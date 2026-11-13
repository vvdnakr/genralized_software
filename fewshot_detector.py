"""
fewshot_detector.py — find an object in images/video from 1-3 reference photos.

Handles any in-plane rotation (full 360 deg), large scale changes, moderate
perspective, partial occlusion, lighting change, and MULTIPLE instances per frame.
No training needed: add reference images, call detect().

Pipeline (features mode, default for textured objects)
  1. RootSIFT keypoints on reference(s) and frame (CLAHE-normalised).
     SIFT keypoints carry their own scale + orientation -> rotation/scale invariant.
  2. Per-reference k-NN matching + Lowe ratio test (spatially-aware 2nd neighbour).
  3. Generalised Hough voting: every match predicts the object's centre, scale and
     rotation in the frame. Matches from the same instance pile up in one bin;
     clutter spreads out. This is what makes multi-instance detection reliable.
  4. Per cluster: similarity seed (RANSAC) -> expand -> homography (MAGSAC++).
  5. Verification: geometric sanity (convex, no mirroring, bounded scale/shear/
     perspective) + appearance check (ZNCC of the frame warped back into the
     reference). Then polygon NMS across clusters and references.

Shape mode (automatic for textureless references: plain parts, logos, tools)
  Coarse search over a rotation x scale template bank on a polarity-invariant
  edge-orientation field (works when the background flips from darker to lighter
  than the object), continuous pose refinement, verification by edge coverage.

Speed
  Config.preset("accurate" | "balanced" (default) | "fast") trades small-object recall for
  speed by limiting the resolution SIFT sees (960 / 640 / 480 px).
  Tracker: Lucas-Kanade on the verified inlier points + homography re-fit each frame, with
  full detection running in a background thread every N frames -> the video loop never
  waits for detection (a few ms per frame on CPU).
"""
from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np
from scipy import fft as _fft
from scipy.spatial import cKDTree

__all__ = ["Config", "Detection", "FewShotDetector", "Tracker", "FrameDetector", "draw_detections", "poly_iou"]


# ═════════════════════════════════════════════════════════════════════ config
@dataclass
class Config:
    mode: str = "auto"                # "auto" | "features" | "shape" | "both" | "template" (advanced template matching)
    max_side: int = 960               # frames are downscaled to this (long side) for detection

    # --- keypoints / matching
    features: str = "sift_fast"       # "sift_fast": SIFT sees at most `fast_side` px | "sift": full detection resolution
    fast_side: int = 640              # see Config.preset(): 640 = balanced, 480 = fast
    scene_features: int = 6000       # budget AFTER spatially-balanced selection (busy clutter can't starve the object)
    ref_features: int = 3000
    contrast_threshold: float = 0.02  # lower -> more keypoints on low-contrast objects
    use_clahe: bool = True            # local contrast normalisation (lighting robustness)
    ref_size: Tuple[int, int] = (320, 800)  # references resized so their long side is in this range
    ratio: float = 0.85               # Lowe ratio; Hough + verification tolerate a loose value
    guided_sim: float = 0.87          # descriptor similarity needed in guided (pose-constrained) matching
    affine_views: bool = False        # ASIFT-style simulated tilts: robust to strong out-of-plane tilt (slower)
    scale_views: Tuple[float, ...] = (0.5, 0.3)  # extra downscaled reference views -> better on small/far objects

    # --- geometry
    hough_pos_frac: float = 0.25      # Hough position bin = this x projected object size
    min_votes: int = 5
    min_inliers: int = 8
    reproj_thresh: float = 4.0        # px, at detection resolution
    min_scale: float = 0.08           # object size relative to the (resized) reference
    max_scale: float = 10.0
    max_anisotropy: float = 4.0       # max ratio of local stretch along two axes
    max_perspective: float = 3.0      # max ratio of local scale between object corners

    # --- verification / output
    verify_appearance: bool = True
    min_ncc: float = 0.10
    min_score: float = 0.50           # verifier probability a detection needs (same scale for every kind of object)
    nms_iou: float = 0.30
    max_instances: int = 20
    deep_rounds: int = 2              # LightGlue backend: matching passes (each can add instances)

    # --- shape mode (textureless objects)
    # auto mode looks at how much of the object's INTERIOR has edges (texture):
    auto_shape_below: float = 0.15    # less than this -> shape mode is used
    auto_features_above: float = 0.10 # more than this -> feature mode is used (between the two: both run)
    auto_plain_fraction: float = 0.25 # plain region fills at least this much of the crop -> a plain object
    auto_outside_texture: float = 0.10  # edge density of the crop outside the plain region that counts as textured
    shape_coarse_px: int = 24                # object size (px) in the coarse search image; smaller = faster, less sure
    shape_angle_step: float = 12.0
    shape_angle_range: Tuple[float, float] = (0.0, 360.0)
    shape_scale_range: Tuple[float, float] = (0.5, 2.0)   # object size relative to the reference crop; a narrower range is faster
    shape_scale_step: float = 1.20
    shape_proposal_threshold: float = 0.35   # coarse orientation agreement (0-1) to become a proposal
    shape_min_score: float = 0.50            # mean edge-orientation agreement (-1..1)
    shape_min_coverage: float = 0.60         # fraction of template edges found in frame
    shape_min_balanced: float = 0.30         # ... in EVERY edge-orientation group (rejects stripes, single lines)
    shape_mag_thresh: float = 30.0           # Sobel magnitude that counts as an edge

    # --- advanced template matching (mode="template", run.py --advance-template)
    template_min_score: float = 0.50         # edge-direction agreement a match needs (1 = every template edge present)
    template_scale_range: Tuple[float, float] = (0.5, 2.0)    # object size relative to the reference crop
    template_angle_range: Tuple[float, float] = (0.0, 360.0)
    template_min_zncc: float = 0.70          # also required: pixel correlation of the whole crop (light-invariant)
    template_polarity: bool = False          # True: light/dark contrast must not reverse (more selective)
    template_coarse_px: int = 24             # object size in the coarse search image
    template_max_candidates: int = 20        # coarse candidates refined per frame
    shape_pre_blur: float = 1.0              # Gaussian sigma before edge detection (camera-noise robustness)
    shape_max_tilt: float = 1.6              # max stretch of the fitted shape (1.6 ~ 50 deg out-of-plane tilt)
    shape_min_flat: float = 0.75             # object interior must be edge-free in the frame where it is in the reference
    shape_max_side: int = 640                # shape search resolution (640 was both faster and more accurate than 960)
    shape_max_proposals: int = 40            # coarse candidates refined per frame
    shape_use_color: bool = True             # compare the interior colour with the reference
    shape_max_color_diff: float = 35.0       # max chroma (Lab a,b) difference, 0-255 scale
    shape_min_ring: float = 0.80             # share of the band just outside the outline that differs from the interior
    shape_grey_strictness: float = 1.1       # colourless objects: edge thresholds x this (fewer look-alike matches)
    shape_max_light_diff: float = 110.0       # max lightness (Lab L) difference (loose: lighting changes)
    shape_auto_mask: bool = True             # GrabCut silhouette of the reference (ignores crop background)


    @classmethod
    def preset(cls, speed: str = "balanced", **overrides) -> "Config":
        """'accurate' : full SIFT at up to 960 px          (best on small objects; ~250 ms at 960x720 on 2 cores)
           'balanced' : SIFT on a <=640 px copy             (~2x faster; same as accurate for 640x480 webcams)
           'fast'     : SIFT on a <=480 px copy             (~3x faster; misses more small objects)
        Plain (textureless) objects are searched at 0.3-2.5x / 0.5-2x / 0.6-1.7x the size they have
        in the reference crop (accurate / balanced / fast); override with shape_scale_range."""
        # plain objects: the cost is set by the SMALLEST size searched (relative to the reference crop)
        table = {"accurate": dict(features="sift", shape_scale_range=(0.3, 2.5), shape_max_proposals=60),
                 "balanced": dict(features="sift_fast", fast_side=640, shape_scale_range=(0.5, 2.0), shape_max_proposals=40),
                 "fast": dict(features="sift_fast", fast_side=480, shape_scale_range=(0.6, 1.7), shape_max_proposals=24,
                              shape_angle_step=15.0, shape_coarse_px=20)}
        if speed not in table:
            raise ValueError(f"speed must be one of {list(table)}")
        return cls(**{**table[speed], **overrides})


@dataclass
class Detection:
    polygon: np.ndarray          # (4,2) reference corners projected into the frame (px)
    H: np.ndarray                # 3x3 homography: reference image px -> frame px
    ref_id: int
    score: float                 # 0..1 confidence
    inliers: int
    angle: float                 # in-plane rotation, deg, counter-clockwise (cv2.getRotationMatrix2D convention)
    scale: float                 # size relative to the reference image
    method: str = "features"
    frame_pts: Optional[np.ndarray] = None  # inlier correspondences (for tracking)
    ref_pts: Optional[np.ndarray] = None
    track_id: int = -1
    # rotated rectangle (cx, cy, w, h, angle_deg_ccw): always the reference's aspect ratio,
    # best similarity fit to `polygon`; smoothed over time when it comes from a Tracker
    rect: Optional[Tuple[float, float, float, float, float]] = None
    shape_pose: Optional[np.ndarray] = None   # shape mode: 2x3 affine at working resolution (for edge tracking)
    evidence: Optional[dict] = None           # shape mode: the verification measurements

    @property
    def center(self) -> np.ndarray:
        if self.rect is not None:
            return np.float32(self.rect[:2])
        return self.polygon.mean(axis=0)

    @property
    def box_points(self) -> np.ndarray:
        """(4,2) corners of the rotated rectangle: object's top-left, top-right, bottom-right, bottom-left."""
        if self.rect is None:
            return self.polygon
        return _rect_points(self.rect)

    @property
    def bbox(self) -> Tuple[int, int, int, int]:
        """Upright (axis-aligned) box x, y, w, h enclosing the rotated rectangle."""
        p = self.box_points
        x0, y0 = np.floor(p.min(0))
        x1, y1 = np.ceil(p.max(0))
        return int(x0), int(y0), int(x1 - x0), int(y1 - y0)


# ═════════════════════════════════════════════════════════════════ helpers
def _fit_rect(poly: np.ndarray, ref_w: float, ref_h: float):
    """Least-squares similarity fit (rotation + uniform scale + shift) of the reference
    rectangle to the projected quadrilateral -> a true rectangle with fixed aspect ratio."""
    P = np.float64([[-ref_w / 2, -ref_h / 2], [ref_w / 2, -ref_h / 2], [ref_w / 2, ref_h / 2], [-ref_w / 2, ref_h / 2]])
    c = poly.astype(np.float64).mean(0)
    Q = poly.astype(np.float64) - c
    z, w = P[:, 0] + 1j * P[:, 1], Q[:, 0] + 1j * Q[:, 1]
    a = np.sum(np.conj(z) * w) / np.sum(np.abs(z) ** 2)       # s * e^(i*theta), image coords (y down)
    s = abs(a)
    ang = -math.degrees(np.angle(a))                             # counter-clockwise on screen
    return (float(c[0]), float(c[1]), float(ref_w * s), float(ref_h * s), (ang + 180.0) % 360.0 - 180.0)


def _rect_points(rect) -> np.ndarray:
    cx, cy, w, h, ang = rect
    t = -math.radians(ang)
    ca, sa = math.cos(t), math.sin(t)
    P = np.float32([[-w / 2, -h / 2], [w / 2, -h / 2], [w / 2, h / 2], [-w / 2, h / 2]])
    return (P @ np.float32([[ca, sa], [-sa, ca]]) + np.float32([cx, cy])).astype(np.float32)


def _attach_rect(d: "Detection", ref) -> "Detection":
    d.rect = _fit_rect(d.polygon, ref.w, ref.h)
    d.angle = d.rect[4]
    return d


def _gray(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img


def _rootsift(desc: Optional[np.ndarray]) -> Optional[np.ndarray]:
    if desc is None:
        return None
    desc = desc / (np.abs(desc).sum(axis=1, keepdims=True) + 1e-7)
    return np.sqrt(desc).astype(np.float32)


def _project(H: np.ndarray, pts: np.ndarray) -> np.ndarray:
    p = np.c_[pts, np.ones(len(pts))] @ H.T
    return p[:, :2] / p[:, 2:3]


def _local_affine(H: np.ndarray, x: float, y: float) -> np.ndarray:
    """Jacobian of the homography at (x, y)."""
    w = H[2, 0] * x + H[2, 1] * y + H[2, 2]
    u = (H[0, 0] * x + H[0, 1] * y + H[0, 2]) / w
    v = (H[1, 0] * x + H[1, 1] * y + H[1, 2]) / w
    return np.array([[H[0, 0] - u * H[2, 0], H[0, 1] - u * H[2, 1]],
                     [H[1, 0] - v * H[2, 0], H[1, 1] - v * H[2, 1]]]) / w


def _pose_from_H(H: np.ndarray, cx: float, cy: float) -> Tuple[float, float]:
    J = _local_affine(H, cx, cy)
    scale = math.sqrt(max(abs(np.linalg.det(J)), 1e-12))
    ang_img = math.degrees(math.atan2(J[1, 0] - J[0, 1], J[0, 0] + J[1, 1]))
    ang = -ang_img  # y-down image rotation -> counter-clockwise on screen
    return (ang + 180.0) % 360.0 - 180.0, scale


def _is_convex(p: np.ndarray) -> bool:
    d = np.roll(p, -1, axis=0) - p
    dn = np.roll(d, -1, axis=0)
    cr = d[:, 0] * dn[:, 1] - d[:, 1] * dn[:, 0]
    return bool(np.all(cr > 0) or np.all(cr < 0))


def _points_in_convex(poly: np.ndarray, pts: np.ndarray) -> np.ndarray:
    d = np.roll(poly, -1, axis=0) - poly
    inside = np.ones(len(pts), bool)
    area2 = np.sum(poly[:, 0] * np.roll(poly[:, 1], -1) - np.roll(poly[:, 0], -1) * poly[:, 1])
    s = 1.0 if area2 > 0 else -1.0
    for i in range(len(poly)):
        rel = pts - poly[i]
        cr = d[i, 0] * rel[:, 1] - d[i, 1] * rel[:, 0]
        inside &= (s * cr) >= 0
    return inside


def poly_iou(a: np.ndarray, b: np.ndarray) -> float:
    a = np.ascontiguousarray(a, np.float32)
    b = np.ascontiguousarray(b, np.float32)
    inter, _ = cv2.intersectConvexConvex(a, b)
    union = abs(cv2.contourArea(a)) + abs(cv2.contourArea(b)) - inter
    return float(inter / union) if union > 1e-6 else 0.0


def _nms(dets: List[Detection], iou: float) -> List[Detection]:
    dets = sorted(dets, key=lambda d: -d.score)
    keep: List[Detection] = []
    for d in dets:
        # suppress if overlapping OR one centre inside another (catches nested partial fits)
        if all(poly_iou(d.polygon, k.polygon) < iou and
               not _points_in_convex(k.polygon, d.center[None])[0] for k in keep):
            keep.append(d)
    return keep


def _balanced_subset(kps, shape, budget: int, grid: int = 8) -> np.ndarray:
    """Pick `budget` keypoints round-robin over a grid (strongest first in each cell), so
    a highly textured region (text, foliage) cannot use up the whole keypoint budget."""
    h, w = shape[:2]
    pts = np.float32([k.pt for k in kps])
    resp = np.float32([k.response for k in kps])
    cell = (np.minimum((pts[:, 1] * grid / h).astype(int), grid - 1) * grid +
            np.minimum((pts[:, 0] * grid / w).astype(int), grid - 1))
    o = np.lexsort((-resp, cell))
    sc = cell[o]
    first = np.r_[0, np.nonzero(np.diff(sc))[0] + 1]
    starts = np.repeat(first, np.diff(np.r_[first, len(sc)]))
    rank = np.empty(len(kps), np.int64)
    rank[o] = np.arange(len(sc)) - starts
    return np.lexsort((-resp, rank))[:budget]


def _scale_det(d: Detection, f: float) -> Detection:
    """Convert a detection from detection-resolution coords (frame*f) to full-res."""
    if f == 1.0:
        return d
    S = np.diag([1.0 / f, 1.0 / f, 1.0])
    d.H = S @ d.H
    d.polygon = d.polygon / f
    if d.frame_pts is not None:
        d.frame_pts = d.frame_pts / f
    d.scale = d.scale / f
    return d


# ═══════════════════════════════════════════════════════════════ features
class _Extractor:
    """Rotation- and scale-invariant keypoints + RootSIFT descriptors.
      sift      : OpenCV SIFT on the detection image (upsampled 2x internally) - most accurate
      sift_fast : SIFT on a copy whose long side is at most `fast_side` (480) px. At 960 px
                  this skips the expensive upsampled octave (~3x faster); only the tiniest
                  keypoints are lost, so very small objects are found a bit less often.
    The reference always uses full SIFT (computed once), so both modes match the same bank.
    """

    def __init__(self, kind: str, cfg: Config, for_ref: bool):
        if kind not in ("sift", "sift_fast"):
            raise ValueError(f"unknown features '{kind}' (use 'sift_fast' or 'sift')")
        self.fast_side = cfg.fast_side if (kind == "sift_fast" and not for_ref) else 0
        self.det = cv2.SIFT_create(nfeatures=cfg.ref_features if for_ref else 0,
                                   contrastThreshold=cfg.contrast_threshold)

    def __call__(self, img, mask=None, budget: int = 0):
        """-> (pts (N,2), size (N,), angle (N,), desc (N,128)) or None"""
        work, m, k = img, mask, 1.0
        if self.fast_side and max(img.shape[:2]) > self.fast_side:
            r = self.fast_side / max(img.shape[:2])
            size = (max(8, round(img.shape[1] * r)), max(8, round(img.shape[0] * r)))
            work = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
            m = None if mask is None else cv2.resize(mask, size, interpolation=cv2.INTER_NEAREST)
            k = img.shape[1] / size[0]
        kps, des = self.det.detectAndCompute(work, m)
        if des is None or len(kps) == 0:
            return None
        if budget and len(kps) > budget:
            sel = _balanced_subset(kps, work.shape, budget)
            kps = [kps[i] for i in sel]
            des = des[sel]
        pts = np.float32([kp.pt for kp in kps])
        if k != 1.0:
            pts = (pts + 0.5) * k - 0.5   # pixel-centre aware rescale
        size = np.float32([kp.size for kp in kps]) * k
        ang = np.float32([kp.angle for kp in kps])
        return pts, size, ang, _rootsift(des)


# ═══════════════════════════════════════════════════════════════ verifier
# Learned verifier: turns the evidence measured on one lined-up candidate into a single
# probability that it really is the object. Logistic models fitted on candidates from many
# objects placed in real photographs (true detections vs. everything else the detector
# proposed there), with signs checked; see README "Verification".
_VERIFY_SHAPE = dict(cov=3.988, flat=9.338, ring=9.693, dab=-12.432, dl=-0.932, hog=8.174, lsize=2.928,
                     grey=-4.075, bias=-36.071)
_VERIFY_FEAT = dict(hog=25.007, linl=3.215, ncc=14.728, bias=-32.451)


def _sigmoid(z: float) -> float:
    return 1.0 / (1.0 + math.exp(-max(min(z, 30.0), -30.0)))


def _shape_probability(ev: dict, grey: bool) -> float:
    """Plain objects. Uses only measures that survive shading, highlights and shadows: outline
    coverage, edge-free interior, a border that differs from the interior all around, colour
    and brightness agreement with the reference, the registered-template (HOG) match, size."""
    w = _VERIFY_SHAPE
    lsize = math.log(min(max(ev.get("size", 100.0), 60.0), 220.0))     # stay inside the trained size range
    z = (w["bias"] + w["cov"] * ev["cov"] + w["flat"] * ev["flat"] + w["ring"] * ev["ring"]
         + w["dab"] * min(ev["dab"], 60.0) / 60.0 + w["dl"] * min(ev["dl"], 120.0) / 120.0
         + w["hog"] * ev["hog"] + w["lsize"] * lsize + w["grey"] * float(grey))
    return _sigmoid(z)


def _feature_probability(hog: float, inliers: int, ncc: float) -> float:
    w = _VERIFY_FEAT
    return _sigmoid(w["bias"] + w["hog"] * hog + w["linl"] * math.log(max(inliers, 1)) + w["ncc"] * ncc)


# ═══════════════════════════════════════════════════════════ appearance model
class _Appearance:
    """Registered appearance of one object, used to VERIFY a candidate once it has been lined
    up with the reference (template matching on a normalised view, as registered-image
    matching in industrial vision sensors or aligned face matching does):

      * the candidate is warped into a canonical 64x64 grey patch,
      * the patch is described by HOG (histograms of gradient directions in 8x8 cells,
        contrast-normalised per block, direction sign ignored): robust to light, exposure
        and small misalignment, sensitive to the actual pattern and outline,
      * the score is the best cosine similarity to the registered templates: the reference
        and, after calibration, the other shots of the object. 1 = same pattern.

    (A PCA "eigen-patch" model on raw pixels was tested for this step and separated true
    from false candidates clearly worse than HOG templates on real photos.)
    """
    N = 64
    _hog = None

    def __init__(self, gray: np.ndarray):
        if _Appearance._hog is None:
            _Appearance._hog = cv2.HOGDescriptor((64, 64), (16, 16), (8, 8), (8, 8), 9)
        self.h, self.w = gray.shape
        self.templates = [self._describe(cv2.resize(gray, (self.N, self.N), interpolation=cv2.INTER_AREA))]

    @classmethod
    def _describe(cls, patch: np.ndarray) -> np.ndarray:
        v = cls._hog.compute(patch).ravel()
        return v / (np.linalg.norm(v) + 1e-6)

    def describe(self, img_gray: np.ndarray, H: np.ndarray) -> Optional[np.ndarray]:
        """H: reference px (the grey image given at construction) -> image px."""
        N = self.N
        Hp = H @ np.diag([self.w / N, self.h / N, 1.0])
        scale = math.sqrt(abs(float(np.linalg.det(_local_affine(Hp, N / 2, N / 2)))))    # image px per patch px
        if scale > 1.5:      # object much larger than the patch: average instead of skipping pixels
            k = 1.0 / scale
            img_gray = cv2.resize(img_gray, None, fx=k, fy=k, interpolation=cv2.INTER_AREA)
            Hp = np.diag([k, k, 1.0]) @ Hp
        return self._describe(cv2.warpPerspective(img_gray, Hp, (N, N), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                                                  borderMode=cv2.BORDER_REPLICATE))

    def score(self, v: Optional[np.ndarray]) -> float:
        return 0.0 if v is None else float(max(v @ t for t in self.templates))

    def learn(self, v: Optional[np.ndarray]):
        if v is not None:
            self.templates.append(v)


# ═══════════════════════════════════════════════════════════════ reference
class _Reference:
    """One reference image prepared for both feature and shape matching."""

    def __init__(self, img: np.ndarray, mask: Optional[np.ndarray], cfg: Config,
                 extractor: "_Extractor", clahe):
        g = _gray(img)
        h, w = g.shape
        lo, hi = cfg.ref_size
        k = 1.0
        if max(h, w) > hi:
            k = hi / max(h, w)
        elif max(h, w) < lo:
            k = lo / max(h, w)
        self.k = k                                   # original ref px * k = internal ref px
        if k != 1.0:
            g = cv2.resize(g, (round(w * k), round(h * k)),
                           interpolation=cv2.INTER_AREA if k < 1 else cv2.INTER_CUBIC)
            if mask is not None:
                mask = cv2.resize(mask, (g.shape[1], g.shape[0]), interpolation=cv2.INTER_NEAREST)
        self.h, self.w = g.shape
        self.mask = (mask > 0).astype(np.uint8) * 255 if mask is not None else None
        self.user_mask = mask is not None
        self.color = cv2.resize(img, (self.w, self.h)) if img.ndim == 3 else cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)
        self.gray_raw = g
        self.gray = clahe.apply(g) if clahe is not None else g
        self.corners = np.float32([[0, 0], [self.w, 0], [self.w, self.h], [0, self.h]])
        self.center = np.float32([self.w / 2, self.h / 2])
        self.dim = float(max(self.w, self.h))
        self._build_features(cfg, extractor)
        # texture measure for auto mode (always SIFT, independent of the chosen descriptor)
        self.n_texture_kp = len(cv2.SIFT_create(contrastThreshold=cfg.contrast_threshold).detect(self.gray, self.mask))
        self._build_appearance()
        self.app = _Appearance(self.gray_raw)
        self.shape = None  # built lazily if needed

    # -- features ------------------------------------------------------------
    def _build_features(self, cfg: Config, extractor):
        views = [(1.0, 0.0, 1.0)]
        views += [(1.0, 0.0, s) for s in cfg.scale_views if s * max(self.w, self.h) >= 48]
        if cfg.affine_views:
            views += [(t, phi, 1.0) for t in (1.6,) for phi in (0.0, 45.0, 90.0, 135.0)]
            views += [(2.4, phi, 1.0) for phi in (0.0, 60.0, 120.0)]
        pts_ref, pts_view, ctr_view, sizes, angs, descs = [], [], [], [], [], []
        self.n_base_kp = 0
        for t, phi, sc in views:
            img, msk, A = self._affine_view(t, phi, sc)
            r = extractor(img, msk)
            if r is None:
                continue
            pv, sz, an, des = r
            if t == 1.0 and sc == 1.0:
                self.n_base_kp = len(pv)
            Ainv = cv2.invertAffineTransform(A)
            pr = pv @ Ainv[:, :2].T + Ainv[:, 2]
            cv = (A[:, :2] @ self.center + A[:, 2]).astype(np.float32)
            pts_ref.append(pr); pts_view.append(pv)
            ctr_view.append(np.repeat(cv[None], len(pv), axis=0))
            sizes.append(sz)
            angs.append(an)
            descs.append(des)
        if not descs:
            self.desc = None
            return
        self.pts_ref = np.concatenate(pts_ref).astype(np.float32)
        self.pts_view = np.concatenate(pts_view).astype(np.float32)
        self.ctr_view = np.concatenate(ctr_view).astype(np.float32)
        self.kp_size = np.concatenate(sizes)
        self.kp_angle = np.concatenate(angs)
        self.desc = np.concatenate(descs)
        if len(self.desc) > 12000:  # smaller banks use exact BLAS matching instead
            self.matcher = cv2.FlannBasedMatcher(dict(algorithm=1, trees=4), dict(checks=64))
            self.matcher.add([self.desc])
            self.matcher.train()

    def _affine_view(self, t: float, phi: float, sc: float = 1.0):
        if sc != 1.0:
            size = (max(8, round(self.w * sc)), max(8, round(self.h * sc)))
            img = cv2.resize(self.gray, size, interpolation=cv2.INTER_AREA)
            msk = None if self.mask is None else cv2.resize(self.mask, size, interpolation=cv2.INTER_NEAREST)
            A = np.float32([[size[0] / self.w, 0, 0], [0, size[1] / self.h, 0]])
            return img, msk, A
        if t == 1.0:
            return self.gray, self.mask, np.float32([[1, 0, 0], [0, 1, 0]])
        # ASIFT: rotate by phi on an expanded canvas, anti-alias along x, compress x by t
        R = cv2.getRotationMatrix2D((0, 0), phi, 1.0)
        c = self.corners @ R[:, :2].T
        off = -c.min(0)
        W, Hh = (np.ceil(c.max(0) - c.min(0)).astype(int) + 1).tolist()
        M1 = np.hstack([R[:, :2], off[:, None]]).astype(np.float32)
        base_mask = self.mask if self.mask is not None else np.full((self.h, self.w), 255, np.uint8)
        rot = cv2.warpAffine(self.gray, M1, (W, Hh), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        msk = cv2.warpAffine(base_mask, M1, (W, Hh), flags=cv2.INTER_NEAREST)
        rot = cv2.GaussianBlur(rot, (0, 0), sigmaX=0.8 * math.sqrt(t * t - 1), sigmaY=0.01)
        Wt = max(8, int(round(W / t)))
        img = cv2.resize(rot, (Wt, Hh), interpolation=cv2.INTER_LINEAR)
        msk = cv2.resize(msk, (Wt, Hh), interpolation=cv2.INTER_NEAREST)
        msk = cv2.erode(msk, np.ones((5, 5), np.uint8))
        A = np.diag([Wt / W, 1.0]).astype(np.float32) @ M1
        return img, msk, A.astype(np.float32)

    # -- appearance ----------------------------------------------------------
    def _build_appearance(self, side: int = 96):
        s = side / max(self.w, self.h)
        self.app_k = s
        self.app_size = (max(8, round(self.w * s)), max(8, round(self.h * s)))
        small = cv2.resize(self.gray, self.app_size, interpolation=cv2.INTER_AREA)
        self.app_img = cv2.GaussianBlur(small, (3, 3), 0).astype(np.float32)
        m = np.ones(small.shape, bool) if self.mask is None else \
            cv2.resize(self.mask, self.app_size, interpolation=cv2.INTER_NEAREST) > 0
        self.app_mask = m


# ═══════════════════════════════════════════════════════════ shape model
def _color_grad(img: np.ndarray, pre_blur: float = 0.0):
    """Sobel gradient of the strongest colour channel at each pixel (grey images: plain Sobel).
    Colour edges stay visible where the grey levels of object and background are equal.
    pre_blur: Gaussian sigma applied first, so camera noise is not mistaken for edges."""
    if pre_blur > 0:
        img = cv2.GaussianBlur(img, (0, 0), pre_blur)
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)
    bx = by = bm = None
    for ch in (cv2.split(img) if img.ndim == 3 else [img]):
        gx, gy = cv2.spatialGradient(ch)          # 3x3 Sobel, SIMD
        gx, gy = gx.astype(np.float32), gy.astype(np.float32)
        if img.ndim == 2:
            return gx, gy
        m = cv2.magnitude(gx, gy)
        if bm is None:
            bx, by, bm = gx, gy, m
        else:
            mask = cv2.compare(m, bm, cv2.CMP_GT)
            bx, by, bm = cv2.copyTo(gx, mask, bx), cv2.copyTo(gy, mask, by), cv2.max(bm, m)
    return bx, by


def _edge_threshold(mag: np.ndarray, base: float) -> float:
    """Edge threshold adapted to the image's noise. The 5th percentile of the gradient
    magnitude sits in the noise floor (flat regions) even in cluttered images; pure noise is
    Rayleigh distributed, whose 5th percentile is 0.32 sigma. Edges must exceed 4.5 sigma,
    never less than `base` and (so that texture is not mistaken for noise) at most 2 x base."""
    v = mag[::3, ::3].ravel()
    sigma = float(np.partition(v, len(v) // 20)[len(v) // 20]) / 0.32
    return float(min(max(base, 4.5 * sigma), 2.0 * base))


class _ShapeModel:
    """Textureless objects (plain parts, circles, simple shapes, logos).
    Search: coarse-to-fine over rotation x scale on a polarity-invariant colour-edge
    orientation field (doubled-angle unit vectors), then similarity + affine (edge-ICP)
    refinement. Verification: edge agreement and coverage, interior flatness, and the
    colour of the object's interior compared with the reference."""


    def __init__(self, ref: _Reference, cfg: Config, r: float = 1.0, parent: "Optional[_ShapeModel]" = None):
        """r: the frame is searched at r x detection resolution."""
        self.cfg = cfg
        self.ref = ref
        self.r = r
        if parent is not None:      # same edge model, different working resolution
            for k in ("ref_thresh", "mask", "edge_xy", "edge_n", "edge_2t", "n_edges", "groups", "flat_xy", "flat_d", "_deep_xy", "sym", "sym_c", "alt_angles", "grey", "ring_xy", "ref_lab",
                      "interior_density", "plain", "plain_frac", "outside_density", "_sub", "_E", "_c2t", "_s2t"):
                setattr(self, k, getattr(parent, k))
        else:
            self._build_model()
        self._spec = {}
        self._build_bank()

    def _build_model(self):
        cfg, ref = self.cfg, self.ref
        img = ref.color
        h, w = img.shape[:2]
        gx, gy = _color_grad(img, cfg.shape_pre_blur)
        mag = cv2.magnitude(gx, gy)
        self.ref_thresh = thr = _edge_threshold(mag, cfg.shape_mag_thresh)
        mask = ref.mask
        plain = False
        if mask is None and cfg.shape_auto_mask:
            mask = self._plain_mask(mag, thr)           # a plain object = the big edge-free region in the crop
            plain = mask is not None
            if mask is None:
                mask = self._grabcut(img)
        self.mask = mask
        border = np.zeros((h, w), bool)
        border[4:-4, 4:-4] = True                       # never use the crop border itself
        region = border if mask is None else border & (cv2.dilate(mask, np.ones((7, 7), np.uint8)) > 0)
        strong = mag > thr
        # thin the edges to their centre line: keep a pixel if it is the maximum ALONG its
        # gradient direction (gives a dense, clean outline even for small, blurry crops)
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        ux, uy = gx / (mag + 1e-6), gy / (mag + 1e-6)
        fwd = cv2.remap(mag, xx + ux, yy + uy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        bwd = cv2.remap(mag, xx - ux, yy - uy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        nms = (mag >= fwd) & (mag > bwd)
        ys, xs = np.nonzero(strong & nms & region)
        if len(xs) < 20:
            ys, xs = np.nonzero(strong & region)
        if len(xs) > 800:
            sel = np.random.default_rng(0).choice(len(xs), 800, replace=False)
            ys, xs = ys[sel], xs[sel]
        theta = np.arctan2(gy[ys, xs], gx[ys, xs])
        xs, ys = xs.astype(np.float32), ys.astype(np.float32)
        if plain:
            # plain object: the outline IS the silhouette's contour. Walk the contour, snap each
            # point to the strongest real edge along the silhouette normal. Background texture
            # next to the object (a busy table) cannot leak into the model this way.
            soft = cv2.GaussianBlur(mask.astype(np.float32), (0, 0), 2.0)
            nx_, ny_ = cv2.Sobel(soft, cv2.CV_32F, 1, 0, ksize=3), cv2.Sobel(soft, cv2.CV_32F, 0, 1, ksize=3)
            cs, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
            pts = np.concatenate([c.reshape(-1, 2) for c in cs if len(c) >= 12]).astype(np.float32) if cs else np.zeros((0, 2), np.float32)
            if len(pts):
                pts = pts[(pts[:, 0] >= 4) & (pts[:, 0] < w - 4) & (pts[:, 1] >= 4) & (pts[:, 1] < h - 4)]
            if len(pts) >= 20:
                step = max(1, len(pts) // 800)
                pts = pts[::step]
                pn = np.c_[cv2.remap(nx_, pts[:, 0][None], pts[:, 1][None], cv2.INTER_LINEAR)[0],
                           cv2.remap(ny_, pts[:, 0][None], pts[:, 1][None], cv2.INTER_LINEAR)[0]]
                pn /= np.linalg.norm(pn, axis=1, keepdims=True) + 1e-9
                d = np.arange(-6, 7, dtype=np.float32)
                sx = pts[:, 0, None] + d[None] * pn[:, 0, None]
                sy = pts[:, 1, None] + d[None] * pn[:, 1, None]
                prof = cv2.remap(mag, sx, sy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
                j = prof.argmax(1)
                on_edge = prof[np.arange(len(j)), j] > thr
                off = np.where(on_edge, d[j], 0.0)
                off = np.where(on_edge, off, np.median(off[on_edge]) if on_edge.any() else 0.0)   # unsnapped points follow the rest
                xs, ys = pts[:, 0] + off * pn[:, 0], pts[:, 1] + off * pn[:, 1]
                theta = np.arctan2(pn[:, 1], pn[:, 0])
        self.edge_xy = np.c_[xs - ref.center[0], ys - ref.center[1]].astype(np.float32)
        self.edge_n = np.c_[np.cos(theta), np.sin(theta)].astype(np.float32)   # edge normals
        self.edge_2t = (2 * theta).astype(np.float32)
        self.n_edges = len(xs)
        # edge-orientation groups: a true match must explain edges of every major
        # orientation, not just two parallel lines (bricks, stripes, door frames)
        ob = (((theta % np.pi) + np.pi / 8) // (np.pi / 4)).astype(int) % 4
        self.groups = [g for g in [ob == i for i in range(4)] if g.sum() >= max(8, 0.08 * len(xs))]
        # interior: flat points (no edges in the reference) and the object's colour
        edge_any = (mag > 0.7 * thr).astype(np.uint8)
        inner = border if mask is None else (cv2.erode(mask, np.ones((9, 9), np.uint8)) > 0) & border
        self.interior_density = float(edge_any[inner].mean()) if inner.any() else 1.0
        # how much of the crop the plain region fills, and how textured the rest of the crop is
        self.plain = plain
        self.plain_frac = float((mask > 0).mean()) if plain else 0.0
        outside = border & (cv2.dilate(mask, np.ones((15, 15), np.uint8)) == 0) if plain else border
        self.outside_density = float(edge_any[outside].mean()) if outside.any() else 0.0
        self.flat_xy = np.zeros((0, 2), np.float32)
        self.flat_d = np.zeros(0, np.float32)
        self.ring_xy = np.zeros((0, 2), np.float32)
        self.ref_lab = None
        if mask is not None:
            dist = cv2.distanceTransform(1 - edge_any, cv2.DIST_L2, 3)
            far = dist > 4
            fy, fx = np.nonzero(far & inner)
            if len(fx) > 400:
                sel = np.random.default_rng(1).choice(len(fx), 400, replace=False)
                fy, fx = fy[sel], fx[sel]
            if len(fx) >= 30:
                self.flat_xy = np.c_[fx - ref.center[0], fy - ref.center[1]].astype(np.float32)
                self.flat_d = dist[fy, fx].astype(np.float32)   # distance to the nearest reference edge
                lab = cv2.cvtColor(img, cv2.COLOR_BGR2Lab).astype(np.float32)
                self.ref_lab = np.median(lab[fy, fx], axis=0)
                # ring of points just OUTSIDE the silhouette: in a real match they look
                # different from the interior (the object ends at its outline)
                pad = int(0.15 * ref.dim) + 8
                mp = cv2.copyMakeBorder(mask, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=0)
                k0, k1 = 9, 9 + 2 * max(3, int(0.05 * ref.dim))
                band = (cv2.dilate(mp, np.ones((k1, k1), np.uint8)) > 0) & ~(cv2.dilate(mp, np.ones((k0, k0), np.uint8)) > 0)
                ry, rx = np.nonzero(band)
                if len(rx) > 300:
                    sel = np.random.default_rng(3).choice(len(rx), 300, replace=False)
                    ry, rx = ry[sel], rx[sel]
                self.ring_xy = np.c_[rx - pad - ref.center[0], ry - pad - ref.center[1]].astype(np.float32)
        # the deepest third of the interior points (robust to a roughly placed candidate)
        self._deep_xy = np.zeros((0, 2), np.float32)
        if len(self.flat_xy) >= 30:
            order = np.argsort(-self.flat_d)[: max(24, len(self.flat_d) // 3)][:60]
            self._deep_xy = self.flat_xy[order]
        self.sym = self._symmetry()
        self.grey = self.ref_lab is None or float(np.hypot(self.ref_lab[1] - 128.0, self.ref_lab[2] - 128.0)) < 20.0
        k = min(200, self.n_edges)
        self._sub = np.random.default_rng(2).choice(self.n_edges, k, replace=False) if k else np.zeros(0, int)
        self._E = self.edge_xy[self._sub]
        self._c2t, self._s2t = np.cos(self.edge_2t[self._sub]), np.sin(self.edge_2t[self._sub])

    def _symmetry(self) -> int:
        """Rotational symmetry of the outline: 0 = looks the same at every angle (circle,
        ring), k = k-fold (square 4, triangle 3, ellipse 2, ...), 1 = none. Used to report
        one canonical angle, so the box of a circle does not spin from frame to frame."""
        self.alt_angles = []
        self.sym_c = np.zeros(2)
        if self.n_edges < 12:
            return 1
        # rotate about the outline's own centre (the object is rarely centred in the crop)
        self.sym_c = self.edge_xy.mean(0).astype(np.float64)
        if self.mask is not None:
            mo = cv2.moments(self.mask, binaryImage=True)
            if mo["m00"] > 0:
                self.sym_c = np.array([mo["m10"] / mo["m00"] - self.ref.center[0], mo["m01"] / mo["m00"] - self.ref.center[1]])
        E = self.edge_xy - self.sym_c.astype(np.float32)
        R = float(np.linalg.norm(E, axis=1).max()) + 1e-6
        n = 41

        def field(deg):
            a = math.radians(deg)
            ca, sa = math.cos(a), math.sin(a)
            x = (ca * E[:, 0] - sa * E[:, 1]) / R * (n // 2 - 3) + n // 2
            y = (sa * E[:, 0] + ca * E[:, 1]) / R * (n // 2 - 3) + n // 2
            T = np.zeros((n, n), np.complex64)
            np.add.at(T, (np.rint(y).astype(int), np.rint(x).astype(int)), np.exp(1j * (self.edge_2t + 2 * a)))
            mag = np.abs(T)
            T = np.where(mag > 1e-6, T / np.maximum(mag, 1e-6), 0)     # one vote per cell, whatever the point density
            T = cv2.GaussianBlur(T.real.astype(np.float32), (0, 0), 1.5) + 1j * cv2.GaussianBlur(T.imag.astype(np.float32), (0, 0), 1.5)
            return T / (np.linalg.norm(T) + 1e-9)

        T0 = field(0.0)
        same = lambda deg: float((T0 * np.conj(field(deg))).real.sum())
        # rotations at which the outline looks SIMILAR (not identical) to itself, e.g. an almost
        # equilateral triangle at 120 deg: the search can lock onto these, so they are re-checked
        curve = np.array([same(a) for a in range(0, 360, 6)])
        self.alt_angles = [6.0 * i for i in range(4, 57)
                           if curve[i] > 0.45 and curve[i] >= curve[i - 1] and curve[i] >= curve[(i + 1) % 60]]
        if min(same(17.0), same(49.0), same(73.0), same(131.0)) > 0.75:
            return 0
        for k in (8, 6, 5, 4, 3, 2):
            if all(same(360.0 * j / k) > 0.75 for j in range(1, k)):
                return k
        return 1

    def canonical(self, A: np.ndarray) -> np.ndarray:
        """Among the equivalent poses of a symmetric shape, the one rotated least from the reference."""
        if self.sym == 1:
            return A
        ang = math.degrees(math.atan2(A[1, 0] - A[0, 1], A[0, 0] + A[1, 1]))     # image-coordinate rotation
        step = 360.0 / self.sym if self.sym else 0.0
        phi = -ang if self.sym == 0 else -round(ang / step) * step
        t = math.radians(phi)
        Rm = np.array([[math.cos(t), -math.sin(t)], [math.sin(t), math.cos(t)]])
        out = A.copy()
        out[:, :2] = A[:, :2] @ Rm
        out[:, 2] = A[:, 2] + A[:, :2] @ self.sym_c - out[:, :2] @ self.sym_c    # turn about the outline centre
        return out

    def _build_bank(self):
        cfg, ref = self.cfg, self.ref
        a0, a1 = cfg.shape_angle_range
        angles = np.arange(a0, a1 - 1e-6, cfg.shape_angle_step)
        # rotational symmetry: equivalent rotations need not be searched twice
        if self.sym == 0:
            angles = angles[:1]
        elif self.sym > 1:
            angles = angles[angles < a0 + 360.0 / self.sym - 1e-6]
        smin, smax = cfg.shape_scale_range
        scales = [smin]
        while scales[-1] * cfg.shape_scale_step <= smax + 1e-6:
            scales.append(scales[-1] * cfg.shape_scale_step)
        self.scales = [s * self.r / ref.k for s in scales]   # range is relative to the ORIGINAL reference size
        self.bank = []                              # [(ds, [(angle, scale, T, wsum, r), ...]), ...]
        self.n_templates = 0
        if self.n_edges < 12:
            return
        for s in self.scales:
            ds = max(1.0, s * ref.dim / cfg.shape_coarse_px)
            ents = []
            for a in angles:
                T, wsum, r = self._raster(float(a), s, ds)
                if wsum < 8:
                    continue
                ents.append((float(a), s, T, wsum, r))
            self.bank.append((ds, ents))
            self.n_templates += len(ents)

    @staticmethod
    def _plain_mask(mag, thr):
        """Silhouette of a plain object: the largest connected edge-free region of the crop that
        is not the surrounding surface (the surface runs along the crop border, the object does
        not). Works on plain and on busy backgrounds; None if the crop has no such region."""
        h, w = mag.shape
        free = (cv2.dilate((mag > 0.7 * thr).astype(np.uint8), np.ones((3, 3), np.uint8)) == 0).astype(np.uint8)
        n, lab, stats, _ = cv2.connectedComponentsWithStats(free, connectivity=4)
        if n <= 1:
            return None
        edge_ids = np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]])
        on_border = np.bincount(edge_ids, minlength=n) / float(len(edge_ids))
        area = stats[:, cv2.CC_STAT_AREA].astype(np.float64)
        score = area * (1.0 - np.minimum(on_border * 2.0, 1.0)) ** 2
        score[0] = 0
        best = int(np.argmax(score))
        if area[best] < 0.12 * h * w or on_border[best] > 0.35:
            return None
        m = np.where(lab == best, 255, 0).astype(np.uint8)
        return cv2.dilate(m, np.ones((7, 7), np.uint8))      # give back the edge band that was cut away

    @staticmethod
    def _grabcut(color):
        """Silhouette of the object inside the reference crop; None if it looks unreliable."""
        h, w = color.shape[:2]
        if min(h, w) < 24:
            return None
        m = np.zeros((h, w), np.uint8)
        mx, my = max(2, w // 20), max(2, h // 20)
        try:
            bg, fg = np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64)
            cv2.grabCut(color, m, (mx, my, w - 2 * mx, h - 2 * my), bg, fg, 4, cv2.GC_INIT_WITH_RECT)
        except cv2.error:
            return None
        sil = np.where((m == cv2.GC_FGD) | (m == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
        frac = (sil > 0).mean()
        if not (0.08 < frac < 0.92):
            return None
        n, lab, stats, _ = cv2.connectedComponentsWithStats(sil)
        if n > 1:   # keep the largest blob
            big = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
            sil = np.where(lab == big, 255, 0).astype(np.uint8)
        return cv2.morphologyEx(sil, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))

    def _raster(self, ang_ccw, s, ds):
        """Render the edge model (rotated/scaled) as a 2-channel doubled-angle template.
        Built from points, so rotation never creates fake edges at a canvas border."""
        a = -math.radians(ang_ccw)
        ca, sa = math.cos(a), math.sin(a)
        ex, ey = self.edge_xy[:, 0] * s / ds, self.edge_xy[:, 1] * s / ds
        px, py = ca * ex - sa * ey, sa * ex + ca * ey
        r = int(math.ceil(max(np.abs(px).max(), np.abs(py).max()))) + 2
        T = np.zeros((2 * r + 1, 2 * r + 1, 2), np.float32)
        ix, iy = np.rint(px + r).astype(int), np.rint(py + r).astype(int)
        t2 = self.edge_2t + 2 * a
        np.add.at(T, (iy, ix, 0), np.cos(t2))
        np.add.at(T, (iy, ix, 1), np.sin(t2))
        n = np.linalg.norm(T, axis=2)
        m = n > 1e-6
        T[m] /= n[m][:, None]
        return T, float(m.sum()), r

    @staticmethod
    def frame_field(img: np.ndarray, mag_thresh: float, blur: float = 1.2, pre_blur: float = 0.0):
        gx, gy = _color_grad(img, pre_blur)
        mag = cv2.magnitude(gx, gy)
        _, m = cv2.threshold(mag, _edge_threshold(mag, mag_thresh), 1.0, cv2.THRESH_BINARY)
        gxx, gyy = cv2.multiply(gx, gx), cv2.multiply(gy, gy)
        den = cv2.add(cv2.add(gxx, gyy), 1e-6)
        c2 = cv2.multiply(cv2.divide(cv2.subtract(gxx, gyy), den), m)          # cos(2θ): contrast-polarity invariant
        s2 = cv2.multiply(cv2.divide(cv2.multiply(gx, gy, scale=2.0), den), m)  # sin(2θ)
        c2 = cv2.GaussianBlur(c2, (0, 0), blur)   # spread: tolerance to small misalignment
        s2 = cv2.GaussianBlur(s2, (0, 0), blur)
        return c2, s2, m

    COARSE_BLUR = 1.0
    COARSE_NORM = 0.0

    def coarse_field(self, small):
        """Edge-orientation field for the coarse search. With COARSE_NORM > 0 the blurred
        vectors are re-normalised (orientation spreading): an edge keeps its full weight a few
        pixels away from where it is, which tolerates the rotation/scale steps of the bank."""
        c2, s2, _ = self.frame_field(small, getattr(self.ref, 'edge_thr', None) or self.cfg.shape_mag_thresh, self.COARSE_BLUR)   # downscaled: already denoised
        if self.COARSE_NORM > 0:
            n = cv2.magnitude(c2, s2)
            k = np.where(n > self.COARSE_NORM, 1.0 / np.maximum(n, 1e-6), 0.0).astype(np.float32)
            c2, s2 = c2 * k, s2 * k
        return c2, s2

    def propose(self, img: np.ndarray, thr: float, per_template: int = 4):
        """Coarse search. All rotations of one scale are correlated in a single batched FFT:
        the frame field is a complex image c2 + i*s2, so Re(conj(T) * F) summed over the
        template is exactly the orientation agreement."""
        h, w = img.shape[:2]
        props = []
        prev = img
        for gi, (ds, ents) in enumerate(self.bank):      # bank is ordered small -> large object = large -> small image
            if not ents:
                continue
            sw, sh = max(8, round(w / ds)), max(8, round(h / ds))
            src = prev if prev.shape[1] >= 1.15 * sw else img    # shrink the previous level, not the full frame
            small = cv2.resize(src, (sw, sh), interpolation=cv2.INTER_AREA) if ds > 1 else img
            prev = small
            c2, s2 = self.coarse_field(small)
            fx, fy = w / sw, h / sh
            fits = [i for i, e in enumerate(ents) if e[2].shape[0] < sh and e[2].shape[1] < sw]
            if not fits:
                continue
            ph, pw = _fft.next_fast_len(sh), _fft.next_fast_len(sw)   # zero padding: fast FFT sizes
            key = (gi, ph, pw)
            spec = self._spec.get(key)
            if spec is None:               # template spectra for this frame size (computed once)
                Tp = np.zeros((len(fits), ph, pw), np.complex64)
                for j, i in enumerate(fits):
                    T = ents[i][2]
                    Tp[j, :T.shape[0], :T.shape[1]] = T[..., 0] + 1j * T[..., 1]
                spec = self._spec[key] = np.conj(_fft.fft2(Tp, workers=-1))
            Fp = np.zeros((ph, pw), np.complex64)
            Fp[:sh, :sw] = c2 + 1j * s2
            Fs = _fft.fft2(Fp, workers=-1)
            buf = self._spec.get(("buf",) + key)
            if buf is None:
                buf = self._spec[("buf",) + key] = np.empty_like(spec)
            np.multiply(spec, Fs[None], out=buf)
            R = _fft.ifft2(buf, workers=-1, overwrite_x=True).real
            for j, i in enumerate(fits):
                a, s, T, wsum, r = ents[i]
                res = R[j, : sh - T.shape[0] + 1, : sw - T.shape[1] + 1] / wsum
                if res.max() < thr:
                    continue
                k = max(3, r | 1)
                res = np.ascontiguousarray(res, np.float32)
                peaks = (res >= cv2.dilate(res, np.ones((k, k), np.uint8))) & (res >= thr)
                ys, xs = np.nonzero(peaks)
                vals = res[ys, xs]
                for q in np.argsort(-vals)[:per_template]:
                    props.append((float(vals[q]), (xs[q] + r) * fx, (ys[q] + r) * fy, a, s))
        return props

    # ---- pose = 2x3 affine A mapping reference-centred coords -> frame
    @staticmethod
    def sim_pose(cx, cy, ang_ccw, s):
        a = -math.radians(ang_ccw)
        ca, sa = math.cos(a) * s, math.sin(a) * s
        return np.array([[ca, -sa, cx], [sa, ca, cy]], np.float64)

    def _edge_frame(self, A):
        p = self.edge_xy @ A[:, :2].T.astype(np.float32) + A[:, 2].astype(np.float32)
        # normals transform with the inverse-transpose of the linear part
        Lit = np.linalg.inv(A[:, :2]).T
        n = self.edge_n @ Lit.T.astype(np.float32)
        n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-9
        return p, n

    def score(self, field, A, lab=None):   # lab: the BGR working image (only sampled pixels are converted to Lab)
        """Evidence for pose A: sc = mean edge-orientation agreement (-1..1), cov = share of
        model edges found, bal = worst coverage over edge orientations, flat = share of the
        plain interior that is also plain in the frame, dab / dl = colour (chroma) and
        lightness difference between the object's interior here and in the reference,
        ring = share of the band just outside the outline that looks different from the interior."""
        c2, s2, m = field
        p, n = self._edge_frame(A)
        px, py = p[:, 0][None], p[:, 1][None]
        fc = cv2.remap(c2, px, py, cv2.INTER_LINEAR, borderValue=0)[0]
        fs = cv2.remap(s2, px, py, cv2.INTER_LINEAR, borderValue=0)[0]
        c2t, s2t = n[:, 0] ** 2 - n[:, 1] ** 2, 2 * n[:, 0] * n[:, 1]
        agree = fc * c2t + fs * s2t
        hit = agree > 0.5
        out = dict(sc=float(agree.mean()), cov=float(hit.mean()),
                   bal=min([float(hit[g].mean()) for g in self.groups], default=float(hit.mean())),
                   flat=1.0, dab=0.0, dl=0.0, ring=1.0, inside=1.0)
        if len(self.flat_xy):
            q = self.flat_xy @ A[:, :2].T.astype(np.float32) + A[:, 2].astype(np.float32)
            qx, qy = q[:, 0][None], q[:, 1][None]
            # only interior points that stay clear of the (few px wide) edge band at this size
            deep = self.flat_d * math.sqrt(abs(float(np.linalg.det(A[:, :2])))) >= 4.0
            e = cv2.remap(m, qx, qy, cv2.INTER_NEAREST, borderValue=1)[0]
            out["flat"] = float((e[deep] < 0.5).mean()) if deep.sum() >= 15 else 1.0
            if lab is not None and self.ref_lab is not None:
                v = cv2.cvtColor(cv2.remap(lab, qx, qy, cv2.INTER_NEAREST, borderMode=cv2.BORDER_REPLICATE),
                                 cv2.COLOR_BGR2Lab)[0].astype(np.float32)
                dab = np.hypot(v[:, 1] - self.ref_lab[1], v[:, 2] - self.ref_lab[2])
                # 35th percentile: "at least a third of the interior has the right colour",
                # so a hand or another object covering part of it does not flip the answer
                k35 = int(0.35 * (len(dab) - 1))
                out["dab"] = float(np.partition(dab, k35)[k35])
                good = dab <= max(out["dab"], 1e-3) * 1.5 + 5.0
                vin = np.median(v[good], axis=0)
                out["dl"] = float(abs(vin[0] - self.ref_lab[0]))
                out["inside"] = float((np.abs(v - vin).max(axis=1) <= 12.0).mean())   # interior is one material
                if len(self.ring_xy):
                    o = self.ring_xy @ A[:, :2].T.astype(np.float32) + A[:, 2].astype(np.float32)
                    w = cv2.cvtColor(cv2.remap(lab, o[:, 0][None], o[:, 1][None], cv2.INTER_NEAREST,
                                               borderMode=cv2.BORDER_REPLICATE), cv2.COLOR_BGR2Lab)[0].astype(np.float32)
                    out["ring"] = float((np.abs(w - vin).max(axis=1) > 10.0).mean())
        return out

    def precheck(self, field, lab, cx, cy, ang, s, max_dab: float) -> bool:
        """Very cheap test of a coarse candidate before any refinement: a plain object's deep
        interior must be free of edges in the frame and (for coloured objects) roughly the
        reference colour. Most clutter candidates fail here in a few microseconds."""
        if not len(self._deep_xy):
            return True
        A = self.sim_pose(cx, cy, ang, s).astype(np.float32)
        q = self._deep_xy @ A[:, :2].T + A[:, 2]
        qx, qy = q[:, 0][None], q[:, 1][None]
        e = cv2.remap(field[2], qx, qy, cv2.INTER_NEAREST, borderValue=1)[0]
        if float((e < 0.5).mean()) < 0.5:
            return False
        if lab is not None and self.ref_lab is not None and max_dab > 0:
            v = cv2.cvtColor(cv2.remap(lab, qx, qy, cv2.INTER_NEAREST, borderMode=cv2.BORDER_REPLICATE),
                             cv2.COLOR_BGR2Lab)[0].astype(np.float32)
            dab = np.hypot(v[:, 1] - self.ref_lab[1], v[:, 2] - self.ref_lab[2])
            k = int(0.35 * (len(dab) - 1))
            if float(np.partition(dab, k)[k]) > max_dab:
                return False
        return True

    def refine_similarity(self, F2, cx, cy, ang, s, a_step, s_step, levels: int = 4, grid: bool = False,
                          dp0: Optional[float] = None):
        """Pattern search over (x, y, angle, scale); all 8 neighbour poses are scored in one
        vectorised remap on a 200-point subset of the edge model.
        grid=True first tries a small exhaustive neighbourhood (position x scale x angle):
        single-parameter moves cannot leave a pose where, say, two of three sides already fit."""
        E, c2t, s2t = self._E, self._c2t, self._s2t

        def evaluate(P):
            a = -np.radians(P[:, 2])[:, None]
            ca, sa, sc = np.cos(a), np.sin(a), P[:, 3][:, None]
            px = P[:, 0][:, None] + sc * (ca * E[:, 0] - sa * E[:, 1])
            py = P[:, 1][:, None] + sc * (sa * E[:, 0] + ca * E[:, 1])
            v = cv2.remap(F2, px.astype(np.float32), py.astype(np.float32), cv2.INTER_LINEAR, borderValue=0)
            c2a, s2a = np.cos(2 * a), np.sin(2 * a)
            return (v[..., 0] * (c2t * c2a - s2t * s2a) + v[..., 1] * (s2t * c2a + c2t * s2a)).mean(1)

        cur = np.array([cx, cy, ang, s], np.float64)
        if grid:
            R = max(2.0, 0.12 * s * self.ref.dim)
            off = np.linspace(-R, R, 5)
            gx, gy, ga, gs = np.meshgrid(off, off, [-a_step / 2, 0.0, a_step / 2],
                                         [1 / (1 + 0.6 * (s_step - 1)), 1.0, 1 + 0.6 * (s_step - 1)], indexing="ij")
            P = np.c_[cx + gx.ravel(), cy + gy.ravel(), ang + ga.ravel(), s * gs.ravel()]
            E, c2t, s2t = E[::4], c2t[::4], s2t[::4]     # 50 points are enough to rank the grid
            cur = P[int(np.argmax(evaluate(P)))]
            E, c2t, s2t = self._E, self._c2t, self._s2t
        best = evaluate(cur[None])[0]
        da, ds = a_step / 2, (s_step - 1) / 2
        dp = max(1.0, 0.03 * s * self.ref.dim) if dp0 is None else dp0
        for _ in range(levels):
            for _ in range(8):
                x, y, a, sc = cur
                cand = np.array([[x + dp, y, a, sc], [x - dp, y, a, sc], [x, y + dp, a, sc], [x, y - dp, a, sc],
                                 [x, y, a + da, sc], [x, y, a - da, sc],
                                 [x, y, a, sc * (1 + ds)], [x, y, a, sc / (1 + ds)]])
                vals = evaluate(cand)
                j = int(np.argmax(vals))
                if vals[j] <= best + 1e-4:
                    break
                best, cur = vals[j], cand[j]
            da, ds, dp = da / 2, ds / 2, max(dp / 2, 0.25)
        return self.sim_pose(*cur), float(best)

    def refine_pose(self, pyr, cx, cy, ang, s, a_step, s_step, reject: float = -1.0):
        """Coarse-to-fine: local grid + pattern search on a reduced image where the object is
        ~40-80 px (wide capture range), then pattern search on each finer level."""
        S = s * self.ref.dim
        L = 0
        while L + 1 < len(pyr) and S * pyr[L + 1][0] >= 40:
            L += 1
        A = None
        first = True
        val = 0.0
        for q, F2 in reversed(pyr[: L + 1]):
            if first:
                A, val = self.refine_similarity(F2, cx * q, cy * q, ang, s * q, a_step, s_step, levels=2, grid=True)
                if val < reject:        # hopeless candidate: stop before the finer levels
                    return None, val
            else:
                A, val = self.refine_similarity(F2, cx * q, cy * q, ang, s * q, a_step / 4, 1 + (s_step - 1) / 4,
                                                levels=2 if q < 1 else 3, dp0=2.0)
            sc = math.sqrt(abs(np.linalg.det(A[:, :2])))
            cx, cy, s = A[0, 2] / q, A[1, 2] / q, sc / q
            ang = -math.degrees(math.atan2(A[1, 0], A[0, 0]))
            first = False
        return self.sim_pose(cx, cy, ang, s), val

    def refine_affine(self, field, A, max_aniso, radii=(6, 4, 3, 2, 2)):
        """Edge-ICP: move each model edge point along its normal to the best-agreeing frame
        edge, then solve a weighted least-squares affine. Absorbs mild perspective/shear."""
        c2, s2, _ = field
        P1 = np.c_[self.edge_xy, np.ones(len(self.edge_xy))]
        for r in radii:
            p, n = self._edge_frame(A)
            d = np.arange(-r, r + 1, dtype=np.float32)
            sx = (p[:, 0, None] + d[None] * n[:, 0, None]).astype(np.float32)
            sy = (p[:, 1, None] + d[None] * n[:, 1, None]).astype(np.float32)
            fc = cv2.remap(c2, sx, sy, cv2.INTER_LINEAR, borderValue=0)
            fs = cv2.remap(s2, sx, sy, cv2.INTER_LINEAR, borderValue=0)
            c2t, s2t = n[:, 0] ** 2 - n[:, 1] ** 2, 2 * n[:, 0] * n[:, 1]
            ag = fc * c2t[:, None] + fs * s2t[:, None]
            j = ag.argmax(1)
            best = ag[np.arange(len(j)), j]
            ok = best > 0.4
            if ok.sum() < max(20, 0.25 * len(ok)):
                break
            tgt = p + d[j][:, None] * n
            wts = np.sqrt(best[ok])[:, None]
            X, *_ = np.linalg.lstsq(P1[ok] * wts, tgt[ok] * wts, rcond=None)
            An = X.T
            sv = np.linalg.svd(An[:, :2], compute_uv=False)
            if np.linalg.det(An[:, :2]) <= 0 or sv[0] / sv[1] > max_aniso:
                break
            A = An
        return A


# ═══════════════════════════════════════════════════════════════ detector
class FewShotDetector:
    def __init__(self, cfg: Optional[Config] = None, deep_backend=None):
        self.cfg = cfg or Config()
        c = self.cfg
        self.ext_ref = _Extractor(c.features, c, for_ref=True)
        self.ext_scene = _Extractor(c.features, c, for_ref=False)
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)) if c.use_clahe else None
        self.refs: List[_Reference] = []
        self.deep = deep_backend
        self.last_timing = {}
        self.calibration = None
        import threading
        self._shape_lock = threading.Lock()

    # ------------------------------------------------------------- references
    def add_reference(self, img: np.ndarray, mask: Optional[np.ndarray] = None, mode: Optional[str] = None) -> int:
        """Add a reference view (BGR or gray). Optional mask (non-zero = object).
        mode overrides Config.mode for this reference ("auto" | "features" | "shape" | "both")."""
        ref = _Reference(img, mask, self.cfg, self.ext_ref, self.clahe)
        ref.cal = None
        ref.edge_thr = None
        ref.min_score = None
        mode = mode or self.cfg.mode
        ref.shape = None
        ref.tm = None
        if mode == "template":
            ref.tm = self._build_template(ref, mask is not None)
            ref.mode = "template"
            self.refs.append(ref)
            return len(self.refs) - 1
        if mode in ("auto", "shape", "both"):
            ref.shape = _ShapeModel(ref, self.cfg)
            ref.shape_cache = {1.0: ref.shape}
        if mode == "auto":
            sm = ref.shape
            dens = sm.interior_density
            use_shape = dens < self.cfg.auto_shape_below and sm.n_templates > 0
            use_feat = (dens > self.cfg.auto_features_above or not use_shape) and ref.desc is not None
            if sm.plain and use_shape and ref.desc is not None:
                # A big plain region was found. If the rest of the crop is textured, that texture
                # may be the object (a cup with a plain patch) or just the table around a plain
                # object: unless the plain region clearly dominates, use both methods.
                if sm.outside_density >= self.cfg.auto_outside_texture and sm.plain_frac < self.cfg.auto_plain_fraction:
                    use_feat = True
            mode = "both" if (use_shape and use_feat) else ("shape" if use_shape else "features")
            if not use_shape:
                ref.shape = None
        ref.mode = mode
        if self.deep is not None:
            self.deep.add_reference(ref.gray)
        self.refs.append(ref)
        return len(self.refs) - 1

    # ------------------------------------------------------------- calibration
    def calibrate(self, images: Sequence[np.ndarray], rois: Optional[Sequence[Tuple[int, int, int, int]]] = None,
                  verbose: bool = True) -> dict:
        """Learn one object from 2-5 images of the SAME side of it.

        images : frames (BGR). rois : the object's box (x, y, w, h) in each frame. Without rois
                 every image is taken to be a crop of the object (then no background is learned).
        The object is taken in different positions / angles / lighting; everything outside the
        boxes is treated as background. Calibration then
          1. picks the image that makes the best model (finds the object in the other images),
          2. picks the matching method (keypoints or outline) that actually finds it in all of them,
          3. sets the accept limits from how the real object scores, and tightens them until the
             background candidates of your own frames are rejected,
          4. learns how much the object's colour and brightness vary between the images.
        Replaces the current references. Returns a report (also kept in self.calibration)."""
        n = len(images)
        if n < 2:
            raise ValueError("calibration needs at least 2 images of the object")
        if rois is None:      # crops only: pad them so the detector has something to search in
            frames, boxes = [], []
            for im in images:
                p = max(16, int(0.3 * max(im.shape[:2])))
                frames.append(cv2.copyMakeBorder(im, p, p, p, p, cv2.BORDER_REPLICATE))
                boxes.append((p, p, im.shape[1], im.shape[0]))
            has_bg = False
        else:
            frames, boxes, has_bg = list(images), [tuple(int(v) for v in r) for r in rois], True
        crops = [f[y:y + h, x:x + w].copy() for f, (x, y, w, h) in zip(frames, boxes)]

        def hit(d, box):          # is detection d the object marked by box?
            x, y, w, h = box
            cx, cy = d.rect[0], d.rect[1]
            area = d.rect[2] * d.rect[3]
            return x <= cx <= x + w and y <= cy <= y + h and 0.35 * w * h <= area <= 2.8 * w * h

        def background(d, box):   # clearly somewhere else in the frame
            x, y, w, h = box
            cx, cy = d.rect[0], d.rect[1]
            return not (x - 0.25 * w <= cx <= x + 1.25 * w and y - 0.25 * h <= cy <= y + 1.25 * h)

        if self.cfg.mode == "template":
            return self._calibrate_template(frames, boxes, crops, hit, background, has_bg, verbose)

        saved_refs, trials = self.refs, []
        for p in range(n):
            self.refs = []
            self.add_reference(crops[p], mode="auto")
            ref = self.refs[0]
            auto_mode = ref.mode
            if ref.shape is None:
                sm = _ShapeModel(ref, self.cfg)
                if sm.n_templates > 0:
                    ref.shape, ref.shape_cache = sm, {1.0: sm}
            t = dict(primary=p, ref=ref, auto_mode=auto_mode, pos=[], neg=[], shape_found=0,
                     feat_found=0, feat_false=0, feat_scores=[], edge_thr=None)

            def outline_pass():
                t["pos"], t["neg"], t["shape_found"] = [], [], 0
                for j in range(n):
                    frame, box = frames[j], boxes[j]
                    g_raw, f = self.prepare(frame)
                    img = frame if f == 1.0 else cv2.resize(frame, (g_raw.shape[1], g_raw.shape[0]), interpolation=cv2.INTER_AREA)
                    cands = [_attach_rect(_scale_det(d, f), ref) for d in self._detect_shape(ref, 0, img, collect=True)]
                    own = [d for d in cands if hit(d, box)]
                    # background = candidates elsewhere in the frame that the detector really evaluates
                    t["neg"] += [d.evidence for d in cands if background(d, box) and d.evidence["gated"]]
                    if j != p and own:
                        best = max(own, key=lambda d: d.evidence["sc"])
                        if best.evidence["sc"] >= 0.35 and best.evidence["cov"] >= 0.40:
                            t["pos"].append(best.evidence)
                            t["shape_found"] += 1

            if ref.shape is not None:
                outline_pass()          # with the default edge sensitivity
            if ref.shape is not None and t["shape_found"] < n - 1:
                # the outline is too faint in some images: search again, very sensitive, and
                # measure how strong the outline really is there
                ref.edge_thr = 0.4 * self.cfg.shape_mag_thresh
                outline_pass()
                if t["pos"]:
                    weakest = min(e["contrast"] for e in t["pos"])
                    # edges count from ~35 % of the weakest outline seen, so a faint object gets a
                    # more sensitive detector. Never less sensitive than the default: the light
                    # may later be dimmer than in the calibration shots.
                    t["edge_thr"] = float(np.clip(0.35 * weakest, 0.4 * self.cfg.shape_mag_thresh,
                                                  self.cfg.shape_mag_thresh))
                ref.edge_thr = t["edge_thr"]
                outline_pass()          # evidence as the detector will really see it
            for j in range(n):
                frame, box = frames[j], boxes[j]
                if ref.desc is not None and j != p:
                    ref.mode = "features"
                    dets = self.detect(frame)
                    ok = [d for d in dets if hit(d, box)]
                    t["feat_found"] += bool(ok)
                    t["feat_scores"] += [max(d.score for d in ok)] if ok else []
                    t["feat_false"] += sum(background(d, box) for d in dets)
                    ref.mode = auto_mode
            # most images found first; then the simpler model (a crop where the symmetry of a
            # circle or square was recognised needs far fewer rotations); otherwise the first image
            t["quality"] = (max(t["shape_found"], t["feat_found"] if t["feat_false"] == 0 else 0),
                            -round(math.log2(max(ref.shape.n_templates, 1))) if ref.shape is not None else 0,
                            -p,
                            float(np.mean(t["feat_scores"])) if t["feat_scores"] else 0.0)
            trials.append(t)
        self.refs = saved_refs

        best = max(trials, key=lambda t: t["quality"])
        ref, others = best["ref"], n - 1
        shape_ok = best["shape_found"] == others
        feat_ok = best["feat_found"] == others and best["feat_false"] == 0
        if shape_ok and feat_ok:
            mode = best["auto_mode"]
        elif shape_ok:
            mode = "shape"
        elif feat_ok:
            mode = "features"
        elif best["shape_found"] != best["feat_found"]:
            mode = "shape" if best["shape_found"] > best["feat_found"] else "features"
        else:
            mode = best["auto_mode"]
        if mode in ("shape", "both") and ref.shape is None:
            mode = "features"
        if mode == "features" and ref.desc is None:
            mode = "shape"

        report = dict(images=n, primary=best["primary"], mode=mode, auto_mode=best["auto_mode"],
                      learned_background=has_bg,
                      found_by_outline=f"{best['shape_found']}/{others}", found_by_keypoints=f"{best['feat_found']}/{others}",
                      keypoint_matches_on_background=best["feat_false"], limits_before=None, limits=None,
                      edge_sensitivity=(self.cfg.shape_mag_thresh, best["edge_thr"] or self.cfg.shape_mag_thresh),
                      background_candidates=len(best["neg"]), background_accepted_before=0, background_accepted_after=0)

        ref.mode = mode
        ref.cal = None
        if mode in ("shape", "both") and best["pos"]:
            sm = ref.shape
            base = self._shape_limits(sm)
            cal = dict(base)
            pos, neg = best["pos"], best["neg"]
            margin = dict(sc=0.08, cov=0.10, bal=0.10, flat=0.10, ring=0.10)
            lo = {k: min(e[k] for e in pos) for k in margin}
            hi = {k: max(e[k] for e in pos) for k in ("dab", "dl")}
            n_bad = lambda lim: sum(self._shape_accept(sm, e, lim=lim) for e in neg)
            report["background_accepted_before"] = n_bad(base)
            # 1) loosen, one measure at a time, so the real object (as it looks in these images)
            #    passes - by at most 20 %, and only if that lets no background candidate in
            want = {k: max(lo[k] - margin[k], 0.8 * base[k]) for k in margin if lo[k] - margin[k] < base[k]}
            want.update({k: float(min(hi[k] * 1.4 + (6.0 if k == "dab" else 15.0), 1.5 * base[k]))
                         for k in ("dab", "dl") if hi[k] * 1.4 + (6.0 if k == "dab" else 15.0) > base[k]})
            for k, v in want.items():
                trial = dict(cal, **{k: v})
                if n_bad(trial) <= n_bad(cal):
                    cal = trial
            # 2) background candidates that pass: tighten the ONE measure that separates the
            #    real object from them best, and only just enough to reject them
            for _ in range(8):
                passing = [e for e in neg if self._shape_accept(sm, e, lim=cal)]
                if not passing:
                    break
                gap = {k: (lo[k] - max(e[k] for e in passing)) / 0.3 for k in margin}          # higher is better
                gap["dab"] = (min(e["dab"] for e in passing) - hi["dab"]) / 30.0               # lower is better
                gap["dl"] = (min(e["dl"] for e in passing) - hi["dl"]) / 60.0
                k = max(gap, key=gap.get)
                if gap[k] <= 0.15:
                    break                     # too close to the real object: tightening would cost detections
                if k in margin:
                    worst = max(e[k] for e in passing)
                    cal[k] = worst + 0.25 * (lo[k] - worst)
                else:
                    worst = min(e[k] for e in passing)
                    cal[k] = worst - 0.25 * (worst - hi[k])
            report["background_accepted_after"] = sum(self._shape_accept(sm, e, lim=cal) for e in neg)
            report["limits_before"] = {k: round(float(v), 2) for k, v in base.items()}
            report["limits"] = {k: round(float(v), 2) for k, v in cal.items()}
            ref.cal = cal
        if mode == "features":
            ref.shape = None

        self.clear()
        self.refs.append(ref)
        if self.deep is not None:
            self.deep.add_reference(ref.gray)
        report["keypoints"] = None
        if mode in ("features", "both") and ref.desc is not None:
            report["keypoints"] = self._keep_stable_keypoints(ref, frames, boxes, best["primary"], hit)
        report["templates"], report["min_score"] = self._learn_templates_and_threshold(ref, frames, boxes, best["primary"],
                                                                                        hit, background)
        self.calibration = report
        if verbose:
            print(self.calibration_summary())
        return report

    def _learn_templates_and_threshold(self, ref, frames, boxes, primary, hit, background):
        """Register the object's appearance in the other shots as extra templates, then place the
        score threshold for THIS object between how the object scores and how the background of
        the calibration frames scores."""
        ref.min_score = 0.0                      # look at everything the detector proposes
        pos, descs = [], []
        for j, (frame, box) in enumerate(zip(frames, boxes)):
            if j == primary:
                continue
            own = [d for d in self.detect(frame) if hit(d, box)]
            if own:
                d = max(own, key=lambda q: q.score)
                pos.append(d.score)              # scored BEFORE its own shot becomes a template
                descs.append(ref.app.describe(_gray(frame), d.H @ np.diag([1.0 / ref.k, 1.0 / ref.k, 1.0])))
        for v in descs:
            ref.app.learn(v)
        neg = [d.score for frame, box in zip(frames, boxes) for d in self.detect(frame) if background(d, box)]
        base = self.cfg.min_score
        lo, hi = (min(pos) if pos else 1.0), (max(neg) if neg else 0.0)
        thr = base
        if hi >= thr:                            # background of your scene scores high: raise the bar
            thr = min(hi + 0.05, 0.5 * (lo + hi)) if lo > hi + 0.1 else min(hi + 0.05, 0.9)
        if lo - 0.1 < thr and lo > hi + 0.1:     # the real object scores low here: lower the bar, staying above the background
            thr = max(lo - 0.1, hi + 0.05, 0.2)
        ref.min_score = None if abs(thr - base) < 0.01 else float(thr)
        return len(ref.app.templates), dict(default=base, learned=float(thr), object_low=float(lo), background_high=float(hi))

    def _keep_stable_keypoints(self, ref, frames, boxes, primary, hit):
        """Keep only the reference keypoints that are seen again, at the right place, in the other
        calibration images. Keypoints on the surface around the object (inside the crop) do not
        move with the object, so they are not confirmed and are dropped: the model is then the
        object alone, which is both safer and faster to match."""
        saved_mode, ref.mode = ref.mode, "features"
        votes = np.zeros(len(ref.desc), np.int32)
        used = 0
        for j, (frame, box) in enumerate(zip(frames, boxes)):
            if j == primary:
                continue
            own = [d for d in self.detect(frame) if hit(d, box)]
            if not own:
                continue
            d = max(own, key=lambda q: q.score)
            g_raw, f = self.prepare(frame)
            g = self.clahe.apply(g_raw) if self.clahe is not None else g_raw
            r = self.ext_scene(g, None, 0)
            if r is None:
                continue
            s_pts, _, _, s_desc = r
            H = np.diag([f, f, 1.0]) @ d.H @ np.diag([1.0 / ref.k, 1.0 / ref.k, 1.0])   # internal ref px -> prepared frame
            proj = _project(H, ref.pts_ref)
            radius = 3.0 + 0.02 * float(np.sqrt(max(d.rect[2] * d.rect[3], 1.0))) * f
            pairs = cKDTree(proj).sparse_distance_matrix(cKDTree(s_pts), radius, output_type="ndarray")
            if len(pairs):
                sim = np.einsum("ij,ij->i", ref.desc[pairs["i"]], s_desc[pairs["j"]])
                votes[np.unique(pairs["i"][sim > 0.75])] += 1
            used += 1
        ref.mode = saved_mode
        total = len(votes)
        if used == 0:
            return dict(before=total, after=total)
        keep = votes >= max(1, (used + 1) // 2)
        if keep.sum() < max(40, 0.15 * total):       # too few confirmed: keep the full model
            return dict(before=total, after=total)
        for name in ("pts_ref", "pts_view", "ctr_view", "kp_size", "kp_angle", "desc"):
            setattr(ref, name, getattr(ref, name)[keep])
        if hasattr(ref, "matcher"):
            del ref.matcher
        if len(ref.desc) > 12000:
            ref.matcher = cv2.FlannBasedMatcher(dict(algorithm=1, trees=4), dict(checks=64))
            ref.matcher.add([ref.desc])
            ref.matcher.train()
        return dict(before=total, after=int(keep.sum()))

    def calibration_summary(self) -> str:
        r = getattr(self, "calibration", None)
        if not r:
            return "not calibrated"
        how = {"shape": "outline + colour", "features": "keypoints", "both": "keypoints and outline"}[r["mode"]]
        out = [f"calibrated from {r['images']} images: model from image {r['primary'] + 1}, matched by {how}",
               f"  object found in the other images: outline {r['found_by_outline']}, keypoints {r['found_by_keypoints']}"
               + (f" (keypoints also matched the background {r['keypoint_matches_on_background']}x)"
                  if r["keypoint_matches_on_background"] else "")]
        if r.get("keypoints") and r["keypoints"]["after"] < r["keypoints"]["before"]:
            out.append(f"  keypoints kept: {r['keypoints']['after']} of {r['keypoints']['before']} "
                       f"(the rest were not on the object)")
        if r["mode"] != "features" and abs(r["edge_sensitivity"][1] - r["edge_sensitivity"][0]) > 0.5:
            a, b = r["edge_sensitivity"]
            out.append(f"  edge threshold {a:.0f}->{b:.0f} (this object's outline is faint, so fainter edges now count)")
        if r.get("min_score"):
            m = r["min_score"]
            out.append(f"  registered templates: {r['templates']}; object scores at least {m['object_low']:.2f}, background at most "
                       f"{m['background_high']:.2f} -> score threshold {m['default']:.2f}" +
                       (f" -> {m['learned']:.2f}" if abs(m["learned"] - m["default"]) >= 0.01 else " (kept)"))
        if r["limits"]:
            ch = [f"{k} {r['limits_before'][k]}->{v}" for k, v in r["limits"].items() if abs(v - r["limits_before"][k]) > 0.005]
            out.append("  accept limits: " + (", ".join(ch) if ch else "defaults kept"))
            if r["learned_background"]:
                out.append(f"  background candidates in your frames: {r['background_candidates']}, accepted before "
                           f"{r['background_accepted_before']}, after {r['background_accepted_after']}")
        return "\n".join(out)

    def clear(self):
        self.calibration = None
        self.refs.clear()
        if self.deep is not None:
            self.deep.clear()

    def _build_template(self, ref, user_mask: bool):
        from advanced_template import TemplateMatcher
        c = self.cfg
        # Made for scenes whose background is like the reference's and where nothing covers the
        # object. On a plain surface the whole crop (object + its surroundings) is the template.
        # On a textured surface (wood grain, fabric, gravel) the surface pattern around the object
        # differs at every spot, so the object is cut out (keeping a thin band for its outline).
        mask = ref.mask if user_mask else None
        zmask = None
        ref.template_surface = "your mask" if user_mask else "plain surface: whole crop"
        if not user_mask:
            # measured on the crop at its original size (the internal copy may be enlarged, which smooths edges)
            orig = cv2.resize(ref.color, (max(8, round(ref.w / ref.k)), max(8, round(ref.h / ref.k))),
                              interpolation=cv2.INTER_AREA) if ref.k != 1.0 else ref.color
            gx, gy = _color_grad(orig, c.shape_pre_blur)
            mag = cv2.magnitude(gx, gy)
            thr = _edge_threshold(mag, c.shape_mag_thresh)
            h, w = mag.shape
            band = np.ones((h, w), bool)
            band[4:-4, 4:-4] = False                                       # the margin around the object: pure surface
            if float((mag[band] > thr).mean()) > 0.05:                     # the surface itself has texture
                m = self._cut_out(ref.color)
                if m is not None:
                    # edges: the outline itself must be inside (about 2.5 original px of margin);
                    # grey check: plus a thin ring of surface, since a one-colour object has no grey detail
                    # of its own and the object/surface contrast is what stays put on a similar surface
                    ke = 2 * int(math.ceil(2.5 * ref.k)) + 1
                    mask = cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ke, ke)))
                    kz = 2 * int(0.06 * max(ref.w, ref.h)) + 1
                    zmask = cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kz, kz)))
                    ref.template_surface = "textured surface: object cut out"
                else:
                    ref.template_surface = "textured surface: cut-out failed, whole crop"
        ref.template_plain = False
        smin, smax = c.template_scale_range
        return TemplateMatcher(ref.gray_raw, mask, scale_range=(smin / ref.k, smax / ref.k),
                               angle_range=c.template_angle_range, coarse_px=c.template_coarse_px,
                               polarity=c.template_polarity, zncc_mask=zmask,
                               upsampled=ref.k, color=ref.color)

    def _calibrate_template(self, frames, boxes, crops, hit, background, has_bg, verbose) -> dict:
        """calibrate() for --advance-template: every shot is tried as the model; the one that finds the
        object best in all the other shots is kept, and its edge / correlation limits are set from the
        real scores (and, with full frames, raised above the best background candidate)."""
        c, n = self.cfg, len(frames)
        trials = []
        for p in range(n):
            self.refs = []
            self.add_reference(crops[p])
            ref = self.refs[0]
            ref.template_min, ref.template_zncc = 0.3, -1.0            # see everything while measuring
            own, bg = [], []
            for j in range(n):
                if j == p:
                    continue
                g_raw, f = self.prepare(frames[j])
                img = frames[j] if f == 1.0 else cv2.resize(frames[j], (g_raw.shape[1], g_raw.shape[0]), interpolation=cv2.INTER_AREA)
                ds = [_attach_rect(_scale_det(d, f), ref) for d in self._detect_template(ref, 0, g_raw, img if img.ndim == 3 else None)]
                mine = [d for d in ds if hit(d, boxes[j])]
                own.append(max(((d.evidence["edge"], d.evidence["zncc"]) for d in mine), default=(0.0, -1.0),
                               key=lambda e: e[0] + e[1]))
                if has_bg:
                    bg += [(d.evidence["edge"], d.evidence["zncc"]) for d in ds if background(d, boxes[j])]
            worst = min(e + z for e, z in own)
            trials.append((worst, p, ref, own, bg))
        worst, p, ref, own, bg = max(trials, key=lambda t: t[0])
        e_min = min(e for e, _ in own)
        z_min = min(z for _, z in own)
        # limits a little below the weakest real view, never looser than 0.35 / 0.4
        e_thr = float(np.clip(e_min - 0.08, 0.35, c.template_min_score))
        z_thr = float(np.clip(z_min - 0.12, 0.40, max(c.template_min_zncc, 0.40)))
        # a background candidate that passes both: tighten the limit that separates it (if the object allows)
        for e, z in sorted(bg, key=lambda t: -(t[0] + t[1])):
            if e >= e_thr and z >= z_thr:
                if z + 0.03 <= z_min - 0.02:
                    z_thr = z + 0.03
                elif e + 0.03 <= e_min - 0.02:
                    e_thr = e + 0.03
        ref.template_min, ref.template_zncc = e_thr, z_thr
        self.refs = [ref]
        found = sum(1 for e, z in own if e >= e_thr and z >= z_thr)
        rep = dict(method="template", model_image=p, found=f"{found}/{len(own)}",
                   edge_scores=[round(e, 2) for e, _ in own], zncc_scores=[round(z, 2) for _, z in own],
                   edge_limit=round(e_thr, 2), zncc_limit=round(z_thr, 2), background_candidates=len(bg),
                   surface=ref.template_surface)
        self.calibration = rep
        if verbose:
            print(f"calibration (advanced template): model = image {p + 1}, finds the object in {rep['found']} other shots; "
                  f"limits edge >= {e_thr:.2f}, correlation >= {z_thr:.2f} [{ref.template_surface}]")
        return rep

    @staticmethod
    def _cut_out(color: np.ndarray) -> Optional[np.ndarray]:
        """Object silhouette in a crop whose outer border is surface: GrabCut with the border as
        definite background, then the largest region with its holes filled."""
        h, w = color.shape[:2]
        k = min(1.0, 240.0 / max(h, w))                                   # GrabCut is slow on big crops
        small = cv2.resize(color, (max(16, round(w * k)), max(16, round(h * k))), interpolation=cv2.INTER_AREA) if k < 1 else color
        sh, sw = small.shape[:2]
        gm = np.full((sh, sw), cv2.GC_PR_FGD, np.uint8)
        b = max(2, round(4 * k))
        gm[:b, :] = gm[-b:, :] = cv2.GC_BGD
        gm[:, :b] = gm[:, -b:] = cv2.GC_BGD
        try:
            bg, fg = np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64)
            cv2.grabCut(small, gm, None, bg, fg, 5, cv2.GC_INIT_WITH_MASK)
        except cv2.error:
            return None
        m = np.where((gm == cv2.GC_FGD) | (gm == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
        n, lab, stats, _ = cv2.connectedComponentsWithStats(m)
        if n <= 1:
            return None
        big = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        m = np.where(lab == big, 255, 0).astype(np.uint8)
        cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        m = cv2.drawContours(np.zeros_like(m), cs, -1, 255, -1)            # fill holes
        frac = float((m > 0).mean())
        if not 0.1 < frac < 0.97:
            return None
        return cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST) if k < 1 else m

    def _detect_template(self, ref, ri, g, color=None) -> List[Detection]:
        c = self.cfg
        m = getattr(ref, "template_min", None)
        thr = c.template_min_score if m is None else m
        mz = getattr(ref, "template_zncc", None)
        zthr = c.template_min_zncc if mz is None else mz
        out = []
        for sc, cx, cy, ang, s, z in ref.tm.find(g, min_score=thr, max_instances=c.max_instances,
                                                 max_candidates=c.template_max_candidates, min_zncc=zthr,
                                                 color=color):
            H = ref.tm.pose_matrix(cx, cy, ang, s)                        # internal ref px -> detection-res frame
            d = Detection(polygon=_project(H, ref.corners).astype(np.float32), H=H @ np.diag([ref.k, ref.k, 1.0]),
                          ref_id=ri, score=float(sc), inliers=0, angle=float(ang), scale=float(s * ref.k), method="template")
            d.evidence = dict(edge=float(sc), zncc=float(z))
            out.append(d)
        return out

    def describe(self) -> str:
        out = []
        for i, r in enumerate(self.refs):
            parts = []
            if r.mode == "template" and r.tm is not None:
                parts.append(f"advanced template: {r.tm.n_templates} rotated/scaled templates, angle step "
                             f"{r.tm.angle_step:.0f} deg, symmetry {('round' if r.tm.sym == 0 else str(r.tm.sym) + '-fold') if r.tm.sym != 1 else 'none'}, "
                             f"{getattr(r, 'template_surface', 'whole crop')}, "
                             f"{'polarity kept' if self.cfg.template_polarity else 'any contrast polarity'}")
            if r.mode in ("shape", "both") and r.shape is not None:
                parts.append(f"shape: {r.shape.n_edges} edge points, {r.shape.n_templates} templates, "
                             f"colour check {'on' if r.shape.ref_lab is not None else 'off'}")
            if r.mode in ("features", "both") and r.desc is not None:
                parts.append(f"features: {r.n_base_kp} keypoints")
            out.append(f"ref{i}: {r.w}x{r.h} mode={r.mode} ({'; '.join(parts)})")
        return "\n".join(out)

    # ------------------------------------------------------------- detection
    def prepare(self, frame: np.ndarray):
        g = _gray(frame)
        h, w = g.shape
        f = min(1.0, self.cfg.max_side / max(h, w))
        if f < 1.0:
            g = cv2.resize(g, (round(w * f), round(h * f)), interpolation=cv2.INTER_AREA)
        return g, f

    def detect(self, frame: np.ndarray) -> List[Detection]:
        import time
        t0 = time.perf_counter()
        g_raw, f = self.prepare(frame)
        self._gray_raw = g_raw
        dets: List[Detection] = []
        timing = {}

        feat_refs = [i for i, r in enumerate(self.refs) if r.mode in ("features", "both") and r.desc is not None]
        shape_refs = [i for i, r in enumerate(self.refs) if r.mode in ("shape", "both") and r.shape is not None]
        tmpl_refs = [i for i, r in enumerate(self.refs) if r.mode == "template" and r.tm is not None]
        if tmpl_refs:
            t = time.perf_counter()
            img = frame if f == 1.0 else cv2.resize(frame, (g_raw.shape[1], g_raw.shape[0]), interpolation=cv2.INTER_AREA)
            img = img if img.ndim == 3 else None
            for ri in tmpl_refs:
                dets += self._detect_template(self.refs[ri], ri, g_raw, img)
            timing["template"] = time.perf_counter() - t
        g = self.clahe.apply(g_raw) if (self.clahe is not None and feat_refs) else g_raw

        if feat_refs and self.deep is not None:
            t = time.perf_counter()
            found: List[Detection] = []
            for _ in range(self.cfg.deep_rounds):   # LightGlue is one-to-one: re-match to find more instances
                new = []
                for ri, (rp, sp) in self.deep.match(g, exclude=[d.polygon for d in found]).items():
                    if ri in feat_refs:
                        new += self._detect_from_correspondences(self.refs[ri], ri, rp, sp, g)
                new = [d for d in _nms(new, self.cfg.nms_iou)
                       if all(poly_iou(d.polygon, f.polygon) < self.cfg.nms_iou for f in found)]
                if not new:
                    break
                found += new
            dets += found
            timing["deep"] = time.perf_counter() - t
        elif feat_refs:
            t = time.perf_counter()
            r = self.ext_scene(g, None, self.cfg.scene_features)
            timing["keypoints"] = time.perf_counter() - t
            if r is not None and len(r[0]) >= self.cfg.min_votes:
                s_pts, s_size, s_ang, desc = r
                t = time.perf_counter()
                for ri in feat_refs:
                    dets += self._detect_features(self.refs[ri], ri, s_pts, s_size, s_ang, desc, g)
                timing["match+geom"] = time.perf_counter() - t

        if shape_refs:
            t = time.perf_counter()
            img = frame if f == 1.0 else cv2.resize(frame, (g_raw.shape[1], g_raw.shape[0]), interpolation=cv2.INTER_AREA)
            ctxs = {}
            shape_dets = []
            for ri in shape_refs:
                shape_dets += self._detect_shape(self.refs[ri], ri, img, ctxs)
            # "both" mode: a feature match must also show the reference's plain region; otherwise
            # it matched only the surroundings that were inside the reference crop
            dets = [d for d in dets if self._plain_region_present(d, img, ctxs)]
            dets += shape_dets
            timing["shape"] = time.perf_counter() - t

        dets = _nms(dets, self.cfg.nms_iou)[: self.cfg.max_instances]
        dets = [_attach_rect(_scale_det(d, f), self.refs[d.ref_id]) for d in dets]
        timing["total"] = time.perf_counter() - t0
        self.last_timing = timing
        return dets

    # -- features path -------------------------------------------------------
    def _knn3(self, ref: _Reference, desc: np.ndarray):
        """Exact 3-NN. Returns (idx, dist, sim_fn); sim_fn(q, t) gives descriptor similarity
        for arbitrary (scene, reference) pairs, used by guided matching."""
        n_ref = len(ref.desc)
        if n_ref <= 12000:
            # RootSIFT vectors are unit-norm, so d^2 = 2 - 2<a,b>: one BLAS matmul
            sims = desc @ ref.desc.T
            rows = np.arange(len(desc))
            idx = np.zeros((len(desc), 3), np.int64)
            val = np.full((len(desc), 3), -1.0, np.float32)
            for j in range(min(3, n_ref)):        # 3 argmax passes: ~3x faster than argpartition
                idx[:, j] = sims.argmax(1)
                val[:, j] = sims[rows, idx[:, j]]
                sims[rows, idx[:, j]] = -2.0
            for j in range(min(3, n_ref)):        # restore for guided matching
                sims[rows, idx[:, j]] = val[:, j]
            dist = np.sqrt(np.maximum(2.0 - 2.0 * val, 0.0))
            return idx, dist, (lambda q, t: sims[q, t])
        knn = ref.matcher.knnMatch(desc, k=3)  # very big ASIFT banks: approximate (FLANN)
        idx = np.zeros((len(desc), 3), np.int64); dist = np.full((len(desc), 3), np.inf, np.float32)
        for i, m in enumerate(knn):
            for j, mm in enumerate(m[:3]):
                idx[i, j] = mm.trainIdx; dist[i, j] = mm.distance
        return idx, dist, (lambda q, t: np.einsum("ij,ij->i", desc[q], ref.desc[t]))

    def _ratio_matches(self, ref: _Reference, desc: np.ndarray):
        idx, dist, sim_fn = self._knn3(ref, desc)
        pr = ref.pts_ref
        min_sep2 = (0.02 * ref.dim) ** 2
        # 2nd neighbour must be a *different* reference location (views / duplicate orientations)
        second = dist[:, 1].copy() if idx.shape[1] > 1 else np.full(len(idx), np.inf)
        if idx.shape[1] > 2:
            same1 = np.sum((pr[idx[:, 1]] - pr[idx[:, 0]]) ** 2, axis=1) <= min_sep2
            same2 = np.sum((pr[idx[:, 2]] - pr[idx[:, 0]]) ** 2, axis=1) <= min_sep2
            second = np.where(same1, np.where(same2, np.inf, dist[:, 2]), second)
        ok = dist[:, 0] < self.cfg.ratio * second
        q = np.nonzero(ok)[0].astype(np.int32)
        return q, idx[ok, 0].astype(np.int32), sim_fn

    def _hough(self, ref, s_pts, s_size, s_ang, q, t):
        c = self.cfg
        sigma = s_size[q] / ref.kp_size[t]
        phi = np.deg2rad(s_ang[q] - ref.kp_angle[t])
        v = ref.ctr_view[t] - ref.pts_view[t]
        cs, sn = np.cos(phi), np.sin(phi)
        cx = s_pts[q, 0] + sigma * (cs * v[:, 0] - sn * v[:, 1])
        cy = s_pts[q, 1] + sigma * (sn * v[:, 0] + cs * v[:, 1])
        us = np.log2(sigma)
        ut = (np.rad2deg(phi) % 360.0) / 30.0
        s0 = np.floor(us).astype(np.int64)
        t0 = np.floor(ut).astype(np.int64)
        n = len(q)
        keys, members = [], []
        idx = np.arange(n)
        for dsb in (0, 1):          # each vote goes to the 2 nearest bins in every dimension
            sb = s0 + dsb
            L = np.maximum(c.hough_pos_frac * ref.dim * np.power(2.0, sb), 4.0)
            px, py = cx / L, cy / L
            x0, y0 = np.floor(px).astype(np.int64), np.floor(py).astype(np.int64)
            for dtb in (0, 1):
                tb = (t0 + dtb) % 12
                for dx in (0, 1):
                    for dy in (0, 1):
                        k = (((sb + 64) * 16 + tb) << 40) + ((x0 + dx + (1 << 19)) << 20) + (y0 + dy + (1 << 19))
                        keys.append(k)
                        members.append(idx)
        keys = np.concatenate(keys)
        members = np.concatenate(members)
        uniq, inv, counts = np.unique(keys, return_inverse=True, return_counts=True)
        good = np.nonzero(counts >= c.min_votes)[0]
        if len(good) == 0:
            return []
        order = np.argsort(inv, kind="stable")
        starts = np.concatenate([[0], np.cumsum(counts)])
        good = good[np.argsort(-counts[good], kind="stable")][:400]
        return [members[order[starts[b]:starts[b + 1]]] for b in good]

    def _detect_features(self, ref, ri, s_pts, s_size, s_ang, desc, g):
        c = self.cfg
        q, t, sim_fn = self._ratio_matches(ref, desc)
        if len(q) < c.min_votes:
            return []
        clusters = self._hough(ref, s_pts, s_size, s_ang, q, t)
        ctx = dict(s_pts=s_pts, sim_fn=sim_fn, gsim=self.cfg.guided_sim, q=q, t=t,
                   src=ref.pts_ref[t], dst=s_pts[q], used_scene=np.zeros(len(s_pts), bool))
        failed = np.zeros(len(q), bool)
        dets = []
        for members in clusters:
            members = members[~ctx["used_scene"][q[members]]]
            if len(members) < c.min_votes or np.all(failed[members]):
                continue
            det = self._fit_cluster(ref, ri, members, ctx, g)
            if det is None:
                failed[members] = True
                continue
            dets.append(det)
            ctx["used_scene"] |= _points_in_convex(det.polygon, s_pts)
            if len(dets) >= c.max_instances:
                break
        return dets

    def _fit_model(self, src, dst):
        """Affine for few points (better conditioned), homography (MAGSAC++) for many."""
        c = self.cfg
        if len(src) >= 15:
            H, m = cv2.findHomography(src, dst, cv2.USAC_MAGSAC, c.reproj_thresh, maxIters=2000, confidence=0.999)
        else:
            A, m = cv2.estimateAffine2D(src, dst, method=cv2.RANSAC, ransacReprojThreshold=c.reproj_thresh)
            H = None if A is None else np.vstack([A, [0, 0, 1]])
        if H is None or not np.all(np.isfinite(H)) or abs(H[2, 2]) < 1e-9:
            return None
        return H / H[2, 2]

    def _guided(self, ref, H, ctx, radius):
        """Guided matching: for scene keypoints inside the predicted object, accept the most
        similar reference keypoint whose projection lands within `radius` px. The geometric
        constraint replaces the ratio test, recovering many matches on small / blurry objects."""
        s_pts = ctx["s_pts"]
        poly = _project(H, ref.corners)
        ctr = poly.mean(0)
        poly_d = ctr + (poly - ctr) * 1.08
        cand = np.nonzero(_points_in_convex(poly_d, s_pts) & ~ctx["used_scene"])[0]
        if len(cand) == 0:
            return cand, cand
        proj = _project(H, ref.pts_ref)
        rin = np.nonzero(_points_in_convex(poly_d, proj))[0]
        if len(rin) == 0:
            return rin, rin
        pairs = cKDTree(s_pts[cand]).sparse_distance_matrix(cKDTree(proj[rin]), radius, output_type="ndarray")
        if len(pairs) == 0:
            return pairs["i"], pairs["j"]
        qi, tj = cand[pairs["i"]], rin[pairs["j"]]
        sim = ctx["sim_fn"](qi, tj)
        keep = sim > ctx["gsim"]
        qi, tj, sim = qi[keep], tj[keep], sim[keep]
        o = np.lexsort((-sim, qi))                 # best reference match per scene keypoint
        qi, tj = qi[o], tj[o]
        _, first = np.unique(qi, return_index=True)
        return qi[first], tj[first]

    def _fit_cluster(self, ref, ri, members, ctx, g):
        c = self.cfg
        src_all, dst_all = ctx["src"], ctx["dst"]
        src, dst = src_all[members], dst_all[members]
        M, inl = cv2.estimateAffinePartial2D(src, dst, method=cv2.RANSAC,
                                             ransacReprojThreshold=3 * c.reproj_thresh,
                                             maxIters=500, confidence=0.995)
        if M is None or inl is None or inl.sum() < 3:
            return None
        H = np.vstack([M, [0, 0, 1]])
        s0 = math.sqrt(abs(np.linalg.det(M[:, :2])))
        if not (c.min_scale <= s0 <= c.max_scale):
            return None
        # 1) expand with ratio-test matches consistent with the seed, refit
        cand = np.nonzero(~ctx["used_scene"][ctx["q"]])[0]
        src_c, dst_c = src_all[cand], dst_all[cand]
        thr_seed = max(3 * c.reproj_thresh, 0.06 * s0 * ref.dim)
        for thr in (thr_seed, 2 * c.reproj_thresh):
            err = np.linalg.norm(_project(H, src_c) - dst_c, axis=1)
            sel = cand[err < thr]
            if len(sel) < 4:
                return None
            Hn = self._fit_model(src_all[sel], dst_all[sel])
            if Hn is None:
                return None
            H = Hn
        err = np.linalg.norm(_project(H, src_c) - dst_c, axis=1)
        n_ratio = len(np.unique(np.round(dst_c[err < 2 * c.reproj_thresh]).astype(np.int32), axis=0))
        if n_ratio < max(5, c.min_inliers // 2) or not self._geometry_ok(ref, H, g.shape):
            return None   # guided matching is only spent on clusters that already look real
        # 2) guided matching around the current pose, refit (twice, shrinking the radius)
        best = None
        if ctx["sim_fn"] is not None:
            for radius in (max(2.5 * c.reproj_thresh, 0.05 * s0 * ref.dim), 1.5 * c.reproj_thresh):
                gq, gt = self._guided(ref, H, ctx, radius)
                if len(gq) < 4:
                    break
                src_g, dst_g = ref.pts_ref[gt], ctx["s_pts"][gq]
                Hn = self._fit_model(src_g, dst_g)
                if Hn is None:
                    break
                H = Hn
                e = np.linalg.norm(_project(H, src_g) - dst_g, axis=1) < c.reproj_thresh
                best = (src_g[e], dst_g[e])
        if best is None:  # no similarity matrix (FLANN path) -> ratio matches only
            err = np.linalg.norm(_project(H, src_c) - dst_c, axis=1)
            sel = cand[err < c.reproj_thresh]
            best = (src_all[sel], dst_all[sel])
        return self._verify(ref, ri, H, best[0], best[1], g, "features")

    def _detect_from_correspondences(self, ref, ri, rp, sp, g):
        """Multi-instance sequential MAGSAC for matchers without keypoint scale/orientation."""
        c = self.cfg
        dets = []
        alive = np.ones(len(rp), bool)
        for _ in range(c.max_instances):
            idx = np.nonzero(alive)[0]
            if len(idx) < c.min_inliers:
                break
            H, m = cv2.findHomography(rp[idx], sp[idx], cv2.USAC_MAGSAC, c.reproj_thresh,
                                      maxIters=5000, confidence=0.999)
            if H is None or m is None or m.sum() < c.min_inliers:
                break
            inl = idx[m.ravel() > 0]
            det = self._verify(ref, ri, H / H[2, 2], rp[inl], sp[inl], g, "lightglue")
            if det is None:
                alive[inl] = False
                continue
            dets.append(det)
            alive &= ~_points_in_convex(det.polygon, sp)
        return dets

    # -- verification --------------------------------------------------------
    def _verify(self, ref, ri, H, src, dst, g, method) -> Optional[Detection]:
        c = self.cfg
        if src is None or len(src) < c.min_inliers:
            return None
        # inliers counted once per distinct location on both sides
        n_s = len(np.unique(np.round(dst / 2).astype(np.int32), axis=0))
        n_r = len(np.unique(np.round(src / 2).astype(np.int32), axis=0))
        n_in = min(n_s, n_r)
        if n_in < c.min_inliers:
            return None
        if not self._geometry_ok(ref, H, g.shape):
            return None
        # inlier support must not be a tiny blob on the reference
        hull = cv2.convexHull(src.astype(np.float32))
        if cv2.contourArea(hull) < 0.01 * ref.w * ref.h:
            return None
        ncc = self._appearance(ref, H, g) if c.verify_appearance else 1.0
        if ncc < c.min_ncc and n_in < 3 * c.min_inliers:
            return None
        hog = ref.app.score(ref.app.describe(self._gray_raw, H)) if c.verify_appearance else 1.0   # registered-template match
        score = _feature_probability(hog, n_in, max(ncc, 0.0)) if c.verify_appearance else 1.0 - math.exp(-n_in / 15.0)
        if score < self._min_score(ref):
            return None
        poly = _project(H, ref.corners).astype(np.float32)
        ang, scl = _pose_from_H(H, *ref.center)
        Hn = H @ np.diag([ref.k, ref.k, 1.0])   # map ORIGINAL reference px -> frame
        d = Detection(polygon=poly, H=Hn, ref_id=ri, score=float(score), inliers=int(n_in),
                      angle=ang, scale=scl * ref.k, method=method,
                      frame_pts=dst.astype(np.float32), ref_pts=(src / ref.k).astype(np.float32))
        d.evidence = dict(inliers=int(n_in), ncc=float(ncc), hog=float(hog))
        return d

    def _geometry_ok(self, ref, H, frame_shape) -> bool:
        c = self.cfg
        if not np.all(np.isfinite(H)):
            return False
        ph = np.c_[ref.corners, np.ones(4)] @ H.T
        if np.any(ph[:, 2] <= 1e-6):
            return False
        poly = ph[:, :2] / ph[:, 2:3]
        if not _is_convex(poly):
            return False
        J = _local_affine(H, *ref.center)
        det = np.linalg.det(J)
        if det <= 0:                       # mirrored
            return False
        s = math.sqrt(det)
        if not (c.min_scale <= s <= c.max_scale):
            return False
        sv = np.linalg.svd(J, compute_uv=False)
        if sv[0] / max(sv[1], 1e-9) > c.max_anisotropy:
            return False
        sc = [math.sqrt(max(np.linalg.det(_local_affine(H, *p)), 1e-12)) for p in ref.corners]
        if max(sc) / max(min(sc), 1e-9) > c.max_perspective:
            return False
        area = abs(cv2.contourArea(poly.astype(np.float32)))
        if area < 100:
            return False
        fh, fw = frame_shape[:2]
        frame_poly = np.float32([[0, 0], [fw, 0], [fw, fh], [0, fh]])
        inter, _ = cv2.intersectConvexConvex(poly.astype(np.float32), frame_poly)
        return inter >= 0.25 * area

    def _appearance(self, ref, H, g) -> float:
        S = np.diag([ref.app_k, ref.app_k, 1.0])
        Hs = H @ np.linalg.inv(S)          # small-ref px -> frame px
        warped = cv2.warpPerspective(g, Hs, ref.app_size, flags=cv2.INTER_AREA | cv2.WARP_INVERSE_MAP,
                                     borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        valid = cv2.warpPerspective(np.full(g.shape, 255, np.uint8), Hs, ref.app_size,
                                    flags=cv2.INTER_NEAREST | cv2.WARP_INVERSE_MAP) > 0
        m = ref.app_mask & valid
        if m.sum() < 0.2 * ref.app_mask.sum():
            return 0.0
        a = cv2.GaussianBlur(warped, (3, 3), 0).astype(np.float32)[m]
        b = ref.app_img[m]
        a = a - a.mean(); b = b - b.mean()
        den = math.sqrt(float((a * a).sum() * (b * b).sum())) + 1e-6
        return float((a * b).sum() / den)

    # -- shape path ----------------------------------------------------------
    # -- shape path ----------------------------------------------------------
    def _shape_model(self, ref, shape_hw) -> Tuple["_ShapeModel", float]:
        """Shape model (template bank) for the working resolution of this frame size."""
        rs = round(min(1.0, self.cfg.shape_max_side / max(shape_hw)), 3)
        with self._shape_lock:
            sm = ref.shape_cache.get(rs)
            if sm is None:
                sm = ref.shape_cache[rs] = _ShapeModel(ref, self.cfg, rs, parent=ref.shape)
        return sm, rs

    def _edge_thr(self, ref) -> float:
        """Edge sensitivity for one reference: learned by calibrate(), else the default."""
        return float(getattr(ref, "edge_thr", None) or self.cfg.shape_mag_thresh)

    def shape_context(self, img: np.ndarray, rs: float, thr: Optional[float] = None) -> dict:
        """Per-frame data shared by all shape candidates: working-resolution image,
        colour-edge field and Lab image."""
        work = img if rs == 1.0 else cv2.resize(img, (round(img.shape[1] * rs), round(img.shape[0] * rs)),
                                                interpolation=cv2.INTER_AREA)
        thr = thr or self.cfg.shape_mag_thresh
        field = _ShapeModel.frame_field(work, thr, pre_blur=self.cfg.shape_pre_blur)
        lab = work if work.ndim == 3 else None
        F2 = cv2.merge([field[0], field[1]])
        pyr, cur = [(1.0, F2)], work
        for q in (0.5, 0.25):
            if min(cur.shape[:2]) < 64:
                break
            cur = cv2.resize(cur, (cur.shape[1] // 2, cur.shape[0] // 2), interpolation=cv2.INTER_AREA)
            c2, s2, _ = _ShapeModel.frame_field(cur, thr)
            pyr.append((cur.shape[1] / work.shape[1], cv2.merge([c2, s2])))
        return dict(img=work, field=field, F2=F2, pyr=pyr, lab=lab, rs=rs)

    def _min_score(self, ref) -> float:
        """Score a detection needs: learned by calibrate() for this object, else Config.min_score."""
        m = getattr(ref, "min_score", None)
        return float(self.cfg.min_score if m is None else m)

    def _shape_limits(self, sm) -> dict:
        """Accept limits for one reference: learned by calibrate(), otherwise the defaults."""
        cal = getattr(sm.ref, "cal", None)
        if cal is not None:
            return cal
        c = self.cfg
        # a colourless (grey / black / white) object has no colour evidence to back it up,
        # so its outline must fit more tightly
        k = c.shape_grey_strictness if sm.grey else 1.0
        return dict(sc=c.shape_min_score * k, cov=c.shape_min_coverage * k, bal=c.shape_min_balanced * k,
                    flat=c.shape_min_flat, ring=c.shape_min_ring, dab=c.shape_max_color_diff,
                    dl=c.shape_max_light_diff)

    def _shape_accept(self, sm, ev, relax: float = 1.0, lim: Optional[dict] = None) -> bool:
        L = lim or self._shape_limits(sm)
        if (ev["sc"] < L["sc"] * relax or ev["cov"] < L["cov"] * relax
                or ev["flat"] < L["flat"] * relax or ev["bal"] < L["bal"] * relax):
            return False
        if sm.ref_lab is not None and self.cfg.shape_use_color:
            if ev["dab"] > L["dab"] / relax or ev["dl"] > L["dl"] / relax or ev["ring"] < L["ring"] * relax:
                return False
        return True

    def refit_shape(self, d: Detection, frame: np.ndarray) -> Optional[Detection]:
        """Re-detect a plain object near its previous pose `d` in a new frame: the outline is
        fitted again and must pass the same checks as a fresh detection. Only a padded region
        around the old box is processed, so this takes a few ms."""
        ref = self.refs[d.ref_id] if d.ref_id < len(self.refs) else None
        if ref is None or ref.shape is None:
            return None
        c = self.cfg
        h, w = frame.shape[:2]
        f = min(1.0, c.max_side / max(h, w))
        sm, rs = self._shape_model(ref, (round(h * f), round(w * f)))
        q = f * rs                                              # full-resolution frame -> working px
        p = d.polygon
        pad = 0.4 * float(np.sqrt(max(cv2.contourArea(p.astype(np.float32)), 1.0)))
        x0, y0 = np.maximum(np.floor(p.min(0) - pad), 0).astype(int)
        x1, y1 = np.minimum(np.ceil(p.max(0) + pad), [w, h]).astype(int)
        if x1 - x0 < 16 or y1 - y0 < 16:
            return None
        ctx = self.shape_context(frame[y0:y1, x0:x1], q, self._edge_thr(ref))
        H = np.diag([q, q, 1.0]) @ np.array([[1, 0, -x0], [0, 1, -y0], [0, 0, 1.0]]) @ d.H \
            @ np.diag([1.0 / ref.k, 1.0 / ref.k, 1.0])           # internal ref px -> ROI working px
        J = _local_affine(H, *ref.center)
        cx, cy = _project(H, ref.center[None].astype(np.float64))[0]
        s0 = math.sqrt(abs(float(np.linalg.det(J))))
        ang = -math.degrees(math.atan2(J[1, 0] - J[0, 1], J[0, 0] + J[1, 1]))
        A, _ = sm.refine_pose(ctx["pyr"], cx, cy, ang, s0, c.shape_angle_step, c.shape_scale_step)
        A2 = sm.refine_affine(ctx["field"], A, c.shape_max_tilt)
        ev, ev2 = sm.score(ctx["field"], A, ctx["lab"]), sm.score(ctx["field"], A2, ctx["lab"])
        if ev2["sc"] > ev["sc"] + 0.03 or not self._shape_accept(sm, ev):
            A, ev = A2, ev2
        if not self._shape_accept(sm, ev):
            return None
        T_c = np.array([[1, 0, -ref.center[0]], [0, 1, -ref.center[1]], [0, 0, 1.0]])
        ev["hog"] = ref.app.score(ref.app.describe(_gray(ctx["img"]), np.vstack([A, [0, 0, 1.0]]) @ T_c))
        ev["size"] = math.sqrt(abs(float(np.linalg.det(A[:, :2])))) * ref.dim / ctx["rs"]
        ev["p"] = _shape_probability(ev, sm.grey)
        if ev["p"] < self._min_score(ref):
            return None
        n = self._shape_detection(ref, d.ref_id, sm, A, ev, ctx)
        if n is None:
            return None
        T = np.array([[1, 0, x0], [0, 1, y0], [0, 0, 1.0]])      # ROI -> full frame
        n.H = T @ n.H
        n.polygon = n.polygon + np.float32([x0, y0])
        return _attach_rect(n, ref)

    def _shape_detection(self, ref, ri, sm, A, ev, ctx) -> Optional[Detection]:
        A = sm.canonical(A)
        T_c = np.array([[1, 0, -ref.center[0]], [0, 1, -ref.center[1]], [0, 0, 1]])  # ref px -> centred
        H = np.vstack([A, [0, 0, 1]]) @ T_c
        if not self._geometry_ok(ref, H, ctx["img"].shape):
            return None
        rs = ctx["rs"]
        H = np.diag([1.0 / rs, 1.0 / rs, 1.0]) @ H   # working resolution -> detection resolution
        poly = _project(H, ref.corners).astype(np.float32)
        ang, scl = _pose_from_H(H, *ref.center)
        conf = float(ev["p"]) if "p" in ev else float(np.clip(0.5 * ev["sc"] + 0.3 * ev["cov"] + 0.2 * ev["flat"], 0, 1))
        d = Detection(polygon=poly, H=H @ np.diag([ref.k, ref.k, 1.0]), ref_id=ri, score=conf,
                      inliers=int(ev["cov"] * sm.n_edges), angle=ang, scale=scl * ref.k, method="shape")
        d.shape_pose = A
        return d

    def _plain_region_present(self, d: Detection, img, ctxs) -> bool:
        ref = self.refs[d.ref_id]
        if ref.mode != "both" or ref.shape is None or not ref.shape.plain:
            return True
        sm, rs = self._shape_model(ref, img.shape[:2])
        key = (rs, self._edge_thr(ref))
        ctx = ctxs.get(key)
        if ctx is None:
            ctx = ctxs[key] = self.shape_context(img, rs, key[1])
        H = np.diag([rs, rs, 1.0]) @ d.H @ np.diag([1.0 / ref.k, 1.0 / ref.k, 1.0])    # internal ref px -> working px
        J = _local_affine(H, *ref.center)
        c0 = _project(H, ref.center[None].astype(np.float64))[0]
        ev = sm.score(ctx["field"], np.c_[J, c0], ctx["lab"])     # pose expressed about the reference centre
        # the plain region must be there: plain inside, roughly the right colour, and at least
        # part of its outline visible
        c = self.cfg
        return (ev["flat"] >= 0.45 and ev["cov"] >= 0.15 and ev["dab"] <= 2.0 * c.shape_max_color_diff
                and ev["dl"] <= c.shape_max_light_diff)

    def _detect_shape(self, ref, ri, img, ctxs=None, collect: bool = False):
        """collect=True (used by calibrate): return EVERY refined candidate with its evidence,
        without applying the accept rule."""
        c = self.cfg
        if ref.shape is None or ref.shape.n_templates == 0:
            return []
        sm, rs = self._shape_model(ref, img.shape[:2])
        key = (rs, self._edge_thr(ref))
        ctx = None if ctxs is None else ctxs.get(key)
        if ctx is None:
            ctx = self.shape_context(img, rs, key[1])
            if ctxs is not None:
                ctxs[key] = ctx
        props = sm.propose(ctx["img"], c.shape_proposal_threshold)
        if not props:
            return []
        props.sort(key=lambda p: -p[0])
        kept = []
        for p in props:   # centre-distance NMS on coarse proposals
            r = 0.3 * p[4] * ref.dim
            if all((p[1] - k[1]) ** 2 + (p[2] - k[2]) ** 2 > r * r for k in kept):
                kept.append(p)
            if len(kept) >= max(c.shape_max_proposals, 60 if collect else 0):
                break
        dets = []
        lim = self._shape_limits(sm)
        max_dab = 1.6 * lim["dab"] if c.shape_use_color else 0.0
        need = lim["sc"]
        for rank, (_, cx, cy, a, s) in enumerate(kept):
            gated = sm.precheck(ctx["field"], ctx["lab"], cx, cy, a, s, max_dab)
            if not collect and not gated:
                continue
            A, val = sm.refine_pose(ctx["pyr"], cx, cy, a, s, c.shape_angle_step, c.shape_scale_step,
                                    reject=-1.0 if collect else 0.55 * need)
            if A is None or (not collect and val < 0.75 * need):   # weak outline fit: not worth the full check
                continue
            # would the normal (non-collect) detector have looked at this candidate at all?
            gated = gated and val >= 0.75 * need and rank < c.shape_max_proposals
            size = math.sqrt(abs(np.linalg.det(A[:, :2]))) * ref.dim
            r0 = int(np.clip(0.05 * size, 6, 14))
            A_sim = A
            A = sm.refine_affine(ctx["field"], A, c.shape_max_tilt, radii=(r0, max(4, r0 // 2), 3, 2, 2))
            ev = sm.score(ctx["field"], A, ctx["lab"])
            if A is not A_sim:      # keep the undistorted pose unless the affine fit is clearly better
                ev_sim = sm.score(ctx["field"], A_sim, ctx["lab"])
                if ev_sim["sc"] >= ev["sc"] - 0.03 and (self._shape_accept(sm, ev_sim) or not self._shape_accept(sm, ev)):
                    A, ev = A_sim, ev_sim
            if not collect and not self._shape_accept(sm, ev):
                continue
            if sm.sym == 1 and sm.alt_angles:      # nearly symmetric outline: is a rotated pose a better fit?
                ang0 = -math.degrees(math.atan2(A[1, 0] - A[0, 1], A[0, 0] + A[1, 1]))
                s0 = math.sqrt(abs(np.linalg.det(A[:, :2])))
                for alt in sm.alt_angles:
                    A2, _ = sm.refine_pose(ctx["pyr"], A[0, 2], A[1, 2], ang0 + alt, s0,
                                           c.shape_angle_step, c.shape_scale_step)
                    ev2 = sm.score(ctx["field"], A2, ctx["lab"])
                    if ev2["sc"] > ev["sc"] + 0.02 and (collect or self._shape_accept(sm, ev2)):
                        A, ev = A2, ev2
            if "gray" not in ctx:
                ctx["gray"] = _gray(ctx["img"])
            T_c = np.array([[1, 0, -ref.center[0]], [0, 1, -ref.center[1]], [0, 0, 1.0]])
            ev["hog"] = ref.app.score(ref.app.describe(ctx["gray"], np.vstack([A, [0, 0, 1.0]]) @ T_c))
            ev["size"] = math.sqrt(abs(float(np.linalg.det(A[:, :2])))) * ref.dim / ctx["rs"]      # object size, px
            ev["p"] = _shape_probability(ev, sm.grey)       # learned verifier: one probability from all the evidence
            if not collect and ev["p"] < self._min_score(ref):
                continue
            d = self._shape_detection(ref, ri, sm, A, ev, ctx)
            if d is not None:
                d.evidence = ev
                if collect:       # how strong is this outline in the frame? (lower quartile of the edge strength)
                    if "mag" not in ctx:
                        ctx["mag"] = cv2.magnitude(*_color_grad(ctx["img"], c.shape_pre_blur))
                    pts, _ = sm._edge_frame(A)
                    v = cv2.remap(ctx["mag"], pts[:, 0][None], pts[:, 1][None], cv2.INTER_LINEAR, borderValue=0)[0]
                    d.evidence = dict(ev, contrast=float(np.percentile(v, 25)), gated=bool(gated))
                dets.append(d)
        return dets


# ═════════════════════════════════════════════════════════════════ tracker
class _RectFilter:
    """One-Euro filter (Casiez et al. 2012) on (cx, cy, log w, log h, angle).
    Cut-off frequency rises with speed: steady when still, responsive when moving."""

    def __init__(self, min_cutoff: float, beta: float, d_cutoff: float = 1.0):
        # per-dimension speed scaling: pixels, log-size (x100 ~ percent), degrees
        self.k = np.array([1.0, 1.0, 100.0, 100.0, 1.0])
        self.min_cutoff, self.beta, self.d_cutoff = min_cutoff, beta, d_cutoff
        self.x = None
        self.dx = np.zeros(5)

    @staticmethod
    def _alpha(cutoff, dt):
        tau = 1.0 / (2 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def __call__(self, rect, dt):
        cx, cy, w, h, ang = rect
        z = np.array([cx, cy, math.log(max(w, 1e-3)), math.log(max(h, 1e-3)), ang], np.float64)
        if self.x is None:
            self.x = z
        else:
            z[4] = self.x[4] + ((z[4] - self.x[4] + 180.0) % 360.0 - 180.0)   # unwrap angle
            dz = (z - self.x) / dt
            self.dx += self._alpha(self.d_cutoff, dt) * (dz - self.dx)
            cutoff = self.min_cutoff + self.beta * np.abs(self.dx * self.k)
            a = np.array([self._alpha(c, dt) for c in cutoff])
            self.x = self.x + a * (z - self.x)
        x = self.x
        return (float(x[0]), float(x[1]), float(math.exp(x[2])), float(math.exp(x[3])),
                (float(x[4]) + 180.0) % 360.0 - 180.0)


class Tracker:
    """
    Real-time wrapper. Full detection every `redetect_every` frames (or as soon as a track
    weakens); Lucas-Kanade + homography tracking on every frame in between.

    async_detect=True (default) runs detection in a background thread: the video loop never
    waits for it, so the frame rate is the tracking rate (a few ms per frame). When a
    detection finishes, its results are carried forward with optical flow from the frame
    it was computed on to the current frame, then merged with the live tracks.
    """

    def __init__(self, detector: FewShotDetector, redetect_every: int = 5, max_misses: int = 3,
                 async_detect: bool = True, smooth: bool = True, fps: float = 30.0,
                 min_cutoff: float = 0.5, beta: float = 0.3):
        """smooth: One-Euro filter on each track's rectangle (centre, size, angle): strong
        smoothing when the object is still (no jitter), little lag when it moves fast.
        min_cutoff lower = steadier box; beta higher = less lag on fast motion."""
        self.det = detector
        self.smooth = smooth
        self.dt = 1.0 / fps
        self.min_cutoff, self.beta = min_cutoff, beta
        self.redetect_every = redetect_every
        self.max_misses = max_misses
        self.async_detect = async_detect
        self.tracks: List[dict] = []
        self.prev = None
        self.frame_idx = 0
        self.last_det_frame = -10 ** 9
        self.next_id = 0
        self.force = True
        self.lk = dict(winSize=(21, 21), maxLevel=3,
                       criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
        self.last_was_detection = False
        self._pool = None
        self._future = None
        self._hist = []        # grays since the pending detection's frame (for step-wise catch-up)
        self._frame = None     # current colour frame (plain objects are re-fitted on it)

    def reset(self):
        """Call after adding/removing references."""
        if self._future is not None:
            try:
                self._future.result()
            except Exception:
                pass
            self._future = None
            self._hist = []
        self.tracks.clear(); self.prev = None; self.force = True

    def close(self):
        self.reset()
        if self._pool is not None:
            self._pool.shutdown(wait=True)
            self._pool = None

    # ---------------------------------------------------------------- main loop
    def update(self, frame: np.ndarray) -> List[Detection]:
        g, f = self.det.prepare(frame)
        self._frame = frame
        self.last_was_detection = False
        if self.prev is not None and self.tracks:
            self._track_all(self.prev, g, f)

        if self._future is not None:
            self._hist.append(g)
            if self._future.done():                                 # async result arrived
                dets = self._future.result()
                self._future = None
                self._merge(dets, self._hist, f)
                self._hist = []
                self.last_was_detection = True

        due = self.force or not self.tracks or (self.frame_idx - self.last_det_frame >= self.redetect_every)
        if due and self._future is None and self.det.refs:
            self.force = False
            self.last_det_frame = self.frame_idx
            if self.async_detect:
                if self._pool is None:
                    from concurrent.futures import ThreadPoolExecutor
                    self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="fewshot-detect")
                self._hist = [g]
                self._future = self._pool.submit(self.det.detect, frame.copy())
            else:
                self._merge(self.det.detect(frame), [g], f)
                self.last_was_detection = True

        self.prev = g
        self.frame_idx += 1
        out = []
        for tr in self.tracks:
            d = tr["det"]
            d.track_id = tr["id"]
            if self.smooth and d.rect is not None:
                if "filt" not in tr:
                    tr["filt"] = _RectFilter(self.min_cutoff, self.beta)
                d = dataclasses.replace(d, rect=tr["filt"](d.rect, self.dt))
                d.angle = d.rect[4]
            out.append(d)
        return out

    # ---------------------------------------------------------------- helpers
    def _seed(self, d: Detection, g, f) -> dict:
        tr = dict(id=-1, det=d, misses=0, pts=None, ref=None, n0=0)
        if d.method == "shape":       # plain object: followed by re-fitting its outline, no points needed
            return tr
        if d.frame_pts is not None and len(d.frame_pts) >= 8:
            tr["pts"] = (d.frame_pts * f).astype(np.float32)
            tr["ref"] = d.ref_pts.astype(np.float32)
        else:   # shape detections: track corners found inside the object
            mask = np.zeros(g.shape, np.uint8)
            cv2.fillConvexPoly(mask, (d.polygon * f).astype(np.int32), 255)
            mask = cv2.erode(mask, np.ones((7, 7), np.uint8))
            pts = cv2.goodFeaturesToTrack(g, 150, 0.01, 5, mask=mask)
            if pts is not None and len(pts) >= 8:
                pts = pts.reshape(-1, 2).astype(np.float32)
                Hs = np.diag([f, f, 1.0]) @ d.H
                tr["pts"] = pts
                tr["ref"] = _project(np.linalg.inv(Hs), pts).astype(np.float32)
        tr["n0"] = 0 if tr["pts"] is None else len(tr["pts"])
        return tr

    def _refit(self, tr) -> bool:
        """Plain objects have nothing for optical flow to hold on to: fit the outline again."""
        n = self.det.refit_shape(tr["det"], self._frame)
        if n is None:
            return False
        tr["det"] = n
        return True

    def _lk(self, tr, g_from, g_to, f) -> bool:
        """Move a track's points from g_from to g_to and refit its homography."""
        p0 = tr["pts"]
        if p0 is None or len(p0) < 6:
            return False
        p1, st, _ = cv2.calcOpticalFlowPyrLK(g_from, g_to, p0, None, **self.lk)
        pb, st2, _ = cv2.calcOpticalFlowPyrLK(g_to, g_from, p1, None, **self.lk)
        ok = (st.ravel() == 1) & (st2.ravel() == 1) & (np.linalg.norm(pb - p0, axis=1) < 1.0)
        if ok.sum() < 6:
            return False
        src, dst = tr["ref"][ok], p1[ok]
        H, m = cv2.findHomography(src, dst / f, cv2.USAC_MAGSAC, 3.0 / f, maxIters=500, confidence=0.99)
        if H is None or m is None or m.sum() < 6:
            return False
        m = m.ravel() > 0
        d = tr["det"]
        ref = self.det.refs[d.ref_id] if d.ref_id < len(self.det.refs) else None
        if ref is None:
            return False
        Hint = H @ np.diag([1.0 / ref.k, 1.0 / ref.k, 1.0])   # internal ref px -> full-res frame
        if not self.det._geometry_ok(ref, Hint, (g_to.shape[0] / f, g_to.shape[1] / f)):
            return False
        d.H = H
        d.polygon = _project(Hint, ref.corners).astype(np.float32)
        _, s = _pose_from_H(Hint, *ref.center)
        d.scale = s * ref.k
        _attach_rect(d, ref)
        tr["pts"], tr["ref"] = dst[m], src[m]
        d.frame_pts, d.ref_pts = dst[m] / f, src[m]
        return True

    def _track_all(self, g_prev, g, f):
        keep = []
        for tr in self.tracks:
            if tr["det"].method == "shape":
                if self._refit(tr):
                    keep.append(tr)
                else:
                    self.force = True
            elif self._lk(tr, g_prev, g, f):
                if len(tr["pts"]) < 0.4 * max(tr["n0"], 1):
                    self.force = True          # points depleting: refresh soon
                keep.append(tr)
            else:
                self.force = True
        self.tracks = keep

    def _merge(self, dets: List[Detection], hist, f):
        """hist: grays from the detection's frame up to the current one. Async results are
        carried forward frame by frame (small steps keep optical flow accurate)."""
        fresh = []
        for d in dets:
            tr = self._seed(d, hist[0], f)
            if d.method == "shape":
                if len(hist) == 1 or self._refit(tr):     # late result: fit again on the current frame
                    fresh.append(tr)
            elif all(self._lk(tr, a, b, f) for a, b in zip(hist[:-1], hist[1:])):
                fresh.append(tr)
        matched = set()
        for tr in fresh:
            best, best_iou = None, 0.3
            for old in self.tracks:
                iou = poly_iou(tr["det"].polygon, old["det"].polygon)
                if iou > best_iou and old["id"] not in matched:
                    best, best_iou = old, iou
            if best is None:
                tr["id"] = self.next_id
                self.next_id += 1
                self.tracks.append(tr)
                matched.add(tr["id"])
            else:
                best.update(det=tr["det"], pts=tr["pts"], ref=tr["ref"], n0=tr["n0"], misses=0)
                matched.add(best["id"])
        keep = []
        for tr in self.tracks:
            if tr["id"] not in matched:
                tr["misses"] += 1
            if tr["misses"] <= self.max_misses:
                keep.append(tr)
        self.tracks = keep


class FrameDetector:
    """Detection on EVERY frame (no optical-flow tracking): each box you get was found in that
    frame by a full search. Between frames it only (a) gives the same object the same ID and
    (b) optionally smooths the rectangle so it does not shake.
    hold: how many frames a box may be kept when its object is not detected (0 = never)."""

    def __init__(self, detector: FewShotDetector, smooth: bool = True, hold: int = 0, fps: float = 30.0,
                 min_cutoff: float = 0.5, beta: float = 0.3):
        self.det = detector
        self.smooth, self.hold = smooth, hold
        self.id_memory = 15           # frames an unseen object keeps its ID (no box is drawn meanwhile)
        self.dt = 1.0 / fps
        self.min_cutoff, self.beta = min_cutoff, beta
        self.objs: List[dict] = []
        self.next_id = 0

    def reset(self):
        self.objs.clear()

    def close(self):
        pass

    def update(self, frame: np.ndarray) -> List[Detection]:
        dets = self.det.detect(frame)
        used = set()
        for d in dets:
            best, bd = None, 1e9
            size = math.sqrt(max(d.rect[2] * d.rect[3], 1.0))
            for o in self.objs:
                if id(o) in used or o["det"].ref_id != d.ref_id:
                    continue
                dist = float(np.hypot(d.rect[0] - o["det"].rect[0], d.rect[1] - o["det"].rect[1]))
                if dist < (0.6 + 0.1 * o.get("miss", 0)) * size and dist < bd:
                    best, bd = o, dist
            if best is None:
                best = dict(id=self.next_id, filt=_RectFilter(self.min_cutoff, self.beta))
                self.next_id += 1
                self.objs.append(best)
            if best.get("miss", 0) > 0:            # back after a gap: do not smooth from the stale position
                best["filt"] = _RectFilter(self.min_cutoff, self.beta)
            best.update(det=d, raw=d.rect, miss=0, seen=True)
            used.add(id(best))
        out = []
        keep = []
        for o in self.objs:
            if id(o) not in used:
                o["miss"] += 1
                o["seen"] = False
                if o["miss"] > max(self.hold, self.id_memory):
                    continue                       # forgotten: a later detection here gets a new ID
                keep.append(o)
                if o["miss"] > self.hold:
                    continue                       # remembered for its ID only; no box is shown
                out.append(o.get("shown", o["det"]))
                continue
            d = o["det"]
            d.track_id = o["id"]
            if self.smooth and o["seen"]:
                d = dataclasses.replace(d, rect=o["filt"](o["raw"], self.dt))
                d.angle = d.rect[4]
            o["shown"] = d
            keep.append(o)
            out.append(d)
        self.objs = keep
        return out


# ═════════════════════════════════════════════════════════════════ drawing
_COLORS = [(0, 255, 0), (255, 128, 0), (0, 128, 255), (255, 0, 255), (0, 255, 255), (128, 255, 0)]


def draw_detections(img: np.ndarray, dets: Sequence[Detection], show_pts: bool = False,
                    box: str = "rotated", thickness: int = 2) -> np.ndarray:
    """box: "rotated" (rectangle that turns with the object, fixed size/aspect),
            "axis" (upright rectangle enclosing it), or "poly" (raw projected quadrilateral)."""
    out = img.copy()
    for i, d in enumerate(dets):
        col = _COLORS[(d.track_id if d.track_id >= 0 else i) % len(_COLORS)]
        if box == "axis":
            x, y, w, h = d.bbox
            cv2.rectangle(out, (x, y), (x + w, y + h), col, thickness, cv2.LINE_AA)
            tl = (x, y)
        else:
            pts = d.polygon if box == "poly" else d.box_points
            cv2.polylines(out, [np.round(pts).astype(np.int32)], True, col, thickness, cv2.LINE_AA)
            x, y = np.floor(pts.min(0)).astype(int)
            tl = (int(x), int(y))
        if show_pts and d.frame_pts is not None:
            for p in d.frame_pts[:200]:
                cv2.circle(out, (int(p[0]), int(p[1])), 2, col, -1)
        x, y = tl
        label = f"#{d.track_id if d.track_id >= 0 else i} {d.score:.2f} | {d.angle:+.0f} deg | x{d.scale:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        y = max(th + 6, y - 4)
        cv2.rectangle(out, (x, y - th - 6), (x + tw + 6, y), col, -1)
        cv2.putText(out, label, (x + 3, y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    return out