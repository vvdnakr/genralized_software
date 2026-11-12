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

__all__ = ["Config", "Detection", "FewShotDetector", "Tracker", "draw_detections", "poly_iou"]


# ═════════════════════════════════════════════════════════════════════ config
@dataclass
class Config:
    mode: str = "auto"                # "auto" | "features" | "shape"
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
    min_score: float = 0.30
    nms_iou: float = 0.30
    max_instances: int = 20
    deep_rounds: int = 2              # LightGlue backend: matching passes (each can add instances)

    # --- shape mode (textureless objects)
    auto_min_ref_features: int = 80   # fewer SIFT keypoints than this on a reference -> shape mode
    shape_angle_step: float = 12.0
    shape_angle_range: Tuple[float, float] = (0.0, 360.0)
    shape_scale_range: Tuple[float, float] = (0.35, 2.5)
    shape_scale_step: float = 1.20
    shape_proposal_threshold: float = 0.35   # coarse orientation agreement (0-1) to become a proposal
    shape_min_score: float = 0.30            # mean edge-orientation agreement (-1..1)
    shape_min_coverage: float = 0.40         # fraction of template edges found in frame
    shape_min_balanced: float = 0.10         # ... in EVERY edge-orientation group (rejects stripes, single lines)
    shape_mag_thresh: float = 40.0
    shape_min_flat: float = 0.80             # object interior must be edge-free in the frame where it is in the reference
    shape_max_side: int = 640                # shape search resolution (640 was both faster and more accurate than 960)
    shape_max_proposals: int = 16            # coarse candidates refined per frame
    shape_auto_mask: bool = True             # GrabCut silhouette of the reference (ignores crop background)


    @classmethod
    def preset(cls, speed: str = "balanced", **overrides) -> "Config":
        """'accurate' : full SIFT at up to 960 px          (best on small objects; ~250 ms at 960x720 on 2 cores)
           'balanced' : SIFT on a <=640 px copy             (~2x faster; same as accurate for 640x480 webcams)
           'fast'     : SIFT on a <=480 px copy             (~3x faster; misses more small objects)"""
        table = {"accurate": dict(features="sift"),
                 "balanced": dict(features="sift_fast", fast_side=640),
                 "fast": dict(features="sift_fast", fast_side=480)}
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
class _ShapeModel:
    """Textureless objects: coarse-to-fine search over rotation x scale on a
    polarity-invariant edge-orientation field (doubled-angle unit vectors), then
    similarity + affine (edge-ICP) refinement and verification by edge agreement,
    edge coverage and interior flatness."""

    COARSE_PX = 28   # template long side (px) at the coarse search level

    def __init__(self, ref: _Reference, cfg: Config, r: float = 1.0):
        """r: the frame is searched at r x detection resolution (speed presets)."""
        self.cfg = cfg
        self.ref = ref
        self.r = r
        g = ref.gray_raw
        h, w = g.shape
        mask = ref.mask
        if mask is None and cfg.shape_auto_mask:
            mask = self._grabcut(ref.color)
        self.mask = mask
        gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
        mag = cv2.magnitude(gx, gy)
        border = np.zeros((h, w), bool)
        border[4:-4, 4:-4] = True                       # never use the crop border itself
        region = border if mask is None else border & (cv2.dilate(mask, np.ones((7, 7), np.uint8)) > 0)
        strong = mag > cfg.shape_mag_thresh
        nms = mag >= cv2.dilate(mag, np.ones((3, 3), np.uint8))
        ys, xs = np.nonzero(strong & nms & region)
        if len(xs) < 20:
            ys, xs = np.nonzero(strong & region)
        if len(xs) > 800:
            sel = np.random.default_rng(0).choice(len(xs), 800, replace=False)
            ys, xs = ys[sel], xs[sel]
        theta = np.arctan2(gy[ys, xs], gx[ys, xs])
        self.edge_xy = np.c_[xs - ref.center[0], ys - ref.center[1]].astype(np.float32)
        self.edge_n = np.c_[np.cos(theta), np.sin(theta)].astype(np.float32)   # edge normals
        self.edge_2t = (2 * theta).astype(np.float32)
        self.n_edges = len(xs)
        # edge-orientation groups: a true match must explain edges of every major
        # orientation, not just two parallel lines (bricks, stripes, door frames)
        ob = (((theta % np.pi) + np.pi / 8) // (np.pi / 4)).astype(int) % 4
        self.groups = [g for g in [ob == i for i in range(4)]
                       if g.sum() >= max(8, 0.08 * len(xs))]
        # flat points: object interior with no edges in the reference (needs a silhouette)
        self.flat_xy = np.zeros((0, 2), np.float32)
        if mask is not None:
            edge_any = (mag > 0.5 * cfg.shape_mag_thresh).astype(np.uint8)
            far = cv2.distanceTransform(1 - edge_any, cv2.DIST_L2, 3) > 4
            inner = cv2.erode(mask, np.ones((9, 9), np.uint8)) > 0
            fy, fx = np.nonzero(far & inner & border)
            if len(fx) > 400:
                sel = np.random.default_rng(1).choice(len(fx), 400, replace=False)
                fy, fx = fy[sel], fx[sel]
            if len(fx) >= 30:
                self.flat_xy = np.c_[fx - ref.center[0], fy - ref.center[1]].astype(np.float32)

        a0, a1 = cfg.shape_angle_range
        self.angles = np.arange(a0, a1 - 1e-6, cfg.shape_angle_step)
        smin, smax = cfg.shape_scale_range
        scales = [smin]
        while scales[-1] * cfg.shape_scale_step <= smax + 1e-6:
            scales.append(scales[-1] * cfg.shape_scale_step)
        self.scales = [s * r / ref.k for s in scales]   # range is relative to the ORIGINAL reference size
        self.bank = []                              # [(ds, [(angle, scale, T, wsum, r), ...]), ...]
        self.n_templates = 0
        if self.n_edges < 12:
            return
        for s in self.scales:
            ds = max(1.0, s * ref.dim / self.COARSE_PX)
            ents = []
            for a in self.angles:
                T, wsum, r = self._raster(float(a), s, ds)
                if wsum >= 8:
                    ents.append((float(a), s, T, wsum, r))
            self.bank.append((ds, ents))
            self.n_templates += len(ents)

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
        if n > 1:   # keep the largest blob, fill its holes' noise a little
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
    def frame_field(gray: np.ndarray, mag_thresh: float, blur: float = 1.2):
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        mag = cv2.magnitude(gx, gy) + 1e-6
        m = (mag > mag_thresh).astype(np.float32)
        c, s = gx / mag, gy / mag
        c2 = (c * c - s * s) * m           # cos(2θ): contrast-polarity invariant
        s2 = (2 * c * s) * m               # sin(2θ)
        c2 = cv2.GaussianBlur(c2, (0, 0), blur)   # spread: tolerance to small misalignment
        s2 = cv2.GaussianBlur(s2, (0, 0), blur)
        return c2, s2, m

    def propose(self, g_raw: np.ndarray, thr: float, per_template: int = 4):
        """Coarse search. All rotations of one scale are correlated in a single batched FFT:
        the frame field is a complex image c2 + i*s2, so Re(conj(T) * F) summed over the
        template is exactly the orientation agreement."""
        h, w = g_raw.shape
        props = []
        if not hasattr(self, "_spec"):
            self._spec = {}
        for gi, (ds, ents) in enumerate(self.bank):
            if not ents:
                continue
            sw, sh = max(8, round(w / ds)), max(8, round(h / ds))
            small = cv2.resize(g_raw, (sw, sh), interpolation=cv2.INTER_AREA) if ds > 1 else g_raw
            c2, s2, _ = self.frame_field(small, self.cfg.shape_mag_thresh, 1.0)
            fx, fy = w / sw, h / sh
            fits = [i for i, e in enumerate(ents) if e[2].shape[0] < sh and e[2].shape[1] < sw]
            if not fits:
                continue
            ph, pw = _fft.next_fast_len(sh), _fft.next_fast_len(sw)   # zero padding: fast FFT sizes
            key = (gi, ph, pw)
            if key not in self._spec:      # template spectra for this frame size (computed once)
                Tp = np.zeros((len(fits), ph, pw), np.complex64)
                for j, i in enumerate(fits):
                    T = ents[i][2]
                    Tp[j, :T.shape[0], :T.shape[1]] = T[..., 0] + 1j * T[..., 1]
                self._spec[key] = np.conj(_fft.fft2(Tp, workers=-1))
            Fp = np.zeros((ph, pw), np.complex64)
            Fp[:sh, :sw] = c2 + 1j * s2
            Fs = _fft.fft2(Fp, workers=-1)
            R = _fft.ifft2(self._spec[key] * Fs[None], workers=-1).real
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

    def score(self, field, A):
        """(mean orientation agreement in [-1,1], edge coverage, interior flatness) of pose A."""
        c2, s2, m = field
        p, n = self._edge_frame(A)
        px, py = p[:, 0][None], p[:, 1][None]
        fc = cv2.remap(c2, px, py, cv2.INTER_LINEAR, borderValue=0)[0]
        fs = cv2.remap(s2, px, py, cv2.INTER_LINEAR, borderValue=0)[0]
        c2t, s2t = n[:, 0] ** 2 - n[:, 1] ** 2, 2 * n[:, 0] * n[:, 1]
        agree = fc * c2t + fs * s2t
        flat = 1.0
        if len(self.flat_xy):
            q = self.flat_xy @ A[:, :2].T.astype(np.float32) + A[:, 2].astype(np.float32)
            e = cv2.remap(m, q[:, 0][None], q[:, 1][None], cv2.INTER_NEAREST, borderValue=0)[0]
            flat = float((e < 0.5).mean())
        hit = agree > 0.5
        bal = min([float(hit[g].mean()) for g in self.groups], default=float(hit.mean()))
        return float(agree.mean()), float(hit.mean()), flat, bal

    def refine_similarity(self, F2, cx, cy, ang, s, a_step, s_step, levels: int = 4):
        """Pattern search over (x, y, angle, scale); all 8 neighbour poses are scored in one
        vectorised remap on a 200-point subset of the edge model."""
        if not hasattr(self, "_sub"):
            k = min(200, self.n_edges)
            self._sub = np.random.default_rng(2).choice(self.n_edges, k, replace=False)
            self._E = self.edge_xy[self._sub]
            self._c2t, self._s2t = np.cos(self.edge_2t[self._sub]), np.sin(self.edge_2t[self._sub])
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
        best = evaluate(cur[None])[0]
        da, ds, dp = a_step / 2, (s_step - 1) / 2, max(1.0, 0.03 * s * self.ref.dim)
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
        return self.sim_pose(*cur)

    def refine_affine(self, field, A, max_aniso):
        """Edge-ICP: move each model edge point along its normal to the best-agreeing frame
        edge, then solve a weighted least-squares affine. Absorbs mild perspective/shear."""
        c2, s2, _ = field
        P1 = np.c_[self.edge_xy, np.ones(len(self.edge_xy))]
        for r in (6, 4, 3, 2, 2):
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

    # ------------------------------------------------------------- references
    def add_reference(self, img: np.ndarray, mask: Optional[np.ndarray] = None) -> int:
        """Add a reference view (BGR or gray). Optional mask (non-zero = object)."""
        ref = _Reference(img, mask, self.cfg, self.ext_ref, self.clahe)
        mode = self.cfg.mode
        if mode == "auto":
            mode = "features" if (ref.desc is not None and ref.n_texture_kp >= self.cfg.auto_min_ref_features) \
                else "shape"
        ref.mode = mode
        if mode == "shape":
            ref.shape = _ShapeModel(ref, self.cfg)
            ref.shape_cache = {1.0: ref.shape}
        if self.deep is not None:
            self.deep.add_reference(ref.gray)
        self.refs.append(ref)
        return len(self.refs) - 1

    def clear(self):
        self.refs.clear()
        if self.deep is not None:
            self.deep.clear()

    def describe(self) -> str:
        out = []
        for i, r in enumerate(self.refs):
            if r.mode == "shape":
                out.append(f"ref{i}: {r.w}x{r.h} shape mode, {r.shape.n_templates} templates, {r.shape.n_edges} edge pts")
            else:
                out.append(f"ref{i}: {r.w}x{r.h} features mode, {r.n_base_kp} keypoints ({len(r.desc)} incl. views)")
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
        g = self.clahe.apply(g_raw) if self.clahe is not None else g_raw
        dets: List[Detection] = []
        timing = {}

        feat_refs = [i for i, r in enumerate(self.refs) if r.mode == "features"]
        shape_refs = [i for i, r in enumerate(self.refs) if r.mode == "shape"]

        if feat_refs and self.deep is not None:
            t = time.perf_counter()
            found: List[Detection] = []
            for _ in range(self.cfg.deep_rounds):   # LightGlue is one-to-one: re-match to find more instances
                new = []
                for ri, (rp, sp) in self.deep.match(g, exclude=[d.polygon for d in found]).items():
                    if self.refs[ri].mode == "features":
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
            for ri in shape_refs:
                dets += self._detect_shape(self.refs[ri], ri, g_raw)
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
        inl_conf = 1.0 - math.exp(-n_in / 15.0)
        score = 0.6 * inl_conf + 0.4 * max(ncc, 0.0) if c.verify_appearance else inl_conf
        if score < c.min_score:
            return None
        poly = _project(H, ref.corners).astype(np.float32)
        ang, scl = _pose_from_H(H, *ref.center)
        Hn = H @ np.diag([ref.k, ref.k, 1.0])   # map ORIGINAL reference px -> frame
        return Detection(polygon=poly, H=Hn, ref_id=ri, score=float(score), inliers=int(n_in),
                         angle=ang, scale=scl * ref.k, method=method,
                         frame_pts=dst.astype(np.float32), ref_pts=(src / ref.k).astype(np.float32))

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
    def _detect_shape(self, ref, ri, g_full):
        c = self.cfg
        if ref.shape is None or ref.shape.n_templates == 0:
            return []
        rs = round(min(1.0, c.shape_max_side / max(g_full.shape)), 3)
        sm: _ShapeModel = ref.shape_cache.get(rs)
        if sm is None:   # template bank for this working resolution (built once per frame size)
            sm = ref.shape_cache[rs] = _ShapeModel(ref, c, rs)
        g_raw = g_full if rs == 1.0 else cv2.resize(g_full, (round(g_full.shape[1] * rs), round(g_full.shape[0] * rs)),
                                                      interpolation=cv2.INTER_AREA)
        props = sm.propose(g_raw, c.shape_proposal_threshold)
        if not props:
            return []
        props.sort(key=lambda p: -p[0])
        kept = []
        for p in props:   # centre-distance NMS on coarse proposals
            r = 0.3 * p[4] * ref.dim
            if all((p[1] - k[1]) ** 2 + (p[2] - k[2]) ** 2 > r * r for k in kept):
                kept.append(p)
            if len(kept) >= c.shape_max_proposals:
                break
        field = _ShapeModel.frame_field(g_raw, c.shape_mag_thresh)
        F2 = cv2.merge([field[0], field[1]])
        dets = []
        T_c = np.array([[1, 0, -ref.center[0]], [0, 1, -ref.center[1]], [0, 0, 1]])  # ref px -> centred
        for _, cx, cy, a, s in kept:
            A = sm.refine_similarity(F2, cx, cy, a, s, c.shape_angle_step, c.shape_scale_step)
            A = sm.refine_affine(field, A, c.max_anisotropy)
            sc, cov, flat, bal = sm.score(field, A)
            if (sc < c.shape_min_score or cov < c.shape_min_coverage or flat < c.shape_min_flat
                    or bal < c.shape_min_balanced):
                continue
            H = np.vstack([A, [0, 0, 1]]) @ T_c
            if not self._geometry_ok(ref, H, g_raw.shape):
                continue
            H = np.diag([1.0 / rs, 1.0 / rs, 1.0]) @ H   # working resolution -> detection resolution
            poly = _project(H, ref.corners).astype(np.float32)
            ang, scl = _pose_from_H(H, *ref.center)
            Hn = H @ np.diag([ref.k, ref.k, 1.0])
            conf = float(np.clip(0.5 * sc + 0.3 * cov + 0.2 * flat, 0, 1))
            dets.append(Detection(polygon=poly, H=Hn, ref_id=ri, score=conf, inliers=int(cov * sm.n_edges),
                                  angle=ang, scale=scl * ref.k, method="shape"))
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
            if self._lk(tr, g_prev, g, f):
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
            if all(self._lk(tr, a, b, f) for a, b in zip(hist[:-1], hist[1:])):
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
