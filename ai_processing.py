"""
ai_processing.py - learned image-processing pattern matcher (run.py --ai_imageprocessing).

For a fixed inspection scene (same background, slightly different light, shiny parts): give one or
more MASTER images of the scene and one or more small TEMPLATE crops of the object. Training then

  1. turns every image into a bank of processed channels (grey, log-grey, colour a/b, saturation,
     locally normalised grey, edge strength, x/y gradients, band-pass),
  2. matches every template in every channel (zero-mean normalised correlation: gain/offset free),
  3. finds all instances of the object in the masters (the templates' own places and look-alike places),
  4. makes many "other parts under other light" from the masters (brightness, gamma, colour cast, noise,
     blur; local tone, slight shape change, small rotation / size change), uses every instance found
     as one more template, and learns
       - how much to trust each channel for THIS object (weights), and
       - a patch classifier on the instances against the hardest background places,
  5. sets the decision threshold in the middle of the gap between object and background.

Detection = the same channels, weighted correlation maps, peaks, patch classifier. OpenCV + NumPy only.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

CHANNELS = ("gray", "loggray", "lab_a", "lab_b", "sat", "lnorm", "gmag", "gx", "gy", "dog")


def channels(img: np.ndarray, sigma: float) -> Dict[str, np.ndarray]:
    """The image-processing bank. sigma: neighbourhood of the local normalisation (about half the object)."""
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2Lab).astype(np.float32)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    gb = cv2.GaussianBlur(g, (0, 0), 1.0)
    gx = cv2.Sobel(gb, cv2.CV_32F, 1, 0)
    gy = cv2.Sobel(gb, cv2.CV_32F, 0, 1)
    mu = cv2.GaussianBlur(g, (0, 0), sigma)
    sd = np.sqrt(np.maximum(cv2.GaussianBlur(g * g, (0, 0), sigma) - mu * mu, 0)) + 4.0
    return dict(gray=g, loggray=np.log1p(g), lab_a=lab[..., 1], lab_b=lab[..., 2], sat=hsv[..., 1],
                lnorm=(g - mu) / sd, gmag=np.log1p(cv2.magnitude(gx, gy)), gx=gx, gy=gy,
                dog=cv2.GaussianBlur(g, (0, 0), 1.5) - cv2.GaussianBlur(g, (0, 0), 4.0))


def relight(img: np.ndarray, rng: np.random.Generator, strength: float = 1.0) -> np.ndarray:
    """A random plausible change of lighting / camera: brightness, gamma, colour cast, offset, blur, noise."""
    s = strength
    x = img.astype(np.float32) / 255.0
    x = np.power(np.clip(x, 0, 1), rng.uniform(1 - 0.2 * s, 1 + 0.25 * s))
    x = x * rng.uniform(1 - 0.25 * s, 1 + 0.3 * s) * (1 + rng.uniform(-0.06, 0.06, 3) * s).astype(np.float32)
    x = x * 255.0 + rng.uniform(-8, 8) * s
    if rng.random() < 0.5:
        x = cv2.GaussianBlur(x, (0, 0), rng.uniform(0.3, 0.8))
    x = x + rng.normal(0, rng.uniform(0, 3.0) * s, x.shape).astype(np.float32)
    return np.clip(x, 0, 255).astype(np.uint8)


def vary(img: np.ndarray, rng: np.random.Generator, box: Tuple[int, int], strength: float = 1.0,
         pose: bool = True) -> Tuple[np.ndarray, np.ndarray, float]:
    """A random "other part": smooth local tone / colour differences, a slight change of shape and
    (pose=True) a small rotation / size change. Returns (image, 2x3 map master -> image, shape slack px)."""
    h, w = img.shape[:2]
    R = float(max(box))
    grid = (max(3, int(round(w / (0.6 * R)))), max(3, int(round(h / (0.6 * R)))))

    def smooth(n):
        f = rng.normal(0, 1, (grid[1], grid[0], n)).astype(np.float32)
        return cv2.resize(f, (w, h), interpolation=cv2.INTER_CUBIC).reshape(h, w, n)

    x = img.astype(np.float32)
    x = x * np.clip(1 + 0.10 * strength * smooth(1) + 0.03 * strength * smooth(3), 0.5, 1.6) + 10.0 * strength * smooth(1)
    amp = 0.03 * R * strength
    d = smooth(2) * (amp / 2.0)
    if pose:
        M = cv2.getRotationMatrix2D((w / 2, h / 2), rng.uniform(-5, 5) * strength, 1 + rng.uniform(-0.05, 0.05) * strength)
    else:
        M = np.float32([[1, 0, 0], [0, 1, 0]]).astype(np.float64)
    Mi = cv2.invertAffineTransform(M)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    mx = Mi[0, 0] * xx + Mi[0, 1] * yy + Mi[0, 2] + d[..., 0]
    my = Mi[1, 0] * xx + Mi[1, 1] * yy + Mi[1, 2] + d[..., 1]
    out = cv2.remap(np.clip(x, 0, 255).astype(np.uint8), mx.astype(np.float32), my.astype(np.float32),
                    cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    return out, M, amp


def distract(img: np.ndarray, keep: Sequence[Tuple[float, float]], box: Tuple[int, int],
             rng: np.random.Generator) -> np.ndarray:
    """Paint random foreign shapes (ellipses, rings, polygons, strokes) on the background, never on an
    object: training then knows that a new mark of about the object's size is not the object."""
    h, w = img.shape[:2]
    bw, bh = box
    R = float(max(box))
    out = img.copy()
    n = int(np.clip(h * w / (R * R * 5.0), 4, 60))
    for _ in range(n):
        cx, cy = int(rng.uniform(0, w)), int(rng.uniform(0, h))
        s = rng.uniform(0.15, 0.65) * R
        if any(abs(cx - x) < 0.6 * bw + s and abs(cy - y) < 0.6 * bh + s for x, y in keep):
            continue
        col = tuple(float(c) for c in rng.uniform(0, 255, 3))
        lay = out.copy()
        kind = rng.integers(0, 4)
        if kind == 0:
            cv2.ellipse(lay, (cx, cy), (int(s), int(s * rng.uniform(0.5, 1.0))), rng.uniform(0, 180), 0, 360, col, -1, cv2.LINE_AA)
        elif kind == 1:                                                       # outline ring
            cv2.ellipse(lay, (cx, cy), (int(s), int(s * rng.uniform(0.7, 1.0))), rng.uniform(0, 180), 0, 360, col,
                        max(1, int(s * rng.uniform(0.08, 0.3))), cv2.LINE_AA)
        elif kind == 2:
            k = rng.integers(3, 7)
            ang = np.sort(rng.uniform(0, 2 * np.pi, k))
            pts = np.stack([cx + s * rng.uniform(0.5, 1, k) * np.cos(ang), cy + s * rng.uniform(0.5, 1, k) * np.sin(ang)], 1)
            cv2.fillPoly(lay, [pts.astype(np.int32)], col, cv2.LINE_AA)
        else:
            a = rng.uniform(0, np.pi)
            cv2.line(lay, (int(cx - s * np.cos(a)), int(cy - s * np.sin(a))), (int(cx + s * np.cos(a)), int(cy + s * np.sin(a))),
                     col, max(1, int(s * rng.uniform(0.05, 0.25))), cv2.LINE_AA)
        al = rng.uniform(0.45, 1.0)
        out = cv2.addWeighted(lay, al, out, 1 - al, 0)
    return out


def _zncc_map(C: np.ndarray, T: np.ndarray) -> np.ndarray:
    """Correlation map with the value at each pixel = template CENTRED there (-1 where it does not fit)."""
    th, tw = T.shape[:2]
    out = np.full(C.shape[:2], -1.0, np.float32)
    if th > C.shape[0] or tw > C.shape[1]:
        return out
    r = cv2.matchTemplate(C, T, cv2.TM_CCOEFF_NORMED)
    r = np.clip(np.nan_to_num(r, nan=0.0, posinf=0.0, neginf=0.0), -1, 1)
    out[th // 2: th // 2 + r.shape[0], tw // 2: tw // 2 + r.shape[1]] = r
    return out


def _peaks(F: np.ndarray, radius: int, top: int, floor: float = -0.5) -> List[Tuple[float, int, int]]:
    k = 2 * max(1, radius) + 1
    mx = cv2.dilate(F, np.ones((k, k), np.uint8))
    ys, xs = np.nonzero((F >= mx) & (F > floor))
    order = np.argsort(-F[ys, xs])[:top]
    return [(float(F[ys[i], xs[i]]), int(xs[i]), int(ys[i])) for i in order]


def _descriptor(img: np.ndarray, cx: int, cy: int, w: int, h: int) -> np.ndarray:
    """Patch descriptor for the classifier: contrast-normalised brightness layout (12x12), colour layout
    (6x6 a/b, mean-free so a colour cast drops out) and edge-direction histograms (3x3 cells x 8 bins)."""
    x0, y0 = int(round(cx - w / 2)), int(round(cy - h / 2))
    pad = cv2.copyMakeBorder(img, h, h, w, w, cv2.BORDER_REFLECT)
    p = pad[y0 + h: y0 + 2 * h, x0 + w: x0 + 2 * w]
    lab = cv2.cvtColor(cv2.resize(p, (24, 24), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2Lab).astype(np.float32)
    L = lab[..., 0]
    Ln = (L - L.mean()) / (L.std() + 6.0)
    f1 = cv2.resize(Ln, (12, 12), interpolation=cv2.INTER_AREA).ravel()
    ab = cv2.resize(lab[..., 1:], (6, 6), interpolation=cv2.INTER_AREA)
    f2 = ((ab - ab.reshape(-1, 2).mean(0)) / 12.0).ravel()
    f2b = (lab[..., 1:].reshape(-1, 2).std(0) / 12.0)
    gx, gy = cv2.Sobel(Ln, cv2.CV_32F, 1, 0), cv2.Sobel(Ln, cv2.CV_32F, 0, 1)
    mag, ang = cv2.cartToPolar(gx, gy)
    b = np.minimum((ang / (2 * math.pi) * 8).astype(int), 7)
    hog = np.zeros((3, 3, 8), np.float32)
    for i in range(3):
        for j in range(3):
            hog[i, j] = np.bincount(b[8 * i: 8 * i + 8, 8 * j: 8 * j + 8].ravel(),
                                    mag[8 * i: 8 * i + 8, 8 * j: 8 * j + 8].ravel(), 8)
    hog = hog.ravel()
    hog /= np.linalg.norm(hog) + 1e-3
    f1 = f1 / (np.linalg.norm(f1) + 1e-3)
    return np.concatenate([f1, f2 / (np.linalg.norm(f2) + 0.5), f2b, hog]).astype(np.float32)


def _logistic(X: np.ndarray, y: np.ndarray, l2: float = 1e-2, iters: int = 400) -> Tuple[np.ndarray, float]:
    """Class-balanced L2 logistic regression (plain gradient descent; small problems)."""
    w, b = np.zeros(X.shape[1], np.float64), 0.0
    sw = np.where(y > 0.5, 0.5 / max(y.sum(), 1), 0.5 / max((1 - y).sum(), 1))
    lr = 1.0
    for _ in range(iters):
        p = 1 / (1 + np.exp(-np.clip(X @ w + b, -30, 30)))
        g = (p - y) * sw
        w -= lr * (X.T @ g + l2 * w)
        b -= lr * g.sum()
    return w, float(b)


@dataclass
class Match:
    x: int            # box (upright), frame pixels
    y: int
    w: int
    h: int
    score: float      # 0..1, >= 0.5 = object
    match: float      # weighted channel correlation
    patch: float      # patch classifier probability
    template: int
    center: Tuple[float, float] = (0.0, 0.0)
    evidence: dict = field(default_factory=dict)


class AIPatternMatcher:
    def __init__(self, min_score: float = 0.4, augment: int = 16, seed: int = 0, expect: Optional[int] = None,
                 verbose: bool = True, tolerance: float = 0.85, variation: float = 1.0, max_templates: int = 4,
                 harvest: bool = True):
        """min_score: the threshold on the 0..1 score (0 = strongest background place in training, 1 = typical
        object). tolerance: no longer used (kept so old calls work). variation: how different the simulated "other parts" are in training.
        max_templates: user templates + instances found in the masters that are used as extra templates."""
        self.min_score, self.n_aug, self.verbose = min_score, augment, verbose
        self.expect = expect                     # known number of objects per image (optional)
        self.tolerance, self.variation = float(np.clip(tolerance, 0.05, 0.95)), variation
        self.max_templates, self.harvest = max_templates, harvest
        self.rng = np.random.default_rng(seed)
        self.trained = False

    # ------------------------------------------------------------------ helpers
    def _maps(self, img: np.ndarray, use: Sequence[str]) -> np.ndarray:
        """S[t, c] maps (template centred) for the channels in `use`, stored as int8 (x127)."""
        C = channels(img, self.sigma)
        S = np.empty((len(self.templates), len(use)) + img.shape[:2], np.int8)
        for ti, TC in enumerate(self.tchan):
            for ci, c in enumerate(use):
                S[ti, ci] = np.round(_zncc_map(C[c], TC[c]) * 127.0)
        return S

    def _fuse(self, S: np.ndarray, w: np.ndarray, skip: Optional[Dict[int, List[Tuple[float, float]]]] = None):
        """Weighted map per template, then the best template at each pixel."""
        F = np.tensordot(S.astype(np.float32), (w / 127.0).astype(np.float32), axes=([1], [0]))   # (T, H, W)
        if skip:                                                              # a template never votes for its own crop
            r = self.radius + 2
            for ti, pts in skip.items():
                for x, y in pts:
                    x, y = int(round(x)), int(round(y))
                    F[ti, max(y - r, 0): y + r + 1, max(x - r, 0): x + r + 1] = -1
        return F.max(0), F.argmax(0)

    @staticmethod
    def _map_pts(M, pts):
        return [(M[0, 0] * x + M[0, 1] * y + M[0, 2], M[1, 0] * x + M[1, 1] * y + M[1, 2]) for x, y in pts]

    def _best_near(self, F, x, y, r):
        """Highest value of F within r pixels of (x, y): the object's place in a varied copy."""
        h, w = F.shape
        x, y = int(round(x)), int(round(y))
        if not (0 <= x < w and 0 <= y < h):
            return None
        y0, x0 = max(y - r, 0), max(x - r, 0)
        sub = F[y0: y + r + 1, x0: x + r + 1]
        dy, dx = np.unravel_index(int(sub.argmax()), sub.shape)
        return float(sub[dy, dx]), x0 + dx, y0 + dy

    def _set_templates(self, templates):
        self.templates = templates
        self.tchan = [channels(t, self.sigma) for t in templates]

    # ------------------------------------------------------------------ training
    def train(self, masters: Sequence[np.ndarray], templates: Sequence[np.ndarray]) -> dict:
        masters = [m if m.ndim == 3 else cv2.cvtColor(m, cv2.COLOR_GRAY2BGR) for m in masters]
        user = [t if t.ndim == 3 else cv2.cvtColor(t, cv2.COLOR_GRAY2BGR) for t in templates]
        hs, ws = [t.shape[0] for t in user], [t.shape[1] for t in user]
        self.radius = max(2, int(0.45 * min(min(hs), min(ws))))
        self.sigma = 0.5 * max(max(hs), max(ws))
        self.box = (int(np.median(ws)), int(np.median(hs)))
        # objects do not overlap: a peak within one box size of a stronger object is part of that object
        self.r_ex = max(2 * self.radius, int(0.9 * max(self.box)))
        self._set_templates(user)
        nc = len(CHANNELS)
        near = lambda x, y, pts, r: any((x - a) ** 2 + (y - b) ** 2 <= r * r for a, b in pts)

        # a template cut from a master: that place is a certain instance, and the template must not
        # be allowed to "find" its own pixels there (it would look perfect and teach nothing)
        own: List[Dict[int, List[Tuple[int, int]]]] = [dict() for _ in masters]
        known: List[List[Tuple[int, int]]] = [[] for _ in masters]
        for ti, t in enumerate(user):
            for mi, m in enumerate(masters):
                if t.shape[0] <= m.shape[0] and t.shape[1] <= m.shape[1]:
                    r = cv2.matchTemplate(m, t, cv2.TM_CCOEFF_NORMED)
                    _, v, _, loc = cv2.minMaxLoc(r)
                    if v > 0.93:
                        p = (loc[0] + t.shape[1] // 2, loc[1] + t.shape[0] // 2)
                        own[mi].setdefault(ti, []).append(p)
                        known[mi].append(p)

        # training copies of every master: as given; re-lit; re-lit "other parts" (tone, shape, pose)
        eye = np.float64([[1, 0, 0], [0, 1, 0]])
        views = []                                                            # (master, image, map, slack px)
        for mi, m in enumerate(masters):
            views.append((mi, m, eye, 0.0))
            for k in range(0, self.n_aug, 4):
                views.append((mi, relight(m, self.rng), eye, 0.0))
        aligned = list(views)

        self.dropped = []
        for attempt in range(2):          # second pass only when a bad template was left out
            # ---- 1. find the instances with the user's templates (aligned copies): alternate
            #         (label places with the current weights, learn the weights)
            w = np.full(nc, 1.0 / nc)
            S_al = [(v[0], self._maps(v[1], CHANNELS)) for v in aligned]
            inst: List[List[Tuple[int, int]]] = [list(k) for k in known]
            for rnd in range(4):
                pos_x, neg_x = [], []
                new_inst = [list(k) for k in known]
                for mi in range(len(masters)):
                    mv = [S for m_, S in S_al if m_ == mi]
                    fused = [self._fuse(S, w, own[mi]) for S in mv]
                    # mean fused map over the re-lit copies: a place must be good under every light
                    Fm = np.mean([f[0] for f in fused], axis=0)
                    pk = _peaks(Fm, self.radius, 400)
                    vals = np.array([p[0] for p in pk])
                    bg = float(np.median(vals[: min(60, len(vals))]))            # level of the strong background peaks
                    kn = [float(Fm[y, x]) for x, y in known[mi]]
                    # reference = a certain instance scored by the OTHER templates (or, without one, the best place)
                    ref = float(np.median(kn)) if kn and len(self.templates) > 1 else float(vals[0])
                    # ONE template that is a crop of this master: its own place is certain but blocked, so the
                    # question is only whether there are MORE objects. Without a clear outlier there are none.
                    solo_known = len(self.templates) == 1 and bool(known[mi])
                    top = vals[: min(60, len(vals))]                             # the same peaks that define bg
                    mad = 1.4826 * float(np.median(np.abs(top - bg))) + 1e-3
                    if solo_known and self.expect:
                        need = self.expect - len(known[mi])
                        cut = vals[need - 1] - 1e-6 if need > 0 else float("inf")
                    elif solo_known and ((vals[0] - bg) < max(6.0 * mad, 0.12) or vals[0] < 0.3):
                        cut = float("inf")
                    elif self.expect:
                        cut = vals[min(self.expect, len(vals)) - 1] - 1e-6
                    else:
                        cut = bg + 0.55 * (ref - bg)
                        cand = [i for i in range(len(vals) - 1) if vals[i] >= bg + 0.3 * (ref - bg)]
                        if cand:                                                  # the instances end where the scores drop most
                            gmax, gi = max((vals[i] - vals[i + 1], i) for i in cand)
                            if gmax > 0.25 * (ref - bg):
                                cut = 0.5 * (vals[gi] + vals[gi + 1])
                    for v, x, y in pk:
                        if v >= cut and not near(x, y, new_inst[mi], self.r_ex):
                            new_inst[mi].append((x, y))
                    neg_pts = [(x, y) for v, x, y in pk if not near(x, y, new_inst[mi], self.r_ex)][:150]
                    for (F, A), S in zip(fused, mv):
                        for x, y in new_inst[mi]:
                            if solo_known and (x, y) in known[mi]:
                                continue                                      # blocked there: nothing honest to learn
                            _, xx, yy = self._best_near(F, x, y, 2)
                            pos_x.append(S[A[yy, xx], :, yy, xx] / 127.0)
                        for x, y in neg_pts:
                            neg_x.append(S[A[y, x], :, y, x] / 127.0)
                inst = new_inst
                w_new = self._weights(pos_x, neg_x, nc) if pos_x else w
                done = np.abs(w_new - w).max() < 0.02 and rnd > 0
                w = w_new
                if done:
                    break
            # ---- a template that does not look like the objects found (drawn off the object, wrong part)
            #      would pull everything down: test each one alone on the instances, leave bad ones out
            if len(user) > 1 and attempt == 0:
                good = []
                for ti in range(len(user)):
                    sc, bgs = [], []
                    for m_, S in S_al:
                        Ft = np.tensordot(S[ti].astype(np.float32), (w / 127.0).astype(np.float32), axes=([0], [0]))
                        mine = own[m_].get(ti, [])
                        for x, y in inst[m_]:
                            if not near(x, y, mine, self.radius):
                                sc.append(float(Ft[max(y - 2, 0): y + 3, max(x - 2, 0): x + 3].max()))
                        bgs.append(float(np.percentile(Ft, 99.5)))
                    good.append((float(np.median(sc)) if sc else 1.0) > float(np.median(bgs)) + 0.02)
                if any(good) and not all(good):
                    self.dropped = [i for i, g in enumerate(good) if not g]
                    keep_i = [i for i, g in enumerate(good) if g]
                    remap = {old: new for new, old in enumerate(keep_i)}
                    user = [user[i] for i in keep_i]
                    own = [{remap[ti]: pts for ti, pts in o.items() if ti in remap} for o in own]
                    known = [[p for pts in o.values() for p in pts] for o in own]
                    self._set_templates(user)
                    if self.verbose:
                        print(f"   WARNING: template(s) {self.dropped} do not look like the other objects (drawn off the "
                              f"object?) - left out")
            if not self.dropped:
                break
        del S_al
        self.instances = inst

        # ---- 2. every instance found is a real, different look of the object: use the most different
        #         ones as extra templates (each skips its own place while learning)
        n_user = len(user)
        if self.harvest and self.max_templates > n_user:
            bw, bh = self.box
            pool = []
            for mi, m in enumerate(masters):
                taken = [p for pts in own[mi].values() for p in pts]
                for x, y in inst[mi]:
                    x0, y0 = int(round(x - bw / 2)), int(round(y - bh / 2))
                    if near(x, y, taken, self.radius) or x0 < 0 or y0 < 0 or x0 + bw > m.shape[1] or y0 + bh > m.shape[0]:
                        continue
                    crop = m[y0: y0 + bh, x0: x0 + bw].copy()
                    pool.append((mi, (x, y), crop, _descriptor(crop, bw // 2, bh // 2, bw, bh)))
            chosen = [_descriptor(t, t.shape[1] // 2, t.shape[0] // 2, t.shape[1], t.shape[0]) for t in user]
            temps = list(user)
            while pool and len(temps) < self.max_templates:
                d = [min(float(np.linalg.norm(p[3] - c)) for c in chosen) for p in pool]
                mi, pt, crop, desc = pool.pop(int(np.argmax(d)))
                own[mi].setdefault(len(temps), []).append(pt)
                temps.append(crop)
                chosen.append(desc)
            self._set_templates(temps)

        # "other parts" copies, made now that the objects' places are known: re-lit, foreign shapes on
        # the background (never on an object), local tone / shape differences, small pose changes
        for mi, m in enumerate(masters):
            for k in range(self.n_aug):
                if k % 4 == 0:
                    continue
                im = relight(m, self.rng)
                if k % 2 == 1:
                    im = distract(im, inst[mi], self.box, self.rng)
                im, M, amp = vary(im, self.rng, self.box, self.variation * self.rng.uniform(0.5, 2.0), pose=(k % 4 != 1))
                views.append((mi, im, M, amp))

        # ---- 3. learn on all copies (instances fixed): channel weights, threshold, patch classifier
        S_all = [self._maps(v[1], CHANNELS) for v in views]
        slack = lambda amp: int(math.ceil(2 + amp))

        def collect(wts, idx=None, with_desc=False):
            zp, zn, px, nx, dp, dn = [], [], [], [], [], []
            jp, jn = [], []                                                   # (match value, view index) per descriptor
            for vi, ((mi, im, M, amp), S) in enumerate(zip(views, S_all)):
                Sv = S if idx is None else S[:, idx]
                solo = len(self.templates) == 1
                skip = None if solo else {ti: self._map_pts(M, pts) for ti, pts in own[mi].items()}
                F, A = self._fuse(Sv, wts, skip)
                pts = self._map_pts(M, inst[mi])
                found = []
                for (x, y), src in zip(pts, inst[mi]):
                    if solo and im is masters[mi] and src in known[mi]:
                        continue                                              # the template on its own pixels: says nothing
                    b = self._best_near(F, x, y, slack(amp))
                    if b is None:
                        continue
                    v, xx, yy = b
                    found.append((xx, yy))
                    zp.append(v)
                    px.append(Sv[A[yy, xx], :, yy, xx] / 127.0)
                    if with_desc:
                        for jx, jy in ((0, 0), (1, 0), (-1, 0), (0, 1), (0, -1)):
                            dp.append(_descriptor(im, xx + jx, yy + jy, *self.box))
                            jp.append((v, vi, jx == 0 and jy == 0))
                k = 0
                for v, x, y in _peaks(F, self.radius, 120):
                    if near(x, y, pts, self.r_ex + slack(amp)):
                        continue
                    zn.append(v)
                    if k < 40:
                        nx.append(Sv[A[y, x], :, y, x] / 127.0)
                        if with_desc:
                            dn.append(_descriptor(im, x, y, *self.box))
                            jn.append((v, vi, True))
                    k += 1
            self._joint_samples = (jp, jn)
            return np.array(zp), np.array(zn), px, nx, dp, dn

        for _ in range(2):
            _, _, px, nx, _, _ = collect(w)
            w = self._weights(px, nx, nc)
        self.weights = w
        keep = w > 0.03
        self.use = [c for c, k in zip(CHANNELS, keep) if k]
        self.w_use = w[keep] / w[keep].sum()
        idx = [i for i, k in enumerate(keep) if k]
        zp, zn, _, _, dp, dn = collect(self.w_use, idx, with_desc=True)
        S_all.clear()
        z_lo, z_hi = float(np.percentile(zn, 99.9)), float(np.percentile(zp, 1))
        self.z_thr = 0.5 * (z_lo + z_hi) if z_hi > z_lo else float(np.percentile(zp, 5))
        self.z_soft = max(0.25 * abs(z_hi - z_lo), 0.02)
        self.z_floor = float(np.median(zn))                                   # candidates: above the usual background peak
        D = np.array(dp + dn, np.float64)
        self.d_mu, self.d_sd = D.mean(0), D.std(0) + 1e-3
        yv = np.array([1.0] * len(dp) + [0.0] * len(dn))
        Dn = (D - self.d_mu) / self.d_sd / math.sqrt(D.shape[1])
        self.d_w, self.d_b = _logistic(Dn, yv, l2=3e-3, iters=500)

        # ---- joint decision on (match, patch): a part whose match dropped can be carried by a clear
        #      patch result and the other way round. The patch values used here are cross-fitted (each
        #      half of the copies is judged by a classifier trained on the other half) so they are honest.
        jp, jn = self._joint_samples
        vv = np.array([a for a, _, _ in jp] + [a for a, _, _ in jn])
        vi = np.array([b for _, b, _ in jp] + [b for _, b, _ in jn])
        main = np.array([c for _, _, c in jp] + [c for _, _, c in jn])
        logit = np.zeros(len(yv))
        for fold in (0, 1):
            tr, te = vi % 2 != fold, vi % 2 == fold
            if yv[tr].sum() < 1 or (1 - yv[tr]).sum() < 1 or not te.any():
                logit[te] = Dn[te] @ self.d_w + self.d_b
                continue
            fw, fb = _logistic(Dn[tr], yv[tr], l2=3e-3, iters=300)
            logit[te] = Dn[te] @ fw + fb
        J = np.stack([self._f_match(vv[main]), self._f_patch(logit[main])], 1)
        self.j_w, self.j_b = _logistic(J, yv[main], l2=1e-2, iters=800)
        self.j_w = np.maximum(self.j_w, 0.05)
        # ---- the score: a straight scale between two levels measured in training
        #        0 = the strongest background place, 1 = a typical object. A part that differs from the
        #        templates lands in between; --threshold says how far up it must be.
        jv = J @ self.j_w
        y_main = yv[main]
        negj, posj = jv[y_main < 0.5], jv[y_main > 0.5]
        # robust levels: one odd training sample must not decide the scale (top 0.5 % ignored)
        b_hi = float(np.percentile(negj, 99.5)) if len(negj) >= 200 else float(negj.max())
        o_med, o_lo = float(np.median(posj)), float(np.percentile(posj, 1))
        self.j_lo = b_hi
        self.j_span = max(o_med - b_hi, 1e-3)
        self.score_weakest = float(np.clip((o_lo - b_hi) / self.j_span, 0, 1))  # weakest training object
        self.joint_separable = bool(o_lo > b_hi)
        # objects are never closer than in the masters: half the smallest spacing is kept free
        dmin = [math.hypot(a[0] - b[0], a[1] - b[1]) for pts in inst for i, a in enumerate(pts) for b in pts[i + 1:]]
        self.min_dist = max(0.5 * min(dmin) if dmin else 0.0, float(self.r_ex))
        del self._joint_samples
        self.trained = True

        report = dict(instances_per_master=[len(i) for i in inst], instances=inst,
                      templates=len(self.templates), templates_from_user=n_user,
                      channel_weights={c: round(float(wc), 2) for c, wc in zip(CHANNELS, w) if wc > 0.03},
                      match_object_min=round(float(z_hi), 3), match_background_max=round(float(z_lo), 3),
                      match_threshold=round(self.z_thr, 3), separable=bool(z_hi > z_lo),
                      training_copies=self.n_aug, templates_cut_from_master=sum(len(o) for o in own))
        self.report = report
        if self.verbose:
            print(f"AI image processing: learned from {len(masters)} master(s), {n_user} template(s) "
                  f"(+{len(self.templates) - n_user} taken from the masters), {self.n_aug} varied copies each")
            print(f"   objects found in the master(s): {report['instances_per_master']}")
            print(f"   channel weights: {report['channel_weights']}")
            print(f"   score scale: 0 = strongest background place, 1 = typical object; the weakest training object "
                  f"scored {self.score_weakest:.2f}; boxes are shown from {self.min_score:.2f} (--threshold)"
                  + ("" if self.joint_separable else "   (WARNING: an object scored like the background)"))
            print(f"   match score: object >= {z_hi:.2f}, background <= {z_lo:.2f}, threshold {self.z_thr:.2f}"
                  + ("" if report["separable"] else "   (WARNING: object and background overlap)"))
        return report

    def _f_match(self, v):
        return np.clip((np.asarray(v, np.float64) - self.z_thr) / self.z_soft, -6, 6)

    @staticmethod
    def _f_patch(logit):
        return np.clip(np.asarray(logit, np.float64) / 2.0, -6, 6)

    @staticmethod
    def _weights(pos_x, neg_x, nc):
        """Channel weights: logistic regression, kept non-negative (a channel can help or be ignored)."""
        X = np.array(list(pos_x) + list(neg_x), np.float64)
        yv = np.array([1.0] * len(pos_x) + [0.0] * len(neg_x))
        ww, _ = _logistic(X, yv, l2=1e-3, iters=600)
        ww = np.maximum(ww, 0)
        if ww.sum() <= 1e-9:
            ww = np.ones(nc)
        return ww / ww.sum()

    # ------------------------------------------------------------------ detection
    def detect(self, img: np.ndarray, roi: Optional[Tuple[int, int, int, int]] = None) -> List[Match]:
        """roi = (x, y, w, h): search only this area of the image (boxes are returned in image pixels)."""
        assert self.trained, "call train() first"
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        ox = oy = 0
        if roi is not None:
            x, y, w, h = [int(v) for v in roi]
            x, y = max(x, 0), max(y, 0)
            img = img[y: y + h, x: x + w]
            ox, oy = x, y
        S = self._maps(img, self.use)
        F, A = self._fuse(S, self.w_use)
        out = []
        for v, x, y in _peaks(F, self.radius, 60, floor=self.z_floor):
            d = (_descriptor(img, x, y, *self.box) - self.d_mu) / self.d_sd / math.sqrt(len(self.d_mu))
            lg = float(d @ self.d_w) + self.d_b
            pc = 1 / (1 + math.exp(-np.clip(lg, -30, 30)))
            j = self.j_w[0] * float(self._f_match(v)) + self.j_w[1] * float(self._f_patch(lg))
            score = float(np.clip((j - self.j_lo) / self.j_span, 0.0, 1.0))  # 0 = background level, 1 = typical object
            ti = int(A[y, x])
            th, tw = self.templates[ti].shape[:2]
            out.append(Match(ox + x - tw // 2, oy + y - th // 2, tw, th, score, v, pc, ti,
                             (float(ox + x), float(oy + y)), dict(match=round(v, 3), patch=round(pc, 3))))
        out.sort(key=lambda m: -m.score)
        kept = []                                                             # the stronger one wins a shared place
        for m in out:
            if all(math.hypot(m.center[0] - k.center[0], m.center[1] - k.center[1]) >= self.min_dist for k in kept):
                kept.append(m)
        out = kept
        if self.expect:
            return out[: self.expect]
        return [m for m in out if m.score >= self.min_score]


def draw_matches(img: np.ndarray, matches: Sequence[Match]) -> np.ndarray:
    vis = img.copy()
    for i, m in enumerate(matches):
        cv2.rectangle(vis, (m.x, m.y), (m.x + m.w, m.y + m.h), (0, 255, 0), 2)
        label = f"#{i} {m.score:.2f}"
        cv2.putText(vis, label, (m.x, max(12, m.y - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(vis, label, (m.x, max(12, m.y - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
    return vis
