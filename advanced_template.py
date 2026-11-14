"""
advanced_template.py — rotation-, scale- and light-invariant template matching.

Opt-in matcher (run.py --advance-template, or Config(mode="template")). It works like the
"shape-based" pattern matching of industrial vision systems: the template is the set of
edge points of the reference image (texture and outline), each with its gradient direction,
and a pose (position, angle, size) is scored by how well the gradient directions of the
frame agree with the template's at those points:

    score = mean over template points of cos(2 * (frame direction - template direction))

    with frame points that have no edge counting 0 (so 1 = every template edge is present
    with the right direction, 0 = nothing in common).

  * light invariance: a gradient DIRECTION does not change with brightness, contrast,
    gamma, or a smooth light gradient across the object; doubling the angle also ignores
    contrast polarity, so a light object on a dark background matches a dark-on-light one
    (polarity=True keeps the sign when the contrast never reverses: more selective).
  * rotation and scale are applied to the edge points analytically (exact, no re-rendering
    of images), with point sets taken from a template pyramid matched to the search
    resolution.
  * coarse-to-fine: every size is searched on a reduced copy of the frame in which the object
    is about `coarse_px` px across; all rotations of one size are correlated with the frame's
    direction field in one batched FFT. The best candidates are refined level by level by a
    vectorised pose search (sub-pixel position, sub-degree angle) up to a resolution where
    the object is ~`final_px` px across.
  * rotational symmetry of the template (circle, square, ...) is detected; equivalent
    rotations are searched once and a circle is reported at 0 deg.
  * partial occlusion lowers the score in proportion to the hidden edges, instead of breaking
    the match; clutter cannot raise it above what the template's own edges allow.
"""
from __future__ import annotations

import math
from typing import Optional, Tuple

import cv2
import numpy as np
from scipy import fft as _fft


def _edge_thr(mag: np.ndarray, base: float) -> float:
    """Noise-adaptive edge threshold (5th percentile of the magnitude sits in the noise floor)."""
    v = mag[::3, ::3].ravel()
    sigma = float(np.partition(v, len(v) // 20)[len(v) // 20]) / 0.32
    return float(min(max(base, 4.5 * sigma), 2.0 * base))


def _grad(img: np.ndarray, sigma: float):
    """Gradient of the strongest colour channel at each pixel (plain Sobel for grey images), so an
    outline stays visible where object and surface have the same brightness but different colour."""
    g = cv2.GaussianBlur(img, (0, 0), sigma)
    g = np.clip(g, 0, 255).astype(np.uint8)
    if g.ndim == 2:
        gx, gy = cv2.spatialGradient(g)
        return gx.astype(np.float32), gy.astype(np.float32)
    bx = by = bm = None
    for ch in cv2.split(g):
        gx, gy = cv2.spatialGradient(ch)
        gx, gy = gx.astype(np.float32), gy.astype(np.float32)
        m = gx * gx + gy * gy
        if bm is None:
            bx, by, bm = gx, gy, m
        else:
            sel = m > bm
            bx, by, bm = np.where(sel, gx, bx), np.where(sel, gy, by), np.maximum(m, bm)
    return bx, by


class TemplateMatcher:
    def __init__(self, gray: np.ndarray, mask: Optional[np.ndarray] = None,
                 scale_range: Tuple[float, float] = (0.5, 2.0), angle_range: Tuple[float, float] = (0.0, 360.0),
                 coarse_px: int = 24, scale_step: float = 1.15, polarity: bool = False,
                 edge_base: float = 30.0, max_points: int = 400, final_px: float = 160.0,
                 zncc_mask: Optional[np.ndarray] = None, upsampled: float = 1.0,
                 color: Optional[np.ndarray] = None):
        """gray: the template (reference crop). mask: non-zero = pixels whose edges are matched.
        zncc_mask: pixels used by the grey-level correlation check (default: mask).
        color: optional BGR copy of the template; its edges are then taken from the colour channels.
        upsampled: how much the template was enlarged from the camera image (smooths its edge ramps).
        scale_range: object sizes searched, relative to this template's size."""
        self.T = gray.astype(np.float32)
        self.upsampled = max(1.0, float(upsampled))
        self.C = None if color is None else color.astype(np.float32)
        self.h, self.w = gray.shape
        self.M = np.full(gray.shape, 255, np.uint8) if mask is None else (mask > 0).astype(np.uint8) * 255
        self.dim = float(max(self.h, self.w))
        self.c0 = np.array([(self.w - 1) / 2.0, (self.h - 1) / 2.0])           # template centre, pixel-index coords
        self.coarse_px, self.final_px = coarse_px, final_px
        self.coarse_px_requested = coarse_px
        self.polarity = polarity
        self.mult = 1.0 if polarity else 2.0                                  # angle multiplier
        self.edge_base = edge_base
        self.max_points = max_points
        self.angle_range = angle_range
        smin, smax = scale_range
        self.scales = [smin]
        while self.scales[-1] * scale_step <= smax * 1.0001:
            self.scales.append(self.scales[-1] * scale_step)
        self.scale_step = scale_step
        self.octaves = {}                                                     # k -> (xy, theta) edge points
        for base in (edge_base, edge_base / 2, edge_base / 4):               # low-contrast template: lower the bar
            self.edge_base, self.octaves = base, {}
            if len(self._points(1.0)[0]) >= 30:
                break
        if len(self.octaves[1.0][0]) < 12:
            raise ValueError("the template has (almost) no edges: nothing to match")
        # small grey copy of the whole template for the pixel-correlation check
        self.kv = min(1.0, 96.0 / self.dim)
        self.Tv = cv2.resize(self.T if self.C is None else self.C,
                             (max(8, round(self.w * self.kv)), max(8, round(self.h * self.kv))),
                             interpolation=cv2.INTER_AREA)
        src0 = self.T if self.C is None else self.C
        lo, hi = np.percentile(src0[self.M > 0], [2, 98])
        self.contrast = float(max(hi - lo, 1.0))                              # template's grey-level range
        Mz = self.M if zncc_mask is None else (zncc_mask > 0).astype(np.uint8) * 255
        self.Mv = cv2.resize(Mz, self.Tv.shape[1::-1], interpolation=cv2.INTER_NEAREST) > 127
        # fine detail (print, small marks) vanishes at the coarse size: then search on a finer grid
        if self._coarse_self_score(coarse_px) < 0.35:
            coarse_px *= 2
            self.coarse_px = coarse_px
        self.fine_detail = coarse_px != self.coarse_px_requested
        # angle step: the farthest edge point moves ~1.5 px at the coarse size
        r = 0.5 * coarse_px * float(np.linalg.norm(self.octaves[1.0][0], axis=1).max()) / (0.5 * self.dim)
        self.angle_step = float(np.clip(math.degrees(1.5 / max(r, 1.0)), 3.0, 12.0))
        self.sym = self._symmetry()
        a0, a1 = angle_range
        angles = np.arange(a0, a1 - 1e-6, self.angle_step)
        if self.sym == 0:
            angles = angles[:1]
        elif self.sym > 1:
            angles = angles[angles < a0 + 360.0 / self.sym - 1e-6]
        self.angles = angles
        self.bank = [(max(1.0, s * self.dim / coarse_px), s) for s in self.scales]
        self.n_templates = len(self.scales) * len(self.angles)
        self._spec = {}

    def _coarse_self_score(self, cp: float) -> float:
        """How well the template matches itself when seen at cp pixels (half a pixel off)."""
        q = min(1.0, cp / self.dim)
        src = self.T if self.C is None else self.C
        img = cv2.resize(src, (max(8, round(self.w * q)), max(8, round(self.h * q))), interpolation=cv2.INTER_AREA)
        c, s = self._field(img, 1.0)
        xy, th = self._points(self._octave(q))
        P = np.array([[self.c0[0] + 0.5 / q, self.c0[1] + 0.5 / q, 0.0, 1.0]])
        return float(self._eval(cv2.merge([c, s]), q, P, xy, th)[0])

    # ------------------------------------------------------------------ template edge points
    def _points(self, k: float):
        """Edge points of the template resized by k: positions relative to the template centre in
        FULL-template pixels, gradient directions (rad). Cached per pyramid octave."""
        if k in self.octaves:
            return self.octaves[k]
        src = self.T if self.C is None else self.C
        if k >= 1.0:
            T, M = src, self.M
        else:
            sz = (max(6, round(self.w * k)), max(6, round(self.h * k)))
            T = cv2.resize(src, sz, interpolation=cv2.INTER_AREA)
            M = cv2.resize(self.M, sz, interpolation=cv2.INTER_AREA)
        kx, ky = T.shape[1] / self.w, T.shape[0] / self.h
        # an enlarged template has linear ramps instead of edges: blur to give them one ridge line
        gx, gy = _grad(T, 0.8 * max(1.0, self.upsampled * kx))
        mag = cv2.magnitude(gx, gy)
        thr = _edge_thr(mag, self.edge_base)
        h, w = mag.shape
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        ux, uy = gx / (mag + 1e-6), gy / (mag + 1e-6)
        fwd = cv2.remap(mag, xx + ux, yy + uy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        bwd = cv2.remap(mag, xx - ux, yy - uy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        keep = (mag > thr) & (mag >= fwd) & (mag > bwd) & (M > 127)          # thin edges to their centre line
        keep[:2, :] = keep[-2:, :] = False
        keep[:, :2] = keep[:, -2:] = False
        ys, xs = np.nonzero(keep)
        if len(xs) > self.max_points:
            sel = np.random.default_rng(0).choice(len(xs), self.max_points, replace=False)
            ys, xs = ys[sel], xs[sel]
        xy = np.c_[(xs + 0.5) / kx - 0.5 - self.c0[0], (ys + 0.5) / ky - 0.5 - self.c0[1]].astype(np.float32)
        th = np.arctan2(gy[ys, xs], gx[ys, xs]).astype(np.float32)
        if len(xs) < 12 and k < 1.0:
            # too few edges survive at this reduction: reuse the next finer level's points
            # (positions are in full-template pixels, so they are valid at any level)
            xy, th = self._points(min(1.0, 2.0 * k))
            if len(xy) > 40:
                sel = np.random.default_rng(1).choice(len(xy), max(40, int(len(xy) * k)), replace=False)
                xy, th = xy[sel], th[sel]
        self.octaves[k] = (xy, th)
        return self.octaves[k]

    def _octave(self, k: float) -> float:
        return float(min(1.0, 2.0 ** math.floor(math.log2(max(k, 1e-3)) + 0.5)))

    # ------------------------------------------------------------------ frame direction field
    def _field(self, gray: np.ndarray, blur: float):
        gx, gy = _grad(gray, 0.8)
        mag = cv2.magnitude(gx, gy)
        w = (mag > _edge_thr(mag, self.edge_base)).astype(np.float32)
        th = np.arctan2(gy, gx) * self.mult
        c, s = np.cos(th) * w, np.sin(th) * w
        if blur > 0:
            c, s = cv2.GaussianBlur(c, (0, 0), blur), cv2.GaussianBlur(s, (0, 0), blur)
        return c, s

    # ------------------------------------------------------------------ symmetry
    def _symmetry(self) -> int:
        """0 = same at every angle, k = k-fold, 1 = none (self-similarity of the direction field)."""
        xy, th = self._points(self._octave(48.0 / self.dim))
        R = float(np.linalg.norm(xy, axis=1).max()) + 1e-6
        n = 41

        def field(deg):
            a = math.radians(deg)
            ca, sa = math.cos(a), math.sin(a)
            x = (ca * xy[:, 0] + sa * xy[:, 1]) / R * (n // 2 - 3) + n // 2
            y = (-sa * xy[:, 0] + ca * xy[:, 1]) / R * (n // 2 - 3) + n // 2
            T = np.zeros((n, n), np.complex64)
            np.add.at(T, (np.rint(y).astype(int), np.rint(x).astype(int)), np.exp(1j * self.mult * (th - a)))
            m = np.abs(T)
            T = np.where(m > 1e-6, T / np.maximum(m, 1e-6), 0)
            T = cv2.GaussianBlur(T.real.astype(np.float32), (0, 0), 1.5) + 1j * cv2.GaussianBlur(T.imag.astype(np.float32), (0, 0), 1.5)
            return T / (np.linalg.norm(T) + 1e-9)

        T0 = field(0.0)
        same = lambda d: float((T0 * np.conj(field(d))).real.sum())
        # only a really round outline skips the angle search; near-round polygons (octagon, ~0.8)
        # fall through to the k-fold test below and keep a short angle search
        if min(same(17.0), same(22.5), same(49.0), same(73.0), same(131.0)) > 0.9:
            return 0
        for kf in (8, 6, 5, 4, 3, 2):
            if all(same(360.0 * j / kf) > 0.75 for j in range(1, kf)):
                return kf
        return 1

    def canonical_angle(self, ang: float) -> float:
        if self.sym == 0:
            return 0.0
        if self.sym > 1:
            step = 360.0 / self.sym
            ang = ang - round(ang / step) * step
        return (ang + 180.0) % 360.0 - 180.0

    # ------------------------------------------------------------------ coarse search
    def _raster(self, xy, th, ang, k):
        a = math.radians(ang)
        ca, sa = math.cos(a), math.sin(a)
        px = k * (ca * xy[:, 0] + sa * xy[:, 1])
        py = k * (-sa * xy[:, 0] + ca * xy[:, 1])
        r = int(math.ceil(max(np.abs(px).max(), np.abs(py).max()))) + 1
        T = np.zeros((2 * r + 1, 2 * r + 1), np.complex64)
        np.add.at(T, (np.rint(py + r).astype(int), np.rint(px + r).astype(int)), np.exp(1j * self.mult * (th - a)))
        m = np.abs(T)
        T = np.where(m > 1e-6, T / np.maximum(m, 1e-6), 0).astype(np.complex64)
        return T, float((m > 1e-6).sum()), r

    def _coarse(self, gray: np.ndarray, thr: float, per_template: int = 3, per_scale: int = 40):
        h, w = gray.shape[:2]
        out = []
        prev = gray
        for gi, (ds, s) in enumerate(self.bank):
            sw, sh = max(8, round(w / ds)), max(8, round(h / ds))
            src = prev if prev.shape[1] >= 1.15 * sw else gray
            f = cv2.resize(src, (sw, sh), interpolation=cv2.INTER_AREA) if ds > 1 else gray
            prev = f
            k = s / ds                                                        # template px -> level px
            P, Q = _fft.next_fast_len(sh), _fft.next_fast_len(sw)
            key = (gi, P, Q)
            if key not in self._spec:
                xy, th = self._points(self._octave(k))
                ents, stack = [], []
                for a in self.angles:
                    T, n, r = self._raster(xy, th, float(a), k)
                    if 2 * r + 1 >= min(sh, sw) or n < 8:
                        continue
                    Tp = np.zeros((P, Q), np.complex64)
                    Tp[:T.shape[0], :T.shape[1]] = T
                    stack.append(Tp)
                    ents.append((float(a), n, r))
                spec = _fft.fft2(np.stack(stack), workers=-1).conj() if stack else None
                self._spec[key] = (ents, spec)
            ents, spec = self._spec[key]
            if spec is None:
                continue
            c, sn = self._field(f, 1.0)
            Fp = np.zeros((P, Q), np.complex64)
            Fp[:sh, :sw] = c + 1j * sn
            R = _fft.ifft2(spec * _fft.fft2(Fp, workers=-1)[None], workers=-1).real
            # all angles centre-aligned in one stack -> one peak search per scale
            Z = np.full((len(ents), sh, sw), -1.0, np.float32)
            rmin = min(r for _, _, r in ents)
            for j, (a, n, r) in enumerate(ents):
                Z[j, r: sh - r, r: sw - r] = R[j, : sh - 2 * r, : sw - 2 * r] / n
            zmax = Z.max(0)
            if zmax.max() < thr:
                continue
            kk = max(3, rmin | 1)
            peaks = (zmax >= cv2.dilate(zmax, np.ones((kk, kk), np.uint8))) & (zmax >= thr)
            ys, xs = np.nonzero(peaks)
            order = np.argsort(-zmax[ys, xs])[:per_scale]
            na = len(ents)
            for y, x in zip(ys[order], xs[order]):
                prof = Z[:, y, x]
                # local maxima along the angle axis (circular when the full turn is searched)
                if na >= 3:
                    lm = (prof >= np.roll(prof, 1)) & (prof >= np.roll(prof, -1)) & (prof >= thr)
                else:
                    lm = prof >= thr
                js = np.nonzero(lm)[0]
                js = js[np.argsort(-prof[js])][:per_template]
                cx = (x + 0.5) * (w / sw) - 0.5                               # template centre, frame pixel-index coords
                cy = (y + 0.5) * (h / sh) - 0.5
                for j in js:
                    out.append((float(prof[j]), cx, cy, ents[j][0], s))
        return out

    # ------------------------------------------------------------------ refinement
    def _level(self, gray, q):
        key = round(q, 4)
        lv = self._levels.get(key)
        if lv is None:
            img = gray if q >= 1.0 else cv2.resize(gray, (max(8, round(gray.shape[1] * q)), max(8, round(gray.shape[0] * q))),
                                                    interpolation=cv2.INTER_AREA)
            c, s = self._field(img, 0.7)
            lv = self._levels[key] = cv2.merge([c, s])
        return lv

    def _eval(self, F, q, P, xy, th):
        """Scores of poses P (N x 4: cx, cy, angle deg, scale; frame coords) on level field F."""
        a = np.radians(P[:, 2])[:, None]
        ca, sa, k = np.cos(a), np.sin(a), (P[:, 3] * q)[:, None]
        px = (P[:, 0, None] + 0.5) * q - 0.5 + k * (ca * xy[:, 0] + sa * xy[:, 1])
        py = (P[:, 1, None] + 0.5) * q - 0.5 + k * (-sa * xy[:, 0] + ca * xy[:, 1])
        v = cv2.remap(F, px.astype(np.float32), py.astype(np.float32), cv2.INTER_LINEAR, borderValue=0)
        t = self.mult * (th[None, :] - a)
        return (v[..., 0] * np.cos(t) + v[..., 1] * np.sin(t)).mean(1)

    def _refine(self, gray, score, cx, cy, ang, s, reject: float = -1.0, levels: int = 99):
        size = s * self.dim
        snap = lambda v: min(1.0, 2.0 ** (math.ceil(math.log2(v) * 2.0 - 1e-6) / 2.0))   # shared sqrt(2) levels
        q = snap(min(1.0, self.coarse_px / size))
        q_final = snap(min(1.0, self.final_px / size))
        da = 0.0 if self.sym == 0 else self.angle_step / 2.0
        dsr = (self.scale_step - 1.0) / 2.0
        cur = np.array([cx, cy, ang, s], np.float64)
        first = True
        while True:
            q = min(2.0 * q, q_final)
            F = self._level(gray, q)
            xy, th = self._points(self._octave(cur[3] * q))
            dp = 1.0 / q                                                      # one level pixel, in frame px
            if first:     # small exhaustive grid: the coarse peak is only accurate to about one coarse pixel
                o = np.array([-1.0, 0.0, 1.0])
                gx_, gy_, ga, gs = np.meshgrid(o * dp, o * dp, o * da if da else [0.0], [1 / (1 + dsr), 1.0, 1 + dsr], indexing="ij")
                P = np.c_[cur[0] + gx_.ravel(), cur[1] + gy_.ravel(), cur[2] + ga.ravel(), cur[3] * gs.ravel()]
                v = self._eval(F, q, P, xy, th)
                j = int(np.argmax(v))
                cur, best = P[j], float(v[j])
                if best < reject:
                    return best, cur
                first = False
            else:
                best = float(self._eval(F, q, cur[None], xy, th)[0])
            # pattern search with halving steps
            step = np.array([dp, dp, da, dsr])
            for _ in range(4):
                for _ in range(6):
                    x, y, a, sc = cur
                    C = np.array([[x + step[0], y, a, sc], [x - step[0], y, a, sc], [x, y + step[1], a, sc], [x, y - step[1], a, sc],
                                  [x, y, a + step[2], sc], [x, y, a - step[2], sc], [x, y, a, sc * (1 + step[3])], [x, y, a, sc / (1 + step[3])]])
                    v = self._eval(F, q, C, xy, th)
                    j = int(np.argmax(v))
                    if v[j] <= best + 1e-4:
                        break
                    best, cur = float(v[j]), C[j]
                step = step / 2.0
            da, dsr = da / 2.0, dsr / 2.0
            levels -= 1
            if levels <= 0:
                return best, cur
            if q >= q_final:
                break
        # final score on the full point set at the final resolution
        xy, th = self._points(self._octave(cur[3] * q))
        best = float(self._eval(self._level(gray, q), q, cur[None], xy, th)[0])
        return best, cur

    def zncc(self, img: np.ndarray, cx, cy, ang, s) -> float:
        """Zero-mean normalised correlation of the template (colour when it has colour; its own
        surroundings included where the mask allows) with the frame at this pose. Each channel's mean
        is removed, so it is invariant to brightness offset and contrast gain."""
        hv, wv = self.Tv.shape[:2]
        if (img.ndim == 3) != (self.Tv.ndim == 3):
            return 0.0
        a = math.radians(ang)
        R = s * np.array([[math.cos(a), math.sin(a)], [-math.sin(a), math.cos(a)]])
        # patch pixel (u, v) -> template index coords -> frame index coords
        Lin = R / self.kv
        off = np.array([cx, cy]) + R @ ((0.5 / self.kv - 0.5) - self.c0)
        A = np.hstack([Lin, off[:, None]])
        sig = 0.5 * s / self.kv - 0.5                                         # anti-alias when the frame is finer
        src = img
        if sig > 0.3:
            ext = 0.6 * s * self.dim + 4
            x0, y0 = int(max(cx - ext, 0)), int(max(cy - ext, 0))
            x1, y1 = int(min(cx + ext, img.shape[1])), int(min(cy + ext, img.shape[0]))
            if x1 - x0 < 4 or y1 - y0 < 4:
                return 0.0
            src = cv2.GaussianBlur(img[y0:y1, x0:x1], (0, 0), sig)
            A = A.copy()
            A[:, 2] -= (x0, y0)
        patch = cv2.warpAffine(src, A.astype(np.float32), (wv, hv), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                               borderMode=cv2.BORDER_CONSTANT, borderValue=(-1, -1, -1, -1))
        inside = patch >= 0 if patch.ndim == 2 else (patch >= 0).all(2)
        ok = self.Mv & inside
        if ok.sum() < 0.7 * max(int(self.Mv.sum()), 1):                     # most of the object inside the frame
            return 0.0
        p, t = patch[ok], self.Tv[ok]
        p, t = (p - p.mean(0)).ravel(), (t - t.mean(0)).ravel()
        den = math.sqrt(float(p @ p) * float(t @ t))
        return float(p @ t) / den if den > 1e-6 else 0.0

    # ------------------------------------------------------------------ main
    def find(self, gray: np.ndarray, min_score: float = 0.6, max_instances: int = 20,
             coarse_thr: Optional[float] = None, max_candidates: int = 20, reject_margin: float = 0.25,
             min_zncc: float = -1.0, per_location: int = 4, color: Optional[np.ndarray] = None):
        """gray: frame (color: the same frame in BGR, used for the edges when the template has colour). Returns [(edge_score, cx, cy, angle_deg, scale, zncc)], scale relative to the
        template, (cx, cy) = template centre in frame pixel-index coordinates."""
        g = gray.astype(np.float32)
        e = color.astype(np.float32) if (color is not None and self.C is not None) else g   # edge source
        # a dim / low-contrast frame (dark lighting): stretch it to the template's contrast so the edge
        # threshold means the same thing (directions are unchanged; only boosts, never reduces)
        zimg = e if e.ndim == 3 else g                                        # correlation: raw values
        lo, hi = np.percentile(e[::4, ::4], [2, 98])
        gain = min(4.0, self.contrast / max(float(hi - lo), 1.0))
        if gain > 1.15:
            e = (e - float(lo)) * gain
        self._levels = {}
        thr = coarse_thr if coarse_thr is not None else max(0.15, min_score - 0.35)
        cands = self._coarse(e, thr)
        cands.sort(key=lambda c: -c[0])
        # group the coarse peaks by location. A coarse view of a simple shape scores almost the same at
        # several angles / sizes, so each location keeps a few clearly different poses to refine.
        period = 360.0 / self.sym if self.sym > 0 else 0.0
        min_da, min_ds = 1.5 * self.angle_step, math.log(self.scale_step) * 1.5

        def distinct(c, h):
            if abs(math.log(c[4] / h[4])) > min_ds:
                return True
            if period == 0.0:
                return False
            d = abs(c[3] - h[3]) % period
            return min(d, period - d) > min_da

        groups = []
        for c in cands:
            for gr in groups:
                L = gr[0]
                r = 0.35 * min(c[4], L[4]) * self.dim
                if (c[1] - L[1]) ** 2 + (c[2] - L[2]) ** 2 <= r * r:
                    if len(gr) < per_location and all(distinct(c, h) for h in gr):
                        gr.append(c)
                    break
            else:
                if len(groups) < max_candidates:
                    groups.append([c])
        res = []
        floor = min_score - reject_margin
        for gr in groups:
            # one cheap level for every pose of this location, then the full refinement for the best 1-2
            pre = sorted((self._refine(e, *c, levels=1) for c in gr), key=lambda t: -t[0])
            if pre[0][0] < floor:
                continue
            best = None
            for sc0, p0 in pre[:2]:
                if sc0 < max(floor, pre[0][0] - 0.1):
                    break
                sc, p = self._refine(e, sc0, *p0, reject=floor)
                if best is None or sc > best[0]:
                    best = (sc, p)
            sc, p = best
            if sc >= min_score:
                z = self.zncc(zimg, *p)
                if z >= min_zncc:
                    res.append((sc, float(p[0]), float(p[1]), self.canonical_angle(float(p[2])), float(p[3]), z))
        res.sort(key=lambda r: -r[0])
        out = []
        for r in res:
            rad = 0.5 * r[4] * self.dim
            if all((r[1] - o[1]) ** 2 + (r[2] - o[2]) ** 2 > rad * rad for o in out):
                out.append(r)
            if len(out) >= max_instances:
                break
        return out

    def pose_matrix(self, cx, cy, ang, s) -> np.ndarray:
        """3x3 similarity: template pixel coords (corner convention, (0,0) = top-left corner)
        -> frame pixel coords (corner convention)."""
        a = math.radians(ang)
        R = s * np.array([[math.cos(a), math.sin(a)], [-math.sin(a), math.cos(a)]])
        t = np.array([cx + 0.5, cy + 0.5]) - R @ (self.c0 + 0.5)
        return np.array([[R[0, 0], R[0, 1], t[0]], [R[1, 0], R[1, 1], t[1]], [0, 0, 1.0]])