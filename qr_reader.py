"""
qr_reader.py  -  accurate, repeatable QR-code reading from camera frames.

Why this is more reliable than a single ``decode(frame)`` call
--------------------------------------------------------------
1. THREE decoders, tried in order, so a code that one engine misses is
   usually caught by the next:
       pyzbar / ZBar         fast, very robust on clean industrial codes
       OpenCV WeChat QR      (only if opencv-contrib is installed) best on
                             small / blurred / low-contrast codes
       OpenCV QR detector    always available (ArUco variant when present)
2. PRE-PROCESSING ladder per frame (cheapest first, stops at the first hit):
   quiet-zone padding, up-scaling of small codes, CLAHE, Otsu, adaptive
   threshold, unsharp mask, denoise + Otsu, inverted (light-on-dark).
3. MULTI-FRAME CONSENSUS (the part that gives repeatability): a value is
   only accepted when it has been decoded from ``min_agree`` DIFFERENT
   camera frames and no other value ties or beats it.  A single lucky /
   corrupted decode can therefore never be returned as the result.
4. FRESH FRAMES ONLY: frames are identified by a sequence number, frames
   that were captured before the scan started are ignored, and the same
   frame is never counted twice.
5. RULES (optional): exact length and/or starting characters.  A code that
   decodes but breaks the rules is reported as a failure with the value in
   the message, so the operator sees *why*.

Public API
----------
    QrRules(length=0, starts_with="")           0 / "" = not checked
    QrReader().read_stream(frame_source, rules, roi_norm, timeout_s, ...)
    QrReader().read_image(image, rules, roi_norm)
    norm_from_rect(rect, frame_w, frame_h) / rect_from_norm(norm, w, h)
    available_backends()

``frame_source()`` must return ``(seq, frame_bgr)``.  ``seq`` is an int that
increases with every NEW camera frame (or None if the source cannot tell).
"""
import time
from collections import Counter
from dataclasses import dataclass, field

import cv2
import numpy as np

try:                                            # pip install pyzbar
    from pyzbar import pyzbar as _pyzbar        # + apt install libzbar0
    from pyzbar.pyzbar import ZBarSymbol as _ZBarSymbol
    HAVE_ZBAR = True
except Exception:                               # lib or wrapper missing
    _pyzbar = None
    _ZBarSymbol = None
    HAVE_ZBAR = False

HAVE_WECHAT = hasattr(cv2, "wechat_qrcode_WeChatQRCode")


def available_backends():
    names = []
    if HAVE_ZBAR:
        names.append("zbar")
    if HAVE_WECHAT:
        names.append("wechat")
    names.append("opencv")
    return names


# ---------------------------------------------------------------------------
# rules
# ---------------------------------------------------------------------------
@dataclass
class QrRules:
    length: int = 0            # 0  -> any length
    starts_with: str = ""      # "" -> any start

    @classmethod
    def from_dict(cls, d):
        d = d or {}
        try:
            length = int(d.get("length") or 0)
        except (TypeError, ValueError):
            length = 0
        return cls(length=max(0, length),
                   starts_with=str(d.get("starts_with") or ""))

    def validate(self, value):
        """-> (ok, reason).  Empty rules always pass."""
        if self.length and len(value) != self.length:
            return False, (f"length is {len(value)}, "
                           f"expected {self.length}")
        if self.starts_with and not value.startswith(self.starts_with):
            return False, (f"does not start with '{self.starts_with}'")
        return True, ""

    def describe(self):
        parts = []
        if self.length:
            parts.append(f"length={self.length}")
        if self.starts_with:
            parts.append(f"starts='{self.starts_with}'")
        return ", ".join(parts) if parts else "no rules"


# ---------------------------------------------------------------------------
# result
# ---------------------------------------------------------------------------
@dataclass
class QrResult:
    ok: bool = False
    value: str = ""
    reason: str = ""
    frames: int = 0                         # distinct frames analysed
    elapsed: float = 0.0
    backend: str = ""
    polygon: list = field(default_factory=list)   # full-frame pixel coords
    votes: dict = field(default_factory=dict)     # value -> agreeing frames
    frame_size: tuple = (0, 0)              # (w, h) of the frame it came from
    last_crop: object = None                # last analysed ROI (for debugging)

    def bbox(self):
        """(x, y, w, h) around the decoded code in full-frame pixels."""
        if not self.polygon:
            return None
        xs = [p[0] for p in self.polygon]
        ys = [p[1] for p in self.polygon]
        return (int(min(xs)), int(min(ys)),
                int(max(xs) - min(xs)), int(max(ys) - min(ys)))


# ---------------------------------------------------------------------------
# ROI helpers  (ROIs are stored as fractions so they survive a change of the
# display scale / camera resolution)
# ---------------------------------------------------------------------------
def norm_from_rect(rect, frame_w, frame_h):
    x, y, w, h = rect
    return [round(x / frame_w, 6), round(y / frame_h, 6),
            round(w / frame_w, 6), round(h / frame_h, 6)]


def rect_from_norm(norm, frame_w, frame_h):
    """-> (x0, y0, x1, y1) clamped to the frame, or the whole frame."""
    if not norm:
        return 0, 0, frame_w, frame_h
    x, y, w, h = norm
    x0 = int(max(0, min(frame_w - 1, round(x * frame_w))))
    y0 = int(max(0, min(frame_h - 1, round(y * frame_h))))
    x1 = int(max(x0 + 1, min(frame_w, round((x + w) * frame_w))))
    y1 = int(max(y0 + 1, min(frame_h, round((y + h) * frame_h))))
    return x0, y0, x1, y1


# ---------------------------------------------------------------------------
# reader
# ---------------------------------------------------------------------------
def _to_text(data):
    if isinstance(data, str):
        txt = data
    else:
        try:
            txt = data.decode("utf-8")
        except UnicodeDecodeError:
            txt = data.decode("latin-1", errors="replace")
    return txt.strip().strip("\x00").strip()


def _poly_area(poly):
    if not poly or len(poly) < 3:
        return 0.0
    xs = [p[0] for p in poly]
    ys = [p[1] for p in poly]
    return 0.5 * abs(sum(xs[i] * ys[(i + 1) % len(poly)]
                         - xs[(i + 1) % len(poly)] * ys[i]
                         for i in range(len(poly))))


class QrReader:
    """One instance per scan (decoder objects are not shared across threads)."""

    def __init__(self):
        self._wechat = None
        if HAVE_WECHAT:
            try:
                self._wechat = cv2.wechat_qrcode_WeChatQRCode()
            except Exception:
                self._wechat = None
        try:
            self._cv = (cv2.QRCodeDetectorAruco()
                        if hasattr(cv2, "QRCodeDetectorAruco")
                        else cv2.QRCodeDetector())
        except Exception:
            self._cv = cv2.QRCodeDetector()

    # -- single decoders: return [(text, polygon_in_image_coords)] ------------
    @staticmethod
    def _dec_zbar(img):
        out = []
        for s in _pyzbar.decode(img, symbols=[_ZBarSymbol.QRCODE]):
            out.append((_to_text(s.data), [(p.x, p.y) for p in s.polygon]))
        return out

    _WECHAT_MAX_SIDE = 800      # its run-time explodes on big inputs
                                # (~4 s at 1 MP, ~18 s at 2.5 MP)

    def _dec_wechat(self, img):
        h, w = img.shape[:2]
        k = 1.0
        if max(h, w) > self._WECHAT_MAX_SIDE:
            k = self._WECHAT_MAX_SIDE / float(max(h, w))
            img = cv2.resize(img, None, fx=k, fy=k, interpolation=cv2.INTER_AREA)
        texts, pts = self._wechat.detectAndDecode(img)
        out = []
        for t, p in zip(texts, pts):
            if t:
                out.append((_to_text(t),
                            [(float(a) / k, float(b) / k)
                             for a, b in p.reshape(-1, 2)]))
        return out

    def _dec_cv(self, img):
        text, pts, _ = self._cv.detectAndDecode(img)
        if text:
            poly = ([] if pts is None else
                    [(float(a), float(b)) for a, b in pts.reshape(-1, 2)])
            return [(_to_text(text), poly)]
        return []

    def _decoders(self):
        """[(name, fn, max_variants)] in priority order.

        ZBar is cheap (a few ms) so it gets the whole pre-processing ladder.
        WeChat / OpenCV can take 100-500 ms on a cluttered frame, so they only
        get the first few variants - they do their own internal scaling and
        are here as a safety net for what ZBar cannot read."""
        d = []
        if HAVE_ZBAR:
            d.append(("zbar", self._dec_zbar, 99))
        if self._wechat is not None:
            d.append(("wechat", self._dec_wechat, 3))
        d.append(("opencv", self._dec_cv, 99 if not d else 3))
        return d

    # -- pre-processing ladder ------------------------------------------------
    @staticmethod
    def _variants(gray):
        """Yield (img, scale, pad) - cheapest / most common first.
        scale & pad let us map a polygon back to the un-processed crop."""
        h, w = gray.shape[:2]
        longest = max(h, w)
        pad = max(16, int(0.08 * longest))
        bg = int(np.percentile(gray, 95))

        def padded(im, p):
            return cv2.copyMakeBorder(im, p, p, p, p,
                                      cv2.BORDER_CONSTANT, value=bg)

        # 1. plain, with a quiet zone (tight ROIs often cut it off)
        yield padded(gray, pad), 1.0, pad

        # small codes decode far better when enlarged
        scale = 1.0
        base = gray
        if longest < 500:
            scale = min(3.0, 800.0 / longest)
            base = cv2.resize(gray, None, fx=scale, fy=scale,
                              interpolation=cv2.INTER_CUBIC)
        bpad = max(16, int(0.08 * max(base.shape[:2])))
        if scale != 1.0:
            yield padded(base, bpad), scale, bpad

        # 2. local contrast
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8)).apply(base)
        yield padded(clahe, bpad), scale, bpad

        # 3. global threshold
        _, otsu = cv2.threshold(base, 0, 255,
                                cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        yield padded(otsu, bpad), scale, bpad

        # 4. adaptive threshold (uneven lighting / glare gradients)
        blk = int(min(base.shape[:2]) / 8) | 1
        blk = max(11, min(51, blk))
        ada = cv2.adaptiveThreshold(base, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                    cv2.THRESH_BINARY, blk, 5)
        yield padded(ada, bpad), scale, bpad

        # 5. unsharp mask (slightly out of focus)
        soft = cv2.GaussianBlur(base, (0, 0), 2.0)
        sharp = cv2.addWeighted(base, 1.8, soft, -0.8, 0)
        yield padded(sharp, bpad), scale, bpad

        # 6. denoise then Otsu (sensor noise / moire)
        den = cv2.GaussianBlur(base, (3, 3), 0)
        _, den_o = cv2.threshold(den, 0, 255,
                                 cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        yield padded(den_o, bpad), scale, bpad

        # 7. inverted (light code on dark background)
        yield padded(255 - base, bpad), scale, bpad

    # -- one image -> best candidate -----------------------------------------
    def decode_image(self, bgr, rules=None, budget_s=1.5, abort=None):
        """Try the backends, in priority order, on the pre-processing ladder.
        -> (text, polygon_in_bgr_coords, backend) or None.

        Prefers a code that satisfies the rules, then the largest one.
        ``budget_s`` caps the time spent on ONE frame so a hard / cluttered
        frame can never block cancel or the overall timeout."""
        if bgr is None or bgr.size == 0:
            return None
        gray = (bgr if bgr.ndim == 2 else cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY))
        deadline = time.time() + budget_s
        ladder_iter = self._variants(gray)
        ladder = []                     # built lazily, shared by all backends

        def variant(i):
            while len(ladder) <= i:
                try:
                    ladder.append(next(ladder_iter))
                except StopIteration:
                    return None
            return ladder[i]

        fallback = None
        for name, fn, max_variants in self._decoders():
            for i in range(max_variants):
                if time.time() > deadline or (abort is not None and abort()):
                    return fallback
                v = variant(i)
                if v is None:
                    break
                img, scale, pad = v
                try:
                    found = fn(img)
                except Exception:
                    break               # a broken backend must never kill a scan
                if not found:
                    continue
                cands = []
                for text, poly in found:
                    if not text:
                        continue
                    poly = [((x - pad) / scale, (y - pad) / scale)
                            for x, y in poly]
                    cands.append((text, poly, name))
                if not cands:
                    continue
                cands.sort(key=lambda c: _poly_area(c[1]), reverse=True)
                if rules is not None:
                    for c in cands:
                        if rules.validate(c[0])[0]:
                            return c
                    fallback = fallback or cands[0]
                    continue            # a later variant may decode a valid one
                return cands[0]
        return fallback

    # -- single still image ---------------------------------------------------
    def read_image(self, image, rules=None, roi_norm=None):
        """One frame, no consensus (setup / testing)."""
        rules = rules or QrRules()
        t0 = time.time()
        h, w = image.shape[:2]
        x0, y0, x1, y1 = rect_from_norm(roi_norm, w, h)
        crop = image[y0:y1, x0:x1]
        res = QrResult(frames=1, frame_size=(w, h), last_crop=crop)
        hit = self.decode_image(crop, rules)
        res.elapsed = time.time() - t0
        if hit is None:
            res.reason = "no QR code found in the region"
            return res
        text, poly, backend = hit
        res.backend = backend
        res.polygon = [(px + x0, py + y0) for px, py in poly]
        ok, why = rules.validate(text)
        res.votes = {text: 1}
        if not ok:
            res.value = text
            res.reason = f"read '{text}' but {why}"
            return res
        res.ok, res.value = True, text
        return res

    # -- the accurate / repeatable path ----------------------------------------
    def read_stream(self, frame_source, rules=None, roi_norm=None,
                    timeout_s=6.0, min_agree=2, stop=None):
        """Read from live frames until ``min_agree`` different frames agree.

        frame_source() -> (seq, frame_bgr);  stop() -> True cancels.
        """
        rules = rules or QrRules()
        min_agree = max(1, int(min_agree))
        t0 = time.time()
        votes = Counter()           # valid values  -> frames that read it
        bad = Counter()             # rule-breaking -> frames that read it
        why_bad = {}
        polys = {}
        backends = {}
        res = QrResult()

        start_seq, _ = frame_source()
        last_seq = start_seq
        frames = 0

        while time.time() - t0 < timeout_s:
            if stop is not None and stop():
                res.reason = "cancelled"
                break
            seq, frame = frame_source()
            if frame is None:
                time.sleep(0.02)
                continue
            if seq is not None:
                if last_seq is not None and seq <= last_seq:
                    time.sleep(0.005)       # no new frame yet
                    continue
                last_seq = seq
            else:
                time.sleep(0.03)            # source can't tell: pace ourselves

            frames += 1
            h, w = frame.shape[:2]
            x0, y0, x1, y1 = rect_from_norm(roi_norm, w, h)
            crop = frame[y0:y1, x0:x1]
            res.frame_size = (w, h)
            res.last_crop = crop
            hit = self.decode_image(crop, rules, abort=stop)
            if hit is None:
                continue
            text, poly, backend = hit
            ok, why = rules.validate(text)
            if not ok:
                bad[text] += 1
                why_bad[text] = why
                if bad[text] >= max(2, min_agree) and not votes:
                    break               # consistent + rule-breaking: stop early
                continue
            votes[text] += 1
            polys[text] = [(px + x0, py + y0) for px, py in poly]
            backends[text] = backend
            top = votes.most_common(2)
            best, n = top[0]
            second = top[1][1] if len(top) > 1 else 0
            # the winner must be seen in >= min_agree frames AND lead every
            # other value by min_agree frames - a lone misread / a second code
            # drifting through the region can never win by a single frame
            if n >= min_agree and n - second >= min_agree:
                res.ok, res.value = True, best
                res.backend = backends[best]
                res.polygon = polys[best]
                res.votes = dict(votes)
                res.frames = frames
                res.elapsed = time.time() - t0
                return res

        res.frames = frames
        res.elapsed = time.time() - t0
        res.votes = dict(votes)
        if res.reason == "cancelled":
            return res
        if votes:
            listing = ", ".join(f"'{v}' x{n}" for v, n in votes.most_common())
            res.reason = (f"unstable read - frames disagree ({listing}); "
                          f"need {min_agree} matching frames and a clear lead")
        elif bad:
            v, n = bad.most_common(1)[0]
            res.value = v
            res.reason = f"read '{v}' but {why_bad[v]}"
        elif frames == 0:
            res.reason = ("no new camera frames arrived "
                          "(camera stalled or not started?)")
        else:
            res.reason = (f"no QR code found in the region "
                          f"({frames} frame(s) analysed)")
        return res
