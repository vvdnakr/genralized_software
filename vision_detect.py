"""
vision_detect.py  -  detection helpers shared by the scene builder and the run.

  * method per area:  "normal" (multi-scale template matching, two_point.py)
                      "ai"     (few-shot detector, fewshot_detector.py)
  * nested areas   :  an area can sit INSIDE one found object of another area.
                      Its region then moves with that object, so the inner
                      object is measured relative to the outer one.
  * scene scale    :  each scene can work on a resized picture (0.1x .. 3x).

Old models have none of the new keys and run exactly as before.

Area config.json (new, optional keys):
    "method"      : "normal" | "ai"
    "ai_mode"     : "auto"      auto | features | shape | both | template
    "ai_score"    : 0.50        lowest confidence that counts (0-1)
    "ai_speed"    : "balanced"  accurate | balanced | fast
    "ai_max_angle": 180         largest allowed rotation in degrees (180 = any)
    "ai_params"   : {...}       any other detector setting (All AI Settings)
  <area>/ai_calib/shot_N.png + rois.json : 2-5 calibration shots
  method "aip" (AI image processing, trained from master + template):
    "aip_score", "aip_expect", "aip_max_templates", "aip_augment",
    "aip_variation", "aip_seed", "aip_harvest"      (see AIP_PARAMS)
    <area>/aip/master_N.png + setup.json : the master picture(s) = the region
    <area>/aip/trained.pkl               : the trained matcher (made again
                                           by itself when anything changes,
                                           or when ai_processing.py's model
                                           version changes - AIP_MODEL_VERSION)
    Long thin objects (a pin row, a strip, a slot: at least 2.5x longer than
    thick) are matched in parts by ai_processing.py; their tilt comes back in
    the match "angle" (degrees, counter-clockwise like the "ai" method) and
    "poly" (the tilted box).  "top_left"/"bottom_right" stay the upright
    template-sized box around the object's centre, so pins, the relation
    check and nested areas use exactly the same centre as before.
    "inside"      : {"area": "a1", "object": 1, "anchor": [x, y]}
                    anchor = centre of that object (picture pixels) at the
                    moment this area's region was drawn
Scene file  <scene>/scene.json :  {"image_scale": 1.0}
"""
import json
import math
import os
import threading
import time

import cv2
import numpy as np

from two_point import detect_multi_scale_templates, check_pin

METHOD_NORMAL = "normal"
METHOD_AI = "ai"
METHOD_AIP = "aip"            # AI image processing (ai_processing.py)
SCALES = [0.1, 0.2, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0]
AI_SPEEDS = ["accurate", "balanced", "fast"]

_AI_ERR = None
try:
    from fewshot_detector import Config as _AiConfig, FewShotDetector
except Exception as _e:                              # scipy / file missing
    _AiConfig = FewShotDetector = None
    _AI_ERR = f"{type(_e).__name__}: {_e}"


_AIP_ERR = None
try:
    from ai_processing import AIPatternMatcher
except Exception as _e:
    AIPatternMatcher = None
    _AIP_ERR = f"{type(_e).__name__}: {_e}"


def aip_available():
    """(ok, reason).  AI image processing needs ai_processing.py."""
    if AIPatternMatcher is None:
        return False, ("AI image processing is not installed on this PC "
                       f"({_AIP_ERR}).  Needed: ai_processing.py next to "
                       "the program.")
    return True, ""


def ai_available():
    """(ok, reason).  AI detection needs fewshot_detector.py and scipy."""
    if FewShotDetector is not None and not hasattr(cv2, "HOGDescriptor"):
        return False, ("AI detection needs OpenCV 4 (this PC has OpenCV "
                       f"{cv2.__version__}, which has no HOGDescriptor).  "
                       "Install it with  pip install \"opencv-python<5\"")
    if FewShotDetector is None:
        return False, ("AI detection is not installed on this PC "
                       f"({_AI_ERR}).  Needed: fewshot_detector.py and "
                       "advanced_template.py next to the program, and  "
                       "pip install scipy")
    return True, ""


# ----------------------------------------------------------------------------
# scene picture scale
# ----------------------------------------------------------------------------
def _scene_file(parent_store, scene):
    return os.path.join(parent_store._parent_dir(scene), "scene.json")


def get_scene_scale(parent_store, scene):
    try:
        with open(_scene_file(parent_store, scene)) as f:
            s = float(json.load(f).get("image_scale", 1.0))
        return s if 0.05 <= s <= 4.0 else 1.0
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return 1.0


def set_scene_scale(parent_store, scene, scale):
    path = _scene_file(parent_store, scene)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    data = {}
    try:
        with open(path) as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        pass
    data["image_scale"] = float(scale)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def scale_frame(frame, s):
    """Resize keeping the aspect ratio.  s == 1 returns the same picture."""
    if frame is None or abs(float(s) - 1.0) < 1e-6:
        return frame
    h, w = frame.shape[:2]
    nw, nh = max(8, int(round(w * s))), max(8, int(round(h * s)))
    return cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_AREA
                      if s < 1 else cv2.INTER_LINEAR)


# ----------------------------------------------------------------------------
# AI detection (fewshot_detector.py)
# ----------------------------------------------------------------------------
AI_MODES = [("auto", "Automatic"), ("features", "Keypoints (textured parts)"),
            ("shape", "Outline + colour (plain parts)"),
            ("both", "Keypoints and outline"),
            ("template", "Advanced template (fixed background)")]
AI_CHOICES = {"mode": [m for m, _ in AI_MODES],
              "features": ["sift_fast", "sift"]}
# set from the area form, so not offered again in "All AI Settings"
AI_FORM_KEYS = {"mode", "min_score", "max_instances", "shape_scale_range",
                "template_scale_range", "shape_angle_range",
                "template_angle_range", "min_scale", "max_scale"}
MAX_SHOTS = 5

_ai_cache = {}
_ai_lock = threading.Lock()          # the detector is not re-entrant


def ai_fields():
    """Every detector setting: [(name, default, comment)] in file order."""
    if _AiConfig is None:
        return []
    import dataclasses
    import inspect
    import re
    notes = {}
    try:
        for line in inspect.getsource(_AiConfig).splitlines():
            m = re.match(r"\s+(\w+)\s*:\s*[^=]+=\s*[^#]*#\s*(.+)$", line)
            if m:
                notes[m.group(1)] = m.group(2).strip()
    except (OSError, TypeError):
        pass
    return [(f.name, f.default, notes.get(f.name, ""))
            for f in dataclasses.fields(_AiConfig)]


def ai_parse(name, text, default):
    """Text typed by the user -> value of the same kind as `default`.
    Raises ValueError when it does not fit."""
    if isinstance(default, bool):
        return bool(text) if isinstance(text, bool) else \
            str(text).strip().lower() in ("1", "true", "yes", "on")
    if isinstance(default, int):
        return int(float(text))
    if isinstance(default, float):
        return float(text)
    if isinstance(default, tuple):
        parts = [p for p in str(text).replace("(", "").replace(")", "")
                 .replace(";", ",").split(",") if p.strip()]
        kind = int if all(isinstance(v, int) for v in default) else float
        vals = tuple(kind(float(p)) for p in parts)
        if not vals:
            raise ValueError("empty")
        return vals
    text = str(text).strip()
    if name in AI_CHOICES and text not in AI_CHOICES[name]:
        raise ValueError(text)
    return text


def ai_overrides(cfg):
    """Area settings -> keyword arguments for the detector Config."""
    lo, hi = [float(v) for v in cfg.get("scale_range", (0.5, 2.0))]
    over = dict(mode=str(cfg.get("ai_mode", "auto")),
                min_score=float(cfg.get("ai_score", 0.5)),
                max_instances=max(1, int(cfg.get("max_matches", 20))))
    if 0 < lo < hi:
        over["shape_scale_range"] = (lo, hi)
        over["template_scale_range"] = (lo, hi)
    ang = float(cfg.get("ai_max_angle", 180))
    if ang < 179.5:
        over["shape_angle_range"] = (-ang, ang)
        over["template_angle_range"] = (-ang, ang)
    defaults = {n: d for n, d, _ in ai_fields()}
    for k, v in (cfg.get("ai_params") or {}).items():
        if k in defaults and k not in AI_FORM_KEYS:
            try:
                over[k] = ai_parse(k, v if not isinstance(v, (list, tuple))
                                   else ",".join(str(x) for x in v),
                                   defaults[k])
            except (TypeError, ValueError):
                pass
    return over


def _tmpl_sig(templates):
    return tuple((t.shape, int(t[::4, ::4].sum())) for t in templates)


# -- calibration shots (2-5 pictures of the same object, each with its box) --
def calib_dir(area_dir):
    return os.path.join(area_dir, "ai_calib")


def save_calib(area_dir, shots):
    """shots = [(picture, (x, y, w, h))].  Replaces what was saved."""
    d = calib_dir(area_dir)
    if os.path.isdir(d):
        for f in os.listdir(d):
            try:
                os.remove(os.path.join(d, f))
            except OSError:
                pass
    if not shots:
        return
    os.makedirs(d, exist_ok=True)
    rois = []
    for i, (img, roi) in enumerate(shots[:MAX_SHOTS]):
        cv2.imwrite(os.path.join(d, f"shot_{i}.png"), img)
        rois.append([int(v) for v in roi])
    with open(os.path.join(d, "rois.json"), "w") as f:
        json.dump(rois, f)


def load_calib(area_dir):
    """[(picture, (x, y, w, h))] or []."""
    d = calib_dir(area_dir)
    try:
        with open(os.path.join(d, "rois.json")) as f:
            rois = json.load(f)
    except (OSError, json.JSONDecodeError):
        return []
    out = []
    for i, roi in enumerate(rois):
        img = cv2.imread(os.path.join(d, f"shot_{i}.png"))
        if img is not None and len(roi) == 4:
            out.append((img, tuple(int(v) for v in roi)))
    return out


def _ai_detectors(key, templates, speed, over, shots):
    """Build (or reuse) the detectors of one area.  Returns (detectors,
    calibration text).  With 2+ calibration shots ONE calibrated detector is
    built; otherwise one detector per 3 templates."""
    sig = (_tmpl_sig(templates), speed,
           json.dumps(over, sort_keys=True, default=str),
           tuple((im.shape, int(im[::8, ::8].sum()), tuple(r))
                 for im, r in shots))
    hit = _ai_cache.get(key)
    if hit and hit[0] == sig:
        return hit[1], hit[2]
    dets, note = [], ""
    if len(shots) >= 2:
        d = FewShotDetector(_AiConfig.preset(speed, **over))
        d.calibrate([im for im, _ in shots], [r for _, r in shots],
                    verbose=False)
        # calibration learns its own accept limits; the user's score and
        # count still apply
        d.cfg.max_instances = over["max_instances"]
        try:
            note = d.calibration_summary()
        except Exception:               # template mode has its own report
            r = getattr(d, "calibration", None) or {}
            note = "calibrated from %d images" % len(shots) + "".join(
                f"\n  {k}: {v}" for k, v in r.items()
                if isinstance(v, (int, float, str, tuple, list)))
        dets.append(d)
    else:
        for i in range(0, len(templates), 3):
            d = FewShotDetector(_AiConfig.preset(speed, **over))
            for t in templates[i:i + 3]:
                d.add_reference(t)
            dets.append(d)
    if len(_ai_cache) > 200:
        _ai_cache.clear()
    _ai_cache[key] = (sig, dets, note)
    return dets, note


def ai_note(key):
    """Calibration text of the detector last built for `key` ('' if none)."""
    hit = _ai_cache.get(key)
    return hit[2] if hit else ""


def _iou(a, b):
    ax1, ay1 = a["top_left"]; ax2, ay2 = a["bottom_right"]
    bx1, by1 = b["top_left"]; bx2, by2 = b["bottom_right"]
    iw = min(ax2, bx2) - max(ax1, bx1)
    ih = min(ay2, by2) - max(ay1, by1)
    if iw <= 0 or ih <= 0:
        return 0.0
    inter = iw * ih
    return inter / float((ax2 - ax1) * (ay2 - ay1)
                         + (bx2 - bx1) * (by2 - by1) - inter)


def detect_ai(image, templates, cfg, key=None, config=None, name=None,
              shots=None):
    """Same return shape as detect_multi_scale_templates:
    (matches, output, mask).  Each match also has 'angle' and 'poly'.
    cfg = the area settings; shots = calibration shots or None."""
    ok, why = ai_available()
    if not ok:
        raise RuntimeError(why)
    templates = [t for t in (templates if isinstance(templates, list)
                             else [templates]) if t is not None]
    speed = str(cfg.get("ai_speed", "balanced"))
    speed = speed if speed in AI_SPEEDS else "balanced"
    over = ai_overrides(cfg)
    max_matches = over["max_instances"]
    max_angle = float(cfg.get("ai_max_angle", 180))
    lo, hi = [float(v) for v in cfg.get("scale_range", (0.5, 2.0))]
    with _ai_lock:
        try:
            dets, _ = _ai_detectors(key or id(templates), templates, speed,
                                    over, list(shots or []))
        except (ValueError, TypeError) as e:
            raise RuntimeError(f"AI settings are not valid: {e}")
        found = []
        for d in dets:
            found += d.detect(image)
    matches = []
    for d in found:
        ang = ((float(d.angle) + 180.0) % 360.0) - 180.0
        if abs(ang) > max_angle + 0.5:
            continue
        if 0 < lo < hi and not (lo * 0.9 <= float(d.scale) <= hi * 1.1):
            continue
        x, y, w, h = d.bbox
        matches.append({"score": float(d.score), "scale": float(d.scale),
                        "top_left": (int(x), int(y)),
                        "bottom_right": (int(x + w), int(y + h)),
                        "angle": ang,
                        "poly": [[float(p[0]), float(p[1])]
                                 for p in d.box_points]})
    matches.sort(key=lambda m: m["score"], reverse=True)
    kept = []
    for m in matches:                       # overlap between detector groups
        if all(_iou(m, k) < 0.3 for k in kept):
            kept.append(m)
    matches = kept[:int(max_matches)]
    t_h = max(8, templates[0].shape[0]) if templates else 16
    row = max(1, t_h // 2)
    matches.sort(key=lambda m: (round((m["top_left"][1]
                                       + m["bottom_right"][1]) / 2 / row),
                                m["top_left"][0]))
    output = image.copy()
    for i, m in enumerate(matches):
        pts = np.int32(np.round(m["poly"])).reshape(-1, 1, 2)
        cv2.polylines(output, [pts], True, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(output, str(i + 1),
                    (m["top_left"][0], max(12, m["top_left"][1] - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    mask = None
    if matches and config is not None:
        try:
            mask = check_pin(matches, config, name=name)
        except Exception:
            mask = None
    return matches, output, mask


# ----------------------------------------------------------------------------
# AI image processing (ai_processing.py): MASTER picture(s) + TEMPLATE(s),
# trained.  master = the area's region, template = the object inside it.
# ----------------------------------------------------------------------------
MAX_MASTERS = 6
# Raise this whenever ai_processing.py changes what a trained matcher holds.
# It is part of the training signature, so a trained.pkl made by an older
# ai_processing.py is never loaded into the new code (it would fail or give
# wrong scores) - that area simply trains once again and stores the new file.
#   1 = first version
#   2 = long thin objects matched in parts, tilt, working scale (f), and a
#       patch descriptor that keeps the object's proportions
AIP_MODEL_VERSION = 2
# (key in the area config, default, label, help)
AIP_PARAMS = [
    ("aip_score", 0.40, "Score (0-1)",
     "Lowest score that counts.  0 = the strongest background place seen "
     "in training, 1 = a typical object.  Raise it if something else is "
     "found; lower it if good parts are missed.  No new training needed."),
    ("aip_expect", 0, "Objects expected",
     "How many objects are in the area.  0 = not fixed.  With a number, "
     "exactly that many (the best ones) are reported."),
    ("aip_max_templates", 4, "Templates used",
     "Your templates plus objects found in the masters that are used as "
     "extra templates.  More = finds more looks, but slower."),
    ("aip_augment", 16, "Training copies",
     "Copies of each master made with other light, tone and small moves.  "
     "More = more careful training, but slower."),
    ("aip_variation", 1.0, "Variation",
     "How different the made-up 'other parts' are in training.  Raise it "
     "when real parts differ a lot from the master."),
    ("aip_seed", 0, "Seed",
     "Start number of the random copies.  The same number gives the same "
     "training again."),
    ("aip_harvest", True, "Use found objects",
     "Use the objects found in the masters as extra templates (one box "
     "on one pin then learns all pins)."),
]
_AIP_TRAIN_KEYS = [k for k, *_ in AIP_PARAMS if k != "aip_score"]

_aip_cache = {}                    # key -> (signature, matcher, report)
_aip_lock = threading.Lock()


def aip_params(cfg):
    """Area settings -> the values of AIP_PARAMS (with defaults)."""
    out = {}
    for k, default, *_ in AIP_PARAMS:
        v = cfg.get(k, default)
        try:
            out[k] = (bool(v) if isinstance(default, bool) else
                      int(float(v)) if isinstance(default, int) else float(v))
        except (TypeError, ValueError):
            out[k] = default
    return out


def _img_sig(imgs):
    return tuple((im.shape, int(im[::4, ::4].sum(dtype=np.int64)))
                 for im in imgs)


def aip_signature(masters, templates, cfg):
    """Changes whenever a new training is needed (also when the trained
    model format of ai_processing.py changes: AIP_MODEL_VERSION)."""
    p = aip_params(cfg)
    return repr((AIP_MODEL_VERSION, _img_sig(masters), _img_sig(templates),
                 [p[k] for k in _AIP_TRAIN_KEYS]))


def aip_dir(area_dir):
    return os.path.join(area_dir, "aip")


def save_masters(area_dir, masters):
    """Replace the stored master pictures (and drop the old trained file)."""
    d = aip_dir(area_dir)
    if os.path.isdir(d):
        for f in os.listdir(d):
            try:
                os.remove(os.path.join(d, f))
            except OSError:
                pass
    if not masters:
        return
    os.makedirs(d, exist_ok=True)
    for i, m in enumerate(masters[:MAX_MASTERS]):
        cv2.imwrite(os.path.join(d, f"master_{i}.png"), m)
    with open(os.path.join(d, "setup.json"), "w") as f:
        json.dump({"masters": [f"master_{i}.png"
                               for i in range(min(len(masters),
                                                  MAX_MASTERS))]}, f)


def load_masters(area_dir):
    d = aip_dir(area_dir)
    try:
        with open(os.path.join(d, "setup.json")) as f:
            names = json.load(f).get("masters", [])
    except (OSError, json.JSONDecodeError):
        return []
    out = []
    for n in names:
        im = cv2.imread(os.path.join(d, n))
        if im is not None:
            out.append(im)
    return out


def _aip_model_file(area_dir):
    return os.path.join(aip_dir(area_dir), "trained.pkl")


def aip_store_model(area_dir, key):
    """Write the trained matcher of `key` next to the masters, so the next
    program start does not have to train again."""
    import pickle
    hit = _aip_cache.get(key)
    if not hit:
        return False
    try:
        os.makedirs(aip_dir(area_dir), exist_ok=True)
        with open(_aip_model_file(area_dir), "wb") as f:
            pickle.dump({"sig": hit[0], "matcher": hit[1],
                         "report": hit[2]}, f)
        return True
    except Exception:
        return False


def aip_load(key, area_dir, masters, templates, cfg):
    """Put the area's stored training into memory under `key` when it still
    fits these masters + templates + settings.  Returns the report or None.
    (Never trains.)"""
    import pickle
    if not aip_available()[0] or not masters or not templates:
        return None
    sig = aip_signature(masters, templates, cfg)
    try:
        with open(_aip_model_file(area_dir), "rb") as f:
            d = pickle.load(f)
    except Exception:
        return None
    if d.get("sig") != sig:
        return None
    with _aip_lock:
        _aip_cache[key] = (sig, d["matcher"], d.get("report") or {})
    return d.get("report") or {}


def aip_get(key, masters, templates, cfg, area_dir=None, on_train=None):
    """The trained matcher for these masters + templates + settings.
    Order: memory -> trained file of the area -> train now (3-20 s).
    Returns (matcher, report).  Raises RuntimeError with a plain reason."""
    ok, why = aip_available()
    if not ok:
        raise RuntimeError(why)
    masters = [m for m in masters if m is not None]
    templates = [t for t in templates if t is not None]
    if not masters:
        raise RuntimeError("AI image processing has no master picture - "
                           "open the area and add the template again")
    if not templates:
        raise RuntimeError("AI image processing has no template")
    for t in templates:
        if not any(t.shape[0] <= m.shape[0] and t.shape[1] <= m.shape[1]
                   for m in masters):
            raise RuntimeError("a template is bigger than the master area")
    sig = aip_signature(masters, templates, cfg)
    p = aip_params(cfg)
    with _aip_lock:
        hit = _aip_cache.get(key)
        if not (hit and hit[0] == sig) and area_dir:
            import pickle
            try:
                with open(_aip_model_file(area_dir), "rb") as f:
                    d = pickle.load(f)
                if d.get("sig") == sig:
                    hit = (sig, d["matcher"], d.get("report") or {})
                    _aip_cache[key] = hit
            except Exception:
                pass
        if not (hit and hit[0] == sig):
            if on_train:
                on_train()
            t0 = time.time()
            m = AIPatternMatcher(
                min_score=p["aip_score"], augment=max(4, p["aip_augment"]),
                seed=p["aip_seed"], expect=p["aip_expect"] or None,
                verbose=False, variation=max(0.0, p["aip_variation"]),
                max_templates=max(1, p["aip_max_templates"]),
                harvest=p["aip_harvest"])
            try:
                rep = dict(m.train(masters, templates))
            except Exception as e:
                # most often: the area is hardly bigger than the template,
                # so there is no background to learn from
                raise RuntimeError(
                    "AI training failed. Make the area clearly bigger than "
                    "the object (it needs some background around it) and "
                    "draw the template on the object. "
                    f"({type(e).__name__}: {e})")
            rep["seconds"] = round(time.time() - t0, 1)
            rep["masters"] = len(masters)
            rep["dropped"] = list(getattr(m, "dropped", []))
            rep["score_weakest"] = round(float(m.score_weakest), 2)
            rep["joint_separable"] = bool(m.joint_separable)
            rep["long_object"] = bool(getattr(m, "long", False))
            if len(_aip_cache) > 100:
                _aip_cache.clear()
            hit = (sig, m, rep)
            _aip_cache[key] = hit
            if area_dir and os.path.isdir(aip_dir(area_dir)):
                aip_store_model(area_dir, key)
        hit[1].min_score = p["aip_score"]        # no training needed
        return hit[1], hit[2]


def aip_report_text(rep):
    """What training learned, in plain words (several lines)."""
    if not rep:
        return ""
    lines = [f"trained in {rep.get('seconds', '?')} s from "
             f"{rep.get('masters', '?')} master(s) and "
             f"{rep.get('templates_from_user', '?')} template(s)"
             + (f" (+{rep['templates'] - rep['templates_from_user']} taken "
                "from the masters)"
                if rep.get("templates", 0) > rep.get("templates_from_user", 0)
                else ""),
             f"objects found in the master(s): "
             f"{rep.get('instances_per_master')}",
             f"picture channels used: {rep.get('channel_weights')}",
             f"weakest training object scored "
             f"{rep.get('score_weakest', '?')} (1 = typical object, "
             "0 = background)"]
    if rep.get("long_object"):
        ws = rep.get("working_scale")
        lines.append("long thin object: matched in parts along its length, "
                     "centre and tilt measured from the parts"
                     + (f", working at {ws:g}x size" if ws else ""))
    if rep.get("dropped"):
        lines.append("WARNING: template(s) "
                     + ", ".join(str(i + 1) for i in rep["dropped"])
                     + " do not look like the other objects (drawn off the "
                     "object?) - left out")
    if not rep.get("separable", True) or not rep.get("joint_separable", True):
        lines.append("WARNING: an object looks like the background here - "
                     "add another template or master")
    return "\n".join(lines)


def _tilted_box(cx, cy, w, h, ang_img):
    """Corners (top-left, top-right, bottom-right, bottom-left of the
    template) of a w x h box centred at (cx, cy), turned by `ang_img`
    degrees in picture coordinates (y down: positive = clockwise on screen,
    the convention of ai_processing.Match.angle and cv2.boxPoints)."""
    t = math.radians(ang_img)
    c, s = math.cos(t), math.sin(t)
    out = []
    for x, y in ((-w / 2, -h / 2), (w / 2, -h / 2), (w / 2, h / 2),
                 (-w / 2, h / 2)):
        out.append([cx + c * x - s * y, cy + s * x + c * y])
    return out


def detect_aip(image, templates, cfg, key=None, config=None, name=None,
               masters=None, area_dir=None):
    """Same return shape as the other methods: (matches, output, mask).
    A long thin object comes back tilted: 'angle' (deg, counter-clockwise,
    as in the "ai" method) and 'poly' (tilted box).  'top_left' and
    'bottom_right' stay the upright box around its centre."""
    m, _rep = aip_get(key or "aip", list(masters or []), templates, cfg,
                      area_dir=area_dir)
    with _aip_lock:
        try:
            found = m.detect(image)
        except Exception as e:
            raise RuntimeError(f"AI image processing failed: "
                               f"{type(e).__name__}: {e}")
    found = found[:max(1, int(cfg.get("max_matches", 20)))]
    matches = []
    for d in found:
        ang_img = float(getattr(d, "angle", 0.0) or 0.0)
        if abs(ang_img) > 0.3:               # long object: tilted box
            cx, cy = (d.center if getattr(d, "center", None)
                      else (d.x + d.w / 2.0, d.y + d.h / 2.0))
            poly = _tilted_box(float(cx), float(cy), d.w, d.h, ang_img)
        else:
            ang_img = 0.0
            poly = [[d.x, d.y], [d.x + d.w, d.y], [d.x + d.w, d.y + d.h],
                    [d.x, d.y + d.h]]
        matches.append({
            "score": float(d.score), "scale": 1.0,
            "angle": round(-ang_img, 2) + 0.0,     # counter-clockwise
            "top_left": (int(d.x), int(d.y)),
            "bottom_right": (int(d.x + d.w), int(d.y + d.h)),
            "poly": poly,
            "match": float(d.match), "patch": float(d.patch)})
    t_h = max(8, int(np.median([t.shape[0] for t in templates])))
    row = max(1, t_h // 2)
    matches.sort(key=lambda q: (round((q["top_left"][1]
                                       + q["bottom_right"][1]) / 2 / row),
                                q["top_left"][0]))
    output = image.copy()
    for i, q in enumerate(matches):
        if q["angle"]:
            pts = np.int32(np.round(q["poly"])).reshape(-1, 1, 2)
            cv2.polylines(output, [pts], True, (0, 255, 0), 2, cv2.LINE_AA)
        else:
            cv2.rectangle(output, q["top_left"], q["bottom_right"],
                          (0, 255, 0), 2)
        cv2.putText(output, str(i + 1),
                    (q["top_left"][0], max(12, q["top_left"][1] - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    mask = None
    if matches and config is not None:
        try:
            mask = check_pin(matches, config, name=name)
        except Exception:
            mask = None
    return matches, output, mask


def detect_area(image, templates, cfg, config=None, name=None, key=None,
                shots=None, masters=None, area_dir=None):
    """Run the method chosen for this area.  cfg = the area's settings."""
    if str(cfg.get("method", METHOD_NORMAL)) == METHOD_AIP:
        return detect_aip(image, templates, cfg, key=key, config=config,
                          name=name, masters=masters, area_dir=area_dir)
    if str(cfg.get("method", METHOD_NORMAL)) == METHOD_AI:
        return detect_ai(image, templates, cfg, key=key, config=config,
                         name=name, shots=shots)
    res = detect_multi_scale_templates(
        image=image, template=templates,
        scale_range=tuple(cfg["scale_range"]),
        scale_steps=int(cfg["scale_steps"]),
        threshold=float(cfg["threshold"]),
        max_matches=int(cfg["max_matches"]),
        visualize=True, t=time.time(), config=config, name=name)
    if isinstance(res, tuple) and len(res) == 3:
        return res
    return res[0], res[1], None


# ----------------------------------------------------------------------------
# relation check: objects judged against EACH OTHER, not against the picture
# ----------------------------------------------------------------------------
# Area key  "position_check":
#     "fixed"    each object must be at its master place (pin limit)  [default]
#     "relation" the objects must keep their distances to each other
#                ("relation_limit" px); the whole group may move
#     "both"     both must pass
# Scene file  scene.json  "relation": {"enabled", "limit", "ref"}
#     the same check between the objects of DIFFERENT areas, in picture
#     pixels.  ref = {area: [[x, y] | None per master object]}
POS_FIXED, POS_RELATION, POS_BOTH = "fixed", "relation", "both"


def _centers(matches):
    return [((m["top_left"][0] + m["bottom_right"][0]) / 2.0,
             (m["top_left"][1] + m["bottom_right"][1]) / 2.0)
            for m in matches]


def pair_shifted(master, cur, tol):
    """Pair each master point with a current point, allowing ONE common
    shift of the whole group.  Returns ([index into cur or None], (sx, sy)).
    Which object sits where does not matter - only that the place is taken."""
    if not master or not cur:
        return [None] * len(master), (0.0, 0.0)
    M = np.asarray(master, float)
    C = np.asarray(cur, float)

    def assign(shift):
        want = M + shift
        d = np.hypot(want[:, None, 0] - C[None, :, 0],
                     want[:, None, 1] - C[None, :, 1])
        pairs, used = [None] * len(M), set()
        for flat in np.argsort(d, axis=None):
            i, j = divmod(int(flat), len(C))
            if d[i, j] > tol:
                break
            if pairs[i] is None and j not in used:
                pairs[i] = j
                used.add(j)
        got = [(i, j) for i, j in enumerate(pairs) if j is not None]
        cost = sum(d[i, j] for i, j in got)
        return pairs, len(got), cost

    best = None
    for i in range(len(M)):                 # every "this is that one" guess
        for j in range(len(C)):
            sh = C[j] - M[i]
            pairs, n, cost = assign(sh)
            if n >= 1:                      # refine on the pairs it gave
                got = [(a, b) for a, b in enumerate(pairs) if b is not None]
                sh2 = np.median([C[b] - M[a] for a, b in got], axis=0)
                p2, n2, c2 = assign(sh2)
                if (n2, -c2) >= (n, -cost):
                    pairs, n, cost, sh = p2, n2, c2, sh2
            if best is None or (n, -cost) > (best[1], -best[2]):
                best = (pairs, n, cost, sh)
    pairs, _n, _c, sh = best
    return pairs, (float(sh[0]), float(sh[1]))


def _blame(points_ref, points_cur, limit, skip=None):
    """Compare every distance between two objects with its reference.
    Returns (worst error per object, set of objects that are out, pairs
    checked, worst error).  skip(i, j) -> True leaves that pair out."""
    n = len(points_ref)
    err = np.zeros((n, n))
    used = np.zeros((n, n), bool)
    for i in range(n):
        for j in range(i + 1, n):
            if skip is not None and skip(i, j):
                continue
            dr = np.hypot(points_ref[i][0] - points_ref[j][0],
                          points_ref[i][1] - points_ref[j][1])
            dc = np.hypot(points_cur[i][0] - points_cur[j][0],
                          points_cur[i][1] - points_cur[j][1])
            err[i, j] = err[j, i] = abs(dc - dr)
            used[i, j] = used[j, i] = True
    worst = [float(err[i][used[i]].max()) if used[i].any() else 0.0
             for i in range(n)]
    bad = (err > float(limit)) & used
    out, alive = set(), set(range(n))
    while True:                              # the object in the most wrong
        cnt = {i: int(bad[i][list(alive)].sum()) for i in alive}   # pairs
        top = max(cnt.values()) if cnt else 0
        if top == 0:
            break
        tied = [i for i in alive if cnt[i] == top]
        if top == 1 and len(tied) == 2 and bad[tied[0], tied[1]]:
            # one wrong pair between two objects: the one that is further
            # off to ALL the others is the wrong one; if that does not
            # tell them apart, both are marked
            tot = [float(err[q][list(alive)].sum()) for q in tied]
            if max(tot) > 1.5 * min(tot) + 1e-6:
                i = tied[int(np.argmax(tot))]
                out.add(i)
                alive.discard(i)
            else:
                out.update(tied)
                alive.difference_update(tied)
            continue
        i = max(tied, key=lambda q: float(err[q][list(alive)].sum()))
        out.add(i)
        alive.discard(i)
    return worst, out, int(used.sum() // 2), float(err[used].max()
                                                   if used.any() else 0.0)


def analyze_relation(master_matches, matches, limit):
    """Like analyze_pins, but the objects are judged against each other:
    every distance between two objects must be within `limit` px of the
    master distance.  A common shift of all objects is allowed.
    Returns (ok, pins, summary) - pins/summary as analyze_pins gives them;
    pin 'dist' = its worst distance error, 'dx','dy' = offset after the
    common shift is taken out, summary['shift'] = the common shift."""
    master = [tuple(float(v) for v in m["center"])
              for m in (master_matches or [])]
    cur = _centers(matches)
    if not master:
        return (len(matches) > 0, [], {
            "paired": len(matches), "in_range": 0, "total": 0,
            "max_dist": 0.0, "mean_dx": 0.0, "mean_dy": 0.0,
            "relation": True, "shift": (0.0, 0.0), "pairs": 0})
    limit = float(limit)
    tol = max(3.0 * limit, 12.0)
    if len(master) > 1:                      # never past half the spacing
        M = np.asarray(master)
        d = np.hypot(M[:, None, 0] - M[None, :, 0],
                     M[:, None, 1] - M[None, :, 1])
        d[d == 0] = np.inf
        tol = max(1.5 * limit, min(tol, 0.5 * float(d.min())))
    pairs, shift = pair_shifted(master, cur, tol)
    idx = [i for i, j in enumerate(pairs) if j is not None]
    worst, out, npairs, max_err = _blame(
        [master[i] for i in idx], [cur[pairs[i]] for i in idx], limit)
    pins = []
    for i, mc in enumerate(master):
        j = pairs[i]
        if j is None:
            pins.append({"master": mc, "current": None, "dist": None,
                         "dx": None, "dy": None, "paired": False,
                         "in_range": False})
            continue
        k = idx.index(i)
        cc = cur[j]
        pins.append({"master": mc, "current": cc, "dist": worst[k],
                     "dx": cc[0] - mc[0] - shift[0],
                     "dy": cc[1] - mc[1] - shift[1],
                     "paired": True, "in_range": k not in out})
    n_in = sum(1 for p in pins if p["in_range"])
    return n_in == len(pins), pins, {
        "paired": len(idx), "in_range": n_in, "total": len(pins),
        "max_dist": max_err, "mean_dx": shift[0], "mean_dy": shift[1],
        "relation": True, "shift": shift, "pairs": npairs}


def get_scene_relation(parent_store, scene):
    """{"enabled": bool, "limit": px, "ref": {area: [[x, y] | None]}}"""
    out = {"enabled": False, "limit": 8.0, "ref": {}}
    try:
        with open(_scene_file(parent_store, scene)) as f:
            d = json.load(f).get("relation") or {}
        out["enabled"] = bool(d.get("enabled", False))
        out["limit"] = float(d.get("limit", 8.0))
        out["ref"] = dict(d.get("ref") or {})
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        pass
    return out


def set_scene_relation(parent_store, scene, **changes):
    path = _scene_file(parent_store, scene)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    data = {}
    try:
        with open(path) as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        pass
    rel = dict(data.get("relation") or {})
    rel.update(changes)
    data["relation"] = rel
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def relation_reference(found):
    """Reference for the between-areas check from one finished scene run:
    {area: [[x, y] per master object]}.  Raises RuntimeError (plain words)
    when an object is missing - the reference needs a complete, good part."""
    ref = {}
    for name, r in found.items():
        if r.get("skipped"):
            raise RuntimeError(f"area '{name}': {r['skipped']}")
        pts = []
        n = max(len(r.get("pins") or []), 1)
        for k in range(1, n + 1):
            c = object_center(r, k)
            if c is None:
                raise RuntimeError(f"area '{name}': object {k} was not "
                                   "found")
            pts.append([round(c[0], 2), round(c[1], 2)])
        ref[name] = pts
    return ref


def scene_relation(found, ref, limit):
    """Between-areas check on a finished scene run, in picture pixels: the
    distance between objects of DIFFERENT areas must stay within `limit` of
    the reference.  Marks the areas that are out (ok = False, 'scene_rel'
    text) and returns a summary dict."""
    items = []                        # (area, object no, ref xy, current xy)
    no_ref = []
    for name, r in found.items():
        if r.get("skipped"):
            continue
        pts = ref.get(name)
        if not pts:
            no_ref.append(name)
            continue
        pins = r.get("pins") or []
        for k, rp in enumerate(pts, 1):
            c = object_center(r, k)
            # an object already NG inside its own area is left out, so it
            # cannot put the blame on the other areas
            if k <= len(pins) and not pins[k - 1]["in_range"]:
                continue
            if rp is not None and c is not None:
                items.append((name, k, (float(rp[0]), float(rp[1])), c))
    summ = {"objects": len(items), "pairs": 0, "max_err": 0.0,
            "limit": float(limit), "out": [], "no_ref": no_ref,
            "areas": sorted({it[0] for it in items})}
    if len(summ["areas"]) < 2:
        return summ
    worst, out, npairs, max_err = _blame(
        [it[2] for it in items], [it[3] for it in items], limit,
        skip=lambda i, j: items[i][0] == items[j][0])
    summ.update(pairs=npairs, max_err=max_err,
                out=[(items[i][0], items[i][1], round(worst[i], 1))
                     for i in sorted(out)])
    for name, k, e in summ["out"]:
        r = found[name]
        r["ok"] = False
        r["scene_rel"] = (r.get("scene_rel", "") + f" #{k}:{e:.0f}px").strip()
    return summ


# ----------------------------------------------------------------------------
# nested areas
# ----------------------------------------------------------------------------
def inside_of(cfg):
    """(parent area, object number 1.., anchor) or None."""
    ins = cfg.get("inside") or None
    if not isinstance(ins, dict) or not ins.get("area"):
        return None
    try:
        obj = max(1, int(ins.get("object", 1)))
        anchor = ins.get("anchor")
        anchor = (float(anchor[0]), float(anchor[1])) if anchor else None
    except (TypeError, ValueError, IndexError):
        return None
    return str(ins["area"]), obj, anchor


def order_areas(children_data):
    """Parents before the areas inside them.  Returns (ordered names,
    {name: problem}) - a problem is a missing parent or a loop."""
    cfgs = {n: d["config"] for n, d in children_data}
    done, order, bad = set(), [], {}

    def visit(n, trail):
        if n in done or n in bad:
            return
        ins = inside_of(cfgs[n])
        if ins:
            p = ins[0]
            if p not in cfgs:
                bad[n] = f"the area '{p}' it sits in does not exist"
                return
            if p in trail:
                bad[n] = "areas are inside each other in a loop"
                return
            visit(p, trail | {n})
            if p in bad:
                bad[n] = f"the area '{p}' it sits in has a problem"
                return
        done.add(n)
        order.append(n)

    for n, _ in children_data:
        visit(n, frozenset())
    return order, bad


def descendants(children_data, name):
    """Names of every area that sits (directly or not) inside `name`."""
    cfgs = {n: d["config"] for n, d in children_data}
    out, grew = set(), True
    while grew:
        grew = False
        for n, c in cfgs.items():
            ins = inside_of(c)
            if n not in out and ins and (ins[0] == name or ins[0] in out):
                out.add(n)
                grew = True
    return out


def object_center(res, obj):
    """Centre (picture px) of object number `obj` of a finished area result,
    or None when that object was not found."""
    if res is None or res.get("skipped"):
        return None
    ox, oy = res["origin"]
    pins = res.get("pins") or []
    if pins:
        if obj > len(pins) or not pins[obj - 1]["paired"]:
            return None
        c = pins[obj - 1]["current"]
        return ox + float(c[0]), oy + float(c[1])
    ms = res.get("matches") or []
    if obj > len(ms):
        return None
    (x1, y1), (x2, y2) = ms[obj - 1]["top_left"], ms[obj - 1]["bottom_right"]
    return ox + (x1 + x2) / 2.0, oy + (y1 + y2) / 2.0


_shots_cache = {}


def _shots_for(area_dir):
    """Calibration shots of an area, re-read only when the file changes."""
    if not area_dir:
        return []
    f = os.path.join(calib_dir(area_dir), "rois.json")
    try:
        mt = os.path.getmtime(f)
    except OSError:
        return []
    hit = _shots_cache.get(area_dir)
    if hit and hit[0] == mt:
        return hit[1]
    shots = load_calib(area_dir)
    _shots_cache[area_dir] = (mt, shots)
    return shots


_masters_cache = {}


def _masters_for(area_dir):
    """Master pictures of an area, re-read only when the file changes."""
    if not area_dir:
        return []
    f = os.path.join(aip_dir(area_dir), "setup.json")
    try:
        mt = os.path.getmtime(f)
    except OSError:
        return []
    hit = _masters_cache.get(area_dir)
    if hit and hit[0] == mt:
        return hit[1]
    ms = load_masters(area_dir)
    _masters_cache[area_dir] = (mt, ms)
    return ms


def run_areas(frame, children_data, dist_thr, analyze_pins, only=None,
              key_prefix="", stop=None, area_dir=None):
    """Detect every area of one scene on `frame` (already scene-scaled).

    children_data : [(name, loaded area)]  as ParentModelManager gives it
    only          : set of names to report (their parents still run)
    stop          : callable -> True to give up early (run cancelled)
    Returns {name: result} in run order.  result keys:
        skipped (text or None), roi (x0,y0,x1,y1 clipped), origin (x,y of the
        un-clipped region), matches, output, mask, ok, pins, summ, area_thr,
        shift (dx,dy of the parent object), inside, method
    """
    fh, fw = frame.shape[:2]
    data = dict(children_data)
    order, bad = order_areas(children_data)
    need = None
    if only is not None:
        need = set(only)
        for n in list(need):
            c = data.get(n)
            while c is not None:
                ins = inside_of(c["config"])
                if not ins or ins[0] in need:
                    break
                need.add(ins[0])
                c = data.get(ins[0])
    out = {}
    for n, _ in children_data:
        if n in bad and (need is None or n in need):
            cfg = data[n]["config"]
            out[n] = {"skipped": bad[n], "ok": False, "matches": [],
                      "pins": [], "inside": inside_of(cfg),
                      "master_n": len(cfg.get("matches", [])),
                      "method": cfg.get("method", METHOD_NORMAL)}
    for n in order:
        if need is not None and n not in need:
            continue
        if stop is not None and stop():
            break
        cd = data[n]
        cfg = cd["config"]
        ins = inside_of(cfg)
        res = {"skipped": None, "inside": ins, "shift": (0.0, 0.0),
               "method": cfg.get("method", METHOD_NORMAL),
               "master_n": len(cfg.get("matches", [])), "matches": [],
               "pins": [], "ok": False}
        out[n] = res
        x, y, w, h = [int(v) for v in cd["reg"]]
        if ins:
            pc = object_center(out.get(ins[0]), ins[1])
            if pc is None:
                res["skipped"] = (f"object {ins[1]} of '{ins[0]}' was not "
                                  "found")
                continue
            if ins[2] is not None:
                res["shift"] = (pc[0] - ins[2][0], pc[1] - ins[2][1])
                x += int(round(res["shift"][0]))
                y += int(round(res["shift"][1]))
        x0, y0 = max(0, min(x, fw)), max(0, min(y, fh))
        x1, y1 = max(0, min(x + w, fw)), max(0, min(y + h, fh))
        res["origin"] = (x, y)
        res["roi"] = (x0, y0, x1, y1)
        if x1 <= x0 or y1 <= y0:
            res["skipped"] = "region is outside the picture"
            continue
        region = frame[y0:y1, x0:x1]
        try:
            shots = masters = adir = None
            meth = cfg.get("method", METHOD_NORMAL)
            if area_dir is not None and meth == METHOD_AI:
                shots = _shots_for(area_dir(n))
            if area_dir is not None and meth == METHOD_AIP:
                adir = area_dir(n)
                masters = _masters_for(adir)
            matches, output, mask = detect_area(
                region, cd["templates"], cfg, config=cd, name=n,
                key=f"{key_prefix}/{n}", shots=shots, masters=masters,
                area_dir=adir)
        except RuntimeError as e:        # e.g. AI detection not installed
            res["skipped"] = str(e)
            continue
        if (x0, y0) != (x, y):          # region cut by the picture edge:
            for m in matches:           # keep positions in region coordinates
                dx, dy = x0 - x, y0 - y
                m["top_left"] = (m["top_left"][0] + dx, m["top_left"][1] + dy)
                m["bottom_right"] = (m["bottom_right"][0] + dx,
                                     m["bottom_right"][1] + dy)
        try:
            area_thr = float(cfg.get("pin_limit") or dist_thr)
        except (TypeError, ValueError):
            area_thr = float(dist_thr)
        ok, pins, summ = analyze_pins(cfg.get("matches", []), matches,
                                      area_thr)
        mode = str(cfg.get("position_check", POS_FIXED))
        if mode in (POS_RELATION, POS_BOTH) and cfg.get("matches"):
            try:
                rel_thr = float(cfg.get("relation_limit") or area_thr)
            except (TypeError, ValueError):
                rel_thr = area_thr
            rok, rpins, rsumm = analyze_relation(cfg["matches"], matches,
                                                 rel_thr)
            rsumm["limit"] = rel_thr
            if mode == POS_RELATION:
                ok, pins, summ = rok, rpins, rsumm
            else:                      # both: fixed place AND relation
                for p, rp in zip(pins, rpins):
                    p["in_range"] = p["in_range"] and rp["in_range"]
                    p["rel_err"] = rp["dist"]
                summ["in_range"] = sum(1 for p in pins if p["in_range"])
                summ["rel"] = rsumm
                ok = ok and rok
        res.update(matches=matches, output=output, mask=mask, ok=ok,
                   pins=pins, summ=summ, area_thr=area_thr,
                   position_check=mode)
    return out


def warm_area(cd, key, area_dir):
    """Get the AI part of one loaded area ready ahead of the first run (a
    calibrated detector or a training needs seconds).  True when done."""
    cfg = cd["config"]
    meth = cfg.get("method", METHOD_NORMAL)
    tmpl = [t for t in cd["templates"] if t is not None]
    if meth == METHOD_AIP:
        if not aip_available()[0]:
            return False
        aip_get(key, _masters_for(area_dir), tmpl, cfg, area_dir=area_dir)
        return True
    if meth != METHOD_AI or not ai_available()[0]:
        return False
    speed = str(cfg.get("ai_speed", "balanced"))
    with _ai_lock:
        _ai_detectors(key, tmpl,
                      speed if speed in AI_SPEEDS else "balanced",
                      ai_overrides(cfg), _shots_for(area_dir))
    return True
