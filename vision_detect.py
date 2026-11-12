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
    "ai_score"    : 0.30        lowest confidence that counts (0-1)
    "ai_speed"    : "balanced"  accurate | balanced | fast
    "ai_max_angle": 180         largest allowed rotation in degrees (180 = any)
    "inside"      : {"area": "a1", "object": 1, "anchor": [x, y]}
                    anchor = centre of that object (picture pixels) at the
                    moment this area's region was drawn
Scene file  <scene>/scene.json :  {"image_scale": 1.0}
"""
import json
import os
import threading
import time

import cv2
import numpy as np

from two_point import detect_multi_scale_templates, check_pin

METHOD_NORMAL = "normal"
METHOD_AI = "ai"
SCALES = [0.1, 0.2, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0]
AI_SPEEDS = ["accurate", "balanced", "fast"]

_AI_ERR = None
try:
    from fewshot_detector import Config as _AiConfig, FewShotDetector
except Exception as _e:                              # scipy / file missing
    _AiConfig = FewShotDetector = None
    _AI_ERR = f"{type(_e).__name__}: {_e}"


def ai_available():
    """(ok, reason).  AI detection needs fewshot_detector.py and scipy."""
    if FewShotDetector is None:
        return False, ("AI detection is not installed on this PC "
                       f"({_AI_ERR}).  Needed: fewshot_detector.py next to "
                       "the program and  pip install scipy")
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
# AI detection
# ----------------------------------------------------------------------------
_ai_cache = {}
_ai_lock = threading.Lock()          # the detector is not re-entrant


def _tmpl_sig(templates):
    return tuple((t.shape, int(t[::4, ::4].sum())) for t in templates)


def _ai_detectors(key, templates, speed, score, max_n, max_angle,
                  scale_range):
    """One detector per 3 templates (the detector takes up to 3 views)."""
    sig = (_tmpl_sig(templates), speed, round(score, 3), int(max_n),
           round(max_angle, 1), tuple(round(float(v), 3)
                                      for v in scale_range))
    hit = _ai_cache.get(key)
    if hit and hit[0] == sig:
        return hit[1]
    over = dict(min_score=float(score), max_instances=int(max(1, max_n)))
    if max_angle < 179.5:
        over["shape_angle_range"] = (-float(max_angle), float(max_angle))
    lo, hi = float(scale_range[0]), float(scale_range[1])
    if 0 < lo < hi:
        over["shape_scale_range"] = (max(0.2, lo * 0.9), min(4.0, hi * 1.1))
    dets = []
    for i in range(0, len(templates), 3):
        d = FewShotDetector(_AiConfig.preset(speed, **over))
        for t in templates[i:i + 3]:
            d.add_reference(t)
        dets.append(d)
    if len(_ai_cache) > 200:
        _ai_cache.clear()
    _ai_cache[key] = (sig, dets)
    return dets


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


def detect_ai(image, templates, score=0.30, speed="balanced", max_matches=20,
              max_angle=180.0, scale_range=(0.5, 2.0), key=None, config=None,
              name=None):
    """Same return shape as detect_multi_scale_templates:
    (matches, output, mask).  Each match also has 'angle' and 'poly'."""
    ok, why = ai_available()
    if not ok:
        raise RuntimeError(why)
    templates = [t for t in (templates if isinstance(templates, list)
                             else [templates]) if t is not None]
    speed = speed if speed in AI_SPEEDS else "balanced"
    lo, hi = float(scale_range[0]), float(scale_range[1])
    with _ai_lock:
        dets = _ai_detectors(key or id(templates), templates, speed,
                             float(score), int(max_matches),
                             float(max_angle), (lo, hi))
        found = []
        for d in dets:
            found += d.detect(image)
    matches = []
    for d in found:
        ang = ((float(d.angle) + 180.0) % 360.0) - 180.0
        if abs(ang) > float(max_angle) + 0.5:
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


def detect_area(image, templates, cfg, config=None, name=None, key=None):
    """Run the method chosen for this area.  cfg = the area's settings."""
    if str(cfg.get("method", METHOD_NORMAL)) == METHOD_AI:
        return detect_ai(image, templates,
                         score=float(cfg.get("ai_score", 0.30)),
                         speed=str(cfg.get("ai_speed", "balanced")),
                         max_matches=int(cfg.get("max_matches", 20)),
                         max_angle=float(cfg.get("ai_max_angle", 180)),
                         scale_range=tuple(cfg.get("scale_range",
                                                   (0.5, 2.0))),
                         key=key, config=config, name=name)
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


def run_areas(frame, children_data, dist_thr, analyze_pins, only=None,
              key_prefix="", stop=None):
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
            matches, output, mask = detect_area(
                region, cd["templates"], cfg, config=cd, name=n,
                key=f"{key_prefix}/{n}")
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
        res.update(matches=matches, output=output, mask=mask, ok=ok,
                   pins=pins, summ=summ, area_thr=area_thr)
    return out
