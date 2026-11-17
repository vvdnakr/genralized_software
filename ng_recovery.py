"""
ng_recovery.py  -  automatic NG recheck + authorised Master / Template
addition (the logic part; the screens are in ui_pyqt6_improved.py).

Nothing here touches Qt, so it can be tested and used without a screen.

  * per-model settings  (models/heads/<model>/detection_settings.json,
                         key "ng_recovery"):
        auto_recheck     True   an NG run is inspected once more on a new
                                picture; only the second result counts
        popup_enabled    False  NG twice -> offer the Master/Template popup
                                (changed only in MASTER mode)
        popup_timeout_s  60     the first popup reports NG by itself after
                                this many seconds without an answer
                                (0 = wait for the operator)
  * authorisation       (models/app_config.yaml, section "ng_recovery"):
        password         "89012"
        min_id_length    3      shortest operator ID accepted
        allowed_ids      []     empty = any ID; otherwise only these IDs
        scan_timeout_s   30     the ID scan box gives up after this
  * what can be added, by the technique of the area:
        aip    (AI Image Processing)  Master first (then trained); a
                                      Template only when the new master
                                      still cannot find the object
        ai     (few-shot AI)          Template (a new reference); with 2+
                                      calibration shots: a calibration shot
                                      (the shots replace the templates there)
        normal (template matching)    Template
  * safety              before and after every change the area is run on
                        its OWN master picture; if it passed before and
                        fails after, the change is taken back completely
                        (AreaBackup: config, templates, masters, trained
                        file, calibration shots)
  * audit log           models/heads/<model>/ng_recovery_log.csv
                        (never deleted by the 2-day data clean-up)
"""
import csv
import datetime
import json
import os
import re
import threading

import cv2
import numpy as np

import vision_detect as vd
from vision_detect import (METHOD_AI, METHOD_AIP, METHOD_NORMAL, MAX_MASTERS,
                           MAX_SHOTS, load_calib, load_masters, save_calib,
                           scale_frame)

SETTINGS_KEY = "ng_recovery"
DEFAULT_SETTINGS = {"auto_recheck": True, "popup_enabled": False,
                    "popup_timeout_s": 60}
DEFAULT_AUTH = {"password": "89012", "min_id_length": 3, "allowed_ids": [],
                "scan_timeout_s": 30}

TECHNIQUE_LABEL = {METHOD_AIP: "AI Image Processing",
                   METHOD_AI: "AI detection (few-shot)",
                   METHOD_NORMAL: "Template matching"}
ACTION_LABEL = {"master": "Add Master",
                "template": "Add Template",
                "shot": "Add Master shot (AI calibration)"}
# what the audit log calls each operation
OPERATION_NAME = {"master": "Master Added", "template": "Template Added",
                  "shot": "Master Shot Added"}
MIN_TEMPLATE_PX = 8


# ----------------------------------------------------------------------------
# settings
# ----------------------------------------------------------------------------
def _typed(raw, defaults):
    out = dict(defaults)
    for k, d in defaults.items():
        if k not in (raw or {}):
            continue
        v = raw[k]
        try:
            if isinstance(d, bool):
                out[k] = (v if isinstance(v, bool) else
                          str(v).strip().lower() in ("1", "true", "yes", "on"))
            elif isinstance(d, int):
                out[k] = max(0, int(float(v)))
            elif isinstance(d, list):
                out[k] = [str(x).strip() for x in (v or []) if str(x).strip()]
            else:
                out[k] = str(v)
        except (TypeError, ValueError):
            pass
    return out


def get_settings(heads, head):
    """Model-level settings (see the module text)."""
    return _typed(heads.get_settings(head).get(SETTINGS_KEY) or {},
                  DEFAULT_SETTINGS)


def set_settings(heads, head, **changes):
    cur = get_settings(heads, head)
    cur.update(_typed(changes, DEFAULT_SETTINGS))
    heads.set_settings(head, **{SETTINGS_KEY: cur})
    return cur


def auth_config(app_config_data):
    return _typed((app_config_data or {}).get(SETTINGS_KEY) or {},
                  DEFAULT_AUTH)


# ----------------------------------------------------------------------------
# authorisation
# ----------------------------------------------------------------------------
_PRINTABLE = re.compile(r"^[\x20-\x7e]+$")


def validate_operator_id(auth, code):
    """(ok, reason).  The scanned ID must be plain text of a sensible length
    and, when a list of allowed IDs is set, one of them."""
    code = (code or "").strip()
    if not code:
        return False, "Nothing was scanned."
    if not _PRINTABLE.match(code):
        return False, "The scanned ID has characters a badge never has."
    if len(code) < int(auth.get("min_id_length", 3)):
        return False, (f"The ID is too short (at least "
                       f"{auth.get('min_id_length', 3)} characters).")
    if len(code) > 64:
        return False, "The ID is too long (at most 64 characters)."
    allowed = auth.get("allowed_ids") or []
    if allowed and code not in allowed:
        return False, "This ID is not allowed to add masters or templates."
    return True, ""


def check_password(auth, pw):
    return pw is not None and str(pw).strip() == str(
        auth.get("password", DEFAULT_AUTH["password"])).strip()


# ----------------------------------------------------------------------------
# results
# ----------------------------------------------------------------------------
def overall_ok(results):
    """True when every area of every scene in `results` is OK."""
    return bool(results) and all(r.get("ok") for pr in results.values()
                                 for r in pr.values())


def ng_areas(results):
    """[(scene, area, result)] of the NG entries, in run order."""
    return [(p, c, r) for p, pr in (results or {}).items()
            for c, r in pr.items() if not r.get("ok")]


def area_ok(results, scene, area):
    r = (results or {}).get(scene, {}).get(area)
    return None if r is None else bool(r.get("ok"))


def result_text(ok):
    return "OK" if ok else "NG" if ok is not None else "-"


# ----------------------------------------------------------------------------
# what can be added to an area
# ----------------------------------------------------------------------------
def technique_label(method):
    return TECHNIQUE_LABEL.get(method or METHOD_NORMAL, str(method))


def recovery_actions(cfg, area_dir):
    """Ordered operations offered for an area: 'master', 'template', 'shot'.
    AI Image Processing always starts with the master."""
    method = (cfg or {}).get("method", METHOD_NORMAL)
    if method == METHOD_AIP:
        return ["master", "template"]
    if method == METHOD_AI:
        return ["shot"] if len(load_calib(area_dir)) >= 2 else ["template"]
    return ["template"]


# ----------------------------------------------------------------------------
# area pictures (from the geometry kept by the run)
# ----------------------------------------------------------------------------
def area_picture(geo, full=False):
    """The area's region as the detector saw it.
    geo = {"frame", "scale", "origin", "roi", "reg"}.
    full=True: the whole region (needed for a master); None when part of it
    is outside the picture."""
    if not geo or geo.get("frame") is None or geo.get("roi") is None:
        return None
    sframe = scale_frame(geo["frame"], geo.get("scale", 1.0))
    if full:
        x, y = [int(v) for v in geo["origin"]]
        w, h = [int(v) for v in geo["reg"][2:4]]
        H, W = sframe.shape[:2]
        if x < 0 or y < 0 or x + w > W or y + h > H:
            return None
        return sframe[y:y + h, x:x + w].copy()
    x0, y0, x1, y1 = [int(v) for v in geo["roi"]]
    pic = sframe[y0:y1, x0:x1]
    return pic.copy() if pic.size else None


# ----------------------------------------------------------------------------
# templates
# ----------------------------------------------------------------------------
def _template_count(area_dir):
    n = 0
    while os.path.isfile(os.path.join(area_dir, f"template_{n}.png")):
        n += 1
    return n


def crop_template(picture, rect):
    """Cut the template out; raises ValueError (plain words) when the box
    is unusable."""
    if picture is None or rect is None:
        raise ValueError("No box was drawn.")
    x, y, w, h = [int(v) for v in rect]
    H, W = picture.shape[:2]
    x, y = max(0, x), max(0, y)
    w, h = min(w, W - x), min(h, H - y)
    if w < MIN_TEMPLATE_PX or h < MIN_TEMPLATE_PX:
        raise ValueError(f"The box is too small (at least {MIN_TEMPLATE_PX} "
                         "pixels wide and high). Draw it around the whole "
                         "object.")
    return picture[y:y + h, x:x + w].copy(), (x, y, w, h)


def add_template(area_dir, config_path, crop, rect):
    """Store one more template of an area (template_<n>.png + its box in
    config.json 'temp_reg').  Returns n (for remove_template)."""
    n = _template_count(area_dir)
    path = os.path.join(area_dir, f"template_{n}.png")
    if not cv2.imwrite(path, crop):
        raise OSError(f"cannot write {path}")
    try:
        with open(config_path) as f:
            cfg = json.load(f)
        regs = list(cfg.get("temp_reg") or [])
        regs.append([int(v) for v in rect])
        cfg["temp_reg"] = regs
        tmp = config_path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(cfg, f, indent=2)
        os.replace(tmp, config_path)
    except Exception:
        try:
            os.remove(path)
        except OSError:
            pass
        raise
    return n


def remove_template(area_dir, config_path, n):
    """Undo add_template (only the newest template can be taken back)."""
    if n != _template_count(area_dir) - 1:
        return False
    try:
        os.remove(os.path.join(area_dir, f"template_{n}.png"))
        with open(config_path) as f:
            cfg = json.load(f)
        regs = list(cfg.get("temp_reg") or [])
        if regs:
            regs.pop()
        cfg["temp_reg"] = regs
        with open(config_path, "w") as f:
            json.dump(cfg, f, indent=2)
        return True
    except (OSError, json.JSONDecodeError):
        return False


def load_templates(area_dir):
    out = []
    for i in range(_template_count(area_dir)):
        t = cv2.imread(os.path.join(area_dir, f"template_{i}.png"))
        if t is not None:
            out.append(t)
    return out


# ----------------------------------------------------------------------------
# AI Image Processing masters + training
# ----------------------------------------------------------------------------
def merge_masters(masters, new):
    """masters + new, at most MAX_MASTERS.  When full, the oldest ADDED
    master goes (the first one - the original - always stays).
    Returns (list, dropped index or None).  ValueError when the picture
    cannot be a master of this area."""
    masters = list(masters)
    if masters and new.shape != masters[0].shape:
        raise ValueError("The picture is not the size of the area (is the "
                         "area cut by the picture edge?).")
    if any(np.array_equal(new, m) for m in masters):
        raise ValueError("This picture is already a master.")
    dropped = None
    if len(masters) >= MAX_MASTERS:
        dropped = 1 if len(masters) > 1 else 0
        masters.pop(dropped)
    masters.append(new)
    return masters, dropped


def train_aip(key, area_dir, masters, templates, cfg):
    """Train AI Image Processing on masters + templates (seconds; call from
    a worker thread).  Only when training works are the masters written and
    the trained file stored, so a failed training changes nothing on disk.
    Returns (report, stored).  Raises RuntimeError with a plain reason."""
    _m, rep = vd.aip_get(key, masters, templates, cfg)       # in memory
    vd.save_masters(area_dir, masters)                        # drops old .pkl
    stored = vd.aip_store_model(area_dir, key)
    vd._masters_cache.pop(area_dir, None)                     # re-read now
    return rep, stored


def add_calib_shot(area_dir, picture, roi):
    """One more AI calibration shot (the oldest goes when there are
    MAX_SHOTS).  Returns the number of shots."""
    shots = load_calib(area_dir)
    if len(shots) >= MAX_SHOTS:
        shots.pop(0)
    shots.append((picture, tuple(int(v) for v in roi)))
    save_calib(area_dir, shots)
    vd._shots_cache.pop(area_dir, None)
    return len(shots)


def area_masters(area_dir):
    return load_masters(area_dir)


# ----------------------------------------------------------------------------
# safety: a change must not break the area on its own master picture
# ----------------------------------------------------------------------------
class AreaBackup:
    """Everything an NG recovery change can touch in one area folder, so a
    change can be taken back completely: config.json, the templates, the AI
    Image Processing masters + trained file, the AI calibration shots."""

    def __init__(self, area_dir, config_path, key, method):
        self.dir, self.config_path, self.key = area_dir, config_path, key
        self.method = method
        with open(config_path, "rb") as f:
            self.config = f.read()
        self.n_templates = _template_count(area_dir)
        self.masters = load_masters(area_dir) if method == METHOD_AIP else None
        pkl = os.path.join(vd.aip_dir(area_dir), "trained.pkl")
        self.pkl = None
        if method == METHOD_AIP and os.path.isfile(pkl):
            with open(pkl, "rb") as f:
                self.pkl = f.read()
        self.shots = load_calib(area_dir) if method == METHOD_AI else None

    def restore(self):
        with open(self.config_path, "wb") as f:
            f.write(self.config)
        n = self.n_templates
        while os.path.isfile(os.path.join(self.dir, f"template_{n}.png")):
            os.remove(os.path.join(self.dir, f"template_{n}.png"))
            n += 1
        if self.masters is not None:
            vd.save_masters(self.dir, self.masters)          # drops new .pkl
            if self.pkl is not None:
                with open(os.path.join(vd.aip_dir(self.dir), "trained.pkl"),
                          "wb") as f:
                    f.write(self.pkl)
        if self.shots is not None:
            save_calib(self.dir, self.shots)
        for cache, k in ((vd._aip_cache, self.key), (vd._ai_cache, self.key),
                         (vd._masters_cache, self.dir),
                         (vd._shots_cache, self.dir)):
            cache.pop(k, None)


def master_check(child_manager, area, key, analyze_pins, default_limit=20.0):
    """Run the area, as it is stored NOW, on its own master picture (the
    picture the area was taught on).  Returns (ok, text).  analyze_pins is
    the pin check of the program (master pins vs found objects)."""
    data = child_manager.load(area)
    if data is None or data.get("region") is None or not data["region"].size:
        return True, "no master picture"
    cfg = data["config"]
    adir = os.path.dirname(child_manager._paths(area)["config"])
    method = cfg.get("method", METHOD_NORMAL)
    templates = [t for t in data["templates"] if t is not None]
    matches, _o, _m = vd.detect_area(
        data["region"], templates, cfg, key=key,
        shots=load_calib(adir) if method == METHOD_AI else None,
        masters=load_masters(adir) if method == METHOD_AIP else None,
        area_dir=adir if method == METHOD_AIP else None)
    pins = cfg.get("matches") or []
    if not pins:
        return bool(matches), f"{len(matches)} object(s) on the master"
    try:
        lim = float(cfg.get("pin_limit") or default_limit)
    except (TypeError, ValueError):
        lim = float(default_limit)
    ok, _p, summ = analyze_pins(pins, matches, lim)
    return bool(ok), (f"{summ.get('in_range', 0)}/{summ.get('total', 0)} "
                      "pins on the master")


# ----------------------------------------------------------------------------
# audit log
# ----------------------------------------------------------------------------
class AuditLog:
    """One CSV per model: every Master / Template addition (and every
    refused authorisation) with the operator ID."""

    FIELDS = ["date_time", "operator_id", "model", "scene", "area",
              "technique", "operation", "previous_result", "training_status",
              "final_result", "status", "details"]
    _lock = threading.Lock()

    def __init__(self, head_dir):
        self.path = os.path.join(head_dir, "ng_recovery_log.csv")

    def write(self, **row):
        row = {k: str(row.get(k, "") if row.get(k) is not None else "")
               for k in self.FIELDS}
        if not row["date_time"]:
            row["date_time"] = datetime.datetime.now().strftime(
                "%Y-%m-%d %H:%M:%S")
        with self._lock:
            os.makedirs(os.path.dirname(self.path), exist_ok=True)
            new = not os.path.isfile(self.path)
            with open(self.path, "a", newline="") as f:
                w = csv.DictWriter(f, fieldnames=self.FIELDS)
                if new:
                    w.writeheader()
                w.writerow(row)
        return row

    def read(self, last=200):
        try:
            with open(self.path, newline="") as f:
                rows = list(csv.DictReader(f))
        except OSError:
            return []
        return rows[-last:]

    @staticmethod
    def line(row):
        """One readable line for the detection log / comm log."""
        where = "/".join(x for x in (row["model"], row["scene"], row["area"])
                         if x)
        parts = [f"[NG recovery] {row['operation']}",
                 f"operator {row['operator_id']}" if row["operator_id"]
                 else "", where + (f" ({row['technique']})"
                                   if row["technique"] else ""),
                 f"before {row['previous_result']}"
                 if row["previous_result"] else "",
                 f"training {row['training_status']}"
                 if row["training_status"] else "",
                 f"after {row['final_result']}" if row["final_result"] else "",
                 row["status"], row["details"]]
        return " - ".join(p for p in parts if p)
