
import os

os.environ["QT_XCB_FORCE_SOFTWARE_OPENGL"] = "1"
os.environ["LIBGL_ALWAYS_SOFTWARE"] = "1"
import html
import json
import re
import shutil
import time
import socket
import threading
import traceback
import datetime
from collections import deque

import cv2
import numpy as np
import yaml

from PyQt6.QtCore import Qt, QTimer, QRectF, pyqtSignal
from PyQt6.QtGui import (QImage, QPixmap, QPainter, QColor, QPen, QFont,
                         QBrush, QCursor, QShortcut, QKeySequence)
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QFrame, QDialog, QLabel, QPushButton,
    QLineEdit, QTextEdit, QCheckBox, QComboBox, QVBoxLayout, QHBoxLayout,
    QGridLayout, QGroupBox, QMessageBox, QInputDialog, QTreeWidget,
    QTreeWidgetItem, QScrollArea, QSizePolicy)

from two_point import (
    ParentModelManager,
    detect_multi_scale_templates,
    _init_cameras,
)
from qr_reader import (QrReader, QrRules, QrResult, norm_from_rect,
                       available_backends)

MODELS_BASE = "models"
CANVAS_W, CANVAS_H = 940, 540
COMM_LOG_DIR = os.path.join(MODELS_BASE, "comm_logs")
GLOBAL_COMM_PATH = os.path.join(MODELS_BASE, "comm_global.yaml")
APP_CONFIG_PATH = os.path.join(MODELS_BASE, "app_config.yaml")


# ============================================================================
# APP CONFIG  (password, lock state, ui parameters)
# ============================================================================
class AppConfig:
    def __init__(self):
        self.data = {
            "password": "1234",          # change it here (models/app_config.yaml)
            "locked": False,
            "ui": {"frame_scale": 0.6},  # same 0.6 resize as two_point.py
            "scanner": {"test_timeout_s": 10},
            # camera QR reading (only used when the part-scan source = camera)
            "qr": {"timeout_s": 6.0,         # one camera read: give up after
                   "scene_timeout_s": 3.0,   # scene QR check: give up after
                   "min_agree": 2,           # identical FRAMES needed to accept
                   "trigger_wait_s": 20,     # wait for the client's trigger
                   "max_tries": 10,          # master scan: attempts, then NG
                   "retry_delay_s": 1.5,     # pause between two attempts
                   "save_failed": True},     # keep the ROI image of failed reads
            # what is uploaded to MES: small images + the full camera view
            "mes_upload": {"max_width": 1024, "jpeg_quality": 75,
                           "include_camera_view": True},
            # saved data (MES logs/reports/images, comm logs, failed QR
            # images) is kept this many days, then deleted
            "retention": {"days": 2},
        }
        self.load()

    def load(self):
        if os.path.isfile(APP_CONFIG_PATH):
            try:
                with open(APP_CONFIG_PATH) as f:
                    saved = yaml.safe_load(f) or {}
                for k in ("password", "locked"):
                    if k in saved:
                        self.data[k] = saved[k]
                self.data["ui"].update(saved.get("ui") or {})
                self.data["scanner"].update(saved.get("scanner") or {})
                self.data["qr"].update(saved.get("qr") or {})
                self.data["mes_upload"].update(saved.get("mes_upload") or {})
                self.data["retention"].update(saved.get("retention") or {})
            except yaml.YAMLError:
                pass
        self.save()   # make sure the file exists with all parameters

    def save(self):
        os.makedirs(os.path.dirname(APP_CONFIG_PATH), exist_ok=True)
        with open(APP_CONFIG_PATH, "w") as f:
            yaml.dump(self.data, f, default_flow_style=False, sort_keys=False)


# ============================================================================
# 2-DAY DATA BUFFER
# Deletes SAVED RUN DATA older than N days: MES logs / reports / uploaded
# images of every model, the daily communication logs and the failed-QR
# images.  Models, areas, templates, yaml configs and counters are NEVER
# touched.
# ============================================================================
def purge_old_data(days=2, base=MODELS_BASE):
    """-> number of files deleted."""
    try:
        days = float(days)
    except (TypeError, ValueError):
        days = 2.0
    if days <= 0:
        return 0
    cutoff = time.time() - days * 86400.0
    roots = [os.path.join(base, "comm_logs"), os.path.join(base, "qr_fail")]
    heads = os.path.join(base, "heads")
    if os.path.isdir(heads):
        for h in os.listdir(heads):
            roots.append(os.path.join(heads, h, "mes_logs"))
    removed = 0
    for root in roots:
        if not os.path.isdir(root):
            continue
        for dp, _dn, files in os.walk(root, topdown=False):
            for f in files:
                fp = os.path.join(dp, f)
                try:
                    if os.path.getmtime(fp) < cutoff:
                        os.remove(fp)
                        removed += 1
                except OSError:
                    pass
            if dp != root:
                try:
                    if not os.listdir(dp):
                        os.rmdir(dp)
                except OSError:
                    pass
    return removed


# ============================================================================
# MODEL LEVEL  (folder names kept as heads/parents/children for compatibility)
# ============================================================================
class HeadManager:
    """UI names: model / scene / area. Disk names: head / parent / child."""

    def __init__(self, base_dir=MODELS_BASE):
        self.heads_dir = os.path.join(base_dir, "heads")

    def _head_dir(self, head):
        return os.path.join(self.heads_dir, head)

    def _head_config(self, head):
        return os.path.join(self._head_dir(head), "head_config.json")

    def _parents_dir(self, head):
        return os.path.join(self._head_dir(head), "parents")

    def comm_yaml_path(self, head):
        return os.path.join(self._head_dir(head), "communication.yaml")

    def exists(self, head):
        return os.path.isfile(self._head_config(head))

    def create(self, head):
        os.makedirs(self._parents_dir(head), exist_ok=True)
        if not self.exists(head):
            self._save(head, {"parents": [], "finished": False})

    def delete(self, head):
        hdir = self._head_dir(head)
        if os.path.exists(hdir):
            shutil.rmtree(hdir, ignore_errors=True)

    def list_heads(self):
        if not os.path.isdir(self.heads_dir):
            return []
        return sorted(n for n in os.listdir(self.heads_dir)
                      if os.path.isfile(self._head_config(n)))

    def _save(self, head, cfg):
        os.makedirs(self._head_dir(head), exist_ok=True)
        with open(self._head_config(head), "w") as f:
            json.dump(cfg, f, indent=2)

    def _load(self, head):
        if not self.exists(head):
            return {"parents": [], "finished": False}
        with open(self._head_config(head)) as f:
            return json.load(f)

    def _raw_parents(self, head):
        return list(self._load(head).get("parents", []))

    def scene_exists(self, head, scene):
        """Any saved scene - including a master-QR scene that has no areas."""
        return scene in self._raw_parents(head)

    def list_parents(self, head):
        """DETECTION scenes = saved scenes that have at least one area.
        A master-QR scene holds only the QR (no areas), so it is not a
        detection scene: no run button, no OK/NG signals, not part of the
        cycle."""
        raw = self._raw_parents(head)
        if not raw:
            return []
        store = self.get_parent_store(head)
        return [p for p in raw if store.list_children(p)]

    def drop_scene_if_empty(self, head, scene):
        """Forget a saved scene that has neither areas nor QR scanners."""
        if self.get_parent_store(head).list_children(scene):
            return
        if any(q["scene"] == scene for q in self.get_qr_scanners(head)):
            return
        cfg = self._load(head)
        if scene in cfg.get("parents", []):
            cfg["parents"].remove(scene)
            self._save(head, cfg)

    def add_parent(self, head, parent):
        cfg = self._load(head)
        cfg.setdefault("parents", [])
        if parent not in cfg["parents"]:
            cfg["parents"].append(parent)
        self._save(head, cfg)

    def is_finished(self, head):
        return bool(self._load(head).get("finished", False))

    def set_finished(self, head, value=True):
        cfg = self._load(head)
        cfg["finished"] = bool(value)
        self._save(head, cfg)

    def get_parent_store(self, head):
        return ParentModelManager(base_dir=self._parents_dir(head))

    def structure(self, head):
        store = self.get_parent_store(head)
        return {p: list(store.list_children(p))
                for p in self.list_parents(head)}

    # -- detection settings --------------------------------------------------
    def _settings_path(self, head):
        return os.path.join(self._head_dir(head), "detection_settings.json")

    def get_settings(self, head):
        defaults = {"distance_threshold": 20}
        p = self._settings_path(head)
        if os.path.isfile(p):
            try:
                with open(p) as f:
                    defaults.update(json.load(f) or {})
            except (json.JSONDecodeError, OSError):
                pass
        return defaults

    def set_settings(self, head, **kw):
        s = self.get_settings(head)
        s.update(kw)
        os.makedirs(self._head_dir(head), exist_ok=True)
        with open(self._settings_path(head), "w") as f:
            json.dump(s, f, indent=2)

    # -- barcode scanner config (per model) ----------------------------------
    def scanner_path(self, head):
        return os.path.join(self._head_dir(head), "scanner.yaml")

    def get_scanner(self, head):
        d = {"enabled": False, "prefix": "", "send_to_client": True,
             "auto_popup": True, "require_operator": False,
             "source": "scanner"}       # part scan: "scanner" | "camera"
        p = self.scanner_path(head)
        if os.path.isfile(p):
            try:
                with open(p) as f:
                    d.update(yaml.safe_load(f) or {})
            except yaml.YAMLError:
                pass
        d.pop("device", None)      # legacy hidraw path — no longer used
        if d.get("source") not in ("scanner", "camera"):
            d["source"] = "scanner"
        return d

    def set_scanner(self, head, **kw):
        d = self.get_scanner(head)
        d.update(kw)
        os.makedirs(self._head_dir(head), exist_ok=True)
        with open(self.scanner_path(head), "w") as f:
            yaml.dump(d, f, default_flow_style=False, sort_keys=False)
        return d

    # -- camera QR scanners (per model) ---------------------------------------
    # Only active when the model's part-scan source is "camera".  Stored in
    # <model>/qr_scanners.yaml, keyed by scene.  Exactly ONE may be the master:
    # the one the camera reads at the start of every cycle.
    def qr_path(self, head):
        return os.path.join(self._head_dir(head), "qr_scanners.yaml")

    def camera_qr_enabled(self, head):
        sc = self.get_scanner(head)
        return bool(sc.get("enabled")) and sc.get("source") == "camera"

    def get_qr_scanners(self, head):
        items = []
        p = self.qr_path(head)
        if os.path.isfile(p):
            try:
                with open(p) as f:
                    items = (yaml.safe_load(f) or {}).get("scanners") or []
            except yaml.YAMLError:
                items = []
        out = []
        for it in items:
            if isinstance(it, dict) and it.get("id") and it.get("scene"):
                it.setdefault("name", str(it["id"]))
                it.setdefault("roi_norm", None)
                it.setdefault("length", 0)
                it.setdefault("starts_with", "")
                it["master"] = bool(it.get("master", False))
                out.append(it)
        return out

    def _save_qr(self, head, items):
        os.makedirs(self._head_dir(head), exist_ok=True)
        with open(self.qr_path(head), "w") as f:
            yaml.dump({"scanners": items}, f, default_flow_style=False,
                      sort_keys=False)

    def qr_scanners_for_scene(self, head, scene):
        return [q for q in self.get_qr_scanners(head) if q["scene"] == scene]

    def get_master_qr(self, head, include_scene=None):
        """The model's single master QR.  A scanner whose scene was never
        saved (scene build cancelled) does not count - except the scene that
        is being built right now (include_scene)."""
        live = set(self._raw_parents(head))
        if include_scene:
            live.add(include_scene)
        for q in self.get_qr_scanners(head):
            if q["master"] and q["scene"] in live:
                return q
        return None

    def add_qr_scanner(self, head, scene, name, roi_norm, length=0,
                       starts_with="", master=False):
        items = self.get_qr_scanners(head)
        nums = [int(str(q["id"]).split("_")[-1]) for q in items
                if str(q["id"]).split("_")[-1].isdigit()]
        lost = []
        if master:                      # only ONE master per model: the old
            lost = [q["scene"] for q in items if q["master"]]   # one goes
            items = [q for q in items if not q["master"]]
        item = {"id": f"qr_{max(nums or [0]) + 1}", "scene": scene,
                "name": str(name), "master": bool(master),
                "roi_norm": ([float(v) for v in roi_norm]
                             if roi_norm else None),
                "length": int(length or 0),
                "starts_with": str(starts_with or "")}
        items.append(item)
        self._save_qr(head, items)
        for sc in lost:                 # its scene held nothing else
            self.drop_scene_if_empty(head, sc)
        return item

    def remove_qr_scanner(self, head, qid):
        gone = [q for q in self.get_qr_scanners(head) if q["id"] == qid]
        self._save_qr(head, [q for q in self.get_qr_scanners(head)
                             if q["id"] != qid])
        for q in gone:
            self.drop_scene_if_empty(head, q["scene"])

    def remove_qr_for_scene(self, head, scene):
        self._save_qr(head, [q for q in self.get_qr_scanners(head)
                             if q["scene"] != scene])

    # -- MES integration config (per model) ----------------------------------
    def mes_path(self, head):
        return os.path.join(self._head_dir(head), "mes.yaml")

    def mes_log_dir(self, head):
        d = os.path.join(self._head_dir(head), "mes_logs")
        os.makedirs(d, exist_ok=True)
        return d

    def get_mes(self, head):
        d = {
            "enabled": False,
            "mes_ip": "", "username": "", "password": "",
            "login_stage_name": "", "stage_id": "", "stage_name": "",
            "product_id": "", "product_name": "",
            "operator_id": "", "created_by": "", "machine_no": "",
            "file_format": "txt",
            "block_on_prev_fail": True,   # honoured in LOCKED (production) mode
            "upload_images": True,
            "defect_type": "Pin Bend",
        }
        p = self.mes_path(head)
        if os.path.isfile(p):
            try:
                with open(p) as f:
                    d.update(yaml.safe_load(f) or {})
            except yaml.YAMLError:
                pass
        # log/results folder always lives with the model
        d["results_dir"] = self.mes_log_dir(head)
        return d

    def set_mes(self, head, **kw):
        d = self.get_mes(head)
        d.update(kw)
        d.pop("results_dir", None)      # derived, never stored
        os.makedirs(self._head_dir(head), exist_ok=True)
        with open(self.mes_path(head), "w") as f:
            yaml.dump(d, f, default_flow_style=False, sort_keys=False)
        return self.get_mes(head)

    # -- OK/NG counters (per model, persisted) -------------------------------
    def _counters_path(self, head):
        return os.path.join(self._head_dir(head), "counters.json")

    def get_counters(self, head):
        d = {"cycle": {"ok": 0, "ng": 0}, "scenes": {}, "areas": {}}
        p = self._counters_path(head)
        if os.path.isfile(p):
            try:
                with open(p) as f:
                    saved = json.load(f) or {}
                for k in d:
                    if k in saved:
                        d[k] = saved[k]
            except (json.JSONDecodeError, OSError):
                pass
        return d

    def add_result_counts(self, head, scope, results):
        """results: {scene: {area: {'ok': bool, ...}}}.
        Scene + area counters count on every run. The complete-cycle counter
        is NOT incremented here — a cycle completes when ALL scenes of the
        model have been run (see add_cycle_count)."""
        c = self.get_counters(head)
        for scene, areas in results.items():
            if not areas:
                continue
            s = c["scenes"].setdefault(scene, {"ok": 0, "ng": 0})
            s["ok" if all(r["ok"] for r in areas.values()) else "ng"] += 1
            for area, r in areas.items():
                a = c["areas"].setdefault(f"{scene}/{area}", {"ok": 0, "ng": 0})
                a["ok" if r["ok"] else "ng"] += 1
        with open(self._counters_path(head), "w") as f:
            json.dump(c, f, indent=2)
        return c

    def add_cycle_count(self, head, ok):
        """One complete cycle = every scene of the model has been run once
        (in any order / any combination of runs)."""
        c = self.get_counters(head)
        c["cycle"]["ok" if ok else "ng"] += 1
        with open(self._counters_path(head), "w") as f:
            json.dump(c, f, indent=2)
        return c

    def reset_counters(self, head):
        with open(self._counters_path(head), "w") as f:
            json.dump({"cycle": {"ok": 0, "ng": 0},
                       "scenes": {}, "areas": {}}, f, indent=2)


# ============================================================================
# QUICK INPUT SCAN BOX  (global always-on-top grab — replaces the hidraw
# reader).  While this box is open it takes a GLOBAL keyboard grab: no other
# window or application on the desktop receives keystrokes, so a
# keyboard-wedge barcode scanner (which is what almost every USB/BT scanner
# is) types straight into it.  Enter completes the scan and returns the text,
# Escape cancels.  No /dev/hidraw path, no root permissions, no udev rules.
# ============================================================================
class QuickScanInput(QDialog):
    """Floating, frameless, always-on-top single-line input with a system-wide
    keyboard grab.

    Qt::Popup makes X11 grab the keyboard and pointer for the whole session,
    which is exactly the behaviour we want for a scan step: the cursor is
    always in this field and nothing else can be typed into until Enter.

    Note: under a native Wayland session compositors deliberately refuse
    system-wide grabs.  The box still floats on top and still keeps focus, it
    just cannot lock other apps out.  Under Xorg / XWayland the grab is real.
    """

    def __init__(self, parent=None, prompt="SCAN BARCODE",
                 sub="", timeout_s=0, width=620, initial=""):
        super().__init__(parent)
        self.value = None
        self._deadline = (time.time() + timeout_s) if timeout_s else 0.0
        self._timeout_s = timeout_s

        self.setWindowFlags(Qt.WindowType.Popup
                            | Qt.WindowType.FramelessWindowHint
                            | Qt.WindowType.WindowStaysOnTopHint)
        self.setStyleSheet("QDialog { background-color: #14141a;"
                           " border: 2px solid #3b6ef5; border-radius: 8px; }")

        lay = QVBoxLayout(self)
        lay.setContentsMargins(18, 12, 18, 12)
        lay.setSpacing(6)

        self.title_lbl = QLabel(prompt)
        self.title_lbl.setStyleSheet("color: #ffd43b; font-size: 13pt;"
                                     " font-weight: bold;")
        self.title_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lay.addWidget(self.title_lbl)

        if sub:
            s = QLabel(sub)
            s.setStyleSheet(f"color: {C_MUTED}; font-size: 9pt;")
            s.setAlignment(Qt.AlignmentFlag.AlignCenter)
            s.setWordWrap(True)
            lay.addWidget(s)

        self.entry = QLineEdit()
        self.entry.setPlaceholderText("scan now… (or type and press Enter)")
        self.entry.setStyleSheet(
            f"background: {C_FIELD}; color: white; padding: 10px;"
            " border-radius: 6px; font-size: 15pt; font-weight: bold;"
            " font-family: Consolas, monospace;")
        self.entry.returnPressed.connect(self._submit)
        if initial:
            self.entry.setText(initial)
        lay.addWidget(self.entry)

        self.hint = QLabel("Enter = accept        Esc = cancel")
        self.hint.setStyleSheet(f"color: {C_FAINT}; font-size: 8pt;")
        self.hint.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lay.addWidget(self.hint)

        self.setFixedWidth(width)
        self.adjustSize()
        self._place_top_center()

        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)
        self._timer.start(200)

    # -- geometry ------------------------------------------------------------
    def _place_top_center(self):
        scr = QApplication.primaryScreen()
        g = scr.availableGeometry() if scr else None
        if g is None:
            return
        x = g.x() + (g.width() - self.width()) // 2
        y = g.y() + 70
        self.move(x, y)

    # -- lifecycle -----------------------------------------------------------
    def showEvent(self, e):
        super().showEvent(e)
        self.raise_()
        self.activateWindow()
        self.entry.setFocus(Qt.FocusReason.OtherFocusReason)
        self.entry.setCursorPosition(len(self.entry.text()))
        # Qt::Popup already grabs globally on X11; this keeps the caret nailed
        # to the field even if the popup grab is refused (Wayland fallback).
        try:
            self.entry.grabKeyboard()
        except Exception:
            pass

    def _release(self):
        try:
            self.entry.releaseKeyboard()
        except Exception:
            pass
        self._timer.stop()

    def _submit(self):
        txt = self.entry.text().strip()
        if not txt:
            return
        self.value = txt
        self._release()
        self.accept()

    def _cancel(self):
        self.value = None
        self._release()
        self.reject()

    def keyPressEvent(self, e):
        if e.key() == Qt.Key.Key_Escape:
            self._cancel()
            return
        super().keyPressEvent(e)

    def mousePressEvent(self, e):
        # a Qt::Popup normally closes on any outside click — swallow it so the
        # box only ever leaves on Enter / Esc / timeout
        e.accept()

    def _tick(self):
        if not self._deadline:
            return
        left = self._deadline - time.time()
        if left <= 0:
            self.hint.setText("timed out")
            self._cancel()
            return
        self.hint.setText(f"Enter = accept        Esc = cancel"
                          f"        ({int(left) + 1}s)")

    # -- convenience ---------------------------------------------------------
    @staticmethod
    def ask(parent=None, prompt="SCAN BARCODE", sub="", timeout_s=0,
            initial=""):
        """Open the box modally and return the scanned/typed text, or None."""
        dlg = QuickScanInput(parent, prompt=prompt, sub=sub,
                             timeout_s=timeout_s, initial=initial)
        dlg.exec()
        return dlg.value


# ============================================================================
# MES INTEGRATION  (mes_client.MesClient wrapper)
#
# Everything here runs OFF the UI thread — MES calls are HTTP and would
# otherwise freeze the live view.  Results come back through callbacks that
# are marshalled onto the UI thread with app.after(0, ...).
#
# Per unit (serial number) two files are produced in <model>/mes_logs/:
#   <serial>_mes_log.txt      every API hit: time, URL, payload, response
#                             (written by MesClient itself)
#   <serial>_report_<ts>.txt  the detection report: per-area results plus the
#                             MES download link of every uploaded image, and
#                             the final stage-report response.  This is the
#                             file that gets uploaded and referenced by
#                             create_stage_report().
# ============================================================================
try:
    from mes_client import MesClient
    MES_AVAILABLE = True
    MES_IMPORT_ERR = ""
except Exception as _e:                      # requests missing, file missing…
    MesClient = None
    MES_AVAILABLE = False
    MES_IMPORT_ERR = str(_e)


class MesSession:
    """One MES client bound to one model."""

    def __init__(self, app, head, cfg):
        self.app = app
        self.head = head
        self.cfg = dict(cfg)
        self.client = MesClient(dict(cfg)) if MES_AVAILABLE else None
        self._lock = threading.Lock()
        self.logged_in = False
        self.last_msg = ""

    # -- plumbing ------------------------------------------------------------
    def available(self):
        return self.client is not None

    def update_cfg(self, cfg):
        self.cfg = dict(cfg)
        if self.client:
            self.client.update_config(dict(cfg))

    def set_operator(self, operator_id):
        """Use the Operator ID scanned when the model was opened, instead of
        the fixed one from mes.yaml, for every subsequent MES call."""
        if not operator_id:
            return
        self.cfg["operator_id"] = operator_id
        if self.client:
            self.client.cfg["operator_id"] = operator_id

    def _async(self, fn, cb, on_error=None):
        """Run fn() on a worker thread, deliver its return value to cb() on
        the UI thread.  on_error(text) builds the value passed to cb when fn
        raises, so each caller keeps its own return shape."""
        def run():
            try:
                res = fn()
            except Exception:
                txt = "MES error:\n" + traceback.format_exc()
                res = on_error(txt) if on_error else (False, txt)
            self.app.after(0, lambda: cb(res))
        threading.Thread(target=run, daemon=True).start()

    # -- API -----------------------------------------------------------------
    def login_async(self, cb):
        if not self.available():
            cb((False, f"mes_client not available ({MES_IMPORT_ERR})"))
            return

        def do():
            with self._lock:
                ok, msg = self.client.login()
                self.logged_in = ok
                return ok, msg
        self._async(do, cb)

    def logout(self):
        if self.available() and self.logged_in:
            try:
                with self._lock:
                    self.client.logout()
            except Exception:
                pass
            self.logged_in = False

    def check_previous_async(self, serial, cb):
        """cb((state, msg)) with state in pass | all_passed | fail | error."""
        if not self.available():
            cb(("error", f"mes_client not available ({MES_IMPORT_ERR})"))
            return

        def do():
            with self._lock:
                self.client.set_serial_log(serial)
                if not self.logged_in:
                    ok, msg = self.client.login()
                    self.logged_in = ok
                    if not ok:
                        return "error", f"MES login failed: {msg}"
                return self.client.previous_stage_check_ex(serial)
        self._async(do, cb, on_error=lambda t: ("error", t))

    def _upload(self, path, serial, status_label, ext):
        """upload_file() with the extension forced for this one call."""
        old = self.client.cfg.get("file_format")
        self.client.cfg["file_format"] = ext
        try:
            return self.client.upload_file(path, serial, status_label)
        finally:
            self.client.cfg["file_format"] = old

    def submit_cycle_async(self, bundle, cb):
        """Full end-of-cycle submission.

        bundle = {serial, ok, images:[(label, path)], report_lines:[str],
                  defects:[dict], report_path}

        Order (per the MES integration doc):
            1. upload every result image      -> download links
            2. append those links to the detection report, save it
            3. upload the report              -> report download link
            4. create_stage_report(...)       -> final MES response
            5. append the MES response to the report
        cb((ok, msg, report_path))
        """
        if not self.available():
            cb((False, f"mes_client not available ({MES_IMPORT_ERR})", None))
            return

        def do():
            with self._lock:
                serial = bundle["serial"]
                status_label = "PASS" if bundle["ok"] else "FAIL"
                self.client.set_serial_log(serial)
                if not self.logged_in:
                    ok, msg = self.client.login()
                    self.logged_in = ok
                    if not ok:
                        return False, f"MES login failed: {msg}", None

                lines = list(bundle["report_lines"])
                img_links = []
                lines.append("")
                lines.append("-" * 66)
                lines.append("IMAGES UPLOADED TO MES")
                lines.append("-" * 66)
                if bundle["images"]:
                    for label, path in bundle["images"]:
                        ext = os.path.splitext(path)[1].lstrip(".") or "jpg"
                        ok_up, link = self._upload(path, serial,
                                                   status_label, ext)
                        if ok_up:
                            img_links.append((label, link))
                            lines.append(f"{label:<22} {os.path.basename(path)}")
                            lines.append(f"{'':<22} link: {link}")
                        else:
                            lines.append(f"{label:<22} "
                                         f"{os.path.basename(path)}")
                            lines.append(f"{'':<22} UPLOAD FAILED: {link}")
                else:
                    lines.append("(image upload disabled or no images)")

                # 2. write the report with the image links in it
                report_path = bundle["report_path"]
                with open(report_path, "w", encoding="utf-8") as fh:
                    fh.write("\n".join(lines) + "\n")

                # 3. upload the report itself
                ext = self.cfg.get("file_format") or "txt"
                ok_rep, report_link = self._upload(report_path, serial,
                                                   status_label, ext)
                if not ok_rep:
                    with open(report_path, "a", encoding="utf-8") as fh:
                        fh.write(f"\nREPORT UPLOAD FAILED: {report_link}\n")
                    return (False, f"report upload failed: {report_link}",
                            report_path)

                # 4. stage report
                ok_sr, msg = self.client.create_stage_report(
                    serial_no=serial,
                    status=bool(bundle["ok"]),
                    download_link=report_link,
                    defects=bundle.get("defects") or None)

                # 5. close the report off with the MES answer
                with open(report_path, "a", encoding="utf-8") as fh:
                    fh.write("\n" + "-" * 66 + "\n")
                    fh.write("MES STAGE REPORT\n")
                    fh.write("-" * 66 + "\n")
                    fh.write(f"report download link : {report_link}\n")
                    fh.write(f"status sent          : {status_label}\n")
                    fh.write(f"MES response         : "
                             f"{'OK' if ok_sr else 'ERROR'} — {msg}\n")
                    fh.write(f"time                 : "
                             f"{datetime.datetime.now():%Y-%m-%d %H:%M:%S}\n")
                return ok_sr, msg, report_path
        self._async(do, cb, on_error=lambda t: (False, t, None))


# ============================================================================
# GLOBAL COMM SETTINGS + PER-MODEL COMM CONFIG
# ============================================================================
class GlobalCommSettings:
    def __init__(self):
        self.data = {
            "connection": {"ip": "192.168.0.50", "port": 2000},
            "handshake": {"enabled": True, "ping": "PING", "reply": "PONG"},
        }
        self.load()

    def load(self):
        if os.path.isfile(GLOBAL_COMM_PATH):
            try:
                with open(GLOBAL_COMM_PATH) as f:
                    saved = yaml.safe_load(f) or {}
                self.data["connection"].update(saved.get("connection") or {})
                self.data["handshake"].update(saved.get("handshake") or {})
            except yaml.YAMLError:
                pass

    def save(self):
        os.makedirs(os.path.dirname(GLOBAL_COMM_PATH), exist_ok=True)
        with open(GLOBAL_COMM_PATH, "w") as f:
            yaml.dump(self.data, f, default_flow_style=False, sort_keys=False)


class CommConfig:
    """models/heads/<model>/communication.yaml — yaml keys keep the old names
    (head/parents/children/PLC) so existing files still load; only the UI
    wording is model/scene/area/client."""

    HEAD_FIELDS = ["SELECT", "PLC", "OK", "NG"]
    PARENT_FIELDS = ["PLC", "OK", "NG"]
    CHILD_FIELDS = ["OK", "NG"]
    # camera-QR node (only when part-scan source = camera AND a master QR
    # exists):  REQ = we send it when the operator presses SCAN QR,
    #           TRIGGER = client sends it -> camera reads now,
    #           OK / NG = we send the read result
    QR_FIELDS = ["REQ", "TRIGGER", "OK", "NG"]

    def __init__(self, heads: HeadManager, head_name: str):
        self.heads = heads
        self.head = head_name
        self.path = heads.comm_yaml_path(head_name)
        self.data = self.load()

    def _defaults(self):
        structure = self.heads.structure(self.head)
        d = {"head": {"SELECT": "", "PLC": "", "OK": "", "NG": ""},
             "qr": {"REQ": "", "TRIGGER": "", "OK": "", "NG": ""},
             "parents": {}}
        for p, children in structure.items():
            d["parents"][p] = {
                "PLC": "", "OK": "", "NG": "",
                "area_wise": True, "combine": True,
                "children": {c: {"PLC": "", "OK": "", "NG": ""}
                             for c in children},
            }
        return d

    def load(self):
        base = self._defaults()
        if os.path.isfile(self.path):
            try:
                with open(self.path) as f:
                    saved = yaml.safe_load(f) or {}
            except yaml.YAMLError:
                saved = {}
            if "head" in saved:
                for k, v in (saved["head"] or {}).items():
                    if k in base["head"]:
                        base["head"][k] = v
            for k, v in (saved.get("qr") or {}).items():
                if k in base["qr"]:
                    base["qr"][k] = v
            for p, pconf in (saved.get("parents") or {}).items():
                if p not in base["parents"]:
                    continue
                children_saved = (pconf or {}).pop("children", {}) or {}
                base["parents"][p].update(pconf or {})
                for c, cconf in children_saved.items():
                    if c in base["parents"][p]["children"]:
                        base["parents"][p]["children"][c].update(cconf or {})
        return base

    def save(self):
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        with open(self.path, "w") as f:
            yaml.dump(self.data, f, default_flow_style=False, sort_keys=False)

    def qr_required(self):
        """Camera part-scan with a master QR -> the QR signals must be set."""
        return bool(self.heads.camera_qr_enabled(self.head)
                    and self.heads.get_master_qr(self.head))

    def missing_fields(self):
        miss = []
        for f in self.HEAD_FIELDS:
            if not str(self.data["head"].get(f, "")).strip():
                miss.append(f"model.{f}")
        if self.qr_required():
            for f in self.QR_FIELDS:
                if not str(self.data["qr"].get(f, "")).strip():
                    miss.append(f"qr.{f}")
        for p, pc in self.data["parents"].items():
            for f in self.PARENT_FIELDS:
                if not str(pc.get(f, "")).strip():
                    miss.append(f"{p}.{f}")
            if pc.get("area_wise", True):
                for c, cc in pc["children"].items():
                    for f in self.CHILD_FIELDS:
                        if not str(cc.get(f, "")).strip():
                            miss.append(f"{p}/{c}.{f}")
        # scanner (quick-input): nothing to configure — no device needed
        return miss

    def total_fields(self):
        n = len(self.HEAD_FIELDS)
        if self.qr_required():
            n += len(self.QR_FIELDS)
        for p, pc in self.data["parents"].items():
            n += len(self.PARENT_FIELDS)
            if pc.get("area_wise", True):
                n += len(self.CHILD_FIELDS) * len(pc["children"])
        return n

    def progress(self):
        miss = self.missing_fields()
        total = self.total_fields()
        return total - len(miss), total, miss


# ============================================================================
# COMM LOG + ENGINE  (CLIENT wording)
# ============================================================================
class CommLog:
    def __init__(self, maxlen=800):
        self.entries = deque(maxlen=maxlen)
        self.seq = 0
        self._lock = threading.Lock()
        os.makedirs(COMM_LOG_DIR, exist_ok=True)

    def _file(self):
        day = datetime.date.today().isoformat()
        return os.path.join(COMM_LOG_DIR, f"comm_{day}.log")

    def add(self, kind, text):
        ts = datetime.datetime.now().strftime("%H:%M:%S")
        line = f"{ts} | {kind:<3} | {text}"
        with self._lock:
            self.seq += 1
            self.entries.append((self.seq, line))
            try:
                with open(self._file(), "a") as f:
                    f.write(datetime.date.today().isoformat() + " "
                            + line + "\n")
            except OSError:
                pass

    def since(self, last_seq):
        with self._lock:
            return [(s, t) for s, t in self.entries if s > last_seq]


class CommEngine:
    RETRY_S = 3

    def __init__(self, app, settings: GlobalCommSettings, log: CommLog):
        self.app = app
        self.settings = settings
        self.log = log
        self.enabled = False
        self._srv = None
        self._clients = []
        self._lock = threading.Lock()
        self._gen = 0

    def client_count(self):
        with self._lock:
            return len(self._clients)

    def status_text(self):
        if not self.enabled:
            return "communication OFF"
        conn = self.settings.data["connection"]
        hs = self.settings.data["handshake"]
        n = self.client_count()
        cl = f"{n} client connected" if n else "waiting for client…"
        return (f"listening {conn.get('ip')}:{conn.get('port')}  ·  {cl}"
                f"  ·  handshake {'ON' if hs.get('enabled') else 'OFF'}")

    def start(self):
        if self.enabled:
            return
        self.enabled = True
        self._gen += 1
        threading.Thread(target=self._serve, args=(self._gen,),
                         daemon=True).start()
        self.log.add("EVT", "communication ENABLED")

    def stop(self):
        if not self.enabled:
            return
        self.enabled = False
        self._gen += 1
        try:
            if self._srv:
                self._srv.close()
        except OSError:
            pass
        with self._lock:
            for c in self._clients:
                try:
                    c.close()
                except OSError:
                    pass
            self._clients = []
        self._srv = None
        self.log.add("EVT", "communication DISABLED")

    def _serve(self, gen):
        while self.enabled and gen == self._gen:
            conn_cfg = self.settings.data["connection"]
            ip = str(conn_cfg.get("ip", "192.168.0.50")).strip() or "0.0.0.0"
            try:
                port = int(conn_cfg.get("port", 2000))
            except (TypeError, ValueError):
                self.log.add("ERR", "invalid port in settings — comm stopped")
                self.enabled = False
                return
            try:
                srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                srv.bind((ip, port))
                srv.listen(4)
                srv.settimeout(0.5)
            except OSError as e:
                self.log.add("ERR", f"cannot listen on {ip}:{port} ({e}) — "
                                    f"retrying in {self.RETRY_S}s")
                for _ in range(self.RETRY_S * 2):
                    if not (self.enabled and gen == self._gen):
                        return
                    time.sleep(0.5)
                continue
            self._srv = srv
            self.log.add("EVT", f"listening on {ip}:{port}")
            try:
                while self.enabled and gen == self._gen:
                    try:
                        conn, addr = srv.accept()
                    except socket.timeout:
                        continue
                    except OSError:
                        break
                    with self._lock:
                        self._clients.append(conn)
                    self.log.add("EVT",
                                 f"client connected from {addr[0]}:{addr[1]}")
                    threading.Thread(target=self._client_loop,
                                     args=(conn, addr, gen),
                                     daemon=True).start()
            finally:
                try:
                    srv.close()
                except OSError:
                    pass
        self.log.add("EVT", "listener stopped")

    def _client_loop(self, conn, addr, gen):
        buf = b""
        conn.settimeout(0.5)
        while self.enabled and gen == self._gen:
            try:
                chunk = conn.recv(1024)
            except socket.timeout:
                continue
            except OSError:
                break
            if not chunk:
                break
            buf += chunk
            while b"\n" in buf:
                line, buf = buf.split(b"\n", 1)
                self.handle_incoming(line.decode(errors="ignore").strip(),
                                     source=f"CLIENT {addr[0]}")
            if buf and b"\n" not in buf and len(buf) < 512:
                self.handle_incoming(buf.decode(errors="ignore").strip(),
                                     source=f"CLIENT {addr[0]}")
                buf = b""
        with self._lock:
            if conn in self._clients:
                self._clients.remove(conn)
        try:
            conn.close()
        except OSError:
            pass
        self.log.add("EVT", f"client {addr[0]} disconnected "
                            "(will accept reconnection)")

    def handle_incoming(self, text, source="SIM"):
        if not text:
            return
        self.log.add("IN", f"[{source}] '{text}'")
        hs = self.settings.data["handshake"]
        if hs.get("enabled") and text == str(hs.get("ping", "")).strip() and text:
            self.send(hs.get("reply", ""), why="handshake reply")
            return
        scope = self.app.build_trigger_map().get(text)
        if scope is None:
            self.log.add("EVT", f"'{text}' matches no configured signal")
            return
        self.app.after(0, lambda: self.app.dispatch_comm_trigger(scope, source))

    def simulate(self, text):
        self.handle_incoming(text, source="SIM")

    def send(self, signal, why=""):
        signal = str(signal or "").strip()
        if not signal:
            return 0
        data = (signal + "\n").encode()
        n = 0
        with self._lock:
            for c in list(self._clients):
                try:
                    c.sendall(data)
                    n += 1
                except OSError:
                    pass
        tgt = f"-> {n} client(s)" if n else "(no client connected, logged only)"
        self.log.add("OUT", f"'{signal}'  [{why}]  {tgt}")
        return n

    def send_results(self, cfg: CommConfig, scope, results):
        pdata = cfg.data["parents"]
        all_parent_ok = []
        for p, childres in results.items():
            pc = pdata.get(p, {})
            child_oks = []
            for c, r in childres.items():
                child_oks.append(r["ok"])
                if pc.get("area_wise", True):
                    cc = pc.get("children", {}).get(c, {})
                    self.send(cc.get("OK") if r["ok"] else cc.get("NG"),
                              why=f"{p}/{c} {'OK' if r['ok'] else 'NG'}")
            p_ok = bool(child_oks) and all(child_oks)
            all_parent_ok.append(p_ok)
            if pc.get("combine", True):
                self.send(pc.get("OK") if p_ok else pc.get("NG"),
                          why=f"{p} {'OK' if p_ok else 'NG'}")
        if scope and scope[0] == "head":
            h_ok = bool(all_parent_ok) and all(all_parent_ok)
            hc = cfg.data["head"]
            self.send(hc.get("OK") if h_ok else hc.get("NG"),
                      why=f"MODEL {'OK' if h_ok else 'NG'}")


def run_detection(res_call):
    if isinstance(res_call, tuple) and len(res_call) == 3:
        return res_call
    matches, output = res_call
    return matches, output, None


# ============================================================================
# Camera thread
# ============================================================================
class CameraFeed:
    def __init__(self, frame_scale=0.6):
        self.frame_scale = frame_scale
        self._cams = None
        self._latest = None
        self._lock = threading.Lock()
        self._running = False
        self._error = None
        # full-resolution frames for QR reading (the 0.6-scaled preview would
        # throw away exactly the detail a small QR needs).  Kept only while a
        # screen that needs them sets keep_raw, so nothing is copied otherwise.
        self.keep_raw = False
        self._latest_raw = None
        self._raw_seq = 0

    def start(self):
        if self._running:
            return
        self._running = True
        threading.Thread(target=self._loop, daemon=True).start()

    def _loop(self):
        try:
            self._cams = _init_cameras()
            self._cams[0].get_things(exposure=34005, work_mode="Exposure")

        except Exception as e:
            self._error = f"Camera init failed: {e}"
            self._running = False
            return
        while self._running:
            try:
                frame = self._cams[0].grab()
                raw = frame.copy() if self.keep_raw else None
                frame = cv2.resize(frame, None, fx=self.frame_scale,
                                   fy=self.frame_scale)
                with self._lock:
                    self._latest = frame
                    if raw is not None:
                        self._latest_raw = raw
                        self._raw_seq += 1
            except Exception as e:
                self._error = f"Camera grab failed: {e}"
                time.sleep(0.2)

    def get_frame(self):
        with self._lock:
            return None if self._latest is None else self._latest.copy()

    def raw_seq(self):
        with self._lock:
            return self._raw_seq

    def get_raw(self):
        """(seq, full-resolution frame copy).  seq grows with every new frame."""
        with self._lock:
            if self._latest_raw is None:
                return self._raw_seq, None
            return self._raw_seq, self._latest_raw.copy()

    def qr_source(self):
        """Frame source for qr_reader.read_stream(): returns (seq, frame),
        and (seq, None) while nothing new has arrived - so an unchanged frame
        is never copied and never counted twice."""
        self.keep_raw = True
        last = {"seq": None}

        def source():
            seq = self.raw_seq()
            if last["seq"] is not None and seq == last["seq"]:
                return seq, None
            seq, frame = self.get_raw()
            last["seq"] = seq
            return seq, frame
        return source

    def auto_balance(self):
        if self._cams:
            try:
                self._cams[0].get_things(work_mode="auto_balance")
                return True
            except Exception:
                return False
        return False

    @property
    def error(self):
        return self._error


# ============================================================================
# Image composition + pin analysis (pair first, measure after)
# ============================================================================
def banner(width, text, ok=None, height=52):
    color = (70, 60, 40) if ok is None else \
        ((60, 150, 40) if ok else (50, 40, 200))
    img = np.zeros((height, width, 3), np.uint8)
    img[:] = color
    cv2.putText(img, text, (14, height - 17), cv2.FONT_HERSHEY_SIMPLEX,
                0.95, (255, 255, 255), 2, cv2.LINE_AA)
    return img


def label_box(img, x, y, w, h, text, ok):
    color = (80, 220, 80) if ok else (60, 60, 255)
    cv2.rectangle(img, (x, y), (x + w, y + h), color, 3)
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.62, 2)
    ty = y - 8 if y - th - 12 > 0 else y + h + th + 8
    # cv2.rectangle(img, (x, ty - th - 6), (x + tw + 10, ty + 6), color, -1)
    print(text)
    # cv2.putText(img, text, (x + 5, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.62,
    #             (0, 0, 0), 2, cv2.LINE_AA)


def side_by_side(master_img, current_img, left_label="MASTER",
                 right_label="CURRENT"):
    h = max(master_img.shape[0], current_img.shape[0])

    def prep(im, lbl):
        scale = h / im.shape[0]
        im = cv2.resize(im, (int(im.shape[1] * scale), h))
        cv2.rectangle(im, (0, 0), (im.shape[1] - 1, 34), (30, 30, 30), -1)
        cv2.putText(im, lbl, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (80, 220, 255), 2, cv2.LINE_AA)
        return im

    a = prep(master_img.copy(), left_label)
    b = prep(current_img.copy(), right_label)
    divider = np.full((h, 6, 3), 200, np.uint8)
    return np.hstack([a, divider, b])


def overlay_blend(master_img, current_img, alpha=0.5):
    cur = cv2.resize(current_img,
                     (master_img.shape[1], master_img.shape[0]))
    out = cv2.addWeighted(master_img, alpha, cur, 1 - alpha, 0)
    cv2.putText(out, "OVERLAY  master + current", (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (80, 220, 255), 2, cv2.LINE_AA)
    return out


def pair_to_master(master_centers, cur_centers):
    """Globally-nearest one-to-one pairing, NO distance cutoff."""
    if not master_centers or not cur_centers:
        return [None] * len(master_centers)
    m = np.array(master_centers, dtype=np.float32)
    c = np.array(cur_centers, dtype=np.float32)
    d = np.linalg.norm(m[:, None, :] - c[None, :, :], axis=2)
    order = np.dstack(np.unravel_index(np.argsort(d, axis=None), d.shape))[0]
    used_m, used_c = set(), set()
    out = [None] * len(master_centers)
    for mi, ci in order:
        mi, ci = int(mi), int(ci)
        if mi in used_m or ci in used_c:
            continue
        used_m.add(mi)
        used_c.add(ci)
        out[mi] = [int(c[ci][0]), int(c[ci][1])]
        if len(used_m) == len(master_centers) \
                or len(used_c) == len(cur_centers):
            break
    return out


def analyze_pins(master_matches, matches, dist_thr):
    """1) pair (no cutoff)  2) measure distance  3) judge with dist_thr."""
    master_centers = [list(m["center"]) for m in (master_matches or [])]
    cur_centers = []
    for m in matches:
        x1, y1 = m["top_left"]
        x2, y2 = m["bottom_right"]
        cur_centers.append([(x1 + x2) // 2, (y1 + y2) // 2])

    if not master_centers:
        return (len(matches) > 0, [], {"paired": len(matches), "in_range": 0,
                                       "total": 0, "max_dist": 0.0,
                                       "mean_dx": 0.0, "mean_dy": 0.0})
    paired = pair_to_master(master_centers, cur_centers)
    pins = []
    dists, dxs, dys = [], [], []
    for mc, cc in zip(master_centers, paired):
        if cc is None:
            pins.append({"master": tuple(mc), "current": None, "dist": None,
                         "dx": None, "dy": None, "paired": False,
                         "in_range": False})
            continue
        dx = float(cc[0] - mc[0])
        dy = float(cc[1] - mc[1])
        dd = float(np.hypot(dx, dy))
        pins.append({"master": tuple(mc), "current": tuple(cc), "dist": dd,
                     "dx": dx, "dy": dy, "paired": True,
                     "in_range": dd <= float(dist_thr)})
        dists.append(dd); dxs.append(dx); dys.append(dy)
    n_paired = sum(1 for p in pins if p["paired"])
    n_in = sum(1 for p in pins if p["in_range"])
    ok = n_in == len(pins)
    summary = {"paired": n_paired, "in_range": n_in, "total": len(pins),
               "max_dist": max(dists) if dists else 0.0,
               "mean_dx": float(np.mean(dxs)) if dxs else 0.0,
               "mean_dy": float(np.mean(dys)) if dys else 0.0}
    return ok, pins, summary


def draw_pin_overlay(img, pins, off_x, off_y, dist_thr):
    for i, p in enumerate(pins):
        mx, my = int(p["master"][0] + off_x), int(p["master"][1] + off_y)
        cv2.circle(img, (mx, my), 9, (80, 220, 80), 2)
        if p["paired"]:
            cx, cy = int(p["current"][0] + off_x), int(p["current"][1] + off_y)
            col = (80, 220, 80) if p["in_range"] else (60, 60, 255)
            cv2.circle(img, (cx, cy), 4, col, -1)
            if p["dist"] is not None and p["dist"] >= 1.0:
                cv2.line(img, (mx, my), (cx, cy), col, 2, cv2.LINE_AA)
                cv2.putText(img, f"{p['dist']:.0f}px", (cx + 6, cy - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1, cv2.LINE_AA)
        else:
            cv2.circle(img, (mx, my), 12, (60, 60, 255), 2)
            cv2.putText(img, f"MISSING #{i + 1}", (mx + 10, my + 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (60, 60, 255), 1,
                        cv2.LINE_AA)


# ============================================================================
# Qt STYLE HELPERS  (professional dark theme)
# ============================================================================
C_BG = "#1c1c22"
C_PANEL = "#22222a"
C_FIELD = "#15151a"
C_TEXT = "#e2e2e6"
C_MUTED = "#9a9aa5"
C_FAINT = "#6b6b75"

GLOBAL_QSS = f"""
* {{ font-family: 'Segoe UI', 'Noto Sans', sans-serif; }}
QMainWindow, QDialog {{ background: {C_BG}; }}
QWidget {{ color: {C_TEXT}; font-size: 10pt; }}
QLabel {{ background: transparent; }}
QGroupBox {{
    background: {C_PANEL}; border: 1px solid #2e2e38; border-radius: 8px;
    margin-top: 12px; padding: 6px 4px 4px 4px;
    font-weight: 600; color: {C_MUTED}; font-size: 9pt;
}}
QGroupBox::title {{
    subcontrol-origin: margin; left: 10px; padding: 0 4px;
}}
QLineEdit, QComboBox {{
    background: {C_FIELD}; color: white; border: 1px solid #33333d;
    border-radius: 6px; padding: 4px 8px; selection-background-color: #3b6ef5;
}}
QLineEdit:focus, QComboBox:focus {{ border: 1px solid #3b6ef5; }}
QComboBox QAbstractItemView {{
    background: {C_FIELD}; color: white; border: 1px solid #33333d;
    selection-background-color: #3b6ef5;
}}
QTextEdit {{
    background: {C_FIELD}; color: #c8c8d0; border: 1px solid #2b2b34;
    border-radius: 6px; font-family: 'Consolas', 'DejaVu Sans Mono', monospace;
}}
QCheckBox {{ color: #c8c8d0; spacing: 8px; }}
QCheckBox::indicator {{
    width: 16px; height: 16px; border-radius: 4px;
    border: 1px solid #44444e; background: {C_FIELD};
}}
QCheckBox::indicator:checked {{ background: #3b6ef5; border-color: #3b6ef5; }}
QTreeWidget {{
    background: {C_FIELD}; border: 1px solid #2b2b34; border-radius: 6px;
    font-size: 10pt;
}}
QTreeWidget::item {{ height: 26px; }}
QTreeWidget::item:selected {{ background: #2e3a5c; }}
QScrollBar:vertical {{
    background: {C_BG}; width: 10px; margin: 0;
}}
QScrollBar::handle:vertical {{
    background: #3a3a44; border-radius: 5px; min-height: 24px;
}}
QScrollBar::add-line, QScrollBar::sub-line {{ height: 0; }}
QScrollArea {{ border: none; background: transparent; }}
QMessageBox, QInputDialog {{ background: {C_PANEL}; }}
"""


def _shade(hexcolor, f):
    h = hexcolor.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    r, g, b = (max(0, min(255, int(c * f))) for c in (r, g, b))
    return f"#{r:02x}{g:02x}{b:02x}"


def apply_btn_color(btn, color):
    btn.setStyleSheet(
        f"QPushButton {{ background: {color}; color: white; border: none;"
        f" border-radius: 6px; padding: 6px 12px; font-weight: 600;"
        f" font-size: 10pt; }}"
        f" QPushButton:hover {{ background: {_shade(color, 1.18)}; }}"
        f" QPushButton:pressed {{ background: {_shade(color, 0.82)}; }}"
        f" QPushButton:disabled {{ background: #2c2c34; color: #6b6b75; }}")


def styled_btn(master, text, cmd, color="#3b6ef5", **kw):
    """Same signature as the Tkinter version (master kept for call-site
    compatibility; Qt parenting is done by the layouts)."""
    b = QPushButton(text)
    b.clicked.connect(lambda *_: cmd())
    b.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
    # A barcode scanner ends every scan with Enter.  If a button ever holds
    # keyboard focus that Enter *activates the button* — which is how a second
    # scan used to press '← Models' and throw you back to the model list.
    # Buttons here are mouse/touch driven, so they never need focus.
    b.setFocusPolicy(Qt.FocusPolicy.NoFocus)
    b.setAutoDefault(False)
    b.setDefault(False)
    apply_btn_color(b, color)
    return b


def panel(bg=C_PANEL, radius=8):
    w = QFrame()
    w.setStyleSheet(f"QFrame {{ background: {bg}; border-radius: {radius}px; }}"
                    f" QLabel {{ background: transparent; }}")
    return w


def clear_layout(lay):
    while lay.count():
        it = lay.takeAt(0)
        w = it.widget()
        if w is not None:
            w.deleteLater()
        elif it.layout() is not None:
            clear_layout(it.layout())


# -- tiny wrappers so the ported logic can keep tk-var style get()/set() -----
class SVar:
    def __init__(self, edit: QLineEdit):
        self.e = edit

    def get(self):
        return self.e.text()

    def set(self, v):
        self.e.setText(str(v))


class BVar:
    def __init__(self, chk: QCheckBox):
        self.c = chk

    def get(self):
        return self.c.isChecked()

    def set(self, v):
        self.c.setChecked(bool(v))


class FVar(SVar):
    def get(self):
        return float(self.e.text())


class CVar:
    """Combo box of (key, label) items with the same get()/set() style."""

    def __init__(self, combo: QComboBox):
        self.c = combo

    def setup(self, items, current):
        self.c.clear()
        for key, label in items:
            self.c.addItem(label, key)
        self.set(current)

    def get(self):
        return self.c.currentData()

    def set(self, v):
        i = self.c.findData(v)
        if i >= 0:
            self.c.setCurrentIndex(i)


PART_SOURCES = [("scanner", "Barcode scanner (keyboard wedge)"),
                ("camera", "Camera (QR code)")]


class IVar(SVar):
    def get(self):
        return int(float(self.e.text()))


# ============================================================================
# Zoom/pan canvas with drag-rectangle selection  (Qt port of VideoCanvas)
# ============================================================================
class VideoCanvas(QWidget):
    MAX_ZOOM = 12.0

    def __init__(self, master=None):
        super().__init__()
        self.setMinimumSize(600, 360)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self._last_bgr = None
        self._qimg = None
        self._draw_pos = (0.0, 0.0)
        self._S = 1.0
        self._off = (0.0, 0.0)
        self._src_size = (1, 1)
        self.zoom = 1.0
        self.pan = [0.0, 0.0]
        self.select_enabled = False
        self._drag_start = None
        self.pending_rect = None
        self.on_rect_drawn = None
        self._pan_start = None
        self._message = None

    def _canvas_size(self):
        cw = max(self.width(), 100)
        ch = max(self.height(), 100)
        return cw, ch

    def resizeEvent(self, e):
        super().resizeEvent(e)
        self.refresh()

    # -- display -------------------------------------------------------------
    def show(self, bgr):
        if bgr is None:
            return
        self._message = None
        self._last_bgr = bgr
        h, w = bgr.shape[:2]
        self._src_size = (w, h)
        cw, ch = self._canvas_size()
        s0 = min(cw / w, ch / h)
        S = s0 * self.zoom
        off_x = (cw - w * S) / 2 + self.pan[0]
        off_y = (ch - h * S) / 2 + self.pan[1]
        self._S = S
        self._off = (off_x, off_y)
        sx0 = max(0, int((0 - off_x) / S))
        sy0 = max(0, int((0 - off_y) / S))
        sx1 = min(w, int((cw - off_x) / S) + 2)
        sy1 = min(h, int((ch - off_y) / S) + 2)
        self._qimg = None
        if sx1 > sx0 and sy1 > sy0:
            crop = bgr[sy0:sy1, sx0:sx1]
            dw = max(1, int(round((sx1 - sx0) * S)))
            dh = max(1, int(round((sy1 - sy0) * S)))
            disp = cv2.resize(crop, (dw, dh),
                              interpolation=cv2.INTER_AREA if S < 1
                              else cv2.INTER_NEAREST if S > 3
                              else cv2.INTER_LINEAR)
            rgb = cv2.cvtColor(np.ascontiguousarray(disp), cv2.COLOR_BGR2RGB)
            hh, ww = rgb.shape[:2]
            self._qimg = QImage(rgb.data, ww, hh, 3 * ww,
                                QImage.Format.Format_RGB888).copy()
            self._draw_pos = (off_x + sx0 * S, off_y + sy0 * S)
        self.update()

    def show_message(self, text):
        """Equivalent of the Tk canvas delete('all') + centered error text."""
        self._qimg = None
        self._message = text
        self.update()

    def refresh(self):
        if self._last_bgr is not None:
            self.show(self._last_bgr)

    def paintEvent(self, _e):
        p = QPainter(self)
        p.fillRect(self.rect(), QColor("#111111"))
        if self._qimg is not None:
            p.drawImage(int(self._draw_pos[0]), int(self._draw_pos[1]),
                        self._qimg)
        if self.pending_rect:
            x, y, w, h = self.pending_rect
            x0, y0 = self._to_canvas(x, y)
            x1, y1 = self._to_canvas(x + w, y + h)
            pen = QPen(QColor("#ffcc00"))
            pen.setWidth(2)
            p.setPen(pen)
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.drawRect(QRectF(x0, y0, x1 - x0, y1 - y0))
        if self.zoom > 1.001:
            cw, ch = self._canvas_size()
            p.setPen(QColor("#ffd43b"))
            f = QFont("Segoe UI", 9)
            f.setBold(True)
            p.setFont(f)
            p.drawText(8, ch - 10, f"zoom ×{self.zoom:.1f}")
        if self._message:
            p.setPen(QColor("#ff6b6b"))
            p.setFont(QFont("Segoe UI", 12))
            p.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter,
                       self._message)
        p.end()

    # -- zoom / pan ----------------------------------------------------------
    def _zoom_at(self, cx, cy, factor):
        old = self.zoom
        self.zoom = max(1.0, min(self.MAX_ZOOM, self.zoom * factor))
        if abs(self.zoom - old) < 1e-6:
            return
        sx, sy = self._to_src(cx, cy)
        w, h = self._src_size
        cw, ch = self._canvas_size()
        s0 = min(cw / w, ch / h)
        S = s0 * self.zoom
        self.pan[0] = cx - sx * S - (cw - w * S) / 2
        self.pan[1] = cy - sy * S - (ch - h * S) / 2
        if self.zoom <= 1.001:
            self.pan = [0.0, 0.0]
        self.refresh()

    def wheelEvent(self, e):
        pos = e.position()
        self._zoom_at(pos.x(), pos.y(),
                      1.15 if e.angleDelta().y() > 0 else 1 / 1.15)

    def zoom_in(self):
        cw, ch = self._canvas_size()
        self._zoom_at(cw // 2, ch // 2, 1.3)

    def zoom_out(self):
        cw, ch = self._canvas_size()
        self._zoom_at(cw // 2, ch // 2, 1 / 1.3)

    def fit(self):
        self.zoom = 1.0
        self.pan = [0.0, 0.0]
        self.refresh()

    # -- coordinate mapping --------------------------------------------------
    def _to_src(self, cx, cy):
        x = (cx - self._off[0]) / self._S
        y = (cy - self._off[1]) / self._S
        return (max(0, min(self._src_size[0] - 1, x)),
                max(0, min(self._src_size[1] - 1, y)))

    def _to_canvas(self, sx, sy):
        return sx * self._S + self._off[0], sy * self._S + self._off[1]

    # -- mouse: left = select, right/middle = pan ----------------------------
    def mousePressEvent(self, e):
        pos = e.position()
        if e.button() == Qt.MouseButton.LeftButton:
            if self.select_enabled:
                self._drag_start = self._to_src(pos.x(), pos.y())
                self.pending_rect = None
                self.update()
        elif e.button() in (Qt.MouseButton.RightButton,
                            Qt.MouseButton.MiddleButton):
            self._pan_start = (pos.x(), pos.y(), self.pan[0], self.pan[1])

    def mouseMoveEvent(self, e):
        pos = e.position()
        if (e.buttons() & Qt.MouseButton.LeftButton
                and self.select_enabled and self._drag_start is not None):
            x0, y0 = self._drag_start
            x1, y1 = self._to_src(pos.x(), pos.y())
            self.pending_rect = (int(min(x0, x1)), int(min(y0, y1)),
                                 int(abs(x1 - x0)), int(abs(y1 - y0)))
            self.update()
        elif (e.buttons() & (Qt.MouseButton.RightButton
                             | Qt.MouseButton.MiddleButton)
              and self._pan_start is not None):
            if self.zoom <= 1.001:
                return
            x0, y0, px, py = self._pan_start
            self.pan[0] = px + (pos.x() - x0)
            self.pan[1] = py + (pos.y() - y0)
            self.refresh()

    def mouseReleaseEvent(self, e):
        if e.button() == Qt.MouseButton.LeftButton:
            if not self.select_enabled or self._drag_start is None:
                return
            pos = e.position()
            x0, y0 = self._drag_start
            x1, y1 = self._to_src(pos.x(), pos.y())
            self.pending_rect = (int(min(x0, x1)), int(min(y0, y1)),
                                 int(abs(x1 - x0)), int(abs(y1 - y0)))
            self._drag_start = None
            if self.pending_rect and self.pending_rect[2] > 4 \
                    and self.pending_rect[3] > 4:
                if self.on_rect_drawn:
                    self.on_rect_drawn(self.pending_rect)
            else:
                self.pending_rect = None
            self.update()
        elif e.button() in (Qt.MouseButton.RightButton,
                            Qt.MouseButton.MiddleButton):
            self._pan_start = None

    def clear_rect(self):
        self.pending_rect = None
        self._drag_start = None
        self.update()


def zoom_controls(master, canvas):
    row = QWidget()
    lay = QHBoxLayout(row)
    lay.setContentsMargins(0, 0, 0, 0)
    lay.setSpacing(4)
    lay.addStretch()
    lay.addWidget(styled_btn(row, "－", canvas.zoom_out, color="#44444e"))
    lay.addWidget(styled_btn(row, "Fit", canvas.fit, color="#44444e"))
    lay.addWidget(styled_btn(row, "＋", canvas.zoom_in, color="#44444e"))
    hint = QLabel("  wheel = zoom · right-drag = pan")
    hint.setStyleSheet(f"color: {C_FAINT}; font-size: 8pt;")
    lay.addWidget(hint)
    lay.addStretch()
    return row


# ============================================================================
# MAIN APP  (global comm engine + lock management)
# ============================================================================
class App(QMainWindow):
    _invoke = pyqtSignal(int, object)     # cross-thread 'after' (ms, fn)

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Inspection")
        self.resize(1420, 830)
        self.app_config = AppConfig()
        self.heads = HeadManager()
        self.camera = CameraFeed(
            frame_scale=float(self.app_config.data["ui"]
                              .get("frame_scale", 0.6)))
        self.comm_settings = GlobalCommSettings()
        self.comm_log = CommLog()
        self.engine = CommEngine(self, self.comm_settings, self.comm_log)

        self._invoke.connect(self._on_invoke)
        self.purge_now()                       # 2-day buffer: at start-up…
        self._purge_timer = QTimer(self)       # …and then every hour
        self._purge_timer.timeout.connect(self.purge_now)
        self._purge_timer.start(3600 * 1000)

        self.container = QWidget()
        self.container_lay = QVBoxLayout(self.container)
        self.container_lay.setContentsMargins(0, 0, 0, 0)
        self.setCentralWidget(self.container)
        self.current_screen = None
        self._nav = ("home",)
        self._operator_head = None
        self._operator_id = None
        self.show_home()

    def purge_now(self):
        days = self.app_config.data.get("retention", {}).get("days", 2)

        def work():
            n = purge_old_data(days)
            if n:
                self.comm_log.add("EVT", f"data buffer: {n} file(s) older "
                                         f"than {days} day(s) deleted")
        threading.Thread(target=work, daemon=True).start()

    # tk-compatible after(): safe to call from ANY thread (engine/workers)
    def after(self, ms, fn):
        self._invoke.emit(int(ms), fn)

    def _on_invoke(self, ms, fn):
        if ms <= 0:
            fn()
        else:
            QTimer.singleShot(ms, fn)

    def closeEvent(self, e):
        self.engine.stop()
        e.accept()

    # -- lock / controls -----------------------------------------------------
    def is_locked(self):
        return bool(self.app_config.data.get("locked", False))

    def lock(self):
        self.app_config.data["locked"] = True
        self.app_config.save()
        self.comm_log.add("EVT", "UI LOCKED (operator mode)")
        self.refresh_screen()

    def unlock_with_password(self, parent):
        pw, ok = QInputDialog.getText(parent, "Controls", "Enter password:",
                                      QLineEdit.EchoMode.Password)
        if not ok:
            return
        if pw == str(self.app_config.data.get("password", "1234")):
            self.app_config.data["locked"] = False
            self.app_config.save()
            self.comm_log.add("EVT", "UI UNLOCKED (controls open)")
            self.refresh_screen()
        else:
            QMessageBox.critical(parent, "Wrong password",
                                 "Incorrect password.")

    def lock_button(self, master):
        """Shared 'Controls'/'Lock' button for the top bars."""
        if self.is_locked():
            return styled_btn(master, "🔑 Controls",
                              lambda: self.unlock_with_password(master),
                              color="#f59f00")
        return styled_btn(master, "🔒 Lock", self.lock, color="#44444e")

    # -- operator session ----------------------------------------------------
    # The Operator ID is tied to *selecting* a model, not to building the
    # HeadScreen widget.  Going into scene creation / comm config / scanner
    # config and coming back rebuilds HeadScreen many times — those must NOT
    # re-ask.  Only leaving to the model list (or picking a different model)
    # ends the session.
    def get_operator(self, head_name):
        if self._operator_head == head_name:
            return self._operator_id
        return None

    def set_operator(self, head_name, operator_id):
        self._operator_head = head_name
        self._operator_id = operator_id

    def clear_operator(self):
        self._operator_head = None
        self._operator_id = None

    # -- navigation ----------------------------------------------------------
    def _swap(self, screen_cls, *args):
        if self.current_screen is not None:
            self.current_screen.destroy_screen()
        self.current_screen = screen_cls(self.container, self, *args)
        self.container_lay.addWidget(self.current_screen)

    def refresh_screen(self):
        if self._nav[0] == "home":
            self.show_home()
        elif self._nav[0] == "head":
            self.show_head(self._nav[1], announce=False)
        elif self._nav[0] == "comm":
            self.show_comm(self._nav[1])
        elif self._nav[0] == "build":
            self.show_build_parent(self._nav[1], self._nav[2])

    def show_home(self):
        self._nav = ("home",)
        self.clear_operator()          # back at the model list -> logged out
        self._swap(HomeScreen)

    def show_head(self, head_name, announce=True):
        if self._operator_head not in (None, head_name):
            self.clear_operator()      # a different model -> new operator
        self._nav = ("head", head_name)
        self._swap(HeadScreen, head_name)
        if not announce:
            return
        sel = CommConfig(self.heads, head_name).data["head"].get("SELECT", "")
        if not str(sel).strip():
            self.comm_log.add("EVT", f"model '{head_name}' selected in UI "
                                     "(no Select-signal configured)")
        elif self.engine.enabled:
            self.engine.send(sel, why=f"model '{head_name}' selected in UI")
        else:
            self.comm_log.add("EVT", f"model '{head_name}' selected in UI "
                                     f"(comm OFF — '{sel}' not sent)")
       
    def show_build_parent(self, head_name, parent_name):
        self._nav = ("build", head_name, parent_name)
        self._swap(BuildParentScreen, head_name, parent_name)

    def show_comm(self, head_name):
        self._nav = ("comm", head_name)
        self._swap(CommScreen, head_name)

    # -- global trigger map --------------------------------------------------
    def build_trigger_map(self):
        m = {}

        def put(text, scope, where):
            text = str(text or "").strip()
            if not text:
                return
            if text in m and m[text] != scope:
                self.comm_log.add("ERR", f"duplicate signal '{text}' "
                                         f"({where}) ignored — already used")
                return
            m[text] = scope

        for h in self.heads.list_heads():
            cfg = CommConfig(self.heads, h)
            put(cfg.data["head"].get("SELECT"), ("select", h), f"{h}.SELECT")
            put(cfg.data["head"].get("PLC"), ("run", h, ("head",)), f"{h}.run")
            if self.heads.camera_qr_enabled(h):
                put(cfg.data["qr"].get("TRIGGER"), ("qr", h),
                    f"{h}.qr_trigger")
            for p, pc in cfg.data["parents"].items():
                put(pc.get("PLC"), ("run", h, ("parent", p)), f"{h}/{p}")
                for c, cc in pc["children"].items():
                    put(cc.get("PLC"), ("run", h, ("child", p, c)),
                        f"{h}/{p}/{c}")
        return m

    def dispatch_comm_trigger(self, scope, source):
        if scope[0] == "select":
            # SELECTION ONLY — never starts any detection run
            head = scope[1]
            already = (isinstance(self.current_screen, HeadScreen)
                       and self.current_screen.head == head)
            self.comm_log.add("EVT", f"client selected model '{head}'"
                                     + (" (already open)" if already else ""))
            if not already:
                self.show_head(head, announce=False)
            sel = CommConfig(self.heads, head).data["head"].get("SELECT", "")
            self.engine.send(sel, why=f"confirm model '{head}' selected")
            return
        if scope[0] == "run":
            head, run_scope = scope[1], scope[2]
            if not (isinstance(self.current_screen, HeadScreen)
                    and self.current_screen.head == head):
                self.comm_log.add(
                    "EVT", f"run trigger {run_scope} for model '{head}' "
                           "IGNORED — model not selected "
                           "(client must send the Select-signal first)")
                return
            self.current_screen.execute(run_scope, source)
            return
        if scope[0] == "qr":
            # the client says: part is in position -> read the QR by camera
            head = scope[1]
            if not (isinstance(self.current_screen, HeadScreen)
                    and self.current_screen.head == head):
                self.comm_log.add(
                    "EVT", f"QR trigger for model '{head}' IGNORED — model "
                           "not selected (client must send the "
                           "Select-signal first)")
                return
            self.current_screen.qr_client_trigger(source)


class BaseScreen(QWidget):
    def __init__(self, master, app):
        super().__init__()
        self.app = app
        self._alive = True

    def after(self, ms, fn):
        # tk-compatible; safe from worker threads (routes to the UI thread)
        self.app.after(ms, fn)

    def destroy_screen(self):
        self._alive = False
        self.setParent(None)
        self.deleteLater()


# ============================================================================
# COMM BAR + MONITOR
# ============================================================================
class CommBar(QFrame):
    def __init__(self, master, app, mini_log_lines=4):
        super().__init__()
        self.app = app
        self._alive = True
        self._last_seq = 0
        self.setStyleSheet(f"QFrame {{ background: {C_PANEL};"
                           f" border-radius: 8px; }}"
                           f" QLabel {{ background: transparent; }}")
        lay = QVBoxLayout(self)
        lay.setContentsMargins(8, 6, 8, 6)
        lay.setSpacing(4)

        row = QHBoxLayout()
        self.lamp = QLabel("●")
        self.lamp.setStyleSheet("color: #e03131; font-size: 14pt;"
                                " font-weight: bold;")
        row.addWidget(self.lamp)
        self.status_lbl = QLabel("communication OFF")
        self.status_lbl.setStyleSheet("color: #c8c8d0; font-size: 9pt;")
        row.addWidget(self.status_lbl)
        row.addStretch()
        row.addWidget(styled_btn(self, "🖥 Monitor",
                                 lambda: CommMonitor(self, app).show(),
                                 color="#3b6ef5"))
        self.btn_hs = styled_btn(self, "Handshake", self.toggle_handshake,
                                 color="#44444e")
        row.addWidget(self.btn_hs)
        self.btn_toggle = styled_btn(self, "Comm ON", self.toggle_comm,
                                     color="#2f9e44")
        row.addWidget(self.btn_toggle)
        lay.addLayout(row)

        self.mini = QTextEdit()
        self.mini.setReadOnly(True)
        self.mini.setStyleSheet(
            f"QTextEdit {{ background: {C_FIELD}; color: #8f8f9a;"
            f" border: none; border-radius: 6px; font-size: 8pt; }}")
        self.mini.setFixedHeight(mini_log_lines * 16 + 10)
        lay.addWidget(self.mini)

        sim = QHBoxLayout()
        lbl = QLabel("Simulate CLIENT:")
        lbl.setStyleSheet("color: #c8c8d0; font-size: 9pt;")
        sim.addWidget(lbl)
        self.sim_entry = QLineEdit()
        self.sim_entry.returnPressed.connect(self.simulate)
        sim.addWidget(self.sim_entry, stretch=1)
        sim.addWidget(styled_btn(self, "Send ▶", self.simulate,
                                 color="#3b6ef5"))
        lay.addLayout(sim)

        self._timer = QTimer(self)
        self._timer.timeout.connect(self._poll)
        self._timer.start(400)
        self._refresh_now()

    def simulate(self):
        text = self.sim_entry.text().strip()
        if not text:
            return
        self.sim_entry.clear()
        self.app.engine.simulate(text)

    def toggle_comm(self):
        if self.app.engine.enabled:
            self.app.engine.stop()
        else:
            self.app.engine.start()
        self._refresh_now()

    def toggle_handshake(self):
        hs = self.app.comm_settings.data["handshake"]
        hs["enabled"] = not hs.get("enabled", True)
        self.app.comm_settings.save()
        self.app.comm_log.add("EVT", "handshake "
                              + ("ENABLED" if hs["enabled"] else "DISABLED"))
        self._refresh_now()

    def _refresh_now(self):
        eng = self.app.engine
        self.lamp.setStyleSheet(
            f"color: {'#2f9e44' if eng.enabled else '#e03131'};"
            " font-size: 14pt; font-weight: bold;")
        self.status_lbl.setText(eng.status_text())
        self.btn_toggle.setText("Comm OFF" if eng.enabled else "Comm ON")
        apply_btn_color(self.btn_toggle,
                        "#e03131" if eng.enabled else "#2f9e44")
        hs_on = self.app.comm_settings.data["handshake"].get("enabled", True)
        self.btn_hs.setText(f"Handshake {'ON' if hs_on else 'OFF'}")
        apply_btn_color(self.btn_hs, "#845ef7" if hs_on else "#44444e")

    def _poll(self):
        if not self._alive:
            return
        self._refresh_now()
        new = self.app.comm_log.since(self._last_seq)
        if new:
            self._last_seq = new[-1][0]
            for _, line in new:
                self.mini.append(html.escape(line, quote=False))
            self.mini.verticalScrollBar().setValue(
                self.mini.verticalScrollBar().maximum())


class CommMonitor(QDialog):
    COLORS = {"IN": "#74c0fc", "OUT": "#69db7c",
              "ERR": "#ff8787", "EVT": "#9a9aa5"}

    def __init__(self, master, app):
        super().__init__(master)
        self.app = app
        self._alive = True
        self._last_seq = 0
        self.setWindowTitle("Communication Monitor")
        self.resize(760, 640)
        lay = QVBoxLayout(self)

        box = QGroupBox(" Connection & Handshake (global) ")
        grid = QGridLayout()
        s = app.comm_settings.data
        self.vars = {}
        fields = [("Listen IP", "ip", s["connection"].get("ip", "0.0.0.0")),
                  ("TCP Port", "port", s["connection"].get("port", 5000)),
                  ("Handshake ping (client→us)", "ping",
                   s["handshake"].get("ping", "PING")),
                  ("Handshake reply (us→client)", "reply",
                   s["handshake"].get("reply", "PONG"))]
        for i, (lbl, key, val) in enumerate(fields):
            l = QLabel(lbl)
            l.setStyleSheet("color: #c8c8d0; font-size: 9pt;")
            grid.addWidget(l, i // 2, (i % 2) * 2)
            e = QLineEdit(str(val))
            e.setFixedWidth(160)
            self.vars[key] = SVar(e)
            grid.addWidget(e, i // 2, (i % 2) * 2 + 1)
        boxlay = QVBoxLayout(box)
        boxlay.addLayout(grid)
        savew = QHBoxLayout()
        savew.addStretch()
        savew.addWidget(styled_btn(self, "💾 Save settings",
                                   self.save_settings, color="#2f9e44"))
        savew.addStretch()
        boxlay.addLayout(savew)
        lay.addWidget(box)

        t = QLabel("ALL CLIENT TRAFFIC  (IN = client sent · OUT = we sent)")
        t.setStyleSheet(f"color: {C_MUTED}; font-weight: bold;")
        t.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lay.addWidget(t)

        self.txt = QTextEdit()
        self.txt.setReadOnly(True)
        lay.addWidget(self.txt, stretch=1)

        sim = QHBoxLayout()
        sl = QLabel("Simulate CLIENT:")
        sl.setStyleSheet("color: #c8c8d0; font-size: 9pt;")
        sim.addWidget(sl)
        self.sim_entry = QLineEdit()
        self.sim_entry.returnPressed.connect(self.simulate)
        sim.addWidget(self.sim_entry, stretch=1)
        sim.addWidget(styled_btn(self, "Send ▶", self.simulate))
        lay.addLayout(sim)

        pl = QLabel(f"Log files: {os.path.abspath(COMM_LOG_DIR)}"
                    "  (one file per day)")
        pl.setStyleSheet(f"color: {C_FAINT}; font-size: 8pt;")
        pl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lay.addWidget(pl)

        self._append(self.app.comm_log.since(0))
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._poll)
        self._timer.start(300)

    def closeEvent(self, e):
        self._alive = False
        e.accept()

    def save_settings(self):
        try:
            port = int(self.vars["port"].get().strip())
        except ValueError:
            QMessageBox.critical(self, "Bad port",
                                 "TCP port must be a number.")
            return
        s = self.app.comm_settings.data
        s["connection"]["ip"] = self.vars["ip"].get().strip() or "0.0.0.0"
        s["connection"]["port"] = port
        s["handshake"]["ping"] = self.vars["ping"].get().strip()
        s["handshake"]["reply"] = self.vars["reply"].get().strip()
        self.app.comm_settings.save()
        self.app.comm_log.add("EVT", "settings saved "
                              f"({s['connection']['ip']}:{port})")
        if self.app.engine.enabled:
            self.app.engine.stop()
            self.app.engine.start()

    def simulate(self):
        text = self.sim_entry.text().strip()
        if not text:
            return
        self.sim_entry.clear()
        self.app.engine.simulate(text)

    def _append(self, entries):
        if not entries:
            return
        self._last_seq = entries[-1][0]
        for _, line in entries:
            kind = line.split("|")[1].strip() if "|" in line else "EVT"
            col = self.COLORS.get(kind, "#c8c8d0")
            self.txt.append(f'<span style="color:{col}">'
                            f'{html.escape(line, quote=False)}</span>')
        self.txt.verticalScrollBar().setValue(
            self.txt.verticalScrollBar().maximum())

    def _poll(self):
        if not self._alive:
            return
        self._append(self.app.comm_log.since(self._last_seq))


# ============================================================================
# SCANNER CONFIG DIALOG  (quick-input scan box — no device to pick)
# ============================================================================
class ScannerDialog(QDialog):
    def __init__(self, master, app, head_name, on_close=None):
        super().__init__(master)
        self.app = app
        self.head = head_name
        self.on_close = on_close
        self._alive = True
        self.setWindowTitle(f"Barcode Scanner — {head_name}")
        self.resize(560, 520)

        cfg = app.heads.get_scanner(head_name)
        lay = QVBoxLayout(self)
        title = QLabel(f"BARCODE SCANNER  —  model '{head_name}'")
        title.setStyleSheet("color: white; font-size: 13pt;"
                            " font-weight: bold;")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lay.addWidget(title)
        sub = QLabel("The scanner step runs BEFORE the scenes: each cycle "
                     "first needs a scan\nbefore detection can run (sending "
                     "to the client is attempted on scan).")
        sub.setStyleSheet(f"color: {C_MUTED}; font-size: 9pt;")
        sub.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lay.addWidget(sub)

        box = panel()
        boxlay = QVBoxLayout(box)
        self.var_enabled = BVar(QCheckBox(
            "Enable barcode scanner for this model"))
        self.var_enabled.c.setStyleSheet("font-weight: bold;")
        self.var_enabled.set(bool(cfg.get("enabled")))
        boxlay.addWidget(self.var_enabled.c)

        how = QLabel("No device to configure.  A scan opens a floating "
                     "input box on top of\neverything that holds the "
                     "keyboard: the cursor stays in it and nothing\nelse "
                     "can be typed into until Enter.  Any keyboard-wedge "
                     "scanner works.")
        how.setStyleSheet(f"color: {C_FAINT}; font-size: 8pt;")
        boxlay.addWidget(how)

        srow = QHBoxLayout()
        sl = QLabel("Part scan source:")
        sl.setStyleSheet("color: #c8c8d0;")
        srow.addWidget(sl)
        self.var_source = CVar(QComboBox())
        self.var_source.setup(PART_SOURCES, cfg.get("source", "scanner"))
        srow.addWidget(self.var_source.c, stretch=1)
        boxlay.addLayout(srow)
        srcn = QLabel("Operator ID is ALWAYS scanned with the scanner.  With\n"
                      "'Camera' the PART is read by the camera: add QR scanners\n"
                      "while building a scene (5. QR Scanner) and mark ONE as\n"
                      "master.  After Finish Model, a  📷 SCAN QR  button starts\n"
                      "every cycle (set its client signals in Comm Config).")
        srcn.setStyleSheet(f"color: {C_FAINT}; font-size: 8pt;")
        boxlay.addWidget(srcn)

        row2 = QHBoxLayout()
        pl = QLabel("Prefix for client:")
        pl.setStyleSheet("color: #c8c8d0;")
        row2.addWidget(pl)
        pe = QLineEdit(cfg.get("prefix", ""))
        pe.setFixedWidth(180)
        self.var_prefix = SVar(pe)
        row2.addWidget(pe)
        row2.addStretch()
        boxlay.addLayout(row2)
        hint = QLabel("what we send = prefix + barcode "
                      "(e.g. 'BC:' -> 'BC:12345678')")
        hint.setStyleSheet(f"color: {C_FAINT}; font-size: 8pt;")
        boxlay.addWidget(hint)

        self.var_send = BVar(QCheckBox(
            "Send barcode to client (auto-sent on scan; "
            "runs continue even if no client yet)"))
        self.var_send.set(bool(cfg.get("send_to_client", True)))
        boxlay.addWidget(self.var_send.c)

        self.var_auto = BVar(QCheckBox(
            "Pop the scan box up automatically at the start of every cycle"))
        self.var_auto.set(bool(cfg.get("auto_popup", True)))
        boxlay.addWidget(self.var_auto.c)

        self.var_req_op = BVar(QCheckBox(
            "Require Operator ID scan when model is opened"))
        self.var_req_op.set(bool(cfg.get("require_operator", False)))
        boxlay.addWidget(self.var_req_op.c)

        trow = QHBoxLayout()
        self.btn_test = styled_btn(self, "▶ Test scan", self.start_test,
                                   color="#f59f00")
        trow.addWidget(self.btn_test)
        self.test_lbl = QLabel("press Test — the scan box opens, then scan")
        self.test_lbl.setStyleSheet(f"color: {C_MUTED}; font-size: 9pt;")
        self.test_lbl.setWordWrap(True)
        trow.addWidget(self.test_lbl, stretch=1)
        boxlay.addLayout(trow)
        lay.addWidget(box, stretch=1)

        brow = QHBoxLayout()
        brow.addStretch()
        brow.addWidget(styled_btn(self, "💾 Save", self.save,
                                  color="#2f9e44"))
        brow.addWidget(styled_btn(self, "Close", self._close,
                                  color="#44444e"))
        brow.addStretch()
        lay.addLayout(brow)

    # ------------------------------------------------------------------
    def start_test(self):
        timeout = int(self.app.app_config.data["scanner"]
                      .get("test_timeout_s", 10))
        code = QuickScanInput.ask(
            self, prompt="TEST SCAN",
            sub="scan a barcode now — the keyboard is locked to this box",
            timeout_s=timeout)
        if code:
            self.test_lbl.setText(f"✔ scanned:  '{code}'")
            self.test_lbl.setStyleSheet("color: #69db7c; font-size: 9pt;")
        else:
            self.test_lbl.setText("✖ nothing scanned (cancelled / timeout)")
            self.test_lbl.setStyleSheet("color: #ff8787; font-size: 9pt;")

    def save(self):
        self.app.heads.set_scanner(
            self.head,
            enabled=bool(self.var_enabled.get()),
            prefix=self.var_prefix.get(),
            send_to_client=bool(self.var_send.get()),
            auto_popup=bool(self.var_auto.get()),
            require_operator=bool(self.var_req_op.get()),
            source=self.var_source.get() or "scanner")
        self.app.comm_log.add("EVT",
                              f"scanner config saved for model '{self.head}' "
                              f"(enabled={self.var_enabled.get()}, "
                              f"source={self.var_source.get()}, "
                              f"auto_popup={self.var_auto.get()}, "
                              f"require_operator={self.var_req_op.get()})")

    def _close(self):
        self._alive = False
        cb, self.on_close = self.on_close, None
        self.close()
        if cb:
            cb()

    def closeEvent(self, e):
        self._alive = False
        cb, self.on_close = self.on_close, None
        e.accept()
        if cb:
            cb()


# ============================================================================
# MES CONFIG DIALOG  (available once the model is finished)
# ============================================================================
class MesDialog(QDialog):
    FIELDS = [
        ("mes_ip",           "MES IP / host:",      False),
        ("username",         "Username:",           False),
        ("password",         "Password:",           True),
        ("login_stage_name", "Login stage name:",   False),
        ("stage_id",         "Stage ID:",           False),
        ("stage_name",       "Stage name:",         False),
        ("product_id",       "Product ID:",         False),
        ("product_name",     "Product name:",       False),
        ("operator_id",      "Operator ID:",        False),
        ("created_by",       "Created by:",         False),
        ("machine_no",       "Machine no:",         False),
        ("defect_type",      "Defect type (NG):",   False),
        ("file_format",      "Report file format:", False),
    ]

    def __init__(self, master, app, head_name, on_close=None):
        super().__init__(master)
        self.app = app
        self.head = head_name
        self.on_close = on_close
        self.setWindowTitle(f"MES Integration — {head_name}")
        self.resize(600, 700)

        cfg = app.heads.get_mes(head_name)
        lay = QVBoxLayout(self)

        title = QLabel(f"MES INTEGRATION  —  model '{head_name}'")
        title.setStyleSheet("color: white; font-size: 13pt;"
                            " font-weight: bold;")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lay.addWidget(title)

        if not MES_AVAILABLE:
            warn = QLabel("⚠ mes_client could not be imported "
                          f"({MES_IMPORT_ERR}).\nPut mes_client.py next to "
                          "this file and install 'requests'.")
            warn.setStyleSheet("color: #ff8787; font-size: 9pt;")
            warn.setWordWrap(True)
            lay.addWidget(warn)

        self.var_enabled = BVar(QCheckBox(
            "Enable MES integration for this model"))
        self.var_enabled.c.setStyleSheet("font-weight: bold;")
        self.var_enabled.set(bool(cfg.get("enabled")))
        lay.addWidget(self.var_enabled.c)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        inner = panel()
        form = QVBoxLayout(inner)
        self.vars = {}
        for key, label, is_pw in self.FIELDS:
            row = QHBoxLayout()
            l = QLabel(label)
            l.setStyleSheet("color: #c8c8d0;")
            l.setFixedWidth(160)
            row.addWidget(l)
            e = QLineEdit(str(cfg.get(key, "")))
            if is_pw:
                e.setEchoMode(QLineEdit.EchoMode.Password)
            e.setStyleSheet(f"background: {C_FIELD}; color: white;"
                            " padding: 5px; border-radius: 4px;")
            self.vars[key] = SVar(e)
            row.addWidget(e, stretch=1)
            form.addLayout(row)
        scroll.setWidget(inner)
        lay.addWidget(scroll, stretch=1)

        self.var_block = BVar(QCheckBox(
            "Block the run when the previous stage did NOT pass "
            "(applies in 🔒 locked / production mode)"))
        self.var_block.set(bool(cfg.get("block_on_prev_fail", True)))
        lay.addWidget(self.var_block.c)

        self.var_imgs = BVar(QCheckBox(
            "Upload the result images to MES at the end of every cycle"))
        self.var_imgs.set(bool(cfg.get("upload_images", True)))
        lay.addWidget(self.var_imgs.c)

        note = QLabel("Logs are written to:  "
                      + app.heads.mes_log_dir(head_name))
        note.setStyleSheet(f"color: {C_FAINT}; font-size: 8pt;")
        note.setWordWrap(True)
        lay.addWidget(note)

        self.test_lbl = QLabel("press Test login to verify the connection")
        self.test_lbl.setStyleSheet(f"color: {C_MUTED}; font-size: 9pt;")
        self.test_lbl.setWordWrap(True)
        lay.addWidget(self.test_lbl)

        brow = QHBoxLayout()
        brow.addStretch()
        brow.addWidget(styled_btn(self, "🔌 Test login", self.test_login,
                                  color="#f59f00"))
        brow.addWidget(styled_btn(self, "💾 Save", self.save,
                                  color="#2f9e44"))
        brow.addWidget(styled_btn(self, "Close", self._close,
                                  color="#44444e"))
        brow.addStretch()
        lay.addLayout(brow)

    # ------------------------------------------------------------------
    def _collect(self):
        d = {k: self.vars[k].get().strip() for k, _, _ in self.FIELDS}
        d["enabled"] = bool(self.var_enabled.get())
        d["block_on_prev_fail"] = bool(self.var_block.get())
        d["upload_images"] = bool(self.var_imgs.get())
        return d

    def test_login(self):
        if not MES_AVAILABLE:
            self.test_lbl.setText(f"✖ mes_client unavailable "
                                  f"({MES_IMPORT_ERR})")
            self.test_lbl.setStyleSheet("color: #ff8787; font-size: 9pt;")
            return
        cfg = self._collect()
        cfg["results_dir"] = self.app.heads.mes_log_dir(self.head)
        self.test_lbl.setText("connecting…")
        self.test_lbl.setStyleSheet("color: #ffd43b; font-size: 9pt;")
        sess = MesSession(self.app, self.head, cfg)

        def done(res):
            ok, msg = res
            self.test_lbl.setText(("✔ " if ok else "✖ ") + str(msg))
            self.test_lbl.setStyleSheet(
                f"color: {'#69db7c' if ok else '#ff8787'}; font-size: 9pt;")
            sess.logout()
        sess.login_async(done)

    def save(self):
        cfg = self._collect()
        if cfg["enabled"]:
            missing = [lbl for k, lbl, _ in self.FIELDS
                       if k in ("mes_ip", "username", "password", "stage_id",
                                "product_id") and not cfg.get(k)]
            if missing:
                QMessageBox.critical(
                    self, "Missing MES details",
                    "MES is enabled but these are empty:\n\n  "
                    + "\n  ".join(missing))
                return
        self.app.heads.set_mes(self.head, **cfg)
        self.app.comm_log.add("EVT", f"MES config saved for model "
                                     f"'{self.head}' "
                                     f"(enabled={cfg['enabled']})")
        QMessageBox.information(self, "Saved",
                                f"MES settings saved for '{self.head}'.")

    def _close(self):
        cb, self.on_close = self.on_close, None
        self.close()
        if cb:
            cb()

    def closeEvent(self, e):
        cb, self.on_close = self.on_close, None
        e.accept()
        if cb:
            cb()


# ============================================================================
# OPERATOR ID SCAN DIALOG (runs once when opening model if required)
# ============================================================================
class OperatorScanDialog(QDialog):
    def __init__(self, master, app, head_name):
        super().__init__(master)
        self.app = app
        self.head_name = head_name
        self.operator_id = None
        self._alive = True
        self.setWindowTitle(f"Operator Login — {head_name}")
        self.setFixedSize(480, 250)
        self.setStyleSheet("QDialog { background-color: #1c1c22; }")

        lay = QVBoxLayout(self)
        lay.setContentsMargins(20, 16, 20, 16)
        lay.setSpacing(12)

        t = QLabel("👤 SCAN OPERATOR ID")
        t.setStyleSheet("color: #ffd43b; font-size: 15pt; font-weight: bold;")
        t.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lay.addWidget(t)

        sub = QLabel(f"Model '{head_name}' requires Operator ID "
                     "authentication.\nPress Scan (or F2) — the scan box "
                     "takes the keyboard until you press Enter.")
        sub.setStyleSheet(f"color: {C_MUTED}; font-size: 9pt;")
        sub.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lay.addWidget(sub)

        self.status_lbl = QLabel("Waiting for scan…")
        self.status_lbl.setStyleSheet(f"color: {C_FAINT}; font-size: 10pt;")
        self.status_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lay.addWidget(self.status_lbl)

        brow = QHBoxLayout()
        brow.addStretch()
        brow.addWidget(styled_btn(self, "⌕  Scan  (F2)", self.open_scan,
                                  color="#3b6ef5"))
        brow.addWidget(styled_btn(self, "Cancel", self.reject,
                                  color="#44444e"))
        brow.addStretch()
        lay.addLayout(brow)

        QShortcut(QKeySequence("F2"), self, activated=self.open_scan)
        # open the scan box straight away
        QTimer.singleShot(250, self.open_scan)

    def open_scan(self):
        if not self._alive:
            return
        code = QuickScanInput.ask(
            self, prompt="SCAN OPERATOR ID",
            sub=f"model '{self.head_name}' — scan your ID badge, "
                "or type it and press Enter")
        if code:
            self.operator_id = code
            self.accept()
        else:
            self.status_lbl.setText("No ID scanned — press Scan (F2) "
                                    "to try again.")
            self.status_lbl.setStyleSheet("color: #ff8787; font-size: 10pt;")

    def reject(self):
        self._alive = False
        super().reject()

    def closeEvent(self, e):
        self._alive = False
        e.accept()


# ============================================================================
# SCREEN 1 — HOME (model selection + comm + lock)
# ============================================================================
class HomeScreen(BaseScreen):
    def __init__(self, master, app):
        super().__init__(master, app)
        self.setStyleSheet("background-color: #1c1c22;")
        lay = QVBoxLayout(self)
        lay.setContentsMargins(20, 12, 20, 12)
        top = QHBoxLayout()
        t = QLabel("MODELS")
        t.setStyleSheet("color: white; font-size: 20pt; font-weight: bold;")
        top.addWidget(t)
        top.addStretch()
        top.addWidget(app.lock_button(self))
        lay.addLayout(top)
        if not app.is_locked():
            hint = QLabel("1) build detection → 2) Finish Model → "
                          "3) Communication.  The client can also "
                          "select a model by its Select-signal.")
            hint.setStyleSheet(f"color: {C_MUTED}; font-size: 10pt;")
            lay.addWidget(hint)
            lay.addWidget(CommBar(self, app, mini_log_lines=4))
            addrow = QHBoxLayout()
            addrow.addStretch()
            addrow.addWidget(styled_btn(self, "+  Add Model", self.add_head,
                                        color="#2f9e44"))
            addrow.addStretch()
            lay.addLayout(addrow)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        self.list_frame = QWidget()
        self.list_lay = QVBoxLayout(self.list_frame)
        self.list_lay.setContentsMargins(40, 8, 40, 8)
        self.list_lay.setSpacing(8)
        self.list_lay.addStretch()
        scroll.setWidget(self.list_frame)
        lay.addWidget(scroll, stretch=1)
        self.refresh()

    def refresh(self):
        while self.list_lay.count() > 1:      # keep trailing stretch
            it = self.list_lay.takeAt(0)
            if it.widget():
                it.widget().deleteLater()
        heads = self.app.heads.list_heads()
        if not heads:
            empty = QLabel("No models yet — press  + Add Model")
            empty.setStyleSheet("color: #666; font-size: 11pt;")
            empty.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self.list_lay.insertWidget(0, empty)
            return
        for i, h in enumerate(heads):
            row = panel("#26262e")
            rlay = QHBoxLayout(row)
            rlay.setContentsMargins(14, 8, 14, 8)
            name = QLabel(h)
            name.setStyleSheet("color: white; font-size: 13pt;"
                               " font-weight: bold;")
            rlay.addWidget(name)
            cnt = self.app.heads.get_counters(h)["cycle"]
            if self.app.is_locked():
                status = (f"   cycles  OK {cnt['ok']} / NG {cnt['ng']}")
            else:
                cfg = CommConfig(self.app.heads, h)
                done, total, _ = cfg.progress()
                sc = self.app.heads.get_scanner(h)
                status = (f"  {len(self.app.heads.list_parents(h))} scene(s)"
                          " | " + ("detection ✔"
                                   if self.app.heads.is_finished(h)
                                   else "detection…")
                          + f" | comm {done}/{total}"
                          + f" | scanner {'ON' if sc.get('enabled') else 'off'}"
                          + f" | cycles OK {cnt['ok']}/NG {cnt['ng']}")
            st = QLabel(status)
            st.setStyleSheet(f"color: {C_MUTED}; font-size: 9pt;")
            rlay.addWidget(st)
            rlay.addStretch()
            if not self.app.is_locked():
                rlay.addWidget(styled_btn(row, "Delete",
                                          lambda hh=h: self.delete_head(hh),
                                          color="#c92a2a"))
            rlay.addWidget(styled_btn(row, "Open ▶",
                                      lambda hh=h: self.app.show_head(hh)))
            self.list_lay.insertWidget(i, row)

    def add_head(self):
        name, ok = QInputDialog.getText(self, "Add Model", "New model name:")
        if not ok or not name or not name.strip():
            return
        name = name.strip()
        if self.app.heads.exists(name):
            QMessageBox.information(self, "Model exists",
                                    f"Model '{name}' already exists — "
                                    "opening it.")
            self.app.show_head(name)
            return
        self.app.heads.create(name)
        # ask about MES integration at model creation (details are filled in
        # later, once the model is finalised)
        m = QMessageBox.question(
            self, "MES integration",
            f"Enable MES integration for model '{name}'?\n\n"
            "The connection details (IP, login, stage, product…) are "
            "configured with the 'MES' button after you finish the model.")
        if m == QMessageBox.StandardButton.Yes:
            self.app.heads.set_mes(name, enabled=True)
        # ask to add the barcode scanner right at model creation
        r = QMessageBox.question(
            self, "Barcode scanner",
            f"Add a barcode scanner to model '{name}' now?\n\n"
            "(You can also configure it later with the 'Scanner' button.)")
        if r == QMessageBox.StandardButton.Yes:
            ScannerDialog(self, self.app, name,
                          on_close=lambda: self.app.show_head(name)).show()
        else:
            self.app.show_head(name)

    def delete_head(self, head_name):
        reply = QMessageBox.question(
            self,
            "Confirm Delete",
            f"Are you sure you want to delete model '{head_name}'?\n\n"
            f"This will permanently delete all data and files for model '{head_name}'.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No
        )
        if reply == QMessageBox.StandardButton.Yes:
            self.app.heads.delete(head_name)
            self.refresh()


# ============================================================================
# SCREEN 2 — MODEL: scanner-first cycle, manual+client run, counters, lock
# ============================================================================
class HeadScreen(BaseScreen):
    VIEW_LIVE = "live"
    VIEW_RESULT = "result"
    VIEW_COMPARE = "compare"

    def __init__(self, master, app, head_name):
        super().__init__(master, app)
        self.head = head_name
        self.parent_store = app.heads.get_parent_store(head_name)
        self.busy = False
        self.view = self.VIEW_LIVE
        self.result_views = {}
        self.result_key = "result"
        self._roi_cache = {}
        self._children_cache = {}
        self.comm_cfg = CommConfig(app.heads, head_name)
        self.compare_targets = []
        self.compare_mode = "side"
        locked = app.is_locked()

        # barcode scanner state (cycle gate — quick-input scan box)
        self.scanner_cfg = app.heads.get_scanner(head_name)
        self.scan_value = None
        self.scan_sent = False
        self.scan_time = ""
        self._scan_open = False
        # A half-built model must not demand scans: the scanner step (and MES
        # with it) only goes live once 'Finish Model' has been pressed, so you
        # can build and test scenes freely while the model is incomplete.
        self.finished = app.heads.is_finished(head_name)
        self.scan_active = bool(self.scanner_cfg.get("enabled")
                                and self.finished)
        # ---- camera QR: the PART is read by the camera instead of the wedge
        # scanner.  (The Operator ID is ALWAYS scanned with the scanner.)
        self.scan_source = self.scanner_cfg.get("source", "scanner")
        self.qr_camera = bool(self.scanner_cfg.get("enabled")
                              and self.scan_source == "camera")
        self.qr_scan_active = bool(self.scan_active and self.qr_camera)
        self.qr_master = (app.heads.get_master_qr(head_name)
                          if self.qr_camera else None)
        # idle | waiting (for the client's trigger) | scanning | retrying
        self.qr_state = "idle"
        self.qr_msg = ""             # last failure text, shown in the panel
        self.qr_attempt = 0          # current attempt of the master read
        self._qr_toast = None
        self._qr_gen = 0             # invalidates stale timers / threads
        self._qr_stop = threading.Event()
        self.scan_btn_qr = None
        self.btn_qr_now = None
        if self.qr_camera:
            app.camera.keep_raw = True   # full-res frames for QR reading
        # ---- MES integration state
        self.mes_cfg = app.heads.get_mes(head_name)
        self.mes_active = bool(self.mes_cfg.get("enabled") and self.finished)
        self.mes = (MesSession(app, head_name, self.mes_cfg)
                    if self.mes_active else None)
        self.mes_prev_state = "idle"   # idle | checking | pass | fail | error
        self.mes_prev_msg = ""
        self.mes_busy = False          # a cycle submission is in flight
        self.cycle_images = []         # [(label, ndarray)] collected per cycle
        # MES gets ONLY the current unit: runs are kept per scope and a
        # re-run replaces the older run (see _mes_collect); a new serial
        # clears everything
        self.cycle_runs = []
        self.cycle_camera = None       # full camera view of the latest run
        self.cycle_report_lines = []   # detection lines for this cycle
        self.cycle_defects = []
        self.mes_status_lbl = None
        # operator session lives on the app: it survives the HeadScreen being
        # rebuilt (scene creation, comm config, lock/unlock, scanner config…)
        # and only ends when you go back to the model list.
        self.operator_id = app.get_operator(head_name)

        # prompt for Operator ID scan ONLY when there is no live session
        if (self.scan_active
                and self.scanner_cfg.get("require_operator")
                and not self.operator_id):
            dlg = OperatorScanDialog(self, app, head_name)
            if dlg.exec() == QDialog.DialogCode.Accepted and dlg.operator_id:
                self.operator_id = dlg.operator_id
                app.set_operator(head_name, self.operator_id)
                app.comm_log.add("EVT", f"Operator '{self.operator_id}' logged in for model '{head_name}'")
            else:
                QTimer.singleShot(0, self.app.show_home)
                return
        # the scanned Operator ID is what MES gets, not the fixed one in
        # mes.yaml
        if self.mes and self.operator_id:
            self.mes.set_operator(self.operator_id)

        root = QVBoxLayout(self)
        root.setContentsMargins(10, 6, 10, 6)
        # the screen itself holds keyboard focus so a scan that arrives while
        # the scan box is shut lands here (see keyPressEvent) instead of
        # leaking into whatever widget happened to be focused
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

        # ---- top bar
        top = QHBoxLayout()
        top.addWidget(styled_btn(self, "← Models", self.app.show_home,
                                 color="#44444e"))
        t = QLabel(f"MODEL:  {head_name}")
        t.setStyleSheet("color: white; font-size: 15pt; font-weight: bold;")
        top.addWidget(t)
        if self.operator_id:
            op_badge = QLabel(f"👤 Op: {self.operator_id}")
            op_badge.setStyleSheet("color: #74c0fc; font-weight: bold;"
                                   " font-size: 11pt; padding: 3px 10px;"
                                   " background: #1c2333; border-radius: 5px;")
            top.addWidget(op_badge)
            top.addWidget(styled_btn(self, "⇄", self.switch_operator,
                                     color="#44444e"))
        top.addStretch()
        if not locked:
            top.addWidget(styled_btn(self, "Auto Balance", self.auto_balance,
                                     color="#44444e"))
            top.addWidget(styled_btn(self, "+ Add Scene", self.add_parent,
                                     color="#2f9e44"))
            top.addWidget(styled_btn(self, "Scanner",
                                     self.open_scanner_cfg, color="#3b6ef5"))
            top.addWidget(styled_btn(
                self, "MES", self.open_mes_cfg,
                color="#0ca678" if self.mes_cfg.get("enabled") else "#44444e"))
            top.addWidget(styled_btn(self, "Comm Config", self.open_comm,
                                     color="#845ef7" if self.finished
                                     else "#44444e"))
            if not self.finished:
                top.addWidget(styled_btn(self, "Finish Model",
                                         self.finish_head, color="#f59f00"))
        top.addWidget(app.lock_button(self))
        root.addLayout(top)

        # cycle tracking: a cycle completes when EVERY scene has been run
        # (in any order — scene1 then scene2, or one RUN MODEL)
        self.cycle_scene_results = {}

        body = QHBoxLayout()
        root.addLayout(body, stretch=1)

        # ---- left: manual triggers (hidden when locked) + view + setting
        left = panel()
        left.setFixedWidth(270)
        llay = QVBoxLayout(left)
        llay.setContentsMargins(10, 8, 10, 8)
        if not locked:
            lt = QLabel("MANUAL TRIGGER")
            lt.setStyleSheet(f"color: {C_MUTED}; font-weight: bold;")
            lt.setAlignment(Qt.AlignmentFlag.AlignCenter)
            llay.addWidget(lt)
            llay.addWidget(styled_btn(
                left, "▶▶  RUN MODEL (all scenes)",
                lambda: self.execute(("head",), "MANUAL"), color="#e8590c"))
            st = QLabel("SCENES — press to RUN")
            st.setStyleSheet(f"color: {C_MUTED}; font-size: 9pt;"
                             " font-weight: bold;")
            st.setAlignment(Qt.AlignmentFlag.AlignCenter)
            llay.addWidget(st)
            self.parent_btn_frame = QWidget()
            self.parent_btn_lay = QVBoxLayout(self.parent_btn_frame)
            self.parent_btn_lay.setContentsMargins(0, 0, 0, 0)
            self.parent_btn_lay.setSpacing(6)
            llay.addWidget(self.parent_btn_frame)
        else:
            self.parent_btn_frame = None
            self.parent_btn_lay = None
            lt = QLabel("🔒 OPERATOR MODE")
            lt.setStyleSheet(f"color: {C_MUTED}; font-weight: bold;")
            lt.setAlignment(Qt.AlignmentFlag.AlignCenter)
            llay.addWidget(lt)
            sub = QLabel("runs are triggered by\nthe client (communication)")
            sub.setStyleSheet(f"color: {C_FAINT}; font-size: 8pt;")
            llay.addWidget(sub)

        vt = QLabel("VIEW")
        vt.setStyleSheet(f"color: {C_MUTED}; font-size: 9pt;"
                         " font-weight: bold;")
        vt.setAlignment(Qt.AlignmentFlag.AlignCenter)
        llay.addWidget(vt)
        llay.addWidget(styled_btn(left, "Compare Master",
                                  self.enter_compare, color="#3b6ef5"))
        llay.addWidget(styled_btn(left, " Live View", self.back_to_live,
                                  color="#44444e"))

        _thr_edit = QLineEdit(str(self.app.heads.get_settings(head_name)
                                  .get("distance_threshold", 20)))
        self.var_dist_thr = SVar(_thr_edit)
        if not locked:
            setbox = QGroupBox(" OK / NG SETTING ")
            sblay = QVBoxLayout(setbox)
            srow = QHBoxLayout()
            sl = QLabel("Pin distance thr (px):")
            sl.setStyleSheet("color: #c8c8d0; font-size: 9pt;")
            srow.addWidget(sl)
            _thr_edit.setFixedWidth(64)
            _thr_edit.returnPressed.connect(self._save_dist_thr)
            _thr_edit.editingFinished.connect(self._save_dist_thr)
            srow.addWidget(_thr_edit)
            srow.addStretch()
            sblay.addLayout(srow)
            hint = QLabel("Pins are paired to the master first, then\n"
                          "the distance is judged: shifted more than\n"
                          "this (or missing) = RED = NG.")
            hint.setStyleSheet(f"color: {C_FAINT}; font-size: 8pt;")
            sblay.addWidget(hint)
            llay.addWidget(setbox)
        llay.addStretch()
        body.addWidget(left)

        # ---- centre: canvas + per-view controls
        centre = QWidget()
        clay = QVBoxLayout(centre)
        clay.setContentsMargins(4, 0, 4, 0)
        self.canvas = VideoCanvas(centre)
        clay.addWidget(self.canvas, stretch=1)
        clay.addWidget(zoom_controls(centre, self.canvas))

        self.result_ctl = QWidget()
        rc = QHBoxLayout(self.result_ctl)
        rc.setContentsMargins(0, 4, 0, 0)
        rc.addStretch()
        self.btn_pinmap = styled_btn(self.result_ctl, "Pin Map",
                                     self.toggle_pinmap, color="#3b6ef5")
        rc.addWidget(self.btn_pinmap)
        rc.addWidget(styled_btn(self.result_ctl, "Compare with Master",
                                self.enter_compare, color="#845ef7"))
        rc.addWidget(styled_btn(self.result_ctl, "Back to Live",
                                self.back_to_live, color="#44444e"))
        rc.addStretch()
        self.result_ctl.setVisible(False)
        clay.addWidget(self.result_ctl)

        self.compare_ctl = QWidget()
        cc = QHBoxLayout(self.compare_ctl)
        cc.setContentsMargins(0, 4, 0, 0)
        cc.addStretch()
        cl = QLabel("Compare:")
        cl.setStyleSheet("color: #c8c8d0; font-size: 9pt;")
        cc.addWidget(cl)
        self.compare_sel = QComboBox()
        self.compare_sel.setFixedWidth(280)
        cc.addWidget(self.compare_sel)
        self.btn_side = styled_btn(self.compare_ctl, "Side-by-Side",
                                   lambda: self.set_compare_mode("side"),
                                   color="#3b6ef5")
        cc.addWidget(self.btn_side)
        self.btn_overlay = styled_btn(self.compare_ctl, "Overlay",
                                      lambda: self.set_compare_mode("overlay"),
                                      color="#44444e")
        cc.addWidget(self.btn_overlay)
        cc.addWidget(styled_btn(self.compare_ctl, " Back to Live",
                                self.back_to_live, color="#44444e"))
        cc.addStretch()
        self.compare_ctl.setVisible(False)
        clay.addWidget(self.compare_ctl)
        body.addWidget(centre, stretch=1)

        # ---- right column
        right = panel()
        right.setFixedWidth(330)
        rlay = QVBoxLayout(right)
        rlay.setContentsMargins(6, 6, 6, 6)
        self.status_big = QLabel("——")
        self.status_big.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.status_big.setFixedHeight(76)
        rlay.addWidget(self.status_big)
        self.status_sub = QLabel("no run yet")
        self.status_sub.setStyleSheet("color: #8f8f9a; font-size: 9pt;")
        self.status_sub.setAlignment(Qt.AlignmentFlag.AlignCenter)
        rlay.addWidget(self.status_sub)
        self.set_big_status(None)

        # -- OK/NG counters (cycle first; arranged vertically)
        cbox = QGroupBox(" OK / NG COUNTER ")
        cblay = QVBoxLayout(cbox)
        cblay.setSpacing(6)
        crow = QHBoxLayout()
        cyc = QLabel("CYCLE")
        cyc.setStyleSheet("color: #c8c8d0; font-weight: bold; font-size: 10pt;")
        crow.addWidget(cyc)
        crow.addStretch()
        if not locked:
            crow.addWidget(styled_btn(cbox, "⟲ Reset", self.reset_counters,
                                      color="#e03131"))
        self.btn_expand = styled_btn(cbox, "▾ Expand", self.toggle_counters,
                                     color="#44444e")
        crow.addWidget(self.btn_expand)
        cblay.addLayout(crow)

        self.cnt_ok_lbl = QLabel("OK  0")
        self.cnt_ok_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.cnt_ok_lbl.setStyleSheet(
            "background: #2f9e44; color: white; font-weight: bold;"
            " font-size: 14pt; padding: 6px 12px; border-radius: 6px;")

        self.cnt_ng_lbl = QLabel("NG  0")
        self.cnt_ng_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.cnt_ng_lbl.setStyleSheet(
            "background: #c92a2a; color: white; font-weight: bold;"
            " font-size: 14pt; padding: 6px 12px; border-radius: 6px;")

        cblay.addWidget(self.cnt_ok_lbl)
        cblay.addWidget(self.cnt_ng_lbl)

        self.counter_detail = QWidget()
        self.counter_detail.setStyleSheet(
            "QWidget { background: #1a1a20; border-radius: 6px; }"
            " QLabel { background: transparent; }")
        self.counter_detail_lay = QVBoxLayout(self.counter_detail)
        self.counter_detail_lay.setContentsMargins(4, 4, 4, 4)
        self.counter_detail_lay.setSpacing(2)
        self.counter_detail.setVisible(False)
        cblay.addWidget(self.counter_detail)
        self.counters_expanded = False
        rlay.addWidget(cbox)

        # -- barcode scanner panel (only once the model is finished)
        if self.scan_active:
            sbox = QGroupBox(" CAMERA QR SCANNER (press every cycle) "
                             if self.qr_scan_active else
                             " BARCODE SCANNER (runs before scenes) ")
            sblay = QVBoxLayout(sbox)
            self.scan_status_lbl = QLabel("waiting for scan…")
            self.scan_status_lbl.setWordWrap(True)
            self.scan_status_lbl.setStyleSheet("color: #ffd43b;"
                                               " font-weight: bold;")
            sblay.addWidget(self.scan_status_lbl)
            self.scan_value_lbl = QLabel("—")
            self.scan_value_lbl.setStyleSheet(
                f"background: {C_FIELD}; color: white; padding: 4px 8px;"
                " border-radius: 5px; font-family: Consolas, monospace;"
                " font-weight: bold; font-size: 12pt;")
            sblay.addWidget(self.scan_value_lbl)
            if self.qr_scan_active:
                m = self.qr_master
                if m:
                    info = (f"master QR '{m['name']}' (scene {m['scene']}) · "
                            f"{'region set' if m.get('roi_norm') else 'full frame'}"
                            f" · {QrRules.from_dict(m).describe()}")
                else:
                    info = ("⚠ no master QR — build a scene and add one "
                            "(5. QR Scanner)")
                il = QLabel(info)
                il.setWordWrap(True)
                il.setStyleSheet(f"color: {C_FAINT}; font-size: 8pt;")
                sblay.addWidget(il)
                self.scan_btn_qr = styled_btn(sbox, "📷  SCAN QR  (F2)",
                                              self.qr_button_pressed,
                                              color="#3b6ef5")
                sblay.addWidget(self.scan_btn_qr)
                if self._qr_sig("TRIGGER") and not locked:
                    self.btn_qr_now = styled_btn(
                        sbox, "▶ Scan now (don't wait for client)",
                        self.qr_scan_now, color="#f59f00")
                    sblay.addWidget(self.btn_qr_now)
            else:
                sblay.addWidget(styled_btn(sbox, "⌕  SCAN  (F2)",
                                           self.open_quick_scan,
                                           color="#3b6ef5"))
            srow2 = QHBoxLayout()
            srow2.addStretch()
            srow2.addWidget(styled_btn(sbox, "Clear", self.clear_barcode,
                                       color="#44444e"))
            srow2.addWidget(styled_btn(sbox, "Resend", self.resend_barcode,
                                       color="#3b6ef5"))
            sblay.addLayout(srow2)
            rlay.addWidget(sbox)
            QShortcut(QKeySequence("F2"), self,
                      activated=(self.qr_button_pressed
                                 if self.qr_scan_active
                                 else self.open_quick_scan))
        else:
            self.scan_status_lbl = None

        # -- MES previous-stage status panel
        if self.mes_active:
            mbox = QGroupBox(" MES  (previous stage check) ")
            mblay = QVBoxLayout(mbox)
            self.mes_status_lbl = QLabel("idle")
            self.mes_status_lbl.setWordWrap(True)
            self.mes_status_lbl.setStyleSheet("color: #9a9aa5;"
                                              " font-weight: bold;")
            mblay.addWidget(self.mes_status_lbl)
            self.mes_msg_lbl = QLabel("—")
            self.mes_msg_lbl.setWordWrap(True)
            self.mes_msg_lbl.setStyleSheet(
                f"background: {C_FIELD}; color: #c8c8d0; padding: 4px 8px;"
                " border-radius: 5px; font-size: 9pt;")
            mblay.addWidget(self.mes_msg_lbl)
            mrow = QHBoxLayout()
            mrow.addStretch()
            mrow.addWidget(styled_btn(mbox, "Re-check",
                                      self.mes_recheck, color="#44444e"))
            mblay.addLayout(mrow)
            rlay.addWidget(mbox)
        else:
            self.mes_status_lbl = None
            self.mes_msg_lbl = None

        if not locked:
            rlay.addWidget(CommBar(right, app, mini_log_lines=4))
            dt = QLabel("DETECTION LOG")
            dt.setStyleSheet(f"color: {C_MUTED}; font-weight: bold;")
            dt.setAlignment(Qt.AlignmentFlag.AlignCenter)
            rlay.addWidget(dt)
            self.log_box = QTextEdit()
            self.log_box.setReadOnly(True)
            self.log_box.setStyleSheet(
                f"QTextEdit {{ background: {C_FIELD}; color: #c8c8d0;"
                f" border: none; border-radius: 6px; font-size: 9pt; }}")
            rlay.addWidget(self.log_box, stretch=1)
        else:
            self.log_box = None
            rlay.addStretch()
        body.addWidget(right)

        self.reload_models()
        self.refresh_counters()
        self._update_scan_panel()
        self.app.camera.start()
        self.auto_balance()
        if not locked:
            done, total, _ = self.comm_cfg.progress()
            self.log(f"Comm config: {done}/{total} "
                     f"({'complete ✔' if done == total else 'incomplete'})")
        if self.qr_scan_active:
            self.log("Camera QR cycle: press 📷 SCAN QR (F2) at the start of "
                     "every cycle (QR readers: "
                     + ", ".join(available_backends()) + ")")
            if not self.qr_master:
                self.log("⚠ camera QR is selected but the model has NO master "
                         "QR — build a scene and add one (5. QR Scanner)")
            elif self.comm_cfg.qr_required():
                _miss = [f for f in CommConfig.QR_FIELDS
                         if not self._qr_sig(f)]
                if _miss:
                    self.log("[QR] Comm Config → CAMERA QR still empty: "
                             + ", ".join(_miss))
        elif self.scan_active:
            self.log("Scanner cycle: press SCAN (F2) and scan a barcode "
                     "to start")
            if self.scanner_cfg.get("auto_popup", True):
                QTimer.singleShot(500, self.open_quick_scan)
        elif self.scanner_cfg.get("enabled") and not self.finished:
            self.log("Scanner is configured but the model is not finished — "
                     "no scan is required yet. Press '🏁 Finish Model' to "
                     "start the scan-first cycle.")
        if self.mes_active:
            if not MES_AVAILABLE:
                self.log(f"[MES] enabled but mes_client is unavailable "
                         f"({MES_IMPORT_ERR}) — no MES calls will be made")
            else:
                self.log(f"[MES] enabled — stage "
                         f"{self.mes_cfg.get('stage_id', '?')} "
                         f"'{self.mes_cfg.get('stage_name', '')}' @ "
                         f"{self.mes_cfg.get('mes_ip', '?')}")
            self._set_mes_status("idle", "")
        elif self.mes_cfg.get("enabled") and not self.finished:
            self.log("MES is enabled but the model is not finished — no MES "
                     "calls will be made until you press '🏁 Finish Model'.")
        QTimer.singleShot(0, lambda: self.setFocus(
            Qt.FocusReason.OtherFocusReason))
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)
        self._timer.start(30)

    # ------------------------------------------------------------------
    def destroy_screen(self):
        self._qr_stop.set()
        if self.mes:
            m, self.mes = self.mes, None
            threading.Thread(target=m.logout, daemon=True).start()
        super().destroy_screen()

    def log(self, msg):
        if not self._alive or self.log_box is None:
            return
        self.log_box.append(html.escape(msg, quote=False))
        self.log_box.verticalScrollBar().setValue(
            self.log_box.verticalScrollBar().maximum())

    def set_big_status(self, ok=None, sub=""):
        base = ("font-size: 34pt; font-weight: bold; border-radius: 8px;"
                " padding: 2px;")
        if ok is None:
            self.status_big.setText("——")
            self.status_big.setStyleSheet(
                f"background: #2e2e38; color: #8f8f9a; {base}")
        elif ok:
            self.status_big.setText("OK")
            self.status_big.setStyleSheet(
                f"background: #2f9e44; color: white; {base}")
        else:
            self.status_big.setText("NG")
            self.status_big.setStyleSheet(
                f"background: #c92a2a; color: white; {base}")
        self.status_sub.setText(sub or ("no run yet" if ok is None else ""))

    def get_dist_thr(self):
        try:
            v = float(self.var_dist_thr.get())
            return v if v > 0 else 20.0
        except ValueError:
            return 20.0

    def _save_dist_thr(self):
        thr = self.get_dist_thr()
        self.var_dist_thr.set(f"{thr:g}")
        self.app.heads.set_settings(self.head, distance_threshold=thr)
        self.log(f"Pin distance threshold set to {thr:g} px (saved)")

    # ------------------------------------------------------------------
    # OK/NG COUNTERS
    # ------------------------------------------------------------------
    def _counter_row(self, lay, name, ok, ng, indent=8, bold=False):
        row = QWidget()
        rl = QHBoxLayout(row)
        rl.setContentsMargins(indent, 0, 8, 0)
        rl.setSpacing(4)
        nm = QLabel(name)
        nm.setStyleSheet(
            f"color: {'#e2e2d2' if bold else '#c8c8d0'}; font-size: 9pt;"
            + (" font-weight: bold;" if bold else ""))
        rl.addWidget(nm, stretch=1)
        okl = QLabel(f"OK {ok}")
        okl.setStyleSheet(f"background: {'#2f9e44' if ok else '#3a3a44'};"
                          " color: white; font-size: 9pt; font-weight: bold;"
                          " padding: 1px 7px; border-radius: 4px;")
        rl.addWidget(okl)
        ngl = QLabel(f"NG {ng}")
        ngl.setStyleSheet(f"background: {'#c92a2a' if ng else '#3a3a44'};"
                          " color: white; font-size: 9pt; font-weight: bold;"
                          " padding: 1px 7px; border-radius: 4px;")
        rl.addWidget(ngl)
        lay.addWidget(row)

    def _counter_section(self, lay, title):
        s = QLabel(title)
        s.setStyleSheet("color: #8f8f9a; font-size: 8pt; font-weight: bold;")
        lay.addWidget(s)

    def refresh_counters(self):
        c = self.app.heads.get_counters(self.head)
        self.cnt_ok_lbl.setText(f"OK  {c['cycle']['ok']}")
        self.cnt_ng_lbl.setText(f"NG  {c['cycle']['ng']}")
        if not self.counters_expanded:
            return
        clear_layout(self.counter_detail_lay)
        self._counter_section(self.counter_detail_lay, "SCENE WISE")
        if not c["scenes"]:
            e = QLabel("no runs yet")
            e.setStyleSheet("color: #55555f; font-size: 8pt;")
            self.counter_detail_lay.addWidget(e)
        for s, v in sorted(c["scenes"].items()):
            self._counter_row(self.counter_detail_lay, s, v["ok"], v["ng"],
                              indent=8, bold=True)
            # its areas directly underneath, indented
            for a, av in sorted(c["areas"].items()):
                if a.startswith(s + "/"):
                    self._counter_row(self.counter_detail_lay,
                                      "└ " + a.split("/", 1)[1],
                                      av["ok"], av["ng"], indent=20)
        # areas of scenes that no longer exist (renamed etc.)
        orphans = [(a, v) for a, v in sorted(c["areas"].items())
                   if a.split("/", 1)[0] not in c["scenes"]]
        if orphans:
            self._counter_section(self.counter_detail_lay, "OTHER AREAS")
            for a, v in orphans:
                self._counter_row(self.counter_detail_lay, a,
                                  v["ok"], v["ng"], indent=20)

    def toggle_counters(self):
        self.counters_expanded = not self.counters_expanded
        if self.counters_expanded:
            self.counter_detail.setVisible(True)
            self.btn_expand.setText("▴ Hide")
        else:
            self.counter_detail.setVisible(False)
            self.btn_expand.setText("▾ Expand")
        self.refresh_counters()

    def reset_counters(self):
        r = QMessageBox.question(self, "Reset counters",
                                 f"Reset ALL OK/NG counters of model "
                                 f"'{self.head}'?")
        if r == QMessageBox.StandardButton.Yes:
            self.app.heads.reset_counters(self.head)
            self.refresh_counters()
            self.log("Counters reset")

    # ------------------------------------------------------------------
    # BARCODE SCANNER (cycle gate: scan + send BEFORE the scenes)
    # Scanning = the quick-input box: it floats over everything, holds the
    # keyboard so nothing else can be typed into, and returns on Enter.
    # ------------------------------------------------------------------
    def open_scanner_cfg(self):
        ScannerDialog(self, self.app, self.head,
                      on_close=lambda: self.app.show_head(
                          self.head, announce=False)).show()

    def switch_operator(self):
        """End the operator session and ask for a new ID, without having to
        go back to the model list."""
        self.app.clear_operator()
        self.app.comm_log.add("EVT", f"operator '{self.operator_id}' "
                                     f"logged out of model '{self.head}'")
        self.app.show_head(self.head, announce=False)

    def keyPressEvent(self, e):
        """Catch a scan that arrives while the scan box is closed.

        A wedge scanner just types.  Without this the characters (and the
        trailing Enter) go to whatever widget has focus — which is how a
        second scan used to trigger a button and bounce back to the model
        list.  Here they are swallowed and pushed into the scan box instead.
        """
        if (self.scan_active and not self.qr_scan_active
                and not self._scan_open):
            txt = e.text()
            if txt and txt.isprintable() and txt.strip():
                if self.scan_value:
                    self.log("[scanner] already scanned this cycle — press "
                             "Clear (or finish the cycle) to scan again")
                    e.accept()
                    return
                e.accept()
                self.open_quick_scan(seed=txt)
                return
            if e.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
                e.accept()          # a bare Enter must never activate anything
                return
        super().keyPressEvent(e)

    def open_quick_scan(self, seed=""):
        """Pop the global scan box and take the barcode from it."""
        if not self._alive or self._scan_open:
            return
        if not self.scan_active or self.qr_scan_active:
            return
        if self.busy:
            self.log("[scanner] busy — scan box not opened")
            return
        self._scan_open = True
        try:
            op = f"   (Op: {self.operator_id})" if self.operator_id else ""
            code = QuickScanInput.ask(
                self, prompt="SCAN BARCODE",
                sub=f"model '{self.head}'{op} — the keyboard is locked to "
                    "this box until Enter",
                initial=seed)
        finally:
            self._scan_open = False
            self.setFocus(Qt.FocusReason.OtherFocusReason)
        if code:
            self._handle_barcode(code)
        else:
            self.log("[scanner] scan cancelled — detection still blocked")

    # ------------------------------------------------------------------
    # MES  (previous stage check on scan, report + upload at cycle end)
    # ------------------------------------------------------------------
    def open_mes_cfg(self):
        if not self.app.heads.is_finished(self.head):
            QMessageBox.information(
                self, "Finish the model first",
                "MES details can be configured once the model is "
                "finalised.\n\nBuild the scenes, then press "
                "'🏁 Finish Model'.")
            return
        MesDialog(self, self.app, self.head,
                  on_close=lambda: self.app.show_head(
                      self.head, announce=False)).show()

    def _set_mes_status(self, state, msg):
        self.mes_prev_state = state
        self.mes_prev_msg = msg or ""
        if not self.mes_status_lbl:
            return
        text, color = {
            "idle":     ("waiting for a scan…", "#9a9aa5"),
            "checking": ("⟳ checking previous stage…", "#ffd43b"),
            "pass":     ("✔ PREVIOUS STAGE PASSED", "#69db7c"),
            "all_passed": ("✔ ALL STAGES ALREADY PASSED — OK to run",
                           "#69db7c"),
            "fail":     ("✖ NOT PASSED BY PREVIOUS STATION", "#ff6b6b"),
            "error":    ("⚠ MES ERROR — could not verify", "#ffa94d"),
            "sending":  ("⟳ sending report to MES…", "#ffd43b"),
            "sent":     ("✔ report sent to MES", "#69db7c"),
            "senderr":  ("⚠ MES report FAILED", "#ff6b6b"),
        }.get(state, (state, "#9a9aa5"))
        self.mes_status_lbl.setText(text)
        self.mes_status_lbl.setStyleSheet(f"color: {color};"
                                          " font-weight: bold;"
                                          " font-size: 11pt;")
        if self.mes_msg_lbl:
            self.mes_msg_lbl.setText(msg or "—")

    def mes_check_previous(self, serial):
        """Scan -> ask MES whether this unit passed its previous stage."""
        if not self.mes or not self.mes_active:
            return
        self._set_mes_status("checking", f"serial '{serial}'…")
        self.log(f"[MES] previous stage check for '{serial}'…")

        def done(res):
            if not self._alive:
                return
            state, msg = res
            msg = str(msg)
            if state == "pass":
                self._set_mes_status("pass", msg)
                self.log(f"[MES] ✔ previous stage OK — {msg}")
                self.app.comm_log.add("EVT", f"MES prev-stage PASS "
                                             f"'{serial}': {msg}")
                if self.scanner_cfg.get("send_to_client", True):
                                
                                    self._try_send_barcode()
                return
            if state == "all_passed":
                # MES says the board already cleared every stage. That is a
                # PASS, not a failure — carry on and report as normal.
                self._set_mes_status("all_passed", msg)
                self.log(f"[MES] ✔ all stages already passed — {msg}")
                self.log("[MES] treating as PASS — running this unit and "
                         "updating MES as usual")
                self.app.comm_log.add("EVT", f"MES prev-stage ALL-PASSED "
                                             f"'{serial}': {msg}")
                if self.scanner_cfg.get("send_to_client", True):
                                
                                    self._try_send_barcode()
                return
            self._set_mes_status("error" if state == "error" else "fail", msg)
            self.log(f"[MES] ✖ {msg}")
            self.app.comm_log.add("EVT", f"MES prev-stage "
                                         f"{state.upper()} '{serial}': {msg}")
            # Locked = production mode -> hard stop with a popup, then the
            # cycle restarts so the operator can scan the next board.
            # Unlocked = engineering mode -> warn only, runs still allowed.
            if self.app.is_locked() and self.mes_cfg.get(
                    "block_on_prev_fail", True):
                QMessageBox.critical(
                    self, "Previous stage NOT passed",
                    f"Serial:  {serial}\n\n{msg}\n\n"
                    "This unit cannot be tested at this stage.\n"
                    "Remove the board — the next scan will start a new "
                    "cycle.")
                self.log("[scanner] cycle aborted — scan the next barcode")
                self._start_next_cycle()
            else:
                self.log("[MES] not locked — run allowed, but this unit "
                         "did NOT pass the previous station")
        self.mes.check_previous_async(serial, done)

    def mes_recheck(self):
        if not self.scan_value:
            self.log("[MES] nothing to check — scan a barcode first")
            return
        self.mes_check_previous(self.scan_value)

    def mes_gate(self):
        """(allowed, reason) for the MES previous-stage result."""
        if not self.mes or not self.mes_active:
            return True, ""
        if self.mes_busy:
            return False, "MES is still submitting the previous cycle — wait"
        if self.mes_prev_state == "checking":
            return False, "previous-stage check still running — wait a moment"
        if not self.scan_active:
            # no scanner -> no serial number -> nothing MES can verify
            return True, ""
        if not self.app.is_locked() or not self.mes_cfg.get(
                "block_on_prev_fail", True):
            return True, ""          # engineering mode: warn only
        if self.mes_prev_state in ("pass", "all_passed"):
            return True, ""
        if self.mes_prev_state == "fail":
            return False, ("unit did NOT pass the previous station — "
                           f"{self.mes_prev_msg}")
        if self.mes_prev_state == "error":
            return False, ("previous stage could not be verified — "
                           f"{self.mes_prev_msg}")
        return False, "no previous-stage check done yet — scan the barcode"

    @staticmethod
    def _scope_covers(a, b):
        """Does a run of scope a make an older run of scope b obsolete?"""
        if a[0] == "head" or a == b:
            return True
        return a[0] == "parent" and b[0] == "child" and b[1] == a[1]

    def _mes_reset_cycle_data(self):
        self.cycle_runs = []
        self.cycle_camera = None
        self.cycle_images = []
        self.cycle_report_lines = []
        self.cycle_defects = []

    def _mes_collect(self, scope, logs, views, results):
        """Keep the lines / images / defects of this run for the MES report.
        ONLY CURRENT DATA: a run replaces every older run it covers (a
        re-run of a scene, or a whole-model run), so a failed first attempt
        never reaches MES once the part was re-run."""
        if not self.mes_active:
            return
        stamp = datetime.datetime.now().strftime("%H:%M:%S")
        name = ("MODEL (all scenes)" if scope[0] == "head"
                else "/".join(scope[1:]))
        lines = ["", f"[{stamp}]  RUN  {name}"]
        defects, images = [], []
        for line in logs:
            lines.append("    " + line.rstrip())
        for parent_name, children in results.items():
            for child_name, r in children.items():
                if r.get("ok"):
                    continue
                pins = r.get("pins") or {}
                desc = (f"{r.get('detected', 0)}/{r.get('master', 0)} "
                        f"detected")
                if pins.get("total"):
                    desc = (f"{pins.get('in_range', 0)}/{pins['total']} pins "
                            f"in range, max deviation "
                            f"{pins.get('max_dist', 0):.0f}px")
                is_qr = str(child_name).startswith("QR:")
                if is_qr:
                    desc = f"QR read failed: {r.get('reason', '')}"
                defects.append({
                    "defectLocation": f"{parent_name}/{child_name}",
                    "defectType": ("QR Read" if is_qr else
                                   self.mes_cfg.get("defect_type",
                                                    "Pin Bend")),
                    "description": desc,
                })
        if self.mes_cfg.get("upload_images", True):
            for key in ("result", "pins"):
                img = views.get(key)
                if img is not None:
                    label = f"{name} [{key}]".replace("/", "-")
                    images.append((label, img.copy()))
        self.cycle_runs = [r for r in self.cycle_runs
                           if not self._scope_covers(scope, r["scope"])]
        self.cycle_runs.append({"scope": scope, "lines": lines,
                                "defects": defects, "images": images})
        cam = views.get("camera")
        if cam is not None:
            self.cycle_camera = cam.copy()      # always the latest run

    def _mes_flatten(self):
        """cycle_runs -> the lists the submission uses."""
        lines, defects, images = [], [], []
        for r in self.cycle_runs:
            lines += r["lines"]
            defects += r["defects"]
            images += r["images"]
        up = self.app.app_config.data.get("mes_upload", {})
        if (self.cycle_camera is not None and up.get("include_camera_view", True)
                and self.mes_cfg.get("upload_images", True)):
            images.insert(0, ("camera view", self.cycle_camera))
        self.cycle_report_lines, self.cycle_defects = lines, defects
        self.cycle_images = images

    def _mes_submit_cycle(self, serial, cycle_ok):
        """End of cycle: write the detection report, upload the images, put
        their MES links in the report, upload it and file the stage report."""
        if not self.mes or not self.mes_active:
            return False
        if not serial:
            self.log("[MES] no serial number for this cycle — nothing sent")
            return False

        out_dir = os.path.join(self.app.heads.mes_log_dir(self.head),
                               serial.replace("/", "_").replace("\\", "_"))
        os.makedirs(out_dir, exist_ok=True)
        stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

        # only the CURRENT unit's data goes up (latest run per scope)
        self._mes_flatten()
        up = self.app.app_config.data.get("mes_upload", {})
        max_w = max(200, int(up.get("max_width", 1024)))
        jpg_q = max(30, min(95, int(up.get("jpeg_quality", 75))))

        # 1. write the images to disk — reduced to a small resolution
        image_paths = []
        for i, (label, img) in enumerate(self.cycle_images):
            safe = "".join(ch if ch.isalnum() or ch in "-_" else "_"
                           for ch in label)[:40]
            p = os.path.join(out_dir, f"{serial}_{stamp}_{i:02d}_{safe}.jpg")
            try:
                if img.shape[1] > max_w:
                    k = max_w / float(img.shape[1])
                    img = cv2.resize(img, None, fx=k, fy=k,
                                     interpolation=cv2.INTER_AREA)
                if not cv2.imwrite(p, img, [cv2.IMWRITE_JPEG_QUALITY, jpg_q]):
                    raise OSError("imwrite failed")
                image_paths.append((label, p))
            except Exception as e:
                self.log(f"[MES] could not save image {label}: {e}")

        # 2. build the detection report header + body
        head_lines = [
            "=" * 66,
            "DETECTION REPORT",
            "=" * 66,
            f"model            : {self.head}",
            f"serial number    : {serial}",
            f"operator        : "
            f"{self.operator_id or self.mes_cfg.get('operator_id', '') or '—'}"
            + ("  (scanned)" if self.operator_id else "  (from mes.yaml)"),
            f"machine          : {self.mes_cfg.get('machine_no', '')}",
            f"stage            : {self.mes_cfg.get('stage_id', '')} "
            f"{self.mes_cfg.get('stage_name', '')}",
            f"product          : {self.mes_cfg.get('product_id', '')} "
            f"{self.mes_cfg.get('product_name', '')}",
            f"time             : "
            f"{datetime.datetime.now():%Y-%m-%d %H:%M:%S}",
            f"previous stage   : {self.mes_prev_state.upper()} — "
            f"{self.mes_prev_msg}",
            f"CYCLE RESULT     : {'PASS' if cycle_ok else 'FAIL'}",
            "",
            "-" * 66,
            "DETECTION LOG",
            "-" * 66,
        ]
        report_path = os.path.join(out_dir,
                                   f"{serial}_report_{stamp}.txt")
        bundle = {
            "serial": serial,
            "ok": bool(cycle_ok),
            "images": image_paths,
            "report_lines": head_lines + self.cycle_report_lines,
            "defects": [] if cycle_ok else self.cycle_defects,
            "report_path": report_path,
        }

        self.mes_busy = True
        if self.operator_id:
            self.mes.set_operator(self.operator_id)
        self._set_mes_status("sending",
                             f"{len(image_paths)} image(s) + report for "
                             f"'{serial}'…")
        self.log(f"[MES] uploading {len(image_paths)} image(s) and the "
                 f"detection report for '{serial}'…")

        def done(res):
            ok, msg, path = res
            self.mes_busy = False
            if not self._alive:
                return
            if ok:
                self._set_mes_status("sent", str(msg))
                self.log(f"[MES] ✔ stage report accepted — {msg}")
                self.app.comm_log.add("EVT", f"MES stage report '{serial}' "
                                             f"OK: {msg}")
            else:
                self._set_mes_status("senderr", str(msg))
                self.log(f"[MES] ✖ stage report FAILED — {msg}")
                self.app.comm_log.add("ERR", f"MES stage report '{serial}' "
                                             f"FAILED: {msg}")
            if path:
                self.log(f"[MES] report: {path}")
            # only now is the unit finished -> ready for the next one
            self._start_next_cycle()

        self.mes.submit_cycle_async(bundle, done)
        return True

    # ------------------------------------------------------------------
    # CAMERA QR  (part read by camera; Operator ID still uses the scanner)
    #
    #   operator presses  📷 SCAN QR
    #     -> REQ signal to the client ("present the part")
    #     -> client answers with TRIGGER  (part is in position)
    #     -> camera reads the master QR  (multi-frame consensus)
    #     -> OK / NG signal to the client, then the normal barcode flow
    #        (MES previous-stage check, prefix + value to the client)
    #   No TRIGGER configured / no client connected -> reads immediately.
    # ------------------------------------------------------------------
    def _qr_sig(self, key):
        return str(self.comm_cfg.data.get("qr", {}).get(key, "") or "").strip()

    def _qr_max_tries(self):
        return max(1, int(self.app.app_config.data["qr"].get("max_tries", 10)))

    def _qr_client_ready(self):
        eng = self.app.engine
        return bool(eng.enabled and eng.client_count() > 0)

    def _qr_close_toast(self):
        t, self._qr_toast = self._qr_toast, None
        if t is not None:
            try:
                t.close()
            except RuntimeError:
                pass

    def _qr_popup(self, title, text, auto_ms=0, critical=False):
        """auto_ms > 0: non-blocking popup that closes itself (retry notices
        must not stop the retry loop).  auto_ms == 0: normal modal popup."""
        if not auto_ms:
            (QMessageBox.critical if critical else QMessageBox.warning)(
                self, title, text)
            return
        self._qr_close_toast()
        box = QMessageBox(QMessageBox.Icon.Warning, title, text,
                          QMessageBox.StandardButton.NoButton, self)
        box.setWindowModality(Qt.WindowModality.NonModal)
        box.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        box.show()
        self._qr_toast = box

        def close():
            try:
                box.close()
            except RuntimeError:
                pass
        QTimer.singleShot(auto_ms, close)

    def _qr_precheck(self, interactive):
        if not self._alive or not self.qr_scan_active:
            return False
        if not self.qr_master:
            self.log("[QR] no master QR configured — build a scene and add "
                     "one (5. QR Scanner)")
            if interactive:
                QMessageBox.warning(
                    self, "No master QR",
                    "Camera QR scanning is selected, but this model has no "
                    "master QR.\n\nBuild a scene and use  5. QR Scanner → "
                    "Add QR  to create one (answer Yes to 'master').")
            return False
        if self.busy:
            self.log("[QR] busy — a detection run is in progress")
            return False
        return True

    def _qr_status(self):
        """(text, colour) for the panel while there is no scan yet."""
        n, mx = self.qr_attempt, self._qr_max_tries()
        if self.qr_state == "waiting":
            return (f"⏳ attempt {n}/{mx}: waiting for client signal "
                    f"'{self._qr_sig('TRIGGER')}'…  (press again to cancel)",
                    "#74c0fc")
        if self.qr_state == "scanning":
            return f"📷 attempt {n}/{mx}: reading QR…", "#74c0fc"
        if self.qr_state == "retrying":
            return (f"↻ not read (attempt {n}/{mx}) — sending the signal "
                    "again…  (press to cancel)", "#ffa94d")
        if self.qr_msg:
            return f"✖ {self.qr_msg}", "#ff8787"
        if not self.qr_master:
            return "⚠ no master QR configured", "#ff8787"
        return ("① press 📷 SCAN QR / F2  (detection blocked)", "#ffd43b")

    def _qr_sync_buttons(self):
        b = self.scan_btn_qr
        if b is None:
            return
        if self.qr_state in ("waiting", "retrying"):
            b.setText("✖  Cancel")
            apply_btn_color(b, "#e03131")
            b.setEnabled(True)
        elif self.qr_state == "scanning":
            b.setText("📷  reading…")
            apply_btn_color(b, "#3b6ef5")
            b.setEnabled(False)
        else:
            b.setText("📷  SCAN QR  (F2)")
            apply_btn_color(b, "#3b6ef5")
            b.setEnabled(True)
        if self.btn_qr_now is not None:
            self.btn_qr_now.setEnabled(self.qr_state != "scanning")

    def _qr_reset_for_new_scan(self):
        self.scan_value = None
        self.scan_sent = False
        self.qr_msg = ""
        self.qr_attempt = 0
        self._qr_close_toast()
        self._set_mes_status("idle", "")
        self._mes_reset_cycle_data()

    def qr_button_pressed(self):
        """The master button — pressed once per cycle."""
        if not self._qr_precheck(interactive=True):
            return
        if self.qr_state in ("waiting", "retrying"):     # press = cancel
            self._qr_gen += 1
            self.qr_state = "idle"
            self.qr_msg = "cancelled"
            self._qr_close_toast()
            self.log("[QR] cancelled by the operator")
            self._update_scan_panel()
            return
        if self.qr_state == "scanning":
            self.log("[QR] already reading — please wait")
            return
        self._qr_reset_for_new_scan()
        self._qr_request()

    def _qr_request(self):
        """One attempt: send the request signal, then wait for the client's
        answer (trigger), then read.  Called for attempt 1 and for every
        retry (the signal is sent again each time)."""
        if not self._alive or not self.qr_scan_active:
            return
        self.qr_attempt += 1
        n, mx = self.qr_attempt, self._qr_max_tries()
        eng = self.app.engine
        req, trig = self._qr_sig("REQ"), self._qr_sig("TRIGGER")
        locked = self.app.is_locked()

        if not self._qr_client_ready():
            why = ("communication is OFF" if not eng.enabled
                   else "no client is connected")
            if locked:
                # operator mode: nothing may run without the client
                self.qr_state = "idle"
                self.qr_msg = f"{why} — QR scan not started"
                self.log(f"[QR] ✖ {why} — scan NOT started (locked mode)")
                self.app.comm_log.add("EVT", f"QR scan blocked — {why} "
                                             "(locked)")
                self._update_scan_panel()
                self._qr_popup("Client not connected",
                               f"{why[0].upper() + why[1:]}.\n\nThe QR scan "
                               "cannot start in operator (locked) mode.\n"
                               "Check the connection, then press SCAN QR "
                               "again.", critical=True)
                return
            # engineering mode: warn, then read anyway
            if req:
                eng.send(req, why=f"QR scan request {n}/{mx}")
            self.log(f"[QR] ⚠ {why} — reading the camera anyway "
                     "(unlocked mode)")
            if n == 1:
                self._qr_popup("Client not connected",
                               f"{why[0].upper() + why[1:]}.\n\nUnlocked "
                               "mode: the QR is read now without waiting for "
                               "the client.")
                if not self._alive:
                    return
            self._qr_begin("MANUAL")
            return

        if req:
            eng.send(req, why=f"QR scan request {n}/{mx}")
        else:
            self.log("[QR] no request signal set (Comm Config → CAMERA QR)")
        if not trig:
            self.log("[QR] no trigger signal set — reading the camera now")
            self._qr_begin("MANUAL")
            return
        self._qr_gen += 1
        gen = self._qr_gen
        self.qr_state = "waiting"
        wait = float(self.app.app_config.data["qr"].get("trigger_wait_s", 20))
        self.log(f"[QR] attempt {n}/{mx}: request sent — waiting for the "
                 f"client's signal '{trig}' (max {wait:.0f}s)")
        self._update_scan_panel()
        QTimer.singleShot(int(wait * 1000),
                          lambda: self._qr_wait_expired(gen))

    def qr_scan_now(self):
        """Engineering override: read now, do not wait for the client."""
        if self.app.is_locked() or not self._qr_precheck(interactive=True):
            return
        if self.qr_state == "scanning":
            return
        if self.qr_state in ("idle", "retrying"):
            self._qr_reset_for_new_scan()
            self.qr_attempt = 1
        self.log("[QR] operator: reading the camera now (not waiting for "
                 "the client)")
        self._qr_begin("MANUAL")

    def qr_client_trigger(self, source):
        """The client's answer.  Accepted while we are waiting for it; with
        no scan yet it can also start a read by itself."""
        if not self._qr_precheck(interactive=False):
            return
        if self.qr_state == "scanning":
            self.log(f"[QR] signal from {source} ignored — already reading")
            return
        if self.qr_state == "retrying":
            self.log(f"[QR] signal from {source} ignored — retry pending")
            return
        if self.qr_state == "idle":
            if self.scan_value:
                self.log(f"[QR] signal from {source} ignored — this cycle "
                         f"already has a scan ('{self.scan_value}')")
                return
            self._qr_reset_for_new_scan()         # client-initiated
            self.qr_attempt = 1
        self.log(f"[QR] client signal received from {source}")
        self._qr_begin(source)

    def _qr_wait_expired(self, gen):
        if (not self._alive or gen != self._qr_gen
                or self.qr_state != "waiting"):
            return
        self._qr_attempt_failed("the client did not send its signal in time",
                                None)

    def _qr_begin(self, source):
        self._qr_gen += 1
        gen = self._qr_gen
        self.qr_state = "scanning"
        self.qr_msg = ""
        self._qr_stop = stop = threading.Event()
        qc = self.app.app_config.data["qr"]
        rules = QrRules.from_dict(self.qr_master)
        roi = self.qr_master.get("roi_norm")
        timeout = float(qc.get("timeout_s", 6.0))
        agree = int(qc.get("min_agree", 2))
        src = self.app.camera.qr_source()
        self.log(f"[QR] attempt {self.qr_attempt}/{self._qr_max_tries()}: "
                 f"reading '{self.qr_master['name']}' — {rules.describe()}, "
                 f"{agree} matching frames needed, max {timeout:g}s")
        self._update_scan_panel()

        def work():
            try:
                res = QrReader().read_stream(src, rules, roi,
                                             timeout_s=timeout,
                                             min_agree=agree,
                                             stop=stop.is_set)
            except Exception:
                res = QrResult(reason="QR reader error:\n"
                                      + traceback.format_exc())
            self.after(0, lambda: self._qr_done(res, source, gen))
        threading.Thread(target=work, daemon=True).start()

    def _qr_done(self, res, source, gen):
        if not self._alive or gen != self._qr_gen:
            return
        self.qr_state = "idle"
        if res.ok:
            self._qr_close_toast()
            self.log(f"[QR] ✔ '{res.value}'  (attempt {self.qr_attempt}, "
                     f"{res.frames} frames, {res.elapsed:.2f}s, "
                     f"{res.backend})")
            self.app.comm_log.add("EVT", f"QR read OK '{res.value}' "
                                         f"[{source}] attempt "
                                         f"{self.qr_attempt}")
            ok_sig = self._qr_sig("OK")
            if ok_sig:
                self.app.engine.send(ok_sig, why="QR read OK")
            # MES active  -> previous-stage check; the value (with prefix) is
            #                sent to the client only if MES says PASS.
            # MES not set -> nothing to check: send it right away.
            self._handle_barcode(res.value)
            if (not self.mes_active
                    and self.scanner_cfg.get("send_to_client", True)):
                self._try_send_barcode()
                self._update_scan_panel()
            return
        if res.reason == "cancelled":
            self.qr_msg = res.reason
            self.log("[QR] read cancelled")
            self._update_scan_panel()
            return
        self._qr_save_failed(res)
        self._qr_attempt_failed(res.reason, res, source)

    def _qr_attempt_failed(self, reason, res, source="MANUAL"):
        n, mx = self.qr_attempt, self._qr_max_tries()
        self.app.comm_log.add("EVT", f"QR attempt {n}/{mx} FAILED "
                                     f"[{source}]: {reason}")
        if n < mx:
            delay = float(self.app.app_config.data["qr"]
                          .get("retry_delay_s", 1.5))
            self._qr_gen += 1
            gen = self._qr_gen
            self.qr_state = "retrying"
            self.log(f"[QR] ✖ attempt {n}/{mx}: {reason} — trying again, "
                     "sending the signal again…")
            self._update_scan_panel()
            self._qr_popup("QR not read",
                           f"Not able to read the QR code "
                           f"(attempt {n} of {mx}).\n\n{reason}\n\n"
                           "Trying again — the signal is sent again…",
                           auto_ms=int(max(delay, 0.5) * 1000))
            QTimer.singleShot(int(delay * 1000),
                              lambda: self._qr_retry(gen))
            return
        # out of attempts
        self.qr_state = "idle"
        self.qr_msg = f"QR not read after {mx} attempts — {reason}"
        self._qr_close_toast()
        self.log(f"[QR] ✖✖ not read after {mx} attempts — {reason}")
        self.app.comm_log.add("EVT", f"QR FAILED after {mx} attempts")
        ng_sig = self._qr_sig("NG")
        if ng_sig:
            self.app.engine.send(ng_sig, why=f"QR read NG after {mx} tries")
        self._update_scan_panel()
        self._qr_popup("QR not readable",
                       f"The QR code could not be read after {mx} "
                       f"attempts.\n\n{reason}\n\nCheck the part, the "
                       "lighting and the focus, then press SCAN QR again.",
                       critical=True)

    def _qr_retry(self, gen):
        if (not self._alive or gen != self._qr_gen
                or self.qr_state != "retrying"):
            return
        self._qr_close_toast()
        self._qr_request()

    def _qr_save_failed(self, res):
        """Keep the ROI image of a failed read so it can be diagnosed
        (focus, glare, wrong region…).  Only the newest 50 are kept."""
        if (not self.app.app_config.data["qr"].get("save_failed", True)
                or res.last_crop is None):
            return
        try:
            d = os.path.join(MODELS_BASE, "qr_fail")
            os.makedirs(d, exist_ok=True)
            fn = os.path.join(
                d, f"{self.head}_{datetime.datetime.now():%Y%m%d_%H%M%S}.png")
            cv2.imwrite(fn, res.last_crop)
            files = sorted((os.path.join(d, f) for f in os.listdir(d)),
                           key=os.path.getmtime)
            for old in files[:-50]:
                os.remove(old)
            self.log(f"[QR] failed region saved: {fn}")
        except Exception:
            pass

    # -- QR scanners that live inside scenes (read on every scene run) -------
    def _qr_box(self, q, res, fw, fh):
        """Rectangle for the result image (fw x fh = the preview frame)."""
        roi = q.get("roi_norm")
        if roi:
            x, y, w, h = roi
            return int(x * fw), int(y * fh), int(w * fw), int(h * fh)
        bb = res.bbox()
        rw, rh = res.frame_size
        if bb and rw and rh:
            return (int(bb[0] * fw / rw), int(bb[1] * fh / rh),
                    int(bb[2] * fw / rw), int(bb[3] * fh / rh))
        return None

    def _qr_scene_checks(self, scene, fw, fh, results, child_boxes, logs):
        """Worker thread.  Every camera QR scanner of the scene becomes a
        result entry 'QR:<name>' so it counts in the scene / cycle OK-NG."""
        qc = self.app.app_config.data["qr"]
        rules_txt = None
        added = False
        for q in self.app.heads.qr_scanners_for_scene(self.head, scene):
            if q["master"]:        # the master is read by the button, never
                continue           # as part of a scene run
            rules = QrRules.from_dict(q)
            try:
                res = QrReader().read_stream(
                    self.app.camera.qr_source(), rules, q.get("roi_norm"),
                    timeout_s=float(qc.get("scene_timeout_s", 3.0)),
                    min_agree=int(qc.get("min_agree", 2)))
            except Exception:
                res = QrResult(reason="QR reader error:\n"
                                      + traceback.format_exc())
            key = f"QR:{q['name']}"
            results.setdefault(scene, {})[key] = {
                "detected": 1 if res.ok else 0, "master": 1,
                "ok": bool(res.ok), "pins": {},
                "qr": res.value, "reason": res.reason}
            logs.append(f" {key:<12} {'OK ' if res.ok else 'NG '}"
                        + (f" '{res.value}'" if res.ok else f" {res.reason}"))
            box = self._qr_box(q, res, fw, fh)
            if box:
                child_boxes.append((*box, f"{key} {'OK' if res.ok else 'NG'}",
                                    bool(res.ok)))
            added = True
        return added

    def _start_next_cycle(self):
        """Reset the scan gate and (optionally) pop the scan box."""
        if not self._alive:
            return
        self.scan_value = None
        self.scan_sent = False
        self.qr_state = "idle"
        self.qr_msg = ""
        self.qr_attempt = 0
        self._qr_gen += 1              # drop any pending retry / wait timer
        self._qr_close_toast()
        self._update_scan_panel()
        self._set_mes_status("idle", "")
        self._mes_reset_cycle_data()
        if self.qr_scan_active:
            self.log("[QR] ready — press 📷 SCAN QR (F2) for the next part")
        elif self.scan_active:
            self.log("[scanner] ready — scan the next barcode")
            if self.scanner_cfg.get("auto_popup", True):
                QTimer.singleShot(300, self.open_quick_scan)

    def _handle_barcode(self, code):
        # a new serial = a new unit: nothing of the previous one may reach MES
        self._mes_reset_cycle_data()
        self.cycle_scene_results = {}
        self.scan_value = code
        self.scan_time = datetime.datetime.now().strftime("%H:%M:%S")
        self.scan_sent = False
        op_info = f" (Op: {self.operator_id})" if self.operator_id else ""
        self.app.comm_log.add("EVT", f"barcode scanned: '{code}'{op_info}")
        self.log(f"[scanner] scanned '{code}'{op_info}")
        
        self._update_scan_panel()
        self.mes_check_previous(code)
        # if self.scanner_cfg.get("send_to_client", True):
        #         if self.mes_prev_state in ("pass", "all_passed"):
        #             self._try_send_barcode()

    def _try_send_barcode(self):
        if not self.scan_value:
            return
        signal = str(self.scanner_cfg.get("prefix", "")) + self.scan_value
        if self.app.engine.enabled:
            n = self.app.engine.send(signal, why="scanner data")
            self.scan_sent = n > 0
            if not self.scan_sent:
                self.log("[scanner] no client connected — barcode NOT "
                         "delivered yet (Resend when connected)")
        else:
            self.scan_sent = False
            self.app.comm_log.add("EVT",
                                  f"barcode '{signal}' NOT sent (comm OFF)")
            self.log("[scanner] comm OFF — barcode not sent to client")

    def resend_barcode(self):
        if not self.scan_value:
            self.log("[scanner] nothing to resend — scan first")
            return
        self._try_send_barcode()
        self._update_scan_panel()

    def clear_barcode(self):
        self.scan_value = None
        self.scan_sent = False
        self._mes_reset_cycle_data()
        self.log("[scanner] cleared — scan next barcode")
        self._update_scan_panel()

    def _update_scan_panel(self):
        if not self.scan_status_lbl:
            return
        if self.qr_scan_active:
            self._qr_sync_buttons()
            if not self.scan_value:
                txt, col = self._qr_status()
                self.scan_status_lbl.setText(txt)
                self.scan_status_lbl.setStyleSheet(
                    f"color: {col}; font-weight: bold;")
                self.scan_value_lbl.setText("—")
                return
        if not self.scan_value:
            self.scan_status_lbl.setText("① waiting for scan…  "
                                         "press SCAN / F2  (detection "
                                         "blocked)")
            self.scan_status_lbl.setStyleSheet("color: #ffd43b;"
                                               " font-weight: bold;")
            self.scan_value_lbl.setText("—")
            return
        self.scan_value_lbl.setText(
            f"{self.scan_value}   ({self.scan_time})")
        if self.scanner_cfg.get("send_to_client", True):
            if self.scan_sent:
                self.scan_status_lbl.setText(
                    "✔ scanned ")
                self.scan_status_lbl.setStyleSheet("color: green;"
                                                   " font-weight: bold;")
            else:
                self.scan_status_lbl.setText(
                    "⚠ scanned Not sent to client")
                self.scan_status_lbl.setStyleSheet("color: green;"
                                                   " font-weight: bold; font-size: 13pt;")
        else:
            self.scan_status_lbl.setText("✔ scanned — detection allowed")
            self.scan_status_lbl.setStyleSheet("color: #69db7c;"
                                               " font-weight: bold;  font-size: 13pt;")

    def scanner_gate(self):
        """(allowed, reason). Scanning is mandatory before detection, but a
        failed/pending SEND does not block — scenes can run even when no
        client is connected (barcode is kept and can be resent)."""
        sc = self.scanner_cfg
        if not sc.get("enabled"):
            return self.mes_gate()
        if not self.scan_value:
            if self.qr_scan_active:
                if not self.qr_master:
                    return False, ("camera QR scanning is selected but the "
                                   "model has no master QR — add one in a "
                                   "scene (5. QR Scanner)")
                return False, ("scan the QR first — press 📷 SCAN QR (F2); "
                               "the scan step runs before the scenes")
            return False, ("scan the barcode first — press SCAN (F2); "
                           "the scanner step runs before the scenes")
        return self.mes_gate()

    def _end_cycle(self, all_ok):
        """Called when EVERY scene of the model has been run once."""
        self.cycle_scene_results = {}
        serial = self.scan_value
        # MES first: the next unit may only be scanned once this one has been
        # reported.  _mes_submit_cycle() calls _start_next_cycle() when the
        # upload finishes (pass or fail).
        if self.mes_active and self._mes_submit_cycle(serial,
                                                                  all_ok):
            self.log("[scanner] cycle complete — waiting for MES…")
            return
        if self.scan_active:
            self.log("[scanner] cycle complete — scan next barcode")
        self._start_next_cycle()

    # ------------------------------------------------------------------
    def reload_models(self):
        if self.parent_btn_lay is not None:
            clear_layout(self.parent_btn_lay)
        self._roi_cache = {}
        self._children_cache = {}
        self.compare_targets = [("WHOLE VIEW (all ROIs)", None, None)]
        parents = self.app.heads.list_parents(self.head)
        if not parents and self.parent_btn_lay is not None:
            e = QLabel("No scenes yet.\nPress + Add Scene")
            e.setStyleSheet("color: #666; font-size: 10pt;")
            e.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self.parent_btn_lay.addWidget(e)
        for p in parents:
            if self.parent_btn_lay is not None:
                n = len(self.parent_store.list_children(p))
                self.parent_btn_lay.addWidget(styled_btn(
                    self.parent_btn_frame, f"▶  {p}   ({n} area)",
                    lambda pp=p: self.execute(("parent", pp), "MANUAL")))
            try:
                cm = self.parent_store.get_child_manager(p)
                regs, loaded = [], []
                for cname in self.parent_store.list_children(p):
                    d = cm.load(cname)
                    if d:
                        regs.append((cname, d["reg"]))
                        loaded.append((cname, d))
                        self.compare_targets.append(
                            (f"{p}  /  {cname}", p, cname))
                self._roi_cache[p] = regs
                self._children_cache[p] = loaded
            except Exception:
                pass
        self.compare_sel.clear()
        self.compare_sel.addItems([t[0] for t in self.compare_targets])
        if self.compare_targets:
            self.compare_sel.setCurrentIndex(0)

    def _master_whole(self):
        for p, loaded in self._children_cache.items():
            for cname, d in loaded:
                if d.get("frame") is not None:
                    img = d["frame"].copy()
                    for pp, regs in self._roi_cache.items():
                        for cn, reg in regs:
                            x, y, w, h = [int(v) for v in reg]
                            cv2.rectangle(img, (x, y), (x + w, y + h),
                                          (255, 120, 0), 2)
                            cv2.putText(img, f"{pp}/{cn}",
                                        (x, max(14, y - 6)),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                        (255, 120, 0), 1, cv2.LINE_AA)
                    return img
        return None

    # ------------------------------------------------------------------
    def _tick(self):
        if not self._alive:
            return
        if not self.busy:
            if self.view == self.VIEW_LIVE:
                frame = self.app.camera.get_frame()
                if frame is not None:
                    disp = frame.copy()
                    for regs in self._roi_cache.values():
                        for _, reg in regs:
                            x, y, w, h = [int(v) for v in reg]
                            cv2.rectangle(disp, (x, y), (x + w, y + h),
                                          (255, 0, 0), 2)
                    self.canvas.show(disp)
                elif self.app.camera.error:
                    self._show_cam_error()
            elif self.view == self.VIEW_COMPARE:
                self._tick_compare()

    def _show_cam_error(self):
        self.canvas.show_message(self.app.camera.error)

    # ------------------------------------------------------------------
    def enter_compare(self):
        if not self.compare_targets or not self._children_cache:
            QMessageBox.information(self, "No master",
                                    "No saved areas yet — nothing to "
                                    "compare.")
            return
        self.view = self.VIEW_COMPARE
        self.result_ctl.setVisible(False)
        self.compare_ctl.setVisible(True)
        self.canvas.fit()

    def set_compare_mode(self, mode):
        self.compare_mode = mode
        apply_btn_color(self.btn_side,
                        "#3b6ef5" if mode == "side" else "#44444e")
        apply_btn_color(self.btn_overlay,
                        "#3b6ef5" if mode == "overlay" else "#44444e")

    def _tick_compare(self):
        frame = self.app.camera.get_frame()
        if frame is None:
            if self.app.camera.error:
                self._show_cam_error()
            return
        idx = self.compare_sel.currentIndex()
        if idx < 0 or idx >= len(self.compare_targets):
            return
        label, p, c = self.compare_targets[idx]
        if p is None:
            master = self._master_whole()
            current = frame.copy()
            for regs in self._roi_cache.values():
                for _, reg in regs:
                    x, y, w, h = [int(v) for v in reg]
                    cv2.rectangle(current, (x, y), (x + w, y + h),
                                  (255, 120, 0), 2)
        else:
            d = dict(self._children_cache.get(p, [])).get(c)
            if d is None:
                return
            master = d["region"]
            x, y, w, h = [int(v) for v in d["reg"]]
            fh, fw = frame.shape[:2]
            x0, y0 = max(0, x), max(0, y)
            x1, y1 = min(fw, x + w), min(fh, y + h)
            if x1 <= x0 or y1 <= y0:
                return
            current = frame[y0:y1, x0:x1]
        if master is None:
            return
        if self.compare_mode == "side":
            img = side_by_side(master, current, "MASTER",
                               f"CURRENT  ({label})")
        else:
            img = overlay_blend(master, current)
        self.canvas.show(img)

    def back_to_live(self):
        self.view = self.VIEW_LIVE
        self.result_ctl.setVisible(False)
        self.compare_ctl.setVisible(False)
        self.canvas.fit()

    def toggle_pinmap(self):
        self.result_key = "pins" if self.result_key == "result" else "result"
        self.btn_pinmap.setText(
            "Result" if self.result_key == "pins" else "Map")
        img = self.result_views.get(self.result_key)
        if img is not None:
            self.canvas.show(img)

    def auto_balance(self):
        ok = self.app.camera.auto_balance()
        self.log("auto_balance " + ("done" if ok
                                    else "FAILED (camera not ready)"))

    def add_parent(self):
        name, ok = QInputDialog.getText(self, "Add Scene", "New scene name:")
        if not ok or not name or not name.strip():
            return
        name = name.strip()
        if self.app.heads.scene_exists(self.head, name):
            QMessageBox.information(self, "Exists",
                                    f"Scene '{name}' already exists.")
            return
        self.app.show_build_parent(self.head, name)

    def finish_head(self):
        structure = self.app.heads.structure(self.head)
        if not any(children for children in structure.values()):
            QMessageBox.warning(
                self, "Detection not finished",
                "The model needs at least one scene with at least one area "
                "before it can be finished.")
            return
        r = QMessageBox.question(
            self, "Finish model",
            f"Mark model '{self.head}' as FINISHED?\n\n"
            "After this you can build the Communication part.")
        if r == QMessageBox.StandardButton.Yes:
            self.app.heads.set_finished(self.head, True)
            self.app.show_head(self.head, announce=False)

    def open_comm(self):
        if not self.app.heads.is_finished(self.head):
            QMessageBox.information(
                self, "Finish detection first",
                "First finish the detection part completely and press "
                "'🏁 Finish Model'. Then the Communication builder unlocks.")
            return
        self.app.show_comm(self.head)

    # ------------------------------------------------------------------
    # EXECUTION (manual buttons AND client triggers land here)
    # ------------------------------------------------------------------
    def execute(self, scope, source):
        if self.busy:
            self.log(f"BUSY — ignored trigger {scope} from {source}")
            self.app.comm_log.add("EVT", f"BUSY — trigger {scope} ignored")
            return
        allowed, reason = self.scanner_gate()
        if not allowed:
            self.log(f"BLOCKED [{source}]: {reason}")
            self.app.comm_log.add("EVT", f"run {scope} BLOCKED — {reason}")
            if source == "MANUAL":
                if (not self.scan_value and self.scan_active
                        and not self.qr_scan_active):
                    # just missing a scan — pop the box and retry
                    self.open_quick_scan()
                    if self.scan_value and not self.mes_active:
                        self.execute(scope, source)
                else:
                    QMessageBox.warning(self, "Cannot run", reason)
            return
        frame = self.app.camera.get_frame()
        if frame is None:
            self.log(f"No camera frame — cannot run {scope} ({source})")
            if source == "MANUAL":
                QMessageBox.critical(self, "No camera",
                                     "No camera frame available yet.")
            return
        if scope[0] == "head":
            plan = [(p, None) for p in self.app.heads.list_parents(self.head)]
        elif scope[0] == "parent":
            plan = [(scope[1], None)]
        else:
            plan = [(scope[1], {scope[2]})]
        if not plan:
            self.log("Nothing to run — model has no scenes.")
            return
        self.busy = True
        dist_thr = self.get_dist_thr()
        name = "/".join(scope) if scope[0] != "head" else "MODEL (all scenes)"
        self.log(f"\n── RUN {name}  [{source}] · thr {dist_thr:g}px ──")
        threading.Thread(target=self._worker,
                         args=(scope, plan, frame, dist_thr),
                         daemon=True).start()

    def _worker(self, scope, plan, frame, dist_thr):
        logs = []
        results = {}
        views = {}
        run_summary = None
        try:
            frame_h, frame_w = frame.shape[:2]
            out_frame = frame.copy()
            proc = np.ones_like(frame)
            child_boxes = []
            pin_overlays = []
            any_output = False
            for parent_name, child_filter in plan:
                children_data = self.parent_store.load_all_children(
                    parent_name)
                if child_filter is not None:
                    children_data = [(n, d) for n, d in children_data
                                     if n in child_filter]
                if not children_data:
                    logs.append(f"[{parent_name}] no loadable areas.")
                    continue
                results[parent_name] = {}
                for child_name, cd in children_data:
                    reg = cd["reg"]
                    cfg = cd["config"]
                    x, y, w, h = [int(v) for v in reg]
                    x0 = max(0, min(x, frame_w))
                    y0 = max(0, min(y, frame_h))
                    x1 = max(0, min(x + w, frame_w))
                    y1 = max(0, min(y + h, frame_h))
                    if x1 <= x0 or y1 <= y0:
                        logs.append(f"[{parent_name}/{child_name}] SKIPPED: "
                                    "ROI outside frame.")
                        results[parent_name][child_name] = {
                            "detected": 0,
                            "master": len(cfg.get("matches", [])),
                            "ok": False}
                        continue
                    region = frame[y0:y1, x0:x1]
                    matches, output, mask = run_detection(
                        detect_multi_scale_templates(
                            image=region, template=cd["templates"],
                            scale_range=tuple(cfg["scale_range"]),
                            scale_steps=cfg["scale_steps"],
                            threshold=cfg["threshold"],
                            max_matches=cfg["max_matches"],
                            visualize=True, t=time.time(),
                            config=cd, name=child_name))
                    ok, pins, summ = analyze_pins(cfg.get("matches", []),
                                                  matches, dist_thr)
                    master_n = summ["total"] if summ["total"] else \
                        len(cfg.get("matches", []))
                    results[parent_name][child_name] = {
                        "detected": len(matches), "master": master_n,
                        "ok": ok, "pins": summ}
                    if summ["total"]:
                        line = (f" {child_name:<12} "
                                f"{'OK ' if ok else 'NG '}"
                                f" {summ['in_range']}/{summ['total']} pins ok"
                                f"   shift {summ['mean_dx']:+.0f},"
                                f"{summ['mean_dy']:+.0f} px")
                        n_shift = sum(1 for p in pins
                                      if p["paired"] and not p["in_range"])
                        n_miss = sum(1 for p in pins if not p["paired"])
                        if n_shift:
                            line += (f"   {n_shift} too far "
                                     f"(max {summ['max_dist']:.0f} px)")
                        if n_miss:
                            miss_ids = ",".join(str(i + 1)
                                                for i, p in enumerate(pins)
                                                if not p["paired"])
                            line += f"   missing #{miss_ids}"
                        logs.append(line)
                    else:
                        logs.append(f" {child_name:<12} "
                                    f"{'OK ' if ok else 'NG '}"
                                    f" {len(matches)} found "
                                    "(no master pins)")
                    oh, ow = output.shape[:2]
                    out_frame[y0:y0 + oh, x0:x0 + ow] = output
                    if mask is not None:
                        mh, mw = mask.shape[:2]
                        mh = min(mh, y1 - y0)
                        mw = min(mw, x1 - x0)
                        proc[y0:y0 + mh, x0:x0 + mw] = mask[:mh, :mw]
                    if summ["total"]:
                        shift_txt = (f"  d({summ['mean_dx']:+.0f},"
                                     f"{summ['mean_dy']:+.0f})px")
                        count_txt = f"{summ['in_range']}/{summ['total']}"
                    else:
                        shift_txt = ""
                        count_txt = f"{len(matches)}/{master_n}"
                    child_boxes.append(
                        (x0, y0, x1 - x0, y1 - y0,
                         f"{child_name} {'OK' if ok else 'NG'} "
                         f"{count_txt}{shift_txt}", ok))
                    pin_overlays.append((pins, x0, y0))
                    any_output = True

                # camera QR scanners of this scene (full-scene / model runs)
                if self.qr_camera and child_filter is None:
                    if self._qr_scene_checks(parent_name, frame_w, frame_h,
                                             results, child_boxes, logs):
                        any_output = True

            if any_output:
                for pins, ox, oy in pin_overlays:
                    draw_pin_overlay(out_frame, pins, ox, oy, dist_thr)
                for (bx, by, bw, bh, text, ok) in child_boxes:
                    label_box(out_frame, bx, by, bw, bh, text, ok)
                    label_box(proc, bx, by, bw, bh, text, ok)
                all_ok = all(r["ok"] for pr in results.values()
                             for r in pr.values())
                n_ok = sum(1 for pr in results.values()
                           for r in pr.values() if r["ok"])
                n_all = sum(len(pr) for pr in results.values())
                title = ("MODEL" if scope[0] == "head"
                         else f"SCENE {scope[1]}" if scope[0] == "parent"
                         else f"AREA {scope[1]}/{scope[2]}")
                txt = (f"{title}  —  {'OK' if all_ok else 'NG'}   "
                       f"({n_ok}/{n_all} areas OK · pin thr {dist_thr:g}px)  "
                       + datetime.datetime.now().strftime("%H:%M:%S"))
                logs.append(f" RESULT       {'OK' if all_ok else 'NG'}"
                            f"   ({n_ok}/{n_all} areas ok)")
                views["result"] = np.vstack(
                    [banner(out_frame.shape[1], txt, all_ok), out_frame])
                views["pins"] = np.vstack(
                    [banner(proc.shape[1], txt + "   [PIN MAP]", all_ok),
                     proc])
                views["camera"] = frame.copy()      # full view, no overlays
                run_summary = (all_ok, f"{title}  ·  {n_ok}/{n_all} areas ok"
                                       f"  ·  thr {dist_thr:g}px")
        except Exception:
            logs.append("ERROR:\n" + traceback.format_exc())

        full_scenes = [p for p, f in plan if f is None]

        def apply():
            if not self._alive:
                return
            for m in logs:
                self.log(m)
            if run_summary is not None:
                self.set_big_status(run_summary[0], run_summary[1])
            if views:
                self.result_views = views
                self.result_key = "result"
                self.btn_pinmap.setText("Map")
                self.view = self.VIEW_RESULT
                self.compare_ctl.setVisible(False)
                self.result_ctl.setVisible(True)
                self.canvas.fit()
                self.canvas.show(views["result"])
            self.busy = False
            if results:
                self._mes_collect(scope, logs, views, results)
                self.app.heads.add_result_counts(self.head, scope, results)
                try:
                    self.app.engine.send_results(self.comm_cfg, scope,
                                                 results)
                except Exception:
                    self.log("send_results ERROR:\n" + traceback.format_exc())
                # --- cycle tracking: mark each FULLY-run scene, and when
                #     every scene of the model has been run -> one cycle
                for scene in full_scenes:
                    if scene in results and results[scene]:
                        self.cycle_scene_results[scene] = all(
                            r["ok"] for r in results[scene].values())
                all_scenes = set(self.app.heads.list_parents(self.head))
                if all_scenes and all_scenes <= set(
                        self.cycle_scene_results):
                    cycle_ok = all(self.cycle_scene_results[s]
                                   for s in all_scenes)
                    self.app.heads.add_cycle_count(self.head, cycle_ok)
                    self.log(f" CYCLE COMPLETE — "
                             f"{'OK' if cycle_ok else 'NG'}  "
                             f"(all {len(all_scenes)} scene(s) run)")
                    self.app.comm_log.add(
                        "EVT", f"cycle complete on model '{self.head}': "
                               f"{'OK' if cycle_ok else 'NG'}")
                    self._end_cycle(cycle_ok)
                elif all_scenes:
                    left = sorted(all_scenes
                                  - set(self.cycle_scene_results))
                    if left:
                        self.log(f" cycle progress: "
                                 f"{len(self.cycle_scene_results)}"
                                 f"/{len(all_scenes)} scenes — "
                                 "left: " + ", ".join(left))
                self.refresh_counters()
        self.after(0, apply)


# ============================================================================
# SCREEN 3 — COMMUNICATION BUILDER (scanner node FIRST, then model/scenes)
# ============================================================================
class CommScreen(BaseScreen):
    def __init__(self, master, app, head_name):
        super().__init__(master, app)
        self.head = head_name
        self.cfg = CommConfig(app.heads, head_name)
        self.current_sel = None
        self.form_vars = {}
        self._suppress_sel = False

        root = QVBoxLayout(self)
        root.setContentsMargins(10, 8, 10, 8)
        top = QHBoxLayout()
        top.addWidget(styled_btn(self, "← Back to Model",
                                 lambda: app.show_head(head_name,
                                                       announce=False),
                                 color="#44444e"))
        t = QLabel(f"COMMUNICATION —  {head_name}  (software / TCP)")
        t.setStyleSheet("color: white; font-size: 14pt; font-weight: bold;")
        top.addWidget(t)
        top.addStretch()
        top.addWidget(styled_btn(self, "🖥 Monitor / Connection",
                                 lambda: CommMonitor(self, app).show(),
                                 color="#845ef7"))
        _mes_on = app.heads.get_mes(head_name).get("enabled")
        top.addWidget(styled_btn(
            self, "MES Config", self.open_mes_cfg,
            color="#0ca678" if _mes_on else "#44444e"))
        top.addWidget(styled_btn(self, "Show YAML", self.show_yaml,
                                 color="#3b6ef5"))
        root.addLayout(top)

        self.progress_lbl = QLabel("")
        self.progress_lbl.setStyleSheet("color: #ffd43b; font-size: 11pt;"
                                        " font-weight: bold;")
        root.addWidget(self.progress_lbl)
        self.remaining_lbl = QLabel("")
        self.remaining_lbl.setStyleSheet(f"color: {C_MUTED};"
                                         " font-size: 9pt;")
        self.remaining_lbl.setWordWrap(True)
        root.addWidget(self.remaining_lbl)

        body = QHBoxLayout()
        root.addLayout(body, stretch=1)
        left = panel()
        llay = QVBoxLayout(left)
        llay.setContentsMargins(8, 6, 8, 8)
        lt = QLabel("STRUCTURE — ✔ done   ✖ missing   – optional/skipped")
        lt.setStyleSheet(f"color: {C_MUTED}; font-weight: bold;")
        lt.setAlignment(Qt.AlignmentFlag.AlignCenter)
        llay.addWidget(lt)
        self.tree = QTreeWidget()
        self.tree.setHeaderHidden(True)
        self.tree.itemSelectionChanged.connect(self._on_tree_select)
        llay.addWidget(self.tree, stretch=1)
        body.addWidget(left, stretch=1)

        self.form_holder = panel()
        self.form_holder.setFixedWidth(520)
        self.form_lay = QVBoxLayout(self.form_holder)
        self.form_lay.setContentsMargins(0, 0, 0, 0)
        body.addWidget(self.form_holder)
        self.refresh_tree(select=("scanner",))

    # ------------------------------------------------------------------
    def _node_status(self, kind, p=None, c=None):
        d = self.cfg.data
        if kind == "scanner":
            # quick-input scanning has nothing that can be left blank
            return (True, [])
        if kind == "qr":
            miss = [f for f in CommConfig.QR_FIELDS
                    if not str(d["qr"].get(f, "")).strip()]
            return (len(miss) == 0, miss)
        if kind == "head":
            miss = [f for f in CommConfig.HEAD_FIELDS
                    if not str(d["head"].get(f, "")).strip()]
        elif kind == "parent":
            miss = [f for f in CommConfig.PARENT_FIELDS
                    if not str(d["parents"][p].get(f, "")).strip()]
        else:
            cc = d["parents"][p]["children"][c]
            miss = [f for f in CommConfig.CHILD_FIELDS
                    if not str(cc.get(f, "")).strip()]
        return (len(miss) == 0, miss)

    def _mk_item(self, parent, iid, text, tag, sel_tuple):
        it = QTreeWidgetItem([text])
        colors = {"done": "#69db7c", "todo": "#ff8787",
                  "optional": "#6b6b75"}
        it.setForeground(0, QBrush(QColor(colors[tag])))
        it.setData(0, Qt.ItemDataRole.UserRole, sel_tuple)
        it.setData(0, Qt.ItemDataRole.UserRole + 1, iid)
        if parent is None:
            self.tree.addTopLevelItem(it)
        else:
            parent.addChild(it)
        self._items[iid] = it
        return it

    def refresh_tree(self, select=None):
        sel = select or self.current_sel
        self._suppress_sel = True
        self.tree.clear()
        self._items = {}

        def label(name, done, miss):
            return f"{'✔' if done else '✖'}  {name}" + \
                   ("" if done else f"   (left: {', '.join(miss)})")

        # scanner node FIRST — it runs before the scenes
        sc = self.app.heads.get_scanner(self.head)
        done, miss = self._node_status("scanner")
        sc_name = ("BARCODE SCANNER (step 1 — before scenes)"
                   if sc.get("enabled") else
                   "BARCODE SCANNER (disabled)")
        self._mk_item(None, "scanner",
                      (label(sc_name, done, miss) if sc.get("enabled")
                       else f"–  {sc_name}"),
                      (("done" if done else "todo")
                       if sc.get("enabled") else "optional"),
                      ("scanner",))

        if self.app.heads.camera_qr_enabled(self.head):
            master = self.app.heads.get_master_qr(self.head)
            if master:
                done, miss = self._node_status("qr")
                self._mk_item(
                    None, "qr",
                    label(f"CAMERA QR  (master '{master['name']}' — "
                          "request / trigger / result)", done, miss),
                    "done" if done else "todo", ("qr",))
            else:
                self._mk_item(None, "qr",
                              "–  CAMERA QR (no master QR yet — add one "
                              "in a scene)", "optional", ("qr",))

        done, miss = self._node_status("head")
        head_item = self._mk_item(
            None, "head",
            label(f"MODEL  '{self.head}'  (Select-signal + run trigger)",
                  done, miss),
            "done" if done else "todo", ("head",))
        head_item.setExpanded(True)
        for p in self.cfg.data["parents"]:
            done, miss = self._node_status("parent", p)
            pid = f"parent::{p}"
            area_wise = bool(self.cfg.data["parents"][p]
                             .get("area_wise", True))
            ptxt = label(f"scene: {p}", done, miss)
            if not area_wise:
                ptxt += "   [scene-only: trigger runs all areas]"
            p_item = self._mk_item(head_item, pid, ptxt,
                                   "done" if done else "todo",
                                   ("parent", p))
            p_item.setExpanded(True)
            for c in self.cfg.data["parents"][p]["children"]:
                cid = f"child::{p}::{c}"
                if not area_wise:
                    self._mk_item(p_item, cid,
                                  f"–  area: {c}   "
                                  "(optional — area-wise results off)",
                                  "optional", ("child", p, c))
                else:
                    done, miss = self._node_status("child", p, c)
                    self._mk_item(p_item, cid, label(f"area: {c}",
                                                     done, miss),
                                  "done" if done else "todo",
                                  ("child", p, c))
        self.tree.expandAll()

        dn, total, missing = self.cfg.progress()
        self.progress_lbl.setText(
            f"Progress: {dn} / {total} fields configured"
            + ("   —  COMPLETE ✔" if dn == total else ""))
        self.remaining_lbl.setText(
            "" if not missing else "Still left:  " + ",  ".join(missing))
        if sel:
            iid = {"scanner": "scanner", "qr": "qr", "head": "head",
                   "parent": f"parent::{sel[1]}" if len(sel) > 1 else None,
                   "child": (f"child::{sel[1]}::{sel[2]}"
                             if len(sel) > 2 else None)}[sel[0]]
            if iid and iid in self._items:
                self.tree.setCurrentItem(self._items[iid])
                self.tree.scrollToItem(self._items[iid])
        self._suppress_sel = False
        self._build_form(sel or ("scanner",))

    def _on_tree_select(self):
        if self._suppress_sel:
            return
        items = self.tree.selectedItems()
        if not items:
            return
        sel = items[0].data(0, Qt.ItemDataRole.UserRole)
        if sel:
            self._build_form(tuple(sel))

    # ------------------------------------------------------------------
    def _clear_form(self):
        clear_layout(self.form_lay)
        self.form_vars = {}

    def _add_field(self, lay, label, initial, hint=""):
        row = QHBoxLayout()
        row.setContentsMargins(12, 4, 12, 0)
        l = QLabel(label)
        l.setStyleSheet("color: #c8c8d0;")
        l.setFixedWidth(160)
        row.addWidget(l)
        e = QLineEdit(str(initial))
        row.addWidget(e, stretch=1)
        lay.addLayout(row)
        if hint:
            h = QLabel(hint)
            h.setStyleSheet(f"color: {C_FAINT}; font-size: 8pt;")
            h.setContentsMargins(12, 0, 12, 0)
            lay.addWidget(h)
        return SVar(e)

    def _chk(self, lay, text, value):
        c = QCheckBox(text)
        c.setChecked(bool(value))
        w = QWidget()
        wl = QHBoxLayout(w)
        wl.setContentsMargins(12, 2, 12, 2)
        wl.addWidget(c)
        wl.addStretch()
        lay.addWidget(w)
        return BVar(c)

    def _build_form(self, sel):
        self.current_sel = sel
        self._clear_form()
        f = self.form_lay
        kind = sel[0]
        d = self.cfg.data

        def title(text):
            t = QLabel(text)
            t.setStyleSheet("color: white; font-size: 12pt;"
                            " font-weight: bold;")
            t.setAlignment(Qt.AlignmentFlag.AlignCenter)
            t.setContentsMargins(0, 14, 0, 8)
            f.addWidget(t)

        def note(text, color=C_FAINT, size=8):
            n = QLabel(text)
            n.setStyleSheet(f"color: {color}; font-size: {size}pt;")
            n.setWordWrap(True)
            n.setContentsMargins(12, 4, 12, 0)
            f.addWidget(n)

        if kind == "scanner":
            sc = self.app.heads.get_scanner(self.head)
            title("BARCODE SCANNER  (step 1 — before scenes)")
            self.form_vars["enabled"] = self._chk(
                f, "Enable barcode scanner for this model",
                sc.get("enabled"))
            srow = QHBoxLayout()
            srow.setContentsMargins(12, 4, 12, 0)
            sl = QLabel("Part scan source:")
            sl.setStyleSheet("color: #c8c8d0;")
            sl.setFixedWidth(160)
            srow.addWidget(sl)
            cv = CVar(QComboBox())
            cv.setup(PART_SOURCES, sc.get("source", "scanner"))
            srow.addWidget(cv.c, stretch=1)
            f.addLayout(srow)
            self.form_vars["source"] = cv
            note("Operator ID is ALWAYS scanned with the scanner.  With\n"
                 "'Camera' the PART is read by the camera: add QR scanners\n"
                 "while building a scene and mark ONE as master; a CAMERA QR\n"
                 "node then appears here for the client signals.",
                 color="#f59f00")
            note("No device to pick.  A scan opens a floating input box\n"
                 "on top of everything that holds the keyboard — the\n"
                 "cursor stays in it and nothing else can be typed into\n"
                 "until Enter.  Any keyboard-wedge scanner works.")
            self.form_vars["prefix"] = self._add_field(
                f, "Prefix for client:", sc.get("prefix", ""),
                "what we send = prefix + barcode")
            self.form_vars["send_to_client"] = self._chk(
                f, "Send barcode to client (auto-sent on scan)",
                sc.get("send_to_client", True))
            self.form_vars["auto_popup"] = self._chk(
                f, "Pop the scan box up automatically each cycle",
                sc.get("auto_popup", True))
            note("If enabled: every cycle needs a SCAN before any\n"
                 "scene can run (sending is attempted on scan,\n"
                 "but a missing client does not block runs). Press\n"
                 "SCAN / F2 on the model page, or use  Scanner  there\n"
                 "for the live scan test.")

        elif kind == "qr":
            master = self.app.heads.get_master_qr(self.head)
            title("CAMERA QR  (master scan)")
            if master:
                note(f"Master QR:  '{master['name']}'  (scene "
                     f"'{master['scene']}')  ·  "
                     f"{'region set' if master.get('roi_norm') else 'full frame'}"
                     f"  ·  {QrRules.from_dict(master).describe()}",
                     color="#74c0fc", size=9)
            self.form_vars["REQ"] = self._add_field(
                f, "Out to client / request:", d["qr"].get("REQ", ""),
                "sent when the operator presses  📷 SCAN QR  —\n"
                "tells the client to present the part")
            self.form_vars["TRIGGER"] = self._add_field(
                f, "In from client / trigger:", d["qr"].get("TRIGGER", ""),
                "client sends it when the part is in position -> the\n"
                "camera reads the QR now (also works without the button)")
            self.form_vars["OK"] = self._add_field(
                f, "Out to client / read OK:", d["qr"].get("OK", ""),
                "sent when the QR was read (and matched length/start)")
            self.form_vars["NG"] = self._add_field(
                f, "Out to client / read NG:", d["qr"].get("NG", ""),
                "sent when the QR could not be read reliably")
            note("Each cycle the operator presses  📷 SCAN QR  on the model\n"
                 "page.  If no trigger is set (or no client is connected)\n"
                 "the camera is read immediately.  After a good read the\n"
                 "normal flow continues: MES check, then the barcode\n"
                 "(prefix + value) goes to the client.")

        elif kind == "head":
            title(f"MODEL LEVEL  —  '{self.head}'")
            self.form_vars["SELECT"] = self._add_field(
                f, "Select signal:", d["head"].get("SELECT", ""),
                "two directions: client sends it -> this model opens;\n"
                "you open this model in the UI -> we send it to the client")
            self.form_vars["PLC"] = self._add_field(
                f, "In from client (run):", d["head"].get("PLC", ""),
                "when received -> the whole model runs (all scenes)")
            self.form_vars["OK"] = self._add_field(
                f, "Out to client / OK:", d["head"].get("OK", ""),
                "sent when EVERY scene result is OK")
            self.form_vars["NG"] = self._add_field(
                f, "Out to client / NG:", d["head"].get("NG", ""),
                "sent when any scene result is NG")
            note("Connection (IP/port) and handshake are GLOBAL "
                 "— edit them in   Monitor / Connection.")

        elif kind == "parent":
            p = sel[1]
            pc = d["parents"][p]
            title(f"SCENE  —  '{p}'")
            self.form_vars["PLC"] = self._add_field(
                f, "In from client:", pc.get("PLC", ""),
                "when received -> only this scene runs (with ALL areas)")
            self.form_vars["OK"] = self._add_field(
                f, "Out to client / OK:", pc.get("OK", ""),
                "sent when all areas of this scene are OK")
            self.form_vars["NG"] = self._add_field(
                f, "Out to client / NG:", pc.get("NG", ""))
            self.form_vars["area_wise"] = self._chk(
                f, "Send area wise results", pc.get("area_wise", True))
            note("Turn OFF for scene-only communication: the scene\n"
                 "trigger still runs ALL areas, but area OK/NG\n"
                 "fields become optional and are not sent.")
            self.form_vars["combine"] = self._chk(
                f, "Send combined scene result", pc.get("combine", True))

        elif kind == "child":
            p, c = sel[1], sel[2]
            cc = d["parents"][p]["children"][c]
            title(f"AREA  —  '{p} / {c}'")
            if not bool(d["parents"][p].get("area_wise", True)):
                note("ℹ  Scene-only mode: area-wise results are "
                     f"OFF for '{p}'.\nThese fields are OPTIONAL "
                     "— the scene trigger already runs this "
                     "area.\nFill them only if you also want "
                     "per-area signals.", color="#f59f00", size=9)
            self.form_vars["PLC"] = self._add_field(
                f, "In from client (optional):", cc.get("PLC", ""),
                "optional — when received, ONLY this area runs")
            self.form_vars["OK"] = self._add_field(
                f, "Out to client / OK:", cc.get("OK", ""),
                "sent when all pins of this area are found in range")
            self.form_vars["NG"] = self._add_field(
                f, "Out to client / NG:", cc.get("NG", ""),
                "sent when pins are missing or shifted too far")

        btnrow = QHBoxLayout()
        btnrow.setContentsMargins(0, 16, 0, 8)
        btnrow.addStretch()
        btnrow.addWidget(styled_btn(self.form_holder, " Save",
                                    self.save_current, color="#2f9e44"))
        btnrow.addWidget(styled_btn(self.form_holder, " Clear fields",
                                    self.clear_current, color="#e03131"))
        btnrow.addStretch()
        f.addLayout(btnrow)
        f.addStretch()

    # ------------------------------------------------------------------
    def _check_duplicate_triggers(self):
        dups = []
        m = {}

        def put(text, where):
            text = str(text or "").strip()
            if not text:
                return
            if text in m:
                dups.append(f"'{text}' used by {m[text]} AND {where}")
            else:
                m[text] = where

        for h in self.app.heads.list_heads():
            cfg = self.cfg if h == self.head \
                else CommConfig(self.app.heads, h)
            put(cfg.data["head"].get("SELECT"), f"{h}.SELECT")
            put(cfg.data["head"].get("PLC"), f"{h}.run")
            if self.app.heads.camera_qr_enabled(h):
                put(cfg.data["qr"].get("TRIGGER"), f"{h}.qr_trigger")
            for p, pc in cfg.data["parents"].items():
                put(pc.get("PLC"), f"{h}/{p}")
                for c, cc in pc["children"].items():
                    put(cc.get("PLC"), f"{h}/{p}/{c}")
        return dups

    def save_current(self):
        sel = self.current_sel
        d = self.cfg.data
        v = {k: (var.get() if isinstance(var, BVar)
                 else var.get().strip())
             for k, var in self.form_vars.items()}
        if sel[0] == "scanner":
            self.app.heads.set_scanner(
                self.head, enabled=bool(v.get("enabled")),
                prefix=v.get("prefix", ""),
                send_to_client=bool(v.get("send_to_client", True)),
                auto_popup=bool(v.get("auto_popup", True)),
                source=v.get("source") or "scanner")
            self.refresh_tree(select=sel)
            return
        if sel[0] == "head":
            d["head"].update({k: v.get(k, "")
                              for k in ("SELECT", "PLC", "OK", "NG")})
        elif sel[0] == "qr":
            d["qr"].update({k: v.get(k, "") for k in CommConfig.QR_FIELDS})
        elif sel[0] == "parent":
            pc = d["parents"][sel[1]]
            pc.update({k: v.get(k, "") for k in ("PLC", "OK", "NG")})
            pc["area_wise"] = bool(v.get("area_wise", True))
            pc["combine"] = bool(v.get("combine", True))
        elif sel[0] == "child":
            cc = d["parents"][sel[1]]["children"][sel[2]]
            cc.update({k: v.get(k, "") for k in ("PLC", "OK", "NG")})
        dups = self._check_duplicate_triggers()
        if dups:
            QMessageBox.warning(self, "Duplicate signal",
                                "Warning — same signal string used more "
                                "than once:\n\n" + "\n".join(dups))
        self.cfg.save()
        self.refresh_tree(select=sel)

    def clear_current(self):
        for var in self.form_vars.values():
            if isinstance(var, BVar):
                var.set(True)
            else:
                var.set("")

    def open_mes_cfg(self):
        MesDialog(self, self.app, self.head).show()

    def show_yaml(self):
        self.cfg.save()
        win = QDialog(self)
        win.setWindowTitle(f"communication.yaml — {self.head}")
        win.resize(620, 600)
        lay = QVBoxLayout(win)
        txt = QTextEdit()
        txt.setReadOnly(True)
        parts = []
        with open(self.cfg.path) as fh:
            parts.append("# communication.yaml\n" + fh.read())
        sp = self.app.heads.scanner_path(self.head)
        if os.path.isfile(sp):
            with open(sp) as fh:
                parts.append("\n# scanner.yaml\n" + fh.read())
        qp = self.app.heads.qr_path(self.head)
        if os.path.isfile(qp):
            with open(qp) as fh:
                parts.append("\n# qr_scanners.yaml\n" + fh.read())
        mp = self.app.heads.mes_path(self.head)
        if os.path.isfile(mp):
            with open(mp) as fh:
                mes_txt = fh.read()
            mes_txt = re.sub(r"(?m)^(password:).*$", r"\1 '****'", mes_txt)
            parts.append("\n# mes.yaml\n" + mes_txt)
        txt.setPlainText("\n".join(parts))
        lay.addWidget(txt, stretch=1)
        pl = QLabel(self.cfg.path)
        pl.setStyleSheet(f"color: {C_FAINT}; font-size: 8pt;")
        pl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lay.addWidget(pl)
        win.show()


# ============================================================================
# QR SCANNER SETTINGS  (step 1 of adding a camera QR scanner to a scene)
# ============================================================================
class QrConfigDialog(QDialog):
    """Name + the two OPTIONAL rules.  Leave a rule empty and it is not
    checked; fill it and a code that breaks it is rejected."""

    def __init__(self, master, default_name, taken_names):
        super().__init__(master)
        self.setWindowTitle("Add QR scanner")
        self.setMinimumWidth(440)
        self.taken = {n.lower() for n in taken_names}
        self.result_cfg = None
        lay = QVBoxLayout(self)
        lay.setContentsMargins(18, 14, 18, 14)
        lay.setSpacing(8)
        t = QLabel("📷  QR SCANNER SETTINGS")
        t.setStyleSheet("color: #ffd43b; font-size: 13pt; font-weight: bold;")
        t.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lay.addWidget(t)

        grid = QGridLayout()
        self.e_name = QLineEdit(default_name)
        self.e_len = QLineEdit()
        self.e_len.setPlaceholderText("empty = any length")
        self.e_start = QLineEdit()
        self.e_start.setPlaceholderText("empty = any start")
        for i, (label, w) in enumerate([("Name", self.e_name),
                                        ("Length (chars)", self.e_len),
                                        ("Starts with", self.e_start)]):
            l = QLabel(label)
            l.setStyleSheet("color: #c8c8d0;")
            grid.addWidget(l, i, 0)
            grid.addWidget(w, i, 1)
        lay.addLayout(grid)

        n = QLabel("Both rules are optional.  Empty = not checked, the QR is\n"
                   "accepted as read.  Filled = a code that does not match is\n"
                   "rejected (the value it read is shown in the log).\n\n"
                   "Next: mark the region where the QR sits on the live\n"
                   "image (or use the full frame).")
        n.setStyleSheet(f"color: {C_FAINT}; font-size: 8pt;")
        lay.addWidget(n)

        row = QHBoxLayout()
        row.addStretch()
        row.addWidget(styled_btn(self, "Next ▶  select region", self._ok,
                                 color="#2f9e44"))
        row.addWidget(styled_btn(self, "Cancel", self.reject,
                                 color="#44444e"))
        row.addStretch()
        lay.addLayout(row)

    def _ok(self):
        name = self.e_name.text().strip()
        if not name or "/" in name or "\\" in name:
            QMessageBox.warning(self, "Name", "Enter a name (no / or \\).")
            return
        if name.lower() in self.taken:
            QMessageBox.warning(self, "Name",
                                f"'{name}' is already used in this scene.")
            return
        ltxt = self.e_len.text().strip()
        if ltxt and (not ltxt.isdigit() or int(ltxt) <= 0):
            QMessageBox.warning(self, "Length",
                                "Length must be a positive whole number "
                                "— or empty for 'any'.")
            return
        self.result_cfg = {"name": name,
                           "length": int(ltxt) if ltxt else 0,
                           "starts_with": self.e_start.text().strip()}
        self.accept()


# ============================================================================
# SCREEN 4 — BUILD SCENE (areas with ROI confirm/reset)
# ============================================================================
class BuildParentScreen(BaseScreen):
    MODE_LIVE = "live"
    MODE_ROI = "roi"
    MODE_TEMPLATE = "template"
    MODE_RESULT = "result"
    MODE_QR = "qr"

    def __init__(self, master, app, head_name, parent_name):
        super().__init__(master, app)
        self.head = head_name
        self.parent_name = parent_name
        self.parent_store = app.heads.get_parent_store(head_name)
        self.child_manager = self.parent_store.get_child_manager(parent_name)
        self.reg = None
        self.template_list = []
        self.temp_reg_list = []
        self.matches = None
        self.child_index = 1
        self.current_child_name = None
        self.current_child_saved = False
        self.region_snapshot = None
        self.mode = self.MODE_LIVE
        self.result_img = None
        self.busy = False
        # camera QR scanners (only when the model's part-scan source = camera)
        self.qr_camera = app.heads.camera_qr_enabled(head_name)
        self.qr_pending = None       # rules typed in, region still to draw
        self._qr_items = app.heads.qr_scanners_for_scene(head_name,
                                                         parent_name)
        self._live_wh = None         # size of the preview frame on the canvas
        self._qr_testing = False
        self.btn_qr_add = self.btn_qr_test = self.btn_qr_del = None
        self.btn_qr_region = self.btn_qr_full = self.btn_qr_cancel = None
        self.qr_list_lbl = None
        # a scene that holds the master QR holds ONLY that QR - no areas
        self.master_scene = any(q["master"] for q in self._qr_items)
        self._area_boxes = []
        if self.qr_camera:
            app.camera.keep_raw = True

        root = QVBoxLayout(self)
        root.setContentsMargins(10, 8, 10, 8)
        top = QHBoxLayout()
        top.addWidget(styled_btn(self, "← Cancel", self.cancel,
                                 color="#44444e"))
        t = QLabel(f"BUILD SCENE  {head_name} / {parent_name}")
        t.setStyleSheet("color: white; font-size: 14pt; font-weight: bold;")
        top.addWidget(t)
        top.addStretch()
        self.status_lbl = QLabel("")
        self.status_lbl.setStyleSheet("color: #ffd43b; font-weight: bold;")
        top.addWidget(self.status_lbl)
        root.addLayout(top)

        body = QHBoxLayout()
        root.addLayout(body, stretch=1)
        centre = QWidget()
        clay = QVBoxLayout(centre)
        clay.setContentsMargins(0, 0, 4, 0)
        self.canvas = VideoCanvas(centre)
        clay.addWidget(self.canvas, stretch=1)
        clay.addWidget(zoom_controls(centre, self.canvas))
        clay.addStretch()
        body.addWidget(centre, stretch=1)
        self.canvas.on_rect_drawn = lambda r: self._update_buttons()

        right = panel()
        right.setFixedWidth(300)
        rlay = QVBoxLayout(right)
        rlay.setContentsMargins(8, 6, 8, 6)

        sec = QGroupBox(" 1. Region (ROI) ")
        s1 = QVBoxLayout(sec)
        self.btn_sel_roi = styled_btn(sec, "Select ROI (drag on video)",
                                      self.start_roi)
        s1.addWidget(self.btn_sel_roi)
        row = QHBoxLayout()
        self.btn_confirm_roi = styled_btn(sec, "✔ Confirm ROI",
                                          self.confirm_roi, color="#2f9e44")
        row.addWidget(self.btn_confirm_roi)
        self.btn_reset_roi = styled_btn(sec, "✖ Reset ROI", self.reset_roi,
                                        color="#e03131")
        row.addWidget(self.btn_reset_roi)
        s1.addLayout(row)
        rlay.addWidget(sec)

        sec2 = QGroupBox(" 2. Templates ")
        s2 = QVBoxLayout(sec2)
        self.btn_add_tmpl = styled_btn(sec2, "Select Template (drag)",
                                       self.start_template)
        s2.addWidget(self.btn_add_tmpl)
        row2 = QHBoxLayout()
        self.btn_confirm_tmpl = styled_btn(sec2, "✔ Confirm",
                                           self.confirm_template,
                                           color="#2f9e44")
        row2.addWidget(self.btn_confirm_tmpl)
        self.btn_reset_tmpl = styled_btn(sec2, "✖ Reset",
                                         self.reset_template,
                                         color="#e03131")
        row2.addWidget(self.btn_reset_tmpl)
        s2.addLayout(row2)
        self.tmpl_count_lbl = QLabel("Templates: 0")
        self.tmpl_count_lbl.setStyleSheet("color: #c8c8d0; font-size: 9pt;")
        self.tmpl_count_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        s2.addWidget(self.tmpl_count_lbl)
        rlay.addWidget(sec2)

        sec3 = QGroupBox(" 3. Settings ")
        s3 = QGridLayout(sec3)
        self.var_thresh = FVar(QLineEdit("0.8"))
        self.var_smin = FVar(QLineEdit("0.8"))
        self.var_smax = FVar(QLineEdit("1.5"))
        self.var_steps = IVar(QLineEdit("20"))
        self.var_maxm = IVar(QLineEdit("20"))
        for i, (label, var) in enumerate([("Threshold", self.var_thresh),
                                          ("Scale min", self.var_smin),
                                          ("Scale max", self.var_smax),
                                          ("Scale steps", self.var_steps),
                                          ("Max matches", self.var_maxm)]):
            l = QLabel(label)
            l.setStyleSheet("color: #c8c8d0; font-size: 9pt;")
            s3.addWidget(l, i, 0)
            var.e.setFixedWidth(72)
            s3.addWidget(var.e, i, 1)
        rlay.addWidget(sec3)
        self._area_boxes = [sec, sec2, sec3]

        sec4 = QGroupBox(" 4. Actions ")
        s4 = QVBoxLayout(sec4)
        self.btn_test = styled_btn(sec4, "▶ Test Detect", self.test_detect,
                                   color="#f59f00")
        s4.addWidget(self.btn_test)
        self.btn_back_live = styled_btn(sec4, "Back to Live",
                                        self.back_to_live, color="#44444e")
        s4.addWidget(self.btn_back_live)
        self.btn_save = styled_btn(sec4, "💾 Save Area", self.save_child,
                                   color="#2f9e44")
        s4.addWidget(self.btn_save)
        self.btn_next = styled_btn(sec4, "➕ Next Area", self.next_child)
        s4.addWidget(self.btn_next)
        self.btn_finish = styled_btn(sec4, "✅ Finish Scene",
                                     self.finish_parent, color="#845ef7")
        s4.addWidget(self.btn_finish)
        s4.addWidget(styled_btn(sec4, "Auto Balance", self.auto_balance,
                                color="#44444e"))
        rlay.addWidget(sec4)

        if self.qr_camera:
            sec5 = QGroupBox(" 5. QR Scanner (camera) ")
            s5 = QVBoxLayout(sec5)
            self.qr_list_lbl = QLabel("")
            self.qr_list_lbl.setWordWrap(True)
            self.qr_list_lbl.setStyleSheet("color: #c8c8d0; font-size: 8pt;")
            s5.addWidget(self.qr_list_lbl)
            r5 = QHBoxLayout()
            self.btn_qr_add = styled_btn(sec5, "➕ Add QR", self.qr_add_start,
                                         color="#0ca678")
            self.btn_qr_test = styled_btn(sec5, "🔍 Test", self.qr_test_read,
                                          color="#f59f00")
            self.btn_qr_del = styled_btn(sec5, "🗑", self.qr_remove,
                                         color="#e03131")
            for b in (self.btn_qr_add, self.btn_qr_test, self.btn_qr_del):
                r5.addWidget(b)
            s5.addLayout(r5)
            r5b = QHBoxLayout()
            self.btn_qr_region = styled_btn(sec5, "✔ Region",
                                            self.qr_confirm_region,
                                            color="#2f9e44")
            self.btn_qr_full = styled_btn(sec5, "⛶ Full frame",
                                          self.qr_use_full_frame,
                                          color="#3b6ef5")
            self.btn_qr_cancel = styled_btn(sec5, "✖", self.qr_cancel,
                                            color="#44444e")
            for b in (self.btn_qr_region, self.btn_qr_full,
                      self.btn_qr_cancel):
                r5b.addWidget(b)
            s5.addLayout(r5b)
            rlay.addWidget(sec5)
            self._qr_refresh_list()

        self.log_box = QTextEdit()
        self.log_box.setReadOnly(True)
        self.log_box.setStyleSheet(
            f"QTextEdit {{ background: {C_FIELD}; color: #c8c8d0;"
            f" border: none; border-radius: 6px; font-size: 8pt; }}")
        rlay.addWidget(self.log_box, stretch=1)
        body.addWidget(right)

        self.app.camera.start()
        self._update_buttons()
     
        self._set_status(f"Area: area_{self.child_index} — select ROI first")
        self._apply_master_lock()
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)
        self._timer.start(30)

    def log(self, msg):
        if not self._alive:
            return
        self.log_box.append(html.escape(msg, quote=False))
        self.log_box.verticalScrollBar().setValue(
            self.log_box.verticalScrollBar().maximum())

    def _set_status(self, txt):
        self.status_lbl.setText(txt)

    def _update_buttons(self):
        pend = self.canvas.pending_rect is not None
        roi_mode = self.mode == self.MODE_ROI
        tmpl_mode = self.mode == self.MODE_TEMPLATE
        self.btn_confirm_roi.setEnabled(roi_mode and pend)
        self.btn_reset_roi.setEnabled(
            (roi_mode and pend) or self.reg is not None)
        self.btn_add_tmpl.setEnabled(
            self.reg is not None and not tmpl_mode)
        self.btn_confirm_tmpl.setEnabled(tmpl_mode and pend)
        self.btn_reset_tmpl.setEnabled(tmpl_mode)
        can_test = self.reg is not None and len(self.template_list) > 0
        self.btn_test.setEnabled(can_test and not self.busy)
        self.btn_save.setEnabled(can_test and self.matches is not None)
        self.btn_back_live.setEnabled(self.mode == self.MODE_RESULT)
        self.tmpl_count_lbl.setText(
            f"Templates: {len(self.template_list)}")
        if self.qr_camera:
            qr_mode = self.mode == self.MODE_QR and self.qr_pending is not None
            self.btn_qr_add.setEnabled(not qr_mode and not self.busy)
            self.btn_qr_region.setEnabled(qr_mode and pend)
            self.btn_qr_full.setEnabled(qr_mode)
            self.btn_qr_cancel.setEnabled(qr_mode)
            self.btn_qr_test.setEnabled(
                (qr_mode or bool(self._qr_items)) and not self._qr_testing)
            self.btn_qr_del.setEnabled(bool(self._qr_items) and not qr_mode)
            if self.master_scene:          # the master QR is all this scene has
                self.btn_qr_add.setEnabled(False)
        if self.master_scene:
            for b in (self.btn_test, self.btn_save, self.btn_next):
                b.setEnabled(False)

    def _tick(self):
        if not self._alive:
            return
        if self.mode in (self.MODE_LIVE, self.MODE_ROI, self.MODE_QR):
            frame = self.app.camera.get_frame()
            if frame is not None:
                disp = frame.copy()
                self._live_wh = (disp.shape[1], disp.shape[0])
                if self.reg is not None:
                    x, y, w, h = [int(v) for v in self.reg]
                    cv2.rectangle(disp, (x, y), (x + w, y + h),
                                  (0, 255, 0), 2)
                    cv2.putText(disp, "ROI confirmed", (x, max(15, y - 6)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                                (0, 255, 0), 2)
                fh, fw = disp.shape[:2]
                for q in self._qr_items:
                    roi = q.get("roi_norm")
                    if not roi:
                        continue
                    x, y = int(roi[0] * fw), int(roi[1] * fh)
                    w, h = int(roi[2] * fw), int(roi[3] * fh)
                    col = (0, 215, 255) if q["master"] else (255, 0, 255)
                    cv2.rectangle(disp, (x, y), (x + w, y + h), col, 2)
                    cv2.putText(disp, f"QR {q['name']}"
                                + (" (MASTER)" if q["master"] else ""),
                                (x, max(15, y - 6)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, col, 2)
                self.canvas.show(disp)
            elif self.app.camera.error:
                self.canvas.show_message(self.app.camera.error)
        elif self.mode == self.MODE_TEMPLATE \
                and self.region_snapshot is not None:
            self.canvas.show(self.region_snapshot)
        elif self.mode == self.MODE_RESULT and self.result_img is not None:
            self.canvas.show(self.result_img)

    def start_roi(self):
        self._qr_cancel_quiet()
        self.mode = self.MODE_ROI
        self.canvas.select_enabled = True
        self.canvas.clear_rect()
        self._set_status("Drag a rectangle over the region, then press "
                         "Confirm ROI (zoom in for precision)")
        self._update_buttons()

    def confirm_roi(self):
        if self.canvas.pending_rect is None:
            return
        self.reg = tuple(self.canvas.pending_rect)
        self.canvas.select_enabled = False
        self.canvas.clear_rect()
        self.mode = self.MODE_LIVE
        self.log(f"ROI confirmed: {self.reg}")
        self._set_status("ROI confirmed ✔ — now add template(s)")
        self._update_buttons()

    def reset_roi(self):
        self.reg = None
        self.matches = None
        self.canvas.clear_rect()
        self.mode = self.MODE_ROI
        self.canvas.select_enabled = True
        self.log("ROI reset — draw it again")
        self._set_status("ROI reset — drag a new rectangle, then Confirm ROI")
        self._update_buttons()

    def start_template(self):
        if self.reg is None:
            return
        self._qr_cancel_quiet()
        frame = self.app.camera.get_frame()
        if frame is None:
            QMessageBox.critical(self, "No camera",
                                 "No camera frame available yet.")
            return
        x, y, w, h = [int(v) for v in self.reg]
        self.region_snapshot = frame[y:y + h, x:x + w].copy()
        self.mode = self.MODE_TEMPLATE
        self.canvas.select_enabled = True
        self.canvas.clear_rect()
        self.canvas.fit()
        self._set_status("Drag over ONE pin/feature inside the region, "
                         "then Confirm (wheel to zoom)")
        self._update_buttons()

    def confirm_template(self):
        if self.canvas.pending_rect is None or self.region_snapshot is None:
            return
        tx, ty, tw, th = self.canvas.pending_rect
        template = self.region_snapshot[ty:ty + th, tx:tx + tw].copy()
        if template.size == 0:
            QMessageBox.critical(self, "Bad template",
                                 "Template area is empty, redraw it.")
            return
        self.template_list.append(template)
        self.temp_reg_list.append((tx, ty, tw, th))
        self.log(f"Template {len(self.template_list)} added "
                 f"({self.current_child_name or f'area_{self.child_index}'})")
        self.canvas.select_enabled = False
        self.canvas.clear_rect()
        self.region_snapshot = None
        self.mode = self.MODE_LIVE
        self.canvas.fit()
        self._set_status("Template added ✔ — add more, or Test Detect")
        self._update_buttons()

    def reset_template(self):
        self.canvas.clear_rect()
        self._set_status("Template cleared — drag again")
        self._update_buttons()

    def test_detect(self):
        if self.busy or self.reg is None or not self.template_list:
            return
        frame = self.app.camera.get_frame()
        if frame is None:
            return
        self.busy = True
        self._set_status("Detecting…")
        self._update_buttons()
        threading.Thread(target=self._test_worker, args=(frame,),
                         daemon=True).start()

    def _test_worker(self, frame):
        try:
            x, y, w, h = [int(v) for v in self.reg]
            region = frame[y:y + h, x:x + w]
            matches, output, mask = run_detection(
                detect_multi_scale_templates(
                    image=region, template=self.template_list,
                    scale_range=(self.var_smin.get(), self.var_smax.get()),
                    scale_steps=int(self.var_steps.get()),
                    threshold=float(self.var_thresh.get()),
                    max_matches=int(self.var_maxm.get()),
                    visualize=True, t=time.time(), config=None))
            self._last_frame_for_save = frame
            msg = f"Detected: {len(matches)}"
            res = np.vstack([banner(output.shape[1],
                                    f"TEST — {len(matches)} match(es)",
                                    None, height=40), output])
        except Exception:
            matches, res, msg = None, None, \
                "ERROR:\n" + traceback.format_exc()

        def apply():
            if not self._alive:
                return
            self.matches = matches
            self.log(msg)
            if res is not None:
                self.result_img = res
                self.mode = self.MODE_RESULT
                self.canvas.fit()
                self._set_status(f"{msg} — Save Area if OK, or Back to Live")
            self.busy = False
            self._update_buttons()
        self.after(0, apply)

    def back_to_live(self):
        self.mode = self.MODE_LIVE
        self.result_img = None
        self.canvas.fit()
        self._update_buttons()

    def save_child(self):
        if self.reg is None or not self.template_list or self.matches is None:
            QMessageBox.warning(self, "Not ready",
                                "Need a confirmed ROI, at least 1 "
                                "template and a successful Test Detect "
                                "before saving.")
            return
        if self.current_child_name is None:
            default = f"area_{self.child_index}"
            name, ok = QInputDialog.getText(
                self, "Area name", f"Area name (default '{default}'):")
            if not ok:
                name = None
            self.current_child_name = (name.strip()
                                       if name and name.strip() else default)
        frame = getattr(self, "_last_frame_for_save", None)
        if frame is None:
            frame = self.app.camera.get_frame()
        self.child_manager.save(self.current_child_name, frame, self.reg,
                                self.temp_reg_list, self.template_list,
                                float(self.var_thresh.get()),
                                (self.var_smin.get(), self.var_smax.get()),
                                int(self.var_steps.get()),
                                int(self.var_maxm.get()),
                                self.matches)
        self.parent_store.add_child(self.parent_name,
                                    self.current_child_name)
        self.app.heads.add_parent(self.head, self.parent_name)
        self.current_child_saved = True
        self.log(f"Saved area '{self.current_child_name}' under "
                 f"'{self.head}/{self.parent_name}'")
        self._set_status(f"Area '{self.current_child_name}' saved ✔ — "
                         "Next Area or Finish Scene")

    def next_child(self):
        if not self.current_child_saved:
            r = QMessageBox.question(
                self, "Not saved",
                "Current area isn't saved yet — discard it and start "
                "a new one?")
            if r != QMessageBox.StandardButton.Yes:
                return
        else:
            self.child_index += 1
        self.current_child_name = None
        self.current_child_saved = False
        self.reg = None
        self.template_list = []
        self.temp_reg_list = []
        self.matches = None
        self.region_snapshot = None
        self.result_img = None
        self.mode = self.MODE_LIVE
        self.canvas.select_enabled = False
        self.canvas.clear_rect()
        self.log(f"--- New area: area_{self.child_index} ---")
        self._set_status(f"Area: area_{self.child_index} — select ROI first")
        self._update_buttons()

    def finish_parent(self):
        children = self.parent_store.list_children(self.parent_name)
        if not children and self.master_scene:
            self.app.heads.add_parent(self.head, self.parent_name)
            self.log(f"Master QR scene '{self.parent_name}' saved")
            QMessageBox.information(
                self, "Scene finished",
                f"Master QR scene '{self.parent_name}' saved.\n\nIt has no "
                "areas — the camera reads its QR when the operator presses "
                "SCAN QR.")
            self.app.show_head(self.head, announce=False)
            return
        if not children:
            QMessageBox.warning(self, "No areas",
                                "No areas saved yet — save at least one "
                                "first.")
            return
        self.app.heads.add_parent(self.head, self.parent_name)
        self.log(f"Scene '{self.parent_name}' finished with areas: "
                 f"{children}")
        QMessageBox.information(self, "Scene finished",
                                f"Scene '{self.parent_name}' saved with "
                                f"{len(children)} area(s).")
        self.app.show_head(self.head, announce=False)

    def cancel(self):
        # a scene that never got an area does not exist: forget its QR
        # scanners too (otherwise a "master" could point at a ghost scene)
        if not self.app.heads.scene_exists(self.head, self.parent_name):
            self.app.heads.remove_qr_for_scene(self.head, self.parent_name)
        self.app.show_head(self.head, announce=False)

    # ------------------------------------------------------------------
    # 5. QR SCANNER (camera)
    #   Add QR -> rules -> draw the region -> "is it the master?" -> saved.
    #   Exactly one master per model; it is read at the start of each cycle.
    #   Every other QR scanner is read whenever its scene runs.
    # ------------------------------------------------------------------
    def _apply_master_lock(self):
        """Master QR scene: area building (ROI, templates, detection, save,
        next area) is switched off - the scene is only the QR."""
        for b in self._area_boxes:
            b.setEnabled(not self.master_scene)
        self._update_buttons()
        if self.master_scene:
            self._set_status("MASTER QR scene — only the QR lives here, no "
                             "areas.  Press ✅ Finish Scene to save it.")

    def _qr_refresh_list(self):
        self._qr_items = self.app.heads.qr_scanners_for_scene(
            self.head, self.parent_name)
        self.master_scene = any(q["master"] for q in self._qr_items)
        if self.qr_list_lbl is None:
            return
        if self._qr_items:
            lines = [f"{'★' if q['master'] else '•'} {q['name']}"
                     f"{' (MASTER)' if q['master'] else ''} — "
                     f"{'region' if q.get('roi_norm') else 'full frame'}, "
                     f"{QrRules.from_dict(q).describe()}"
                     for q in self._qr_items]
        else:
            lines = ["no QR scanner in this scene yet"]
        m = self.app.heads.get_master_qr(self.head,
                                         include_scene=self.parent_name)
        lines.append(f"master: '{m['name']}' (scene {m['scene']})" if m
                     else "⚠ the model has no master QR yet")
        self.qr_list_lbl.setText("\n".join(lines))

    def qr_add_start(self):
        if self.master_scene:
            QMessageBox.information(
                self, "Master QR scene",
                "This scene holds the master QR — a master scene contains "
                "only that QR (no areas, no other QR).\n\nRemove it first "
                "(🗑) to change it, or create another scene.")
            return
        if self.mode not in (self.MODE_LIVE, self.MODE_RESULT):
            QMessageBox.information(
                self, "Finish this step first",
                "Finish or reset the ROI / template selection first, "
                "then add the QR scanner.")
            return
        if self.mode == self.MODE_RESULT:
            self.result_img = None
            self.canvas.fit()
        taken = [q["name"] for q in self._qr_items]
        n = 1
        while f"QR_{n}" in taken:
            n += 1
        dlg = QrConfigDialog(self, f"QR_{n}", taken)
        if dlg.exec() != QDialog.DialogCode.Accepted or not dlg.result_cfg:
            return
        self.qr_pending = dlg.result_cfg
        self.mode = self.MODE_QR
        self.canvas.select_enabled = True
        self.canvas.clear_rect()
        self._set_status("Drag a rectangle around the QR code, then ✔ Region "
                         "(or ⛶ Full frame to search the whole image)")
        self._update_buttons()

    def qr_confirm_region(self):
        if (self.mode != self.MODE_QR or not self.qr_pending
                or self.canvas.pending_rect is None):
            return
        if not self._live_wh:
            QMessageBox.critical(self, "No camera",
                                 "No camera frame available yet.")
            return
        w, h = self._live_wh
        self._qr_save_pending(norm_from_rect(self.canvas.pending_rect, w, h))

    def qr_use_full_frame(self):
        if self.mode != self.MODE_QR or not self.qr_pending:
            return
        self._qr_save_pending(None)

    def _qr_save_pending(self, norm):
        cfg = self.qr_pending
        Y, N = QMessageBox.StandardButton.Yes, QMessageBox.StandardButton.No
        existing = self.app.heads.get_master_qr(self.head,
                                                include_scene=self.parent_name)
        # a master scene holds ONLY the master QR: it must be alone in a
        # scene that has no areas (saved, or being drawn) and no other QR
        has_areas = bool(
            self.parent_store.list_children(self.parent_name)
            or self.current_child_saved or self.reg is not None
            or self.template_list)
        alone = not self._qr_items and not has_areas
        master = False
        if not alone:
            QMessageBox.information(
                self, "Not a master scene",
                f"'{cfg['name']}' is saved as a normal QR of this scene.\n\n"
                "A MASTER QR needs a scene of its own: no areas and no other "
                "QR.  To make a master, press + Add Scene on the model page "
                "and add the QR there.")
        elif existing is None:
            r = QMessageBox.question(
                self, "Master QR?",
                f"Is '{cfg['name']}' the MASTER QR?\n\n"
                "The master QR is the one the camera reads at the start of "
                "every cycle (📷 SCAN QR button, after Finish Model).  "
                "Only ONE master is allowed per model, and its scene then "
                "contains only this QR — no areas can be added.\n\n"
                "Yes = master      No = normal QR of this scene "
                "(read whenever the scene runs)", Y | N, N)
            master = (r == Y)
        else:
            r = QMessageBox.question(
                self, "Master QR?",
                f"The model already has a master QR: '{existing['name']}' "
                f"(scene '{existing['scene']}').\nOnly one is allowed.\n\n"
                f"Make '{cfg['name']}' the master instead?  The old master "
                "is removed.  This scene then holds only this QR — no areas "
                "can be added.", Y | N, N)
            master = (r == Y)
        item = self.app.heads.add_qr_scanner(
            self.head, self.parent_name, cfg["name"], norm,
            cfg["length"], cfg["starts_with"], master)
        self.log(f"QR scanner '{item['name']}' saved"
                 f"{' as MASTER' if master else ''} — "
                 f"{'region' if norm else 'full frame'}, "
                 f"{QrRules.from_dict(item).describe()}")
        self.qr_pending = None
        self.mode = self.MODE_LIVE
        self.canvas.select_enabled = False
        self.canvas.clear_rect()
        self._qr_refresh_list()
        self._set_status(f"QR scanner '{item['name']}' saved ✔ — use 🔍 Test "
                         "to check the reading")
        self._apply_master_lock()

    def _qr_cancel_quiet(self):
        if self.qr_pending is None and self.mode != self.MODE_QR:
            return
        self.qr_pending = None
        if self.mode == self.MODE_QR:
            self.mode = self.MODE_LIVE
            self.canvas.select_enabled = False
            self.canvas.clear_rect()

    def qr_cancel(self):
        self._qr_cancel_quiet()
        self._set_status("QR setup cancelled")
        self._update_buttons()

    def qr_remove(self):
        if not self._qr_items:
            return
        names = [f"{q['name']}{' (MASTER)' if q['master'] else ''}"
                 for q in self._qr_items]
        choice, ok = QInputDialog.getItem(self, "Remove QR scanner",
                                          "Remove:", names, 0, False)
        if not ok:
            return
        q = self._qr_items[names.index(choice)]
        self.app.heads.remove_qr_scanner(self.head, q["id"])
        self.log(f"QR scanner '{q['name']}' removed")
        self._qr_refresh_list()
        self._apply_master_lock()
        if q["master"]:
            QMessageBox.warning(
                self, "Master removed",
                "That was the master QR — the model has no master now.  "
                "Add another QR scanner and mark it as master.")

    def qr_test_read(self):
        """Read now with the real reader (same code path as production):
        the QR being set up (its region so far), else every QR of the scene."""
        if self._qr_testing:
            return
        targets = []
        if self.mode == self.MODE_QR and self.qr_pending:
            roi = None
            if self.canvas.pending_rect is not None and self._live_wh:
                roi = norm_from_rect(self.canvas.pending_rect,
                                     *self._live_wh)
            targets.append((f"{self.qr_pending['name']} (setup)", roi,
                            QrRules.from_dict(self.qr_pending)))
        else:
            for q in self._qr_items:
                targets.append((q["name"], q.get("roi_norm"),
                                QrRules.from_dict(q)))
        if not targets:
            self.log("Nothing to test — add a QR scanner first")
            return
        qc = self.app.app_config.data["qr"]
        timeout = float(qc.get("scene_timeout_s", 3.0))
        agree = int(qc.get("min_agree", 2))
        src = self.app.camera.qr_source()
        self._qr_testing = True
        self._update_buttons()
        self.log(f"Testing {len(targets)} QR scanner(s) — readers: "
                 + ", ".join(available_backends()))

        def work():
            out = []
            for label, roi, rules in targets:
                try:
                    res = QrReader().read_stream(src, rules, roi,
                                                 timeout_s=timeout,
                                                 min_agree=agree)
                except Exception:
                    res = QrResult(reason="ERROR:\n"
                                          + traceback.format_exc())
                out.append((label, res))

            def apply():
                if not self._alive:
                    return
                for label, res in out:
                    if res.ok:
                        self.log(f"QR '{label}': ✔ '{res.value}'  "
                                 f"({res.frames} frames · {res.elapsed:.2f}s "
                                 f"· {res.backend})")
                    else:
                        self.log(f"QR '{label}': ✖ {res.reason}")
                self._qr_testing = False
                self._update_buttons()
            self.after(0, apply)
        threading.Thread(target=work, daemon=True).start()

    def auto_balance(self):
        ok = self.app.camera.auto_balance()
        self.log("auto_balance " + ("done" if ok else "FAILED"))


# ============================================================================
if __name__ == "__main__":
    import sys
    qt_app = QApplication(sys.argv)
    qt_app.setStyleSheet(GLOBAL_QSS)
    app = App()
    app.show()
    sys.exit(qt_app.exec())