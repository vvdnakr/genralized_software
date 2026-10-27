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

try:                      # optional: precise Ethernet link detection
    import psutil
except ImportError:       # pip install psutil
    psutil = None

from PyQt6.QtCore import Qt, QTimer, QRectF, pyqtSignal
from PyQt6.QtGui import (QImage, QPixmap, QPainter, QColor, QPen, QFont,
                         QBrush, QCursor, QShortcut, QKeySequence)
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QFrame, QDialog, QLabel, QPushButton,
    QLineEdit, QTextEdit, QCheckBox, QComboBox, QVBoxLayout, QHBoxLayout,
    QGridLayout, QGroupBox, QMessageBox, QInputDialog, QTreeWidget,
    QTreeWidgetItem, QScrollArea, QSizePolicy, QListWidget)

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
            # DIFFERENT password: needed to register a VALIDATION part
            # (a part that is run but NEVER reported to MES)
            "validation_password": "5678",
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
            # popups for lost Ethernet / lost camera, and the cycle-result popup
            "health": {"check_ethernet": True, "check_camera": True,
                       "interval_s": 2,
                       "comm_no_client_s": 10},   # popup if no client after
                                                  # Comm ON (0 = never)
            "cycle_popup": {"enabled": True,
                            "auto_close_s": 5},   # 0 = stay until OK pressed
        }
        self.load()

    def load(self):
        if os.path.isfile(APP_CONFIG_PATH):
            try:
                with open(APP_CONFIG_PATH) as f:
                    saved = yaml.safe_load(f) or {}
                for k in ("password", "validation_password", "locked"):
                    if k in saved:
                        self.data[k] = saved[k]
                self.data["ui"].update(saved.get("ui") or {})
                self.data["scanner"].update(saved.get("scanner") or {})
                self.data["qr"].update(saved.get("qr") or {})
                self.data["mes_upload"].update(saved.get("mes_upload") or {})
                self.data["retention"].update(saved.get("retention") or {})
                self.data["health"].update(saved.get("health") or {})
                self.data["cycle_popup"].update(saved.get("cycle_popup") or {})
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
            "validation_serials": [],     # MES disabled for these serials
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

    UNLOCK_CODE = "##**1"     # scan/type this to close the box, like Esc —
                              # for stations with a scanner but no keyboard
    last_unlocked = False     # set by ask() after the box closes

    def __init__(self, parent=None, prompt="SCAN BARCODE",
                 sub="", timeout_s=0, width=620, initial=""):
        super().__init__(parent)
        self.value = None
        self.unlocked = False
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
        if txt == self.UNLOCK_CODE:
            self.unlocked = True
            self._cancel()
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
        QuickScanInput.last_unlocked = dlg.unlocked
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
        self._listen_alerted = False

    # -- popups (thread-safe: always marshalled onto the UI thread) ---------
    def _popup(self, key, title, text):
        self.app.after(0, lambda: self.app.alert(key, title, text))

    def _popup_clear(self, *keys):
        def go():
            for k in keys:
                self.app.clear_alert(k)
        self.app.after(0, go)

    def _check_no_client(self, gen):
        if self.enabled and gen == self._gen and self.client_count() == 0 \
                and not self._listen_alerted:
            conn = self.settings.data["connection"]
            self.log.add("ERR", "no client connected")
            self._popup("comm_noclient", "MACHINE NOT CONNECTED",
                        "Reason: machine is off or its cable is out.")

    def client_count(self):
        with self._lock:
            return len(self._clients)

    def status_text(self):
        if not self.enabled:
            return "Machine connection is OFF"
        conn = self.settings.data["connection"]
        hs = self.settings.data["handshake"]
        n = self.client_count()
        cl = (f"{n} machine connected" if n
              else "waiting for the machine to connect")
        return (f"{cl}  ·  {conn.get('ip')}:{conn.get('port')}"
                f"  ·  handshake {'ON' if hs.get('enabled') else 'OFF'}")

    def start(self):
        if self.enabled:
            return
        self.enabled = True
        self._gen += 1
        threading.Thread(target=self._serve, args=(self._gen,),
                         daemon=True).start()
        self.log.add("EVT", "communication ENABLED")
        wait = float(self.app.app_config.data["health"]
                     .get("comm_no_client_s", 10))
        if wait > 0:
            g = self._gen
            self.app.after(int(wait * 1000), lambda: self._check_no_client(g))

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
        self._listen_alerted = False
        self._popup_clear("comm_listen", "comm_client", "comm_noclient")
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
                if not self._listen_alerted:
                    self._listen_alerted = True
                    self._popup("comm_listen", "CONNECTION FAILED",
                                "Reason: network cable is out or the IP "
                                f"address changed ({ip}).")
                for _ in range(self.RETRY_S * 2):
                    if not (self.enabled and gen == self._gen):
                        return
                    time.sleep(0.5)
                continue
            self._srv = srv
            self._listen_alerted = False
            self._popup_clear("comm_listen")
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
                    self._popup_clear("comm_client", "comm_noclient")
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
        if self.enabled and gen == self._gen:
            self._popup("comm_client", "MACHINE DISCONNECTED",
                        "Reason: machine is off or its cable is out.")

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
        self._reset_cb = None
        self._reset_req = False

    def is_usb(self):
        return bool(self._cams) and getattr(self._cams[0], "camtype",
                                            "") == "usbcam"

    def reset_camera(self, cb=None):
        """Ask the grab thread to reset the camera to its default settings.
        cb(ok, message) is called from that thread when done."""
        if not self._running or not self._cams:
            if cb:
                cb(False, "camera is not running")
            return
        self._reset_cb = cb
        self._reset_req = True

    def _do_reset(self):
        cam = self._cams[0]
        try:
            if getattr(cam, "camtype", "") == "usbcam":
                ok, msg = cam.reset_settings()
            else:
                self._cams = self._open()          # MindVision: re-apply
                ok, msg = True, "camera settings re-applied"
        except Exception as e:
            ok, msg = False, f"reset failed: {e}"
        cb, self._reset_cb = self._reset_cb, None
        if cb:
            try:
                cb(ok, msg)
            except Exception:
                pass

    def start(self):
        if self._running:
            return
        self._running = True
        threading.Thread(target=self._loop, daemon=True).start()

    def _open(self):
        """Open the camera and apply the fixed settings (raises if absent)."""
        cams = _init_cameras()
        if not cams:
            raise IndexError("no camera")
        cams[0].get_things(exposure=34005, work_mode="Exposure")
        cams[0].get_things(exposure=34005, work_mode="Exposure")
        cams[0].get_things(exposure=74005, work_mode="Exposure")
        cams[0].get_things(gain=4, work_mode="Gain")
        cams[0].get_things(rGain=187, gGain=100, bGain=169,
                           work_mode="ColorGain")
        cams[0].get_things(saturation=130, work_mode="Saturation")
        cams[0].get_things(sharpnes=30, work_mode="Sharpness")
        cams[0].get_things(work_mode="auto_balance")
        return cams

    def _loop(self):
        try:
            self._cams = self._open()
            self._error = None
        except Exception as e:
            if isinstance(e, IndexError):
                self._error = ("No camera found.\n"
                               "Please check the camera cable and power.")
            else:
                self._error = ("The camera could not be started.\n"
                               "Please check the camera cable and power.\n"
                               f"({e})")
            self._running = False
            return
        fails = 0
        last_reopen = 0.0
        while self._running:
            if self._reset_req:
                self._reset_req = False
                self._do_reset()
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
                fails = 0
                self._error = None            # camera is (back) OK
            except Exception as e:
                fails += 1
                if fails >= 3:
                    # stop serving a stale picture as if it were live
                    self._error = ("No picture from the camera.\n"
                                   "Please check the camera cable and "
                                   "power.")
                    with self._lock:
                        self._latest = None
                    # try to re-open the camera every few seconds
                    if time.time() - last_reopen > 3.0:
                        last_reopen = time.time()
                        try:
                            self._cams = self._open()
                        except Exception:
                            pass
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
# -- palette: graphite + teal (industrial inspection look) -------------------
C_BG      = "#0d1117"      # window
C_PANEL   = "#161b22"      # panels / group boxes
C_ELEV    = "#1c232d"      # raised rows
C_LINE    = "#232b35"      # borders
C_FIELD   = "#0a0e13"      # inputs
C_TEXT    = "#e6edf3"
C_MUTED   = "#8b98a5"
C_FAINT   = "#5c6773"
C_ACCENT  = "#12a594"      # primary action (teal)
C_BLUE    = "#3d6ff5"
C_OK      = "#1f9d55"
C_NG      = "#d93a3a"
C_WARN    = "#c9821a"
C_NEUTRAL = "#242c38"

# THE WHOLE UI USES FOUR BUTTON COLOURS, NOTHING ELSE:
#   teal  = the main action of the screen
#   grey  = every ordinary action
#   amber = something that needs attention / a decision
#   red   = deletes and resets
# The call sites still pass the old colour names; they are mapped here, so
# the look of the whole program is decided in this one place.
# Only 3 button colours in the whole app: teal = the one main action in a
# panel, grey = an ordinary button, red = delete / reset. Nothing else.
_BTN_MAP = {
    "#2f9e44": C_ACCENT,      # save / confirm / add  -> main action
    "#e8590c": C_ACCENT,      # run
    "#0ca678": C_ACCENT,
    "#f59f00": C_ACCENT,      # finish / unlock -> same as main action
    "#e03131": C_NG,          # delete / reset
    "#c92a2a": C_NG,
    "#3b6ef5": C_NEUTRAL,     # everything else is grey
    "#44444e": C_NEUTRAL,
    "#845ef7": C_NEUTRAL,
    "#1971c2": C_NEUTRAL,
    "#74c0fc": C_NEUTRAL,
}

GLOBAL_QSS = f"""
* {{ font-family: 'Inter', 'Segoe UI', 'Noto Sans', sans-serif; }}
QMainWindow, QDialog {{ background: {C_BG}; }}
QWidget {{ color: {C_TEXT}; font-size: 10pt; }}
QLabel {{ background: transparent; border: none; }}
QFrame#card {{
    background: {C_PANEL}; border: 1px solid {C_LINE}; border-radius: 10px;
}}
QGroupBox {{
    background: {C_PANEL}; border: 1px solid {C_LINE}; border-radius: 10px;
    margin-top: 14px; padding: 10px 8px 8px 8px;
    font-weight: 700; color: {C_MUTED}; font-size: 8pt;
}}
QGroupBox::title {{
    subcontrol-origin: margin; left: 12px; padding: 0 6px;
    letter-spacing: 1px;
}}
QLineEdit, QComboBox {{
    background: {C_FIELD}; color: {C_TEXT}; border: 1px solid #2b333e;
    border-radius: 6px; padding: 7px 10px; font-size: 10pt;
    selection-background-color: {C_ACCENT};
}}
QLineEdit:focus, QComboBox:focus {{ border: 1px solid {C_ACCENT}; }}
QLineEdit:disabled {{ background: #10151b; color: {C_FAINT}; }}
QComboBox QAbstractItemView {{
    background: {C_PANEL}; color: {C_TEXT}; border: 1px solid {C_LINE};
    selection-background-color: {C_ACCENT};
}}
QTextEdit {{
    background: {C_FIELD}; color: #b9c4cf; border: 1px solid #222a34;
    border-radius: 8px;
    font-family: 'Consolas', 'DejaVu Sans Mono', monospace; font-size: 9pt;
}}
QCheckBox {{ color: #c3ccd6; spacing: 9px; font-size: 10pt; }}
QCheckBox::indicator {{
    width: 17px; height: 17px; border-radius: 4px;
    border: 1px solid #39424e; background: {C_FIELD};
}}
QCheckBox::indicator:checked {{
    background: {C_ACCENT}; border-color: {C_ACCENT};
}}
QListWidget, QTreeWidget {{
    background: {C_FIELD}; border: 1px solid #222a34; border-radius: 8px;
    font-size: 10pt;
}}
QListWidget::item, QTreeWidget::item {{ height: 26px; padding-left: 4px; }}
QListWidget::item:selected, QTreeWidget::item:selected {{
    background: #1d3a4d; color: white;
}}
QScrollBar:vertical {{ background: transparent; width: 11px; margin: 2px; }}
QScrollBar::handle:vertical {{
    background: #333c48; border-radius: 5px; min-height: 28px;
}}
QScrollBar::handle:vertical:hover {{ background: #414c5a; }}
QScrollBar::add-line, QScrollBar::sub-line {{ height: 0; width: 0; }}
QScrollArea {{ border: none; background: transparent; }}
QMessageBox, QInputDialog {{ background: {C_PANEL}; font-size: 11pt; }}
QMessageBox QLabel {{ font-size: 12pt; }}
QToolTip {{
    background: {C_ELEV}; color: {C_TEXT}; border: 1px solid {C_LINE};
    padding: 5px;
}}
"""


def _shade(hexcolor, f):
    h = hexcolor.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    r, g, b = (max(0, min(255, int(c * f))) for c in (r, g, b))
    return f"#{r:02x}{g:02x}{b:02x}"


def fit_size(want_w, want_h, min_w=420, min_h=320):
    """Shrink (w, h) so a window / dialog always fits on the screen it is
    opened on, however small that screen is.  Never used to grow a window
    bigger than what was asked for."""
    screen = QApplication.primaryScreen()
    if screen is None:
        return want_w, want_h
    geo = screen.availableGeometry()
    margin = 40
    w = min(want_w, max(min_w, geo.width() - margin))
    h = min(want_h, max(min_h, geo.height() - margin))
    return w, h


def apply_btn_color(btn, color):
    color = _BTN_MAP.get(str(color).lower(), color)
    border = ("1px solid #2f3a46" if color == C_NEUTRAL
              else f"1px solid {_shade(color, 0.86)}")
    btn.setStyleSheet(
        f"QPushButton {{ background: {color}; color: white;"
        f" border: {border};"
        f" border-radius: 6px; padding: 8px 14px; font-weight: 600;"
        f" font-size: 10pt; }}"
        f" QPushButton:hover {{ background: {_shade(color, 1.16)}; }}"
        f" QPushButton:pressed {{ background: {_shade(color, 0.84)}; }}"
        f" QPushButton:disabled {{ background: #141a21;"
        f" border: 1px solid #1e252d; color: #46505b; }}")


class FitButton(QPushButton):
    """A button that can never be squeezed narrower than its own text: the
    layout must give it at least the width of the words plus the padding,
    and it re-measures itself whenever its text changes."""
    PAD = 34

    def _text_w(self):
        lines = (self.text() or " ").split("\n")
        fm = self.fontMetrics()
        return max(fm.horizontalAdvance(t) for t in lines) + self.PAD

    def minimumSizeHint(self):
        h = super().minimumSizeHint()
        h.setWidth(max(h.width(), self._text_w()))
        return h

    def sizeHint(self):
        h = super().sizeHint()
        h.setWidth(max(h.width(), self._text_w()))
        return h

    def setText(self, text):
        super().setText(text)
        self.updateGeometry()


def styled_btn(master, text, cmd, color="#3b6ef5", **kw):
    """Same signature as the Tkinter version (master kept for call-site
    compatibility; Qt parenting is done by the layouts)."""
    b = FitButton(text)
    b.setSizePolicy(QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Fixed)
    b.clicked.connect(lambda *_: cmd())
    b.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
    # A barcode scanner ends every scan with Enter.  If a button ever holds
    # keyboard focus that Enter *activates the button* — which is how a second
    # scan used to press '<- Models' and throw you back to the model list.
    # Buttons here are mouse/touch driven, so they never need focus.
    b.setFocusPolicy(Qt.FocusPolicy.NoFocus)
    b.setAutoDefault(False)
    b.setDefault(False)
    b.setMinimumHeight(36)
    apply_btn_color(b, color)
    return b


def section_title(text, size=8):
    """Small upper-case caption used above a group of controls."""
    l = QLabel(text.upper())
    l.setStyleSheet(f"color: {C_FAINT}; font-size: {size}pt;"
                    " font-weight: 700; letter-spacing: 1px;")
    l.setAlignment(Qt.AlignmentFlag.AlignCenter)
    return l


def screen_title(text):
    l = QLabel(text)
    l.setStyleSheet(f"color: {C_TEXT}; font-size: 15pt; font-weight: 700;"
                    " letter-spacing: 0.3px;")
    return l


def hint_label(text, color=None):
    l = QLabel(text)
    l.setWordWrap(True)
    l.setStyleSheet(f"color: {color or C_MUTED}; font-size: 9pt;")
    return l


def btn_group(buttons, spacing=6, pad=(6, 5, 6, 5)):
    """Several buttons shown as ONE block, so a toolbar never looks like
    loose buttons scattered over the screen."""
    f = QFrame()
    f.setStyleSheet(f"QFrame {{ background: {C_ELEV};"
                    f" border: 1px solid {C_LINE}; border-radius: 8px; }}")
    lay = QHBoxLayout(f)
    lay.setContentsMargins(*pad)
    lay.setSpacing(spacing)
    for b in buttons:
        lay.addWidget(b)
    return f


def caption(text):
    """Small grey upper-case caption above a group of controls."""
    l = QLabel(str(text).upper())
    l.setStyleSheet(f"color: {C_FAINT}; font-size: 8pt; font-weight: 700;"
                    " letter-spacing: 1px; background: transparent;"
                    " border: none;")
    return l


def card(title=None, name="card", bg=None):
    """Rounded panel with an optional caption.  The style is bound to the
    frame's object name, so it never leaks onto the widgets inside."""
    f = QFrame()
    f.setObjectName(name)
    f.setStyleSheet(f"QFrame#{name} {{ background: {bg or C_PANEL};"
                    f" border: 1px solid {C_LINE}; border-radius: 10px; }}"
                    " QLabel { background: transparent; border: none; }")
    lay = QVBoxLayout(f)
    lay.setContentsMargins(12, 10, 12, 12)
    lay.setSpacing(8)
    if title:
        lay.addWidget(caption(title))
    return f, lay


def vline():
    l = QFrame()
    l.setFixedWidth(1)
    l.setStyleSheet(f"background: {C_LINE}; border: none;")
    return l


def simulate_group(app, master):
    """Compact 'send a test signal' box, for the main screen (the Monitor
    window has the same send box plus the full connect/handshake controls)."""
    w = QWidget()
    v = QVBoxLayout(w)
    v.setContentsMargins(0, 0, 0, 0)
    v.setSpacing(4)
    v.addWidget(caption("Simulate Machine Signal"))
    h = QHBoxLayout()
    h.setSpacing(6)
    e = QLineEdit()
    e.setPlaceholderText("signal text")
    e.setFixedWidth(150)

    def send():
        text = e.text().strip()
        if not text:
            return
        e.clear()
        app.engine.simulate(text)
    e.returnPressed.connect(send)
    h.addWidget(e)
    h.addWidget(styled_btn(master, "Send", send, color="#44444e"))
    v.addLayout(h)
    w.setSizePolicy(QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Fixed)
    return w


def tool_group(title, buttons):
    """Caption + a row of buttons: one block of the toolbar."""
    w = QWidget()
    v = QVBoxLayout(w)
    v.setContentsMargins(0, 0, 0, 0)
    v.setSpacing(4)
    v.addWidget(caption(title))
    h = QHBoxLayout()
    h.setSpacing(6)
    for b in buttons:
        h.addWidget(b)
    v.addLayout(h)
    w.setSizePolicy(QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Fixed)
    return w


class StatusChip(QFrame):
    """Pill with a coloured dot: green = working, red = problem,
    grey = off / not known yet."""

    def __init__(self, name):
        super().__init__()
        self.setObjectName("chip")
        self.setStyleSheet(f"QFrame#chip {{ background: {C_ELEV};"
                           f" border: 1px solid {C_LINE};"
                           " border-radius: 14px; }"
                           " QLabel { background: transparent; border: none; }")
        lay = QHBoxLayout(self)
        lay.setContentsMargins(10, 5, 12, 5)
        lay.setSpacing(7)
        self.dot = QLabel()
        self.dot.setFixedSize(10, 10)
        lay.addWidget(self.dot)
        self.lbl = QLabel(name)
        lay.addWidget(self.lbl)
        self.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        self.state = "init"
        self.set(None)

    def set(self, state, tip=""):
        if state == self.state:
            return
        self.state = state
        col = {True: C_OK, False: C_NG}.get(state, C_FAINT)
        self.dot.setStyleSheet(f"background: {col}; border-radius: 5px;")
        self.lbl.setStyleSheet(
            f"color: {C_TEXT if state is not None else C_MUTED};"
            " font-size: 9pt; font-weight: 600;")
        if tip:
            self.setToolTip(tip)


def btn_row(buttons, spacing=6):
    """One row of equally wide buttons."""
    w = QWidget()
    lay = QHBoxLayout(w)
    lay.setContentsMargins(0, 0, 0, 0)
    lay.setSpacing(spacing)
    for b in buttons:
        lay.addWidget(b, stretch=1)
    return w


def step_bar(steps, current):
    """A '1 2 3' guide strip: the step the user is on is highlighted and
    carries the instruction.  steps = [(short name, what to do), ...]"""
    f = QFrame()
    f.setStyleSheet(f"QFrame {{ background: {C_PANEL};"
                    f" border: 1px solid {C_LINE}; border-radius: 10px; }}"
                    " QLabel { background: transparent; border: none; }")
    lay = QVBoxLayout(f)
    lay.setContentsMargins(14, 10, 14, 10)
    lay.setSpacing(8)
    row = QHBoxLayout()
    row.setSpacing(8)
    for i, (name, _) in enumerate(steps):
        done = i < current
        here = i == current
        chip = QLabel(f"  {i + 1}.  {name}  ")
        chip.setAlignment(Qt.AlignmentFlag.AlignCenter)
        if here:
            chip.setStyleSheet(f"background: {C_ACCENT}; color: white;"
                               " font-size: 10pt; font-weight: 700;"
                               " padding: 7px 10px; border-radius: 6px;")
        elif done:
            chip.setStyleSheet(f"background: #16323a; color: {C_ACCENT};"
                               " font-size: 10pt; font-weight: 600;"
                               " padding: 7px 10px; border-radius: 6px;")
        else:
            chip.setStyleSheet(f"background: {C_ELEV}; color: {C_FAINT};"
                               " font-size: 10pt; padding: 7px 10px;"
                               " border-radius: 6px;")
        row.addWidget(chip, stretch=1)
    lay.addLayout(row)
    if 0 <= current < len(steps):
        tip = QLabel(steps[current][1])
        tip.setWordWrap(True)
        tip.setStyleSheet(f"color: {C_TEXT}; font-size: 11pt;")
        lay.addWidget(tip)
    return f


def panel(bg=C_PANEL, radius=10):
    w = QFrame()
    w.setStyleSheet(f"QFrame {{ background: {bg};"
                    f" border: 1px solid {C_LINE};"
                    f" border-radius: {radius}px; }}"
                    " QLabel { background: transparent; border: none; }")
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
    zb = []
    for txt, cmd in (("-", canvas.zoom_out), ("Fit", canvas.fit),
                     ("+", canvas.zoom_in)):
        b = styled_btn(row, txt, cmd, color="#44444e")
        b.setFixedWidth(64 if txt == "Fit" else 44)
        zb.append(b)
    lay.addWidget(btn_group(zb, spacing=4))
    hint = QLabel("wheel = zoom · right-drag = pan")
    hint.setStyleSheet(f"color: {C_FAINT}; font-size: 8pt;"
                       " padding-left: 10px;")
    lay.addWidget(hint)
    lay.addStretch()
    return row


# ============================================================================
# MAIN APP  (global comm engine + lock management)
# ============================================================================
# ============================================================================
# HEALTH CHECKS + POPUPS
# ============================================================================
ETH_REASON = ("Ethernet cable disconnected, switch/router off, "
              "or IP address changed.")
CAM_REASON = ("camera cable or power disconnected, camera IP changed, "
              "or another program is using the camera.")
COMM_REASON = ("Ethernet cable disconnected, or the IP address changed.")
_NOT_ETHERNET = ("loopback", "wi-fi", "wifi", "wlan", "wireless", "bluetooth",
                 "vethernet", "vmware", "virtualbox", "vbox", "docker",
                 "veth", "tailscale", "zerotier", "tun", "tap", "br-")


def ethernet_connected():
    """True when at least one wired network adapter is up and has an IPv4
    address (the camera, the MES server and the client all sit on it).
    Needs psutil for a real link check (pip install psutil); without it
    only 'has a non-loopback IP' can be tested."""
    try:
        if psutil is not None:
            stats = psutil.net_if_stats()
            addrs = psutil.net_if_addrs()
            for name, st in stats.items():
                low = name.lower()
                if not st.isup or low == "lo" or any(
                        w in low for w in _NOT_ETHERNET):
                    continue
                for a in addrs.get(name, []):
                    if (a.family == socket.AF_INET
                            and not a.address.startswith(("127.", "169.254."))):
                        return True
            return False
        ips = socket.gethostbyname_ex(socket.gethostname())[2]
        return any(not ip.startswith(("127.", "169.254.")) for ip in ips)
    except Exception:
        return True       # cannot tell -> do not raise a false alarm


def ask_text(parent, headline, body="", placeholder="", password=False):
    """Big, clear text box.  Returns the typed text, or None if cancelled."""
    dlg = QDialog(parent)
    dlg.setWindowTitle(headline)
    dlg.setModal(True)
    dlg.setMinimumWidth(560)
    accent = C_ACCENT
    dlg.setStyleSheet(f"QDialog {{ background: {C_PANEL};"
                      f" border: 2px solid {accent}; border-radius: 12px; }}")
    lay = QVBoxLayout(dlg)
    lay.setContentsMargins(0, 0, 0, 0)
    lay.setSpacing(0)
    band = QLabel(headline)
    band.setAlignment(Qt.AlignmentFlag.AlignCenter)
    band.setWordWrap(True)
    band.setStyleSheet(f"background: {accent}; color: white;"
                       " font-size: 22pt; font-weight: 800; padding: 20px;"
                       " border-top-left-radius: 10px;"
                       " border-top-right-radius: 10px;")
    lay.addWidget(band)
    inner = QVBoxLayout()
    inner.setContentsMargins(26, 22, 26, 22)
    inner.setSpacing(14)
    if body:
        t = QLabel(body)
        t.setWordWrap(True)
        t.setAlignment(Qt.AlignmentFlag.AlignCenter)
        t.setStyleSheet(f"color: {C_MUTED}; font-size: 12pt;")
        inner.addWidget(t)
    edit = QLineEdit()
    edit.setPlaceholderText(placeholder)
    edit.setAlignment(Qt.AlignmentFlag.AlignCenter)
    edit.setMinimumHeight(54)
    if password:
        edit.setEchoMode(QLineEdit.EchoMode.Password)
    edit.setStyleSheet(f"background: {C_FIELD}; color: white;"
                       f" border: 1px solid {C_LINE}; border-radius: 8px;"
                       " font-size: 18pt;"
                       + (" letter-spacing: 6px;" if password else ""))
    edit.returnPressed.connect(dlg.accept)
    inner.addWidget(edit)
    row = QHBoxLayout()
    row.setSpacing(10)
    bc = QPushButton("Cancel")
    bc.setMinimumHeight(50)
    bc.setFocusPolicy(Qt.FocusPolicy.NoFocus)
    apply_btn_color(bc, C_NEUTRAL)
    bc.setStyleSheet(bc.styleSheet() + " QPushButton { font-size: 13pt; }")
    bc.clicked.connect(dlg.reject)
    row.addWidget(bc, stretch=1)
    bo = QPushButton("Continue")
    bo.setMinimumHeight(50)
    bo.setFocusPolicy(Qt.FocusPolicy.NoFocus)
    apply_btn_color(bo, C_OK)
    bo.setStyleSheet(bo.styleSheet() + " QPushButton { font-size: 13pt; }")
    bo.clicked.connect(dlg.accept)
    row.addWidget(bo, stretch=1)
    inner.addLayout(row)
    lay.addLayout(inner)
    edit.setFocus()
    if dlg.exec() == QDialog.DialogCode.Accepted:
        return edit.text()
    return None


def ask_password(parent, headline, body=""):
    """Big, clear password box.  Returns the typed text, or None."""
    dlg = QDialog(parent)
    dlg.setWindowTitle("Password")
    dlg.setModal(True)
    dlg.setMinimumWidth(560)
    dlg.setStyleSheet(f"QDialog {{ background: {C_PANEL};"
                      f" border: 2px solid {C_WARN}; border-radius: 12px; }}")
    lay = QVBoxLayout(dlg)
    lay.setContentsMargins(0, 0, 0, 0)
    lay.setSpacing(0)
    band = QLabel(headline)
    band.setAlignment(Qt.AlignmentFlag.AlignCenter)
    band.setWordWrap(True)
    band.setStyleSheet(f"background: {C_WARN}; color: white;"
                       " font-size: 22pt; font-weight: 800; padding: 20px;"
                       " border-top-left-radius: 10px;"
                       " border-top-right-radius: 10px;")
    lay.addWidget(band)
    inner = QVBoxLayout()
    inner.setContentsMargins(26, 22, 26, 22)
    inner.setSpacing(14)
    if body:
        t = QLabel(body)
        t.setWordWrap(True)
        t.setAlignment(Qt.AlignmentFlag.AlignCenter)
        t.setStyleSheet(f"color: {C_MUTED}; font-size: 12pt;")
        inner.addWidget(t)
    edit = QLineEdit()
    edit.setEchoMode(QLineEdit.EchoMode.Password)
    edit.setAlignment(Qt.AlignmentFlag.AlignCenter)
    edit.setMinimumHeight(54)
    edit.setStyleSheet(f"background: {C_FIELD}; color: white;"
                       f" border: 1px solid {C_LINE}; border-radius: 8px;"
                       " font-size: 20pt; letter-spacing: 6px;")
    edit.returnPressed.connect(dlg.accept)
    inner.addWidget(edit)
    row = QHBoxLayout()
    row.setSpacing(10)
    bc = QPushButton("Cancel")
    bc.setMinimumHeight(50)
    bc.setFocusPolicy(Qt.FocusPolicy.NoFocus)
    apply_btn_color(bc, C_NEUTRAL)
    bc.setStyleSheet(bc.styleSheet() + " QPushButton { font-size: 13pt; }")
    bc.clicked.connect(dlg.reject)
    row.addWidget(bc, stretch=1)
    bo = QPushButton("Continue")
    bo.setMinimumHeight(50)
    bo.setFocusPolicy(Qt.FocusPolicy.NoFocus)
    apply_btn_color(bo, C_OK)
    bo.setStyleSheet(bo.styleSheet() + " QPushButton { font-size: 13pt; }")
    bo.clicked.connect(dlg.accept)
    row.addWidget(bo, stretch=1)
    inner.addLayout(row)
    lay.addLayout(inner)
    edit.setFocus()
    if dlg.exec() == QDialog.DialogCode.Accepted:
        return edit.text()
    return None


class Message(QDialog):
    """The ONE popup used everywhere: a big colour band with a short
    headline, then plain-English text in large print and one wide button.

    Non-modal and always on top, so a signal from the client is never
    blocked by a message left open on the screen.
    """

    KINDS = {
        "error": (C_NG,     "Problem"),
        "warn":  (C_WARN,   "Please check"),
        "ok":    (C_OK,     "Done"),
        "info":  ("#2563c9", "Information"),
    }

    AUTO_CLOSE_MS = 2000          # every popup closes itself after 2 s

    def __init__(self, parent, title, headline, lines=(), color=C_OK,
                 auto_ms=0, retry=None, retry_text="Try again"):
        super().__init__(parent)
        self.setWindowTitle(title or "Message")
        self.setWindowFlag(Qt.WindowType.WindowStaysOnTopHint, True)
        self.setModal(False)
        self.setMinimumWidth(min(600, fit_size(600, 300)[0]))
        self.setStyleSheet(f"QDialog {{ background: {C_PANEL};"
                           f" border: 2px solid {color}; border-radius: 12px; }}")

        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)

        band = QLabel(str(headline))
        band.setAlignment(Qt.AlignmentFlag.AlignCenter)
        band.setWordWrap(True)
        band.setStyleSheet(f"background: {color}; color: white;"
                           " font-size: 26pt; font-weight: 800;"
                           " padding: 22px 20px; border-top-left-radius: 10px;"
                           " border-top-right-radius: 10px;"
                           " letter-spacing: 0.5px;")
        lay.addWidget(band)

        body = QVBoxLayout()
        body.setContentsMargins(26, 22, 26, 22)
        body.setSpacing(16)
        text = "\n".join(str(x) for x in lines) if not isinstance(lines, str) \
            else lines
        text = _short_text(text)
        if text.strip():
            t = QLabel(text)
            t.setWordWrap(True)
            t.setAlignment(Qt.AlignmentFlag.AlignCenter)
            t.setStyleSheet(f"color: {C_TEXT}; font-size: 16pt;"
                            " line-height: 150%;")
            body.addWidget(t)

        brow = QHBoxLayout()
        brow.setSpacing(10)
        if retry is not None:
            rb = QPushButton(retry_text)
            rb.setMinimumHeight(52)
            rb.setFocusPolicy(Qt.FocusPolicy.NoFocus)
            apply_btn_color(rb, C_ACCENT)
            rb.setStyleSheet(rb.styleSheet()
                             + " QPushButton { font-size: 14pt; }")

            def _retry():
                self.close()
                try:
                    retry()
                except Exception:
                    pass
            rb.clicked.connect(_retry)
            brow.addWidget(rb, stretch=1)
        if brow.count():
            body.addLayout(brow)
        lay.addLayout(body)
        QTimer.singleShot(self.AUTO_CLOSE_MS, self.close)

    def mousePressEvent(self, e):          # a tap closes it early
        self.close()

    # -- the only entry points the screens use -----------------------------
    @classmethod
    def show_msg(cls, parent, kind, headline, body="", auto_ms=0, retry=None,
                 title=None):
        color, deftitle = cls.KINDS.get(kind, cls.KINDS["info"])
        dlg = cls(parent, title or deftitle, headline, body, color,
                  auto_ms, retry)
        dlg.show()
        dlg.raise_()
        return dlg

    @classmethod
    def error(cls, parent, headline, body="", auto_ms=0, retry=None,
              title=None):
        return cls.show_msg(parent, "error", headline, body, auto_ms, retry,
                            title)

    @classmethod
    def warn(cls, parent, headline, body="", auto_ms=0, retry=None,
             title=None):
        return cls.show_msg(parent, "warn", headline, body, auto_ms, retry,
                            title)

    @classmethod
    def ok(cls, parent, headline, body="", auto_ms=2600, retry=None,
           title=None):
        return cls.show_msg(parent, "ok", headline, body, auto_ms, retry,
                            title)

    @classmethod
    def info(cls, parent, headline, body="", auto_ms=0, retry=None,
             title=None):
        return cls.show_msg(parent, "info", headline, body, auto_ms, retry,
                            title)

    # -- yes / no question (this one DOES wait for the answer) -------------
    @staticmethod
    def confirm(parent, headline, body="", yes="Yes", no="No",
                color="#2563c9"):
        dlg = QDialog(parent)
        dlg.setWindowTitle("Please confirm")
        dlg.setModal(True)
        dlg.setMinimumWidth(640)
        dlg.setStyleSheet(f"QDialog {{ background: {C_PANEL};"
                          f" border: 2px solid {color};"
                          f" border-radius: 12px; }}")
        lay = QVBoxLayout(dlg)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)
        band = QLabel(str(headline))
        band.setAlignment(Qt.AlignmentFlag.AlignCenter)
        band.setWordWrap(True)
        band.setStyleSheet(f"background: {color}; color: white;"
                           " font-size: 24pt; font-weight: 800;"
                           " padding: 22px 20px;"
                           " border-top-left-radius: 10px;"
                           " border-top-right-radius: 10px;")
        lay.addWidget(band)
        inner = QVBoxLayout()
        inner.setContentsMargins(26, 22, 26, 22)
        inner.setSpacing(16)
        if str(body).strip():
            t = QLabel(str(body))
            t.setWordWrap(True)
            t.setAlignment(Qt.AlignmentFlag.AlignCenter)
            t.setStyleSheet(f"color: {C_TEXT}; font-size: 14pt;")
            inner.addWidget(t)
        row = QHBoxLayout()
        row.setSpacing(10)
        bn = QPushButton(no)
        bn.setMinimumHeight(52)
        bn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        apply_btn_color(bn, C_NEUTRAL)
        bn.setStyleSheet(bn.styleSheet() + " QPushButton { font-size: 13pt; }")
        bn.clicked.connect(dlg.reject)
        row.addWidget(bn, stretch=1)
        by = QPushButton(yes)
        by.setMinimumHeight(52)
        by.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        apply_btn_color(by, C_OK)
        by.setStyleSheet(by.styleSheet() + " QPushButton { font-size: 13pt; }")
        by.clicked.connect(dlg.accept)
        row.addWidget(by, stretch=1)
        inner.addLayout(row)
        lay.addLayout(inner)
        return dlg.exec() == QDialog.DialogCode.Accepted


ResultPopup = Message        # old name kept for existing call sites


def _first_line(text, n=70):
    t = str(text).strip().splitlines()
    t = t[-1] if t else ""
    return t if len(t) <= n else t[:n - 3] + "..."


def _short_text(text):
    """Popups carry one or two short lines, never a paragraph of detail."""
    paras = [p.strip() for p in re.split(r"\n\s*\n", str(text))
             if p.strip()]
    return "\n".join(paras[:2])


class App(QMainWindow):
    _invoke = pyqtSignal(int, object)     # cross-thread 'after' (ms, fn)

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Inspection")
        self.setMinimumSize(1000, 620)
        self._fit_window_to_screen()
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
        self._alerts = {}                      # key -> open popup
        self._health = {"ethernet": True, "camera": True}
        self._health_timer = QTimer(self)      # Ethernet / camera watchdog
        self._health_timer.timeout.connect(self._health_tick)
        self._health_timer.start(max(500, int(float(
            self.app_config.data["health"].get("interval_s", 2)) * 1000)))
        QTimer.singleShot(800, self._health_tick)
        self.show_home()

    # -- window always matches the screen it opens on ------------------------
    def _fit_window_to_screen(self):
        """Size and place the window from the real screen, instead of a
        fixed pixel size — so it never runs off a small monitor and never
        looks tiny and lost on a big one."""
        screen = self.screen() or QApplication.primaryScreen()
        if screen is None:
            self.resize(1420, 830)
            return
        geo = screen.availableGeometry()
        margin = 60
        w = min(1420, geo.width() - margin)
        h = min(830, geo.height() - margin)
        w = max(1000, w)
        h = max(620, h)
        self.resize(w, h)
        self.move(geo.x() + (geo.width() - w) // 2,
                  geo.y() + (geo.height() - h) // 2)
        # A small panel screen: use all of it rather than leaving a margin.
        if geo.width() <= 1366 or geo.height() <= 800:
            self.showMaximized()

    # -- popups: lost Ethernet / lost camera ----------------------------------
    def alert(self, key, headline, text, retry=None):
        """Big red popup (one per key: a second one replaces the first)."""
        old = self._alerts.pop(key, None)
        if old is not None:
            old.close()
        self._alerts[key] = Message.error(self, headline, text, retry=retry)

    def clear_alert(self, key):
        box = self._alerts.pop(key, None)
        if box is not None:
            box.close()

    def _health_tick(self):
        cfg = self.app_config.data["health"]
        if cfg.get("check_ethernet", True):
            ok = ethernet_connected()
            if ok != self._health["ethernet"]:
                self._health["ethernet"] = ok
                if ok:
                    self.clear_alert("ethernet")
                    self.comm_log.add("EVT", "Ethernet connection restored")
                else:
                    self.comm_log.add("ERR", "Ethernet NOT connected")
                    self.alert("ethernet", "NETWORK NOT CONNECTED",
                               "Reason: network cable is out or the switch is off.")
        if cfg.get("check_camera", True):
            err = self.camera.error      # only set once the camera was started
            ok = not err
            if ok != self._health["camera"]:
                self._health["camera"] = ok
                if ok:
                    self.clear_alert("camera")
                    self.comm_log.add("EVT", "Camera connection restored")
                else:
                    self.comm_log.add("ERR", f"Camera NOT connected: {err}")
                    self.alert("camera", "CAMERA NOT CONNECTED",
                               "Reason: camera cable is out or the camera has no power.",
                               retry=self.camera.start)

    def hardware_ok(self, parent=None):
        """Re-raise the popup for anything still broken (used before a run).
        Returns (ethernet_ok, camera_ok)."""
        eth = ethernet_connected() if self.app_config.data["health"].get(
            "check_ethernet", True) else True
        cam = not self.camera.error
        if not eth:
            self._health["ethernet"] = False
            self.alert("ethernet", "NETWORK NOT CONNECTED",
                       "Reason: network cable is out or the switch is off.")
        if not cam:
            self._health["camera"] = False
            self.alert("camera", "CAMERA NOT CONNECTED",
                       "Reason: camera cable is out or the camera has no power.",
                       retry=self.camera.start)
        return eth, cam

    def ask_validation_password(self, parent):
        pw = ask_password(parent, "Validation password needed",
                          "These parts are tested but never reported to "
                          "MES, so a separate password is needed.")
        if pw is None:
            return False
        if pw == str(self.app_config.data.get("validation_password", "5678")):
            return True
        Message.error(parent, "Wrong password",
                      "That validation password is not correct.")
        return False

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
        Message.ok(self, "LOCKED",
                  "Setup is now hidden. Operator mode.")
        self.refresh_screen()

    def unlock_with_password(self, parent):
        pw = ask_password(parent, "Enter the password",
                          "The setup buttons stay hidden until the correct "
                          "password is entered.")
        if pw is None:
            return
        if pw == str(self.app_config.data.get("password", "1234")):
            self.app_config.data["locked"] = False
            self.app_config.save()
            self.comm_log.add("EVT", "UI UNLOCKED (controls open)")
            self.refresh_screen()
        else:
            Message.error(parent, "Wrong password",
                          "That password is not correct. Please try again.")

    def lock_button(self, master):
        """Shared 'Controls'/'Lock' button for the top bars."""
        if self.is_locked():
            return styled_btn(master, "Unlock Setup",
                              lambda: self.unlock_with_password(master),
                              color="#f59f00")
        return styled_btn(master, "Lock", self.lock, color="#44444e")

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
    """One line: machine connection lamp + status + the Monitor button.
    Connect / handshake / test signals all live in the Monitor window."""

    def __init__(self, master, app, mini_log_lines=0):
        super().__init__()
        self.app = app
        self._alive = True
        self.setStyleSheet(f"QFrame {{ background: {C_PANEL};"
                           f" border: 1px solid {C_LINE};"
                           f" border-radius: 10px; }}"
                           " QLabel { background: transparent;"
                           " border: none; }")
        row = QHBoxLayout(self)
        row.setContentsMargins(12, 8, 8, 8)
        row.setSpacing(10)
        self.lamp = QLabel()
        self.lamp.setFixedSize(12, 12)
        row.addWidget(self.lamp)
        cap = QLabel("MACHINE")
        cap.setStyleSheet(f"color: {C_FAINT}; font-size: 8pt;"
                          " font-weight: 700; letter-spacing: 1px;")
        row.addWidget(cap)
        self.status_lbl = QLabel("")
        self.status_lbl.setStyleSheet(f"color: {C_MUTED}; font-size: 9pt;")
        row.addWidget(self.status_lbl, stretch=1)
        row.addWidget(styled_btn(self, "Monitor",
                                 lambda: CommMonitor(self, app).show(),
                                 color="#44444e"))
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._poll)
        self._timer.start(500)
        self._poll()

    def _poll(self):
        if not self._alive:
            return
        eng = self.app.engine
        on = eng.enabled and eng.client_count() > 0
        col = C_OK if on else (C_WARN if eng.enabled else C_FAINT)
        self.lamp.setStyleSheet(f"background: {col}; border-radius: 6px;")
        self.status_lbl.setText(eng.status_text())


class CommMonitor(QDialog):
    COLORS = {"IN": "#74c0fc", "OUT": "#69db7c",
              "ERR": "#ff8787", "EVT": "#9a9aa5"}

    def __init__(self, master, app):
        super().__init__(master)
        self.app = app
        self._alive = True
        self._last_seq = 0
        self.setWindowTitle("Monitor - machine communication")
        self.resize(*fit_size(780, 660))
        lay = QVBoxLayout(self)

        # connection on/off + handshake: the only place these are switched
        ctl = QFrame()
        ctl.setStyleSheet(f"QFrame {{ background: {C_ELEV};"
                          f" border: 1px solid {C_LINE}; border-radius: 8px; }}"
                          " QLabel { background: transparent; border: none; }")
        cl = QHBoxLayout(ctl)
        cl.setContentsMargins(12, 8, 8, 8)
        cl.setSpacing(10)
        self.lamp = QLabel()
        self.lamp.setFixedSize(12, 12)
        cl.addWidget(self.lamp)
        self.status_lbl = QLabel("")
        self.status_lbl.setWordWrap(True)
        self.status_lbl.setStyleSheet(f"color: {C_TEXT}; font-size: 10pt;")
        cl.addWidget(self.status_lbl, stretch=1)
        self.btn_toggle = styled_btn(self, "Connect", self.toggle_comm,
                                     color="#2f9e44")
        self.btn_hs = styled_btn(self, "Handshake ON", self.toggle_handshake,
                                 color="#44444e")
        cl.addWidget(self.btn_toggle)
        cl.addWidget(self.btn_hs)
        lay.addWidget(ctl)

        box = QGroupBox(" Connection & Handshake (global) ")
        grid = QGridLayout()
        s = app.comm_settings.data
        self.vars = {}
        fields = [("Listen IP", "ip", s["connection"].get("ip", "0.0.0.0")),
                  ("TCP Port", "port", s["connection"].get("port", 5000)),
                  ("Handshake ping (client->us)", "ping",
                   s["handshake"].get("ping", "PING")),
                  ("Handshake reply (us->client)", "reply",
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
        savew.addWidget(styled_btn(self, "Save settings",
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
        sl = QLabel("Simulate machine:")
        sl.setStyleSheet("color: #c8c8d0; font-size: 9pt;")
        sim.addWidget(sl)
        self.sim_entry = QLineEdit()
        self.sim_entry.returnPressed.connect(self.simulate)
        sim.addWidget(self.sim_entry, stretch=1)
        sim.addWidget(styled_btn(self, "Send", self.simulate))
        lay.addLayout(sim)

        pl = QLabel(f"Log files: {os.path.abspath(COMM_LOG_DIR)}"
                    "  (one file per day)")
        pl.setStyleSheet(f"color: {C_FAINT}; font-size: 8pt;")
        pl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lay.addWidget(pl)

        self._append(self.app.comm_log.since(0))
        self._refresh_ctl()
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._poll)
        self._timer.start(300)

    def closeEvent(self, e):
        self._alive = False
        e.accept()

    def toggle_comm(self):
        if self.app.engine.enabled:
            self.app.engine.stop()
        else:
            self.app.engine.start()
        self._refresh_ctl()

    def toggle_handshake(self):
        hs = self.app.comm_settings.data["handshake"]
        hs["enabled"] = not hs.get("enabled", True)
        self.app.comm_settings.save()
        self.app.comm_log.add("EVT", "handshake "
                              + ("ENABLED" if hs["enabled"] else "DISABLED"))
        self._refresh_ctl()

    def _refresh_ctl(self):
        eng = self.app.engine
        on = eng.enabled and eng.client_count() > 0
        col = C_OK if on else (C_WARN if eng.enabled else C_FAINT)
        self.lamp.setStyleSheet(f"background: {col}; border-radius: 6px;")
        self.status_lbl.setText(eng.status_text())
        self.btn_toggle.setText("Disconnect" if eng.enabled else "Connect")
        apply_btn_color(self.btn_toggle,
                        "#44444e" if eng.enabled else "#2f9e44")
        hs_on = self.app.comm_settings.data["handshake"].get("enabled", True)
        self.btn_hs.setText(f"Handshake {'ON' if hs_on else 'OFF'}")

    def save_settings(self):
        try:
            port = int(self.vars["port"].get().strip())
        except ValueError:
            Message.error(self, "Bad port",
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
        self._refresh_ctl()
        self._append(self.app.comm_log.since(self._last_seq))


class DetectionLogDialog(QDialog):
    """What-happened log for one model — opened on demand with a button,
    the same way the Monitor dialog opens for communication events."""

    def __init__(self, master, lines):
        super().__init__(master)
        self.setWindowTitle("Detection Logs")
        self.resize(*fit_size(640, 560))
        lay = QVBoxLayout(self)
        t = QLabel("WHAT THE MODEL HAS BEEN DOING")
        t.setStyleSheet(f"color: {C_MUTED}; font-weight: bold;")
        t.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lay.addWidget(t)
        self.txt = QTextEdit()
        self.txt.setReadOnly(True)
        self.txt.setStyleSheet(
            f"QTextEdit {{ background: {C_FIELD}; color: #c8c8d0;"
            f" border: none; border-radius: 6px; font-size: 9pt; }}")
        for msg in lines:
            self.txt.append(html.escape(msg, quote=False))
        self.txt.verticalScrollBar().setValue(
            self.txt.verticalScrollBar().maximum())
        lay.addWidget(self.txt, stretch=1)
        close_row = QHBoxLayout()
        close_row.addStretch()
        close_row.addWidget(styled_btn(self, "Close", self.close,
                                       color="#44444e"))
        lay.addLayout(close_row)

    def append_line(self, msg):
        self.txt.append(html.escape(msg, quote=False))
        self.txt.verticalScrollBar().setValue(
            self.txt.verticalScrollBar().maximum())


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
        self.resize(*fit_size(560, 520))

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
                      "master.  After Finish Model, a SCAN QR CODE button starts\n"
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
        self.btn_test = styled_btn(self, "Test scan", self.start_test,
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
        brow.addWidget(styled_btn(self, "Save", self.save,
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
            self.test_lbl.setText(f"Scanned:  '{code}'")
            self.test_lbl.setStyleSheet("color: #69db7c; font-size: 9pt;")
        else:
            self.test_lbl.setText("Nothing was scanned.")
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
        self.setWindowTitle(f"MES settings - {head_name}")
        self.resize(*fit_size(600, 700, min_h=480))

        cfg = app.heads.get_mes(head_name)
        lay = QVBoxLayout(self)

        title = QLabel(f"MES SETTINGS   -   model '{head_name}'")
        title.setStyleSheet("color: white; font-size: 13pt;"
                            " font-weight: bold;")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lay.addWidget(title)

        if not MES_AVAILABLE:
            warn = QLabel("MES cannot be used: mes_client is missing "
                          f"({MES_IMPORT_ERR}).\nPut mes_client.py next to "
                          "this file and install 'requests'.")
            warn.setStyleSheet("color: #ff8787; font-size: 9pt;")
            warn.setWordWrap(True)
            lay.addWidget(warn)

        self.var_enabled = BVar(QCheckBox(
            "Send the results of this model to MES"))
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
            l.setMinimumWidth(160)
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

        # ---- validation serials: MES is DISABLED for these (password)
        self._val_serials = [str(x).strip() for x in
                             (cfg.get("validation_serials") or [])
                             if str(x).strip()]
        self._val_unlocked = False
        vbox = QGroupBox(" VALIDATION BARCODES (never sent to MES) ")
        vlay = QVBoxLayout(vbox)
        vlay.addWidget(hint_label(
            "A part whose barcode is in this list is tested as usual, but "
            "nothing about it is sent to MES. Every other barcode uses MES "
            "normally.", C_FAINT))
        self.val_list = QListWidget()
        self.val_list.setFixedHeight(90)
        vlay.addWidget(self.val_list)
        vrow = QHBoxLayout()
        self.val_edit = QLineEdit()
        self.val_edit.setPlaceholderText("type or scan a barcode, then "
                                         "press Enter")
        self.val_edit.setStyleSheet(f"background: {C_FIELD}; color: white;"
                                    " padding: 5px; border-radius: 4px;")
        self.val_edit.returnPressed.connect(self._val_add)
        vrow.addWidget(self.val_edit, stretch=1)
        self.val_add_btn = styled_btn(vbox, "+ Add", self._val_add,
                                      color="#2f9e44")
        vrow.addWidget(self.val_add_btn)
        self.val_del_btn = styled_btn(vbox, "Remove selected",
                                      self._val_remove, color="#c92a2a")
        vrow.addWidget(self.val_del_btn)
        vlay.addLayout(vrow)
        self.val_unlock_btn = styled_btn(
            vbox, "Unlock with the validation password", self._val_unlock,
            color="#f59f00")
        vlay.addWidget(self.val_unlock_btn)
        lay.addWidget(vbox)
        self._val_refresh()

        self.var_block = BVar(QCheckBox(
            "Do not allow a part that failed the station before this one "
            "(applies once the screen is locked for production)"))
        self.var_block.set(bool(cfg.get("block_on_prev_fail", True)))
        lay.addWidget(self.var_block.c)

        self.var_imgs = BVar(QCheckBox(
            "Send the result pictures to MES after every part"))
        self.var_imgs.set(bool(cfg.get("upload_images", True)))
        lay.addWidget(self.var_imgs.c)

        note = QLabel("Logs are written to:  "
                      + app.heads.mes_log_dir(head_name))
        note.setStyleSheet(f"color: {C_FAINT}; font-size: 8pt;")
        note.setWordWrap(True)
        lay.addWidget(note)

        self.test_lbl = QLabel("Press Test connection to check these "
                               "details.")
        self.test_lbl.setStyleSheet(f"color: {C_MUTED}; font-size: 9pt;")
        self.test_lbl.setWordWrap(True)
        lay.addWidget(self.test_lbl)

        brow = QHBoxLayout()
        brow.addStretch()
        brow.addWidget(styled_btn(self, "Test connection", self.test_login,
                                  color="#f59f00"))
        brow.addWidget(styled_btn(self, "Save", self.save,
                                  color="#2f9e44"))
        brow.addWidget(styled_btn(self, "Close", self._close,
                                  color="#44444e"))
        brow.addStretch()
        lay.addLayout(brow)

    # ------------------------------------------------------------------
    def _val_refresh(self):
        self.val_list.clear()
        self.val_list.addItems(self._val_serials)
        for w in (self.val_edit, self.val_add_btn, self.val_del_btn):
            w.setEnabled(self._val_unlocked)
        self.val_unlock_btn.setVisible(not self._val_unlocked)

    def _val_unlock(self):
        if self.app.ask_validation_password(self):
            self._val_unlocked = True
            self._val_refresh()
            self.val_edit.setFocus()

    def _val_add(self):
        if not self._val_unlocked:
            return
        code = self.val_edit.text().strip()
        self.val_edit.clear()
        if not code:
            return
        if code in self._val_serials:
            Message.info(self, "ALREADY IN THE LIST",
                         f"The barcode '{code}' is already saved.",
                         auto_ms=3000)
        else:
            self._val_serials.append(code)
            self._val_refresh()
            self.val_list.scrollToBottom()
        self.val_edit.setFocus()

    def _val_remove(self):
        if not self._val_unlocked:
            return
        sel = [i.text() for i in self.val_list.selectedItems()]
        if sel:
            self._val_serials = [x for x in self._val_serials
                                 if x not in sel]
            self._val_refresh()

    def _collect(self):
        d = {k: self.vars[k].get().strip() for k, _, _ in self.FIELDS}
        d["validation_serials"] = list(self._val_serials)
        d["enabled"] = bool(self.var_enabled.get())
        d["block_on_prev_fail"] = bool(self.var_block.get())
        d["upload_images"] = bool(self.var_imgs.get())
        return d

    def test_login(self):
        if not MES_AVAILABLE:
            self.test_lbl.setText(f"Not available: "
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
            self.test_lbl.setText(("Connected. " if ok else "Failed. ")
                                  + str(msg))
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
                Message.error(
                    self, "MES DETAILS MISSING",
                    "Fill in: " + ", ".join(missing))
                return
        self.app.heads.set_mes(self.head, **cfg)
        self.app.comm_log.add("EVT", f"MES config saved for model "
                                     f"'{self.head}' "
                                     f"(enabled={cfg['enabled']})")
        Message.ok(self, "SAVED",
                   f"The MES settings for '{self.head}' are saved.")

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

        t = QLabel("SCAN YOUR ID BADGE")
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
        brow.addWidget(styled_btn(self, "Scan  (F2)", self.open_scan,
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
        self.setStyleSheet(f"background-color: {C_BG};")
        lay = QVBoxLayout(self)
        lay.setContentsMargins(26, 16, 26, 16)
        lay.setSpacing(12)
        top = QHBoxLayout()
        head = QVBoxLayout()
        head.setSpacing(2)
        t = QLabel("Vision Inspection")
        t.setStyleSheet(f"color: {C_TEXT}; font-size: 22pt; font-weight: 800;"
                        " letter-spacing: 0.4px;")
        head.addWidget(t)
        sub = section_title("select a model to open")
        sub.setAlignment(Qt.AlignmentFlag.AlignLeft)
        head.addWidget(sub)
        top.addLayout(head)
        top.addStretch()
        top.addWidget(app.lock_button(self))
        lay.addLayout(top)
        if not app.is_locked():
            lay.addWidget(hint_label(
                "A model holds everything needed to check one part: what the "
                "camera looks at, the good/bad rule, the barcode scanner and "
                "the machine connection."))
            lay.addWidget(CommBar(self, app, mini_log_lines=4))
            addrow = QHBoxLayout()
            addrow.addStretch()
            addrow.addWidget(styled_btn(self, "Add Model", self.add_head,
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
            empty = QLabel("No models yet.\nPress  Add Model  to make the "
                           "first one.")
            empty.setStyleSheet(f"color: {C_FAINT}; font-size: 13pt;")
            empty.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self.list_lay.insertWidget(0, empty)
            return
        for i, h in enumerate(heads):
            row = panel(C_ELEV)
            rlay = QHBoxLayout(row)
            rlay.setContentsMargins(16, 10, 16, 10)
            rlay.setSpacing(10)
            name = QLabel(h)
            name.setStyleSheet(f"color: {C_TEXT}; font-size: 13pt;"
                               " font-weight: 700;")
            rlay.addWidget(name)
            cnt = self.app.heads.get_counters(h)["cycle"]
            if self.app.is_locked():
                status = (f"   cycles  OK {cnt['ok']} / NG {cnt['ng']}")
            else:
                cfg = CommConfig(self.app.heads, h)
                done, total, _ = cfg.progress()
                sc = self.app.heads.get_scanner(h)
                status = (f"  {len(self.app.heads.list_parents(h))} scene(s)"
                          "   ·   " + ("finished"
                                       if self.app.heads.is_finished(h)
                                       else "still being built")
                          + f"   ·   connection {done}/{total}"
                          + "   ·   scanner "
                          + ('on' if sc.get('enabled') else 'off')
                          + f"   ·   OK {cnt['ok']} / NG {cnt['ng']}")
            st = QLabel(status)
            st.setStyleSheet(f"color: {C_MUTED}; font-size: 9pt;")
            rlay.addWidget(st)
            rlay.addStretch()
            rowbtns = []
            if not self.app.is_locked():
                rowbtns.append(styled_btn(
                    row, "Delete", lambda hh=h: self.delete_head(hh),
                    color="#c92a2a"))
            rowbtns.append(styled_btn(row, "Open",
                                      lambda hh=h: self.app.show_head(hh),
                                      color="#2f9e44"))
            for b in rowbtns:
                b.setFixedWidth(104)
            rlay.addWidget(btn_group(rowbtns))
            self.list_lay.insertWidget(i, row)

    def add_head(self):
        name = ask_text(self, "Name the new model",
                        "Use the name of the part this station checks.",
                        placeholder="for example:  bracket A12")
        if not name or not name.strip():
            return
        name = name.strip()
        if self.app.heads.exists(name):
            Message.info(self, "Model exists",
                                    f"Model '{name}' already exists — "
                                    "opening it.")
            self.app.show_head(name)
            return
        self.app.heads.create(name)
        # ask about MES integration at model creation (details are filled in
        # later, once the model is finalised)
        if Message.confirm(
                self, "Use MES for this model?",
                "Choose Yes if the results of this model must be sent to "
                "MES.\n\nYou can fill in the MES details later, with the "
                "MES button, after the model is finished."):
            self.app.heads.set_mes(name, enabled=True)
        # ask to add the barcode scanner right at model creation
        if Message.confirm(
                self, "Use a barcode scanner?",
                "Choose Yes if the operator must scan a barcode before "
                "every test.\n\nYou can also set this up later, with the "
                "Scanner button."):
            ScannerDialog(self, self.app, name,
                          on_close=lambda: self.app.show_head(name)).show()
        else:
            self.app.show_head(name)

    def delete_head(self, head_name):
        if Message.confirm(
                self, "Delete this model?",
                f"Everything saved for the model '{head_name}' will be "
                "deleted for good: its scenes, its master pictures and its "
                "settings.\n\nThis cannot be undone.",
                yes="Delete", no="Keep it"):
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
        # VALIDATION part: serial typed/scanned via the password-protected
        # button; run normally but NEVER sent to MES (no check, no report)
        self.validation_mode = False
        self._popups = {}              # key -> open ResultPopup
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

        self._mes_online = None        # None = not checked yet
        root = QVBoxLayout(self)
        root.setContentsMargins(12, 10, 12, 10)
        root.setSpacing(10)
        # the screen itself holds keyboard focus so a scan that arrives while
        # the scan box is shut lands here (see keyPressEvent) instead of
        # leaking into whatever widget happened to be focused
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.n_scenes = len(app.heads.list_parents(head_name))
        # cycle tracking: a cycle completes when EVERY scene has been run
        # (in any order — scene1 then scene2, or one RUN MODEL)
        self.cycle_scene_results = {}

        # ============================================================ HEADER
        #  [Back]  MODEL name   OPERATOR id [Change]     o Camera o Network
        #                                                o MES o Machine [Lock]
        hdr, _hl = card(None, "hdr")
        hl = QHBoxLayout()
        _hl.setContentsMargins(10, 8, 10, 8)
        _hl.addLayout(hl)
        hl.setSpacing(14)
        hl.addWidget(styled_btn(self, "Back to Models", self.app.show_home,
                                color="#44444e"))
        tcol = QVBoxLayout()
        tcol.setSpacing(0)
        tcol.addWidget(caption("Model"))
        tcol.addWidget(screen_title(head_name))
        hl.addLayout(tcol)
        if self.operator_id:
            hl.addWidget(vline())
            ocol = QVBoxLayout()
            ocol.setSpacing(0)
            ocol.addWidget(caption("Operator"))
            ol = QLabel(str(self.operator_id))
            ol.setStyleSheet(f"color: {C_TEXT}; font-size: 12pt;"
                             " font-weight: 600;")
            ocol.addWidget(ol)
            hl.addLayout(ocol)
            hl.addWidget(styled_btn(self, "Change", self.switch_operator,
                                    color="#44444e"))
        hl.addStretch()
        self.chips = {}
        for key, name in (("camera", "Camera"), ("network", "Network"),
                          ("mes", "MES"), ("machine", "Machine")):
            if key == "mes" and not self.mes_active:
                continue
            self.chips[key] = StatusChip(name)
            hl.addWidget(self.chips[key])
        hl.addSpacing(6)
        hl.addWidget(app.lock_button(self))
        root.addWidget(hdr)

        # =========================================================== TOOLBAR
        # setup only.  A new model shows Add Scene; the rest appears when
        # the step before it is done.
        if not locked:
            tb, _tl = card(None, "tb")
            _tl.setContentsMargins(12, 8, 12, 10)
            tl_host = QWidget()
            tl_host.setStyleSheet("background: transparent;")
            tl = QHBoxLayout(tl_host)
            tl.setContentsMargins(0, 0, 0, 0)
            tl.setSpacing(18)
            tl_scroll = QScrollArea()
            tl_scroll.setWidget(tl_host)
            tl_scroll.setWidgetResizable(True)
            tl_scroll.setFrameShape(QFrame.Shape.NoFrame)
            tl_scroll.setVerticalScrollBarPolicy(
                Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
            tl_scroll.setHorizontalScrollBarPolicy(
                Qt.ScrollBarPolicy.ScrollBarAsNeeded)
            tl_scroll.setStyleSheet("QScrollArea { background: transparent; }")
            _tl.addWidget(tl_scroll)
            mb = [styled_btn(self, "Add Scene", self.add_parent,
                             color="#2f9e44" if not self.n_scenes
                             else "#44444e")]
            if self.n_scenes and not self.finished:
                mb.append(styled_btn(self, "Finish Model", self.finish_head,
                                     color="#f59f00"))
            tl.addWidget(tool_group("Model", mb))
            if self.finished:
                tl.addWidget(vline())
                tl.addWidget(tool_group("Station", [
                    styled_btn(self, "Scanner", self.open_scanner_cfg,
                               color="#44444e"),
                    styled_btn(self, "MES", self.open_mes_cfg,
                               color="#44444e"),
                    styled_btn(self, "Machine", self.open_comm,
                               color="#44444e")]))
            tl.addStretch()
            tools = []
            if self.n_scenes:
                tools.append(styled_btn(self, "Reset Camera",
                                        self.reset_camera, color="#44444e"))
            tools.append(styled_btn(self, "Detection Logs",
                                    self.open_log_dialog, color="#44444e"))
            tools.append(styled_btn(self, "Monitor",
                                    lambda: CommMonitor(self, app).show(),
                                    color="#44444e"))
            tl.addWidget(tool_group("Tools", tools))
            tl.addWidget(vline())
            tl.addWidget(simulate_group(app, self))
            root.addWidget(tb)

        body = QHBoxLayout()
        body.setSpacing(10)
        root.addLayout(body, stretch=1)

        # ====================================================== CAMERA (left)
        leftcol = QVBoxLayout()
        leftcol.setSpacing(10)
        cam, clay = card(None, "cam")
        clay.setContentsMargins(10, 8, 10, 10)
        vrow = QHBoxLayout()
        vrow.setSpacing(8)
        vrow.addWidget(caption("Camera"))
        vrow.addStretch()
        vbtns = [styled_btn(cam, "Live", self.back_to_live, color="#44444e")]
        if self.n_scenes:
            vbtns.append(styled_btn(cam, "Compare with Master",
                                    self.enter_compare, color="#44444e"))
        vrow.addWidget(btn_group(vbtns, spacing=4, pad=(4, 4, 4, 4)))
        clay.addLayout(vrow)
        self.canvas = VideoCanvas(cam)
        self.canvas.setMinimumSize(420, 260)
        clay.addWidget(self.canvas, stretch=1)
        clay.addWidget(zoom_controls(cam, self.canvas))

        self.result_ctl = QWidget()
        rc = QHBoxLayout(self.result_ctl)
        rc.setContentsMargins(0, 2, 0, 0)
        rc.addStretch()
        self.btn_pinmap = styled_btn(self.result_ctl, "Pin Map",
                                     self.toggle_pinmap, color="#44444e")
        rc.addWidget(btn_group([
            self.btn_pinmap,
            styled_btn(self.result_ctl, "Compare with Master",
                       self.enter_compare, color="#44444e"),
            styled_btn(self.result_ctl, "Back to Live",
                       self.back_to_live, color="#44444e")]))
        rc.addStretch()
        self.result_ctl.setVisible(False)
        clay.addWidget(self.result_ctl)

        self.compare_ctl = QWidget()
        cc = QHBoxLayout(self.compare_ctl)
        cc.setContentsMargins(0, 2, 0, 0)
        cc.addStretch()
        cl = QLabel("Compare:")
        cl.setStyleSheet(f"color: {C_MUTED}; font-size: 9pt;")
        cc.addWidget(cl)
        self.compare_sel = QComboBox()
        self.compare_sel.setMinimumWidth(220)
        cc.addWidget(self.compare_sel)
        self.btn_side = styled_btn(self.compare_ctl, "Side-by-Side",
                                   lambda: self.set_compare_mode("side"),
                                   color="#3b6ef5")
        self.btn_overlay = styled_btn(self.compare_ctl, "Overlay",
                                      lambda: self.set_compare_mode("overlay"),
                                      color="#44444e")
        cc.addWidget(btn_group([
            self.btn_side, self.btn_overlay,
            styled_btn(self.compare_ctl, "Back to Live",
                       self.back_to_live, color="#44444e")]))
        cc.addStretch()
        self.compare_ctl.setVisible(False)
        clay.addWidget(self.compare_ctl)
        leftcol.addWidget(cam, stretch=1)

        # ---- run bar under the camera (setup only)
        self.parent_btn_frame = None
        self.parent_btn_lay = None
        _thr_edit = QLineEdit(str(self.app.heads.get_settings(head_name)
                                  .get("distance_threshold", 20)))
        self.var_dist_thr = SVar(_thr_edit)
        if not locked:
            runbar, _rl = card(None, "run")
            _rl.setContentsMargins(12, 8, 12, 8)
            rl_host = QWidget()
            rl_host.setStyleSheet("background: transparent;")
            rl = QHBoxLayout(rl_host)
            rl.setContentsMargins(0, 0, 0, 0)
            rl.setSpacing(12)
            rl_scroll = QScrollArea()
            rl_scroll.setWidget(rl_host)
            rl_scroll.setWidgetResizable(True)
            rl_scroll.setFrameShape(QFrame.Shape.NoFrame)
            rl_scroll.setVerticalScrollBarPolicy(
                Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
            rl_scroll.setHorizontalScrollBarPolicy(
                Qt.ScrollBarPolicy.ScrollBarAsNeeded)
            rl_scroll.setStyleSheet("QScrollArea { background: transparent; }")
            _rl.addWidget(rl_scroll)
            if self.n_scenes:
                rl.addWidget(styled_btn(
                    runbar, "Run All Scenes",
                    lambda: self.execute(("head",), "MANUAL"),
                    color="#e8590c"))
                rl.addWidget(vline())
                rl.addWidget(caption("One scene"))
                self.parent_btn_frame = QWidget()
                self.parent_btn_frame.setStyleSheet("background: transparent;")
                self.parent_btn_lay = QHBoxLayout(self.parent_btn_frame)
                self.parent_btn_lay.setContentsMargins(0, 0, 0, 0)
                self.parent_btn_lay.setSpacing(6)
                self.parent_btn_lay.setAlignment(Qt.AlignmentFlag.AlignLeft)
                sc = QScrollArea()
                sc.setWidget(self.parent_btn_frame)
                sc.setWidgetResizable(True)
                sc.setFrameShape(QFrame.Shape.NoFrame)
                sc.setVerticalScrollBarPolicy(
                    Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
                sc.setHorizontalScrollBarPolicy(
                    Qt.ScrollBarPolicy.ScrollBarAsNeeded)
                sc.setFixedHeight(54)
                sc.setStyleSheet("QScrollArea { background: transparent; }")
                rl.addWidget(sc, stretch=1)
                rl.addWidget(vline())
                tcap = QVBoxLayout()
                tcap.setSpacing(2)
                tcap.addWidget(caption("Pin limit (px)"))
                _thr_edit.setFixedWidth(80)
                _thr_edit.setToolTip("A pin shifted more than this from the "
                                     "master (or missing) = NG")
                _thr_edit.returnPressed.connect(self._save_dist_thr)
                _thr_edit.editingFinished.connect(self._save_dist_thr)
                tcap.addWidget(_thr_edit)
                rl.addLayout(tcap)
            else:
                nl = QLabel("No scene yet.  Press  Add Scene  to start.")
                nl.setStyleSheet(f"color: {C_MUTED}; font-size: 10pt;")
                rl.addWidget(nl)
                rl.addStretch()
            leftcol.addWidget(runbar)
        body.addLayout(leftcol, stretch=1)

        # ===================================================== SIDEBAR (right)
        side = QWidget()
        side.setObjectName("side")
        side.setStyleSheet("QWidget#side { background: transparent; }")
        rlay = QVBoxLayout(side)
        rlay.setContentsMargins(0, 0, 2, 0)
        rlay.setSpacing(10)

        # -- RESULT
        rcard, rcl = card("Result", "res")
        self.status_big = QLabel("——")
        self.status_big.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.status_big.setWordWrap(True)
        self.status_big.setMinimumHeight(100)
        rcl.addWidget(self.status_big)
        self.status_sub = QLabel("Nothing tested yet")
        self.status_sub.setWordWrap(True)
        self.status_sub.setStyleSheet(f"color: {C_MUTED}; font-size: 9pt;")
        self.status_sub.setAlignment(Qt.AlignmentFlag.AlignCenter)
        rcl.addWidget(self.status_sub)
        self.set_big_status(None)
        # overall CYCLE result (all scenes of the unit) - separate from the
        # big status above, which shows the LAST RUN only
        self.cycle_result_lbl = QLabel("WHOLE PART:  -")
        self.cycle_result_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.cycle_result_lbl.setWordWrap(True)
        rcl.addWidget(self.cycle_result_lbl)
        self._set_cycle_result(None)
        self.val_banner = QLabel("VALIDATION PART\nnot sent to MES")
        self.val_banner.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.val_banner.setStyleSheet(
            f"background: {C_ACCENT}; color: white; font-weight: bold;"
            " font-size: 12pt; padding: 6px; border-radius: 6px;")
        self.val_banner.setVisible(False)
        rcl.addWidget(self.val_banner)
        rlay.addWidget(rcard)

        # -- PART (barcode / QR) — only once the model is finished
        if self.scan_active:
            pcard, pcl = card("Part QR code" if self.qr_scan_active
                              else "Part barcode", "part")
            self.scan_value_lbl = QLabel("—")
            self.scan_value_lbl.setWordWrap(True)
            self.scan_value_lbl.setStyleSheet(
                f"background: {C_FIELD}; color: {C_TEXT}; padding: 6px 10px;"
                " border-radius: 6px; font-family: Consolas, monospace;"
                " font-weight: bold; font-size: 13pt;")
            pcl.addWidget(self.scan_value_lbl)
            self.scan_status_lbl = QLabel("Waiting for the barcode")
            self.scan_status_lbl.setWordWrap(True)
            self.scan_status_lbl.setStyleSheet(f"color: {C_WARN};"
                                               " font-weight: bold;")
            pcl.addWidget(self.scan_status_lbl)
            if self.qr_scan_active:
                if not self.qr_master and not locked:
                    pcl.addWidget(hint_label(
                        "No main QR code yet. Add one while building a "
                        "scene.", C_FAINT))
                self.scan_btn_qr = styled_btn(pcard, "Scan QR Code  (F2)",
                                              self.qr_button_pressed,
                                              color="#2f9e44")
                pcl.addWidget(self.scan_btn_qr)
                if self._qr_sig("TRIGGER") and not locked:
                    self.btn_qr_now = styled_btn(pcard, "Scan Now",
                                                 self.qr_scan_now,
                                                 color="#44444e")
                    self.btn_qr_now.setToolTip("Read the QR now, without "
                                               "waiting for the machine")
                    pcl.addWidget(self.btn_qr_now)
            else:
                pcl.addWidget(styled_btn(pcard, "Scan Barcode  (F2)",
                                         self.open_quick_scan,
                                         color="#2f9e44"))
            pcl.addWidget(btn_row([
                styled_btn(pcard, "Clear", self.clear_barcode,
                           color="#44444e"),
                styled_btn(pcard, "Resend", self.resend_barcode,
                           color="#44444e")]))
            rlay.addWidget(pcard)
            QShortcut(QKeySequence("F2"), self,
                      activated=(self.qr_button_pressed
                                 if self.qr_scan_active
                                 else self.open_quick_scan))
        else:
            self.scan_status_lbl = None

        # -- MES part check
        if self.mes_active:
            mcard, mcl = card("MES part check", "mes")
            self.mes_status_lbl = QLabel("Waiting for the barcode")
            self.mes_status_lbl.setWordWrap(True)
            mcl.addWidget(self.mes_status_lbl)
            self.mes_msg_lbl = QLabel("—")
            self.mes_msg_lbl.setWordWrap(True)
            self.mes_msg_lbl.setStyleSheet(
                f"background: {C_FIELD}; color: {C_MUTED}; padding: 4px 8px;"
                " border-radius: 5px; font-size: 9pt;")
            self.mes_msg_lbl.setVisible(not locked)
            mcl.addWidget(self.mes_msg_lbl)
            mcl.addWidget(styled_btn(mcard, "Check Again",
                                     self.mes_recheck, color="#44444e"))
            nv = len(self.mes_cfg.get("validation_serials") or [])
            if nv and not locked:
                mcl.addWidget(hint_label(
                    f"{nv} validation serial(s) - never sent to MES.",
                    C_FAINT))
            rlay.addWidget(mcard)
        else:
            self.mes_status_lbl = None
            self.mes_msg_lbl = None

        # -- COUNTER
        ccard, ccl = card("Counter", "cnt")
        crow = QHBoxLayout()
        crow.setSpacing(8)
        self.cnt_ok_lbl = QLabel("OK  0")
        self.cnt_ok_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.cnt_ok_lbl.setStyleSheet(
            f"background: {C_OK}; color: white; font-weight: bold;"
            " font-size: 14pt; padding: 8px 10px; border-radius: 6px;")
        self.cnt_ng_lbl = QLabel("NG  0")
        self.cnt_ng_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.cnt_ng_lbl.setStyleSheet(
            f"background: {C_NG}; color: white; font-weight: bold;"
            " font-size: 14pt; padding: 8px 10px; border-radius: 6px;")
        crow.addWidget(self.cnt_ok_lbl, stretch=1)
        crow.addWidget(self.cnt_ng_lbl, stretch=1)
        ccl.addLayout(crow)
        self.btn_expand = styled_btn(ccard, "Show Details",
                                     self.toggle_counters, color="#44444e")
        if locked:
            ccl.addWidget(self.btn_expand)
        else:
            ccl.addWidget(btn_row([
                self.btn_expand,
                styled_btn(ccard, "Reset", self.reset_counters,
                           color="#e03131")]))
        self.counter_detail = QWidget()
        self.counter_detail.setObjectName("cdet")
        self.counter_detail.setStyleSheet(
            f"QWidget#cdet {{ background: {C_FIELD}; border-radius: 6px; }}"
            " QLabel { background: transparent; }")
        self.counter_detail_lay = QVBoxLayout(self.counter_detail)
        self.counter_detail_lay.setContentsMargins(4, 4, 4, 4)
        self.counter_detail_lay.setSpacing(2)
        self.counter_detail.setVisible(False)
        ccl.addWidget(self.counter_detail)
        self.counters_expanded = False
        rlay.addWidget(ccard)
        rlay.addStretch()

        self._log_lines = deque(maxlen=400)
        self._log_dialog = None

        sscroll = QScrollArea()
        sscroll.setWidget(side)
        sscroll.setWidgetResizable(True)
        sscroll.setFrameShape(QFrame.Shape.NoFrame)
        sscroll.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        sscroll.setStyleSheet("QScrollArea { background: transparent;"
                              " border: none; }")
        # sidebar width follows the screen, but never narrower than its
        # widest button / label needs
        sscroll.setFixedWidth(max(self._side_width(),
                                  side.minimumSizeHint().width() + 12))
        body.addWidget(sscroll)

        self.reload_models()
        self.refresh_counters()
        self._update_scan_panel()
        self.app.camera.start()
        self.auto_balance()
        if not locked:
            done, total, _ = self.comm_cfg.progress()
            self.log(f"Comm config: {done}/{total} "
                     f"({'complete OK' if done == total else 'incomplete'})")
        if self.qr_scan_active:
            self.log("Camera QR cycle: press SCAN QR CODE (F2) at the start of "
                     "every cycle (QR readers: "
                     + ", ".join(available_backends()) + ")")
            if not self.qr_master:
                self.log("Warning: camera QR is selected but the model has NO master "
                         "QR — build a scene and add one (5. QR Scanner)")
            elif self.comm_cfg.qr_required():
                _miss = [f for f in CommConfig.QR_FIELDS
                         if not self._qr_sig(f)]
                if _miss:
                    self.log("[QR] Comm Config -> CAMERA QR still empty: "
                             + ", ".join(_miss))
        elif self.scan_active:
            self.log("Scanner cycle: press SCAN (F2) and scan a barcode "
                     "to start")
            if self.scanner_cfg.get("auto_popup", True):
                QTimer.singleShot(500, self.open_quick_scan)
        elif self.scanner_cfg.get("enabled") and not self.finished:
            self.log("Scanner is configured but the model is not finished — "
                     "no scan is required yet. Press 'Finish Model' to "
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
                     "calls will be made until you press 'Finish Model'.")
        QTimer.singleShot(0, lambda: self.setFocus(
            Qt.FocusReason.OtherFocusReason))
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)
        self._timer.start(30)
        # header lamps + MES reachability
        self._chip_timer = QTimer(self)
        self._chip_timer.timeout.connect(self._refresh_chips)
        self._chip_timer.timeout.connect(self._refresh_ready)
        self._chip_timer.start(1000)
        QTimer.singleShot(50, self._refresh_chips)
        if self.mes_active:
            self._mes_timer = QTimer(self)
            self._mes_timer.timeout.connect(self._mes_probe)
            self._mes_timer.start(10000)
            QTimer.singleShot(700, self._mes_probe)

    # ------------------------------------------------------------------
    # header lamps / MES reachability
    # ------------------------------------------------------------------
    @staticmethod
    def _side_width():
        scr = QApplication.primaryScreen()
        w = scr.availableGeometry().width() if scr else 1400
        return max(330, min(420, int(w * 0.24)))

    def _refresh_chips(self):
        if not self._alive:
            return
        h = self.app._health
        c = self.chips
        c["camera"].set(not getattr(self.app.camera, "error", None))
        c["network"].set(bool(h.get("ethernet", True)))
        eng = self.app.engine
        c["machine"].set(None if not eng.enabled
                         else eng.client_count() > 0, eng.status_text())
        if "mes" in c:
            c["mes"].set(self._mes_online)

    def _mes_hostport(self):
        txt = str(self.mes_cfg.get("mes_ip", "") or "").strip()
        txt = re.sub(r"^\w+://", "", txt).split("/", 1)[0]
        if not txt:
            return None
        host, port = txt, 80
        if ":" in txt:
            host, p = txt.rsplit(":", 1)
            try:
                port = int(p)
            except ValueError:
                port = 80
        return host, port

    def _mes_probe(self):
        """Can the MES server be reached?  (TCP connect, off the UI thread)"""
        if not self._alive or not self.mes_active:
            return
        hp = self._mes_hostport()

        def run():
            ok = False
            if hp:
                try:
                    with socket.create_connection(hp, timeout=2.5):
                        ok = True
                except OSError:
                    ok = False
            self.after(0, lambda: self._set_mes_link(ok))
        threading.Thread(target=run, daemon=True).start()

    def _set_mes_link(self, ok, announce=True):
        if not self._alive:
            return
        prev = self._mes_online
        self._mes_online = ok
        if "mes" in self.chips:
            self.chips["mes"].set(ok)
        if prev is ok:
            return
        if not ok:
            self.log("[MES] NOT connected")
            self.app.comm_log.add("ERR", "MES server not reachable")
            if announce:
                Message.error(self, "MES NOT CONNECTED",
                              "Reason: network cable is out or the MES "
                              "server is off." if self._mes_hostport() else
                              "Reason: MES address is empty.")
        elif prev is False:
            self.log("[MES] connected again")
            self.app.comm_log.add("EVT", "MES server reachable again")
            if announce:
                Message.ok(self, "MES CONNECTED")

    # ------------------------------------------------------------------
    def destroy_screen(self):
        self._qr_stop.set()
        for pp in list(getattr(self, "_popups", {}).values()):
            try:
                pp.close()
            except Exception:
                pass
        if self.mes:
            m, self.mes = self.mes, None
            threading.Thread(target=m.logout, daemon=True).start()
        super().destroy_screen()

    def log(self, msg):
        if not self._alive:
            return
        self._log_lines.append(msg)
        if self._log_dialog is not None:
            self._log_dialog.append_line(msg)

    def open_log_dialog(self):
        if self._log_dialog is not None:
            self._log_dialog.raise_()
            self._log_dialog.activateWindow()
            return
        dlg = DetectionLogDialog(self, list(self._log_lines))
        self._log_dialog = dlg
        dlg.destroyed.connect(lambda *_: setattr(self, "_log_dialog", None))
        dlg.show()

    # ------------------------------------------------------------------
    # CYCLE RESULT (overall OK/NG of every scene) + popups
    # ------------------------------------------------------------------
    def _set_cycle_result(self, ok, sub=""):
        base = ("font-size: 16pt; font-weight: 700; border-radius: 8px;"
                " padding: 8px;")
        if ok is None:
            self.cycle_result_lbl.setText("WHOLE PART:  " + (sub or "-"))
            self.cycle_result_lbl.setStyleSheet(
                f"background: {C_ELEV}; color: {C_FAINT}; {base}")
        else:
            self.cycle_result_lbl.setText(
                f"WHOLE PART:  {'OK' if ok else 'NG'}"
                + (f"\n{sub}" if sub else ""))
            self.cycle_result_lbl.setStyleSheet(
                f"background: {C_OK if ok else C_NG};"
                f" color: white; {base}")

    def _show_popup(self, key, title, headline, lines, color, auto_ms=0):
        old = self._popups.pop(key, None)
        if old is not None:
            old.close()
        pp = ResultPopup(self, title, headline, lines, color, auto_ms)
        pp.show()
        pp.raise_()
        self._popups[key] = pp

    def _popup_cycle_result(self, cycle_ok, scene_results, serial,
                            validation):
        cp = self.app.app_config.data.get("cycle_popup", {})
        if not cp.get("enabled", True):
            return
        lines = [f"Serial: {serial}" if serial else "",
                 "Validation part - not sent to MES" if validation
                 else ("" if cycle_ok else "Put it in the reject bin.")]
        lines = [x for x in lines if x]
        auto = int(float(cp.get("auto_close_s", 5)) * 1000)
        self._show_popup("cycle", "Part finished",
                         f"PART {'OK' if cycle_ok else 'NG'}", lines,
                         C_OK if cycle_ok else C_NG, auto)

    # ------------------------------------------------------------------
    # VALIDATION PART  (MES disabled for this unit)
    # ------------------------------------------------------------------
    def _is_validation_serial(self, code):
        """MES in use and this serial is in the model's validation list."""
        if not self.mes_active:
            return False
        lst = [str(x).strip() for x in
               (self.mes_cfg.get("validation_serials") or [])]
        return str(code).strip() in lst

    def _validation_notice(self, why):
        self._show_popup(
            "validation", "Validation part",
            "VALIDATION PART",
            [f"Serial: {self.scan_value}", "Not sent to MES."],
            "#2563c9", 4500)

    def _not_ready_reason(self):
        """What is stopping the station being READY, or None.  READY only
        shows once camera, MES (when used) and the machine are all
        connected — otherwise this says which one is not."""
        if getattr(self.app.camera, "error", None):
            return "CAMERA NOT CONNECTED", "Check the camera cable"
        if not self.app._health.get("ethernet", True):
            return "NETWORK NOT CONNECTED", "Check the network cable"
        if self.mes_active and self._mes_online is not True:
            return "MES NOT CONNECTED", "Check the MES server"
        if self.app.engine.enabled and self.app.engine.client_count() == 0:
            return "MACHINE NOT CONNECTED", "Waiting for the machine"
        return None

    def set_big_status(self, ok=None, sub=""):
        short_base = ("font-size: 40pt; font-weight: 800;"
                      " border-radius: 10px; padding: 4px;"
                      " letter-spacing: 2px;")
        long_base = ("font-size: 17pt; font-weight: 800;"
                     " border-radius: 10px; padding: 10px 8px;"
                     " letter-spacing: 0.5px;")
        base = short_base
        self._last_ok = ok
        if ok is None:
            reason = self._not_ready_reason()
            if reason:
                base = long_base
                self.status_big.setText(reason[0])
                self.status_big.setStyleSheet(
                    f"background: {C_WARN}; color: white; {base}")
                sub = sub or reason[1]
            else:
                self.status_big.setText("READY")
                self.status_big.setStyleSheet(
                    f"background: {C_ELEV}; color: {C_FAINT}; {base}")
        elif ok:
            self.status_big.setText("OK")
            self.status_big.setStyleSheet(
                f"background: {C_OK}; color: white; {base}")
        else:
            self.status_big.setText("NG")
            self.status_big.setStyleSheet(
                f"background: {C_NG}; color: white; {base}")
        self.status_sub.setText(
            sub or ("Nothing tested yet" if ok is None else ""))

    def _refresh_ready(self):
        """Keep the idle READY / NOT-CONNECTED text live while nothing has
        been tested yet.  Never touches a real OK/NG result on screen."""
        if getattr(self, "_last_ok", None) is None and self._alive:
            self.set_big_status(None)

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
                                      "  " + a.split("/", 1)[1],
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
            self.btn_expand.setText("Hide Details")
        else:
            self.counter_detail.setVisible(False)
            self.btn_expand.setText("Show Details")
        self.refresh_counters()

    def reset_counters(self):
        if Message.confirm(self, "Set all counters back to zero?",
                           "The OK and NG counts of this model will start "
                           "again from zero.",
                           yes="Reset", no="Cancel"):
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

    def reset_camera(self):
        """Reset the camera to its own default settings (USB camera: the raw
        feed, nothing applied; resizing is the only processing)."""
        if not self.app.camera._cams:
            self.app.alert("camera", "CAMERA NOT CONNECTED",
                           "Reason: camera cable is out or the camera has no power.",
                           retry=self.app.camera.start)
            return
        self.log("[camera] resetting camera settings…")

        def done(ok, msg):
            def ui():
                if not self._alive:
                    return
                self.log(f"[camera] {'OK' if ok else 'X'} {msg}")
                self.app.comm_log.add("EVT" if ok else "ERR",
                                      f"camera reset: {msg}")
                if ok:
                    Message.ok(self, "Camera settings reset",
                               "The camera is back to its own settings.")
                else:
                    Message.warn(self, "Camera not reset", str(msg))
            self.app.after(0, ui)
        self.app.camera.reset_camera(done)

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
                    self.log("[scanner] already scanned this cycle - press "
                             "Clear (or finish the cycle) to scan again")
                    Message.info(self, "ALREADY SCANNED",
                             f"Serial: {self.scan_value}")
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
        elif QuickScanInput.last_unlocked:
            self.log("[scanner] unlock code used - screen is free to use")
        else:
            self.log("[scanner] scan cancelled - detection still blocked")
            Message.warn(self, "NO BARCODE SCANNED",
                     "Scan the part first.")

    # ------------------------------------------------------------------
    # MES  (previous stage check on scan, report + upload at cycle end)
    # ------------------------------------------------------------------
    def open_mes_cfg(self):
        if not self.app.heads.is_finished(self.head):
            Message.info(
                self, "Finish the model first",
                "MES details can be configured once the model is "
                "finalised.\n\nBuild the scenes, then press "
                "'Finish Model'.")
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
            "idle":     ("Waiting for the barcode", C_MUTED),
            "checking": ("Checking this part...", C_WARN),
            "pass":     ("Previous station passed", C_OK),
            "all_passed": ("Already passed by this station", C_OK),
            "fail":     ("Previous station NOT passed", C_NG),
            "error":    ("MES not connected", C_NG),
            "sending":  ("Sending the result to MES...", C_WARN),
            "sent":     ("Result sent to MES", C_OK),
            "senderr":  ("Result NOT sent to MES", C_NG),
            "validation": ("Validation part - not sent to MES", C_ACCENT),
        }.get(state, (state, C_MUTED))
        self.mes_status_lbl.setText(text)
        self.mes_status_lbl.setStyleSheet(f"color: {color};"
                                          " font-weight: bold;"
                                          " font-size: 11pt;")
        if self.mes_msg_lbl:
            self.mes_msg_lbl.setText(msg or "—")

    @staticmethod
    def _mes_link_problem(msg):
        """Does the MES error say the server could not be reached?"""
        m = str(msg).lower()
        return any(w in m for w in (
            "connection", "timed out", "timeout", "refused", "unreachable",
            "max retries", "login failed", "no route", "name or service",
            "not available", "failed to establish"))

    @staticmethod
    def _serial_problem(msg):
        """Does the MES answer say the barcode itself is wrong?"""
        m = str(msg).lower()
        return any(w in m for w in ("not found", "no record", "invalid",
                                    "does not exist", "doesn't exist",
                                    "not exist", "unknown serial",
                                    "wrong serial", "no data found"))

    def mes_check_previous(self, serial):
        """Scan -> ask MES whether this part passed the previous station."""
        if not self.mes or not self.mes_active:
            return
        self._set_mes_status("checking", f"serial '{serial}'...")
        self.log(f"[MES] previous stage check for '{serial}'...")

        def done(res):
            if not self._alive:
                return
            state, msg = res
            msg = str(msg)

            if state in ("pass", "all_passed"):
                self._set_mes_status(state, msg)
                if state == "pass":
                    self.log(f"[MES] OK previous stage passed - {msg}")
                    self.app.comm_log.add("EVT", f"MES prev-stage PASS "
                                                 f"'{serial}': {msg}")
                    Message.ok(self, "PART OK TO TEST",
                               f"Serial: {serial}")
                else:
                    self.log(f"[MES] OK all stations already passed - {msg}")
                    self.app.comm_log.add("EVT", f"MES prev-stage ALL-PASSED "
                                                 f"'{serial}': {msg}")
                    Message.ok(self, "ALREADY PASSED",
                               f"Serial: {serial}\n"
                               "Already passed by this station.")
                if self.scanner_cfg.get("send_to_client", True):
                    self._try_send_barcode()
                self._update_scan_panel()
                return

            # ---- not allowed / could not be checked
            bad_serial = (state in ("error", "fail")
                          and self._serial_problem(msg))
            no_link = (state == "error" and not bad_serial
                       and self._mes_link_problem(msg))
            if no_link:
                self._set_mes_link(False, announce=False)
            self._set_mes_status("error" if state == "error" else "fail", msg)
            self.log(f"[MES] FAILED {msg}")
            self.app.comm_log.add("EVT", f"MES prev-stage "
                                         f"{state.upper()} '{serial}': {msg}")

            if state == "fail" and self.scanner_cfg.get("send_to_client",
                                                        False):
                self._try_send_barcode(st="ng")

            blocked = (self.app.is_locked()
                       and self.mes_cfg.get("block_on_prev_fail", True))
            if bad_serial:
                Message.error(self, "WRONG SERIAL NUMBER",
                              f"Serial: {serial}\n"
                              "This serial number is not in MES.")
            elif state == "fail":
                Message.error(self, "PREVIOUS STATION NOT PASSED",
                              f"Serial: {serial}\n"
                              "Take this part off the station.")
            elif no_link:
                Message.error(self, "MES NOT CONNECTED",
                              "Reason: network cable is out or the MES "
                              "server is off.")
            else:
                Message.warn(self, "MES CHECK FAILED",
                             f"Serial: {serial}\n"
                             "Reason: " + _first_line(msg))

            if blocked:
                self.log("[scanner] cycle aborted - scan the next barcode")
                self._start_next_cycle()
            else:
                self.log("[MES] controls open - run allowed, but this part "
                         "did NOT pass the previous station")
        self.mes.check_previous_async(serial, done)

    def mes_recheck(self):
        if self.validation_mode:
            self.log("[MES] validation part — MES is disabled for it")
            return
        if not self.scan_value:
            self.log("[MES] nothing to check - scan a barcode first")
            Message.info(self, "NOTHING TO CHECK",
                         "Scan the barcode on the part first.")
            return
        self.mes_check_previous(self.scan_value)

    def mes_gate(self):
        """(allowed, reason) for the MES previous-stage result."""
        if not self.mes or not self.mes_active:
            return True, ""
        if self.mes_busy:
            return False, "MES is still submitting the previous cycle — wait"
        if self.validation_mode and self.scan_value:
            return True, ""          # validation part: MES not involved
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
        if not self.mes_active or self.validation_mode:
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
                self.log(f"[MES] OK stage report accepted - {msg}")
                self.app.comm_log.add("EVT", f"MES stage report '{serial}' "
                                             f"OK: {msg}")
                Message.ok(self, "SENT TO MES",
                           f"Serial: {serial}")
            else:
                self._set_mes_status("senderr", str(msg))
                self.log(f"[MES] FAILED stage report - {msg}")
                self.app.comm_log.add("ERR", f"MES stage report '{serial}' "
                                             f"FAILED: {msg}")
                Message.error(self, "NOT SENT TO MES",
                              f"Serial: {serial}\n"
                              "Reason: MES did not answer. Saved on this PC.")
            if path:
                self.log(f"[MES] report: {path}")
            # only now is the unit finished -> ready for the next one
            self._start_next_cycle()

        self.mes.submit_cycle_async(bundle, done)
        return True

    # ------------------------------------------------------------------
    # CAMERA QR  (part read by camera; Operator ID still uses the scanner)
    #
    #   operator presses SCAN QR CODE
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
        """Big popup. auto_ms > 0 closes itself (a retry notice must never
        stop the retry loop)."""
        if auto_ms:
            self._qr_close_toast()
        box = (Message.error if critical else Message.warn)(
            self, title, text, auto_ms=auto_ms)
        if auto_ms:
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
                Message.warn(
                    self, "No master QR",
                    "Camera QR scanning is selected, but this model has no "
                    "master QR.\n\nBuild a scene and use  5. QR Scanner -> "
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
            return (f"Attempt {n}/{mx}: waiting for client signal "
                    f"'{self._qr_sig('TRIGGER')}'…  (press again to cancel)",
                    "#74c0fc")
        if self.qr_state == "scanning":
            return f"Attempt {n}/{mx}: reading QR…", "#74c0fc"
        if self.qr_state == "retrying":
            return (f"Not read (attempt {n}/{mx}) — sending the signal "
                    "again…  (press to cancel)", "#ffa94d")
        if self.qr_msg:
            return str(self.qr_msg), "#ff8787"
        if not self.qr_master:
            return "No main QR code set up", "#ff8787"
        return ("Press SCAN QR CODE (F2) to start this part", "#ffd43b")

    def _qr_sync_buttons(self):
        b = self.scan_btn_qr
        if b is None:
            return
        if self.qr_state in ("waiting", "retrying"):
            b.setText("Cancel")
            apply_btn_color(b, "#e03131")
            b.setEnabled(True)
        elif self.qr_state == "scanning":
            b.setText("reading...")
            apply_btn_color(b, "#3b6ef5")
            b.setEnabled(False)
        else:
            b.setText("SCAN QR CODE  (F2)")
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
                self.log(f"[QR] X {why} — scan NOT started (locked mode)")
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
            self.log(f"[QR] Warning: {why} — reading the camera anyway "
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
            self.log("[QR] no request signal set (Comm Config -> CAMERA QR)")
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
            self.log(f"[QR] OK '{res.value}'  (attempt {self.qr_attempt}, "
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
            self.log(f"[QR] X attempt {n}/{mx}: {reason} — trying again, "
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
        self.log(f"[QR] XX not read after {mx} attempts — {reason}")
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
        self._set_validation(False)
        self._set_mes_status("idle", "")
        self._mes_reset_cycle_data()
        if self.qr_scan_active:
            self.log("[QR] ready — press SCAN QR CODE (F2) for the next part")
        elif self.scan_active:
            self.log("[scanner] ready — scan the next barcode")
            if self.scanner_cfg.get("auto_popup", True):
                QTimer.singleShot(300, self.open_quick_scan)

    def _set_validation(self, on):
        self.validation_mode = bool(on)
        self.val_banner.setVisible(self.validation_mode)

    def _handle_barcode(self, code, validation=False):
        code = str(code or "").strip()
        if len(code) < 2:
            self.log(f"[scanner] barcode '{code}' is not valid - ignored")
            Message.error(self, "BARCODE NOT VALID",
                          "That barcode could not be read properly.\n\n"
                          "Please scan it again.")
            return
        # a new serial = a new unit: nothing of the previous one may reach MES
        self._mes_reset_cycle_data()
        self.cycle_scene_results = {}
        validation = validation or self._is_validation_serial(code)
        self._set_validation(validation)
        self._set_cycle_result(None, "cycle in progress…")
        self.scan_value = code
        self.scan_time = datetime.datetime.now().strftime("%H:%M:%S")
        self.scan_sent = False
        op_info = f" (Op: {self.operator_id})" if self.operator_id else ""
        self.app.comm_log.add("EVT", f"barcode scanned: '{code}'{op_info}")
        self.log(f"[scanner] scanned '{code}'{op_info}")
        
        self._update_scan_panel()
        if validation:
            self.app.comm_log.add("EVT", f"VALIDATION part '{code}' — MES "
                                         "disabled, nothing will be sent")
            self.log(f"[validation] '{code}' is a VALIDATION part — MES check "
                     "skipped, nothing will be sent to MES")
            self._set_mes_status("validation", f"serial '{code}' — NOT sent "
                                               "to MES")
            self._validation_notice("Serial found in the validation list.\n"
                                    "Run it as usual.")
            if self.scanner_cfg.get("send_to_client", True):
                self._try_send_barcode()
                self._update_scan_panel()
            return
        self.mes_check_previous(code)
        # if self.scanner_cfg.get("send_to_client", True):
        #         if self.mes_prev_state in ("pass", "all_passed"):
        #             self._try_send_barcode()

    def _try_send_barcode(self,st=None):
        if not self.scan_value:
            return
        if st is None:
            signal = str(self.scanner_cfg.get("prefix", "")) + self.scan_value
        else: 
            signal = "ng" + self.scan_value


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
        self._set_validation(False)
        self._set_mes_status("idle", "")
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
            self.scan_status_lbl.setText("Waiting for the barcode")
            self.scan_status_lbl.setStyleSheet(f"color: {C_WARN};"
                                               " font-weight: bold;")
            self.scan_value_lbl.setText("—")
            return
        self.scan_value_lbl.setText(
            f"{self.scan_value}   ({self.scan_time})")
        if self.scanner_cfg.get("send_to_client", True):
            if self.scan_sent:
                self.scan_status_lbl.setText(
                    "Scanned")
                self.scan_status_lbl.setStyleSheet(f"color: {C_OK};"
                                                   " font-weight: bold;")
            else:
                self.scan_status_lbl.setText(
                    "Scanned - not sent to the machine")
                self.scan_status_lbl.setStyleSheet(f"color: {C_OK};"
                                                   " font-weight: bold;")
        else:
            self.scan_status_lbl.setText("Scanned - you can test the part")
            self.scan_status_lbl.setStyleSheet(f"color: {C_OK};"
                                               " font-weight: bold;")

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
                return False, ("scan the QR first — press  SCAN QR (F2); "
                               "the scan step runs before the scenes")
            return False, ("scan the barcode first — press SCAN (F2); "
                           "the scanner step runs before the scenes")
        return self.mes_gate()

    def _end_cycle(self, all_ok):
        """Called when EVERY scene of the model has been run once."""
        scenes = dict(self.cycle_scene_results)
        self.cycle_scene_results = {}
        serial = self.scan_value
        validation = self.validation_mode
        # overall CYCLE result: on the UI and in a popup
        self._set_cycle_result(all_ok, ("validation part - not sent to MES"
                                        if validation else ""))
        self._popup_cycle_result(all_ok, scenes, serial, validation)
        if validation:
            self.log(f"[validation] cycle {'OK' if all_ok else 'NG'} for "
                     f"'{serial}' — NOT sent to MES")
            self.app.comm_log.add("EVT", f"VALIDATION cycle '{serial}' "
                                         f"{'OK' if all_ok else 'NG'} — not "
                                         "sent to MES")
            self._start_next_cycle()
            return
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
            e = QLabel("No scenes yet.\nPress Add Scene")
            e.setStyleSheet("color: #666; font-size: 10pt;")
            e.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self.parent_btn_lay.addWidget(e)
        for p in parents:
            if self.parent_btn_lay is not None:
                n = len(self.parent_store.list_children(p))
                self.parent_btn_lay.addWidget(styled_btn(
                    self.parent_btn_frame, f"{p}   ({n} area)",
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
            Message.info(self, "No master",
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
                        "#0ca678" if mode == "side" else "#44444e")
        apply_btn_color(self.btn_overlay,
                        "#0ca678" if mode == "overlay" else "#44444e")

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
        name = ask_text(self, "Name the new scene",
                        "A scene is one view of the part. Give it a name "
                        "you will recognise.",
                        placeholder="for example:  top view")
        if not name or not name.strip():
            return
        name = name.strip()
        if self.app.heads.scene_exists(self.head, name):
            Message.info(self, "Exists",
                                    f"Scene '{name}' already exists.")
            return
        self.app.show_build_parent(self.head, name)

    def finish_head(self):
        structure = self.app.heads.structure(self.head)
        if not any(children for children in structure.values()):
            Message.warn(
                self, "Detection not finished",
                "The model needs at least one scene with at least one area "
                "before it can be finished.")
            return
        if Message.confirm(
                self, "Finish this model?",
                "The model is then ready for production.\n\nThe barcode "
                "scanner, MES and the machine communication can be set up "
                "after this step.",
                yes="Finish", no="Not yet"):
            self.app.heads.set_finished(self.head, True)
            self.app.show_head(self.head, announce=False)

    def open_comm(self):
        if not self.app.heads.is_finished(self.head):
            Message.info(
                self, "Finish detection first",
                "First finish the detection part completely and press "
                "'Finish Model'. Then the Communication builder unlocks.")
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
            self.app.comm_log.add("EVT", f"run {scope} BLOCKED - {reason}")
            if (source == "MANUAL" and not self.scan_value
                    and self.scan_active and not self.qr_scan_active):
                # only a missing scan - open the scan box and go again
                self.open_quick_scan()
                if self.scan_value and not self.mes_active:
                    self.execute(scope, source)
                return
            self._blocked_popup(reason)
            return
        eth_ok, cam_ok = self.app.hardware_ok(self)   # popup if broken
        if not eth_ok:
            self.log("Warning: the network cable is not connected")
        frame = self.app.camera.get_frame()
        if frame is None:
            self.log(f"No camera frame — cannot run {scope} ({source})")
            if cam_ok:      # camera not reported broken, just no frame yet
                self.app.alert("camera", "CAMERA NOT CONNECTED",
                               "Reason: camera cable is out or the camera has no power.",
                               retry=self.app.camera.start)
            return
        if self.validation_mode:
            self._validation_notice("This run is NOT sent to MES.")
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

    def _blocked_popup(self, reason):
        """Why nothing ran, in words an operator can act on."""
        if not self.scan_value and self.scan_active:
            if self.qr_scan_active:
                Message.warn(self, "READ THE QR CODE FIRST",
                             "Press SCAN QR CODE before testing the part.")
            else:
                Message.warn(self, "SCAN THE BARCODE FIRST",
                             "Scan the barcode on the part before testing "
                             "it.")
            return
        if self.mes_busy:
            Message.info(self, "PLEASE WAIT",
                         "The result of the last part is still being sent "
                         "to MES.", auto_ms=3000)
            return
        if self.mes_prev_state == "checking":
            Message.info(self, "PLEASE WAIT",
                         "This part is still being checked.", auto_ms=3000)
            return
        if self.mes_prev_state == "fail":
            Message.error(self, "PREVIOUS STATION NOT PASSED",
                          "Take this part off the station.")
            return
        if self.mes_prev_state == "error":
            Message.error(self, "MES NOT CONNECTED",
                          "Reason: network cable is out or the MES server "
                          "is off.")
            return
        Message.warn(self, "CANNOT TEST THIS PART", str(reason))

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
                    self.back_to_live()
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
        top.addWidget(styled_btn(self, "Back to Model",
                                 lambda: app.show_head(head_name,
                                                       announce=False),
                                 color="#44444e"))
        t = QLabel(f"COMMUNICATION —  {head_name}  (software / TCP)")
        t.setStyleSheet("color: white; font-size: 14pt; font-weight: bold;")
        top.addWidget(t)
        top.addStretch()
        top.addWidget(styled_btn(self, "Connection Monitor",
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
        lt = QLabel("STRUCTURE - done / missing / optional")
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
            return f"{'OK' if done else 'X'}  {name}" + \
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
            + ("   —  COMPLETE OK" if dn == total else ""))
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
        l.setMinimumWidth(160)
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
            sl.setMinimumWidth(160)
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
                "sent when the operator presses SCAN QR CODE -\n"
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
            note("Each cycle the operator presses SCAN QR CODE on the model\n"
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
                note("Scene-only mode: area-wise results are "
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
        btnrow.addWidget(styled_btn(self.form_holder, "Save",
                                    self.save_current, color="#2f9e44"))
        btnrow.addWidget(styled_btn(self.form_holder, "Clear fields",
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
            Message.warn(self, "Duplicate signal",
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
        t = QLabel("QR CODE SETTINGS")
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
        row.addWidget(styled_btn(self, "Next: select the region", self._ok,
                                 color="#2f9e44"))
        row.addWidget(styled_btn(self, "Cancel", self.reject,
                                 color="#44444e"))
        row.addStretch()
        lay.addLayout(row)

    def _ok(self):
        name = self.e_name.text().strip()
        if not name or "/" in name or "\\" in name:
            Message.warn(self, "Name", "Enter a name (no / or \\).")
            return
        if name.lower() in self.taken:
            Message.warn(self, "Name",
                                f"'{name}' is already used in this scene.")
            return
        ltxt = self.e_len.text().strip()
        if ltxt and (not ltxt.isdigit() or int(ltxt) <= 0):
            Message.warn(self, "Length",
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
        top.setSpacing(10)
        top.addWidget(btn_group([styled_btn(self, "Cancel", self.cancel,
                                            color="#44444e")]))
        top.addWidget(screen_title(f"{head_name}  /  {parent_name}"))
        top.addStretch()
        root.addLayout(top)

        # what to do next, always on the same line under the title
        self.status_lbl = QLabel("")
        self.status_lbl.setWordWrap(True)
        self.status_lbl.setStyleSheet(
            f"background: {C_PANEL}; border: 1px solid {C_LINE};"
            " border-radius: 8px; padding: 10px 14px;"
            f" color: {C_TEXT}; font-size: 11pt; font-weight: 600;")
        root.addWidget(self.status_lbl)

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
        right.setFixedWidth(310)
        rlay = QVBoxLayout(right)
        rlay.setContentsMargins(10, 10, 10, 10)
        rlay.setSpacing(8)

        sec = QGroupBox(" STEP 1   AREA TO LOOK AT ")
        s1 = QVBoxLayout(sec)
        self.btn_sel_roi = styled_btn(sec, "Select ROI (drag on video)",
                                      self.start_roi)
        s1.addWidget(self.btn_sel_roi)
        self.btn_confirm_roi = styled_btn(sec, "Confirm Area",
                                          self.confirm_roi, color="#2f9e44")
        self.btn_reset_roi = styled_btn(sec, "Clear Area", self.reset_roi,
                                        color="#e03131")
        s1.addWidget(btn_row([self.btn_confirm_roi, self.btn_reset_roi]))
        rlay.addWidget(sec)

        sec2 = QGroupBox(" STEP 2   WHAT TO FIND ")
        s2 = QVBoxLayout(sec2)
        self.btn_add_tmpl = styled_btn(sec2, "Select Template (drag)",
                                       self.start_template)
        s2.addWidget(self.btn_add_tmpl)
        self.btn_confirm_tmpl = styled_btn(sec2, "Confirm",
                                           self.confirm_template,
                                           color="#2f9e44")
        self.btn_reset_tmpl = styled_btn(sec2, "Clear",
                                         self.reset_template,
                                         color="#e03131")
        s2.addWidget(btn_row([self.btn_confirm_tmpl, self.btn_reset_tmpl]))
        self.tmpl_count_lbl = QLabel("Templates: 0")
        self.tmpl_count_lbl.setStyleSheet("color: #c8c8d0; font-size: 9pt;")
        self.tmpl_count_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        s2.addWidget(self.tmpl_count_lbl)
        rlay.addWidget(sec2)

        sec3 = QGroupBox(" STEP 3   FINE TUNING ")
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
        self._box_roi, self._box_tmpl, self._box_tune = sec, sec2, sec3

        sec4 = QGroupBox(" STEP 4   CHECK AND SAVE ")
        s4 = QVBoxLayout(sec4)
        self.btn_test = styled_btn(sec4, "Test Detect", self.test_detect,
                                   color="#f59f00")
        s4.addWidget(self.btn_test)
        self.btn_back_live = styled_btn(sec4, "Back to Live",
                                        self.back_to_live, color="#44444e")
        s4.addWidget(self.btn_back_live)
        self.btn_save = styled_btn(sec4, "Save Area", self.save_child,
                                   color="#2f9e44")
        s4.addWidget(self.btn_save)
        self.btn_next = styled_btn(sec4, "Next Area", self.next_child,
                                   color="#44444e")
        s4.addWidget(self.btn_next)
        self.btn_finish = styled_btn(sec4, "Finish Scene",
                                     self.finish_parent, color="#f59f00")
        s4.addWidget(self.btn_finish)
        s4.addWidget(styled_btn(sec4, "Auto Balance", self.auto_balance,
                                color="#44444e"))
        rlay.addWidget(sec4)

        if self.qr_camera:
            sec5 = QGroupBox(" STEP 5   QR CODE (OPTIONAL) ")
            s5 = QVBoxLayout(sec5)
            self.qr_list_lbl = QLabel("")
            self.qr_list_lbl.setWordWrap(True)
            self.qr_list_lbl.setStyleSheet("color: #c8c8d0; font-size: 8pt;")
            s5.addWidget(self.qr_list_lbl)
            self.btn_qr_add = styled_btn(sec5, "Add QR Code",
                                         self.qr_add_start, color="#44444e")
            self.btn_qr_test = styled_btn(sec5, "Test", self.qr_test_read,
                                          color="#44444e")
            self.btn_qr_del = styled_btn(sec5, "Remove", self.qr_remove,
                                         color="#e03131")
            s5.addWidget(btn_row([self.btn_qr_add, self.btn_qr_test,
                                  self.btn_qr_del]))
            r5b = QHBoxLayout()
            self.btn_qr_region = styled_btn(sec5, "Use this region",
                                            self.qr_confirm_region,
                                            color="#2f9e44")
            self.btn_qr_full = styled_btn(sec5, "Use whole picture",
                                          self.qr_use_full_frame,
                                          color="#44444e")
            self.btn_qr_cancel = styled_btn(sec5, "Cancel", self.qr_cancel,
                                            color="#44444e")
            for b in (self.btn_qr_region, self.btn_qr_full,
                      self.btn_qr_cancel):
                r5b.addWidget(b)
            s5.addLayout(r5b)
            rlay.addWidget(sec5)
            self._qr_refresh_list()
        self._box_actions = sec4

        rlay.addWidget(section_title("Messages"))
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
        # a button that cannot be used yet is hidden, so the operator always
        # sees exactly one next step
        for b in (self.btn_confirm_roi, self.btn_reset_roi,
                  self.btn_confirm_tmpl, self.btn_reset_tmpl,
                  self.btn_test, self.btn_save, self.btn_back_live):
            b.setVisible(b.isEnabled())
        # a whole step only appears once the step before it is done
        if not self.master_scene:
            has_roi = self.reg is not None
            has_tmpl = has_roi and bool(self.template_list)
            self._box_tmpl.setVisible(has_roi)
            self._box_tune.setVisible(has_tmpl)
            self._box_actions.setVisible(has_tmpl)
        if self.qr_camera:
            qr_mode = self.mode == self.MODE_QR and self.qr_pending is not None
            self.btn_qr_add.setEnabled(not qr_mode and not self.busy)
            self.btn_qr_region.setEnabled(qr_mode and pend)
            self.btn_qr_full.setEnabled(qr_mode)
            self.btn_qr_cancel.setEnabled(qr_mode)
            self.btn_qr_test.setEnabled(
                (qr_mode or bool(self._qr_items)) and not self._qr_testing)
            self.btn_qr_del.setEnabled(bool(self._qr_items) and not qr_mode)
            for b in (self.btn_qr_region, self.btn_qr_full,
                      self.btn_qr_cancel, self.btn_qr_test,
                      self.btn_qr_del):
                b.setVisible(b.isEnabled())
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
        self._set_status("ROI confirmed OK — now add template(s)")
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
            Message.error(self, "No camera",
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
            Message.error(self, "Bad template",
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
        self._set_status("Template added OK — add more, or Test Detect")
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
            Message.warn(self, "Not ready",
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
        self._set_status(f"Area '{self.current_child_name}' saved OK — "
                         "Next Area or Finish Scene")

    def next_child(self):
        if not self.current_child_saved:
            if not Message.confirm(
                    self, "This area is not saved",
                    "Start a new area and throw away what you just did?",
                    yes="Throw away", no="Go back"):
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
            Message.info(
                self, "Scene finished",
                f"Master QR scene '{self.parent_name}' saved.\n\nIt has no "
                "areas — the camera reads its QR when the operator presses "
                "SCAN QR.")
            self.app.show_head(self.head, announce=False)
            return
        if not children:
            Message.warn(self, "No areas",
                                "No areas saved yet — save at least one "
                                "first.")
            return
        self.app.heads.add_parent(self.head, self.parent_name)
        self.log(f"Scene '{self.parent_name}' finished with areas: "
                 f"{children}")
        Message.info(self, "Scene finished",
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
                             "areas.  Press Finish Scene to save it.")

    def _qr_refresh_list(self):
        self._qr_items = self.app.heads.qr_scanners_for_scene(
            self.head, self.parent_name)
        self.master_scene = any(q["master"] for q in self._qr_items)
        if self.qr_list_lbl is None:
            return
        if self._qr_items:
            lines = [f"{'*' if q['master'] else '-'} {q['name']}"
                     f"{' (MASTER)' if q['master'] else ''} — "
                     f"{'region' if q.get('roi_norm') else 'full frame'}, "
                     f"{QrRules.from_dict(q).describe()}"
                     for q in self._qr_items]
        else:
            lines = ["no QR scanner in this scene yet"]
        m = self.app.heads.get_master_qr(self.head,
                                         include_scene=self.parent_name)
        lines.append(f"master: '{m['name']}' (scene {m['scene']})" if m
                     else "This model has no main QR code yet")
        self.qr_list_lbl.setText("\n".join(lines))

    def qr_add_start(self):
        if self.master_scene:
            Message.info(
                self, "Master QR scene",
                "This scene holds the master QR — a master scene contains "
                "only that QR (no areas, no other QR).\n\nRemove it first "
                "(Remove) to change it, or create another scene.")
            return
        if self.mode not in (self.MODE_LIVE, self.MODE_RESULT):
            Message.info(
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
        self._set_status("Drag a rectangle around the QR code, then press Use this region "
                         "(or Use whole picture to search everywhere)")
        self._update_buttons()

    def qr_confirm_region(self):
        if (self.mode != self.MODE_QR or not self.qr_pending
                or self.canvas.pending_rect is None):
            return
        if not self._live_wh:
            Message.error(self, "No camera",
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
            Message.info(
                self, "Not a master scene",
                f"'{cfg['name']}' is saved as a normal QR of this scene.\n\n"
                "A MASTER QR needs a scene of its own: no areas and no other "
                "QR.  To make a master, press Add Scene on the model page "
                "and add the QR there.")
        elif existing is None:
            master = Message.confirm(
                self, "Is this the main QR code?",
                f"The main QR code is the one the camera reads at the start "
                f"of every part.\n\nA model can have only one main QR code, "
                f"and its scene then holds nothing else.\n\nChoose No to "
                f"use '{cfg['name']}' as an ordinary QR check in this scene.",
                yes="Yes, main QR", no="No, ordinary QR")
        else:
            master = Message.confirm(
                self, "This model already has a main QR code",
                f"The main QR code is now '{existing['name']}', in the scene "
                f"'{existing['scene']}'. Only one is allowed.\n\nUse "
                f"'{cfg['name']}' as the main QR code instead? The old one "
                f"is removed.",
                yes="Use the new one", no="Keep the old one")
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
        self._set_status(f"QR scanner '{item['name']}' saved OK — use Test "
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
            Message.warn(
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
                        self.log(f"QR '{label}': OK '{res.value}'  "
                                 f"({res.frames} frames · {res.elapsed:.2f}s "
                                 f"· {res.backend})")
                    else:
                        self.log(f"QR '{label}': X {res.reason}")
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