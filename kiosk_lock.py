# coding=utf-8
"""
KIOSK INPUT LOCK  -  while the program is LOCKED the keyboard and the mouse
can only be used inside the program.  The window keeps its size and place
(it is NOT made full screen).

Layers (each one works on its own, together they close the usual ways out):
  1. the MOUSE POINTER is kept inside the program window (and inside the
     program's own popups): when it leaves, it is put back at once - so
     nothing outside can be clicked, not even the desktop top bar / dock,
     which are drawn above every window.  Also, everything OUTSIDE the
     program window is covered by dark "locked" covers (the rest of this monitor, the desktop top bar, every other
     monitor), so a mouse click outside the program does nothing.  The
     window loses its minimise / maximise / close buttons, stays on top and
     is put back where it was if it is minimised, moved or resized
  2. the keyboard is grabbed (X11): Alt+Tab, the Windows/Super key, Alt+F4,
     Alt+F7..F10 (move / resize / minimise / maximise), Ctrl+Alt+... never
     reach the desktop.  Normal keys (and the barcode scanner, which is a
     keyboard too) are passed on to the program
  3. GNOME desktop shortcuts and the hot corner are switched off while
     locked and restored afterwards (a backup file restores them even after
     a crash, at the next start).  This also works on Wayland, where layer 2
     is not possible.

Master password (default 8252, stored only as a SHA-256 hash):
  frees the keyboard and mouse and turns on MASTER mode (hidden features),
  but the program itself stays locked - its own password is still needed.

Not blocked (done by the system itself): Ctrl+Alt+Del and switching to a
text console with Ctrl+Alt+F1..F12 under X11 - see the notes at the end of
this file.  On Wayland a program may not place windows itself, so the
covers of layer 1 need an X11 session ("Ubuntu on Xorg").
"""
import os
import json
import shutil
import hashlib
import threading
import subprocess

from PyQt6.QtCore import Qt, QObject, QEvent, QTimer, QRect
from PyQt6.QtGui import QGuiApplication, QKeyEvent, QCursor
from PyQt6.QtWidgets import (QApplication, QWidget, QLabel, QVBoxLayout,
                             QDialog, QSizePolicy)

MASTER_DEFAULT_HASH = hashlib.sha256(b"8252").hexdigest()


def hash_password(pw):
    return hashlib.sha256(str(pw).strip().encode("utf-8")).hexdigest()


# ----------------------------------------------------------------------------
# GNOME shortcuts (layer 3)
# ----------------------------------------------------------------------------
_WM = "org.gnome.desktop.wm.keybindings"
_SH = "org.gnome.shell.keybindings"
_MK = "org.gnome.settings-daemon.plugins.media-keys"
GNOME_KEYS = (
    [("org.gnome.mutter", "overlay-key"),
     ("org.gnome.desktop.interface", "enable-hot-corners"),
     ("org.gnome.mutter", "edge-tiling")]
    + [(_WM, k) for k in (
        "switch-applications", "switch-applications-backward",
        "switch-windows", "switch-windows-backward", "switch-group",
        "switch-group-backward", "cycle-windows", "cycle-windows-backward",
        "cycle-group", "cycle-group-backward", "cycle-panels",
        "cycle-panels-backward", "switch-panels", "switch-panels-backward",
        "close", "minimize", "maximize", "unmaximize", "toggle-maximized",
        "toggle-fullscreen", "show-desktop", "panel-main-menu",
        "panel-run-dialog", "activate-window-menu", "begin-move",
        "begin-resize", "switch-to-workspace-left",
        "switch-to-workspace-right", "switch-to-workspace-up",
        "switch-to-workspace-down", "switch-to-workspace-last",
        "move-to-workspace-left", "move-to-workspace-right",
        "move-to-monitor-left", "move-to-monitor-right")]
    + [(_SH, k) for k in (
        "toggle-overview", "toggle-application-view", "toggle-message-tray",
        "focus-active-notification", "toggle-quick-settings", "screenshot",
        "screenshot-window", "show-screenshot-ui", "show-screen-recording-ui",
        "shift-overview-up", "shift-overview-down")]
    + [(_MK, k) for k in (
        "terminal", "logout", "screensaver", "home", "search", "www",
        "email", "control-center", "help", "calculator")]
    + [("org.gnome.mutter.wayland.keybindings", f"switch-to-session-{i}")
       for i in range(1, 13)]
    + [("org.gnome.mutter.wayland.keybindings", "restore-shortcuts")])


def _off_value(orig):
    o = orig.strip()
    if o in ("true", "false"):
        return "false"
    if o.startswith("[") or o.startswith("@as"):
        return "[]"
    if o.startswith("'") or o.startswith('"'):
        return "''"
    return None


class GnomeShortcuts:
    """Switch the desktop shortcuts off / back on.  Never raises."""

    def __init__(self, backup_path):
        self.backup_path = backup_path
        self._lock = threading.Lock()
        self.available = shutil.which("gsettings") is not None

    def _run(self, *args):
        try:
            r = subprocess.run(["gsettings", *args], capture_output=True,
                               text=True, timeout=3)
            return r.returncode, r.stdout.strip()
        except Exception:
            return 1, ""

    def disable(self):
        if not self.available:
            return
        with self._lock:
            if os.path.isfile(self.backup_path):
                return                     # already off (backup kept)
            saved = {}
            for schema, key in GNOME_KEYS:
                code, val = self._run("get", schema, key)
                if code != 0 or not val:
                    continue
                off = _off_value(val)
                if off is None or val == off:
                    continue
                saved[f"{schema}|{key}"] = val
            try:
                os.makedirs(os.path.dirname(self.backup_path) or ".",
                            exist_ok=True)
                with open(self.backup_path, "w") as f:
                    json.dump(saved, f, indent=1)
            except OSError:
                return                     # no backup -> change nothing
            for sk, val in saved.items():
                schema, key = sk.split("|", 1)
                self._run("set", schema, key, _off_value(val))

    def restore(self):
        if not self.available:
            return
        with self._lock:
            try:
                with open(self.backup_path) as f:
                    saved = json.load(f)
            except (OSError, json.JSONDecodeError):
                return
            for sk, val in saved.items():
                schema, key = sk.split("|", 1)
                self._run("set", schema, key, val)
            try:
                os.remove(self.backup_path)
            except OSError:
                pass


# ----------------------------------------------------------------------------
# covers: everything outside the program window (layer 1)
# ----------------------------------------------------------------------------
class _Shield(QWidget):
    """A dark cover that swallows the mouse and the keyboard."""

    def __init__(self, rect, text="LOCKED"):
        super().__init__(None, Qt.WindowType.FramelessWindowHint
                         | Qt.WindowType.WindowStaysOnTopHint
                         | Qt.WindowType.Tool
                         | Qt.WindowType.WindowDoesNotAcceptFocus)
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating, True)
        self.setStyleSheet("background: #101114;")
        self.setWindowOpacity(0.82)        # the desktop shows through a bit
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        t = QLabel(text)
        t.setAlignment(Qt.AlignmentFlag.AlignCenter)
        # the text must never make the cover bigger than its strip (it
        # would then lie over the program window)
        t.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored)
        t.setMinimumSize(0, 0)
        self.setMinimumSize(1, 1)
        big = rect.width() > 300 and rect.height() > 120
        t.setStyleSheet("color: #4a4d58; font-weight: 800; letter-spacing: 6px;"
                        f" font-size: {'34pt' if big else '9pt'};")
        t.setVisible(bool(text) and rect.width() > 60 and rect.height() > 24)
        lay.addWidget(t)
        self.setGeometry(rect)

    def mousePressEvent(self, e):
        e.accept()

    def mouseReleaseEvent(self, e):
        e.accept()

    def mouseDoubleClickEvent(self, e):
        e.accept()

    def wheelEvent(self, e):
        e.accept()

    def keyPressEvent(self, e):
        e.accept()


def _outside(screen_rect, win_rect):
    """The parts of screen_rect not covered by win_rect (up to 4 rects)."""
    s = screen_rect
    w = win_rect.intersected(s)
    if w.isEmpty():
        return [s]
    out = [QRect(s.left(), s.top(), s.width(), w.top() - s.top()),            # above
           QRect(s.left(), w.bottom() + 1, s.width(), s.bottom() - w.bottom()),  # below
           QRect(s.left(), w.top(), w.left() - s.left(), w.height()),         # left
           QRect(w.right() + 1, w.top(), s.right() - w.right(), w.height())]  # right
    return [r for r in out if r.width() > 0 and r.height() > 0]


# ----------------------------------------------------------------------------
# the lock itself
# ----------------------------------------------------------------------------
class InputLock(QObject):
    """engage() / release().  `on_master_key` is called for Ctrl+Shift+F12
    (opens the password box from anywhere while locked)."""

    def __init__(self, window, backup_path, grab_keyboard=True,
                 gnome_shortcuts=True, on_master_key=None, log=None):
        super().__init__(window)
        self.win = window
        self.engaged = False
        self.grab = bool(grab_keyboard)
        self.gnome = GnomeShortcuts(backup_path) if gnome_shortcuts else None
        self.on_master_key = on_master_key
        self.log = log or (lambda msg: None)
        self._shields = []
        self._shield_key = None
        self._timer = QTimer(self)
        self._timer.setInterval(250)
        self._timer.timeout.connect(self._keep)
        self._mouse = QTimer(self)          # keeps the pointer inside
        self._mouse.setInterval(15)
        self._mouse.timeout.connect(self._confine)
        self._was_flags = None
        self._geom = None                  # window place / size to keep
        self._maxed = False
        self.wayland = QGuiApplication.platformName().startswith("wayland")
        self._fwd = False

    # -- crash recovery ------------------------------------------------------------
    def recover(self, wait=False):
        """Restore desktop shortcuts left switched off by a crash."""
        if self.gnome is None:
            return
        if wait:
            self.gnome.restore()
        else:
            threading.Thread(target=self.gnome.restore, daemon=True).start()

    # -- on / off -----------------------------------------------------------------------
    def engage(self):
        if self.engaged:
            return
        self.engaged = True
        w = self.win
        self._was_flags = w.windowFlags()
        self._maxed = w.isMaximized()
        self._geom = QRect(w.geometry())
        # the window keeps its size and place.  Title bar without the
        # minimise / maximise / close buttons, always on top.  Changing the
        # flags re-creates the window: put it back exactly as it was.
        T = Qt.WindowType
        want = ((self._was_flags | T.WindowStaysOnTopHint
                 | T.CustomizeWindowHint | T.WindowTitleHint)
                & ~T.WindowMinimizeButtonHint & ~T.WindowMaximizeButtonHint
                & ~T.WindowCloseButtonHint & ~T.WindowMinMaxButtonsHint)
        if w.windowFlags() != want:
            w.setWindowFlags(want)
        self._show_as_before()
        for ms in (100, 400, 1000, 2500):
            QTimer.singleShot(ms, self._front)
        QTimer.singleShot(300, self._cover_outside)
        QGuiApplication.instance().focusWindowChanged.connect(
            self._focus_changed)
        QApplication.instance().installEventFilter(self)
        if self.grab and not self.wayland:
            QTimer.singleShot(300, self._grab)
        if self.gnome is not None:
            threading.Thread(target=self.gnome.disable, daemon=True).start()
        self._timer.start()
        self._mouse.start()
        if self.wayland:
            self.log("keyboard lock: Wayland session - desktop shortcuts are "
                     "switched off, but the full keyboard grab and the covers "
                     "outside the window need an X11 session ('Ubuntu on "
                     "Xorg')")
        self.log("keyboard and mouse locked to the program window")

    def release(self, wait=False):
        if not self.engaged:
            return
        self.engaged = False
        self._timer.stop()
        self._mouse.stop()
        try:
            QGuiApplication.instance().focusWindowChanged.disconnect(
                self._focus_changed)
        except Exception:
            pass
        QApplication.instance().removeEventFilter(self)
        try:
            self.win.releaseKeyboard()
        except Exception:
            pass
        self._drop_shields()
        w = self.win
        if self._was_flags is not None:
            w.setWindowFlags(self._was_flags)
        self._show_as_before()
        if self.gnome is not None:
            if wait:
                self.gnome.restore()
            else:
                threading.Thread(target=self.gnome.restore,
                                 daemon=True).start()
        self.log("keyboard and mouse free again")

    # -- helpers ------------------------------------------------------------------------
    def _show_as_before(self):
        w = self.win
        if self._maxed:
            w.showMaximized()
        else:
            w.showNormal()
            if self._geom is not None:
                w.setGeometry(self._geom)
        w.raise_()
        w.activateWindow()

    def _drop_shields(self):
        for s in self._shields:
            s.close()
            s.deleteLater()
        self._shields = []
        self._shield_key = None

    def _cover_outside(self):
        """Dark covers on every screen area outside the window."""
        if not self.engaged:
            return
        fr = self.win.frameGeometry()
        screens = QGuiApplication.screens()
        key = (fr.x(), fr.y(), fr.width(), fr.height(),
               tuple((sc.geometry().x(), sc.geometry().y(),
                      sc.geometry().width(), sc.geometry().height())
                     for sc in screens))
        if key == self._shield_key:
            return
        self._drop_shields()
        self._shield_key = key
        for sc in screens:
            for r in _outside(sc.geometry(), fr):
                s = _Shield(r)
                s.show()
                self._shields.append(s)
        self._raise_ours()

    def _raise_ours(self):
        """Our window and our open popups above the covers."""
        self.win.raise_()
        for tl in QApplication.topLevelWidgets():
            if tl is not self.win and tl.isVisible() and \
                    isinstance(tl, QDialog):
                tl.raise_()
        m = QApplication.activeModalWidget()
        if m is not None:
            m.raise_()

    def _allowed_rects(self):
        """Where the pointer may be: the main window and every visible
        window of the program itself (dialogs, popups), not the covers."""
        out = [self.win.frameGeometry()]
        shields = set(id(s) for s in self._shields)
        for tl in QApplication.topLevelWidgets():
            if tl is self.win or id(tl) in shields or not tl.isVisible() \
                    or not tl.isWindow():
                continue
            out.append(tl.frameGeometry())
        m = QApplication.activeModalWidget()
        return out, (m.frameGeometry() if m is not None else None)

    def _confine(self):
        """Pointer outside the program: put it back to the nearest point
        inside (inside the open popup, when one is open)."""
        if not self.engaged or self.wayland:
            return
        try:
            pos = QCursor.pos()
            rects, modal = self._allowed_rects()
        except RuntimeError:
            return
        if any(r.contains(pos) for r in rects):
            return
        targets = [modal] if modal is not None else rects
        best, bd = None, None
        for r in targets:
            if r.isEmpty():
                continue
            x = min(max(pos.x(), r.left() + 2), r.right() - 2)
            y = min(max(pos.y(), r.top() + 2), r.bottom() - 2)
            d = (x - pos.x()) ** 2 + (y - pos.y()) ** 2
            if bd is None or d < bd:
                best, bd = (x, y), d
        if best is not None:
            QCursor.setPos(best[0], best[1])

    def _grab(self):
        if self.engaged and QApplication.activePopupWidget() is None:
            self.win.grabKeyboard()

    def _focus_changed(self, _win):
        # another program took the focus (file manager, a USB pop-up ...):
        # take it back at once
        if self.engaged:
            QTimer.singleShot(0, self._front)

    def _front(self):
        """Bring the main window back to the front, at its own size."""
        if not self.engaged:
            return
        w = self.win
        if QApplication.activeModalWidget() is not None or \
                QApplication.activePopupWidget() is not None:
            self._raise_ours()
            return
        if not w.isVisible() or w.isMinimized():
            self._show_as_before()
        if w.isActiveWindow():
            return
        self._raise_ours()
        w.activateWindow()
        h = w.windowHandle()
        if h is not None:
            h.requestActivate()
        self._x11_front()

    def _x11_front(self):
        """Some window managers refuse to give the focus back to a program;
        wmctrl / xdotool (if installed) can force it.  At most once a
        second."""
        import time
        now = time.time()
        if self.wayland or now - getattr(self, "_x11_t", 0) < 1.0:
            return
        self._x11_t = now
        wid = int(self.win.winId())
        if shutil.which("wmctrl"):
            cmd = ["sh", "-c", f"wmctrl -i -r {hex(wid)} -b add,above; "
                               f"wmctrl -i -a {hex(wid)}"]
        elif shutil.which("xdotool"):
            cmd = ["xdotool", "windowactivate", str(wid)]
        else:
            return
        try:
            subprocess.Popen(cmd, stdout=subprocess.DEVNULL,
                             stderr=subprocess.DEVNULL)
        except Exception:
            pass

    def _keep(self):
        """Every quarter second: keep the window where it was, on top, with
        the keyboard; covers follow screens plugged in later."""
        if not self.engaged:
            return
        w = self.win
        if w.isMinimized() or not w.isVisible():
            self._show_as_before()
        elif self._maxed and not w.isMaximized():
            w.showMaximized()
        elif not self._maxed and self._geom is not None and \
                w.geometry() != self._geom and not w.isMaximized():
            w.setGeometry(self._geom)       # moved / resized: put it back
        elif not self._maxed and w.isMaximized():
            w.showNormal()
            w.setGeometry(self._geom)
        self._cover_outside()
        self._front()
        if self.grab and not self.wayland and \
                QApplication.activePopupWidget() is None and \
                QWidget.keyboardGrabber() is not w:
            w.grabKeyboard()

    # -- keys -----------------------------------------------------------------------------
    @staticmethod
    def _blocked(e):
        k, m = e.key(), e.modifiers().value
        M = Qt.KeyboardModifier
        META, ALT, CTRL = (M.MetaModifier.value, M.AltModifier.value,
                           M.ControlModifier.value)
        if k in (Qt.Key.Key_Meta, Qt.Key.Key_Super_L, Qt.Key.Key_Super_R,
                 Qt.Key.Key_Hyper_L, Qt.Key.Key_Hyper_R, Qt.Key.Key_Menu):
            return True
        if m & META:
            return True
        # Alt+Tab, Alt+F4 (close), Alt+F1/F2 (menus, run), Alt+F3, Alt+F5
        # (un-maximise), Alt+F6, Alt+F7 (move), Alt+F8 (resize), Alt+F9
        # (minimise), Alt+F10 (maximise), Alt+Esc, Alt+Space (window menu)
        if (m & ALT) and k in (
                Qt.Key.Key_Tab, Qt.Key.Key_Backtab, Qt.Key.Key_F4,
                Qt.Key.Key_Escape, Qt.Key.Key_Space, Qt.Key.Key_F1,
                Qt.Key.Key_F2, Qt.Key.Key_F3, Qt.Key.Key_F5, Qt.Key.Key_F6,
                Qt.Key.Key_F7, Qt.Key.Key_F8, Qt.Key.Key_F9,
                Qt.Key.Key_F10, Qt.Key.Key_F11, Qt.Key.Key_F12):
            return True
        if (m & CTRL) and (m & ALT):
            return True
        if (m & CTRL) and k in (Qt.Key.Key_Escape, Qt.Key.Key_Q,
                                Qt.Key.Key_W):
            return True
        if k == Qt.Key.Key_F11:            # full-screen toggle
            return True
        return False

    def eventFilter(self, obj, e):
        t = e.type()
        if self.engaged and obj is self.win:
            # minimised / hidden / sent behind: bring it back at once
            if t == QEvent.Type.WindowStateChange and self.win.isMinimized():
                QTimer.singleShot(0, self._front)
            elif t in (QEvent.Type.Hide, QEvent.Type.WindowDeactivate):
                QTimer.singleShot(60, self._front)
        if self.engaged and t == QEvent.Type.Show and \
                isinstance(obj, QDialog) and obj.isWindow():
            QTimer.singleShot(0, obj.raise_)      # our popups above the covers
        if not self.engaged or t not in (QEvent.Type.KeyPress,
                                         QEvent.Type.KeyRelease,
                                         QEvent.Type.ShortcutOverride):
            return False
        if t == QEvent.Type.KeyPress and e.key() == Qt.Key.Key_F12 and \
                e.modifiers() == (Qt.KeyboardModifier.ControlModifier
                                  | Qt.KeyboardModifier.ShiftModifier):
            if self.on_master_key:
                QTimer.singleShot(0, self.on_master_key)
            return True
        if self._blocked(e):
            return True
        # keyboard grabbed by the main window: hand the key to the widget
        # that has the focus (a text box, a dialog, the scan catcher ...)
        # Only a REAL key from the system is handed on, and only once: a key
        # the target does not use travels back up to the main window, and
        # handing it on again would loop forever.
        if self.grab and obj is self.win and not self._fwd and \
                e.spontaneous() and t != QEvent.Type.ShortcutOverride:
            target = QApplication.focusWidget()
            if target is not None and target is not self.win:
                ev = QKeyEvent(t, e.key(), e.modifiers(), e.text(),
                               e.isAutoRepeat(), e.count())
                self._fwd = True
                try:
                    QApplication.sendEvent(target, ev)
                finally:
                    self._fwd = False
                return True
        return False


# NOTES for the station PC (one-time setup, outside this program):
#  * Ctrl+Alt+Del: on Ubuntu run   sudo systemctl mask ctrl-alt-del.target
#  * Ctrl+Alt+F1..F12 (text console) under X11: add to /etc/X11/xorg.conf.d/
#      Section "ServerFlags"
#          Option "DontVTSwitch" "true"
#          Option "DontZap"      "true"
#      EndSection
#  * best: log in with "Ubuntu on Xorg" so the keyboard grab and the covers
#    outside the window work fully.