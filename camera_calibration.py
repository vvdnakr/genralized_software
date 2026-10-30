# coding=utf-8
"""
CAMERA SETUP  -  calibration and tuning of one or many cameras.

  * works on its own:        python camera_calibration.py
  * and inside the main UI:  Home screen -> "Camera Setup"
    (ui_pyqt6_improved.py embeds CalibrationPanel)

Cameras:   MindVision industrial cameras (mvsdk)  +  any USB / OpenCV camera.
Streaming: no threads of our own.  MindVision cameras stream through the
           SDK's callback engine, USB cameras through Qt Multimedia (the
           system's native video pipeline).  Only the newest picture of each
           camera is kept (old ones are dropped, never queued), a small
           preview is made when it arrives, and the screen paints it.  So 4
           cameras stay smooth and a slow camera never slows the others.
Settings:  exposure / gain / colour / gamma ..., hardware or software
           trigger with delay, external-signal type and jitter filter,
           strobe (flash) mode / polarity / delay / pulse width,
           the camera's input and output pins (IO, PWM), resolution and ROI,
           parameter groups stored inside the camera, and a profile file
           (models/camera_profiles.json) that the main program re-applies
           every time it starts the camera.
"""
import os
import re
import sys
import json
import time
import platform
import threading
import datetime
from dataclasses import dataclass, field

import cv2
import numpy as np

try:                                   # MindVision SDK (Windows / Linux)
    import mvsdk
    HAS_MV = True
except Exception:                      # no SDK installed: USB cameras only
    mvsdk = None
    HAS_MV = False

PROFILE_PATH = os.path.join("models", "camera_profiles.json")
SNAP_DIR = os.path.join("Camera", "Data")


# ============================================================================
# PROFILES  (saved settings per camera, which camera the inspection uses)
# ============================================================================
def load_profiles():
    try:
        with open(PROFILE_PATH) as f:
            d = json.load(f)
        d.setdefault("cameras", {})
        d.setdefault("main", "")
        return d
    except (OSError, json.JSONDecodeError):
        return {"cameras": {}, "main": ""}


def save_profiles(d):
    os.makedirs(os.path.dirname(PROFILE_PATH) or ".", exist_ok=True)
    with open(PROFILE_PATH, "w") as f:
        json.dump(d, f, indent=2)


# the order matters: picture size first, trigger last
APPLY_ORDER = ["resolution_index", "roi", "frame_speed", "mono", "inverse",
               "mirror_h", "mirror_v", "rotate", "noise_filter",
               "ae", "ae_target", "anti_flick", "light_freq",
               "exposure_us", "analog_gain", "gamma", "contrast",
               "saturation", "sharpness", "wb_auto", "rgb_gain",
               "strobe_mode", "strobe_polarity", "strobe_delay_us",
               "strobe_width_us", "ext_signal", "ext_shutter",
               "ext_delay_us", "ext_jitter_us", "trigger_count",
               "trigger_delay_us", "trigger_mode"]

ERR_TEXT = {-4: "not supported by this camera", -6: "value not valid",
            -7: "value out of range", -8: "not enabled", -12: "timeout",
            -14: "communication error", -28: "camera busy"}


def _err_text(code):
    return ERR_TEXT.get(code, f"error {code}")


# ============================================================================
# MINDVISION SETTINGS  (shared by this tool and by the main program)
# ============================================================================
def mv_apply(h, name, v):
    """Apply ONE setting to a MindVision camera handle.
    Returns (ok, message).  A refused value never raises."""
    S = mvsdk
    try:
        m = re.match(r"io_(out|in)_(mode|state)_(\d+)$", name)
        if m:
            kind, what, idx = m.group(1), m.group(2), int(m.group(3))
            if what == "mode":
                fn = (S.CameraSetOutPutIOMode if kind == "out"
                      else S.CameraSetInPutIOMode)
                code = fn(h, idx, int(v))
            else:
                code = S.CameraSetIOState(h, idx, 1 if v else 0)
        elif name == "ae":
            code = S.CameraSetAeState(h, 1 if v else 0)
        elif name == "exposure_us":
            code = S.CameraSetExposureTime(h, float(v))
        elif name == "ae_target":
            code = S.CameraSetAeTarget(h, int(v))
        elif name == "anti_flick":
            code = S.CameraSetAntiFlick(h, 1 if v else 0)
        elif name == "light_freq":
            code = S.CameraSetLightFrequency(h, int(v))
        elif name == "analog_gain":
            code = S.CameraSetAnalogGain(h, int(v))
        elif name == "gamma":
            code = S.CameraSetGamma(h, int(v))
        elif name == "contrast":
            code = S.CameraSetContrast(h, int(v))
        elif name == "saturation":
            code = S.CameraSetSaturation(h, int(v))
        elif name == "sharpness":
            code = S.CameraSetSharpness(h, int(v))
        elif name == "wb_auto":
            code = S.CameraSetWbMode(h, 1 if v else 0)
        elif name == "rgb_gain":
            code = S.CameraSetGain(h, int(v[0]), int(v[1]), int(v[2]))
        elif name == "mono":
            code = S.CameraSetMonochrome(h, 1 if v else 0)
        elif name == "inverse":
            code = S.CameraSetInverse(h, 1 if v else 0)
        elif name == "noise_filter":
            code = S.CameraSetNoiseFilter(h, 1 if v else 0)
        elif name == "mirror_h":
            code = S.CameraSetMirror(h, 0, 1 if v else 0)
        elif name == "mirror_v":
            code = S.CameraSetMirror(h, 1, 1 if v else 0)
        elif name == "rotate":
            code = S.CameraSetRotate(h, int(v))
        elif name == "frame_speed":
            code = S.CameraSetFrameSpeed(h, int(v))
        elif name == "trigger_mode":
            code = S.CameraSetTriggerMode(h, int(v))
        elif name == "trigger_count":
            code = S.CameraSetTriggerCount(h, int(v))
        elif name == "trigger_delay_us":
            code = S.CameraSetTriggerDelayTime(h, int(v))
        elif name == "ext_signal":
            code = S.CameraSetExtTrigSignalType(h, int(v))
        elif name == "ext_shutter":
            code = S.CameraSetExtTrigShutterType(h, int(v))
        elif name == "ext_delay_us":
            code = S.CameraSetExtTrigDelayTime(h, int(v))
        elif name == "ext_jitter_us":
            code = S.CameraSetExtTrigJitterTime(h, int(v))
        elif name == "strobe_mode":
            code = S.CameraSetStrobeMode(h, int(v))
        elif name == "strobe_polarity":
            code = S.CameraSetStrobePolarity(h, int(v))
        elif name == "strobe_delay_us":
            code = S.CameraSetStrobeDelayTime(h, int(v))
        elif name == "strobe_width_us":
            code = S.CameraSetStrobePulseWidth(h, int(v))
        elif name == "resolution_index":
            res = S.CameraGetImageResolution(h)
            res.iIndex = int(v)
            code = S.CameraSetImageResolution(h, res)
        elif name == "roi":
            res = S.CameraGetImageResolution(h)
            x, y, w, hh = [int(t) for t in v]
            res.iIndex = 0xFF
            res.iHOffsetFOV, res.iVOffsetFOV = x, y
            res.iWidthFOV, res.iHeightFOV = w, hh
            res.iWidth, res.iHeight = w, hh
            res.iWidthZoomHd = res.iHeightZoomHd = 0
            res.iWidthZoomSw = res.iHeightZoomSw = 0
            code = S.CameraSetImageResolution(h, res)
        else:
            return False, f"unknown setting '{name}'"
    except Exception as e:
        return False, str(e)
    if code not in (0, None):
        return False, _err_text(code)
    return True, "ok"


def mv_apply_all(h, params):
    """Apply a whole saved profile in the right order."""
    bad = []
    for name in APPLY_ORDER:
        if name in params:
            ok, msg = mv_apply(h, name, params[name])
            if not ok:
                bad.append(f"{name}: {msg}")
    for name, v in params.items():
        if name.startswith("io_"):
            ok, msg = mv_apply(h, name, v)
            if not ok:
                bad.append(f"{name}: {msg}")
    return bad


def mv_read(h):
    """Current values from the camera (what the camera really has now)."""
    S = mvsdk
    out = {}

    def g(name, fn):
        try:
            out[name] = fn()
        except Exception:
            pass
    g("ae", lambda: bool(S.CameraGetAeState(h)))
    g("exposure_us", lambda: float(S.CameraGetExposureTime(h)))
    g("ae_target", lambda: int(S.CameraGetAeTarget(h)))
    g("anti_flick", lambda: bool(S.CameraGetAntiFlick(h)))
    g("light_freq", lambda: int(S.CameraGetLightFrequency(h)))
    g("analog_gain", lambda: int(S.CameraGetAnalogGain(h)))
    g("gamma", lambda: int(S.CameraGetGamma(h)))
    g("contrast", lambda: int(S.CameraGetContrast(h)))
    g("saturation", lambda: int(S.CameraGetSaturation(h)))
    g("sharpness", lambda: int(S.CameraGetSharpness(h)))
    g("wb_auto", lambda: bool(S.CameraGetWbMode(h)))
    g("rgb_gain", lambda: [int(t) for t in S.CameraGetGain(h)])
    g("mono", lambda: bool(S.CameraGetMonochrome(h)))
    g("inverse", lambda: bool(S.CameraGetInverse(h)))
    g("noise_filter", lambda: bool(S.CameraGetNoiseFilterState(h)))
    g("mirror_h", lambda: bool(S.CameraGetMirror(h, 0)))
    g("mirror_v", lambda: bool(S.CameraGetMirror(h, 1)))
    g("rotate", lambda: int(S.CameraGetRotate(h)))
    g("frame_speed", lambda: int(S.CameraGetFrameSpeed(h)))
    g("trigger_mode", lambda: int(S.CameraGetTriggerMode(h)))
    g("trigger_count", lambda: int(S.CameraGetTriggerCount(h)))
    g("trigger_delay_us", lambda: int(S.CameraGetTriggerDelayTime(h)))
    g("ext_signal", lambda: int(S.CameraGetExtTrigSignalType(h)))
    g("ext_shutter", lambda: int(S.CameraGetExtTrigShutterType(h)))
    g("ext_delay_us", lambda: int(S.CameraGetExtTrigDelayTime(h)))
    g("ext_jitter_us", lambda: int(S.CameraGetExtTrigJitterTime(h)))
    g("strobe_mode", lambda: int(S.CameraGetStrobeMode(h)))
    g("strobe_polarity", lambda: int(S.CameraGetStrobePolarity(h)))
    g("strobe_delay_us", lambda: int(S.CameraGetStrobeDelayTime(h)))
    g("strobe_width_us", lambda: int(S.CameraGetStrobePulseWidth(h)))
    return out


def apply_saved_profile(wrapper):
    """MAIN PROGRAM HOOK.  Re-apply the profile saved by Camera Setup to a
    CameraWrapper (multi_cam_infrence.py).  Returns a list of problems."""
    try:
        if getattr(wrapper, "camtype", "") != "mindvision" or not HAS_MV:
            return []
        prof = load_profiles()["cameras"].get(camera_id_of(wrapper), {})
        params = prof.get("params") or {}
        if not params:
            return []
        bad = mv_apply_all(wrapper.h, params)
        wrapper.trigger_mode = int(params.get("trigger_mode", 0))
        return bad
    except Exception as e:                     # never stop the program
        return [str(e)]


def camera_id_of(wrapper):
    if getattr(wrapper, "camtype", "") == "usbcam":
        return f"usb:{wrapper.cam_port}"
    dev = getattr(wrapper, "dev_info", None)
    sn = ""
    try:
        sn = dev.GetSn() if dev is not None else ""
    except Exception:
        pass
    return f"mv:{sn}" if sn else "mv:?"


def pick_main_first(cams):
    """MAIN PROGRAM HOOK.  Put the camera the user marked 'use for
    inspection' at position 0 of the list."""
    try:
        want = load_profiles().get("main", "")
        if not want:
            return cams
        for i, c in enumerate(cams):
            if camera_id_of(c) == want:
                cams.insert(0, cams.pop(i))
                break
    except Exception:
        pass
    return cams


# ============================================================================
# DISCOVERY
# ============================================================================
@dataclass
class CamInfo:
    id: str
    kind: str                 # "mv" | "usb" | "demo"
    name: str
    detail: str = ""
    dev: object = None
    index: int = 0


def discover(include_usb=True, skip_usb=()):
    found = []
    if HAS_MV:
        try:
            for i, d in enumerate(mvsdk.CameraEnumerateDevice()):
                sn = d.GetSn()
                found.append(CamInfo(f"mv:{sn}", "mv",
                                     d.GetFriendlyName() or d.GetProductName(),
                                     f"MindVision  {d.GetPortType()}  "
                                     f"SN {sn}", d, i))
        except Exception as e:
            print(f"[camera setup] MindVision scan failed: {e}")
    # A camera that is already open (kept in the pool) is NOT listed again by
    # the MindVision scan ("user control fd open failed"), so add it back
    ids = {i.id for i in found}
    for cam in list(_POOL.values()):
        if cam.info.id not in ids and cam.info.kind == "mv":
            found.append(cam.info)
    if include_usb:
        try:
            from PyQt6.QtMultimedia import QMediaDevices
            for i, d in enumerate(QMediaDevices.videoInputs()):
                if i in skip_usb:
                    continue
                uid = bytes(d.id()).decode("utf-8", "ignore").strip() or str(i)
                found.append(CamInfo(f"usb:{uid}", "usb", d.description()
                                     or f"USB camera {i}",
                                     "USB camera", None, i))
        except Exception as e:
            print(f"[camera setup] USB scan failed: {e}")
    return found


# ============================================================================
# CAMERA OBJECTS  -  no threads of our own
#   MindVision : the SDK's own streaming engine pushes every picture to a
#                callback (CameraSetCallbackFunction).
#   USB        : Qt Multimedia (QCamera + QVideoSink), the system's native
#                video pipeline (DirectShow / Media Foundation / V4L2).
#   Every camera only keeps the newest picture; the screen paints it.
# ============================================================================
class BaseCam:
    kind = "base"
    TILE_W = 960                  # preview width made when a picture arrives

    def __init__(self, info):
        self.info = info
        self.params = {}
        self.caps = {}
        self.error = ""
        self.running = False
        self.opened = False
        self.analyze = False            # selected camera only
        self.idle = False               # open, but nobody is using it
        self.preview_w = self.TILE_W
        self._lock = threading.Lock()
        self._sdk = threading.RLock()
        self._preview = None            # newest small picture (BGR / gray)
        self._full = None               # newest full picture
        self.seq = 0
        self.frame_size = (0, 0)
        self.fps = 0.0
        self.stats = {}                 # focus, mean, clipped, hist
        self._t_last = None

    # -- life cycle -----------------------------------------------------------
    def open(self):
        raise NotImplementedError

    def close(self):
        self.running = False
        self._stop_stream()
        with self._sdk:
            self._close_device()
        self.opened = False

    def start(self):
        if not self.running:
            self.running = True
            self._start_stream()

    def _start_stream(self):
        pass

    def _stop_stream(self):
        pass

    def _close_device(self):
        pass

    # -- a new picture arrived (called by the SDK / Qt / a timer) --------------
    def _push(self, frame):
        if not self.running or self.idle or frame is None:
            return
        now = time.time()
        if self._t_last is not None:
            dt = now - self._t_last
            if dt > 0:
                self.fps = (0.85 * self.fps + 0.15 / dt) if self.fps \
                    else 1.0 / dt
        self._t_last = now
        self.error = ""
        h, w = frame.shape[:2]
        pw = max(160, int(self.preview_w))
        small = (cv2.resize(frame, (pw, max(1, int(h * pw / w))),
                            interpolation=cv2.INTER_AREA)
                 if w > pw else frame)
        with self._lock:
            self._full = frame
            self._preview = small
            self.frame_size = (w, h)
            self.seq += 1

    def analyze_now(self):
        """Focus / brightness numbers from the newest picture (UI timer)."""
        _, img = self.latest()
        if img is None:
            return
        g = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if g.shape[1] > 640:
            g = cv2.resize(g, (640, int(g.shape[0] * 640 / g.shape[1])))
        focus = float(cv2.Laplacian(g, cv2.CV_64F).var())
        hist = cv2.calcHist([g], [0], None, [64], [0, 256]).flatten()
        tot = float(hist.sum()) or 1.0
        self.stats = {"focus": focus, "mean": float(g.mean()),
                      "clipped": 100.0 * float((g >= 250).sum()) / g.size,
                      "dark": 100.0 * float((g <= 5).sum()) / g.size,
                      "hist": (hist / tot).tolist()}

    def latest(self):
        with self._lock:
            return self.seq, self._preview

    def latest_full(self):
        with self._lock:
            return None if self._full is None else self._full.copy()

    # -- settings -----------------------------------------------------------------
    def set(self, name, value):
        with self._sdk:
            ok, msg = self._apply(name, value)
        if ok:
            self.params[name] = value
        return ok, msg

    def _apply(self, name, value):
        return False, "not supported"

    def read_params(self):
        return dict(self.params)

    def specs(self):
        return {}

    def action(self, name):
        return False, "not supported"

    def factory_reset(self):
        return False, "not supported"

    def apply_params(self, params):
        """Apply a saved settings set.  Returns a list of refused values."""
        bad = []
        for k, v in params.items():
            ok, msg = self.set(k, v)
            if not ok:
                bad.append(f"{k}: {msg}")
        return bad

    def autocal_plan(self):
        """(settings to fix first, [candidate settings], changing names).
        An empty list = auto calibration is not possible."""
        return {}, [], ()

    def autocal_defaults(self):
        return {}


_MV_CLOSED_AT = {}          # camera id -> when it was last released


# -- MindVision ---------------------------------------------------------------
class MVCam(BaseCam):
    kind = "mv"

    def open(self):
        S = mvsdk
        # a MindVision camera needs a moment after it was released before it
        # can be opened again (otherwise it reports "in use"): wait, and try
        # a few times
        wait = _MV_CLOSED_AT.get(self.info.id, 0) + 1.0 - time.time()
        if wait > 0:
            time.sleep(min(wait, 1.0))
        for _ in range(15):                  # wait until it is really free
            try:
                if not S.CameraIsOpened(self.info.dev):
                    break
            except Exception:
                break
            time.sleep(0.2)
        last = None
        for _ in range(6):
            try:
                self.h = S.CameraInit(self.info.dev, -1, -1)
                last = None
                break
            except Exception as e:
                last = e
                time.sleep(0.5)
        if last is not None:
            raise last
        try:
            self.cap = S.CameraGetCapability(self.h)
            self.mono = bool(self.cap.sIspCapacity.bMonoSensor)
            S.CameraSetIspOutFormat(self.h, S.CAMERA_MEDIA_TYPE_MONO8
                                    if self.mono else S.CAMERA_MEDIA_TYPE_BGR8)
            S.CameraSetTriggerMode(self.h, 0)
            r = self.cap.sResolutionRange
            self._fb_size = r.iWidthMax * r.iHeightMax * (1 if self.mono else 3)
            self._fb = S.CameraAlignMalloc(self._fb_size, 16)
            self._build_caps()
            S.CameraPlay(self.h)
            self.opened = True
            self.params = mv_read(self.h)
        except Exception:
            self._close_device()      # never leave the camera half open
            raise

    def _close_device(self):
        """Release the camera for real.  CameraUnInit only reports an error
        code (it does not raise), so check it and try again."""
        S = mvsdk
        _MV_CLOSED_AT[self.info.id] = time.time()
        h = getattr(self, "h", None)
        if h is not None:
            try:
                S.CameraSetCallbackFunction(h, S.CAMERA_SNAP_PROC(0), 0)
            except Exception:
                pass
            time.sleep(0.15)               # a picture may still be in flight
            try:
                S.CameraStop(h)
            except Exception:
                pass
            code = -1
            for _ in range(15):
                try:
                    code = S.CameraUnInit(h)
                except Exception:
                    code = -1
                if code == 0:
                    break
                time.sleep(0.2)
            if code != 0:
                print(f"[camera] could not release {self.info.name}: "
                      f"{_err_text(code)}")
        try:
            if getattr(self, "_fb", None):
                S.CameraAlignFree(self._fb)
        except Exception:
            pass
        self.h = None
        self._fb = None

    def _build_caps(self):
        S, cap, h = mvsdk, self.cap, self.h
        c = {}

        def rng(r):
            return (int(r.iMin), int(r.iMax))
        try:
            c["exposure"] = tuple(S.CameraGetExposureTimeRange(h))
        except Exception:
            c["exposure"] = (100.0, 1000000.0, 10.0)
        e = cap.sExposeDesc
        c["analog_gain"] = (int(e.uiAnalogGainMin), int(e.uiAnalogGainMax),
                            float(e.fAnalogGainStep))
        c["ae_target"] = (int(e.uiTargetMin), int(e.uiTargetMax))
        c["gamma"] = rng(cap.sGammaRange)
        c["contrast"] = rng(cap.sContrastRange)
        c["saturation"] = rng(cap.sSaturationRange)
        c["sharpness"] = rng(cap.sSharpnessRange)
        g = cap.sRgbGainRange
        c["rgb"] = ((int(g.iRGainMin), int(g.iRGainMax)),
                    (int(g.iGGainMin), int(g.iGGainMax)),
                    (int(g.iBGainMin), int(g.iBGainMax)))
        r = cap.sResolutionRange
        c["max_size"] = (int(r.iWidthMax), int(r.iHeightMax))
        c["min_size"] = (int(r.iWidthMin), int(r.iHeightMin))
        c["resolutions"] = [
            (int(cap.pImageSizeDesc[i].iIndex),
             f"{cap.pImageSizeDesc[i].GetDescription()}  "
             f"{cap.pImageSizeDesc[i].iWidth}x{cap.pImageSizeDesc[i].iHeight}")
            for i in range(int(cap.iImageSizeDesc))]
        c["frame_speeds"] = [(int(cap.pFrameSpeedDesc[i].iIndex),
                              cap.pFrameSpeedDesc[i].GetDescription())
                             for i in range(int(cap.iFrameSpeedDesc))]
        c["io_out"] = int(cap.iOutputIoCounts)
        c["io_in"] = int(cap.iInputIoCounts)
        i = cap.sIspCapacity
        c["mono"] = bool(i.bMonoSensor)
        c["auto_wb"] = bool(i.bAutoWb)
        c["wb_once"] = bool(i.bWbOnce)
        c["auto_exposure"] = bool(i.bAutoExposure)
        c["anti_flick"] = bool(i.bAntiFlick)
        try:
            c["groups"] = 4
        except Exception:
            pass
        self.caps = c

    def _start_stream(self):
        S = mvsdk
        self._cb = S.CAMERA_SNAP_PROC(self._on_frame)     # keep a reference
        S.CameraSetCallbackFunction(self.h, self._cb, 0)

    def _stop_stream(self):
        try:
            mvsdk.CameraSetCallbackFunction(self.h, mvsdk.CAMERA_SNAP_PROC(0),
                                            0)
        except Exception:
            pass

    def _on_frame(self, hCamera, pRaw, pHead, pContext):
        """Called by the MindVision SDK for every picture (its own thread)."""
        if not self.running or self.idle:
            return
        S = mvsdk
        try:
            head = pHead.contents
            S.CameraImageProcess(self.h, pRaw, self._fb, head)
            if platform.system() == "Windows":
                S.CameraFlipFrameBuffer(self._fb, head, 1)
            chan = 1 if head.uiMediaType == S.CAMERA_MEDIA_TYPE_MONO8 else 3
            buf = (S.c_ubyte * head.uBytes).from_address(self._fb)
            frame = np.frombuffer(buf, dtype=np.uint8).reshape(
                head.iHeight, head.iWidth, chan).copy()
            self._push(frame[:, :, 0] if chan == 1 else frame)
        except Exception as e:
            self.error = f"camera error: {e}"

    def _apply(self, name, value):
        return mv_apply(self.h, name, value)

    def read_params(self):
        self.params.update(mv_read(self.h))
        return dict(self.params)

    def action(self, name):
        S = mvsdk
        try:
            with self._sdk:
                if name == "wb_once":
                    code = S.CameraSetOnceWB(self.h)
                elif name == "bb_once":
                    code = S.CameraSetOnceBB(self.h)
                elif name == "soft_trigger":
                    code = S.CameraSoftTrigger(self.h)
                elif name == "pause_level":
                    code = S.CameraPauseLevelTrigger(self.h)
                elif name.startswith("save_group_"):
                    code = S.CameraSaveParameter(self.h, int(name[-1]))
                elif name.startswith("load_group_"):
                    code = S.CameraLoadParameter(self.h, int(name[-1]))
                elif name == "read_inputs":
                    return True, "  ".join(
                        f"IN{i}={S.CameraGetIOState(self.h, i)}"
                        for i in range(self.caps.get("io_in", 0)))
                else:
                    return False, "unknown action"
        except Exception as e:
            return False, str(e)
        if name == "wb_once":
            self.read_params()
        return (code in (0, None)), ("done" if code in (0, None)
                                     else _err_text(code))

    def apply_params(self, params):
        with self._sdk:
            bad = mv_apply_all(self.h, params)
        self.params.update(params)
        return bad

    def factory_reset(self):
        S = mvsdk
        try:
            with self._sdk:
                code = S.CameraLoadParameter(
                    self.h, getattr(S, "PARAMETER_TEAM_DEFAULT", 0xFF))
                S.CameraSetTriggerMode(self.h, 0)
        except Exception as e:
            return False, str(e)
        self.read_params()
        return (code in (0, None)), ("camera is back to its factory settings"
                                     if code in (0, None) else _err_text(code))

    def autocal_plan(self):
        c = self.caps
        lo, hi = float(c["exposure"][0]), float(c["exposure"][1])
        lo = max(lo, 100.0)
        hi = min(hi, 80000.0)
        exps = [float(round(x)) for x in np.geomspace(lo, hi, 9)]
        g0, g1 = c["analog_gain"][0], c["analog_gain"][1]
        gains = sorted({int(g0), int(g0 + (g1 - g0) * 0.2),
                        int(g0 + (g1 - g0) * 0.45)})
        prep = {"ae": False, "trigger_mode": 0}
        cands = [{"exposure_us": e, "analog_gain": g}
                 for g in gains for e in exps]
        return prep, cands, ("exposure_us", "analog_gain")

    def save_file(self, path):
        with self._sdk:
            return mvsdk.CameraSaveParameterToFile(self.h, path)

    def load_file(self, path):
        with self._sdk:
            return mvsdk.CameraReadParameterFromFile(self.h, path)

    def specs(self):
        c = self.caps
        ex = c["exposure"]
        T = {}
        T["Image"] = [
            dict(kind="check", name="ae", label="Auto exposure",
                 tip="The camera chooses the exposure time by itself."),
            dict(kind="slider", name="ae_target", label="Brightness target",
                 range=c["ae_target"], step=1,
                 tip="Brightness the auto exposure aims for."),
            dict(kind="slider", name="exposure_us", label="Exposure time",
                 range=(ex[0], ex[1]), step=max(1.0, ex[2]), unit="us",
                 log=True, tip="How long the sensor collects light."),
            dict(kind="slider", name="analog_gain", label="Analog gain",
                 range=c["analog_gain"][:2], step=1,
                 tip="Sensor gain. More gain = brighter but noisier."),
            dict(kind="slider", name="gamma", label="Gamma",
                 range=c["gamma"], step=1),
            dict(kind="slider", name="contrast", label="Contrast",
                 range=c["contrast"], step=1),
            dict(kind="slider", name="sharpness", label="Sharpness",
                 range=c["sharpness"], step=1),
            dict(kind="check", name="anti_flick", label="Anti-flicker",
                 tip="Removes banding from lamps on the mains supply."),
            dict(kind="combo", name="light_freq", label="Mains frequency",
                 options=[(0, "50 Hz"), (1, "60 Hz")]),
            dict(kind="check", name="noise_filter", label="Noise filter"),
        ]
        if not c["mono"]:
            T["Colour"] = [
                dict(kind="check", name="wb_auto",
                     label="Automatic white balance"),
                dict(kind="button", name="wb_once",
                     label="White balance once (point at a white / grey part)"),
                dict(kind="slider3", name="rgb_gain", label="Colour gain",
                     range=c["rgb"], labels=("Red", "Green", "Blue")),
                dict(kind="slider", name="saturation", label="Saturation",
                     range=c["saturation"], step=1),
                dict(kind="check", name="mono",
                     label="Black and white picture"),
            ]
        T["Trigger"] = [
            dict(kind="combo", name="trigger_mode", label="Trigger mode",
                 options=[(0, "Free running (live)"),
                          (1, "Software trigger"),
                          (2, "Hardware trigger (signal on the trigger pin)")],
                 tip="Free running = always live.  Software = you press the "
                     "button.  Hardware = the camera takes one picture for "
                     "every signal on its trigger input."),
            dict(kind="button", name="soft_trigger",
                 label="Take one picture now (software trigger)"),
            dict(kind="spin", name="trigger_count", label="Pictures per trigger",
                 range=(1, 1000), step=1),
            dict(kind="spin", name="trigger_delay_us",
                 label="Delay after trigger", range=(0, 10000000), step=100,
                 unit="us", tip="Wait this long after the trigger, then "
                                "take the picture."),
            dict(kind="combo", name="ext_signal", label="Trigger signal",
                 options=[(0, "Rising edge"), (1, "Falling edge"),
                          (2, "High level"), (3, "Low level"),
                          (4, "Both edges")],
                 tip="Which change of the input pin counts as a trigger."),
            dict(kind="combo", name="ext_shutter", label="Shutter type",
                 options=[(0, "Standard"), (1, "Global reset release")]),
            dict(kind="spin", name="ext_delay_us",
                 label="External signal delay", range=(0, 10000000),
                 step=100, unit="us"),
            dict(kind="spin", name="ext_jitter_us",
                 label="Signal jitter filter", range=(0, 1000000), step=10,
                 unit="us", tip="Ignores very short bounces on the input "
                                "line."),
            dict(kind="button", name="pause_level",
                 label="Stop a level trigger"),
        ]
        T["Strobe / IO"] = [
            dict(kind="combo", name="strobe_mode", label="Strobe (flash) mode",
                 options=[(0, "Automatic - with the exposure"),
                          (1, "Manual - own delay and width"),
                          (2, "Always on"), (3, "Always off")]),
            dict(kind="combo", name="strobe_polarity", label="Strobe polarity",
                 options=[(0, "Active low"), (1, "Active high")]),
            dict(kind="spin", name="strobe_delay_us", label="Strobe delay",
                 range=(0, 10000000), step=10, unit="us",
                 tip="Manual mode: time from exposure start to the flash."),
            dict(kind="spin", name="strobe_width_us", label="Strobe pulse width",
                 range=(0, 10000000), step=10, unit="us",
                 tip="Manual mode: how long the flash signal stays on."),
        ]
        for i in range(c["io_out"]):
            T["Strobe / IO"].append(dict(
                kind="combo", name=f"io_out_mode_{i}",
                label=f"Output pin {i}",
                options=[(1, "Strobe output"), (3, "General output"),
                         (4, "PWM output")]))
            T["Strobe / IO"].append(dict(
                kind="check", name=f"io_out_state_{i}",
                label=f"Output pin {i} on (general output)"))
        for i in range(c["io_in"]):
            T["Strobe / IO"].append(dict(
                kind="combo", name=f"io_in_mode_{i}",
                label=f"Input pin {i}",
                options=[(0, "Trigger input"), (2, "General input")]))
        if c["io_in"]:
            T["Strobe / IO"].append(dict(kind="button", name="read_inputs",
                                         label="Read the input pins now",
                                         show_result=True))
        sizes = c["resolutions"]
        T["Picture"] = [
            dict(kind="combo", name="resolution_index",
                 label="Preset size", options=sizes or [(0, "default")]),
            dict(kind="roi", name="roi", label="Custom area (ROI)",
                 range=c["max_size"], minsize=c["min_size"]),
            dict(kind="combo", name="frame_speed", label="Frame speed",
                 options=c["frame_speeds"] or [(0, "normal")]),
            dict(kind="check", name="mirror_h", label="Mirror left-right"),
            dict(kind="check", name="mirror_v", label="Mirror up-down"),
            dict(kind="combo", name="rotate", label="Rotate",
                 options=[(0, "0"), (1, "90"), (2, "180"), (3, "270")]),
            dict(kind="check", name="inverse", label="Invert the picture"),
        ]
        T["Save"] = [
            dict(kind="button", name=f"save_group_{i}",
                 label=f"Store in camera - group {chr(65 + i)}")
            for i in range(4)] + [
            dict(kind="button", name=f"load_group_{i}",
                 label=f"Load from camera - group {chr(65 + i)}")
            for i in range(4)]
        return T


# -- USB: Qt Multimedia ---------------------------------------------------------
class UsbCam(BaseCam):
    kind = "usb"

    def open(self):
        from PyQt6.QtMultimedia import (QCamera, QMediaCaptureSession,
                                        QVideoSink, QMediaDevices)
        devs = QMediaDevices.videoInputs()
        want = self.info.id.split(":", 1)[1]
        self.qdev = next((d for d in devs if bytes(d.id()).decode(
            "utf-8", "ignore").strip() == want), None)
        if self.qdev is None and self.info.index < len(devs):
            self.qdev = devs[self.info.index]
        if self.qdev is None:
            raise RuntimeError("the camera is not there any more")
        self.qcam = QCamera(self.qdev)
        self.sink = QVideoSink()
        self.session = QMediaCaptureSession()
        self.session.setCamera(self.qcam)
        self.session.setVideoSink(self.sink)
        self.sink.videoFrameChanged.connect(self._on_frame)
        self.qcam.errorOccurred.connect(
            lambda _e, msg: setattr(self, "error", msg))
        self.opened = True
        self.params = {}

    def _start_stream(self):
        self.qcam.start()

    def _stop_stream(self):
        try:
            self.qcam.stop()
        except Exception:
            pass

    def _close_device(self):
        try:
            self.sink.videoFrameChanged.disconnect(self._on_frame)
            self.session.setCamera(None)
        except Exception:
            pass

    def _on_frame(self, vf):
        if not self.running or not vf.isValid():
            return
        from PyQt6.QtGui import QImage as _QI
        img = vf.toImage()
        if img.isNull():
            return
        img = img.convertToFormat(_QI.Format.Format_BGR888)
        w, h = img.width(), img.height()
        ptr = img.constBits()
        ptr.setsize(img.sizeInBytes())
        a = np.frombuffer(ptr, np.uint8).reshape(
            h, img.bytesPerLine())[:, :w * 3].reshape(h, w, 3).copy()
        self._push(a)

    def factory_reset(self):
        for k, v in (("exposure_auto", True), ("exposure_comp", 0.0),
                     ("wb_auto", True), ("zoom", 1.0), ("focus_auto", True)):
            self._apply(k, v)
        self.params = {}
        return True, "camera is back to automatic settings"

    def autocal_defaults(self):
        return {"exposure_comp": 0.0, "exposure_auto": True}

    def autocal_plan(self):
        """Manual exposure time when the driver has it; otherwise the
        brightness correction (EV), which nearly every USB camera accepts."""
        from PyQt6.QtMultimedia import QCamera
        c = self.qcam
        lo = float(c.minimumExposureTime())
        hi = min(float(c.maximumExposureTime()), 0.1)
        manual = (c.isExposureModeSupported(QCamera.ExposureMode.ExposureManual)
                  and lo > 0 and hi > lo)
        if manual:
            exps = [round(float(x) * 1000.0, 3)          # seconds -> ms
                    for x in np.geomspace(lo, hi, 9)]
            cands = [{"exposure_time": e} for e in exps]
            isos = []
            if int(c.maximumIsoSensitivity()) > 0:
                lo_i = int(c.minimumIsoSensitivity())
                hi_i = int(c.maximumIsoSensitivity())
                isos = sorted({lo_i, int(lo_i + (hi_i - lo_i) * 0.25)})
            if len(isos) > 1:
                cands = [dict(x, iso=i) for i in isos for x in cands]
            return {"exposure_auto": False}, cands, ("exposure_time", "iso")
        evs = [-3.0, -2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 3.0]
        return ({"exposure_auto": True}, [{"exposure_comp": e} for e in evs],
                ("exposure_comp",))

    def _apply(self, name, value):
        from PyQt6.QtMultimedia import QCamera
        c = self.qcam
        try:
            if name == "exposure_auto":
                c.setExposureMode(QCamera.ExposureMode.ExposureAuto if value
                                  else QCamera.ExposureMode.ExposureManual)
            elif name == "exposure_time":
                c.setManualExposureTime(float(value) / 1000.0)  # ms -> s
            elif name == "exposure_comp":
                c.setExposureCompensation(float(value))
            elif name == "iso":
                c.setManualIsoSensitivity(int(value))
            elif name == "wb_auto":
                c.setWhiteBalanceMode(QCamera.WhiteBalanceMode.WhiteBalanceAuto
                                      if value else
                                      QCamera.WhiteBalanceMode.WhiteBalanceManual)
            elif name == "wb_temp":
                c.setColorTemperature(int(value))
            elif name == "zoom":
                c.setZoomFactor(float(value))
            elif name == "focus_auto":
                c.setFocusMode(QCamera.FocusMode.FocusModeAuto if value
                               else QCamera.FocusMode.FocusModeManual)
            elif name == "focus_dist":
                c.setFocusDistance(float(value))
            elif name == "format":
                c.stop()
                c.setCameraFormat(value)
                c.start()
            else:
                return False, "not supported"
        except Exception as e:
            return False, str(e)
        return True, "ok"

    def specs(self):
        from PyQt6.QtMultimedia import QCamera
        c = self.qcam
        dev = self.qdev
        T = {}
        img = [dict(kind="check", name="exposure_auto",
                    label="Auto exposure"),
               dict(kind="slider", name="exposure_comp",
                    label="Brightness correction (EV)", range=(-4.0, 4.0),
                    step=0.5),
               dict(kind="slider", name="exposure_time",
                    label="Exposure time (manual)", range=(0.1, 1000.0),
                    step=0.1, unit="ms", log=True),
               dict(kind="slider", name="iso", label="ISO (manual)",
                    range=(100, 6400), step=100)]
        T["Image"] = img
        T["Colour"] = [
            dict(kind="check", name="wb_auto", label="Automatic white balance"),
            dict(kind="slider", name="wb_temp", label="White balance temperature",
                 range=(2000, 10000), step=100, unit="K")]
        fmts = sorted(dev.videoFormats(), key=lambda f: (
            -f.resolution().width() * f.resolution().height(),
            -f.maxFrameRate()))
        seen, opts = set(), []
        for f in fmts:
            r = f.resolution()
            key = (r.width(), r.height(), round(f.maxFrameRate()))
            if key in seen:
                continue
            seen.add(key)
            opts.append((f, f"{r.width()} x {r.height()}   "
                            f"{f.maxFrameRate():.0f} fps"))
        T["Picture"] = [
            dict(kind="combo", name="format", label="Size and speed",
                 options=opts or [(None, "default")]),
            dict(kind="slider", name="zoom", label="Zoom",
                 range=(1.0, max(2.0, float(c.maximumZoomFactor()))),
                 step=0.1),
            dict(kind="check", name="focus_auto", label="Auto focus"),
            dict(kind="slider", name="focus_dist", label="Focus distance",
                 range=(0.0, 1.0), step=0.05)]
        T["Trigger"] = [dict(kind="note", label="A USB camera has no trigger "
                             "or strobe pins. It always runs live.  Settings "
                             "that this camera or its driver does not offer "
                             "are refused with a short message.")]
        return T


# -- Demo (no hardware) -----------------------------------------------------------
class DemoCam(BaseCam):
    """A pretend camera, only so the screen can be tried without hardware.
    A timer on the screen thread makes its pictures (no thread)."""
    kind = "demo"

    def open(self):
        from PyQt6.QtCore import QTimer as _T
        self.opened = True
        self.params = {"exposure_us": 8000.0, "analog_gain": 4,
                       "trigger_mode": 0}
        self._t = 0
        self._shot = False
        self._noise = np.random.randint(0, 12, (540, 960, 3), np.uint8)
        self._timer = _T()
        self._timer.setInterval(33)
        self._timer.timeout.connect(self._make)

    def _start_stream(self):
        self._timer.start()

    def _stop_stream(self):
        self._timer.stop()

    def _make(self):
        if int(self.params.get("trigger_mode", 0)) and not self._shot:
            return
        self._shot = False
        self._t += 1
        W, H = 960, 540
        img = np.full((H, W, 3), (92, 104, 88), np.uint8)
        k = float(self.params.get("exposure_us", 8000)) / 8000.0 * \
            (1 + 0.1 * int(self.params.get("analog_gain", 4)))
        x = int(W / 2 + 270 * np.sin(self._t / 40.0 + self.info.index))
        cv2.rectangle(img, (x - 90, 150), (x + 90, 390), (150, 170, 190), -1)
        cv2.circle(img, (x, 270), 52, (60, 90, 120), -1)
        cv2.putText(img, f"DEMO {self.info.index + 1}", (40, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.6, (230, 225, 210), 3)
        img = cv2.add(img, self._noise)
        self._push(np.clip(img.astype(np.float32) * min(2.5, max(0.2, k)),
                           0, 255).astype(np.uint8))

    def _apply(self, name, value):
        return True, "ok"

    def factory_reset(self):
        self.params = {"exposure_us": 8000.0, "analog_gain": 4,
                       "trigger_mode": 0}
        return True, "camera is back to its factory settings"

    def autocal_plan(self):
        exps = [float(x) for x in np.geomspace(500, 40000, 9).round()]
        return ({"trigger_mode": 0},
                [{"exposure_us": e, "analog_gain": g}
                 for g in (1, 4, 10) for e in exps],
                ("exposure_us", "analog_gain"))

    def action(self, name):
        if name == "soft_trigger":
            self._shot = True
        return True, "done"

    def specs(self):
        return {
            "Image": [
                dict(kind="slider", name="exposure_us", label="Exposure time",
                     range=(100, 40000), step=100, unit="us"),
                dict(kind="slider", name="analog_gain", label="Analog gain",
                     range=(1, 40), step=1),
            ],
            "Trigger": [
                dict(kind="combo", name="trigger_mode", label="Trigger mode",
                     options=[(0, "Free running (live)"),
                              (1, "Software trigger")]),
                dict(kind="button", name="soft_trigger",
                     label="Take one picture now (software trigger)"),
            ],
        }


# ----------------------------------------------------------------------------
# CAMERA POOL.  A camera is opened ONCE and handed from screen to screen
# (picker -> tuning -> inspection).  "Releasing" it only makes it idle; it is
# really closed when the program ends.  Closing and re-opening a camera on
# every screen change is what made a MindVision camera report "in use".
# ----------------------------------------------------------------------------
_POOL = {}


def pool_open(info):
    cam = _POOL.get(info.id)
    if cam is not None and cam.opened:
        cam.idle = False
        cam.error = ""
        cam.preview_w = BaseCam.TILE_W
        cam.fps, cam._t_last = 0.0, None
        return cam
    cam = make_camera(info)
    cam.open()
    cam.start()
    if info.kind != "demo":
        _POOL[info.id] = cam
    return cam


def pool_release(cam):
    """Done with it on this screen: keep it open, but stop the work."""
    if cam is None:
        return
    if _POOL.get(cam.info.id) is cam:
        cam.idle = True
    else:
        cam.close()


def pool_close(cam):
    """Really close it (a broken camera, or program exit)."""
    if cam is None:
        return
    if _POOL.get(cam.info.id) is cam:
        _POOL.pop(cam.info.id, None)
    try:
        cam.close()
    except Exception:
        pass


def pool_flush():
    """Close the idle cameras (Scan again: pick up a replugged camera)."""
    for cam in [c for c in _POOL.values() if c.idle]:
        pool_close(cam)


def close_all():
    for cam in list(_POOL.values()):
        pool_close(cam)


import atexit
atexit.register(close_all)


def make_camera(info):
    return {"mv": MVCam, "usb": UsbCam, "demo": DemoCam}[info.kind](info)


# ============================================================================
# UI  (natural, low-contrast palette: moss, sand, clay)
# ============================================================================
from PyQt6.QtCore import (Qt, QTimer, QRectF, QPointF, QObject,
                          pyqtSignal)
from PyQt6.QtGui import (QImage, QPainter, QColor, QPen, QFont, QBrush,
                         QPixmap)
from PyQt6.QtWidgets import (
    QApplication, QWidget, QFrame, QLabel, QPushButton, QCheckBox, QComboBox,
    QSlider, QSpinBox, QDoubleSpinBox, QTabWidget, QVBoxLayout, QHBoxLayout,
    QGridLayout, QScrollArea, QSizePolicy, QFileDialog, QMessageBox,
    QDialog, QProgressBar)

P = dict(bg="#2c2f2a", panel="#363b33", panel2="#40463b", line="#4f5748",
         text="#e8e4d4", muted="#a9ad9b", faint="#7d8274",
         sage="#8ba67b", sage_d="#6f8a62", clay="#c29a6e", water="#86a7b0",
         ochre="#cdb06a", rust="#c07a64", tile="#262925")

QSS = f"""
QWidget {{ background: {P['bg']}; color: {P['text']}; font-size: 10pt;
          font-family: 'Segoe UI', 'Inter', 'Noto Sans', sans-serif; }}
QLabel {{ background: transparent; }}
QFrame#card {{ background: {P['panel']}; border: 1px solid {P['line']};
              border-radius: 12px; }}
QFrame#row {{ background: {P['panel2']}; border: 1px solid {P['line']};
             border-radius: 10px; }}
QPushButton {{ background: {P['panel2']}; border: 1px solid {P['line']};
              border-radius: 8px; padding: 7px 12px; color: {P['text']}; }}
QPushButton:hover {{ background: #4a5144; }}
QPushButton:pressed {{ background: #353a31; }}
QPushButton:disabled {{ color: {P['faint']}; background: #31352e; }}
QPushButton#primary {{ background: {P['sage_d']}; border: 1px solid {P['sage']};
                      color: #f3f0e4; font-weight: 600; }}
QPushButton#primary:hover {{ background: {P['sage']}; color: #20241d; }}
QPushButton#clay {{ background: #8d6f4f; border: 1px solid {P['clay']};
                   color: #f3f0e4; }}
QPushButton#clay:hover {{ background: {P['clay']}; color: #231d15; }}
QComboBox, QSpinBox, QDoubleSpinBox {{ background: {P['tile']};
    border: 1px solid {P['line']}; border-radius: 7px; padding: 5px 8px; }}
QComboBox QAbstractItemView {{ background: {P['panel']};
    selection-background-color: {P['sage_d']}; }}
QSlider::groove:horizontal {{ height: 5px; background: {P['line']};
                             border-radius: 2px; }}
QSlider::sub-page:horizontal {{ background: {P['sage_d']}; border-radius: 2px; }}
QSlider::handle:horizontal {{ background: {P['sage']}; width: 14px;
    margin: -6px 0; border-radius: 7px; }}
QCheckBox {{ spacing: 8px; }}
QCheckBox::indicator {{ width: 16px; height: 16px; border-radius: 4px;
    border: 1px solid {P['faint']}; background: {P['tile']}; }}
QCheckBox::indicator:checked {{ background: {P['sage_d']};
    border-color: {P['sage']}; }}
QTabWidget::pane {{ border: 1px solid {P['line']}; border-radius: 10px;
                   background: {P['panel']}; top: -1px; }}
QTabBar::tab {{ background: {P['bg']}; color: {P['muted']}; padding: 7px 11px;
    border: 1px solid {P['line']}; border-bottom: none;
    border-top-left-radius: 8px; border-top-right-radius: 8px;
    margin-right: 2px; }}
QTabBar::tab:selected {{ background: {P['panel']}; color: {P['text']}; }}
QScrollArea {{ border: none; background: transparent; }}
QScrollBar:vertical {{ background: transparent; width: 10px; }}
QScrollBar::handle:vertical {{ background: {P['line']}; border-radius: 5px;
                              min-height: 24px; }}
QScrollBar::add-line, QScrollBar::sub-line {{ height: 0; width: 0; }}
QToolTip {{ background: {P['panel']}; color: {P['text']};
           border: 1px solid {P['line']}; padding: 5px; }}
"""


class StreamTile(QWidget):
    """One live picture.  Paints only when a new picture has arrived."""
    clicked = pyqtSignal(str)
    double_clicked = pyqtSignal(str)

    def __init__(self, cam):
        super().__init__()
        self.cam = cam
        self.selected = False
        self.grid = False
        self.cross = False
        self._seq = -1
        self._qimg = None
        self._keep = None
        self.setMinimumSize(200, 130)
        self.setSizePolicy(QSizePolicy.Policy.Expanding,
                           QSizePolicy.Policy.Expanding)

    def poll(self):
        seq, img = self.cam.latest()
        if img is None or seq == self._seq:
            return
        self._seq = seq
        a = np.ascontiguousarray(img)
        h, w = a.shape[:2]
        fmt = (QImage.Format.Format_Grayscale8 if a.ndim == 2
               else QImage.Format.Format_BGR888)
        self._keep = a
        self._qimg = QImage(a.data, w, h, a.strides[0], fmt)
        self.update()

    def paintEvent(self, e):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)
        r = self.rect().adjusted(2, 2, -2, -2)
        p.fillRect(self.rect(), QColor(P["tile"]))
        if self._qimg is not None:
            iw, ih = self._qimg.width(), self._qimg.height()
            s = min(r.width() / iw, r.height() / ih)
            w, h = iw * s, ih * s
            x, y = r.x() + (r.width() - w) / 2, r.y() + (r.height() - h) / 2
            tgt = QRectF(x, y, w, h)
            p.drawImage(tgt, self._qimg)
            pen = QPen(QColor(255, 255, 255, 70), 1)
            p.setPen(pen)
            if self.grid:
                for i in range(1, 3):
                    p.drawLine(QPointF(x + w * i / 3, y),
                               QPointF(x + w * i / 3, y + h))
                    p.drawLine(QPointF(x, y + h * i / 3),
                               QPointF(x + w, y + h * i / 3))
            if self.cross:
                p.setPen(QPen(QColor(P["ochre"]), 1))
                p.drawLine(QPointF(x + w / 2, y), QPointF(x + w / 2, y + h))
                p.drawLine(QPointF(x, y + h / 2), QPointF(x + w, y + h / 2))
        else:
            p.setPen(QColor(P["muted"]))
            msg = self.cam.error or "waiting for the first picture..."
            p.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, msg)
        # name / speed strip
        bar = QRectF(0, self.height() - 24, self.width(), 24)
        p.fillRect(bar, QColor(20, 22, 18, 170))
        p.setPen(QColor(P["text"]))
        w, h = self.cam.frame_size
        p.drawText(bar.adjusted(8, 0, -8, 0),
                   Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft,
                   f"{self.cam.info.name}")
        p.setPen(QColor(P["muted"]))
        p.drawText(bar.adjusted(8, 0, -8, 0),
                   Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignRight,
                   f"{w}x{h}   {self.cam.fps:4.1f} fps")
        p.setPen(QPen(QColor(P["sage"] if self.selected else P["line"]),
                      3 if self.selected else 1))
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawRoundedRect(QRectF(self.rect()).adjusted(1, 1, -1, -1), 6, 6)

    def mousePressEvent(self, e):
        self.clicked.emit(self.cam.info.id)

    def mouseDoubleClickEvent(self, e):
        self.double_clicked.emit(self.cam.info.id)


class HistBar(QWidget):
    def __init__(self):
        super().__init__()
        self.h = []
        self.setFixedHeight(54)

    def set_hist(self, h):
        self.h = h
        self.update()

    def paintEvent(self, e):
        p = QPainter(self)
        p.fillRect(self.rect(), QColor(P["tile"]))
        if not self.h:
            return
        m = max(self.h) or 1
        n = len(self.h)
        w = self.width() / n
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QColor(P["water"]))
        for i, v in enumerate(self.h):
            bh = (v / m) ** 0.5 * (self.height() - 4)
            p.drawRect(QRectF(i * w, self.height() - bh, max(1.0, w - 1), bh))


class ParamRow(QWidget):
    """One setting: label + control(s).  Calls on_change(name, value)."""

    def __init__(self, spec, on_change, on_action):
        super().__init__()
        self.spec, self.on_change, self.on_action = spec, on_change, on_action
        self.name = spec.get("name", "")
        self._quiet = False
        self._debounce = QTimer(self)
        self._debounce.setSingleShot(True)
        self._debounce.setInterval(70)
        self._debounce.timeout.connect(self._emit)
        self._pending = None
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 4, 0, 4)
        lay.setSpacing(3)
        k = spec["kind"]
        tip = spec.get("tip", "")
        self.setToolTip(tip)
        if k == "check":
            self.w = QCheckBox(spec["label"])
            self.w.toggled.connect(lambda v: self.on_change(self.name, bool(v)))
            lay.addWidget(self.w)
        elif k == "button":
            self.w = QPushButton(spec["label"])
            self.w.clicked.connect(lambda: self._act())
            lay.addWidget(self.w)
            self.result = QLabel("")
            self.result.setStyleSheet(f"color: {P['muted']};")
            self.result.setWordWrap(True)
            if spec.get("show_result"):
                lay.addWidget(self.result)
        elif k == "note":
            n = QLabel(spec["label"])
            n.setWordWrap(True)
            n.setStyleSheet(f"color: {P['muted']};")
            lay.addWidget(n)
        elif k == "combo":
            lay.addWidget(self._title(spec["label"]))
            self.w = QComboBox()
            for val, txt in spec["options"]:
                self.w.addItem(txt, val)
            self.w.currentIndexChanged.connect(
                lambda i: self.on_change(self.name, self.w.itemData(i)))
            lay.addWidget(self.w)
        elif k in ("slider", "spin"):
            lay.addWidget(self._title(spec["label"]))
            row = QHBoxLayout()
            lo, hi = spec["range"]
            self.is_float = isinstance(lo, float) or isinstance(hi, float)
            self.log = bool(spec.get("log")) and lo > 0 and hi / lo > 500
            if k == "slider":
                self.sl = QSlider(Qt.Orientation.Horizontal)
                self.sl.setRange(0, 1000)
                self.sl.valueChanged.connect(self._from_slider)
                row.addWidget(self.sl, stretch=1)
            self.sp = QDoubleSpinBox() if self.is_float else QSpinBox()
            if self.is_float:
                self.sp.setDecimals(1)
            self.sp.setRange(lo, hi)
            self.sp.setSingleStep(spec.get("step", 1))
            if spec.get("unit"):
                self.sp.setSuffix(" " + spec["unit"])
            self.sp.setMinimumWidth(110)
            self.sp.setKeyboardTracking(False)
            self.sp.valueChanged.connect(self._from_spin)
            row.addWidget(self.sp)
            lay.addLayout(row)
        elif k == "slider3":
            lay.addWidget(self._title(spec["label"]))
            self.sub = []
            for i, (lab, rng) in enumerate(zip(spec["labels"], spec["range"])):
                row = QHBoxLayout()
                l = QLabel(lab)
                l.setMinimumWidth(48)
                row.addWidget(l)
                s = QSlider(Qt.Orientation.Horizontal)
                s.setRange(rng[0], rng[1])
                sp = QSpinBox()
                sp.setRange(rng[0], rng[1])
                sp.setMinimumWidth(70)
                s.valueChanged.connect(sp.setValue)
                sp.valueChanged.connect(s.setValue)
                sp.valueChanged.connect(lambda _v: self._rgb_changed())
                row.addWidget(s, stretch=1)
                row.addWidget(sp)
                lay.addLayout(row)
                self.sub.append((s, sp))
        elif k == "roi":
            lay.addWidget(self._title(spec["label"]))
            mw, mh = spec["range"]
            g = QGridLayout()
            self.roi = []
            for j, (lab, hi) in enumerate((("X", mw), ("Y", mh),
                                           ("Width", mw), ("Height", mh))):
                g.addWidget(QLabel(lab), j // 2, (j % 2) * 2)
                sp = QSpinBox()
                sp.setRange(0, hi)
                sp.setSingleStep(8)
                g.addWidget(sp, j // 2, (j % 2) * 2 + 1)
                self.roi.append(sp)
            self.roi[2].setValue(mw)
            self.roi[3].setValue(mh)
            lay.addLayout(g)
            b = QPushButton("Apply the area")
            b.clicked.connect(lambda: self.on_change(
                self.name, [s.value() for s in self.roi]))
            lay.addWidget(b)

    def _title(self, t):
        l = QLabel(t)
        l.setStyleSheet(f"color: {P['muted']}; font-size: 9pt;")
        return l

    def _act(self):
        ok, msg = self.on_action(self.name)
        if hasattr(self, "result"):
            self.result.setText(("" if ok else "Not done: ") + str(msg))

    # slider <-> spin
    def _to_pos(self, v):
        lo, hi = self.spec["range"]
        if self.log:
            import math
            return int(1000 * (math.log(max(v, lo)) - math.log(lo))
                       / (math.log(hi) - math.log(lo)))
        return int(1000 * (v - lo) / ((hi - lo) or 1))

    def _from_pos(self, pos):
        lo, hi = self.spec["range"]
        if self.log:
            import math
            return math.exp(math.log(lo) + (math.log(hi) - math.log(lo))
                            * pos / 1000)
        return lo + (hi - lo) * pos / 1000

    def _from_slider(self, pos):
        if self._quiet:
            return
        v = self._from_pos(pos)
        self._quiet = True
        self.sp.setValue(v if self.is_float else int(round(v)))
        self._quiet = False
        self._queue(self.sp.value())

    def _from_spin(self, v):
        if self._quiet:
            return
        if hasattr(self, "sl"):
            self._quiet = True
            self.sl.setValue(self._to_pos(v))
            self._quiet = False
        self._queue(v)

    def _queue(self, v):
        self._pending = v
        self._debounce.start()

    def _emit(self):
        if self._pending is not None:
            self.on_change(self.name, self._pending)

    def _rgb_changed(self):
        if self._quiet:
            return
        self._pending = [sp.value() for _, sp in self.sub]
        self._debounce.start()

    def set_value(self, v):
        """Show the camera's value without sending it back."""
        self._quiet = True
        try:
            k = self.spec["kind"]
            if k == "check":
                self.w.setChecked(bool(v))
            elif k == "combo":
                i = self.w.findData(v)
                if i >= 0:
                    self.w.setCurrentIndex(i)
            elif k in ("slider", "spin"):
                self.sp.setValue(v if self.is_float else int(round(v)))
                if hasattr(self, "sl"):
                    self.sl.setValue(self._to_pos(v))
            elif k == "slider3":
                for (s, sp), t in zip(self.sub, v):
                    sp.setValue(int(t))
            elif k == "roi":
                for s, t in zip(self.roi, v):
                    s.setValue(int(t))
        except Exception:
            pass
        self._quiet = False

    def set_enabled_state(self, on):
        self.setEnabled(on)


class CamRow(QFrame):
    """A line in the camera list: show / hide, name, details, status."""
    toggled = pyqtSignal(str, bool)

    def __init__(self, info):
        super().__init__()
        self.setObjectName("row")
        self.info = info
        lay = QHBoxLayout(self)
        lay.setContentsMargins(10, 8, 10, 8)
        self.chk = QCheckBox("")
        self.chk.toggled.connect(lambda v: self.toggled.emit(info.id, v))
        lay.addWidget(self.chk)
        col = QVBoxLayout()
        col.setSpacing(1)
        n = QLabel(info.name)
        n.setWordWrap(True)
        n.setStyleSheet("font-weight: 600;")
        d = QLabel(info.detail)
        d.setStyleSheet(f"color: {P['muted']}; font-size: 8pt;")
        d.setWordWrap(True)
        self.state = QLabel("not connected")
        self.state.setStyleSheet(f"color: {P['faint']}; font-size: 8pt;")
        col.addWidget(n)
        col.addWidget(d)
        col.addWidget(self.state)
        lay.addLayout(col, stretch=1)

    def set_state(self, text, kind="idle"):
        c = {"ok": P["sage"], "bad": P["rust"], "idle": P["faint"],
             "main": P["clay"]}[kind]
        self.state.setText(text)
        self.state.setStyleSheet(f"color: {c}; font-size: 8pt;")


class CalibrationPanel(QWidget):
    """The whole Camera Setup page.  `back` is emitted by the Back button
    (only shown when embedded in the main program)."""
    back = pyqtSignal()
    done = pyqtSignal()               # model mode: "Done"
    change_cameras = pyqtSignal()     # model mode: go back to the picker

    def __init__(self, embedded=False, parent=None, demo=False, model=None):
        super().__init__(parent)
        self.model = model            # ModelCameras -> "model mode"
        self.cur_profile = {}         # cam id -> settings set shown
        self.setStyleSheet(QSS)
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.embedded = embedded
        self.cams = {}            # id -> BaseCam (opened)
        self.tiles = {}           # id -> StreamTile
        self.rows = {}
        self.infos = []
        self.sel = None
        self.focus_mode = False
        self.param_rows = {}
        self.profiles = load_profiles()
        self._demo_n = 0
        self._build()
        QTimer.singleShot(50, self.rescan)
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)
        self._timer.start(33)                      # ~30 pictures / second
        self._slow = QTimer(self)
        self._slow.timeout.connect(self._slow_tick)
        self._slow.start(400)
        if demo:
            QTimer.singleShot(200, self.add_demo)

    # -- layout ------------------------------------------------------------------
    def _build(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(14, 12, 14, 12)
        root.setSpacing(10)
        top = QHBoxLayout()
        if self.embedded:
            b = QPushButton("Back")
            b.clicked.connect(self.back.emit)
            top.addWidget(b)
        t = QLabel("Tune the cameras" if self.model else "Camera Setup")
        t.setStyleSheet("font-size: 17pt; font-weight: 700;")
        top.addWidget(t)
        self.sub = QLabel("")
        self.sub.setStyleSheet(f"color: {P['muted']};")
        top.addWidget(self.sub)
        top.addStretch()
        self.chk_grid = QCheckBox("Grid")
        self.chk_cross = QCheckBox("Centre lines")
        for c in (self.chk_grid, self.chk_cross):
            c.toggled.connect(self._overlay)
            top.addWidget(c)
        self.btn_focus = QPushButton("Show only the selected")
        self.btn_focus.setCheckable(True)
        self.btn_focus.toggled.connect(self._set_focus_mode)
        top.addWidget(self.btn_focus)
        b = QPushButton("Save picture")
        b.clicked.connect(self.snapshot)
        top.addWidget(b)
        b = QPushButton("Refresh" if self.model else "Scan again")
        b.clicked.connect(self.rescan)
        top.addWidget(b)
        if self.model:
            b = QPushButton("Light check...")
            b.setToolTip("How much the light may change before a warning "
                         "is shown.")
            b.clicked.connect(self.open_drift)
            top.addWidget(b)
            b = QPushButton("Change cameras")
            b.clicked.connect(self.change_cameras.emit)
            top.addWidget(b)
            b = QPushButton("Done")
            b.setObjectName("primary")
            b.clicked.connect(self.done.emit)
            top.addWidget(b)
        root.addLayout(top)

        body = QHBoxLayout()
        body.setSpacing(12)
        # left: camera list
        left = QFrame()
        left.setObjectName("card")
        left.setFixedWidth(235)
        ll = QVBoxLayout(left)
        ll.setContentsMargins(10, 10, 10, 10)
        cap = QLabel("CAMERAS")
        cap.setStyleSheet(f"color: {P['faint']}; font-size: 8pt;"
                          " font-weight: 700; letter-spacing: 1px;")
        ll.addWidget(cap)
        hint = QLabel("Tick a camera to see it.  Tick several to see them "
                      "all at once.")
        hint.setWordWrap(True)
        hint.setStyleSheet(f"color: {P['muted']}; font-size: 8pt;")
        ll.addWidget(hint)
        sc = QScrollArea()
        sc.setWidgetResizable(True)
        sc.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.list_w = QWidget()
        self.list_w.setStyleSheet("background: transparent;")
        self.list_lay = QVBoxLayout(self.list_w)
        self.list_lay.setContentsMargins(0, 0, 0, 0)
        self.list_lay.setSpacing(6)
        self.list_lay.addStretch()
        sc.setWidget(self.list_w)
        ll.addWidget(sc, stretch=1)
        self.btn_demo = QPushButton("Add a demo camera")
        self.btn_demo.clicked.connect(self.add_demo)
        self.btn_demo.setVisible(not self.model)
        ll.addWidget(self.btn_demo)
        body.addWidget(left)

        # centre: the streams
        self.stage = QFrame()
        self.stage.setObjectName("card")
        self.grid_lay = QGridLayout(self.stage)
        self.grid_lay.setContentsMargins(8, 8, 8, 8)
        self.grid_lay.setSpacing(8)
        self.empty = QLabel("No camera is shown.\nTick a camera on the left.")
        self.empty.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.empty.setStyleSheet(f"color: {P['muted']}; font-size: 12pt;")
        self.grid_lay.addWidget(self.empty, 0, 0)
        body.addWidget(self.stage, stretch=1)

        # right: settings
        right = QFrame()
        right.setObjectName("card")
        right.setFixedWidth(380)
        rl = QVBoxLayout(right)
        rl.setContentsMargins(10, 10, 10, 10)
        rl.setSpacing(8)
        self.sel_lbl = QLabel("No camera selected")
        self.sel_lbl.setStyleSheet("font-size: 12pt; font-weight: 700;")
        self.sel_lbl.setWordWrap(True)
        rl.addWidget(self.sel_lbl)
        self.prof_bar = QWidget()
        pb = QHBoxLayout(self.prof_bar)
        pb.setContentsMargins(0, 0, 0, 0)
        pb.addWidget(QLabel("Settings set"))
        self.prof_combo = QComboBox()
        self.prof_combo.setToolTip("One camera can have several settings "
                                   "sets - for example one for a shiny part "
                                   "and one for a dark part.")
        self.prof_combo.activated.connect(self._profile_picked)
        pb.addWidget(self.prof_combo, stretch=1)
        self.btn_prof_new = QPushButton("New")
        self.btn_prof_new.clicked.connect(self.new_profile)
        pb.addWidget(self.btn_prof_new)
        self.btn_prof_del = QPushButton("Delete")
        self.btn_prof_del.clicked.connect(self.delete_profile)
        pb.addWidget(self.btn_prof_del)
        self.prof_bar.setVisible(bool(self.model))
        rl.addWidget(self.prof_bar)
        self.tabs = QTabWidget()
        rl.addWidget(self.tabs, stretch=1)
        # live numbers
        self.metrics = QLabel("")
        self.metrics.setStyleSheet(f"color: {P['muted']}; font-size: 9pt;")
        self.metrics.setWordWrap(True)
        rl.addWidget(self.metrics)
        self.hist = HistBar()
        rl.addWidget(self.hist)
        self.msg = QLabel("")
        self.msg.setWordWrap(True)
        self.msg.setStyleSheet(f"color: {P['ochre']}; font-size: 9pt;")
        rl.addWidget(self.msg)
        r1 = QHBoxLayout()
        self.btn_save = QPushButton("Save settings set" if self.model
                                    else "Save these settings")
        self.btn_save.setObjectName("primary")
        self.btn_save.clicked.connect(self.save_settings)
        r1.addWidget(self.btn_save)
        self.btn_main = QPushButton("Use for inspection")
        self.btn_main.setObjectName("clay")
        self.btn_main.clicked.connect(self.set_main)
        self.btn_main.setVisible(not self.model)
        r1.addWidget(self.btn_main)
        self.btn_auto = QPushButton("Auto calibrate")
        self.btn_auto.setObjectName("clay")
        self.btn_auto.setToolTip("The program tries many exposure and gain "
                                 "values and shows the 5 best pictures.")
        self.btn_auto.clicked.connect(self.auto_calibrate)
        r1.addWidget(self.btn_auto)
        rl.addLayout(r1)
        r2 = QHBoxLayout()
        self.btn_reload = QPushButton("Back to saved")
        self.btn_reload.clicked.connect(self.reload_saved)
        r2.addWidget(self.btn_reload)
        self.btn_copy = QPushButton("Copy to the other cameras")
        self.btn_copy.clicked.connect(self.copy_to_others)
        r2.addWidget(self.btn_copy)
        rl.addLayout(r2)
        r3 = QHBoxLayout()
        self.btn_reset = QPushButton("Reset camera")
        self.btn_reset.setToolTip("Put every setting of this camera back to "
                                  "the factory values.")
        self.btn_reset.clicked.connect(self.reset_camera)
        r3.addWidget(self.btn_reset)
        r3.addStretch()
        rl.addLayout(r3)
        body.addWidget(right)
        root.addLayout(body, stretch=1)
        for b in self.findChildren(QPushButton):
            b.setMinimumWidth(b.sizeHint().width() + 6)
        self.setMinimumSize(1100, 640)
        self._select(None)

    # -- cameras --------------------------------------------------------------------
    def rescan(self):
        if self.model is not None:
            return self._rescan_model()
        busy_usb = {c.info.index for c in self.cams.values()
                    if c.kind == "usb"}
        keep = [i for i in self.infos if i.kind == "demo"]
        for i in list(self.cams):
            if self.cams[i].kind != "demo":
                continue
        found = discover(True, skip_usb=busy_usb)
        known = {c.info.id: c.info for c in self.cams.values()}
        infos = []
        for f in found:
            infos.append(known.get(f.id, f))
        for k, i in known.items():
            if i.id not in {x.id for x in infos}:
                infos.append(i)
        self.infos = infos + [d for d in keep if d.id not in
                              {x.id for x in infos}]
        self._fill_list()

    def _rescan_model(self):
        found = {i.id: i for i in discover(True)}
        known = {c.info.id: c.info for c in self.cams.values()}
        infos = []
        for s in self.model.slots():
            cid = s["id"]
            if cid in known:
                inf = known[cid]
            elif cid.startswith("demo:"):
                inf = CamInfo(cid, "demo", s["name"], "pretend camera", None,
                              int(cid.split(":")[1]) - 1)
            elif cid in found:
                inf = found[cid]
            else:
                inf = CamInfo(cid, "missing", s["name"],
                              "not connected - check the cable", None, 0)
            inf = CamInfo(inf.id, inf.kind, f"Camera {s['slot']}  -  "
                          f"{s['name']}", inf.detail, inf.dev, inf.index)
            infos.append(inf)
        self.infos = infos
        self._fill_list()
        for inf in infos:
            if inf.kind != "missing" and inf.id not in self.cams:
                self.rows[inf.id].chk.setChecked(True)

    def _fill_list(self):
        while self.list_lay.count() > 1:
            it = self.list_lay.takeAt(0)
            w = it.widget()
            if w is not None:
                w.hide()
                w.setParent(None)
                w.deleteLater()
        self.rows = {}
        for info in self.infos:
            row = CamRow(info)
            if info.kind == "missing":
                row.chk.setEnabled(False)
                row.set_state("not connected", "bad")
            row.toggled.connect(self._toggle_cam)
            self.list_lay.insertWidget(self.list_lay.count() - 1, row)
            self.rows[info.id] = row
            if info.id in self.cams:
                row.chk.blockSignals(True)
                row.chk.setChecked(True)
                row.chk.blockSignals(False)
                row.set_state("live", "ok")
            elif self.profiles.get("main") == info.id:
                row.set_state("used for inspection", "main")
        n = len(self.infos)
        if self.model is not None:
            bad = sum(1 for i in self.infos if i.kind == "missing")
            self.sub.setText(f"{n} camera(s) in this model"
                             + (f"  -  {bad} not connected" if bad else ""))
        else:
            self.sub.setText(f"{n} camera(s) found" if n
                             else "no camera found - check cable and power")

    def add_demo(self):
        self._demo_n += 1
        n = self._demo_n
        info = CamInfo(f"demo:{n}", "demo", f"Demo camera {n}",
                       "pretend camera - no hardware", None, n - 1)
        self.infos.append(info)
        self._fill_list()
        self.rows[info.id].chk.setChecked(True)

    def _toggle_cam(self, cid, on):
        info = next(i for i in self.infos if i.id == cid)
        row = self.rows[cid]
        if on:
            try:
                cam = pool_open(info)
            except Exception as e:
                row.chk.blockSignals(True)
                row.chk.setChecked(False)
                row.chk.blockSignals(False)
                row.set_state(f"could not open: {e}", "bad")
                self._say(f"{info.name}: could not open - {e}")
                return
            # re-apply what was saved for this camera
            if self.model is not None:
                name = self.cur_profile.get(cid) or \
                    self.model.default_profile(cid)
                self.cur_profile[cid] = name
                prof = self.model.profile_params(name) if name else {}
            else:
                prof = self.profiles["cameras"].get(cid, {}).get("params")
            if prof:
                bad = cam.apply_params(prof)
                try:
                    cam.read_params()
                except Exception:
                    pass
                if bad:
                    self._say("Some saved values were not accepted: "
                              + "; ".join(bad[:3]))
            self.cams[cid] = cam
            tile = StreamTile(cam)
            tile.grid = self.chk_grid.isChecked()
            tile.cross = self.chk_cross.isChecked()
            tile.clicked.connect(self._select)
            tile.double_clicked.connect(self._double)
            self.tiles[cid] = tile
            row.set_state("live", "ok")
            self._relayout()
            self._select(cid)
        else:
            cam = self.cams.pop(cid, None)
            tile = self.tiles.pop(cid, None)
            if cam is not None:
                pool_release(cam)
            if tile is not None:
                tile.setParent(None)
                tile.deleteLater()
            row.set_state("not connected", "idle")
            if self.sel == cid:
                self._select(next(iter(self.cams), None))
            self._relayout()

    def _relayout(self):
        for i in reversed(range(self.grid_lay.count())):
            it = self.grid_lay.takeAt(i)
            w = it.widget()
            if w is not None:
                w.hide()
        ids = list(self.tiles)
        if self.focus_mode and self.sel in self.tiles:
            ids = [self.sel]
        n = len(ids)
        if n == 0:
            self.grid_lay.addWidget(self.empty, 0, 0)
            self.empty.show()
            return
        cols = 1 if n == 1 else 2 if n <= 4 else 3
        for k, cid in enumerate(ids):
            self.grid_lay.addWidget(self.tiles[cid], k // cols, k % cols)
            self.tiles[cid].show()
        for c in range(cols):
            self.grid_lay.setColumnStretch(c, 1)
        for r in range((n + cols - 1) // cols):
            self.grid_lay.setRowStretch(r, 1)
        # fewer pixels per tile when many are shown
        pw = 1280 if n == 1 else 960 if n <= 4 else 640
        for cid in ids:
            self.cams[cid].preview_w = pw

    def _set_focus_mode(self, on):
        self.focus_mode = on
        self._relayout()

    def _double(self, cid):
        self._select(cid)
        self.btn_focus.setChecked(not self.btn_focus.isChecked())

    def _overlay(self):
        for t in self.tiles.values():
            t.grid = self.chk_grid.isChecked()
            t.cross = self.chk_cross.isChecked()
            t.update()

    # -- selection and the settings tabs ---------------------------------------------
    def _select(self, cid):
        self.sel = cid if cid in self.cams else None
        for k, t in self.tiles.items():
            t.selected = (k == self.sel)
            t.update()
        self._fill_profiles()
        self._build_tabs()
        if self.focus_mode:
            self._relayout()

    # -- settings sets (model mode) -------------------------------------------------
    def _fill_profiles(self):
        if self.model is None:
            return
        self.prof_combo.blockSignals(True)
        self.prof_combo.clear()
        if self.sel:
            for n in self.model.profiles_for(self.sel):
                self.prof_combo.addItem(n, n)
            cur = self.cur_profile.get(self.sel, "")
            i = self.prof_combo.findData(cur)
            if i >= 0:
                self.prof_combo.setCurrentIndex(i)
            elif self.prof_combo.count() == 0:
                self.prof_combo.addItem("(not saved yet)", "")
        self.prof_combo.blockSignals(False)

    def _profile_picked(self, i):
        cam = self.cams.get(self.sel)
        name = self.prof_combo.itemData(i)
        if cam is None or not name:
            return
        self.cur_profile[self.sel] = name
        bad = cam.apply_params(self.model.profile_params(name))
        try:
            cam.read_params()
        except Exception:
            pass
        if bad:
            self._say("Some values were not accepted: " + "; ".join(bad[:3]))
        self._build_tabs()

    def _ask_name(self, title, text, default=""):
        from PyQt6.QtWidgets import QInputDialog
        name, ok = QInputDialog.getText(self, title, text, text=default)
        name = (name or "").strip()
        return name if ok and name else ""

    def new_profile(self):
        cam = self.cams.get(self.sel)
        if cam is None:
            return
        name = self._ask_name("New settings set",
                              "Name for these settings (for example: "
                              "shiny part):")
        if not name:
            return
        if name in self.model.d["profiles"] and \
                self.model.get_profile(name)["cam_id"] != self.sel:
            self._say("This name is used by another camera.")
            return
        self.cur_profile[self.sel] = name
        self._save_to(name)
        self._fill_profiles()

    def delete_profile(self):
        name = self.cur_profile.get(self.sel, "")
        if not name:
            return
        if QMessageBox.question(
                self, "Delete settings set",
                f"Delete the settings set '{name}'?\nScenes that use it "
                "will use the first set of the camera.") \
                != QMessageBox.StandardButton.Yes:
            return
        self.model.delete_profile(name)
        self.cur_profile[self.sel] = self.model.default_profile(self.sel)
        self._fill_profiles()
        self._say(f"'{name}' deleted.", good=True)

    def _save_to(self, name):
        cam = self.cams.get(self.sel)
        try:
            params = cam.read_params()
        except Exception:
            params = dict(cam.params)
        params = {k: v for k, v in params.items() if k != "format"}
        self.model.save_profile(name, cam.info.id, params)
        self._say(f"Saved as '{name}' for {cam.info.name}.", good=True)

    def auto_calibrate(self):
        cam = self.cams.get(self.sel)
        if cam is None:
            return
        dlg = AutoCalDialog(self, cam)
        if dlg.exec() and dlg.chosen:
            for k, v in dlg.chosen["params"].items():
                cam.set(k, v)
            self._build_tabs()
            self._say("Auto calibration applied.  You can still change any "
                      "value, then save.", good=True)

    def reset_camera(self):
        cam = self.cams.get(self.sel)
        if cam is None:
            return
        if QMessageBox.question(
                self, "Reset camera",
                "Put every setting of this camera back to the factory "
                "values?\n\nSaved settings sets are not deleted.") \
                != QMessageBox.StandardButton.Yes:
            return
        ok, msg = cam.factory_reset()
        self._build_tabs()
        self._say(msg, good=ok)

    def open_drift(self):
        DriftDialog(self, self.model).exec()

    def _build_tabs(self):
        self.tabs.clear()
        self.param_rows = {}
        cam = self.cams.get(self.sel)
        has = cam is not None
        for b in (self.btn_save, self.btn_main, self.btn_reload,
                  self.btn_copy, self.btn_auto, self.btn_reset):
            b.setEnabled(has)
        if not has:
            self.sel_lbl.setText("No camera selected")
            self.metrics.setText("Click a picture to tune that camera.")
            self.hist.set_hist([])
            return
        self.sel_lbl.setText(f"{cam.info.name}\n{cam.info.detail}")
        try:
            vals = cam.read_params()
        except Exception:
            vals = dict(cam.params)
        for tab, specs in cam.specs().items():
            page = QWidget()
            page.setStyleSheet("background: transparent;")
            pl = QVBoxLayout(page)
            pl.setContentsMargins(8, 6, 8, 6)
            pl.setSpacing(2)
            for sp in specs:
                row = ParamRow(sp, self._on_change, self._on_action)
                pl.addWidget(row)
                n = sp.get("name")
                if n:
                    self.param_rows[n] = row
                    if n in vals:
                        row.set_value(vals[n])
            pl.addStretch()
            sa = QScrollArea()
            sa.setWidgetResizable(True)
            sa.setWidget(page)
            self.tabs.addTab(sa, tab)
        self._update_dependent()

    def _update_dependent(self):
        """Grey out what makes no sense with the current choice."""
        cam = self.cams.get(self.sel)
        if cam is None or cam.kind != "mv":
            return
        p = cam.params
        ae = bool(p.get("ae"))
        if "exposure_us" in self.param_rows:
            self.param_rows["exposure_us"].set_enabled_state(not ae)
        if "ae_target" in self.param_rows:
            self.param_rows["ae_target"].set_enabled_state(ae)
        hw = int(p.get("trigger_mode", 0)) == 2
        for n in ("ext_signal", "ext_shutter", "ext_delay_us",
                  "ext_jitter_us"):
            if n in self.param_rows:
                self.param_rows[n].set_enabled_state(hw)
        if "soft_trigger" in self.param_rows:
            self.param_rows["soft_trigger"].set_enabled_state(
                int(p.get("trigger_mode", 0)) == 1)
        man = int(p.get("strobe_mode", 0)) == 1
        for n in ("strobe_delay_us", "strobe_width_us"):
            if n in self.param_rows:
                self.param_rows[n].set_enabled_state(man)
        if "wb_auto" in self.param_rows and "rgb_gain" in self.param_rows:
            self.param_rows["rgb_gain"].set_enabled_state(
                not bool(p.get("wb_auto")))

    def _on_change(self, name, value):
        cam = self.cams.get(self.sel)
        if cam is None:
            return
        ok, msg = cam.set(name, value)
        if not ok:
            self._say(f"{name.replace('_', ' ')}: the camera did not accept "
                      f"it ({msg})")
            try:
                vals = cam.read_params()
                if name in vals and name in self.param_rows:
                    self.param_rows[name].set_value(vals[name])
            except Exception:
                pass
        else:
            self._say("")
            if name in ("trigger_mode", "ae", "strobe_mode", "wb_auto"):
                self._update_dependent()
            if name in ("ae", "wb_auto", "exposure_us"):
                try:
                    vals = cam.read_params()
                    for n in ("exposure_us", "analog_gain"):
                        if n in vals and n in self.param_rows and n != name:
                            self.param_rows[n].set_value(vals[n])
                except Exception:
                    pass

    def _on_action(self, name):
        cam = self.cams.get(self.sel)
        if cam is None:
            return False, "no camera"
        ok, msg = cam.action(name)
        if name == "wb_once" and ok and "rgb_gain" in self.param_rows:
            v = cam.params.get("rgb_gain")
            if v:
                self.param_rows["rgb_gain"].set_value(v)
        return ok, msg

    # -- saving ---------------------------------------------------------------------------
    def save_settings(self):
        cam = self.cams.get(self.sel)
        if cam is None:
            return
        if self.model is not None:
            name = self.cur_profile.get(self.sel, "")
            if not name:
                name = self._ask_name("Save settings set",
                                      "Name for these settings (for example: "
                                      "shiny part):", "Standard")
                if not name:
                    return
                self.cur_profile[self.sel] = name
            self._save_to(name)
            self._fill_profiles()
            return
        try:
            params = cam.read_params()
        except Exception:
            params = dict(cam.params)
        self.profiles["cameras"][cam.info.id] = {
            "name": cam.info.name, "kind": cam.kind, "params": _plain(params),
            "saved": datetime.datetime.now().isoformat(timespec="seconds")}
        save_profiles(self.profiles)
        self._say(f"Saved for {cam.info.name}.  The main program uses these "
                  "settings the next time it starts the camera.", good=True)

    def set_main(self):
        cam = self.cams.get(self.sel)
        if cam is None:
            return
        self.profiles["main"] = cam.info.id
        if cam.info.id not in self.profiles["cameras"]:
            self.save_settings()
        save_profiles(self.profiles)
        for cid, row in self.rows.items():
            if cid == cam.info.id:
                row.set_state("live  -  used for inspection", "main")
            elif cid in self.cams:
                row.set_state("live", "ok")
        self._say(f"{cam.info.name} will be used for inspection.", good=True)

    def reload_saved(self):
        cam = self.cams.get(self.sel)
        if cam is None:
            return
        if self.model is not None:
            name = self.cur_profile.get(cam.info.id, "")
            prof = self.model.profile_params(name) if name else {}
        else:
            prof = self.profiles["cameras"].get(cam.info.id, {}).get("params")
        if not prof:
            self._say("Nothing is saved for this camera yet.")
            return
        bad = cam.apply_params(prof)
        if bad:
            self._say("Not accepted: " + "; ".join(bad[:3]))
        try:
            cam.read_params()
        except Exception:
            pass
        self._build_tabs()

    def copy_to_others(self):
        src = self.cams.get(self.sel)
        if src is None:
            return
        others = [c for c in self.cams.values()
                  if c is not src and c.kind == src.kind]
        if not others:
            self._say("There is no other camera of the same kind open.")
            return
        keep_out = {"resolution_index", "roi"}
        params = {k: v for k, v in src.read_params().items()
                  if k not in keep_out}
        n_bad = 0
        for c in others:
            if c.kind == "mv":
                n_bad += len(mv_apply_all(c.h, params))
            else:
                for k, v in params.items():
                    c.set(k, v)
        self._say(f"Copied to {len(others)} camera(s)"
                  + (f", {n_bad} value(s) not accepted" if n_bad else "")
                  + ".", good=not n_bad)

    def snapshot(self):
        os.makedirs(SNAP_DIR, exist_ok=True)
        ids = [self.sel] if self.sel in self.cams else list(self.cams)
        n = 0
        for cid in ids:
            fr = self.cams[cid].latest_full()
            if fr is None:
                continue
            name = re.sub(r"[^\w\-]+", "_", self.cams[cid].info.name)
            path = os.path.join(SNAP_DIR, f"{name}_"
                                f"{datetime.datetime.now():%Y%m%d_%H%M%S}.png")
            cv2.imwrite(path, fr)
            n += 1
        self._say(f"{n} picture(s) saved in {SNAP_DIR}", good=bool(n))

    # -- periodic -------------------------------------------------------------------------
    def _tick(self):
        for t in self.tiles.values():
            if t.isVisible():
                t.poll()

    def _slow_tick(self):
        cam = self.cams.get(self.sel)
        if cam is None:
            return
        cam.analyze_now()
        s = cam.stats
        if s:
            self.hist.set_hist(s.get("hist", []))
            tips = []
            if s["clipped"] > 2.0:
                tips.append("too bright in places - lower the exposure")
            elif s["mean"] < 40:
                tips.append("dark - raise the exposure or the light")
            self.metrics.setText(
                f"Sharpness {s['focus']:.0f}   (higher = better focus)\n"
                f"Brightness {s['mean']:.0f} / 255   "
                f"too bright {s['clipped']:.1f}%   too dark {s['dark']:.1f}%"
                + (("\n" + "; ".join(tips)) if tips else ""))
        for cid, c in self.cams.items():
            if c.error and cid in self.rows:
                self.rows[cid].set_state(c.error, "bad")
            elif cid in self.rows and self.rows[cid].state.text().startswith(
                    ("camera error", "no picture")):
                self.rows[cid].set_state("live", "ok")

    def _say(self, text, good=False):
        self.msg.setStyleSheet(f"color: {P['sage'] if good else P['ochre']};"
                               " font-size: 9pt;")
        self.msg.setText(text)

    # -- shutdown -------------------------------------------------------------------------
    def shutdown(self):
        self._timer.stop()
        self._slow.stop()
        for c in list(self.cams.values()):
            try:
                pool_release(c)
            except Exception:
                pass
        self.cams.clear()
        self.tiles.clear()

    def closeEvent(self, e):
        self.shutdown()
        e.accept()


# ============================================================================
# MODEL CAMERAS  (what a model stores about its cameras: cameras.json)
#   - cameras are known by their SERIAL (id), not by port or order, so the
#     model still finds them when a cable moves to another port
#   - a "settings set" (profile) = one camera tuned for one object / view
#   - a scene uses one camera + one settings set
#   - the light reference of a scene is measured on its master picture
# ============================================================================
DRIFT_DEFAULT = {"enabled": True, "brightness_pct": 20.0,
                 "contrast_pct": 30.0, "sharpness_pct": 40.0}


def image_stats(img):
    """Brightness / contrast / sharpness of a picture (for the light check)."""
    g = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if g.shape[1] > 640:
        g = cv2.resize(g, (640, max(1, int(g.shape[0] * 640 / g.shape[1]))),
                       interpolation=cv2.INTER_AREA)
    hist = cv2.calcHist([g], [0], None, [32], [0, 256]).flatten()
    hist = (hist / (hist.sum() or 1.0)).tolist()
    return {"mean": float(g.mean()), "contrast": float(g.std()),
            "sharp": float(cv2.Laplacian(g, cv2.CV_64F).var()),
            "hist": hist,
            "when": datetime.datetime.now().isoformat(timespec="seconds")}


def drift_check(ref, now, th):
    """Compare the picture now with the reference.  Returns (changes, bad):
    changes = {"brightness": %, "contrast": %, "sharpness": %}
    bad = list of plain-English lines for what is over its limit."""
    def pct(a, b):
        return (a - b) / max(abs(b), 1.0) * 100.0
    ch = {"brightness": pct(now["mean"], ref["mean"]),
          "contrast": pct(now["contrast"], ref["contrast"]),
          "sharpness": pct(now["sharp"], ref["sharp"])}
    lim = {"brightness": float(th.get("brightness_pct", 20)),
           "contrast": float(th.get("contrast_pct", 30)),
           "sharpness": float(th.get("sharpness_pct", 40))}
    word = {"brightness": ("brighter", "darker"),
            "contrast": ("more contrast", "less contrast"),
            "sharpness": ("sharper", "less sharp")}
    bad = []
    for k, v in ch.items():
        if lim[k] > 0 and abs(v) > lim[k]:
            bad.append(f"The picture is {abs(v):.0f}% {word[k][0 if v > 0 else 1]}"
                       f" than when the model was made (limit {lim[k]:.0f}%).")
    return ch, bad


class ModelCameras:
    def __init__(self, path):
        self.path = path
        self.d = self._load()

    def _load(self):
        try:
            with open(self.path) as f:
                d = json.load(f)
        except (OSError, json.JSONDecodeError):
            d = {}
        d.setdefault("cameras", [])        # [{"id","name"}] in order
        d.setdefault("profiles", {})       # name -> {"cam_id","params"}
        d.setdefault("scenes", {})         # scene -> {"cam_id","profile","ref"}
        d.setdefault("drift", dict(DRIFT_DEFAULT))
        for k, v in DRIFT_DEFAULT.items():
            d["drift"].setdefault(k, v)
        return d

    def reload(self):
        self.d = self._load()

    def save(self):
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        with open(self.path, "w") as f:
            json.dump(_plain(self.d), f, indent=2)

    # cameras
    def has_cameras(self):
        return bool(self.d["cameras"])

    def slots(self):
        return [{"slot": i + 1, "id": c["id"], "name": c.get("name", c["id"])}
                for i, c in enumerate(self.d["cameras"])]

    def slot_of(self, cam_id):
        for s in self.slots():
            if s["id"] == cam_id:
                return s["slot"]
        return 0

    def cam_name(self, cam_id):
        for s in self.slots():
            if s["id"] == cam_id:
                return f"Camera {s['slot']}"
        return "Camera ?"

    def set_cameras(self, items):
        """items = [(id, name)] in the order the user chose."""
        self.d["cameras"] = [{"id": i, "name": n} for i, n in items]
        keep = {i for i, _ in items}
        self.d["profiles"] = {k: v for k, v in self.d["profiles"].items()
                              if v.get("cam_id") in keep}
        for sc in self.d["scenes"].values():
            if sc.get("cam_id") not in keep:
                sc["cam_id"] = items[0][0] if items else ""
                sc["profile"] = ""
        self.save()

    # settings sets
    def profiles_for(self, cam_id):
        return [k for k, v in self.d["profiles"].items()
                if v.get("cam_id") == cam_id]

    def get_profile(self, name):
        return self.d["profiles"].get(name)

    def profile_params(self, name):
        p = self.d["profiles"].get(name)
        return dict(p["params"]) if p else {}

    def save_profile(self, name, cam_id, params):
        self.d["profiles"][name] = {
            "cam_id": cam_id, "params": _plain(params),
            "saved": datetime.datetime.now().isoformat(timespec="seconds")}
        self.save()

    def delete_profile(self, name):
        self.d["profiles"].pop(name, None)
        for sc in self.d["scenes"].values():
            if sc.get("profile") == name:
                sc["profile"] = ""
        self.save()

    def default_profile(self, cam_id):
        names = self.profiles_for(cam_id)
        return names[0] if names else ""

    # scenes
    def scene(self, scene):
        sc = dict(self.d["scenes"].get(scene) or {})
        ids = [c["id"] for c in self.d["cameras"]]
        if sc.get("cam_id") not in ids:
            sc["cam_id"] = ids[0] if ids else ""
            sc["profile"] = ""
        if not sc.get("profile") or sc["profile"] not in self.d["profiles"]:
            sc["profile"] = self.default_profile(sc["cam_id"])
        return sc

    def set_scene(self, scene, cam_id, profile):
        sc = self.d["scenes"].setdefault(scene, {})
        sc["cam_id"], sc["profile"] = cam_id, profile
        self.save()

    def get_ref(self, scene):
        return (self.d["scenes"].get(scene) or {}).get("ref")

    def set_ref(self, scene, stats):
        sc = self.d["scenes"].setdefault(scene, {})
        sc["ref"] = _plain(stats)
        self.save()

    def forget_scene(self, scene):
        if self.d["scenes"].pop(scene, None) is not None:
            self.save()

    # light check limits
    def drift(self):
        return dict(self.d["drift"])

    def set_drift(self, **kw):
        self.d["drift"].update(kw)
        self.save()


# ============================================================================
# CAMERA HUB  -  the cameras of the running program
#   Same calls as the old single-camera feed (start, get_frame, get_raw,
#   qr_source, auto_balance, reset_camera, error ...), plus: several cameras
#   at once, one "active" camera, and per-scene settings sets.
# ============================================================================
BUILTIN_MV = {"ae": False, "exposure_us": 74005.0, "analog_gain": 4,
              "rgb_gain": [187, 100, 169], "saturation": 130,
              "sharpness": 30}


class CameraHub:
    STALE_S = 4.0

    def __init__(self, frame_scale=0.6):
        self.frame_scale = frame_scale
        self.keep_raw = True            # kept for the old callers
        self.store = None               # ModelCameras of the open model
        self.ui_call = None             # fn(ms, fn): run on the UI thread
        self._running = False
        self._by_id = {}
        self._infos = {}
        self._applied = {}
        self._active = None
        self._err = None
        self._stale = ""
        self._last_seq = {}
        self._last_move = {}
        self._reopen_t = {}
        self._watch = None
        self.missing = []

    # -- state -------------------------------------------------------------------
    @property
    def _cams(self):
        return list(self._by_id.values())

    @property
    def error(self):
        if self._err:
            return self._err
        cam = self._by_id.get(self._active)
        if cam is not None and cam.error and not cam.latest()[1] is not None:
            return "The camera has a problem.\n" + cam.error
        return self._stale or None

    def active_key(self):
        return (self._active, self._applied.get(self._active, ""))

    def cam(self, cam_id=None):
        return self._by_id.get(cam_id or self._active)

    def is_usb(self):
        c = self.cam()
        return c is not None and c.kind == "usb"

    # -- model ---------------------------------------------------------------------
    def set_model(self, store):
        same = (store is self.store) or (
            store is not None and self.store is not None
            and store.path == self.store.path)
        self.store = store
        if same:
            return
        if self._running:
            self.release()
            self.start()

    # -- start / stop ------------------------------------------------------------------
    def start(self):
        if self._running:
            return
        self._running = True
        self._err = None
        self._stale = ""
        self.missing = []
        try:
            self._open_all()
        except Exception as e:
            self._err = ("The camera could not be started.\n"
                         "Please check the camera cable and power.\n"
                         f"({e})")
            self._running = False
            return
        if not self._by_id:
            self._err = ("No camera found.\n"
                         "Please check the camera cable and power.")
            self._running = False
            return
        if self._watch is None:
            self._watch = QTimer()
            self._watch.setInterval(2000)
            self._watch.timeout.connect(self._check)
        self._watch.start()

    def release(self):
        self._running = False
        self._err, self._stale = None, ""
        if self._watch is not None:
            self._watch.stop()
        for c in list(self._by_id.values()):
            try:
                pool_release(c)
            except Exception:
                pass
        self._by_id = {}
        self._applied = {}
        self._active = None

    def _wanted(self):
        """[(cam_id, name)] the model wants, or None = one camera."""
        if self.store is not None and self.store.has_cameras():
            return [(s["id"], s["name"]) for s in self.store.slots()]
        return None

    def _open_all(self):
        found = {i.id: i for i in discover(True)}
        wanted = self._wanted()
        if wanted is None:
            gp = load_profiles()
            want_id = gp.get("main", "")
            info = found.get(want_id) or (next(iter(found.values()))
                                          if found else None)
            todo = [info] if info is not None else []
        else:
            todo = []
            for cid, name in wanted:
                if cid.startswith("demo:"):
                    todo.append(CamInfo(cid, "demo", name, "demo", None,
                                        int(cid.split(":")[1]) - 1))
                elif cid in found:
                    todo.append(found[cid])
                else:
                    self.missing.append((cid, name))
        for info in todo:
            self._open_one(info)
        if wanted is not None:
            first = next((c for c, _ in wanted if c in self._by_id), None)
        else:
            first = next(iter(self._by_id), None)
        self._active = first
        if self.missing and self._by_id:
            lost = ", ".join(
                f"{self.store.cam_name(c)}" if self.store else c
                for c, _ in self.missing)
            self._stale = f"{lost} not found.\nCheck the camera cable."

    def _open_one(self, info):
        cam = pool_open(info)
        self._infos[info.id] = info
        params = {}
        prof = ""
        if self.store is not None and self.store.has_cameras():
            prof = self.store.default_profile(info.id)
            params = self.store.profile_params(prof)
        else:
            params = (load_profiles()["cameras"].get(info.id, {})
                      .get("params") or {})
        if not params and cam.kind == "mv":
            params = dict(BUILTIN_MV)
        bad = cam.apply_params(params) if params else []
        if bad:
            print(f"[camera] {info.name}: not accepted: {bad}")
        if cam.kind == "mv" and not (
                self.store is not None and self.store.has_cameras()
                and prof):
            cam.action("wb_once")
        self._applied[info.id] = prof
        self._by_id[info.id] = cam
        self._last_seq[info.id] = -1
        self._last_move[info.id] = time.time()

    # -- watchdog: lost camera -> open it again -----------------------------------------
    def _check(self):
        if not self._running:
            return
        now = time.time()
        for cid, cam in list(self._by_id.items()):
            if int(cam.params.get("trigger_mode", 0)) != 0:
                self._last_move[cid] = now
                continue
            if cam.seq != self._last_seq.get(cid):
                self._last_seq[cid] = cam.seq
                self._last_move[cid] = now
                if cid == self._active:
                    self._stale = ""
                continue
            if now - self._last_move.get(cid, now) > self.STALE_S:
                if cid == self._active:
                    self._stale = ("No picture from the camera.\n"
                                   "Please check the camera cable and power.")
                if now - self._reopen_t.get(cid, 0) > 5.0:
                    self._reopen_t[cid] = now
                    try:
                        prof = self._applied.get(cid, "")
                        pool_close(cam)
                        self._by_id.pop(cid, None)
                        self._open_one(self._infos[cid])
                        self._applied[cid] = ""
                        self.use(cid, prof)
                    except Exception:
                        pass

    # -- choosing the camera / settings set -------------------------------------------------
    def _ui(self, fn):
        """Run fn on the UI thread (a USB camera must be touched there)."""
        if threading.current_thread() is threading.main_thread() \
                or self.ui_call is None:
            return fn()
        done = threading.Event()
        box = []

        def go():
            try:
                box.append(fn())
            finally:
                done.set()
        self.ui_call(0, go)
        done.wait(5.0)
        return box[0] if box else None

    def use(self, cam_id, profile=""):
        """Make cam_id the active camera and put its settings set on it."""
        cam = self._by_id.get(cam_id)
        if cam is None:
            return False
        if self._applied.get(cam_id, "") != profile and self.store is not None:
            params = self.store.profile_params(profile) if profile else {}
            if params:
                self._ui(lambda: cam.apply_params(params))
            self._applied[cam_id] = profile
        self._active = cam_id
        return True

    def capture(self, cam_id, profile="", timeout=2.5, frames=3):
        """Worker-thread call: switch to the camera / settings set and wait for
        fresh pictures.  Returns a picture (same size as get_frame) or None."""
        cam = self._by_id.get(cam_id)
        if cam is None:
            return None
        changed = self._applied.get(cam_id, "") != profile
        self.use(cam_id, profile)
        if int(cam.params.get("trigger_mode", 0)) != 0:
            return self._scaled(cam.latest_full())
        seq0 = cam.seq
        need = frames if changed else 1
        t = time.time()
        while time.time() - t < timeout and cam.seq < seq0 + need:
            time.sleep(0.01)
        return self._scaled(cam.latest_full())

    # -- the old CameraFeed calls ---------------------------------------------------------
    def _to_bgr(self, f):
        return cv2.cvtColor(f, cv2.COLOR_GRAY2BGR) if f.ndim == 2 else f

    def _scaled(self, f):
        """Same picture size the program has always worked with."""
        if f is None:
            return None
        f = self._to_bgr(f)
        if self.frame_scale and self.frame_scale != 1:
            f = cv2.resize(f, None, fx=self.frame_scale, fy=self.frame_scale)
        return f

    def get_frame(self):
        cam = self._by_id.get(self._active)
        return self._scaled(cam.latest_full() if cam is not None else None)

    def raw_seq(self):
        cam = self._by_id.get(self._active)
        return cam.seq if cam is not None else 0

    def get_raw(self):
        cam = self._by_id.get(self._active)
        if cam is None:
            return 0, None
        seq = cam.seq
        f = cam.latest_full()
        return seq, (None if f is None else self._to_bgr(f))

    def qr_source(self):
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
        cam = self._by_id.get(self._active)
        if cam is None:
            return False
        try:
            if cam.kind == "mv":
                return bool(cam.action("wb_once")[0])
            return bool(cam.set("wb_auto", True)[0])
        except Exception:
            return False

    def reset_camera(self, cb=None):
        """Put the active camera back to its factory settings."""
        cam = self._by_id.get(self._active)
        if cam is None:
            if cb:
                cb(False, "camera is not running")
            return
        ok, msg = self._ui(cam.factory_reset) or (False, "not done")
        self._applied[self._active] = ""
        if cb:
            cb(ok, msg)


# ============================================================================
# AUTO CALIBRATION  -  try many exposure / gain values, score each picture,
# keep the 5 best.  No thread: a timer steps through the candidates while the
# camera keeps streaming.
# ============================================================================
def score_picture(img, target, centre_only):
    g = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if centre_only:
        h, w = g.shape[:2]
        g = g[h // 4: h - h // 4, w // 4: w - w // 4]
    if g.shape[1] > 480:
        g = cv2.resize(g, (480, max(1, int(g.shape[0] * 480 / g.shape[1]))),
                       interpolation=cv2.INTER_AREA)
    m = float(g.mean())
    return {"mean": m, "contrast": float(g.std()),
            "sharp": float(cv2.Laplacian(g, cv2.CV_64F).var()) / max(m, 8.0) ** 2,
            "clip": 100.0 * float((g >= 250).sum()) / g.size,
            "dark": 100.0 * float((g <= 5).sum()) / g.size}


def rank_results(rows, target):
    """rows: [{"params","m"(stats),"gain_n","exp_n","thumb"}] -> sorted rows
    with a 0-100 'score'."""
    if not rows:
        return []
    smax = max(r["m"]["sharp"] for r in rows) or 1.0
    for r in rows:
        m = r["m"]
        exp_s = 1.0 - min(1.0, abs(m["mean"] - target) / max(target, 1.0))
        con_s = min(1.0, m["contrast"] / 60.0)
        sh_s = m["sharp"] / smax
        sc = (40 * exp_s + 20 * con_s + 20 * sh_s + 20
              - 30 * min(1.0, m["clip"] / 5.0)
              - 15 * min(1.0, m["dark"] / 20.0)
              - 8 * r["gain_n"] - 6 * r["exp_n"])
        r["score"] = max(0.0, min(100.0, sc))
    rows.sort(key=lambda r: -r["score"])
    return rows


def pick_top(rows, n=5):
    """Best n that are really different from each other."""
    out = []
    for r in rows:
        p = r["params"]
        same = False
        for o in out:
            q = o["params"]
            diff = False
            for k in p:
                a, b = p[k], q.get(k)
                if isinstance(a, (int, float)) and isinstance(b, (int, float)):
                    if k.startswith("exposure"):
                        if max(a, b) / max(min(a, b), 1e-9) >= 1.4:
                            diff = True
                    elif a != b:
                        diff = True
                elif a != b:
                    diff = True
            if not diff:
                same = True
                break
        if not same:
            out.append(r)
        if len(out) >= n:
            break
    return out


class AutoCalibrator(QObject):
    progress = pyqtSignal(int, int)
    finished = pyqtSignal(list)            # the best results
    failed = pyqtSignal(str)

    SKIP, TAKE, TIMEOUT = 3, 2, 2.5

    def __init__(self, cam, target=125, centre_only=True, parent=None):
        super().__init__(parent)
        self.cam, self.target, self.centre_only = cam, target, centre_only
        self.rows = []
        self.cands = []
        self.i = 0
        self.state = "idle"
        self._timer = QTimer(self)
        self._timer.setInterval(30)
        self._timer.timeout.connect(self._step)
        self._orig = {}

    def start(self):
        prep, cands, names = self.cam.autocal_plan()
        if not cands:
            self.failed.emit("This camera cannot be set automatically.")
            return
        self.cands, self.names = cands, names
        try:
            self._orig = dict(self.cam.read_params())
        except Exception:
            self._orig = dict(self.cam.params)
        for k, v in prep.items():
            self.cam.set(k, v)
        self.i = 0
        self.rows = []
        self.state = "apply"
        self._timer.start()
        self.progress.emit(0, len(cands))

    def cancel(self):
        if self.state != "idle":
            self._timer.stop()
            self.state = "idle"
            self._restore()

    def _restore(self):
        keys = set(self.names) | {"ae", "trigger_mode", "exposure_auto"}
        back = dict(self.cam.autocal_defaults())
        back.update({k: v for k, v in self._orig.items() if k in keys})
        for k, v in back.items():
            self.cam.set(k, v)

    def _span(self, name):
        vals = [c[name] for c in self.cands if name in c]
        return (min(vals), max(vals)) if vals else (0, 1)

    def _step(self):
        cam = self.cam
        if self.state == "apply":
            if self.i >= len(self.cands):
                return self._done()
            for k, v in self.cands[self.i].items():
                cam.set(k, v)
            self._seq0 = cam.seq
            self._t0 = time.time()
            self._got = []
            self._last = cam.seq
            self.state = "settle"
        elif self.state == "settle":
            if cam.seq >= self._seq0 + self.SKIP:
                self.state = "take"
                self._last = cam.seq
            elif time.time() - self._t0 > self.TIMEOUT:
                self._next(None)
        elif self.state == "take":
            if cam.seq != self._last:
                self._last = cam.seq
                _, img = cam.latest()
                if img is not None:
                    self._got.append(score_picture(img, self.target,
                                                   self.centre_only))
                    self._thumb = img
                if len(self._got) >= self.TAKE:
                    self._next(self._got)
            elif time.time() - self._t0 > self.TIMEOUT * 2:
                self._next(None)

    def _next(self, got):
        if got:
            m = {k: sum(g[k] for g in got) / len(got) for k in got[0]}
            c = self.cands[self.i]
            ge = self._span("analog_gain")
            xe = self._span("exposure_us") if "exposure_us" in c \
                else self._span("exposure_time")
            ex = c.get("exposure_us", c.get("exposure_time", xe[0]))
            gn = ((c.get("analog_gain", ge[0]) - ge[0])
                  / ((ge[1] - ge[0]) or 1))
            en = ((np.log(max(ex, 1e-9)) - np.log(max(xe[0], 1e-9)))
                  / ((np.log(max(xe[1], 1e-9)) - np.log(max(xe[0], 1e-9)))
                     or 1))
            th = self._thumb
            if th.shape[1] > 320:
                th = cv2.resize(th, (320, int(th.shape[0] * 320 / th.shape[1])))
            self.rows.append({"params": dict(c), "m": m,
                              "gain_n": float(gn), "exp_n": float(en),
                              "thumb": th.copy()})
        self.i += 1
        self.progress.emit(self.i, len(self.cands))
        self.state = "apply"

    def _done(self):
        self._timer.stop()
        self.state = "idle"
        self._restore()
        means = [r["m"]["mean"] for r in self.rows]
        if len(means) >= 3 and max(means) - min(means) < 4.0:
            self.failed.emit("The picture did not change when the settings "
                             "changed.  This camera or its driver does not "
                             "accept exposure control from the program.  "
                             "Set it by hand in the Image tab.")
            return
        top = pick_top(rank_results(self.rows, self.target), 5)
        if not top:
            self.failed.emit("No picture came from the camera.")
        else:
            self.finished.emit(top)



class DriftDialog(QDialog):
    """How much the light may change before the operator gets a warning."""

    def __init__(self, parent, model):
        super().__init__(parent)
        self.model = model
        self.setWindowTitle("Light check")
        self.setStyleSheet(QSS)
        d = model.drift()
        lay = QVBoxLayout(self)
        lay.setSpacing(8)
        info = QLabel("When the picture is much brighter, darker or less "
                      "sharp than it was when the model was made, the "
                      "operator gets a warning.  Set how much change is "
                      "allowed.")
        info.setWordWrap(True)
        info.setMinimumWidth(420)
        lay.addWidget(info)
        self.on = QCheckBox("Warn when the light changes")
        self.on.setChecked(bool(d["enabled"]))
        lay.addWidget(self.on)
        self.sp = {}
        for key, label in (("brightness_pct", "Brightness may change by"),
                           ("contrast_pct", "Contrast may change by"),
                           ("sharpness_pct", "Sharpness may change by")):
            row = QHBoxLayout()
            row.addWidget(QLabel(label))
            sp = QSpinBox()
            sp.setRange(1, 500)
            sp.setSuffix(" %")
            sp.setValue(int(d[key]))
            self.sp[key] = sp
            row.addStretch()
            row.addWidget(sp)
            lay.addLayout(row)
        row = QHBoxLayout()
        row.addStretch()
        b = QPushButton("Cancel")
        b.clicked.connect(self.reject)
        row.addWidget(b)
        b = QPushButton("Save")
        b.setObjectName("primary")
        b.clicked.connect(self._save)
        row.addWidget(b)
        lay.addLayout(row)

    def _save(self):
        self.model.set_drift(enabled=self.on.isChecked(),
                             **{k: float(s.value())
                                for k, s in self.sp.items()})
        self.accept()


class AutoCalDialog(QDialog):
    """Auto calibration: measure many exposure / gain values, show the 5
    best pictures, the user picks one."""

    def __init__(self, parent, cam):
        super().__init__(parent)
        self.cam, self.chosen, self.cal = cam, None, None
        self.setWindowTitle("Auto calibrate")
        self.setStyleSheet(QSS)
        self.setModal(True)
        self.lay = QVBoxLayout(self)
        self.lay.setSpacing(10)
        self._intro()

    def _clear(self):
        while self.lay.count():
            it = self.lay.takeAt(0)
            w = it.widget()
            if w is not None:
                w.hide()
                w.setParent(None)
                w.deleteLater()
            elif it.layout() is not None:
                while it.layout().count():
                    x = it.layout().takeAt(0)
                    xw = x.widget()
                    if xw is not None:
                        xw.hide()
                        xw.setParent(None)
                        xw.deleteLater()

    def _intro(self):
        self._clear()
        t = QLabel("Auto calibrate")
        t.setStyleSheet("font-size: 14pt; font-weight: 700;")
        self.lay.addWidget(t)
        info = QLabel("Put the part in front of the camera, as it will be "
                      "during the test.  The program tries many exposure "
                      "and gain values, then shows the 5 best pictures.  "
                      "You pick one and can still change it by hand.")
        info.setWordWrap(True)
        info.setMinimumWidth(460)
        self.lay.addWidget(info)
        row = QHBoxLayout()
        row.addWidget(QLabel("Wanted brightness (0 dark - 255 bright)"))
        self.target = QSpinBox()
        self.target.setRange(40, 220)
        self.target.setValue(125)
        row.addStretch()
        row.addWidget(self.target)
        self.lay.addLayout(row)
        self.centre = QCheckBox("Judge only the centre of the picture "
                                "(where the part is)")
        self.centre.setChecked(True)
        self.lay.addWidget(self.centre)
        row = QHBoxLayout()
        row.addStretch()
        b = QPushButton("Cancel")
        b.clicked.connect(self.reject)
        row.addWidget(b)
        b = QPushButton("Start")
        b.setObjectName("primary")
        b.clicked.connect(self._start)
        row.addWidget(b)
        self.lay.addLayout(row)

    def _start(self):
        tgt, ctr = self.target.value(), self.centre.isChecked()
        self._clear()
        self.lbl = QLabel("Trying settings...  Keep the part still.")
        self.lay.addWidget(self.lbl)
        self.bar = QProgressBar()
        self.lay.addWidget(self.bar)
        b = QPushButton("Stop")
        b.clicked.connect(self._stop)
        self.lay.addWidget(b)
        self.cal = AutoCalibrator(self.cam, tgt, ctr, self)
        self.cal.progress.connect(self._prog)
        self.cal.finished.connect(self._results)
        self.cal.failed.connect(self._failed)
        self.cal.start()

    def _prog(self, i, n):
        self.bar.setRange(0, n)
        self.bar.setValue(i)

    def _stop(self):
        if self.cal:
            self.cal.cancel()
        self.reject()

    def _failed(self, msg):
        self._clear()
        l = QLabel(msg)
        l.setWordWrap(True)
        self.lay.addWidget(l)
        b = QPushButton("Close")
        b.clicked.connect(self.reject)
        self.lay.addWidget(b)

    def _results(self, top):
        self._clear()
        t = QLabel("The 5 best results - pick one")
        t.setStyleSheet("font-size: 13pt; font-weight: 700;")
        self.lay.addWidget(t)
        grid = QGridLayout()
        grid.setSpacing(10)
        for k, r in enumerate(top):
            card = QFrame()
            card.setObjectName("card")
            cl = QVBoxLayout(card)
            pic = QLabel()
            th = np.ascontiguousarray(r["thumb"])
            h, w = th.shape[:2]
            fmt = (QImage.Format.Format_Grayscale8 if th.ndim == 2
                   else QImage.Format.Format_BGR888)
            qi = QImage(th.data, w, h, th.strides[0], fmt).copy()
            pic.setPixmap(QPixmap.fromImage(qi).scaledToWidth(
                230, Qt.TransformationMode.SmoothTransformation))
            cl.addWidget(pic)
            head = QLabel(f"#{k + 1}    score {r['score']:.0f} / 100")
            head.setStyleSheet("font-weight: 700;")
            cl.addWidget(head)
            cl.addWidget(QLabel(self._describe(r["params"])))
            m = r["m"]
            det = QLabel(f"brightness {m['mean']:.0f}   "
                         f"too bright {m['clip']:.1f}%")
            det.setStyleSheet(f"color: {P['muted']}; font-size: 8pt;")
            cl.addWidget(det)
            b = QPushButton("Use this")
            b.setObjectName("primary")
            b.clicked.connect(lambda _=False, rr=r: self._pick(rr))
            cl.addWidget(b)
            grid.addWidget(card, k // 3, k % 3)
        self.lay.addLayout(grid)
        row = QHBoxLayout()
        row.addStretch()
        b = QPushButton("Try again")
        b.clicked.connect(self._intro)
        row.addWidget(b)
        b = QPushButton("Cancel")
        b.clicked.connect(self.reject)
        row.addWidget(b)
        self.lay.addLayout(row)
        self.adjustSize()

    @staticmethod
    def _describe(p):
        out = []
        for k, v in p.items():
            if k == "exposure_us":
                out.append(f"Exposure {v / 1000:.1f} ms")
            elif k == "exposure_time":
                out.append(f"Exposure {v:.1f} ms")
            elif k == "analog_gain":
                out.append(f"Gain {v}")
            elif k == "iso":
                out.append(f"ISO {v}")
            elif k == "exposure_comp":
                out.append(f"Brightness correction {v:+.1f} EV")
            else:
                out.append(f"{k} {v}")
        return "   ".join(out)

    def _pick(self, r):
        self.chosen = r
        self.accept()


class CameraPicker(QWidget):
    """Step 1 of the model's camera setup: how many cameras the system has,
    a live picture of each, and the order (Camera 1, Camera 2, ...)."""
    chosen = pyqtSignal(list)         # [(id, name)] in order
    cancelled = pyqtSignal()

    def __init__(self, store, parent=None, demo=False):
        super().__init__(parent)
        self.setStyleSheet(QSS)
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.store = store
        self.cards = {}          # id -> dict(widgets, cam, tile)
        self.order = []          # ids in the chosen order
        self.infos = []
        self._demo_n = 0
        root = QVBoxLayout(self)
        root.setContentsMargins(14, 12, 14, 12)
        root.setSpacing(10)
        top = QHBoxLayout()
        t = QLabel("Choose the cameras")
        t.setStyleSheet("font-size: 17pt; font-weight: 700;")
        top.addWidget(t)
        self.count = QLabel("")
        self.count.setStyleSheet(f"color: {P['muted']};")
        top.addWidget(self.count)
        top.addStretch()
        b = QPushButton("Scan again")
        b.clicked.connect(lambda: self.rescan(True))
        top.addWidget(b)
        self.btn_demo = QPushButton("Add a demo camera")
        self.btn_demo.clicked.connect(self.add_demo)
        top.addWidget(self.btn_demo)
        root.addLayout(top)
        tip = QLabel("Tick the cameras this model uses.  The order you tick "
                     "them is Camera 1, Camera 2, ...  The program "
                     "remembers each camera by its serial number, so the "
                     "numbers stay the same even if a cable is moved to "
                     "another port.")
        tip.setWordWrap(True)
        tip.setStyleSheet(f"color: {P['muted']};")
        root.addWidget(tip)
        self.warn = QLabel("")
        self.warn.setWordWrap(True)
        self.warn.setStyleSheet(f"color: {P['rust']};")
        root.addWidget(self.warn)
        sc = QScrollArea()
        sc.setWidgetResizable(True)
        self.body = QWidget()
        self.body.setStyleSheet("background: transparent;")
        self.grid = QGridLayout(self.body)
        self.grid.setSpacing(12)
        sc.setWidget(self.body)
        root.addWidget(sc, stretch=1)
        bot = QHBoxLayout()
        self.msg = QLabel("")
        self.msg.setStyleSheet(f"color: {P['ochre']};")
        bot.addWidget(self.msg)
        bot.addStretch()
        self.btn_skip = QPushButton("Skip - no camera setup")
        self.btn_skip.clicked.connect(self.cancelled.emit)
        bot.addWidget(self.btn_skip)
        self.btn_next = QPushButton("Next: tune the cameras")
        self.btn_next.setObjectName("primary")
        self.btn_next.clicked.connect(self._next)
        bot.addWidget(self.btn_next)
        root.addLayout(bot)
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)
        self._timer.start(40)
        self.setMinimumSize(900, 560)
        QTimer.singleShot(50, self.rescan)
        if demo:
            QTimer.singleShot(150, self.add_demo)

    # -- find cameras ------------------------------------------------------------
    def rescan(self, flush=False):
        self._close_cards()
        if flush:
            pool_flush()
        found = discover(True)
        saved = self.store.slots() if self.store else []
        have = {i.id for i in found}
        self.infos = list(found) + [i for i in self.infos
                                    if i.kind == "demo"]
        for s in saved:
            if s["id"].startswith("demo:") and s["id"] not in \
                    {i.id for i in self.infos}:
                n = int(s["id"].split(":")[1])
                self._demo_n = max(self._demo_n, n)
                self.infos.append(CamInfo(s["id"], "demo", f"Demo camera {n}",
                                          "pretend camera", None, n - 1))
        gone = [s for s in saved if s["id"] not in have
                and not s["id"].startswith("demo:")]
        self.warn.setText("")
        if gone:
            self.warn.setText("Not connected now: " + ",  ".join(
                f"Camera {s['slot']} ({s['name']}, {s['id'].split(':', 1)[1]})"
                for s in gone))
        self._fill()
        want = [s["id"] for s in saved if s["id"] in
                {i.id for i in self.infos}]
        for cid in want:
            if cid in self.cards:
                self.cards[cid]["chk"].setChecked(True)

    def add_demo(self):
        self._demo_n += 1
        n = self._demo_n
        self.infos.append(CamInfo(f"demo:{n}", "demo", f"Demo camera {n}",
                                  "pretend camera - no hardware", None,
                                  n - 1))
        keep = list(self.order)
        self._close_cards()
        self._fill()
        for cid in keep:
            if cid in self.cards:
                self.cards[cid]["chk"].setChecked(True)

    def _close_cards(self):
        for c in self.cards.values():
            try:
                pool_release(c["cam"])
            except Exception:
                pass
        self.cards = {}
        self.order = []
        while self.grid.count():
            it = self.grid.takeAt(0)
            w = it.widget()
            if w is not None:
                w.hide()
                w.setParent(None)
                w.deleteLater()

    def _fill(self):
        n = len(self.infos)
        self.count.setText(f"{n} camera(s) found in this system" if n else
                           "no camera found - check the cable and power")
        for k, info in enumerate(self.infos):
            card = QFrame()
            card.setObjectName("card")
            cl = QVBoxLayout(card)
            cl.setContentsMargins(8, 8, 8, 8)
            cam = None
            tile = None
            err = ""
            try:
                cam = pool_open(info)
                cam.preview_w = 480
                if cam.kind == "mv":
                    cam.set("trigger_mode", 0)       # a live picture
                tile = StreamTile(cam)
                tile.setMinimumSize(300, 190)
                cl.addWidget(tile)
            except Exception as e:
                err = str(e)
                lab = QLabel(f"Could not open:\n{err}")
                lab.setWordWrap(True)
                lab.setMinimumSize(300, 190)
                lab.setAlignment(Qt.AlignmentFlag.AlignCenter)
                lab.setStyleSheet(f"color: {P['rust']};")
                cl.addWidget(lab)
            row = QHBoxLayout()
            chk = QCheckBox(info.name)
            chk.setStyleSheet("font-weight: 600;")
            chk.setEnabled(not err)
            row.addWidget(chk, stretch=1)
            badge = QLabel("")
            badge.setStyleSheet(f"color: {P['sage']}; font-weight: 700;")
            row.addWidget(badge)
            left = QPushButton("<")
            right = QPushButton(">")
            for b in (left, right):
                b.setFixedWidth(30)
                b.setVisible(False)
            row.addWidget(left)
            row.addWidget(right)
            cl.addLayout(row)
            d = QLabel(info.detail)
            d.setStyleSheet(f"color: {P['muted']}; font-size: 8pt;")
            d.setWordWrap(True)
            cl.addWidget(d)
            self.grid.addWidget(card, k // 2, k % 2)
            c = dict(chk=chk, badge=badge, left=left, right=right,
                     cam=cam, tile=tile, info=info)
            self.cards[info.id] = c
            chk.toggled.connect(lambda on, cid=info.id: self._ticked(cid, on))
            left.clicked.connect(lambda _=False, cid=info.id:
                                 self._move(cid, -1))
            right.clicked.connect(lambda _=False, cid=info.id:
                                  self._move(cid, +1))
        self._badges()

    # -- order -------------------------------------------------------------------
    def _ticked(self, cid, on):
        if on and cid not in self.order:
            self.order.append(cid)
        if not on and cid in self.order:
            self.order.remove(cid)
        self._badges()

    def _move(self, cid, d):
        i = self.order.index(cid)
        j = i + d
        if 0 <= j < len(self.order):
            self.order[i], self.order[j] = self.order[j], self.order[i]
            self._badges()

    def _badges(self):
        for cid, c in self.cards.items():
            if cid in self.order:
                i = self.order.index(cid)
                c["badge"].setText(f"Camera {i + 1}")
                c["left"].setVisible(len(self.order) > 1)
                c["right"].setVisible(len(self.order) > 1)
                c["left"].setEnabled(i > 0)
                c["right"].setEnabled(i < len(self.order) - 1)
                if c["tile"] is not None:
                    c["tile"].selected = True
            else:
                c["badge"].setText("")
                c["left"].setVisible(False)
                c["right"].setVisible(False)
                if c["tile"] is not None:
                    c["tile"].selected = False
        self.msg.setText(f"{len(self.order)} selected"
                         if self.order else "Tick at least one camera.")

    def _tick(self):
        for c in self.cards.values():
            t = c["tile"]
            if t is not None and t.isVisible():
                t.poll()

    # -- done ------------------------------------------------------------------------
    def _next(self):
        if not self.order:
            self.msg.setText("Tick at least one camera.")
            return
        out = [(cid, self.cards[cid]["info"].name) for cid in self.order]
        self.shutdown()
        self.chosen.emit(out)

    def shutdown(self):
        self._timer.stop()
        self._close_cards()


class ModelCameraWizard(QWidget):
    """Pick the cameras (with previews and order), then tune each one.
    `done` is emitted when the user finishes or skips."""
    done = pyqtSignal()

    def __init__(self, store, start="auto", demo=False, parent=None):
        super().__init__(parent)
        self.setStyleSheet(QSS)
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.store = store
        self._demo = demo
        self.lay = QVBoxLayout(self)
        self.lay.setContentsMargins(0, 0, 0, 0)
        self.page = None
        if start == "picker" or (start == "auto"
                                 and not store.has_cameras()):
            self._show_picker()
        else:
            self._show_panel()

    def _swap(self, w):
        if self.page is not None:
            try:
                self.page.shutdown()
            except Exception:
                pass
            self.page.hide()
            self.page.setParent(None)
            self.page.deleteLater()
        self.page = w
        self.lay.addWidget(w)

    def _show_picker(self):
        p = CameraPicker(self.store, demo=self._demo and
                         not self.store.has_cameras())
        p.chosen.connect(self._picked)
        p.cancelled.connect(self.done.emit)
        self._swap(p)

    def _picked(self, items):
        self.store.set_cameras(items)
        self._show_panel()

    def _show_panel(self):
        p = CalibrationPanel(embedded=False, model=self.store)
        p.done.connect(self.done.emit)
        p.change_cameras.connect(self._show_picker)
        self._swap(p)

    def shutdown(self):
        if self.page is not None:
            try:
                self.page.shutdown()
            except Exception:
                pass


def _plain(v):
    """JSON-safe copy of a settings dict."""
    if isinstance(v, dict):
        return {k: _plain(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_plain(x) for x in v]
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return float(v)
    return v


def main():
    app = QApplication(sys.argv)
    w = CalibrationPanel(demo="--demo" in sys.argv)
    w.setWindowTitle("Camera Setup")
    w.resize(1500, 860)
    w.show()
    code = app.exec()
    w.shutdown()
    close_all()
    sys.exit(code)


if __name__ == "__main__":
    main()