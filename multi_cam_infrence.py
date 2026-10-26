# coding=utf-8
import cv2
import numpy as np
import mvsdk
import platform
import time
import os
import shutil
import re
import subprocess

_V4L2_WARNED = False


def _v4l2(args):
    """Run 'v4l2-ctl <args>' only when the tool exists (Linux, package
    v4l-utils).  Missing tool = skip quietly instead of 'sh: v4l2-ctl: not
    found' on every call.  Install it with:  sudo apt install v4l-utils"""
    global _V4L2_WARNED
    if platform.system() == "Windows":
        return False
    if shutil.which("v4l2-ctl") is None:
        if not _V4L2_WARNED:
            _V4L2_WARNED = True
            print("[camera] v4l2-ctl not installed - USB camera exposure/"
                  "gain/white-balance via v4l2 skipped "
                  "(sudo apt install v4l-utils)")
        return False
    os.system(f"v4l2-ctl {args}")
    return True
def get_camera_info(no=4):
    """Get camera information."""
    devs = mvsdk.CameraEnumerateDevice()
    
    
    index = 0
    usb_cam = []
    max_devices = 10

    while index < max_devices:
        cap = cv2.VideoCapture(index)
        if not cap.isOpened():
            print(f"Checking OpenCV camera index  fail {index}...")
        if cap.read()[0]:
            frame= cap.read()[1]
            if frame is not None:
                cap.release()
                print(f"Camera found at index {index}")
                usb_cam.append(index)
            cap.release()


        index += 1
    if not devs and  len(usb_cam)==0:
        print("No cameras found.")
        return []
    for i, dev in enumerate(devs):
        print(f"Camera {i}: {dev}")
    totcam=devs+usb_cam
    return totcam[:no]

class CameraWrapper:
    def __init__(self,dev_info,exposure=6000, gain=15, rGain=275, gGain=167, bGain=100):
        # Open & configure camera
        self.exposure=exposure
        self.gain=gain
        self.rgain=rGain
        self.ggain=gGain
        self.bgain=bGain 
        self.cam_port='None'
        self.camtype="mindvision"
        self.H_W={}
        self.height=2160
        self.width=3840

        if str(dev_info).isdigit():
            self.camtype='usbcam'
            self.cam_port=dev_info
            self._open_usb()
            self._snapshot_usb()      # the camera's own settings, for Reset

            print(f"Using OpenCV camera index {self.h}")

        
        
        else:
            print(f"Using MVS camera info: {dev_info}")
            self.h = mvsdk.CameraInit(dev_info, -1, -1)
            self.cap = mvsdk.CameraGetCapability(self.h)
            self.mono = (self.cap.sIspCapacity.bMonoSensor != 0)
            out_fmt = (mvsdk.CAMERA_MEDIA_TYPE_MONO8 
                    if self.mono else 
                    mvsdk.CAMERA_MEDIA_TYPE_BGR8)
            mvsdk.CameraSetIspOutFormat(self.h, out_fmt)
            mvsdk.CameraSetTriggerMode(self.h, 0)
            mvsdk.CameraSetAeState(self.h, 0)
            mvsdk.CameraSetExposureTime(self.h, self.exposure)
            self._set_color_gain(self.rgain, self.ggain, self.bgain)
            mvsdk.CameraSetAnalogGain(self.h,self.gain)
            mvsdk.CameraSetOnceWB(self.h)
            mvsdk.CameraPlay(self.h)


            # Pre-allocate the shared frame buffer once
            fb_size = (self.cap.sResolutionRange.iWidthMax *
                    self.cap.sResolutionRange.iHeightMax *
                    (1 if self.mono else 3))
            self.pFrameBuffer = mvsdk.CameraAlignMalloc(fb_size, 16)

    def grab(self, timeout=200):
        """Return a valid NumPy frame, or None on timeout/error."""
        if self.camtype=='usbcam': 
            # OpenCV camera index, use OpenCV to grab frame
            try:
                ret, frame = self.h.read()
                
                if not ret:
                    print(f"[Camera {self.h}] Error: Could not read frame.")
                    self.h.release()
                    self._open_usb()
                else:
                    frame=cv2.resize(frame,(self.width,self.height))
            except cv2.error as e:
                print("5454545455555555555555555555")

                print(f"[Camera {self.h}] OpenCV error: {e}")
                return None
            return frame
        else:
            try:
                pRaw, head = mvsdk.CameraGetImageBuffer(self.h, timeout)
                mvsdk.CameraImageProcess(self.h, pRaw, self.pFrameBuffer, head)
                mvsdk.CameraReleaseImageBuffer(self.h, pRaw)
                # Windows BMP flip
                if platform.system() == "Windows":
                    mvsdk.CameraFlipFrameBuffer(self.pFrameBuffer, head, 1)

                arr = (mvsdk.c_ubyte * head.uBytes).from_address(self.pFrameBuffer)
                frame = np.frombuffer(arr, dtype=np.uint8)
                frame = frame.reshape((
                    head.iHeight, head.iWidth,
                    1 if head.uiMediaType == mvsdk.CAMERA_MEDIA_TYPE_MONO8 else 3
                ))
                frame=cv2.resize(frame,(self.width,self.height))

                return frame
            except mvsdk.CameraException as e:
                print("cvgjsfhvjgsfjvsdfjhvbjsfdv")
                if e.error_code != mvsdk.CAMERA_STATUS_TIME_OUT:
                    print(f"[Camera {self.h}] Error {e.error_code}: {e.message}")
                return None
    # -- USB / OpenCV camera: open, snapshot, reset ---------------------------
    _USB_PROPS = {
        "brightness": cv2.CAP_PROP_BRIGHTNESS, "contrast": cv2.CAP_PROP_CONTRAST,
        "saturation": cv2.CAP_PROP_SATURATION, "hue": cv2.CAP_PROP_HUE,
        "gain": cv2.CAP_PROP_GAIN, "exposure": cv2.CAP_PROP_EXPOSURE,
        "sharpness": cv2.CAP_PROP_SHARPNESS, "gamma": cv2.CAP_PROP_GAMMA,
        "auto_exposure": cv2.CAP_PROP_AUTO_EXPOSURE,
        "auto_wb": cv2.CAP_PROP_AUTO_WB,
        "wb_temperature": cv2.CAP_PROP_WB_TEMPERATURE,
    }

    def _open_usb(self):
        idx = self.cam_port
        if platform.system() == "Windows":
            self.h = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
        else:
            self.h = cv2.VideoCapture(idx)
        self.h.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.h.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)

    def _snapshot_usb(self):
        self._usb_defaults = {}
        for name, pid in self._USB_PROPS.items():
            try:
                self._usb_defaults[name] = self.h.get(pid)
            except Exception:
                pass

    def reset_settings(self):
        """USB camera -> back to its own default settings, nothing applied
        on top.  Uses the driver defaults (v4l2-ctl, Linux) when available,
        otherwise the values the camera had when the program opened it.
        Then re-opens the device.  Returns (ok, message)."""
        if self.camtype != 'usbcam':
            return False, "not a USB camera"
        done = 0
        okset = set()
        try:
            if platform.system() != "Windows" and shutil.which("v4l2-ctl"):
                dev = f"/dev/video{self.cam_port}"
                out = subprocess.run(["v4l2-ctl", "-d", dev, "--list-ctrls"],
                                     capture_output=True, text=True,
                                     timeout=10).stdout
                ctrls = []
                for line in out.splitlines():
                    m = re.match(r"\s*(\w+)\s+0x[0-9a-f]+\s+\((int|bool|menu|"
                                 r"int64)\)\s*:.*?default=(-?\d+)", line)
                    if m and "read-only" not in line:
                        ctrls.append((m.group(1), m.group(3)))
                # two passes: 'auto' controls first so manual ones unlock
                ctrls.sort(key=lambda c: 0 if "auto" in c[0] else 1)
                for _ in range(2):
                    for name, val in ctrls:
                        r = subprocess.run(
                            ["v4l2-ctl", "-d", dev, "--set-ctrl",
                             f"{name}={val}"], capture_output=True, timeout=10)
                        if r.returncode == 0:
                            okset.add(name)
                done = len(okset)
            else:
                for name, val in getattr(self, "_usb_defaults", {}).items():
                    if self.h.set(self._USB_PROPS[name], val):
                        done += 1
            try:
                self.h.release()
            except Exception:
                pass
            time.sleep(0.3)
            self._open_usb()
            ok = self.h.isOpened()
            return ok, (f"USB camera reset to its default settings "
                        f"({done} control(s) restored)" if ok else
                        "USB camera could not be re-opened after reset")
        except Exception as e:
            return False, f"USB camera reset failed: {e}"

    # -- helpers: MindVision-only calls must never crash the program -------
    @staticmethod
    def _clamp(v, lo, hi):
        return max(lo, min(hi, v))

    def _mv(self, fn, *args):
        """Call an mvsdk function; a refused setting (out of range, not
        supported by this model) is reported, not raised."""
        try:
            return fn(*args)
        except mvsdk.CameraException as e:
            print(f"[camera] {fn.__name__}{args[1:]} refused: {e}")
        except Exception as e:           # e.g. ctypes arg error
            print(f"[camera] {fn.__name__} failed: {e}")
        return None

    def _set_color_gain(self, r, g, b):
        if self.camtype=='usbcam':
            return      # RGB digital gain exists only on MindVision cameras
        rg = self.cap.sRgbGainRange
        self._mv(mvsdk.CameraSetGain, self.h,
                 self._clamp(r, rg.iRGainMin, rg.iRGainMax),
                 self._clamp(g, rg.iGGainMin, rg.iGGainMax),
                 self._clamp(b, rg.iBGainMin, rg.iBGainMax))

    def get_things(self,exposure=102000,
                    gain=2,temperature=8652,
                    rGain=275,auto_exposure=1,
                    gGain=167, bGain=100,
                   height=2160,width=3680,saturation=100,sharpnes=0,
                   work_mode="Exposure"):
        self.exposure=exposure
        self.gain=gain
        self.rgain=rGain
        self.ggain=gGain
        self.bgain=bGain
        self.saturation=saturation
        self.sharpnes=sharpnes
        # if work_mode=="Resolution" and self.camtype=='usbcam':
        #         self.h.set(cv2.CAP_PROP_FRAME_HEIGHT,height) 
        #         self.h.set(cv2.CAP_PROP_FRAME_WIDTH,width)
        #         print("change shape")

        if work_mode=="Resolution":
            self.height=height
            self.width=width

        # USB / OpenCV camera: show the camera's OWN feed, untouched - no
        # exposure, gain, colour gain, saturation, sharpness, white balance
        # or v4l2 changes are ever applied to it.
        if self.camtype == 'usbcam':
            return
        # if work_mode=="Width" and self.camtype=='usbcam':
        #     self.H_W["w"]=width
        #     if len(self.H_W)==2:
        #         self.h.set(cv2.CAP_PROP_FRAME_HEIGHT,self.H_W['h']) 
        #         self.h.set(cv2.CAP_PROP_FRAME_WIDTH,self.H_W['w'])
        #         print("change shape")



        if work_mode=="Exposure":
            if self.camtype=='usbcam':
                self.h.set(cv2.CAP_PROP_EXPOSURE,exposure)
                if platform.system() != "Windows":
                    _v4l2(f"-d /dev/video{self.cam_port} --set-ctrl exposure_time_absolute={3000000}")
                    time.sleep(1)
                    _v4l2(f"-d /dev/video{self.cam_port} --set-ctrl exposure_time_absolute={exposure}")
                    time.sleep(0.4)
                    _v4l2(f"-d /dev/video{self.cam_port}  --set-ctrl auto_exposure=3")
                    time.sleep(0.4)

                    _v4l2(f"-d /dev/video{self.cam_port}  --set-ctrl auto_exposure=1")

                print("expos")
            else:
                mvsdk.CameraSetExposureTime(self.h, self.exposure)
        if work_mode=="ColorGain":
            self._set_color_gain(self.rgain, self.ggain, self.bgain)
        if work_mode=="Gain":
            print("gain")

            if self.camtype=='usbcam':
                self.h.set(cv2.CAP_PROP_BRIGHTNESS, self.gain)
                if platform.system() != "Windows":
                    _v4l2(f"-d /dev/video{self.cam_port} --set-ctrl gain={self.gain}")
            
        
            else:
                mvsdk.CameraSetAnalogGain(self.h,self.gain)
                print("here come")

        if work_mode=="Temperature":
            if self.camtype=='usbcam':
                self.h.set(cv2.CAP_PROP_TEMPERATURE, temperature)
                if platform.system() != "Windows":
                  
                    _v4l2(f"-d /dev/video{self.cam_port} --set-ctrl white_balance_temperature={temperature}")
                    print("temp")
        if work_mode=="Auto exposure" and self.camtype=='usbcam':
             if platform.system() != "Windows":
                 _v4l2(f"-d /dev/video{self.cam_port}  --set-ctrl auto_exposure={auto_exposure}")

        if work_mode=="Saturation":
            if self.camtype=='usbcam':
                self.h.set(cv2.CAP_PROP_SATURATION, self.saturation)
            else:
                self._mv(mvsdk.CameraSetSaturation, self.h, self._clamp(
                    self.saturation, self.cap.sSaturationRange.iMin,
                    self.cap.sSaturationRange.iMax))

        # callers use both spellings: "Sharpness" and "Sharpnes"
        if work_mode in ("Sharpnes", "Sharpness"):
            if self.camtype=='usbcam':
                self.h.set(cv2.CAP_PROP_SHARPNESS, self.sharpnes)
            else:
                self._mv(mvsdk.CameraSetSharpness, self.h, self._clamp(
                    self.sharpnes, self.cap.sSharpnessRange.iMin,
                    self.cap.sSharpnessRange.iMax))

        if work_mode=="auto_balance":
            if self.camtype=='usbcam':
                # a USB/OpenCV camera has no MindVision one-shot white
                # balance: use the driver's auto white balance instead
                self.h.set(cv2.CAP_PROP_AUTO_WB, 1)
            else:
                self._mv(mvsdk.CameraSetOnceWB, self.h)
                self._mv(mvsdk.CameraSetOnceBB, self.h)
                self._set_color_gain(187, 100, 169)
        # print(mvsdk.CameraGetFrameSpeed(self.h) ) # 
        #Ensure frame speed is set
        
        
    def close(self):
        if self.camtype=='usbcam':
            self.h.release()
            print(f"Released OpenCV camera index {self.cam_port}")
            
        else:
            print("here i come to relese ind")

            mvsdk.CameraUnInit(self.h)
            mvsdk.CameraAlignFree(self.pFrameBuffer)
            print("release camera")


def main():
    # 1) Enumerate & wrap
    devs=get_camera_info(no=1)
    # print("Camera info:", cameraaaa,len(cameraaaa))
    # devs = mvsdk.CameraEnumerateDevice()
    # if not devs:
    #     print("No cameras found.")
    #     return
    print(devs)
    if not devs:
        print("No camera connected - check power / cable.")
        return
    cams = [CameraWrapper(dev) for dev in devs]
    

    window_names = {i: f"Cam-{i}" for i in range(len(cams))}
    for w in window_names.values():
        cv2.namedWindow(w, cv2.WINDOW_NORMAL)

    def apply_settings(cam):
        # (a USB camera ignores all of these - raw feed, no processing)
        cam.get_things(exposure=74005,work_mode="Exposure")
        cam.get_things(gain=4,work_mode="Gain")
        cam.get_things(rGain=306, gGain=100, bGain=100,work_mode="ColorGain")
        cam.get_things(saturation=130,work_mode="Saturation")
        cam.get_things(sharpnes=30,work_mode="Sharpness")
        cam.get_things(work_mode="auto_balance")

    def reset_camera(cam):
        """USB camera -> its own default settings; MindVision -> re-apply."""
        if cam.camtype == 'usbcam':
            ok, msg = cam.reset_settings()
        else:
            apply_settings(cam)
            ok, msg = True, "camera settings re-applied"
        print(f"[camera] {'OK' if ok else 'FAILED'}: {msg}")

    apply_settings(cams[0])
    print("Keys:  x = RESET camera settings   d = save image   q = quit")
# 2) Grab loop
    try:
        c=0
        while True:
            c+=1
            k= cv2.waitKey(1)


            if k == ord('q'):
                break
            if k == ord('x'):
                reset_camera(cams[0])

            for i in range(len(cams)):
                cam = cams[i]
                frame = cam.grab()
                #cam.get_things(exposure=1500005,work_mode="Exposure")
                


                if i==0:
                    # try:
                    # except Exception as e:
                    #     print(f"Error getting frame shape: {e}")
                    #     continue

                    # Example: adjust exposure dynamically for the first camera
                    # cam.get_things(width=3680,work_mode="Width") 
                    if k== ord('w'):
                        cam.get_things(width=680,height=2160,work_mode="Resolution") 

                    if k== ord('s'):
                        cam.get_things(width=1200,height=1900,work_mode="Resolution") 
                        # Example: adjust exposure dynamically
                         # Example: adjust exposure dynamically
                    if k== ord('e'):
                        cam.get_things(exposure=-5,work_mode="Exposure") # Example: adjust exposure dynamically
                    if k== ord('r'):
                        cam.get_things(exposure=5,work_mode="Exposure")
                    if k== ord('g'):
                        cam.get_things(gain=10,work_mode="Gain") # Example: adjust exposure dynamically
                    if k== ord('t'):
                        cam.get_things(gain=110,work_mode="Gain")

                if k ==ord("c"):
                
                            cams[0].get_things(work_mode="auto_balance")
                            # print("auto_balances")
                            
                if frame is None:
                    # skip display if no data
                    continue

              
                cv2.imshow(window_names[i], frame)

                if k==ord("d"):
                    cv2.imwrite(f"Camera/Data/image_{time.time()}.jpg ",frame)

    finally:
        # cleanup
        for cam in cams:
            cam.close()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()