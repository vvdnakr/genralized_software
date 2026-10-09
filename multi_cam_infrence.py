# coding=utf-8
import cv2
import numpy as np
import mvsdk
import platform
import time
import os
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
        return None
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
            if platform.system() == "Windows":
                self.h=cv2.VideoCapture(dev_info,cv2.CAP_DSHOW)
                self.h.set(cv2.CAP_PROP_FRAME_HEIGHT,self.height) 
                self.h.set(cv2.CAP_PROP_FRAME_WIDTH,self.width)
            else:
                self.h=cv2.VideoCapture(dev_info)
                self.h.set(cv2.CAP_PROP_FRAME_HEIGHT,self.height) 
                self.h.set(cv2.CAP_PROP_FRAME_WIDTH,self.width)

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
            mvsdk.CameraSetGain(self.h,iRGain=self.rgain,iGGain=self.ggain,iBGain=self.bgain) # Optional: set gain to 0
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
                    self.h=cv2.VideoCapture(self.cam_port)
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
                    os.system(f"v4l2-ctl  -d /dev/video{self.cam_port} --set-ctrl exposure_time_absolute={3000000}")
                    time.sleep(1)
                    os.system(f"v4l2-ctl  -d /dev/video{self.cam_port} --set-ctrl exposure_time_absolute={exposure}")
                    time.sleep(0.4)
                    os.system(f"v4l2-ctl  -d /dev/video{self.cam_port}  --set-ctrl auto_exposure=3")
                    time.sleep(0.4)

                    os.system(f"v4l2-ctl  -d /dev/video{self.cam_port}  --set-ctrl auto_exposure=1")

                print("expos")
            else:
                mvsdk.CameraSetExposureTime(self.h, self.exposure)
        if work_mode=="ColorGain":
            mvsdk.CameraSetGain(self.h,iRGain=self.rgain,iGGain=self.ggain,iBGain=self.bgain)
        if work_mode=="Gain":
            print("gain")

            if self.camtype=='usbcam':
                self.h.set(cv2.CAP_PROP_BRIGHTNESS, self.gain)
                if platform.system() != "Windows":
                    os.system(f"v4l2-ctl  -d /dev/video{self.cam_port} --set-ctrl gain={self.gain}")
            
        
            else:
                mvsdk.CameraSetAnalogGain(self.h,self.gain)
                print("here come")

        if work_mode=="Temperature":
            if self.camtype=='usbcam':
                self.h.set(cv2.CAP_PROP_TEMPERATURE, temperature)
                if platform.system() != "Windows":
                  
                    os.system(f"v4l2-ctl  -d /dev/video{self.cam_port} --set-ctrl white_balance_temperature={temperature}")
                    print("temp")
        if work_mode=="Auto exposure" and self.camtype=='usbcam':
             if platform.system() != "Windows":
                 os.system(f"v4l2-ctl  -d /dev/video{self.cam_port}  --set-ctrl auto_exposure={auto_exposure}")

        if work_mode=="Saturation":
                mvsdk.CameraSetSaturation(self.h,self.saturation)


        if work_mode=="Sharpnes":
            mvsdk.CameraSetSharpness(self.h,self.sharpnes)

        if work_mode=="auto_balance":
            mvsdk.CameraSetOnceWB(self.h)
            d=mvsdk.CameraGetUserClrTempGain(self.h)
            print(d)
            mvsdk.CameraSetOnceBB(self.h)
            mvsdk.CameraSetGain(self.h,iRGain=187,iGGain=100,iBGain=169)
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
    cams = [CameraWrapper(dev) for dev in devs]
    

    window_names = {i: f"Cam-{i}" for i in range(len(cams))}
    for w in window_names.values():
        cv2.namedWindow(w, cv2.WINDOW_NORMAL)

    cams[0].get_things(exposure=74005,work_mode="Exposure") 
    cams[0].get_things(gain=4,work_mode="Gain")
    cams[0].get_things(rGain=306, gGain=100, bGain=100,work_mode="ColorGain")
    cams[0].get_things(saturation=130,work_mode="Saturation")
    cams[0].get_things(sharpnes=30,work_mode="Sharpness")
    cams[0].get_things(work_mode="auto_balance")
# 2) Grab loop
    try:
        c=0
        while True:
            c+=1
            k= cv2.waitKey(1)


            if k == ord('q'):
                break

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
