from libcamera import controls
from picamera2 import MappedArray
import cv2
import time

DEBUG_MODE = True
Black_White_Threshold = 50

def Rescue_Camera_Pre_callback(request):
    pass


def Linetrace_Camera_Pre_callback(request):
    if DEBUG_MODE:
        print("precallback called", str(time.time()))
    with MappedArray(request, "lores") as m:
        current = m.array
        image_bgr = cv2.cvtColor(current, cv2.COLOR_RGB2GRAY)
        _, frame = cv2.threshold(image_bgr, Black_White_Threshold, 255, cv2.THRESH_BINARY)
        if DEBUG_MODE:
            cv2.imwrite(f"bin/{str(time.time())}.jpg", frame)
        # Split the frame into left and right halves
        height, width = frame.shape
        left_half = frame[:, :width//2]
        right_half = frame[:, width//2:]

        # Count black pixels (value 0) in each half
        left_black_pixels = cv2.countNonZero(255 - left_half)
        right_black_pixels = cv2.countNonZero(255 - right_half)

        if DEBUG_MODE:
            print(f"Left black pixels: {left_black_pixels}, Right black pixels: {right_black_pixels}")

Rescue_Camera_PORT = 1
Rescue_Camera_Controls = {
    "AfMode": controls.AfModeEnum.Continuous,
    "AfSpeed": controls.AfSpeedEnum.Fast,
    "AeFlickerMode": controls.AeFlickerModeEnum.Manual,
    "AeFlickerPeriod": 10000,
    "AeMeteringMode": controls.AeMeteringModeEnum.Matrix,
    "AwbEnable": True,
    "AwbMode": controls.AwbModeEnum.Indoor,
    "HdrMode": controls.HdrModeEnum.Off
}
Rescue_Camera_size = (4608, 2592)
Rescue_Camera_formats = "RGB888"
Rescue_Camera_lores_size = (Rescue_Camera_size[0] // 4,
                            Rescue_Camera_size[1] // 4)
Rescue_Camera_Pre_Callback_func = Rescue_Camera_Pre_callback

Linetrace_Camera_PORT = 0
Linetrace_Camera_Controls = {
    "AfMode": controls.AfModeEnum.Manual,
    "LensPosition": 1.0 / 0.03,
    "AeFlickerMode": controls.AeFlickerModeEnum.Manual,
    "AeFlickerPeriod": 10000,
    "AeMeteringMode": controls.AeMeteringModeEnum.Matrix,
    "AwbEnable": False,
    "AwbMode": controls.AwbModeEnum.Indoor,
    "HdrMode": controls.HdrModeEnum.Night
}
Linetrace_Camera_size = (4608, 2592)
Linetrace_Camera_formats = "RGB888"
Linetrace_Camera_lores_size = (Linetrace_Camera_size[0] // 4,
                               Linetrace_Camera_size[1] // 4)
Linetrace_Camera_Pre_Callback_func = Linetrace_Camera_Pre_callback