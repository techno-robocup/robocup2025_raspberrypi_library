from libcamera import controls
from picamera2 import MappedArray
import cv2
import time
import threading

DEBUG_MODE = True
Black_White_Threshold = 100
# Number of parts to split each half into
num_parts = 16
vertical_parts = 16
coefficient_base = 1.1
leftturn = 0
rightturn = 0
leftturn_lock = threading.Lock()
rightturn_lock = threading.Lock()
Linetrace_Camera_lores_height = 9
Linetrace_Camera_lores_width = 16
cnt = 0


def Rescue_Camera_Pre_callback(request):
  pass


def Linetrace_Camera_Pre_callback(request):
  if DEBUG_MODE:
    print("precallback called", str(time.time()))
  with MappedArray(request, "lores") as m:
    current = m.array
    image_bgr = cv2.cvtColor(current, cv2.COLOR_RGB2GRAY)
    _, frame = cv2.threshold(image_bgr, Black_White_Threshold, 255,
                             cv2.THRESH_BINARY)
    if DEBUG_MODE:
      cv2.imwrite(f"bin/{str(time.time())}.jpg", frame)
  Upblack = -1
  Upblacke = -1
  row = frame[0, :]
  for i in range(16):
    if row[i] == 255 and Upblack == -1:
      Upblack = int(i)
    if row[i] == 0 and Upblack != -1:
      Upblacke = int(i)
      Upblacke = (Upblacke + Upblack) / 2

  Downblack = -1
  Downblacke = -1
  row = frame[16, :]
  for i in range(16):
    if row[i] == 255 and Downblack == -1:
      Downblack = int(i)
    if row[i] == 0 and Downblack != -1:
      Downblacke = int(i)
      Downblacke = (Downblacke + Downblack) / 2
  return


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
Linetrace_Camera_lores_size = (Linetrace_Camera_lores_width,
                               Linetrace_Camera_lores_height)
Linetrace_Camera_Pre_Callback_func = Linetrace_Camera_Pre_callback
