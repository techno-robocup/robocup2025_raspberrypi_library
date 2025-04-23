from libcamera import controls
from picamera2 import MappedArray
import cv2
import time
import threading

DEBUG_MODE = True
Black_White_Threshold = 50
# Number of parts to split each half into
num_parts = 16
vertical_parts = 16
coefficient_base = 1.1
leftturn = 0
rightturn = 0
leftturn_lock = threading.Lock()
rightturn_lock = threading.Lock()


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

    height, width = frame.shape
    left_half = frame[:, :width // 2]
    right_half = frame[:, width // 2:]

    horizontal_parts = num_parts  #16 * 16 * 2
    vertical_parts = 16

    block_width = (width // 2) // horizontal_parts
    block_height = height // vertical_parts

    horizontal_coefficient = [
        coefficient_base**i for i in range(horizontal_parts)
    ]
    vertical_coefficient = [coefficient_base**i for i in range(vertical_parts)]

    local_leftturn = 0
    local_rightturn = 0

    UpLeft = 0
    UpRight = 0
    DownLeft = 0
    DownRight = 0

    for y in range(vertical_parts):
      for x in range(horizontal_parts):
        section = left_half[y * block_height:(y + 1) * block_height,
                            x * block_width:(x + 1) * block_width]
        white_pixels = cv2.countNonZero(section)
        black_pixels = section.size - white_pixels
        if white_pixels > black_pixels:
          local_leftturn += horizontal_coefficient[x] * vertical_coefficient[y]
          if y < vertical_parts // 2:
            UpLeft += horizontal_coefficient[x] * vertical_coefficient[y]
          else:
            DownLeft += horizontal_coefficient[x] * vertical_coefficient[y]

    for y in range(vertical_parts):
      for x in range(horizontal_parts):
        section = right_half[y * block_height:(y + 1) * block_height,
                             x * block_width:(x + 1) * block_width]
        white_pixels = cv2.countNonZero(section)
        black_pixels = section.size - white_pixels
        if white_pixels > black_pixels:
          local_rightturn += horizontal_coefficient[x] * vertical_coefficient[y]
          if y < vertical_parts // 2:
            UpRight += horizontal_coefficient[x] * vertical_coefficient[y]
          else:
            DownRight += horizontal_coefficient[x] * vertical_coefficient[y]

  global leftturn, rightturn
  with leftturn_lock:
    leftturn = local_leftturn
  with rightturn_lock:
    rightturn = local_rightturn

  if DEBUG_MODE:
    print(leftturn, rightturn)
    print(Upleft, UpRight, DownLeft, DownRight)
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
Linetrace_Camera_lores_size = (16,9)
Linetrace_Camera_Pre_Callback_func = Linetrace_Camera_Pre_callback
