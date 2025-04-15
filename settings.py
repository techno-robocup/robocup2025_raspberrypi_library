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
        # Number of parts to split each half into
        num_parts = 16
        
        # Function to analyze a section and return True for white majority or False for black majority
        def analyze_section(section):
            white_pixels = cv2.countNonZero(section)
            black_pixels = section.size - white_pixels
            return white_pixels > black_pixels
        
        # Split left half into 16 vertical parts and analyze each
        left_sections = []
        section_width = (width // 2) // num_parts
        for i in range(num_parts):
            section = left_half[:, i * section_width:(i + 1) * section_width]
            left_sections.append(analyze_section(section))
        
        # Split right half into 16 vertical parts and analyze each
        right_sections = []
        for i in range(num_parts):
            section = right_half[:, i * section_width:(i + 1) * section_width]
            right_sections.append(analyze_section(section))

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