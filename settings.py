from libcamera import controls
from picamera2 import MappedArray
import cv2
import time
import numpy as np
import threading
import modules.log
from typing import List, Tuple, Optional

logger = modules.log.get_logger()

# Constants
DEBUG_MODE = True
BLACK_WHITE_THRESHOLD = 100
LINETRACE_CAMERA_LORES_HEIGHT = 180
LINETRACE_CAMERA_LORES_WIDTH = 320
COMPUTING_P = 297

# Detection thresholds
MIN_GREEN_AREA = 200
MIN_RED_AREA = 200
MIN_SILVER_AREA = 200
MIN_BLACK_LINE_AREA = 100

# Threading locks
linetracecam_threadlock = threading.Lock()
LASTBLACKLINE_LOCK = threading.Lock()
SLOPE_LOCK = threading.Lock()

# Global state variables
lastblackline = LINETRACE_CAMERA_LORES_WIDTH // 2  # Initialize to center
slope: Optional[float] = None

# Contour storage
red_contours: List[np.ndarray] = []
green_contours: List[np.ndarray] = []
silver_contours: List[np.ndarray] = []

# Detection results
green_marks: List[Tuple[int, int, int, int]] = []
green_black_detected: List[np.ndarray] = []
silver_marks: List[Tuple[int, int, int, int]] = []

# Control flags
stop_requested = False
is_rescue_area = False


def detect_green_marks(orig_image: np.ndarray,
                       blackline_image: np.ndarray) -> None:
  """Detect multiple X-shaped green marks and their relationship with black lines."""
  global green_marks, green_black_detected, green_contours

  # Convert to HSV (avoid copying if possible)
  hsv = cv2.cvtColor(orig_image, cv2.COLOR_RGB2HSV)

  # Define green color range
  lower_green = np.array([30, 40, 20])
  upper_green = np.array([100, 255, 255])

  # Create mask for green color
  green_mask = cv2.inRange(hsv, lower_green, upper_green)

  # Clean up noise with optimized kernel
  kernel = np.ones((3, 3), np.uint8)
  green_mask = cv2.morphologyEx(green_mask,
                                cv2.MORPH_OPEN,
                                kernel,
                                iterations=2)

  # Save green mask for debugging
  if DEBUG_MODE:
    cv2.imwrite(f"bin/{time.time():.3f}_green_mask.jpg", green_mask)

  # Find contours
  green_contours, _ = cv2.findContours(green_mask, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)

  # Reset global variables
  green_marks.clear()
  green_black_detected.clear()

  # Process each contour
  for contour in green_contours:
    if cv2.contourArea(contour) > MIN_GREEN_AREA:
      # Get bounding box
      x, y, w, h = cv2.boundingRect(contour)
      logger.debug(f"Green mark found at ({x}, {y}) with size ({w}, {h})")

      # Calculate center point
      center_x = x + w // 2
      center_y = y + h // 2

      # Store mark info
      green_marks.append((center_x, center_y, w, h))

      # Check for black lines around the mark
      black_detections = _check_black_lines_around_mark(blackline_image,
                                                        center_x, center_y, w,
                                                        h)
      green_black_detected.append(black_detections)

      if DEBUG_MODE:
        _draw_green_mark_debug(orig_image, x, y, w, h, center_x, center_y,
                               black_detections)

  # Save the image with X marks drawn on it
  if DEBUG_MODE and green_marks:
    cv2.imwrite(f"bin/{time.time():.3f}_green_marks_with_x.jpg", orig_image)


def detect_red_marks(orig_image: np.ndarray) -> None:
  """Detect red marks and set stop_requested flag."""
  global stop_requested, red_contours
  hsv = cv2.cvtColor(orig_image, cv2.COLOR_RGB2HSV)

  # Red color range
  lower_red = np.array([160, 70, 110])
  upper_red = np.array([179, 255, 255])

  red_mask = cv2.inRange(hsv, lower_red, upper_red)

  # Clean up noise with morphology
  kernel = np.ones((3, 3), np.uint8)
  red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel, iterations=3)

  if DEBUG_MODE:
    cv2.imwrite(f"bin/{time.time():.3f}_red_mask.jpg", red_mask)

  red_contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL,
                                     cv2.CHAIN_APPROX_SIMPLE)
  read_red = 0
  for contour in red_contours:
    #if cv2.contourArea(contour) > MIN_RED_AREA://TODO: check area
    x, y, w, h = cv2.boundingRect(contour)
    center_x = x + w // 2
    center_y = y + h // 2
    read_red += 1
    if read_red >= 3:
      stop_requested = True

    if DEBUG_MODE:
      _draw_red_mark_debug(orig_image, x, y, w, h, center_x, center_y)


def detect_silver_marks(orig_image: np.ndarray) -> None:
    # Convert to HSV (silver ≈ low saturation + high value)
  hsv = cv2.cvtColor(orig_img, cv2.COLOR_BGR2HSV)
  lower = np.array([0, 0, 180])
  upper = np.array([180, 50, 255])
  mask = cv2.inRange(hsv, lower, upper)

  # Find contours
  contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

  img = cv2.imread("silver_line.png")
  h, w, _ = img.shape
  mid_y = h // 2

  for cnt in contours:
    if cv2.contourArea(cnt) > 100:  # skip small noise
      M = cv2.moments(cnt)
      if M["m00"] != 0:
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        
        # Draw center on image
        cv2.circle(img, (cx, cy), 5, (0, 0, 255), -1)
        
        # Check position vs. middle
        if cy > mid_y:   # center is lower half
          is_rescue_area = True

  #"""Detect silver marks and set rescue area flag."""
  #global silver_marks, is_rescue_area

  #hsv = cv2.cvtColor(orig_image, cv2.COLOR_RGB2HSV)

  ## Silver color range
  #lower_silver = np.array([0, 0, 200])
  #upper_silver = np.array([179, 60, 255])

  #silver_mask = cv2.inRange(hsv, lower_silver, upper_silver)

  ## Clean up noise
  #kernel = np.ones((3, 3), np.uint8)
  #silver_mask = cv2.morphologyEx(silver_mask,
  #                               cv2.MORPH_OPEN,
  #                               kernel,
  #                               iterations=2)

  #if DEBUG_MODE:
  #  cv2.imwrite(f"bin/{time.time():.3f}_silver_mask.jpg", silver_mask)

  #contours, _ = cv2.findContours(silver_mask, cv2.RETR_EXTERNAL,
  #                               cv2.CHAIN_APPROX_SIMPLE)

  #silver_marks.clear()

  #for contour in contours:
  #  if cv2.contourArea(contour) < MIN_SILVER_AREA:
  #    x, y, w, h = cv2.boundingRect(contour)
  #    center_x = x + w // 2
  #    center_y = y + h // 2

  #    silver_marks.append((center_x, center_y, w, h))
  #    if center_y > orig_image.shape[0] // 2:
  #      is_rescue_area = True

  #    if DEBUG_MODE:
  #      _draw_silver_mark_debug(orig_image, x, y, w, h, center_x, center_y)


def _check_black_lines_around_mark(blackline_image: np.ndarray, center_x: int,
                                   center_y: int, w: int, h: int) -> np.ndarray:
  """Check for black lines around a mark in four directions."""
  black_detections = np.zeros(4, dtype=np.int8)  # [bottom, top, left, right]

  # Define ROI sizes relative to mark size
  roi_width = int(w * 0.5)
  roi_height = int(h * 0.5)

  # Check bottom
  roi_b = blackline_image[center_y +
                          h // 2:min(center_y + h // 2 +
                                     roi_height, LINETRACE_CAMERA_LORES_HEIGHT),
                          center_x - roi_width // 2:center_x + roi_width // 2]
  if roi_b.size > 0 and np.mean(roi_b) > BLACK_WHITE_THRESHOLD:
    black_detections[0] = 1

  # Check top
  roi_t = blackline_image[max(center_y - h // 2 -
                              roi_height, 0):center_y - h // 2,
                          center_x - roi_width // 2:center_x + roi_width // 2]
  if roi_t.size > 0 and np.mean(roi_t) > BLACK_WHITE_THRESHOLD:
    black_detections[1] = 1

  # Check left
  roi_l = blackline_image[center_y - roi_height // 2:center_y + roi_height // 2,
                          max(center_x - w // 2 - roi_width, 0):center_x -
                          w // 2]
  if roi_l.size > 0 and np.mean(roi_l) > BLACK_WHITE_THRESHOLD:
    black_detections[2] = 1

  # Check right
  roi_r = blackline_image[center_y - roi_height // 2:center_y + roi_height // 2,
                          center_x +
                          w // 2:min(center_x + w // 2 +
                                     roi_width, LINETRACE_CAMERA_LORES_WIDTH)]
  if roi_r.size > 0 and np.mean(roi_r) > BLACK_WHITE_THRESHOLD:
    black_detections[3] = 1

  return black_detections


def _draw_green_mark_debug(image: np.ndarray, x: int, y: int, w: int, h: int,
                           center_x: int, center_y: int,
                           black_detections: np.ndarray) -> None:
  """Draw debug visualization for green marks."""
  # Draw X mark
  cv2.line(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
  cv2.line(image, (x + w, y), (x, y + h), (0, 255, 0), 2)
  # Draw center point
  cv2.circle(image, (center_x, center_y), 5, (0, 0, 255), -1)
  # Draw black line detection indicators
  if black_detections[0]:
    cv2.line(image, (center_x - 10, center_y + 10),
             (center_x + 10, center_y + 10), (255, 0, 0), 2)
  if black_detections[1]:
    cv2.line(image, (center_x - 10, center_y - 10),
             (center_x + 10, center_y - 10), (255, 0, 0), 2)
  if black_detections[2]:
    cv2.line(image, (center_x - 10, center_y - 10),
             (center_x - 10, center_y + 10), (255, 0, 0), 2)
  if black_detections[3]:
    cv2.line(image, (center_x + 10, center_y - 10),
             (center_x + 10, center_y + 10), (255, 0, 0), 2)


def _draw_red_mark_debug(image: np.ndarray, x: int, y: int, w: int, h: int,
                         center_x: int, center_y: int) -> None:
  """Draw debug visualization for red marks."""
  # Draw X mark
  cv2.line(image, (x, y), (x + w, y + h), (0, 0, 255), 2)
  cv2.line(image, (x + w, y), (x, y + h), (0, 0, 255), 2)
  # Draw center point
  cv2.circle(image, (center_x, center_y), 5, (0, 0, 255), -1)
  cv2.imwrite(f"bin/{time.time():.3f}_red_marks.jpg", image)


def _draw_silver_mark_debug(image: np.ndarray, x: int, y: int, w: int, h: int,
                            center_x: int, center_y: int) -> None:
  """Draw debug visualization for silver marks."""
  # Draw X mark
  cv2.line(image, (x, y), (x + w, y + h), (125, 125, 125), 2)
  cv2.line(image, (x + w, y), (x, y + h), (125, 125, 125), 2)
  # Draw center point
  cv2.circle(image, (center_x, center_y), 5, (125, 125, 125), -1)
  cv2.imwrite(f"bin/{time.time():.3f}_silver_marks.jpg", image)


def Linetrace_Camera_Pre_callback(request):
  """Optimized camera callback for line tracing."""
  global lastblackline, slope
  current_time = time.time()

  try:
    with linetracecam_threadlock:
      with MappedArray(request, "lores") as m:
        # Get image from camera
        image = m.array

        # Save original image for debugging
        if DEBUG_MODE:
          cv2.imwrite(f"bin/{current_time:.3f}_original.jpg", image)

        # Convert image to grayscale
        gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        # Create binary image with threshold for black line detection
        _, binary_image = cv2.threshold(gray_image, BLACK_WHITE_THRESHOLD, 255,
                                        cv2.THRESH_BINARY_INV)

        # Save binary image for debugging
        if DEBUG_MODE:
          cv2.imwrite(f"bin/{current_time:.3f}_binary.jpg", binary_image)

        # Clean up noise with morphological operations (optimized)
        kernel = np.ones((3, 3), np.uint8)
        binary_image = cv2.morphologyEx(binary_image,
                                        cv2.MORPH_OPEN,
                                        kernel,
                                        iterations=3)

        # Detect marks in parallel (if threading is available)
        detect_red_marks(image)
        detect_green_marks(image, binary_image)
        #detect_silver_marks(image)

        # Find contours of the black line
        contours, _ = cv2.findContours(
            binary_image, cv2.RETR_TREE,
            cv2.CHAIN_APPROX_SIMPLE)  # Use SIMPLE for better performance

        # If no contours found, keep previous values and return
        if not contours:
          slope = None
          return

        # Find the best contour to follow
        best_contour = find_best_contour(contours, LINETRACE_CAMERA_LORES_WIDTH,
                                         LINETRACE_CAMERA_LORES_HEIGHT,
                                         lastblackline)

        if best_contour is None:
          slope = None
          return

        # Calculate center point of contour
        cx, cy = calculate_contour_center(best_contour)

        # Update global variables for line following
        with LASTBLACKLINE_LOCK:
          lastblackline = cx

        # Calculate slope for steering
        with SLOPE_LOCK:
          slope = calculate_slope(best_contour, cx, cy)

        # Create debug visualization if needed
        if DEBUG_MODE:
          debug_image = visualize_tracking(image, best_contour, cx, cy)
          _draw_debug_contours(debug_image)
          cv2.imwrite(f"bin/{current_time:.3f}_tracking.jpg", debug_image)

  except SystemExit:
    print("SystemExit caught")
    raise
  except Exception as e:
    if DEBUG_MODE:
      logger.error(f"Error in line tracing: {e}")


def _draw_debug_contours(debug_image: np.ndarray) -> None:
  """Draw debug visualization for all detected contours."""
  # Red contours
  for contour in red_contours:
    x, y, w, h = cv2.boundingRect(contour)
    cv2.line(debug_image, (x, y), (x + w, y + h), (0, 0, 255), 2)
    cv2.line(debug_image, (x + w, y), (x, y + h), (0, 0, 255), 2)
    cv2.circle(debug_image, (x + w // 2, y + h // 2), 5, (0, 0, 255), -1)

  # Green contours
  for contour, black_detection in zip(green_contours, green_black_detected):
    x, y, w, h = cv2.boundingRect(contour)
    cv2.line(debug_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.line(debug_image, (x + w, y), (x, y + h), (0, 255, 0), 2)
    cv2.circle(debug_image, (x + w // 2, y + h // 2), 5, (0, 255, 0), -1)

    # Draw black line detection indicators
    center_x, center_y = x + w // 2, y + h // 2
    if black_detection[0]:
      cv2.line(debug_image, (center_x, center_y), (center_x, center_y + 10),
               (255, 0, 0), 2)
    if black_detection[1]:
      cv2.line(debug_image, (center_x, center_y), (center_x, center_y - 10),
               (255, 0, 0), 2)
    if black_detection[2]:
      cv2.line(debug_image, (center_x, center_y), (center_x - 10, center_y),
               (255, 0, 0), 2)
    if black_detection[3]:
      cv2.line(debug_image, (center_x, center_y), (center_x + 10, center_y),
               (255, 0, 0), 2)


def find_best_contour(contours: List[np.ndarray], camera_x: int, camera_y: int,
                      last_center: int) -> Optional[np.ndarray]:
  """
  Find the best contour to follow from multiple candidates.
  Prioritizes contours at the bottom of the image and close to the center.
  Also considers line width and continuity to handle intersections.
  
  Returns the selected contour or None if no suitable contour found.
  """
  if not contours:
    return None

  # Filter contours by minimum area first
  valid_contours = [(i, contour) for i, contour in enumerate(contours)
                    if cv2.contourArea(contour) >= MIN_BLACK_LINE_AREA]

  if not valid_contours:
    return None

  # Process valid contours
  candidates = []
  bottom_contours = 0

  for i, contour in valid_contours:
    # Get bounding box
    rect = cv2.minAreaRect(contour)
    box = cv2.boxPoints(rect)
    # Sort points by y-coordinate (descending)
    box = box[box[:, 1].argsort()[::-1]]

    # Calculate line width at bottom
    width = abs(box[0][0] - box[1][0])

    # Check if contour extends to bottom of image
    is_bottom = box[0][1] >= (camera_y * 0.95)
    if is_bottom:
      bottom_contours += 1

    candidates.append({
        'index': i,
        'contour': contour,
        'x1': int(box[0][0]),
        'y1': int(box[0][1]),
        'x2': int(box[1][0]),
        'y2': int(box[1][1]),
        'width': width,
        'is_bottom': is_bottom
    })

  # Sort candidates by y-coordinate (prioritize contours at bottom)
  candidates.sort(key=lambda x: x['y1'], reverse=True)

  # If multiple contours at bottom, choose based on width and distance from center
  if bottom_contours > 1:
    bottom_candidates = [c for c in candidates if c['is_bottom']]

    for candidate in bottom_candidates:
      center_x = (candidate['x1'] + candidate['x2']) / 2
      image_center = camera_x / 2
      distance_from_center = abs(image_center - center_x)

      # Penalize very wide lines (likely intersections)
      if candidate['width'] > 20:
        distance_from_center *= 2

      candidate['distance_from_center'] = distance_from_center

    # Sort bottom contours by distance from image center (prioritize middle)
    bottom_candidates.sort(key=lambda x: x['distance_from_center'])
    return bottom_candidates[0]['contour']

  # Return best contour (highest y-coordinate)
  return candidates[0]['contour'] if candidates else None


def calculate_contour_center(contour: np.ndarray) -> Tuple[int, int]:
  """Calculate the center point of a contour."""
  M = cv2.moments(contour)
  if M["m00"] != 0:
    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])
  else:
    # Fallback to bounding box center
    x, y, w, h = cv2.boundingRect(contour)
    cx = x + w // 2
    cy = y + h // 2

  return cx, cy


def calculate_slope(contour: np.ndarray, cx: int, cy: int) -> float:
  """Calculate the slope of the line for steering."""
  try:
    # Set base point
    base_x = LINETRACE_CAMERA_LORES_WIDTH // 2
    base_y = LINETRACE_CAMERA_LORES_HEIGHT

    # Calculate slope between top and center points
    if cx != base_x:  # Avoid division by zero or tiny values
      return (base_y - cy) / (cx - base_x)
    else:
      return 10**9
  except Exception as e:
    if DEBUG_MODE:
      logger.error(f"Error in calculate_slope: {e}")
    return 0.0


def visualize_tracking(image: np.ndarray, contour: np.ndarray, cx: int,
                       cy: int) -> np.ndarray:
  """Create a visualization image showing tracking information."""
  # Make a copy of the image for drawing
  vis_image = image.copy()

  # Draw the contour
  cv2.drawContours(vis_image, [contour], 0, (0, 255, 0), 1)

  # Draw center point
  cv2.circle(vis_image, (cx, cy), 3, (0, 0, 255), -1)

  # Draw horizontal line at center of image
  h, w = vis_image.shape[:2]
  cv2.line(vis_image, (0, h // 2), (w, h // 2), (255, 0, 0), 1)

  # Draw vertical line at the tracked position
  cv2.line(vis_image, (cx, 0), (cx, h), (255, 0, 0), 1)

  return vis_image


def Rescue_Camera_Pre_callback(request):
  """Rescue camera callback function."""
  with MappedArray(request, "lores") as m:
    image = m.array
    fixed_image = cv2.rotate(image, cv2.ROTATE_180)
    cv2.imwrite(f"bin/{str(time.time())}_rescue.jpg", fixed_image)
    print("Rescue_Camera_Pre_callback")


# Camera configuration constants
RESCUE_CAMERA_PORT = 0
RESCUE_CAMERA_CONTROLS = {
    "AfMode": controls.AfModeEnum.Continuous,
    "AfSpeed": controls.AfSpeedEnum.Fast,
    "AeFlickerMode": controls.AeFlickerModeEnum.Manual,
    "AeFlickerPeriod": 10000,
    "AeMeteringMode": controls.AeMeteringModeEnum.Matrix,
    "AwbEnable": True,
    "AwbMode": controls.AwbModeEnum.Indoor,
    "HdrMode": controls.HdrModeEnum.Off
}
RESCUE_CAMERA_SIZE = (4608, 2592)
RESCUE_CAMERA_FORMATS = "RGB888"
RESCUE_CAMERA_LORES_SIZE = (RESCUE_CAMERA_SIZE[0] // 4,
                            RESCUE_CAMERA_SIZE[1] // 4)
RESCUE_CAMERA_PRE_CALLBACK_FUNC = Rescue_Camera_Pre_callback

LINETRACE_CAMERA_PORT = 1
LINETRACE_CAMERA_CONTROLS = {
    "AfMode": controls.AfModeEnum.Manual,
    "LensPosition": 1.0 / 0.03,
    "AeFlickerMode": controls.AeFlickerModeEnum.Manual,
    "AeFlickerPeriod": 10000,
    "AeMeteringMode": controls.AeMeteringModeEnum.Matrix,
    "AwbEnable": False,
    "AwbMode": controls.AwbModeEnum.Indoor,
    "HdrMode": controls.HdrModeEnum.Night
}
LINETRACE_CAMERA_SIZE = (3456, 2592)
LINETRACE_CAMERA_FORMATS = "RGB888"
LINETRACE_CAMERA_LORES_SIZE = (LINETRACE_CAMERA_LORES_WIDTH,
                               LINETRACE_CAMERA_LORES_HEIGHT)
LINETRACE_CAMERA_PRE_CALLBACK_FUNC = Linetrace_Camera_Pre_callback


def timeout_function(func, args=(), kwargs={}, timeout=1):
  result = []
  error = []

  def target():
    try:
      result.append(func(*args, **kwargs))
    except Exception as e:
      error.append(e)

  thread = threading.Thread(target=target)
  thread.daemon = True
  thread.start()
  thread.join(timeout)

  if thread.is_alive():
    # Thread is still running, timeout occurred
    return None, TimeoutError(f"Function timed out after {timeout} seconds")
  if error:
    return None, error[0]
  return result[0] if result else None, None
