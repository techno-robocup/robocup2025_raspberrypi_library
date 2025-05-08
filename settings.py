from libcamera import controls
from picamera2 import MappedArray
import cv2
import time
import threading
import numpy as np

DEBUG_MODE = True
Black_White_Threshold = 125
# Number of parts to split each half into
num_parts = 16
vertical_parts = 16
coefficient_base = 1.1
midh = 0
midw = 0
leftturn_lock = threading.Lock()
rightturn_lock = threading.Lock()
Linetrace_Camera_lores_height = 180
Linetrace_Camera_lores_width = 320

# Line tracing variables
lastblackline = Linetrace_Camera_lores_width // 2  # Initialize to center
slope = 0
Downblacke = Linetrace_Camera_lores_width // 2  # Initialize to center


def Rescue_Camera_Pre_callback(request):
  pass


def Linetrace_Camera_Pre_callback(request):
  if DEBUG_MODE:
    print("Linetrace precallback called", str(time.time()))

  # Global variables for line following
  global lastblackline, slope, Downblacke

  try:
    with MappedArray(request, "lores") as m:
      # Get image from camera
      image = m.array

      # Get camera dimensions
      camera_x = Linetrace_Camera_lores_width
      camera_y = Linetrace_Camera_lores_height

      # Save original image for debugging
      if DEBUG_MODE:
        cv2.imwrite(f"bin/{str(time.time())}_original.jpg", image)

      # Convert image to grayscale
      gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

      # Create binary image with threshold for black line detection
      _, binary_image = cv2.threshold(gray_image, Black_White_Threshold, 255,
                                      cv2.THRESH_BINARY_INV)

      # Save binary image for debugging
      if DEBUG_MODE:
        cv2.imwrite(f"bin/{str(time.time())}_binary.jpg", binary_image)

      # Clean up noise with morphological operations
      kernel = np.ones((3, 3), np.uint8)
      binary_image = cv2.erode(binary_image, kernel, iterations=2)
      binary_image = cv2.dilate(binary_image, kernel, iterations=3)

      # Find contours of the black line
      contours, _ = cv2.findContours(binary_image, cv2.RETR_TREE,
                                     cv2.CHAIN_APPROX_NONE)

      # If no contours found, keep previous values and return
      if not contours:
        return

      # Find the best contour to follow
      best_contour = find_best_contour(contours, camera_x, camera_y,
                                       lastblackline)

      if best_contour is None:
        return

      # Calculate center point of contour
      cx, cy = calculate_contour_center(best_contour)

      # Update global variables for line following
      lastblackline = cx
      Downblacke = cx

      # Calculate slope for steering
      slope = calculate_slope(best_contour, cx, cy)

      # Create debug visualization if needed
      if DEBUG_MODE:
        debug_image = visualize_tracking(image, best_contour, cx, cy)
        cv2.imwrite(f"bin/{str(time.time())}_tracking.jpg", debug_image)

  except Exception as e:
    if DEBUG_MODE:
      print(f"Error in line tracing: {e}")


def find_best_contour(contours, camera_x, camera_y, last_center):
  """
  Find the best contour to follow from multiple candidates.
  Prioritizes contours at the bottom of the image and close to the last position.
  
  Returns the selected contour or None if no suitable contour found.
  """
  # Initial candidate array structure: [contour_index, bottom_x1, bottom_y1, bottom_x2, bottom_y2, distance]
  candidates = np.array([[0, 0, 0, 0, 0, camera_x]])
  bottom_contours = 0

  # Process each contour
  for i, contour in enumerate(contours):
    # Get bounding box
    rect = cv2.minAreaRect(contour)
    box = cv2.boxPoints(rect)
    # Sort points by y-coordinate (descending)
    box = box[box[:, 1].argsort()[::-1]]

    # Add to candidates
    candidates = np.append(candidates, [[
        i,
        int(box[0][0]),
        int(box[0][1]),
        int(box[1][0]),
        int(box[1][1]), camera_x
    ]],
                           axis=0)

    # Check if contour extends to bottom of image
    if box[0][1] >= (camera_y * 0.95):
      bottom_contours += 1

  # Remove initial placeholder row
  candidates = candidates[1:] if len(candidates) > 1 else None

  if candidates is None or len(candidates) == 0:
    return None

  # Sort candidates by y-coordinate (prioritize contours at bottom)
  candidates = candidates[candidates[:, 2].argsort()[::-1]]

  # If multiple contours at bottom, choose closest to previous position
  if bottom_contours > 1:
    for i in range(bottom_contours):
      con_num, x_cor1, y_cor1, x_cor2, y_cor2, _ = candidates[i]
      # Calculate distance from last position
      center_x = (x_cor1 + x_cor2) / 2
      candidates[i, 5] = abs(last_center - center_x)

    # Sort bottom contours by distance from last position
    bottom_indices = list(range(bottom_contours))
    candidates[bottom_indices] = candidates[bottom_indices][
        candidates[bottom_indices][:, 5].argsort()]

  # Return best contour
  return contours[int(candidates[0][0])]


def calculate_contour_center(contour):
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


def calculate_slope(contour, cx, cy):
  """Calculate the slope of the line for steering."""
  try:
    # Find top point of contour
    y_min = np.amin(contour[:, :, 1])
    top_points = contour[np.where(contour[:, 0, 1] == y_min)]
    top_x = int(np.mean(top_points[:, :, 0]))

    # Calculate slope between top and center points
    if cy != y_min and cy - y_min > 1:  # Avoid division by zero or tiny values
      return (cx - top_x) / (cy - y_min)
    else:
      return 0
  except Exception as e:
    if DEBUG_MODE:
      print(f"Error in calculate_slope: {e}")
    return 0


def visualize_tracking(image, contour, cx, cy):
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
