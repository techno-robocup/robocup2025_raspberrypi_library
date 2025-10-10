import modules.settings
import numpy as np
from ultralytics import YOLO
from enum import Enum
import time
import threading
import modules.log

logger = modules.log.get_logger()


class ObjectClasses(Enum):
	BLACK_BALL = 0
	FINAL_TARGET = 1
	GREEN_CAGE = 2
	RED_CAGE = 3
	SILVER_BALL = 4


MODEL = YOLO("best.pt")
KP = 0.1
TP = 1.0
CP = 1.0
THRESHOLD = 10.0
MOTOR_NEUTRAL = 1500
MOTOR_MAX_TURN = 100
BALL_CATCH_SIZE = 1000
RESCUE_CAGE_SIZE = 1200
turning = False

class RobotState:
	def __init__(self):
		self.silver_ball_cnt = 0
		self.black_ball_cnt = 0
		self.is_ball_caching = False
		self.is_task_done = False
		self.is_aligned = False
		self.target_angle = None

Release_flag = False
L_motor_value = MOTOR_NEUTRAL
R_motor_value = MOTOR_NEUTRAL

robot = RobotState()


def find_best_target(boxes, valid_classes, image_width):
	"""Finds all valid targets and returns the one closest to the center."""
	global Release_flag
	best_target_angle = None
	min_dist_from_center = float("inf")
	image_center_x = image_width / 2

	for box in boxes:
		cls = int(box.cls[0])
		if cls in valid_classes:
			x_center, y_center, w, h = map(float, box.xywh[0])
			dist_from_center = abs(x_center - image_center_x)
			area = w * h

			if dist_from_center < min_dist_from_center:
				min_dist_from_center = dist_from_center
				best_target_angle = x_center - image_center_x
				logger.debug(f"size:{area}")
				if robot.is_ball_caching and area > RESCUE_CAGE_SIZE:
					Release_flag = True
				elif not robot.is_ball_caching and area > BALL_CATCH_SIZE:
					Release_flag = True
	return best_target_angle


def get_target_angle(image_frame: np.ndarray) -> None:
	if image_frame is None:
		print("Error: Received an empty image frame.")
		robot.target_angle = None
		return
	results = MODEL(image_frame, verbose=False)
	boxes = results[0].boxes

	valid_classes = []
	if robot.is_task_done:
		logger.debug("Find:Exit")
		valid_classes = [ObjectClasses.FINAL_TARGET.value]
	elif not robot.is_ball_caching:
		if robot.black_ball_cnt < 2:
			logger.debug("Find:Black")
			valid_classes = [ObjectClasses.BLACK_BALL.value]
		else:
			logger.debug("Find:Silver")
			valid_classes = [ObjectClasses.SILVER_BALL.value]
	else:
		if robot.black_ball_cnt < 2:
			logger.debug("Find:RED")
			valid_classes = [ObjectClasses.RED_CAGE.value]
		else:
			logger.debug("Find:GREEN")
			valid_classes = [ObjectClasses.GREEN_CAGE.value]

	robot.target_angle = find_best_target(
		boxes, valid_classes, results[0].orig_shape[1]
	)

	if robot.target_angle is not None:
		print(f"Targeting class(es) {valid_classes}. Best target offset = {robot.target_angle:.1f}")
	else:
		print(f"Targeting class(es) {valid_classes}. No target detected.")


def set_motor_speeds_from_angle():
	global L_motor_value, R_motor_value
	if robot.target_angle is None:
		L_motor_value = MOTOR_NEUTRAL
		R_motor_value = MOTOR_NEUTRAL
		print("No target to align. Stopping motors.")
		return

	if abs(robot.target_angle) < THRESHOLD:
		L_motor_value = MOTOR_NEUTRAL
		R_motor_value = MOTOR_NEUTRAL
	else:
		turn_speed = KP * robot.target_angle
		turn_speed = max(min(turn_speed, MOTOR_MAX_TURN), -MOTOR_MAX_TURN)

		L_motor_value = int(MOTOR_NEUTRAL + turn_speed)
		R_motor_value = int(MOTOR_NEUTRAL - turn_speed)


def turn_threaded(duration=1):
	global L_motor_value, R_motor_value, turning
	turning = True
	L_motor_value = int(TP * (MOTOR_NEUTRAL - MOTOR_MAX_TURN))
	R_motor_value = int(TP * (MOTOR_NEUTRAL + MOTOR_MAX_TURN))
	time.sleep(duration)
	L_motor_value = MOTOR_NEUTRAL
	R_motor_value = MOTOR_NEUTRAL
	turning = False


def turn():
	if not turning:
		t = threading.Thread(target=turn_threaded, args=(1,))
		t.start()


def catch_ball(area):
	"""Camera-based distance estimation: use bounding box area."""
	global R_motor_value, L_motor_value, Release_flag
	Release_flag = False

	if area > BALL_CATCH_SIZE:
		L_motor_value = MOTOR_NEUTRAL
		R_motor_value = MOTOR_NEUTRAL
		Release_flag = True
		if not robot.is_ball_caching:
			robot.is_ball_caching = True
		else:
			if robot.black_ball_cnt < 2:
				robot.black_ball_cnt += 1
				robot.is_ball_caching = False
			else:
				robot.silver_ball_cnt += 1
				robot.is_ball_caching = False
				if robot.black_ball_cnt == 2 and robot.silver_ball_cnt == 1:
					robot.is_task_done = True


def rescue_loop_func(img):
	global L_motor_value, R_motor_value

	robot.is_aligned = False
	#img = modules.settings.Rescue_Camera_Pre_callback()
	results = MODEL(img, verbose=False)
	boxes = results[0].boxes

	get_target_angle(img)
	if robot.target_angle is None:
		turn()
	set_motor_speeds_from_angle()

	if boxes:
		for box in boxes:
			w, h = map(float, box.xywh[0][2:])
			area = w * h
			catch_ball(area)

	if robot.target_angle is not None and abs(robot.target_angle) < THRESHOLD:
		robot.is_aligned = True
		print(f"Robot is aligned! Motor Values: L={L_motor_value}, R={R_motor_value}")
	else:
		print(f"Aligning... Motor Values: L={L_motor_value}, R={R_motor_value}")

# Cage -> u_sonic
# First, get angles of all objects
# Use mutex gard
