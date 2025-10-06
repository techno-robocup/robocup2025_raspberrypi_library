# import cv2
import settings
import numpy as np
from ultralytics import YOLO
from enum import Enum


class ObjectClasses(Enum):
	BLACK_BALL = 0
	RED_CACHE = 1
	GREEN_CACHE = 2
	SILVER_BALL = 4
	FINAL_TARGET = 5


MODEL = YOLO("best.pt")
KP = 0.1
THRESHOLD = 10.0  # Increased slightly for more stability
MOTOR_NEUTRAL = 1500
MOTOR_MAX_TURN = 100


class RobotState:
	def __init__(self):
		self.silver_ball_cnt = 0
		self.black_ball_cnt = 0
		self.is_ball_caching = False
		self.is_task_done = False
		self.is_aligned = False
		self.L_motor_value = MOTOR_NEUTRAL
		self.R_motor_value = MOTOR_NEUTRAL
		self.target_angle = None


robot = RobotState()


def find_best_target(boxes, valid_classes, image_width):
	"""Finds all valid targets and returns the one closest to the center."""
	best_target_angle = None
	min_dist_from_center = float("inf")
	image_center_x = image_width / 2

	for box in boxes:
		cls = int(box.cls[0])
		if cls in valid_classes:
			x_center = float(box.xywh[0][0])
			dist_from_center = abs(x_center - image_center_x)

			if dist_from_center < min_dist_from_center:
				min_dist_from_center = dist_from_center
				best_target_angle = x_center - image_center_x

	return best_target_angle


def get_target_angle(image_frame: np.ndarray) -> None:
	"""
	Performs inference on an image frame, determines the correct target based on state,
	and updates the robot's target_angle.
	"""
	if image_frame is None:
		print("Error: Received an empty image frame.")
		robot.target_angle = None
		return
	results = MODEL(image_frame, verbose=False)
	boxes = results[0].boxes

	valid_classes = []
	if robot.black_ball_cnt == 2 and robot.silver_ball_cnt == 0:
		valid_classes = [ObjectClasses.FINAL_TARGET.value]
	elif not robot.is_ball_caching:
		if robot.black_ball_cnt < 2:
			valid_classes = [ObjectClasses.BLACK_BALL.value]
		else:
			valid_classes = [ObjectClasses.SILVER_BALL.value]
	else:
		if robot.black_ball_cnt < 2:
			valid_classes = [ObjectClasses.RED_CACHE.value]
		else:
			valid_classes = [ObjectClasses.GREEN_CACHE.value]

	robot.target_angle = find_best_target(
		boxes, valid_classes, results[0].orig_shape[1]
	)

	if robot.target_angle is not None:
		print(
			f"Targeting class(es) {valid_classes}. Best target offset = {robot.target_angle:.1f}"
		)
	else:
		print(f"Targeting class(es) {valid_classes}. No target detected.")


def set_motor_speeds_from_angle():
	"""Applies P-control to set motor values based on the target_angle."""
	if robot.target_angle is None:
		robot.L_motor_value = MOTOR_NEUTRAL
		robot.R_motor_value = MOTOR_NEUTRAL
		print("No target to align. Stopping motors.")
		return

	if abs(robot.target_angle) < THRESHOLD:
		robot.L_motor_value = MOTOR_NEUTRAL
		robot.R_motor_value = MOTOR_NEUTRAL
	else:
		turn_speed = KP * robot.target_angle
		turn_speed = max(min(turn_speed, MOTOR_MAX_TURN), -MOTOR_MAX_TURN)

		robot.L_motor_value = int(MOTOR_NEUTRAL + turn_speed)
		robot.R_motor_value = int(MOTOR_NEUTRAL - turn_speed)


def rescue_loop_func():
	"""The main loop for a single cycle of rescue logic."""
	robot.is_aligned = False
	img = settings.Rescue_Camera_Pre_callback()
	get_target_angle(img)
	set_motor_speeds_from_angle()
	if robot.target_angle is not None and abs(robot.target_angle) < THRESHOLD:
		robot.is_aligned = True
		print(
			f"Robot is aligned! Motor Values: L={robot.L_motor_value}, R={robot.R_motor_value}"
		)
	else:
		print(
			f"Aligning... Motor Values: L={robot.L_motor_value}, R={robot.R_motor_value}"
		)
