# import cv2
import modules.settings
import numpy as np
from ultralytics import YOLO
from enum import Enum
import time
import threading


class ObjectClasses(Enum):
	BLACK_BALL = 0
	FINAL_TARGET = 1
	GREEN_CACHE = 2
	RED_CACHE = 3
	SILVER_BALL = 4


MODEL = YOLO("best.pt")
KP = 0.1
TP = 1.0
CP = 1.0
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
		self.target_angle = None

Release_flag = False
L_motor_value = MOTOR_NEUTRAL
R_motor_value = MOTOR_NEUTRAL

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
	if robot.is_task_done:
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
	global L_motor_value, R_motor_value
	L_motor_value = int(TP*(MOTOR_NEUTRAL - MOTOR_MAX_TURN))
	R_motor_value = int(TP*(MOTOR_NEUTRAL + MOTOR_MAX_TURN))
	time.sleep(duration)
	L_motor_value = MOTOR_NEUTRAL
	R_motor_value = MOTOR_NEUTRAL

def turn():
	t = threading.Thread(target=turn_threaded, args=(1,))
	t.start()

def catch_ball(u_sonicU):
	global R_motor_value,L_motor_value
	global Release_flag
	Release_flag = False
	R_motor_value = int(MOTOR_NEUTRAL + (u_sonicU * CP))
	L_motor_value = int(MOTOR_NEUTRAL + (u_sonicU * CP))
	if u_sonicU <= 1:
		L_motor_value = MOTOR_NEUTRAL
		R_motor_value = MOTOR_NEUTRAL
		Release_flag = True
		if robot.is_ball_caching:
			robot.is_ball_caching = False
		else:
			if robot.black_ball_cnt < 2:
				robot.black_ball_cnt += 1
				robot.is_ball_caching = True
			else:
				robot.silver_ball_cnt += 1
				robot.is_ball_caching = True
				if robot.black_ball_cnt == 2 and robot.silver_ball_cnt == 1:
					robot.is_task_done = True

def rescue_loop_func(u_sonicL, u_sonicU, u_sonicR):
	global L_motor_value, R_motor_value

	robot.is_aligned = False
	img = settings.Rescue_Camera_Pre_callback()
	get_target_angle(img)

	if robot.target_angle is None:
		turn()
	set_motor_speeds_from_angle()

	if robot.target_angle is not None and abs(robot.target_angle) < THRESHOLD:
		robot.is_aligned = True
		catch_ball(u_sonicU)
		print(f"Robot is aligned! Motor Values: L={L_motor_value}, R={R_motor_value}")
	else:
		print(f"Aligning... Motor Values: L={L_motor_value}, R={R_motor_value}")
