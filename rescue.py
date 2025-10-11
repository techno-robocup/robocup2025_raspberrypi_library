import modules.settings
import numpy as np
from enum import Enum
import time
import threading
import modules.log
import cv2

logger = modules.log.get_logger()
lock = threading.Lock()

P = 1.0
AP = 1.0

class ObjectClasses(Enum):
	BLACK_BALL = 0
	EXIT = 1
	GREEN_CAGE = 2
	RED_CAGE = 3
	SILVER_BALL = 4
Valid_Classes = []

BALL_CATCH_SIZE = 1000
Turn_Speed = 1600
First_Turn = False

class RobotState:
	def __init__(self):
		self.silver_ball_cnt = 0
		self.black_ball_cnt = 0
		self.is_ball_caching = False
		self.target_position = None
		self.target_size = None

L_Motor_Value = 1500
R_Motor_Value = 1500
Arm_Motor_Value = 1024
Wire_Motor_Value = 0
L_U_SONIC = 99
F_U_SONIC = 99
R_U_SONIC = 99


robot = RobotState()


def catch_ball():
	global Arm_Motor_Value,Wire_Motor_Value
	global L_Motor_Value,R_Motor_Value
	Arm_Motor_Value = 3072
	time.sleep(3)
	L_Motor_Value = 1550
	R_Motor_Value = L_Motor_Value
	time.sleep(0.5)
	L_Motor_Value = 1500
	R_Motor_Value = L_Motor_Value
	Wire_Motor_Value = 1
	Arm_Motor_Value = 1024
	robot.is_ball_caching = True
	time.sleep(2)
	return

def release_ball():
	global Arm_Motor_Value,Wire_Motor_Value
	global L_Motor_Value,R_Motor_Value
	L_Motor_Value = 1550
	R_Motor_Value = L_Motor_Value
	Arm_Motor_Value = 3072
	time.sleep(3)
	Wire_Motor_Value = 0
	Arm_Motor_Value = 1024
	robot.is_ball_caching = False
	if Valid_Classes == [ObjectClasses.GREEN_CAGE.value]:
		robot.silver_ball_cnt += 1
	else:
		robot.black_ball_cnt += 1
	time.sleep(2)
	L_Motor_Value = 1400
	R_Motor_Value = L_Motor_Value
	time.sleep(1)
	L_Motor_Value = 1500
	R_Motor_Value = L_Motor_Value
	return


def find_best_target(results,image_width):# TODO: turn 180 degrees
	"""Finds all valid targets and returns the one closest to the center."""
	global Valid_Classes
	boxes = results[0].boxes
	if not boxes:
		logger.debug("No boxes detected in YOLO results.")
		robot.target_position = None
		robot.target_size = None
		return None

	if robot.target_position is not None:
		logger.debug(f"Targeting class(es) {Valid_Classes}. Best target offset = {robot.target_position:.1f}")
	else:
		logger.debug(f"Targeting class(es) {Valid_Classes}. No target detected.")
	best_target_position = None
	min_dist_from_center = float("inf")
	image_center_x = image_width / 2

	for box in boxes:
		i_class = int(box.cls[0])
		if i_class in Valid_Classes:
			x_center, y_center, w, h = map(float, box.xywh[0])
			dist_from_center = x_center - image_center_x
			area = w * h

			if abs(dist_from_center) < min_dist_from_center:
				min_dist_from_center = abs(dist_from_center)
				best_target_position = dist_from_center
				best_target_area = area

	logger.debug(f"dist: {best_target_area}size:{best_target_area}")
	robot.target_position = best_target_position
	robot.target_size = best_target_area
	if best_target_position is not None:
		logger.debug(
			f"Target found: offset={best_target_position:.1f}, area={best_target_area:.1f}, classes={Valid_Classes}"
		)
	else:
		logger.debug(f"No valid target found for classes {Valid_Classes}.")

def change_position():
	global L_Motor_Value,R_Motor_Value
	"""Turn 180 degrees"""
	pass


def set_motor_speeds():
	global L_Motor_Value,R_Motor_Value

	diff_angle = robot.target_position * P
	if diff_angle < 1*P:
		diff_angle = 0

	if robot.is_ball_caching:
		dist = 100
	else:
		dist = BALL_CATCH_SIZE - robot.target_size * AP

	L_Motor_Value = 1500 + diff_angle + dist#TODO:Edit values
	R_Motor_Value = 1500 - diff_angle + dist


#def set_motor_speeds_from_position():
#	global L_Motor_Value, R_Motor_Value

#	with lock:
#		if robot.target_position is None:
#			L_motor_value = MOTOR_NEUTRAL
#			R_motor_value = MOTOR_NEUTRAL
#			logger.debug("No target to align. Stopping motors.")
#			return

#		if abs(robot.target_position) < THRESHOLD:
#			L_motor_value = MOTOR_NEUTRAL
#			R_motor_value = MOTOR_NEUTRAL
#		else:
#			turn_speed = KP * robot.target_position
#			turn_speed = max(min(turn_speed, MOTOR_MAX_TURN), -MOTOR_MAX_TURN)

#			L_motor_value = int(MOTOR_NEUTRAL + turn_speed)
#			R_motor_value = int(MOTOR_NEUTRAL - turn_speed)



#def turn_threaded(duration=1):
#	global L_Motor_Value, R_Motor_Value
#, turning
#	with lock:
#		turning = True
#		L_motor_value = int(TP * (MOTOR_NEUTRAL - MOTOR_MAX_TURN))
#		R_motor_value = int(TP * (MOTOR_NEUTRAL + MOTOR_MAX_TURN))
#		time.sleep(duration)
#		L_motor_value = MOTOR_NEUTRAL
#		R_motor_value = MOTOR_NEUTRAL
#		turning = False


def rescue_loop_func():
	global L_Motor_Value,R_Motor_Value

	logger.debug("call rescue_loop_func")
	#robot.is_aligned = False

	if modules.settings.yolo_results is None:
		logger.debug("No YOLO results available")
		return

	results = modules.settings.yolo_results
	image_width = results[0].orig_shape[1]

	if robot.silver_ball_cnt== 2 and robot.black_ball_cnt == 1:
		logger.debug("Find:Exit")
		Valid_Classes = [ObjectClasses.EXIT.value]
	else:
		if not robot.is_ball_caching:
			if robot.silver_ball_cnt < 2:
				logger.debug("Find:Silver")
				Valid_Classes = [ObjectClasses.SILVER_BALL.value]
			else:
				logger.debug("Find:Black")
				Valid_Classes = [ObjectClasses.BLACK_BALL.value]
		else:
			if robot.silver_ball_cnt < 2:
				logger.debug("Find:GREEN")
				Valid_Classes = [ObjectClasses.GREEN_CAGE.value]
			else:
				logger.debug("Find:RED")
				Valid_Classes = [ObjectClasses.RED_CAGE.value]

	if modules.settings.DEBUG_MODE:
		annotated = results[0].plot()
		cv2.imwrite(f"bin/{time.time():.3f}_rescue_loop_detections.jpg", annotated)

	# Find best target from detections
	find_best_target(results, image_width)# NOTE: Find target, return dist and area size

	if robot.target_position is not None:
		logger.debug(f"Targeting class(es) {Valid_Classes}. Best target offset = {robot.target_position:.1f} , {robot.target_size}")# TODO: continue 180 degrees
	else:
		logger.debug(f"Targeting class(es) {Valid_Classes}. No target detected.")

	if not robot.is_ball_caching and robot.target_size is not None and robot.target_size >= BALL_CATCH_SIZE:
		catch_ball()
	elif robot.is_ball_caching and F_U_SONIC < 1:
		release_ball()
	else:
		if robot.target_position is not None:
			set_motor_speeds()
		else:
			change_position()
	# Alignment status
	logger.debug(f"Aligning... Motor Values: L={L_Motor_Value}, R={R_Motor_Value}")
