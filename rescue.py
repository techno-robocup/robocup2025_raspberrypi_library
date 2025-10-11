import modules.settings
import numpy as np
from enum import Enum
import time
import threading
import modules.log
import cv2

logger = modules.log.get_logger()
lock = threading.Lock()


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
	#min_dist_from_center = float("inf")
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


			logger.debug(f"size:{area}")
				#if robot.is_ball_caching and F_U_SONIC <= 0.5:# TODO: Use u_sonic in run_motor
				#	release_ball()
				#elif not robot.is_ball_caching and area > BALL_CATCH_SIZE:
				#	catch_ball()
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
	


def run_to_target():
	if robot.target_position is None:
		return# TODO: Turn or change position
	else:
		return


#def set_motor_speeds_from_position():
#	global L_motor_value, R_motor_value
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
#	global L_motor_value, R_motor_value, turning
#	with lock:
#		turning = True
#		L_motor_value = int(TP * (MOTOR_NEUTRAL - MOTOR_MAX_TURN))
#		R_motor_value = int(TP * (MOTOR_NEUTRAL + MOTOR_MAX_TURN))
#		time.sleep(duration)
#		L_motor_value = MOTOR_NEUTRAL
#		R_motor_value = MOTOR_NEUTRAL
#		turning = False


#def turn():
#	if not turning:
#		t = threading.Thread(target=turn_threaded, args=(1,))
#		t.start()


#def catch_ball(area,f_u_sonic):
#	"""Camera-based distance estimation: use bounding box area."""
#	global R_motor_value, L_motor_value, Release_flag
#	Release_flag = False

#	if not robot.is_ball_caching:
#		if area > BALL_CATCH_SIZE:
#			L_motor_value = MOTOR_NEUTRAL
#			R_motor_value = MOTOR_NEUTRAL
#			Release_flag = True
#			if not robot.is_ball_caching:
#				robot.is_ball_caching = True
#			else:
#				if robot.silver_ball_cnt < 2:
#					robot.silver_ball_cnt += 1
#					robot.is_ball_caching = False
#				else:
#					robot.black_ball_cnt += 1
#					robot.is_ball_caching = False
#					if robot.black_ball_cnt == 1 and robot.silver_ball_cnt == 2:
#						robot.is_task_done = True
#	else:
#		if f_u_sonic <= 5:#TODO:Fix value
#			Release_flag = True
#			L_motor_value = MOTOR_NEUTRAL
#			R_motor_value = MOTOR_NEUTRAL
#			robot.is_ball_caching = False


def rescue_loop_func():
	global L_motor_value, R_motor_value

	logger.debug("call rescue_loop_func")
	#robot.is_aligned = False

	if modules.settings.yolo_results is None:
		logger.debug("No YOLO results available")
		return

	results = modules.settings.yolo_results
	boxes = results[0].boxes
	image_width = results[0].orig_shape[1]

	if robot.silver_ball_cnt < 2 or robot.black_ball_cnt < 1:
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
	robot.target_position = find_best_target(results, image_width)# NOTE: Find target, return dist and area size

	if robot.target_position is not None:
		logger.debug(f"Targeting class(es) {Valid_Classes}. Best target offset = {robot.target_position:.1f} , {robot.target_size}")# TODO: continue 180 degrees
	else:
		logger.debug(f"Targeting class(es) {Valid_Classes}. No target detected.")

	if not(robot.is_ball_caching) and robot.target_size <= BALL_CATCH_SIZE:
		catch_ball()
	elif robot.is_ball_caching and F_U_SONIC < 1:
		release_ball()
	else:
		return# TODO: run for target
	# Alignment status
	logger.debug(f"Aligning... Motor Values: L={L_motor_value}, R={R_motor_value}")
