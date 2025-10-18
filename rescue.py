import modules.settings
from enum import Enum
import time
import modules.log

logger = modules.log.get_logger()

P = 1.0
AP = 1.0
CP = 1.0
BALL_CATCH_SIZE = 1000
TURN_45_TIME = 0.45
TURN_180_TIME = 1.0
FORWARD_STEP_TIME = 0.3
WALL_DIST_THRESHOLD = 5.0
FRONT_CLEAR_THRESHOLD = 3.0
MOTOR_MIN = 1000
MOTOR_MAX = 2000
MOTOR_NEUTRAL = 1500


class ObjectClasses(Enum):
  BLACK_BALL = 0
  EXIT = 1
  GREEN_CAGE = 2
  RED_CAGE = 3
  SILVER_BALL = 4


Valid_Classes = []


class RobotState:

  def __init__(self):
    self.silver_ball_cnt = 0
    self.black_ball_cnt = 0
    self.is_ball_caching = False
    self.target_position = None
    self.target_size = None
    self.cnt_turning_degrees = 0
    self.cnt_turning_side = 0


robot = RobotState()

L_Motor_Value = MOTOR_NEUTRAL
R_Motor_Value = MOTOR_NEUTRAL
Arm_Motor_Value = 1024
Wire_Motor_Value = 0
L_U_SONIC = None
F_U_SONIC = None
R_U_SONIC = None


def catch_ball():
  global Arm_Motor_Value, Wire_Motor_Value, L_Motor_Value, R_Motor_Value
  logger.debug("Executing catch_ball()")
  Arm_Motor_Value = 3072
  time.sleep(3)
  L_Motor_Value = 1550
  R_Motor_Value = L_Motor_Value
  time.sleep(0.5)
  L_Motor_Value = MOTOR_NEUTRAL
  R_Motor_Value = MOTOR_NEUTRAL
  Wire_Motor_Value = 1
  Arm_Motor_Value = 1024
  robot.is_ball_caching = True
  time.sleep(1.5)
  logger.debug("catch_ball done")


def release_ball():
  global Arm_Motor_Value, Wire_Motor_Value, L_Motor_Value, R_Motor_Value
  logger.debug("Executing release_ball()")
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
  L_Motor_Value = MOTOR_NEUTRAL
  R_Motor_Value = MOTOR_NEUTRAL
  logger.debug("release_ball done")


def find_best_target(results, image_width):
  global Valid_Classes
  boxes = results[0].boxes
  if not boxes:
    robot.target_position = None
    robot.target_size = None
    return None
  best_target_pos = None
  best_target_area = None
  min_dist = float("inf")
  cx = image_width / 2.0
  for box in boxes:
    try:
      cls = int(box.cls[0])
    except Exception:
      continue
    if cls in Valid_Classes:
      x_center, y_center, w, h = map(float, box.xywh[0])
      dist = x_center - cx
      area = w * h
      if abs(dist) < min_dist:
        min_dist = abs(dist)
        best_target_pos = dist
        best_target_area = area
      logger.debug(f"Detected cls={cls}, area={area:.1f}, offset={dist:.1f}")
  robot.target_position = best_target_pos
  robot.target_size = best_target_area
  if best_target_pos is not None:
    logger.debug(
        f"Target found offset={best_target_pos:.1f}, area={best_target_area:.1f}"
    )
  else:
    logger.debug("No valid target found")
  return best_target_pos


def change_position():
  global L_Motor_Value, R_Motor_Value, F_U_SONIC, L_U_SONIC
  logger.debug("change_position() called")
  if robot.cnt_turning_degrees >= 90:
    if F_U_SONIC is not None and F_U_SONIC >= 3:
      L_Motor_Value = 1700 - (L_U_SONIC * CP)
      R_Motor_Value = 1700 + (L_U_SONIC * CP)
    elif F_U_SONIC <= 3:
      L_Motor_Value = 2000
      R_Motor_Value = 1000
      time.sleep(TURN_180_TIME)
      L_Motor_Value = 1500
      R_Motor_Value = 1500
      robot.cnt_turning_side += 1
      robot.cnt_turning_degrees = 0
    elif F_U_SONIC is None:
      L_Motor_Value = MOTOR_NEUTRAL
      R_Motor_Value = MOTOR_NEUTRAL
    else:
      L_Motor_Value = 1750
      R_Motor_Value = 1250
      time.sleep(TURN_45_TIME)
      L_Motor_Value = 1500
      R_Motor_Value = 1500
      robot.cnt_turning_degrees += 45
  return


def set_motor_speeds():
  global L_Motor_Value, R_Motor_Value, robot
  if robot.target_position is None or robot.target_size is None:
    L_Motor_Value = 1500
    R_Motor_Value = 1500
    logger.debug("No target data for set_motor_speeds(), stopping motors.")
    return
  diff_angle = robot.target_position * P
  if robot.is_ball_caching:
    dist_term = 100
  else:
    dist_term = BALL_CATCH_SIZE - robot.target_size * AP
  base_L = 1500 + diff_angle + dist_term
  base_R = 1500 - diff_angle + dist_term
  L_Motor_Value = int(min(max(base_L, 1000), 2000))
  R_Motor_Value = int(min(max(base_R, 1000), 2000))
  logger.debug(f"Motor speed L:{L_Motor_Value}, R:{R_Motor_Value}")


def rescue_loop_func():
  global L_Motor_Value, R_Motor_Value, Valid_Classes
  if modules.settings.yolo_results is None:
    logger.debug("No YOLO results available, stopping motors.")
    L_Motor_Value = MOTOR_NEUTRAL
    R_Motor_Value = MOTOR_NEUTRAL
    return

  results = modules.settings.yolo_results
  image_width = results[0].orig_shape[1]

  if robot.silver_ball_cnt == 2 and robot.black_ball_cnt == 1:
    Valid_Classes = [ObjectClasses.EXIT.value]
  elif not robot.is_ball_caching:
    Valid_Classes = [
        ObjectClasses.SILVER_BALL.value
    ] if robot.silver_ball_cnt < 2 else [ObjectClasses.BLACK_BALL.value]
  else:
    Valid_Classes = [
        ObjectClasses.GREEN_CAGE.value
    ] if robot.silver_ball_cnt < 2 else [ObjectClasses.RED_CAGE.value]

  find_best_target(results, image_width)

  if robot.target_position is None:
    logger.debug("No target found -> executing change_position()")
    change_position()
    return
  else:
    robot.cnt_turning_degrees = 0
    if not robot.is_ball_caching and robot.target_size >= BALL_CATCH_SIZE:
      logger.debug(
          f"Target is close (size: {robot.target_size:.1f}). Initiating catch_ball()"
      )
      catch_ball()
      return
    if robot.is_ball_caching and F_U_SONIC is not None and F_U_SONIC < 1.0:
      logger.debug(
          f"Close to wall (dist: {F_U_SONIC:.1f}). Initiating release_ball()")
      release_ball()
      return

    logger.debug(
        f"Targeting {Valid_Classes}, offset={robot.target_position:.1f}. Navigating..."
    )
    set_motor_speeds()

  logger.debug(f"Motor Values after run: L={L_Motor_Value}, R={R_Motor_Value}")
