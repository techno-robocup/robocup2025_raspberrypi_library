import cv2
import settings
from ultralytics import YOLO

model = YOLO("best.pt")

silver_ball_cnt = 0
black_ball_cnt = 0
is_ball_caching = False
is_task_done = False

L_motor_value = 1500
R_motor_value = 1500

threshold = 5.0
kp = 0.1

target_angle = None

#Rescue Cam Size:4608, 2592
def get_centers(image_path) -> None:
		global target_angle, is_ball_caching, black_ball_cnt, silver_ball_cnt

		img = cv2.imread(image_path)
		if img is None:
				print(f"Error: could not load {image_path}")
				return

		h, w, _ = img.shape
		cx = w // 2
		target_angle = None

		results = model(image_path, verbose=False)
		boxes = results[0].boxes

		if not (black_ball_cnt == 2 and silver_ball_cnt == 0):
				if not is_ball_caching:
						for box in boxes:
								cls = int(box.cls[0])
								if cls == 0 and black_ball_cnt < 2:
										x_center = float(box.xywh[0][0])
										target_angle = x_center - cx
										break
								elif cls == 4 and black_ball_cnt == 2:
										x_center = float(box.xywh[0][0])
										target_angle = x_center - cx
										break
				else:
						for box in boxes:
								cls = int(box.cls[0])
								if black_ball_cnt < 2 and cls == 1:
										x_center = float(box.xywh[0][0])
										target_angle = x_center - cx
										break
								elif black_ball_cnt >= 2 and cls == 2:
										x_center = float(box.xywh[0][0])
										target_angle = x_center - cx
										break
		else:
				for box in boxes:
						cls = int(box.cls[0])
						if cls == 5:
								x_center = float(box.xywh[0][0])
								target_angle = x_center - cx
								break

		if target_angle is not None:
				print(f"Detected target offset = {target_angle:.1f}")
		else:
				print("No target detected.")


def set_angle_PID():
		global L_motor_value, R_motor_value, target_angle

		if target_angle is None:
				print("No target to align.")
				return

		if abs(target_angle) < threshold:
				L_motor_value = 1500
				R_motor_value = 1500
		else:
				turn_speed = kp * target_angle
				turn_speed = max(min(turn_speed, 100), -100)

				L_motor_value = int(1500 + turn_speed)
				R_motor_value = int(1500 - turn_speed)
