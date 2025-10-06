import cv2
from ultralytics import YOLO

model = YOLO("best.pt")

angle = 0
silver_ball_cnt = 0
black_ball_cnt = 0
is_ball_caching = False
target_angle = None

def get_centers(image_path) -> None:# TODO: Check class,cnt
		global target_angle, is_ball_caching, black_ball_cnt, silver_ball_cnt

		results = model(image_path, verbose=False)
		result = results[0]
		boxes = result.boxes

		target_angle = None

		if not is_ball_caching:
				for box in boxes:
						cls = int(box.cls[0])
						if cls == 0 and black_ball_cnt < 2:
								x_center = float(box.xywh[0][0])
								target_angle = x_center
								break
						elif cls == 4 and black_ball_cnt == 2:
								x_center = float(box.xywh[0][0])
								target_angle = x_center
								break

		else:
				for box in boxes:
						cls = int(box.cls[0])
						if black_ball_cnt < 2 and cls == 1:
								x_center = float(box.xywh[0][0])
								target_angle = x_center
								break
						elif black_ball_cnt >= 2 and cls == 2:
								x_center = float(box.xywh[0][0])
								target_angle = x_center
								break

		if target_angle is not None:
				print(f"Detected target x_center = {target_angle:.1f}")
		else:
				print("No target detected.")

