
import cv2
import time
from pathlib import Path
from ultralytics import YOLO

# SETTINGS
MODEL_PATH = Path("runs/detect/train-2/weights/best.pt")
CAMERA_INDEX = 0
CONFIDENCE = 0.15
IMAGE_SIZE = 640

# LOAD TRAINED MODEL
if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model not found: {MODEL_PATH.resolve()}")

model = YOLO(str(MODEL_PATH))
print("Model loaded:", model.names)

# OPEN WEBCAM
camera = cv2.VideoCapture(CAMERA_INDEX)

if not camera.isOpened():
    raise SystemExit("Cannot open webcam. Check camera connection.")

camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

print("Tablet quality inspection started. Press Q to quit.")

last_prediction = "PLACE TABLET IN VIEW"
last_confidence = 0.0
last_time = 0.0
fps = 0.0
previous_time = time.time()

try:
    while True:
        success, frame = camera.read()

        if not success:
            print("Cannot read camera.")
            break

        current_time = time.time()

        # Run inference approximately every 0.4 seconds
        if current_time - last_time >= 0.4:
            result = model.predict(
                source=frame,
                conf=CONFIDENCE,
                imgsz=IMAGE_SIZE,
                device="cpu",
                max_det=20,
                verbose=False
            )[0]

            last_time = current_time

            if len(result.boxes) > 0:
                # Use the highest-confidence prediction
                best_index = int(result.boxes.conf.argmax().item())
                best_box = result.boxes[best_index]

                class_id = int(best_box.cls.item())
                last_confidence = float(best_box.conf.item())
                class_name = str(model.names[class_id]).lower()

                if class_name in ("defected", "defective", "defect"):
                    last_prediction = "DEFECTIVE"
                elif class_name in ("no-defect", "no_defect",
                                    "non-defective", "non defective"):
                    last_prediction = "NON-DEFECTIVE"
                else:
                    last_prediction = "UNKNOWN CLASS"

            else:
                last_prediction = "CANNOT CLASSIFY"
                last_confidence = 0.0

        # FPS
        elapsed = current_time - previous_time
        if elapsed > 0:
            fps = 1.0 / elapsed
        previous_time = current_time

        # Display without object bounding boxes
        display = frame.copy()

        cv2.putText(
            display, "TABLET QUALITY INSPECTION",
            (15, 35), cv2.FONT_HERSHEY_SIMPLEX,
            0.75, (255, 255, 255), 2
        )

        if last_prediction == "DEFECTIVE":
            status_color = (0, 0, 255)
        elif last_prediction == "NON-DEFECTIVE":
            status_color = (0, 200, 0)
        else:
            status_color = (0, 200, 255)

        cv2.putText(
            display, "RESULT: " + last_prediction,
            (15, 100), cv2.FONT_HERSHEY_SIMPLEX,
            0.85, status_color, 3
        )

        if last_prediction in ("DEFECTIVE", "NON-DEFECTIVE"):
            cv2.putText(
                display,
                f"Model confidence: {last_confidence:.1%}",
                (15, 145), cv2.FONT_HERSHEY_SIMPLEX,
                0.65, (255, 255, 255), 2
            )
        else:
            cv2.putText(
                display,
                "Adjust tablet position or lighting",
                (15, 145), cv2.FONT_HERSHEY_SIMPLEX,
                0.55, (255, 255, 255), 2
            )

        cv2.putText(
            display, f"FPS: {fps:.1f} | Press Q to quit",
            (15, 190), cv2.FONT_HERSHEY_SIMPLEX,
            0.55, (255, 255, 255), 2
        )

        cv2.imshow("Tablet Quality Inspection", display)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

finally:
    camera.release()
    cv2.destroyAllWindows()
    print("Inspection stopped.")
