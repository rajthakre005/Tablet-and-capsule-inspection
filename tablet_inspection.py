import time
from pathlib import Path

import cv2
from ultralytics import YOLO

try:
    import serial
    from serial import SerialException
except ImportError:
    serial = None
    SerialException = Exception


# -------------------- SETTINGS --------------------
PROJECT_DIR = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_DIR / "runs" / "detect" / "train-2" / "weights" / "best.pt"

CAMERA_INDEX = 0
CONFIDENCE = 0.10
IMAGE_SIZE = 640

# ESP32 settings. Change ESP32_PORT to the COM port shown in Arduino IDE.
ESP32_ENABLED = True
ESP32_PORT = "COM3"
ESP32_BAUD = 115200
SERIAL_TIMEOUT = 1

# Fail-safe policy:
# If YOLO sees no object, do NOT declare it non-defective.
# Display REJECT - NO DETECTION and send ALERT to the ESP32.
NO_DETECTION_STATUS = "REJECT - NO DETECTION"


def send_esp32_command(connection, command):
    """Send ALERT or SAFE to the ESP32, if serial is connected."""
    if connection is None:
        return
    try:
        connection.write((command + "\n").encode("utf-8"))
    except (SerialException, OSError) as exc:
        print(f"ESP32 serial write failed: {exc}")


# -------------------- LOAD MODEL --------------------
if not MODEL_PATH.is_file():
    raise FileNotFoundError(
        f"Trained model not found: {MODEL_PATH}\n"
        "Check that runs/detect/train-2/weights/best.pt exists."
    )

model = YOLO(str(MODEL_PATH))
print("Model loaded:", model.names)

# -------------------- CONNECT ESP32 --------------------
esp = None
if ESP32_ENABLED:
    if serial is None:
        print("WARNING: pyserial is not installed. ESP32 alerts are disabled.")
        print("Install it with: python -m pip install pyserial")
    else:
        try:
            esp = serial.Serial(
                ESP32_PORT, ESP32_BAUD, timeout=SERIAL_TIMEOUT
            )
            time.sleep(2)  # ESP32 may reset when the serial port opens.
            print(f"ESP32 connected on {ESP32_PORT} at {ESP32_BAUD} baud.")
        except (SerialException, OSError) as exc:
            print(f"WARNING: Could not connect to ESP32 on {ESP32_PORT}: {exc}")
            print("Camera inspection will continue without buzzer alerts.")

# -------------------- OPEN WEBCAM --------------------
camera = cv2.VideoCapture(CAMERA_INDEX)
if not camera.isOpened():
    if esp is not None:
        esp.close()
    raise SystemExit(
        "Cannot open webcam. Close other camera apps or change CAMERA_INDEX."
    )

camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

print("Tablet quality inspection started. Press Q in the camera window to quit.")

last_prediction = "PLACE TABLET IN VIEW"
last_confidence = 0.0
last_time = 0.0
last_esp_command = None
fps = 0.0
previous_time = time.time()

try:
    while True:
        success, frame = camera.read()
        if not success:
            print("Cannot read camera frame.")
            break

        current_time = time.time()

        # Run inference about 2.5 times per second.
        if current_time - last_time >= 0.4:
            result = model.predict(
                source=frame,
                conf=CONFIDENCE,
                imgsz=IMAGE_SIZE,
                device="cpu",
                max_det=20,
                verbose=False,
            )[0]
            last_time = current_time

            if len(result.boxes) > 0:
                # Use the highest-confidence detection.
                best_index = int(result.boxes.conf.argmax().item())
                best_box = result.boxes[best_index]
                class_id = int(best_box.cls.item())
                last_confidence = float(best_box.conf.item())
                class_name = str(model.names[class_id]).strip().lower()

                if class_name in ("defected", "defective", "defect"):
                    last_prediction = "DEFECTIVE"
                elif class_name in (
                    "no-defect", "no_defect", "no defect",
                    "non-defective", "non defective", "non_defective"
                ):
                    last_prediction = "NON-DEFECTIVE"
                else:
                    # Unknown labels are treated as a rejection, not a pass.
                    last_prediction = "REJECT - UNKNOWN CLASS"
            else:
                last_prediction = NO_DETECTION_STATUS
                last_confidence = 0.0

            # Buzzer alert for defective/reject; SAFE for a predicted pass.
            # This expects ESP32 firmware to understand ALERT and SAFE lines.
            command = (
                "SAFE" if last_prediction == "NON-DEFECTIVE" else "ALERT"
            )
            if command != last_esp_command:
                send_esp32_command(esp, command)
                last_esp_command = command
                print(
                    f"Result: {last_prediction} | "
                    f"Confidence: {last_confidence:.1%} | ESP32: {command}"
                )

        elapsed = current_time - previous_time
        if elapsed > 0:
            fps = 1.0 / elapsed
        previous_time = current_time

        display = frame.copy()

        # Draw detections to help with camera/model alignment.
        for box in result.boxes if "result" in locals() else []:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            cv2.rectangle(display, (x1, y1), (x2, y2), (255, 255, 255), 2)

        cv2.putText(
            display, "TABLET / CAPSULE QUALITY INSPECTION",
            (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2
        )

        if last_prediction == "NON-DEFECTIVE":
            status_color = (0, 200, 0)
        elif last_prediction == "DEFECTIVE" or last_prediction.startswith("REJECT"):
            status_color = (0, 0, 255)
        else:
            status_color = (0, 200, 255)

        cv2.putText(
            display, "RESULT: " + last_prediction,
            (15, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.75, status_color, 3
        )

        if last_prediction in ("DEFECTIVE", "NON-DEFECTIVE"):
            confidence_text = f"Model confidence: {last_confidence:.1%}"
        elif last_prediction.startswith("REJECT"):
            confidence_text = "Fail-safe: do not pass without a detection"
        else:
            confidence_text = "Place one tablet/capsule in view"

        cv2.putText(
            display, confidence_text,
            (15, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 2
        )

        esp_text = "ESP32: CONNECTED" if esp is not None else "ESP32: NOT CONNECTED"
        cv2.putText(
            display, f"{esp_text} | FPS: {fps:.1f} | Press Q to quit",
            (15, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 2
        )

        cv2.imshow("Tablet Quality Inspection", display)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

finally:
    camera.release()
    cv2.destroyAllWindows()
    if esp is not None:
        try:
            esp.close()
        except Exception:
            pass
    print("Inspection stopped.")
