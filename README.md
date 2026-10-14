# Tablet and Capsule Inspection

YOLO-based visual inspection prototype for identifying tablets labelled as **defected** or **no-defect**. The repository includes the dataset splits, dataset configuration, a trained YOLO11n checkpoint, and a webcam inference script.

> **Important:** This is an educational prototype, not a validated pharmaceutical quality-control system. Verify predictions on representative real-world samples before relying on them.

## Repository contents

- `tablet_inspection.py` — webcam inference and on-screen result display.
- `data.yaml` — dataset configuration for training/validation/testing.
- `train/images`, `valid/images`, `test/images` and matching labels — dataset splits.
- `runs/detect/train-2/weights/best.pt` — trained checkpoint used by the webcam script.
- `yolo11n.pt` — original pretrained YOLO11n checkpoint.
- `README.dataset.txt` — dataset source and licence information.
- `README.roboflow.txt` — export/preprocessing information.

## Requirements

- Python 3.10 or newer (Python 3.11 is a reasonable choice)
- A working webcam
- Windows, macOS, or Linux

Install the dependencies:

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

macOS/Linux:

```bash
source .venv/bin/activate
pip install -r requirements.txt
```

## Run webcam inspection

From the repository root:

```bash
python tablet_inspection.py
```

The application loads `runs/detect/train-2/weights/best.pt`, opens camera index `0`, and displays the top-confidence detected class. Press **Q** to quit. If your webcam is not camera 0, change `CAMERA_INDEX` in `tablet_inspection.py`.

If the trained checkpoint is missing, download/copy it into the expected path or update `MODEL_PATH` in the script.

## Train the detector (optional)

The dataset configuration uses repository-relative paths. After installing the requirements, run:

```bash
yolo detect train model=yolo11n.pt data=data.yaml epochs=10 imgsz=640
```

Training creates output under `runs/detect/`. The command above may create a new run directory instead of replacing the existing checkpoint.

## Dataset and attribution

The included dataset metadata identifies the source as [Roboflow Universe — Tablet Defect Detection](https://universe.roboflow.com/fyp-qjwy0/tablet-defect-detection-er87f) and states **CC BY 4.0**. Review the upstream dataset page and its terms before redistribution or commercial use.

## Known limitations

- Webcam predictions depend on lighting, focus, camera angle, tablet size, and how well the dataset matches real samples.
- A high validation score does not guarantee performance on unseen photos or a real production line.
- The current webcam script reports the highest-confidence detection in the frame; it is not a complete multi-tablet inspection/counting system.
- The current script does not control an ESP32 or buzzer. Hardware integration would need to be implemented and tested separately.
- Validate label files and dataset annotations before retraining, particularly if the source export contains mixed detection/segmentation labels.

## Suggested real-photo test

Place a representative tablet under steady lighting, capture a photo, and test it separately using the trained model before using the webcam live. Record false positives and false negatives as well as confidence.
