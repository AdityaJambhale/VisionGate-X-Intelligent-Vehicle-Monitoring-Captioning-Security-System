import cv2
import os

VIDEO_PATH = "data/sample_videos/bike1.mp4"

OUTPUT_DIR = "data/extracted_frames"

FRAME_SKIP = 5

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)

cap = cv2.VideoCapture(
    VIDEO_PATH
)

frame_count = 0
saved_count = 0

while True:

    ret, frame = cap.read()

    if not ret:
        break

    # Save every Nth frame
    if frame_count % FRAME_SKIP == 0:

        frame_path = os.path.join(
            OUTPUT_DIR,
            f"frame_{saved_count:04d}.jpg"
        )

        cv2.imwrite(
            frame_path,
            frame
        )

        print(
            f"Saved: {frame_path}"
        )

        saved_count += 1

    frame_count += 1

cap.release()

print("\nDone extracting frames.")