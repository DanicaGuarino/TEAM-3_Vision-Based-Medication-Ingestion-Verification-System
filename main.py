from __future__ import annotations

import time

import cv2

from config import AppConfig
from detectors import DetectionPipeline
from display import DisplayRenderer
from logger import TrialLogger
from utils import distance_between, enhance_low_light
from verifier import FrameInputs, IngestionVerifier


def main() -> None:
    config = AppConfig()
    detector = DetectionPipeline(config)
    logger = TrialLogger(config.csv_path)
    verifier = IngestionVerifier(config, logger)
    renderer = DisplayRenderer(config)

    cap = cv2.VideoCapture(config.camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open camera index {config.camera_index}.")

    print("Prototype started.")
    print("Press ESC to close the application.")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[WARN] Failed to read frame from camera.")
                break

            current_time = time.time()
            frame = enhance_low_light(frame)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            hand = detector.detect_hand(rgb, frame.shape)
            pill = detector.detect_pill(frame, hand.box)
            mouth = detector.detect_mouth(rgb, frame.shape)

            distance_to_mouth = distance_between(hand.center, mouth.center)
            session = verifier.update(
                FrameInputs(
                    pill_detected=pill.box is not None,
                    pill_in_hand=pill.in_hand,
                    hand_visible=hand.box is not None,
                    mouth_open=mouth.is_open,
                    distance_to_mouth=distance_to_mouth,
                ),
                current_time,
            )

            frame = renderer.draw(
                frame,
                session,
                current_time=current_time,
                hand_box=hand.box,
                pill_box=pill.box,
                pill_conf=pill.confidence,
                mouth_center=mouth.center,
                pill_in_hand_flag=pill.in_hand,
            )

            cv2.imshow(config.window_name, frame)
            if (cv2.waitKey(1) & 0xFF) == 27:
                break
    finally:
        cap.release()
        detector.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
