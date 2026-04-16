from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
from mediapipe import solutions
from ultralytics import YOLO

from config import AppConfig
from utils import Box, Point, box_center, compute_mouth_metrics, pill_in_hand, scale_box


@dataclass
class HandResult:
    box: Optional[Box]
    center: Optional[Point]


@dataclass
class MouthResult:
    center: Optional[Point]
    is_open: bool


@dataclass
class PillResult:
    box: Optional[Box]
    confidence: float
    in_hand: bool


class DetectionPipeline:
    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.model = YOLO(config.model_path)
        self.hands = solutions.hands.Hands(
            max_num_hands=1,
            min_detection_confidence=config.hand_detection_confidence,
            min_tracking_confidence=config.hand_tracking_confidence,
        )
        self.face = solutions.face_mesh.FaceMesh(
            max_num_faces=1,
            min_detection_confidence=config.face_detection_confidence,
            min_tracking_confidence=config.face_tracking_confidence,
        )

    def detect_hand(self, rgb_frame, frame_shape: Tuple[int, int, int]) -> HandResult:
        result = self.hands.process(rgb_frame)
        if not result.multi_hand_landmarks:
            return HandResult(box=None, center=None)

        frame_width = frame_shape[1]
        frame_height = frame_shape[0]
        landmarks = result.multi_hand_landmarks[0]
        xs = [point.x * frame_width for point in landmarks.landmark]
        ys = [point.y * frame_height for point in landmarks.landmark]

        hand_box = scale_box(
            (int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))),
            scale=self.config.hand_box_scale,
        )
        return HandResult(box=hand_box, center=box_center(hand_box))

    def detect_pill(self, frame, hand_box: Optional[Box]) -> PillResult:
        if hand_box is None:
            return PillResult(box=None, confidence=0.0, in_hand=False)

        frame_height, frame_width = frame.shape[:2]
        hx1, hy1, hx2, hy2 = hand_box

        sx1 = max(0, hx1 - self.config.hand_crop_padding)
        sy1 = max(0, hy1 - self.config.hand_crop_padding)
        sx2 = min(frame_width, hx2 + self.config.hand_crop_padding)
        sy2 = min(frame_height, hy2 + self.config.hand_crop_padding)

        hand_crop = frame[sy1:sy2, sx1:sx2]
        if hand_crop.size == 0:
            return PillResult(box=None, confidence=0.0, in_hand=False)

        results = self.model.predict(hand_crop, verbose=False)
        best_box = None
        best_conf = 0.0
        best_score = -1.0

        for result in results:
            if result.boxes is None:
                continue

            boxes = result.boxes.xyxy.cpu().numpy()
            confs = result.boxes.conf.cpu().numpy()

            for index, conf in enumerate(confs):
                if conf < self.config.min_pill_conf:
                    continue

                x1, y1, x2, y2 = map(int, boxes[index])
                width = x2 - x1
                height = y2 - y1
                area = width * height
                if area > self.config.max_pill_box_area:
                    continue

                score = float(conf) - (area / 10000.0)
                if score <= best_score:
                    continue

                best_score = score
                best_conf = float(conf)
                best_box = scale_box((x1, y1, x2, y2), scale=self.config.pill_box_scale)

        if best_box is None:
            return PillResult(box=None, confidence=0.0, in_hand=False)

        x1, y1, x2, y2 = best_box
        full_frame_box = (x1 + sx1, y1 + sy1, x2 + sx1, y2 + sy1)
        return PillResult(
            box=full_frame_box,
            confidence=best_conf,
            in_hand=pill_in_hand(full_frame_box, hand_box, self.config.overlap_threshold),
        )

    def detect_mouth(self, rgb_frame, frame_shape: Tuple[int, int, int]) -> MouthResult:
        result = self.face.process(rgb_frame)
        if not result.multi_face_landmarks:
            return MouthResult(center=None, is_open=False)

        mouth_center, mar = compute_mouth_metrics(result.multi_face_landmarks[0], frame_shape)
        return MouthResult(center=mouth_center, is_open=mar > self.config.mouth_open_threshold)

    def close(self) -> None:
        self.hands.close()
        self.face.close()
