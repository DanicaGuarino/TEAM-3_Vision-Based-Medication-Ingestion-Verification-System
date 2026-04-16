from __future__ import annotations

from typing import Optional, Tuple

import cv2
import numpy as np

Box = Tuple[int, int, int, int]
Point = Tuple[int, int]


def enhance_low_light(frame: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    enhanced_l = clahe.apply(l_channel)

    merged = cv2.merge((enhanced_l, a_channel, b_channel))
    return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)


def scale_box(box: Box, scale: float = 1.0) -> Box:
    x1, y1, x2, y2 = box
    width = x2 - x1
    height = y2 - y1
    center_x = x1 + width / 2
    center_y = y1 + height / 2

    new_width = width * scale
    new_height = height * scale

    return (
        int(center_x - new_width / 2),
        int(center_y - new_height / 2),
        int(center_x + new_width / 2),
        int(center_y + new_height / 2),
    )


def pill_in_hand(pill_box: Box, hand_box: Box, overlap_thresh: float = 0.03) -> bool:
    px1, py1, px2, py2 = pill_box
    hx1, hy1, hx2, hy2 = hand_box

    inter_x1 = max(px1, hx1)
    inter_y1 = max(py1, hy1)
    inter_x2 = min(px2, hx2)
    inter_y2 = min(py2, hy2)

    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    pill_area = max(1, (px2 - px1) * (py2 - py1))
    return (inter_area / pill_area) >= overlap_thresh


def compute_mouth_metrics(face_landmarks, frame_shape: Tuple[int, int, int]) -> Tuple[Point, float]:
    height, width = frame_shape[:2]

    upper_lip = 13
    lower_lip = 14
    left_mouth = 61
    right_mouth = 291

    ux = int(face_landmarks.landmark[upper_lip].x * width)
    uy = int(face_landmarks.landmark[upper_lip].y * height)
    lx = int(face_landmarks.landmark[lower_lip].x * width)
    ly = int(face_landmarks.landmark[lower_lip].y * height)
    lmx = int(face_landmarks.landmark[left_mouth].x * width)
    lmy = int(face_landmarks.landmark[left_mouth].y * height)
    rmx = int(face_landmarks.landmark[right_mouth].x * width)
    rmy = int(face_landmarks.landmark[right_mouth].y * height)

    vertical = np.linalg.norm(np.array([ux, uy]) - np.array([lx, ly]))
    horizontal = np.linalg.norm(np.array([lmx, lmy]) - np.array([rmx, rmy])) + 1e-6
    mouth_aspect_ratio = vertical / horizontal
    mouth_center = (int((ux + lx) / 2), int((uy + ly) / 2))
    return mouth_center, mouth_aspect_ratio


def box_center(box: Optional[Box]) -> Optional[Point]:
    if box is None:
        return None
    x1, y1, x2, y2 = box
    return (int((x1 + x2) / 2), int((y1 + y2) / 2))


def distance_between(point_a: Optional[Point], point_b: Optional[Point]) -> Optional[float]:
    if point_a is None or point_b is None:
        return None
    return float(np.linalg.norm(np.array(point_a) - np.array(point_b)))
