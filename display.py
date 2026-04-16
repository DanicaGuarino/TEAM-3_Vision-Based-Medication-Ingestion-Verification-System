from __future__ import annotations

from typing import Optional

import cv2
import numpy as np

from config import AppConfig
from state import SessionState
from utils import Box, Point


class DisplayRenderer:
    def __init__(self, config: AppConfig) -> None:
        self.config = config

    def draw(
        self,
        frame: np.ndarray,
        session: SessionState,
        current_time: float,
        hand_box: Optional[Box],
        pill_box: Optional[Box],
        pill_conf: float,
        mouth_center: Optional[Point],
        pill_in_hand_flag: bool,
    ) -> np.ndarray:
        if self.config.show_debug_boxes:
            self._draw_boxes(frame, hand_box, pill_box, pill_conf, mouth_center)

        if pill_in_hand_flag:
            cv2.putText(frame, "PILL IN HAND", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        if session.state == "HOLDING" and session.hold_start is not None:
            elapsed = max(0.0, current_time - session.hold_start)
            cv2.putText(
                frame,
                f"Holding {elapsed:.1f}s",
                (20, 120),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 255, 255),
                2,
            )

        result_color = (255, 255, 255)
        if session.final_result == "INGESTED":
            result_color = (0, 255, 0)
        elif session.final_result == "NOT CONFIRMED":
            result_color = (0, 255, 255)

        frame_height, _ = frame.shape[:2]
        cv2.putText(frame, f"State: {session.state}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        cv2.putText(
            frame,
            f"Final Result: {session.final_result}",
            (20, frame_height - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            result_color,
            2,
        )
        return frame

    def _draw_boxes(
        self,
        frame: np.ndarray,
        hand_box: Optional[Box],
        pill_box: Optional[Box],
        pill_conf: float,
        mouth_center: Optional[Point],
    ) -> None:
        if hand_box is not None:
            hx1, hy1, hx2, hy2 = hand_box
            cv2.rectangle(frame, (hx1, hy1), (hx2, hy2), (255, 0, 0), 2)

        if pill_box is not None:
            px1, py1, px2, py2 = pill_box
            cv2.rectangle(frame, (px1, py1), (px2, py2), (0, 255, 0), 2)
            cv2.putText(frame, f"Pill {pill_conf:.2f}", (px1, py1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        if mouth_center is not None:
            cv2.circle(frame, mouth_center, 6, (0, 0, 255), -1)
