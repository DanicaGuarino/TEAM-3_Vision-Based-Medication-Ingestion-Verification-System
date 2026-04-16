from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class AppConfig:
    model_path: str = "models/pill_detector.pt"
    csv_path: str = "trial_results.csv"
    camera_index: int = 0
    window_name: str = "Medication Ingestion Verification"

    min_pill_conf: float = 0.40
    approach_dist: int = 80
    release_dist: int = 120
    hold_time: float = 1.5
    session_timeout: float = 8.0
    result_display_time: float = 3.0
    ready_confirm_time: float = 1.0
    mouth_open_threshold: float = 0.02
    mouth_radius: int = 55
    hand_crop_padding: int = 80

    hand_detection_confidence: float = 0.5
    hand_tracking_confidence: float = 0.5
    face_detection_confidence: float = 0.5
    face_tracking_confidence: float = 0.5

    overlap_threshold: float = 0.03
    hand_box_scale: float = 1.15
    pill_box_scale: float = 1.0
    max_pill_box_area: int = 6000

    show_debug_boxes: bool = True

    @property
    def csv_file(self) -> Path:
        return Path(self.csv_path)
