from __future__ import annotations

import csv
import os
import time
from pathlib import Path

from state import SessionState


class TrialLogger:
    def __init__(self, csv_path: str) -> None:
        self.csv_path = Path(csv_path)
        self._initialize_csv()

    def _initialize_csv(self) -> None:
        if self.csv_path.exists():
            return

        with self.csv_path.open("w", newline="") as file_obj:
            writer = csv.writer(file_obj)
            writer.writerow(
                [
                    "Trial No.",
                    "Observed State Sequence",
                    "Final State",
                    "Final Result",
                    "Minimum Distance (px)",
                    "Maximum Hold Time (s)",
                    "Pill Seen",
                    "Mouth Open Seen",
                    "Timestamp",
                ]
            )

    def save_trial(self, session: SessionState) -> None:
        with self.csv_path.open("a", newline="") as file_obj:
            writer = csv.writer(file_obj)
            writer.writerow(
                [
                    session.trial_no,
                    " -> ".join(session.state_history) if session.state_history else session.state,
                    session.state,
                    session.final_result,
                    round(session.min_distance, 2) if session.min_distance != float("inf") else "N/A",
                    round(session.max_hold_time, 2),
                    session.pill_seen_once,
                    session.mouth_open_seen,
                    time.strftime("%Y-%m-%d %H:%M:%S"),
                ]
            )

        print(f"[INFO] Trial {session.trial_no} saved to {os.path.abspath(self.csv_path)}")
