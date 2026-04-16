from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class SessionState:
    state: str = "WAITING_FOR_PILL"
    final_result: str = "NOT CONFIRMED"

    hold_start: Optional[float] = None
    session_start: Optional[float] = None
    result_time: Optional[float] = None
    ready_confirm_start: Optional[float] = None

    pill_seen_once: bool = False
    mouth_open_seen: bool = False

    trial_no: int = 1
    state_history: List[str] = field(default_factory=lambda: ["WAITING_FOR_PILL"])
    min_distance: float = float("inf")
    max_hold_time: float = 0.0

    def set_state(self, new_state: str) -> None:
        if self.state == new_state:
            return
        if new_state not in self.state_history:
            self.state_history.append(new_state)
        self.state = new_state

    def start_trial_if_needed(self, current_time: float) -> None:
        if self.session_start is None:
            self.session_start = current_time
            if not self.state_history:
                self.state_history = [self.state]

    def reset_for_next_trial(self) -> None:
        self.state = "WAITING_FOR_PILL"
        self.final_result = "NOT CONFIRMED"

        self.hold_start = None
        self.session_start = None
        self.result_time = None
        self.ready_confirm_start = None

        self.pill_seen_once = False
        self.mouth_open_seen = False

        self.state_history = [self.state]
        self.min_distance = float("inf")
        self.max_hold_time = 0.0
