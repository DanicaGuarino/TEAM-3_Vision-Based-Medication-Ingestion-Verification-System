from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from config import AppConfig
from logger import TrialLogger
from state import SessionState


@dataclass
class FrameInputs:
    pill_detected: bool
    pill_in_hand: bool
    hand_visible: bool
    mouth_open: bool
    distance_to_mouth: Optional[float]


class IngestionVerifier:
    def __init__(self, config: AppConfig, logger: TrialLogger) -> None:
        self.config = config
        self.logger = logger
        self.session = SessionState()

    def update(self, inputs: FrameInputs, current_time: float) -> SessionState:
        session = self.session

        if inputs.pill_detected:
            session.pill_seen_once = True
        if inputs.mouth_open:
            session.mouth_open_seen = True
        if inputs.distance_to_mouth is not None:
            session.min_distance = min(session.min_distance, inputs.distance_to_mouth)

        if inputs.pill_in_hand:
            session.start_trial_if_needed(current_time)

        if session.result_time is None:
            self._run_state_machine(inputs, current_time)
            self._handle_timeout(current_time)
        else:
            self._handle_result_display(current_time)

        return session

    def _run_state_machine(self, inputs: FrameInputs, current_time: float) -> None:
        session = self.session

        if session.state == "WAITING_FOR_PILL":
            session.final_result = "NOT CONFIRMED"
            if inputs.pill_in_hand:
                session.set_state("PILL_IN_HAND")

        elif session.state == "PILL_IN_HAND":
            session.final_result = "NOT CONFIRMED"
            if self._is_near_mouth(inputs) and inputs.mouth_open:
                session.hold_start = current_time
                session.set_state("HOLDING")

        elif session.state == "HOLDING":
            session.final_result = "NOT CONFIRMED"
            if self._is_near_mouth(inputs) and inputs.mouth_open and session.hold_start is not None:
                elapsed = current_time - session.hold_start
                session.max_hold_time = max(session.max_hold_time, elapsed)
                if elapsed >= self.config.hold_time:
                    session.set_state("READY_TO_CONFIRM")
                    session.ready_confirm_start = None
            else:
                session.set_state("PILL_IN_HAND")
                session.hold_start = None

        elif session.state == "READY_TO_CONFIRM":
            session.final_result = "NOT CONFIRMED"
            if session.ready_confirm_start is None:
                if inputs.distance_to_mouth is not None and inputs.distance_to_mouth > self.config.release_dist:
                    session.ready_confirm_start = current_time
            else:
                self._finalize_confirmation(inputs, current_time)

    def _finalize_confirmation(self, inputs: FrameInputs, current_time: float) -> None:
        session = self.session
        elapsed_confirm = current_time - session.ready_confirm_start

        if not inputs.hand_visible:
            session.set_state("NOT_CONFIRMED")
            self._end_trial("NOT CONFIRMED", current_time)
            return

        if inputs.pill_detected:
            session.set_state("NOT_CONFIRMED")
            self._end_trial("NOT CONFIRMED", current_time)
            return

        if elapsed_confirm >= self.config.ready_confirm_time:
            if session.pill_seen_once and session.mouth_open_seen:
                session.set_state("INGESTED")
                self._end_trial("INGESTED", current_time)
            else:
                session.set_state("NOT_CONFIRMED")
                self._end_trial("NOT CONFIRMED", current_time)

    def _handle_timeout(self, current_time: float) -> None:
        session = self.session
        if session.session_start is None or session.result_time is not None:
            return
        if (current_time - session.session_start) > self.config.session_timeout:
            self._end_trial("NOT CONFIRMED", current_time, final_state_override="NOT_CONFIRMED")

    def _handle_result_display(self, current_time: float) -> None:
        session = self.session
        if session.result_time is None:
            return
        if (current_time - session.result_time) > self.config.result_display_time:
            session.trial_no += 1
            session.reset_for_next_trial()

    def _end_trial(self, result_label: str, current_time: float, final_state_override: Optional[str] = None) -> None:
        session = self.session
        session.final_result = result_label
        if final_state_override is not None:
            session.set_state(final_state_override)
        session.result_time = current_time
        self.logger.save_trial(session)

    def _is_near_mouth(self, inputs: FrameInputs) -> bool:
        return (
            inputs.distance_to_mouth is not None
            and inputs.distance_to_mouth < (self.config.approach_dist + self.config.mouth_radius)
        )
