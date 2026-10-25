"""Hand-presence proposal detector. It never approves an interaction or exports pose features."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path

DEFAULT_MODEL = Path(__file__).resolve().parent / 'video_inspect_20260913' / 'hand_landmarker.task'

@dataclass
class HandPresence:
    hand_count: int

class HandPresenceDetector:
    def __init__(self, model_path=DEFAULT_MODEL):
        import mediapipe as mp
        self.mp = mp
        self.version = getattr(mp, '__version__', 'unknown')
        self.last_timestamp = -1
        self.model_sha256 = None
        legacy = getattr(getattr(mp, 'solutions', None), 'hands', None)
        if legacy is not None:
            self.backend = 'mediapipe_solutions_hands'
            self.detector = legacy.Hands(static_image_mode=False, max_num_hands=2,
                model_complexity=0, min_detection_confidence=0.30,
                min_tracking_confidence=0.30)
        else:
            model = Path(model_path)
            if not model.is_file():
                raise FileNotFoundError(f'HandLandmarker model is missing: {model}')
            tasks = mp.tasks
            self.backend = 'mediapipe_tasks_hand_landmarker'
            self.model_sha256 = hashlib.sha256(model.read_bytes()).hexdigest()
            options = tasks.vision.HandLandmarkerOptions(
                base_options=tasks.BaseOptions(model_asset_path=str(model)),
                running_mode=tasks.vision.RunningMode.VIDEO, num_hands=2,
                min_hand_detection_confidence=0.30,
                min_hand_presence_confidence=0.30, min_tracking_confidence=0.30)
            self.detector = tasks.vision.HandLandmarker.create_from_options(options)

    def process(self, rgb, time_s):
        if self.backend == 'mediapipe_solutions_hands':
            result = self.detector.process(rgb)
            return HandPresence(len(result.multi_hand_landmarks or []))
        timestamp = max(self.last_timestamp + 1, int(round(float(time_s) * 1000)))
        self.last_timestamp = timestamp
        image = self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=rgb)
        result = self.detector.detect_for_video(image, timestamp)
        return HandPresence(len(result.hand_landmarks or []))

    def close(self):
        self.detector.close()
