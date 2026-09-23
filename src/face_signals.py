# src/face_signals.py
"""
Face Signals: Smile, Blink, and Eye-Closed Detection using MediaPipe FaceMesh.
Computes Eye Aspect Ratio (EAR) and mouth-to-face width ratio with hysteresis.
"""
from dataclasses import dataclass
from typing import Optional, Tuple
import cv2
import mediapipe as mp
import numpy as np

# Canonical MediaPipe FaceMesh landmark indices
LEFT_EYE = (33, 160, 158, 133, 153, 144)
RIGHT_EYE = (362, 385, 387, 263, 373, 380)
LEFT_EYE_CENTER = (159, 145)
RIGHT_EYE_CENTER = (386, 374)
MOUTH_LEFT, MOUTH_RIGHT = 61, 291
LIP_TOP, LIP_BOTTOM = 13, 14
FACE_LEFT, FACE_RIGHT = 234, 454


def distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))


def eye_aspect_ratio(
    points: np.ndarray,
    idx: Tuple[int, ...],
    center_idx: Optional[Tuple[int, int]] = None,
) -> float:
    """
    Computes Eye Aspect Ratio (EAR):
    EAR = (|p2 - p6| + |p3 - p5|) / (2 * |p1 - p4|)
    If center_idx is provided, incorporates center eyelid vertical distance for enhanced blink sensitivity.
    """
    p1, p2, p3, p4, p5, p6 = (points[i] for i in idx)
    width = max(distance(p1, p4), 1e-6)
    if center_idx is not None:
        p_top, p_bot = points[center_idx[0]], points[center_idx[1]]
        return (distance(p2, p6) + distance(p3, p5) + distance(p_top, p_bot)) / (3.0 * width)
    return (distance(p2, p6) + distance(p3, p5)) / (2.0 * width)


@dataclass
class FaceSignals:
    ear: float
    blink: bool
    eyes_closed: bool
    smile_score: float
    smiling: bool
    corner_elevation: float = 0.0


class FaceSignalExtractor:
    def __init__(
        self,
        ear_threshold: float = 0.21,
        blink_min_frames: int = 1,
        blink_max_frames: int = 7,
        closed_frames: int = 8,
        smile_on: float = 0.43,
        smile_off: float = 0.39,
        use_corner_elevation: bool = True,
    ):
        self.ear_threshold = ear_threshold
        self.blink_min_frames = blink_min_frames
        self.blink_max_frames = blink_max_frames
        self.closed_frames = closed_frames
        self.smile_on = smile_on
        self.smile_off = smile_off
        self.use_corner_elevation = use_corner_elevation
        self.low_ear_frames = 0
        self.blink_cooldown = 0
        self.open_ear_baseline = 0.28
        self.smiling = False
        self.mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

    def reset(self) -> None:
        self.low_ear_frames = 0
        self.blink_cooldown = 0
        self.smiling = False

    def close(self) -> None:
        self.mesh.close()

    def calibrate(self, sample_smile_score: float, open_ear: float) -> Tuple[float, float, float]:
        """
        Calibrate both smile and EAR thresholds based on the user's neutral face and open eyes.
        Sets smile_on to ~14% above neutral, smile_off to ~7% above neutral.
        Sets ear_threshold to 75% of measured open-eye EAR.
        """
        self.smile_on = round(sample_smile_score * 1.14, 3)
        self.smile_off = round(sample_smile_score * 1.07, 3)
        self.open_ear_baseline = open_ear
        self.ear_threshold = round(open_ear * 0.75, 3)
        self.smiling = False
        self.low_ear_frames = 0
        return self.smile_on, self.smile_off, self.ear_threshold

    def calibrate_neutral_mouth(self, sample_score: float) -> Tuple[float, float]:
        """Backward-compatible helper to calibrate smile thresholds only."""
        self.smile_on = round(sample_score * 1.14, 3)
        self.smile_off = round(sample_score * 1.07, 3)
        self.smiling = False
        return self.smile_on, self.smile_off

    def analyze(self, frame: np.ndarray, bbox) -> Optional[FaceSignals]:
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = bbox
        bw, bh = x2 - x1, y2 - y1
        pad_x, pad_y = int(0.12 * bw), int(0.18 * bh)
        rx1, ry1 = max(0, x1 - pad_x), max(0, y1 - pad_y)
        rx2, ry2 = min(w, x2 + pad_x), min(h, y2 + pad_y)
        roi = frame[ry1:ry2, rx1:rx2]
        if roi.size == 0:
            return None

        result = self.mesh.process(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))
        if not result.multi_face_landmarks:
            return None

        rh, rw = roi.shape[:2]
        lm = result.multi_face_landmarks[0].landmark
        points = np.array(
            [[p.x * rw + rx1, p.y * rh + ry1] for p in lm],
            dtype=np.float32,
        )

        left_ear = eye_aspect_ratio(points, LEFT_EYE, LEFT_EYE_CENTER)
        right_ear = eye_aspect_ratio(points, RIGHT_EYE, RIGHT_EYE_CENTER)
        ear = 0.5 * (left_ear + right_ear)

        # Smooth baseline when eyes are clearly open
        if ear >= self.ear_threshold and self.low_ear_frames == 0:
            self.open_ear_baseline = 0.95 * self.open_ear_baseline + 0.05 * ear

        if self.blink_cooldown > 0:
            self.blink_cooldown -= 1

        blink = False
        if ear < self.ear_threshold:
            self.low_ear_frames += 1
        else:
            # Eyes transitioned back to open: check duration for valid blink
            if (
                self.blink_min_frames <= self.low_ear_frames <= self.blink_max_frames
                and self.blink_cooldown == 0
            ):
                blink = True
                self.blink_cooldown = 2  # Cooldown to avoid bounce false counts
            self.low_ear_frames = 0

        eyes_closed = self.low_ear_frames >= self.closed_frames

        face_width = max(distance(points[FACE_LEFT], points[FACE_RIGHT]), 1e-6)
        mouth_width = distance(points[MOUTH_LEFT], points[MOUTH_RIGHT])
        smile_score = mouth_width / face_width

        # Check mouth corner elevation relative to lip midpoint
        # In image coordinates, y increases downward.
        # When smiling, corners are pulled upward (smaller y), so (lip_center_y - corners_y) > 0.
        corners_y = (points[MOUTH_LEFT][1] + points[MOUTH_RIGHT][1]) / 2.0
        lip_center_y = (points[LIP_TOP][1] + points[LIP_BOTTOM][1]) / 2.0
        corner_elevation = float((lip_center_y - corners_y) / face_width)

        # Genuine smile requires both mouth widening AND corners not drooping downward
        is_mouth_upturned = (corner_elevation >= -0.008) if self.use_corner_elevation else True

        if self.smiling:
            # Hysteresis: to stay smiling, score must stay above smile_off
            self.smiling = (smile_score >= self.smile_off) and is_mouth_upturned
        else:
            # To enter smiling, score must exceed smile_on and mouth corners must be upturned
            self.smiling = (smile_score >= self.smile_on) and is_mouth_upturned

        return FaceSignals(
            ear=ear,
            blink=blink,
            eyes_closed=eyes_closed,
            smile_score=smile_score,
            smiling=self.smiling,
            corner_elevation=corner_elevation,
        )
