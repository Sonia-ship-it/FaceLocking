# src/person_tracker.py
"""
Whole-Person Tracking and Face-Association Module.
Maintains temporal identity continuity across temporary facial loss through an honest
three-state model (Confirmed, Temporary, and Unknown) with exponential confidence decay.
Designed for CPU-only execution using MediaPipe and OpenCV.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import List, Tuple, Optional, Dict, Union
import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

from src.haar_5pt import Haar5ptDetector, FaceBox5pt, align_face_5pt
from src.recognize import ArcFaceEmbedderONNX, FaceDBMatcher, MatchResult, load_db_npz


class TrackState(Enum):
    """
    Honest Three-State Identity Model:
      CONFIRMED: Face actively detected, spatially aligned with body, and verified by ArcFace.
      TEMPORARY: Face temporarily lost (head turn, occlusion, looking away), body actively
                 tracked with exponential confidence decay. Maintains identity continuity.
      UNKNOWN:   Confidence decayed below threshold, or track never possessed a verified face.
    """
    CONFIRMED = "Confirmed"
    TEMPORARY = "Temporary"
    UNKNOWN = "Unknown"


@dataclass
class PersonDetection:
    """Represents a whole-person or upper-body detection bounding box [x1, y1, x2, y2]."""
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float = 1.0
    keypoints: Optional[np.ndarray] = None  # Optional body landmarks (e.g. shoulders, hips)

    @property
    def box(self) -> Tuple[int, int, int, int]:
        return (int(self.x1), int(self.y1), int(self.x2), int(self.y2))

    @property
    def center(self) -> Tuple[float, float]:
        return ((self.x1 + self.x2) / 2.0, (self.y1 + self.y2) / 2.0)

    @property
    def width(self) -> float:
        return max(1.0, self.x2 - self.x1)

    @property
    def height(self) -> float:
        return max(1.0, self.y2 - self.y1)

    @property
    def upper_center(self) -> Tuple[float, float]:
        """Center of head/shoulder region (top 25% of body box)."""
        return ((self.x1 + self.x2) / 2.0, self.y1 + 0.20 * self.height)


class KalmanBoxTracker:
    """
    Constant-Velocity Kalman Filter for tracking bounding boxes [x1, y1, x2, y2].
    State: [cx, cy, s, r, v_cx, v_cy, v_s, 0] where s = area, r = aspect ratio.
    """
    count = 0

    def __init__(self, bbox: Tuple[float, float, float, float]):
        KalmanBoxTracker.count += 1
        self.id = KalmanBoxTracker.count

        # OpenCV Kalman filter: 7 state variables, 4 measurement variables
        self.kf = cv2.KalmanFilter(7, 4)
        self.kf.transitionMatrix = np.array([
            [1, 0, 0, 0, 1, 0, 0],
            [0, 1, 0, 0, 0, 1, 0],
            [0, 0, 1, 0, 0, 0, 1],
            [0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 1]
        ], dtype=np.float32)

        self.kf.measurementMatrix = np.eye(4, 7, dtype=np.float32)

        # Tuning covariance
        self.kf.processNoiseCov = np.eye(7, dtype=np.float32) * 1e-2
        self.kf.processNoiseCov[4:, 4:] *= 5.0
        self.kf.measurementNoiseCov = np.eye(4, dtype=np.float32) * 1e-1
        self.kf.errorCovPost = np.eye(7, dtype=np.float32) * 1.0

        # Initialize state with bbox
        w = max(1.0, bbox[2] - bbox[0])
        h = max(1.0, bbox[3] - bbox[1])
        cx = bbox[0] + w / 2.0
        cy = bbox[1] + h / 2.0
        s = w * h
        r = w / float(h)
        self.kf.statePost = np.array([[cx], [cy], [s], [r], [0], [0], [0]], dtype=np.float32)

        self.time_since_update = 0
        self.history: List[Tuple[float, float]] = [(cx, cy)]
        self.hits = 1
        self.hit_streak = 1
        self.age = 0

    def update(self, bbox: Tuple[float, float, float, float]):
        self.time_since_update = 0
        self.hits += 1
        self.hit_streak += 1
        w = max(1.0, bbox[2] - bbox[0])
        h = max(1.0, bbox[3] - bbox[1])
        cx = bbox[0] + w / 2.0
        cy = bbox[1] + h / 2.0
        s = w * h
        r = w / float(h)
        meas = np.array([[cx], [cy], [s], [r]], dtype=np.float32)
        self.kf.correct(meas)
        self.history.append((cx, cy))
        if len(self.history) > 60:
            self.history.pop(0)

    def predict(self) -> Tuple[float, float, float, float]:
        self.kf.predict()
        self.age += 1
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1

        cx = float(self.kf.statePost[0, 0])
        cy = float(self.kf.statePost[1, 0])
        s = max(100.0, float(self.kf.statePost[2, 0]))
        r = max(0.1, float(self.kf.statePost[3, 0]))
        w = math.sqrt(s * r)
        h = s / w
        return (cx - w / 2.0, cy - h / 2.0, cx + w / 2.0, cy + h / 2.0)

    def get_state(self) -> Tuple[float, float, float, float]:
        cx = float(self.kf.statePost[0, 0])
        cy = float(self.kf.statePost[1, 0])
        s = max(100.0, float(self.kf.statePost[2, 0]))
        r = max(0.1, float(self.kf.statePost[3, 0]))
        w = math.sqrt(s * r)
        h = s / w
        return (cx - w / 2.0, cy - h / 2.0, cx + w / 2.0, cy + h / 2.0)


@dataclass
class TrackedPerson:
    """
    Maintains temporal identity continuity for a single person track across frames.
    Implements the honest three-state model with exponential confidence decay.
    """
    track_id: int
    kalman: KalmanBoxTracker
    state: TrackState = TrackState.UNKNOWN
    identity: str = "Unknown"
    confidence: float = 0.0

    # Timing and decay parameters
    decay_lambda: float = 0.35              # Exponential decay rate (lambda = ln(2)/half_life)
    confidence_thresh_unknown: float = 0.35 # Confidence threshold below which state drops to Unknown
    max_temporary_seconds: float = 4.0      # Maximum duration track can stay Temporary without face

    # Timestamps
    first_seen_time: float = field(default_factory=time.time)
    last_body_seen_time: float = field(default_factory=time.time)
    last_face_seen_time: Optional[float] = None
    last_confirmed_confidence: float = 1.0

    # Transient observations
    current_face_box: Optional[Tuple[int, int, int, int]] = None
    current_face_kps: Optional[np.ndarray] = None
    frames_since_face: int = 0
    total_frames_tracked: int = 0
    consecutive_face_hits: int = 0

    # Diagnostic logs
    state_transitions: List[Tuple[float, str, str, float]] = field(default_factory=list)

    @property
    def body_box(self) -> Tuple[int, int, int, int]:
        x1, y1, x2, y2 = self.kalman.get_state()
        return (int(x1), int(y1), int(x2), int(y2))

    @property
    def center(self) -> Tuple[float, float]:
        box = self.body_box
        return ((box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0)

    @property
    def upper_center(self) -> Tuple[float, float]:
        """Upper chest / head anchor point for centering logic."""
        box = self.body_box
        h = box[3] - box[1]
        return ((box[0] + box[2]) / 2.0, box[1] + 0.22 * h)

    def record_transition(self, prev_state: TrackState, new_state: TrackState, current_time: float):
        if prev_state != new_state:
            self.state_transitions.append((
                current_time,
                prev_state.value,
                new_state.value,
                float(self.confidence),
            ))

    def update_with_confirmed_face(
        self,
        name: str,
        match_similarity: float,
        face_box: Tuple[int, int, int, int],
        face_kps: Optional[np.ndarray],
        current_time: float,
    ):
        """
        Face actively detected and confirmed with ArcFace.
        Refreshes confidence and transitions track to CONFIRMED.
        """
        prev_state = self.state
        self.state = TrackState.CONFIRMED
        self.identity = name
        # Bound confidence between [similarity, 1.0] with smoothing
        self.confidence = float(np.clip(match_similarity, 0.70, 1.0))
        self.last_confirmed_confidence = self.confidence
        self.last_face_seen_time = current_time
        self.current_face_box = face_box
        self.current_face_kps = face_kps
        self.frames_since_face = 0
        self.consecutive_face_hits += 1

        self.record_transition(prev_state, TrackState.CONFIRMED, current_time)

    def update_without_face(self, current_time: float):
        """
        Body is tracked, but face was NOT detected this frame (e.g. head turn, occlusion).
        Applies exponential confidence decay according to the honest three-state model:
            C(t) = C_0 * exp(-lambda * delta_t)
        """
        self.frames_since_face += 1
        self.current_face_box = None
        self.current_face_kps = None
        self.consecutive_face_hits = 0
        prev_state = self.state

        if self.state == TrackState.CONFIRMED:
            # Transition from Confirmed to Temporary
            self.state = TrackState.TEMPORARY
            self.record_transition(prev_state, TrackState.TEMPORARY, current_time)

        if self.state == TrackState.TEMPORARY:
            if self.last_face_seen_time is not None:
                delta_t = max(0.0, current_time - self.last_face_seen_time)
                # Exponential decay formula
                decay_factor = math.exp(-self.decay_lambda * delta_t)
                self.confidence = float(self.last_confirmed_confidence * decay_factor)

                # Check if confidence decayed below threshold or timeout exceeded
                if (self.confidence < self.confidence_thresh_unknown) or (delta_t > self.max_temporary_seconds):
                    self.state = TrackState.UNKNOWN
                    self.confidence = float(min(self.confidence, 0.25))
                    self.record_transition(TrackState.TEMPORARY, TrackState.UNKNOWN, current_time)
            else:
                self.state = TrackState.UNKNOWN
                self.confidence = 0.20
                self.record_transition(TrackState.TEMPORARY, TrackState.UNKNOWN, current_time)

        elif self.state == TrackState.UNKNOWN:
            # Stays Unknown with nominal baseline confidence
            self.confidence = float(max(0.05, self.confidence * 0.95))


class SpatialFaceBodyMapper:
    """
    Computes spatial face-to-body mapping based on human anatomical geometry.
    Ensures faces are associated with the correct body track and prevents identity swaps.
    """

    @staticmethod
    def compute_spatial_affinity(
        face_box: Tuple[int, int, int, int],
        body_box: Tuple[int, int, int, int],
    ) -> float:
        """
        Calculates anatomical alignment score between face and body in [0.0, 1.0].
        Criteria:
          1. Vertical position: Face center must be in top 40% of body box.
          2. Horizontal alignment: Face center aligns with body center.
          3. Enclosure (IoFA): Face area intersecting body box / Face area.
        """
        fx1, fy1, fx2, fy2 = face_box
        bx1, by1, bx2, by2 = body_box

        fw = max(1.0, fx2 - fx1)
        fh = max(1.0, fy2 - fy1)
        bw = max(1.0, bx2 - bx1)
        bh = max(1.0, by2 - by1)

        fcx = (fx1 + fx2) / 2.0
        fcy = (fy1 + fy2) / 2.0
        bcx = (bx1 + bx2) / 2.0

        # Anatomical Gate 1: Face center vertically in upper torso / head region
        # Allow slight headroom above body box (-0.15*bh) down to 45% of body box
        if not (by1 - 0.15 * bh <= fcy <= by1 + 0.45 * bh):
            return 0.0

        # Anatomical Gate 2: Face horizontal center within body horizontal boundaries (with 15% tolerance)
        if not (bx1 - 0.15 * bw <= fcx <= bx2 + 0.15 * bw):
            return 0.0

        # Intersection over Face Area (IoFA)
        ix1 = max(fx1, bx1)
        iy1 = max(fy1, by1)
        ix2 = min(fx2, bx2)
        iy2 = min(fy2, by2)
        inter_area = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
        face_area = fw * fh
        iofa = inter_area / float(face_area)

        if iofa < 0.40:
            return 0.0

        # Horizontal alignment score (closer to body centerline is better)
        horiz_dist_norm = abs(fcx - bcx) / (bw / 2.0)
        horiz_score = max(0.0, 1.0 - horiz_dist_norm)

        # Expected head vertical anchor (around 15% down from body top)
        expected_head_y = by1 + 0.15 * bh
        vert_dist_norm = abs(fcy - expected_head_y) / (bh * 0.35)
        vert_score = max(0.0, 1.0 - vert_dist_norm)

        # Composite spatial affinity
        affinity = 0.50 * iofa + 0.30 * horiz_score + 0.20 * vert_score
        return float(np.clip(affinity, 0.0, 1.0))

    @classmethod
    def match_faces_to_bodies(
        cls,
        faces: List[Tuple[int, int, int, int]],
        bodies: List[Tuple[int, int, int, int]],
        min_affinity_thresh: float = 0.35,
    ) -> List[Tuple[int, int, float]]:
        """
        Solves optimal bipartite matching between faces and bodies using Hungarian algorithm.
        Returns list of (face_idx, body_idx, affinity_score).
        """
        if not faces or not bodies:
            return []

        cost_matrix = np.ones((len(faces), len(bodies)), dtype=np.float32) * 10.0

        for f_idx, f_box in enumerate(faces):
            for b_idx, b_box in enumerate(bodies):
                affinity = cls.compute_spatial_affinity(f_box, b_box)
                if affinity >= min_affinity_thresh:
                    cost_matrix[f_idx, b_idx] = 1.0 - affinity

        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        matches = []
        for r, c in zip(row_ind, col_ind):
            if cost_matrix[r, c] < (1.0 - min_affinity_thresh):
                affinity = 1.0 - cost_matrix[r, c]
                matches.append((r, c, float(affinity)))

        return matches


class PersonDetector:
    """
    CPU-Optimized Multi-Backend Person Detector.
    Uses MediaPipe Pose landmarks for high-accuracy whole-person and torso detection on CPU,
    with OpenCV HOG / Cascade and anatomical projection fallbacks.
    """

    def __init__(self, min_confidence: float = 0.5):
        self.min_confidence = min_confidence
        self.mp_pose = None

        try:
            import mediapipe as mp
            if hasattr(mp, "solutions") and hasattr(mp.solutions, "pose"):
                self.mp_pose = mp.solutions.pose.Pose(
                    static_image_mode=False,
                    model_complexity=0,  # 0 is the fastest CPU-optimized pose model
                    min_detection_confidence=min_confidence,
                    min_tracking_confidence=min_confidence,
                )
        except Exception:
            self.mp_pose = None

        # Fallback HOG People detector
        self.hog = cv2.HOGDescriptor()
        self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    def detect(
        self,
        frame: np.ndarray,
        detected_faces: Optional[List[FaceBox5pt]] = None,
    ) -> List[PersonDetection]:
        """
        Detects whole persons in the frame.
        Combines MediaPipe Pose with anatomical face-anchored torso projection
        to ensure multi-person capability and zero failure on head turns.
        """
        h, w = frame.shape[:2]
        persons: List[PersonDetection] = []

        # 1. MediaPipe Pose Detection (Primary high-fidelity body detector)
        if self.mp_pose is not None:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.mp_pose.process(rgb)

            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark
                # Key body points: nose(0), shoulders(11, 12), hips(23, 24)
                valid_pts = []
                for idx in [0, 11, 12, 13, 14, 23, 24]:
                    lm = landmarks[idx]
                    if lm.visibility > 0.35:
                        valid_pts.append([lm.x * w, lm.y * h])

                if valid_pts:
                    pts_arr = np.array(valid_pts, dtype=np.float32)
                    min_x, min_y = np.min(pts_arr, axis=0)
                    max_x, max_y = np.max(pts_arr, axis=0)

                    bw = max_x - min_x
                    bh = max_y - min_y

                    # Expand margins around landmarks to capture full body / torso width
                    pad_x = max(bw * 0.35, 40.0)
                    pad_top = max(bh * 0.25, 35.0)
                    pad_bottom = max(bh * 0.50, 60.0)

                    bx1 = max(0.0, min_x - pad_x)
                    by1 = max(0.0, min_y - pad_top)
                    bx2 = min(float(w), max_x + pad_x)
                    by2 = min(float(h), max_y + pad_bottom)

                    # For webcam framing (seated in front of computer):
                    # Ensure the whole torso extends to the bottom of the frame without cutting off
                    if by2 > h * 0.55 or max_y > h * 0.45:
                        by2 = float(h)

                    persons.append(PersonDetection(bx1, by1, bx2, by2, confidence=0.90, keypoints=pts_arr))

        # 2. Multi-Person Complement: Face-Anchored Anatomical Body Projection
        # When multiple people or additional faces are detected, project anatomically realistic
        # upper-body / torso boxes so all individuals are simultaneously tracked.
        if detected_faces:
            for face in detected_faces:
                fx1, fy1, fx2, fy2 = face.x1, face.y1, face.x2, face.y2
                fw = fx2 - fx1
                fh = fy2 - fy1

                # Anatomical human proportions: body is ~2.5x face width, ~3.6x face height
                proj_bw = fw * 2.5
                proj_bh = fh * 3.6
                proj_cx = (fx1 + fx2) / 2.0
                proj_y1 = max(0.0, fy1 - 0.25 * fh)
                proj_x1 = max(0.0, proj_cx - proj_bw / 2.0)
                proj_x2 = min(float(w), proj_cx + proj_bw / 2.0)

                # For sitting computer webcam view: extend torso to fill full visible height
                if (proj_y1 + proj_bh > h * 0.55) or (fy2 > h * 0.35):
                    proj_y2 = float(h)
                else:
                    proj_y2 = min(float(h), proj_y1 + proj_bh)

                # Check if already covered by an existing pose detection
                covered = False
                for existing in persons:
                    # Compute IoU
                    ex_box = (existing.x1, existing.y1, existing.x2, existing.y2)
                    ix1, iy1 = max(proj_x1, ex_box[0]), max(proj_y1, ex_box[1])
                    ix2, iy2 = min(proj_x2, ex_box[2]), min(proj_y2, ex_box[3])
                    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
                    union = (proj_bw * (proj_y2 - proj_y1)) + (existing.width * existing.height) - inter
                    if inter / max(1.0, union) > 0.45:
                        covered = True
                        break

                if not covered:
                    persons.append(PersonDetection(proj_x1, proj_y1, proj_x2, proj_y2, confidence=0.75))

        return persons


class WholePersonFaceTracker:
    """
    Main Tracking & Association Engine.
    Integrates Whole-Person Detection, Face Detection + ArcFace Embeddings,
    Spatial Mapping, Kalman Filtering, and the Honest Three-State Model.
    """

    def __init__(
        self,
        target_name: Optional[str] = None,
        db_path: str = "data/db/face_db.npz",
        decay_half_life: float = 2.5,          # Half-life for confidence decay in seconds (e.g. 2.5s)
        match_thresh: float = 0.40,            # ArcFace cosine distance threshold
        confidence_unknown_thresh: float = 0.35,# Drops to Unknown below this confidence
        max_iou_distance: float = 0.70,        # Kalman association threshold
        max_lost_frames: int = 35,             # Frames before track deletion
    ):
        self.target_name = target_name
        self.db = load_db_npz(db_path)
        self.matcher = FaceDBMatcher(self.db, dist_thresh=match_thresh)
        self.embedder = ArcFaceEmbedderONNX()
        self.face_detector = Haar5ptDetector()
        self.person_detector = PersonDetector()

        # Decay parameter lambda = ln(2) / half_life
        self.decay_lambda = math.log(2.0) / max(0.1, decay_half_life)
        self.confidence_unknown_thresh = confidence_unknown_thresh
        self.max_iou_distance = max_iou_distance
        self.max_lost_frames = max_lost_frames

        # Active person tracks
        self.tracks: List[TrackedPerson] = []
        self.frame_count = 0
        self.selected_target_id: Optional[int] = None

    def box_iou(self, boxA: Tuple[float, float, float, float], boxB: Tuple[float, float, float, float]) -> float:
        ax1, ay1, ax2, ay2 = boxA
        bx1, by1, bx2, by2 = boxB
        ix1 = max(ax1, bx1)
        iy1 = max(ay1, by1)
        ix2 = min(ax2, bx2)
        iy2 = min(ay2, by2)
        inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
        areaA = max(1.0, (ax2 - ax1) * (ay2 - ay1))
        areaB = max(1.0, (bx2 - bx1) * (by2 - by1))
        return inter / float(areaA + areaB - inter)

    def process_frame(
        self,
        frame: np.ndarray,
        current_time: Optional[float] = None,
        suppress_face: bool = False,
        simulated_occlusion_box: Optional[Tuple[int, int, int, int]] = None,
    ) -> List[TrackedPerson]:
        """
        Processes a video frame through detection, association, and state updates.
        Supports simulated real-world scenarios:
          - suppress_face: Simulates looking away or sudden head turn (face detection fails).
          - simulated_occlusion_box: Draws artificial obstacle over target to test occlusion.
        """
        if current_time is None:
            current_time = time.time()
        self.frame_count += 1
        h, w = frame.shape[:2]

        # Apply simulated occlusion if requested
        proc_frame = frame.copy()
        if simulated_occlusion_box is not None:
            ox1, oy1, ox2, oy2 = simulated_occlusion_box
            cv2.rectangle(proc_frame, (ox1, oy1), (ox2, oy2), (30, 30, 30), -1)
            cv2.putText(proc_frame, "[SIMULATED OCCLUDER]", (ox1 + 10, oy1 + 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 1)

        # 1. Step Kalman predictions for existing tracks
        for track in self.tracks:
            track.kalman.predict()

        # 2. Detect Faces + Landmarks
        detected_faces: List[FaceBox5pt] = []
        if not suppress_face:
            detected_faces = self.face_detector.detect(proc_frame)

        # Filter out faces inside simulated occlusion
        if simulated_occlusion_box is not None:
            filtered_faces = []
            for f in detected_faces:
                fcx, fcy = (f.x1 + f.x2) / 2.0, (f.y1 + f.y2) / 2.0
                if not (ox1 <= fcx <= ox2 and oy1 <= fcy <= oy2):
                    filtered_faces.append(f)
            detected_faces = filtered_faces

        # 3. Detect Persons (MediaPipe Pose + Anatomical Projection)
        body_detections = self.person_detector.detect(proc_frame, detected_faces)

        # 4. Associate Detections with Existing Kalman Person Tracks (IoU Matching)
        matched_track_indices = set()
        matched_det_indices = set()

        if self.tracks and body_detections:
            iou_matrix = np.zeros((len(self.tracks), len(body_detections)), dtype=np.float32)
            for t_idx, track in enumerate(self.tracks):
                pred_box = track.kalman.get_state()
                for d_idx, det in enumerate(body_detections):
                    iou_matrix[t_idx, d_idx] = self.box_iou(pred_box, (det.x1, det.y1, det.x2, det.y2))

            cost_matrix = 1.0 - iou_matrix
            t_rows, d_cols = linear_sum_assignment(cost_matrix)

            for r, c in zip(t_rows, d_cols):
                if iou_matrix[r, c] >= (1.0 - self.max_iou_distance):
                    self.tracks[r].kalman.update((
                        body_detections[c].x1,
                        body_detections[c].y1,
                        body_detections[c].x2,
                        body_detections[c].y2,
                    ))
                    self.tracks[r].last_body_seen_time = current_time
                    self.tracks[r].total_frames_tracked += 1
                    matched_track_indices.add(r)
                    matched_det_indices.add(c)

        # Create new tracks for unmatched body detections
        for d_idx, det in enumerate(body_detections):
            if d_idx not in matched_det_indices:
                new_kt = KalmanBoxTracker((det.x1, det.y1, det.x2, det.y2))
                new_track = TrackedPerson(
                    track_id=new_kt.id,
                    kalman=new_kt,
                    decay_lambda=self.decay_lambda,
                    confidence_thresh_unknown=self.confidence_unknown_thresh,
                    first_seen_time=current_time,
                    last_body_seen_time=current_time,
                )
                self.tracks.append(new_track)
                matched_track_indices.add(len(self.tracks) - 1)

        # 5. Spatial Face-to-Body Association
        face_boxes = [(f.x1, f.y1, f.x2, f.y2) for f in detected_faces]
        track_boxes = [t.body_box for t in self.tracks]

        face_track_matches = SpatialFaceBodyMapper.match_faces_to_bodies(face_boxes, track_boxes)
        matched_track_ids = set()

        for f_idx, t_idx, affinity in face_track_matches:
            target_track = self.tracks[t_idx]
            face = detected_faces[f_idx]
            matched_track_ids.add(target_track.track_id)

            # Recognize identity via ArcFace
            aligned, _ = align_face_5pt(proc_frame, face.kps, out_size=(112, 112))
            if aligned is not None and aligned.size > 0:
                emb = self.embedder.embed(aligned)
                match_res = self.matcher.match(emb)
            else:
                match_res = MatchResult(name="Unknown", distance=1.0, similarity=0.0, accepted=False)

            if match_res.accepted:
                target_track.update_with_confirmed_face(
                    name=match_res.name,
                    match_similarity=match_res.similarity,
                    face_box=(face.x1, face.y1, face.x2, face.y2),
                    face_kps=face.kps,
                    current_time=current_time,
                )
            else:
                # Face seen but not recognized in database
                target_track.current_face_box = (face.x1, face.y1, face.x2, face.y2)
                target_track.current_face_kps = face.kps
                target_track.frames_since_face = 0
                if target_track.state == TrackState.UNKNOWN:
                    target_track.identity = "Unknown"
                    target_track.confidence = 0.25

        # 6. Apply Honest Three-State Confidence Decay for Tracks without Faces
        for track in self.tracks:
            if track.track_id not in matched_track_ids:
                track.update_without_face(current_time)

        # 7. Prune dead tracks that have been lost for too many frames
        self.tracks = [
            t for t in self.tracks
            if t.kalman.time_since_update <= self.max_lost_frames
        ]

        # 8. Manage selected target lock
        self._update_target_selection()

        return self.tracks

    def _update_target_selection(self):
        """Automatically locks onto enrolled target name or highest confidence track."""
        if not self.tracks:
            self.selected_target_id = None
            return

        # If a specific target name was requested (e.g. Sonia)
        if self.target_name:
            target_lower = self.target_name.strip().lower()
            for t in self.tracks:
                if t.identity.strip().lower() == target_lower and t.state != TrackState.UNKNOWN:
                    self.selected_target_id = t.track_id
                    return

        # If previously locked track is still active, keep lock
        if self.selected_target_id is not None:
            for t in self.tracks:
                if t.track_id == self.selected_target_id:
                    return

        # Fallback to track with highest confidence
        best_track = max(self.tracks, key=lambda t: t.confidence)
        self.selected_target_id = best_track.track_id

    def get_selected_target(self) -> Optional[TrackedPerson]:
        """Returns the currently locked target person."""
        if self.selected_target_id is None:
            return None
        for t in self.tracks:
            if t.track_id == self.selected_target_id:
                return t
        return None
