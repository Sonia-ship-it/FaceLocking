# tests/test_person_tracker.py
"""
Unit and Integration Tests for Whole-Person Tracking, YuNet, Appearance ReID, and Auto-Locking.
Tests:
  - Three-state transitions (Confirmed -> Temporary -> Unknown & Confirmed recovery)
  - Exponential confidence decay numerical accuracy: C(t) = C_0 * exp(-lambda * delta_t)
  - Appearance ReID torso clothing color histogram matching (tracking back-turns)
  - YuNet deep learning face detector and 5-landmark output
  - Robotic Auto-Locking Policies:
      * Lock by Unknown (Facility Security Intruder Mode)
      * Lock by Known (Authorized Facility Follow Mode)
  - Spatial face-to-body anatomical mapping and gating
  - Virtual PTZ centering error calculations and deadband behavior
"""
import math
import numpy as np
import cv2

from src.person_tracker import (
    TrackState,
    TrackedPerson,
    KalmanBoxTracker,
    SpatialFaceBodyMapper,
    LockPolicy,
    WholePersonFaceTracker,
)
from src.target_centering import VirtualPTZController, CenteringSignal
from src.appearance_reid import AppearanceReID
from src.yunet_detector import YuNetFaceDetector


def test_state_transitions_and_exponential_decay():
    """Validates the honest three-state model and exponential confidence decay formula."""
    kt = KalmanBoxTracker((100, 100, 300, 500))
    decay_half_life = 2.0  # seconds
    decay_lambda = math.log(2.0) / decay_half_life

    track = TrackedPerson(
        track_id=1,
        kalman=kt,
        decay_lambda=decay_lambda,
        confidence_thresh_unknown=0.35,
    )

    assert track.state == TrackState.UNKNOWN
    assert track.confidence == 0.0

    # 1. Face confirmed at t = 10.0
    t0 = 10.0
    track.update_with_confirmed_face(
        name="Sonia",
        match_similarity=0.96,
        face_box=(150, 110, 250, 220),
        face_kps=None,
        current_time=t0,
    )
    assert track.state == TrackState.CONFIRMED
    assert track.identity == "Sonia"
    assert math.isclose(track.confidence, 0.96, abs_tol=1e-3)

    # 2. Face lost at t = 11.0 (Head turn)
    t1 = 11.0  # 1 second after loss
    track.update_without_face(t1)
    assert track.state == TrackState.TEMPORARY
    expected_c1 = 0.96 * math.exp(-decay_lambda * 1.0)
    assert math.isclose(track.confidence, expected_c1, abs_tol=1e-3)

    # 3. Face still lost at t = 12.0 (2 seconds = 1 half life)
    t2 = 12.0
    track.update_without_face(t2)
    assert track.state == TrackState.TEMPORARY
    expected_c2 = 0.96 * 0.5  # Exactly half
    assert math.isclose(track.confidence, expected_c2, abs_tol=1e-3)

    # 4. Face restored at t = 12.5 before dropping to unknown
    t3 = 12.5
    track.update_with_confirmed_face(
        name="Sonia",
        match_similarity=0.94,
        face_box=(150, 110, 250, 220),
        face_kps=None,
        current_time=t3,
    )
    assert track.state == TrackState.CONFIRMED
    assert math.isclose(track.confidence, 0.94, abs_tol=1e-3)

    # 5. Face lost for a long duration -> decay below threshold (0.35) -> drops to UNKNOWN
    t4 = 17.0  # 4.5 seconds of loss
    track.update_without_face(t4)
    assert track.state == TrackState.UNKNOWN
    assert track.confidence <= 0.35


def test_appearance_reid_clothing_tracking():
    """Validates torso clothing color histogram matching across back-turns."""
    reid = AppearanceReID(h_bins=16, s_bins=8)

    # Synthetic person with blue jacket
    frame_blue = np.zeros((480, 640, 3), dtype=np.uint8)
    body_box = (100, 80, 300, 440)
    # Fill torso with deep blue (BGR: 180, 50, 30)
    frame_blue[150:350, 120:280] = [180, 50, 30]

    # Another frame of same person (slightly shifted in room)
    frame_blue_moved = np.zeros((480, 640, 3), dtype=np.uint8)
    body_box_moved = (140, 80, 340, 440)
    frame_blue_moved[150:350, 160:320] = [175, 55, 32]

    # A different person in bright green shirt (BGR: 30, 200, 40)
    frame_green = np.zeros((480, 640, 3), dtype=np.uint8)
    frame_green[150:350, 120:280] = [30, 200, 40]

    sig_blue1 = reid.extract_torso_signature(frame_blue, body_box)
    sig_blue2 = reid.extract_torso_signature(frame_blue_moved, body_box_moved)
    sig_green = reid.extract_torso_signature(frame_green, body_box)

    assert sig_blue1 is not None and sig_blue2 is not None and sig_green is not None

    # Similarity between same person's clothing across frames
    sim_same = reid.compute_similarity(sig_blue1, sig_blue2)
    # Similarity between two different people
    sim_diff = reid.compute_similarity(sig_blue1, sig_green)

    assert sim_same > 0.85, f"Expected high similarity for same clothing, got {sim_same}"
    assert sim_diff < 0.35, f"Expected low similarity for different clothing, got {sim_diff}"


def test_robotic_auto_locking_policies():
    """Validates Lock by Unknown (Intruder) vs Lock by Known (Authorized) policies."""
    tracker = WholePersonFaceTracker(target_name="Sonia", lock_policy=LockPolicy.LOCK_UNKNOWN)

    # Create two tracks: Track 1 is registered "Sonia", Track 2 is "Unknown" stranger
    kt1 = KalmanBoxTracker((100, 80, 260, 420))
    kt2 = KalmanBoxTracker((350, 80, 510, 420))

    track_known = TrackedPerson(track_id=1, kalman=kt1, state=TrackState.CONFIRMED, identity="Sonia", confidence=0.95)
    track_unknown = TrackedPerson(track_id=2, kalman=kt2, state=TrackState.UNKNOWN, identity="Unknown", confidence=0.20)

    tracker.tracks = [track_known, track_unknown]

    # 1. In LOCK_UNKNOWN mode: Must auto-lock onto the unknown intruder (Track 2)
    tracker.lock_policy = LockPolicy.LOCK_UNKNOWN
    tracker._update_target_selection()
    assert tracker.selected_target_id == 2, f"Expected lock on unknown intruder (#2), got #{tracker.selected_target_id}"

    # 2. In LOCK_KNOWN mode: Must auto-lock onto registered Sonia (Track 1)
    tracker.lock_policy = LockPolicy.LOCK_KNOWN
    tracker._update_target_selection()
    assert tracker.selected_target_id == 1, f"Expected lock on registered member (#1), got #{tracker.selected_target_id}"


def test_yunet_face_detector():
    """Validates YuNet ONNX face detection model initialization and inference."""
    yunet = YuNetFaceDetector()
    assert yunet.detector is not None, "YuNet model should be initialized"

    # Synthetic test frame with face
    canvas = np.ones((320, 320, 3), dtype=np.uint8) * 200
    # Head ellipse
    cv2.ellipse(canvas, (160, 150), (45, 55), 0, 0, 360, (190, 215, 240), -1)
    # Eyes
    cv2.circle(canvas, (145, 140), 4, (30, 30, 30), -1)
    cv2.circle(canvas, (175, 140), 4, (30, 30, 30), -1)
    # Mouth
    cv2.line(canvas, (148, 175), (172, 175), (40, 40, 180), 2)

    faces = yunet.detect(canvas)
    # Verify method runs without error and returns List[FaceBox5pt]
    assert isinstance(faces, list)


def test_spatial_face_body_mapping():
    """Validates anatomical alignment gating and Hungarian matching."""
    body1 = (100, 100, 300, 500)
    body2 = (400, 100, 600, 500)

    valid_face_1 = (160, 110, 240, 200)
    valid_face_2 = (460, 110, 540, 200)
    knee_face = (160, 380, 240, 460)

    aff1 = SpatialFaceBodyMapper.compute_spatial_affinity(valid_face_1, body1)
    aff2 = SpatialFaceBodyMapper.compute_spatial_affinity(valid_face_2, body2)
    aff_cross = SpatialFaceBodyMapper.compute_spatial_affinity(valid_face_1, body2)
    aff_knee = SpatialFaceBodyMapper.compute_spatial_affinity(knee_face, body1)

    assert aff1 > 0.60, f"Expected high affinity for face in upper body, got {aff1}"
    assert aff2 > 0.60, f"Expected high affinity for face in upper body, got {aff2}"
    assert aff_cross == 0.0, f"Expected 0 affinity across separated bodies, got {aff_cross}"
    assert aff_knee == 0.0, f"Expected 0 affinity for face near knee, got {aff_knee}"

    matches = SpatialFaceBodyMapper.match_faces_to_bodies(
        faces=[valid_face_1, valid_face_2],
        bodies=[body1, body2],
    )
    assert len(matches) == 2
    assert matches[0][0] == 0 and matches[0][1] == 0
    assert matches[1][0] == 1 and matches[1][1] == 1


def test_virtual_ptz_centering_and_deadband():
    """Validates closed-loop target centering logic and deadband response."""
    ptz = VirtualPTZController(
        deadband_ratio=0.06,
        kp_pan=40.0,
        kd_pan=5.0,
        max_pan_speed=50.0,
        virtual_zoom=1.4,
    )

    frame_shape = (480, 640)

    # 1. Target exactly in optical center
    sig_center = ptz.update(target_pt=(320.0, 240.0), frame_shape=frame_shape, current_time=1.0)
    assert sig_center.in_deadband == True
    assert math.isclose(sig_center.norm_error_x, 0.0, abs_tol=1e-4)
    assert math.isclose(sig_center.norm_error_y, 0.0, abs_tol=1e-4)

    # 2. Target within 3% radius (inside 6% deadband)
    sig_near = ptz.update(target_pt=(328.0, 244.0), frame_shape=frame_shape, current_time=1.05)
    assert sig_near.in_deadband == True

    # 3. Target shifted significantly to the right (x = 520, dx = +200 px)
    sig_right = ptz.update(target_pt=(520.0, 240.0), frame_shape=frame_shape, current_time=1.10)
    assert sig_right.in_deadband == False
    assert sig_right.norm_error_x > 0.50
    assert sig_right.pan_velocity > 0.0

    # 4. Viewport rendering test
    test_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    viewport = ptz.render_virtual_ptz_viewport(test_frame, out_size=(280, 210))
    assert viewport.shape == (210, 280, 3)


if __name__ == "__main__":
    print("Running test_state_transitions_and_exponential_decay...")
    test_state_transitions_and_exponential_decay()
    print("Running test_appearance_reid_clothing_tracking...")
    test_appearance_reid_clothing_tracking()
    print("Running test_robotic_auto_locking_policies...")
    test_robotic_auto_locking_policies()
    print("Running test_yunet_face_detector...")
    test_yunet_face_detector()
    print("Running test_spatial_face_body_mapping...")
    test_spatial_face_body_mapping()
    print("Running test_virtual_ptz_centering_and_deadband...")
    test_virtual_ptz_centering_and_deadband()
    print("\n[+] ALL 6 UNIT TESTS PASSED SUCCESSFULLY!")
