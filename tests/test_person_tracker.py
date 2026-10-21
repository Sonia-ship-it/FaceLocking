# tests/test_person_tracker.py
"""
Unit and Integration Tests for Whole-Person Tracking and Face Association Module.
Tests:
  - Three-state transitions (Confirmed -> Temporary -> Unknown & Confirmed recovery)
  - Exponential confidence decay numerical accuracy: C(t) = C_0 * exp(-lambda * delta_t)
  - Spatial face-to-body anatomical mapping and gating
  - Virtual PTZ centering error calculations and deadband behavior
"""
import math
import numpy as np

from src.person_tracker import (
    TrackState,
    TrackedPerson,
    KalmanBoxTracker,
    SpatialFaceBodyMapper,
)
from src.target_centering import VirtualPTZController, CenteringSignal


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
    # Expected: 0.96 * exp(-ln(2)/2 * 1) = 0.96 * sqrt(0.5) ~ 0.6788
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


def test_spatial_face_body_mapping():
    """Validates anatomical alignment gating and Hungarian matching."""
    # Body 1: (100, 100, 300, 500) -> width 200, height 400
    # Head should be around x in [100, 300], y in [80, 260]
    body1 = (100, 100, 300, 500)
    # Body 2: (400, 100, 600, 500)
    body2 = (400, 100, 600, 500)

    # Valid face in Body 1
    valid_face_1 = (160, 110, 240, 200)
    # Valid face in Body 2
    valid_face_2 = (460, 110, 540, 200)
    # Invalid face near knees of Body 1
    knee_face = (160, 380, 240, 460)

    aff1 = SpatialFaceBodyMapper.compute_spatial_affinity(valid_face_1, body1)
    aff2 = SpatialFaceBodyMapper.compute_spatial_affinity(valid_face_2, body2)
    aff_cross = SpatialFaceBodyMapper.compute_spatial_affinity(valid_face_1, body2)
    aff_knee = SpatialFaceBodyMapper.compute_spatial_affinity(knee_face, body1)

    assert aff1 > 0.60, f"Expected high affinity for face in upper body, got {aff1}"
    assert aff2 > 0.60, f"Expected high affinity for face in upper body, got {aff2}"
    assert aff_cross == 0.0, f"Expected 0 affinity across separated bodies, got {aff_cross}"
    assert aff_knee == 0.0, f"Expected 0 affinity for face near knee, got {aff_knee}"

    # Bipartite matching
    matches = SpatialFaceBodyMapper.match_faces_to_bodies(
        faces=[valid_face_1, valid_face_2],
        bodies=[body1, body2],
    )
    assert len(matches) == 2
    # Match 0 should pair face 0 with body 0
    assert matches[0][0] == 0 and matches[0][1] == 0
    # Match 1 should pair face 1 with body 1
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

    frame_shape = (480, 640)  # Optical center: (320, 240)

    # 1. Target exactly in optical center
    sig_center = ptz.update(target_pt=(320.0, 240.0), frame_shape=frame_shape, current_time=1.0)
    assert sig_center.in_deadband is True
    assert math.isclose(sig_center.norm_error_x, 0.0, abs_tol=1e-4)
    assert math.isclose(sig_center.norm_error_y, 0.0, abs_tol=1e-4)

    # 2. Target within 3% radius (inside 6% deadband)
    sig_near = ptz.update(target_pt=(328.0, 244.0), frame_shape=frame_shape, current_time=1.05)
    assert sig_near.in_deadband is True

    # 3. Target shifted significantly to the right (x = 520, dx = +200 px, norm_error = +200/320 = +0.625)
    sig_right = ptz.update(target_pt=(520.0, 240.0), frame_shape=frame_shape, current_time=1.10)
    assert sig_right.in_deadband is False
    assert sig_right.norm_error_x > 0.50
    assert sig_right.pan_velocity > 0.0, "Pan velocity should direct gimbal to the right"

    # 4. Viewport rendering test
    test_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    viewport = ptz.render_virtual_ptz_viewport(test_frame, out_size=(280, 210))
    assert viewport.shape == (210, 280, 3)


if __name__ == "__main__":
    print("Running test_state_transitions_and_exponential_decay...")
    test_state_transitions_and_exponential_decay()
    print("Running test_spatial_face_body_mapping...")
    test_spatial_face_body_mapping()
    print("Running test_virtual_ptz_centering_and_deadband...")
    test_virtual_ptz_centering_and_deadband()
    print("\n[+] ALL UNIT TESTS PASSED SUCCESSFULLY!")
