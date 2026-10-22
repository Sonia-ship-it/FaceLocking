# src/person_face_tracker.py
"""
Whole-Person Tracking & Face-Association Interactive Live System.
Integrates:
  - Face Detection: Modern Deep Learning YuNet (OpenCV FaceDetectorYN)
  - Person Detection: YOLOX-Nano ONNX + MediaPipe Pose
  - Face Recognition: ArcFace 512-D ONNX Embeddings
  - Appearance Re-Identification: Torso Clothing Color Signature (Tracks across back-turns)
  - Honest Three-State Model: Confirmed, Temporary, Unknown with exponential decay
  - Robotic Automatic Locking:
      * Lock by Unknown: Facility security mode (auto-locks onto unauthorized strangers / intruders)
      * Lock by Known: VIP / authorized personnel follow mode (auto-locks onto enrolled members)
      * Manual Lock: Operator-directed
  - Closed-Loop Virtual PTZ Software Gimbal (No physical hardware required)
"""
from __future__ import annotations

import argparse
import math
import sys
import time
from typing import Optional, Tuple
import cv2
import numpy as np

from src.camera import open_camera, print_camera_list
from src.person_tracker import (
    WholePersonFaceTracker,
    TrackedPerson,
    TrackState,
    LockPolicy,
)
from src.target_centering import VirtualPTZController, CenteringSignal


# Modern color palette (BGR)
COLOR_CONFIRMED = (46, 204, 113)     # Emerald Green
COLOR_TEMPORARY = (30, 165, 245)     # Amber / Orange
COLOR_UNKNOWN = (70, 70, 240)        # Security Alert Crimson / Red
COLOR_HUD_BG = (20, 20, 24)          # Deep Charcoal
COLOR_ACCENT = (255, 191, 0)         # Cyan / Yellow accent
COLOR_RETICLE = (0, 255, 255)        # Electric Yellow
COLOR_ALERT = (0, 0, 255)            # Bright Red


def draw_hud_banner(
    frame: np.ndarray,
    target: Optional[TrackedPerson],
    centering: CenteringSignal,
    fps: float,
    lock_policy: LockPolicy,
    head_turn_sim: bool,
    occlusion_sim: bool,
    show_ptz: bool,
):
    """
    Renders an intelligent floating telemetry HUD showing locking policy,
    intrusion alerts, appearance ReID matching, and gimbal tracking.
    """
    h, w = frame.shape[:2]

    # Top floating telemetry card
    card_h = 62
    card_w = min(w - 20, 960)
    card_x = 10
    card_y = 8

    card_roi = frame[card_y:card_y + card_h, card_x:card_x + card_w]
    overlay = card_roi.copy()
    cv2.rectangle(overlay, (0, 0), (card_w, card_h), COLOR_HUD_BG, -1)
    cv2.addWeighted(overlay, 0.72, card_roi, 0.28, 0, card_roi)
    cv2.rectangle(frame, (card_x, card_y), (card_x + card_w, card_y + card_h), (70, 75, 85), 1)

    # Policy badge text
    if lock_policy == LockPolicy.LOCK_UNKNOWN:
        policy_label = "FACILITY SECURITY [LOCK UNKNOWN INTRUDER]"
        policy_col = (50, 100, 255)
    elif lock_policy == LockPolicy.LOCK_KNOWN:
        policy_label = "AUTHORIZED FOLLOW [LOCK KNOWN MEMBER]"
        policy_col = (0, 220, 150)
    else:
        policy_label = "MANUAL LOCK MODE [OPERATOR DIRECTED]"
        policy_col = (200, 200, 200)

    cv2.putText(frame, f"POLICY: {policy_label}", (card_x + 12, card_y + 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.44, policy_col, 1, cv2.LINE_AA)
    cv2.putText(frame, f"CPU FPS: {fps:.1f}", (card_x + card_w - 110, card_y + 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.44, (0, 255, 200), 1, cv2.LINE_AA)

    # Target info line & intrusion warnings
    if target is not None:
        if target.state == TrackState.UNKNOWN:
            state_col = COLOR_UNKNOWN
            badge_text = "[INTRUDER / UNKNOWN]"
        elif target.state == TrackState.TEMPORARY:
            state_col = COLOR_TEMPORARY
            badge_text = "[TEMPORARY]"
        else:
            state_col = COLOR_CONFIRMED
            badge_text = "[CONFIRMED]"

        cv2.putText(frame, badge_text, (card_x + 12, card_y + 44), cv2.FONT_HERSHEY_SIMPLEX, 0.52, state_col, 2, cv2.LINE_AA)

        target_info = f"ID: {target.identity} #{target.track_id} | Conf: {target.confidence * 100:.1f}%"
        cv2.putText(frame, target_info, (card_x + 175, card_y + 44), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (230, 230, 230), 1, cv2.LINE_AA)

        # Confidence Decay Bar
        bar_x = card_x + 420
        bar_y = card_y + 32
        bar_w = 100
        bar_h = 12
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (45, 45, 50), -1)
        fill_w = int(bar_w * max(0.0, min(1.0, target.confidence)))
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill_w, bar_y + bar_h), state_col, -1)
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (110, 110, 120), 1)

        # Appearance ReID tag or Facial loss timer
        if target.is_facing_away:
            cv2.putText(frame, f"[BACK TURNED // Appearance ReID: {target.appearance_similarity * 100:.0f}%]",
                        (bar_x + bar_w + 10, card_y + 43), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 220, 255), 1, cv2.LINE_AA)
        elif target.state == TrackState.TEMPORARY and target.last_face_seen_time:
            dt_loss = time.time() - target.last_face_seen_time
            cv2.putText(frame, f"Face Lost: {dt_loss:.1f}s", (bar_x + bar_w + 10, card_y + 43),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, COLOR_TEMPORARY, 1, cv2.LINE_AA)
        elif target.state == TrackState.UNKNOWN and lock_policy == LockPolicy.LOCK_UNKNOWN:
            cv2.putText(frame, "🚨 INTRUDER AUTO-LOCKED", (bar_x + bar_w + 10, card_y + 43),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.44, (50, 50, 255), 2, cv2.LINE_AA)

    else:
        status_msg = "FACILITY SECURE // SCANNING FOR UNKNOWN INTRUDERS..." if lock_policy == LockPolicy.LOCK_UNKNOWN else "SEARCHING FOR REGISTERED PERSONNEL..."
        cv2.putText(frame, status_msg, (card_x + 12, card_y + 44),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.46, (160, 160, 160), 1, cv2.LINE_AA)

    # Sleek bottom floating pill for Gimbal & scenarios status
    pill_h = 28
    pill_w = min(w - 20, 920)
    pill_x = 10
    pill_y = h - pill_h - 10

    pill_roi = frame[pill_y:pill_y + pill_h, pill_x:pill_x + pill_w]
    overlay_bot = pill_roi.copy()
    cv2.rectangle(overlay_bot, (0, 0), (pill_w, pill_h), COLOR_HUD_BG, -1)
    cv2.addWeighted(overlay_bot, 0.72, pill_roi, 0.28, 0, pill_roi)
    cv2.rectangle(frame, (pill_x, pill_y), (pill_x + pill_w, pill_y + pill_h), (70, 75, 85), 1)

    deadband_status = "CENTERED" if centering.in_deadband else "ADJUSTING"
    ptz_str = (
        f"GIMBAL: Pan {centering.pan_angle_deg:+.1f} deg | Tilt {centering.tilt_angle_deg:+.1f} deg | "
        f"Err ({centering.norm_error_x:+.2f}, {centering.norm_error_y:+.2f}) [{deadband_status}] | Policy Key: [M]"
    )
    cv2.putText(frame, ptz_str, (pill_x + 10, pill_y + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 235, 255), 1, cv2.LINE_AA)

    indicators = []
    if head_turn_sim:
        indicators.append("[H: TURN ACTIVE]")
    if occlusion_sim:
        indicators.append("[O: OCCLUDED]")
    if show_ptz:
        indicators.append("[P: PTZ]")

    if indicators:
        tag_str = " ".join(indicators)
        cv2.putText(frame, tag_str, (pill_x + pill_w - 230, pill_y + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 220, 255), 1, cv2.LINE_AA)


def draw_centering_reticle(
    frame: np.ndarray,
    target_pt: Optional[Tuple[float, float]],
    centering: CenteringSignal,
):
    h, w = frame.shape[:2]
    cx, cy = w // 2, h // 2

    # Optical center deadband box
    db_w = int(w * 0.06)
    db_h = int(h * 0.06)
    cv2.rectangle(frame, (cx - db_w, cy - db_h), (cx + db_w, cy + db_h), (80, 80, 90), 1)
    cv2.line(frame, (cx - 10, cy), (cx + 10, cy), (120, 120, 140), 1)
    cv2.line(frame, (cx, cy - 10), (cx, cy + 10), (120, 120, 140), 1)

    if target_pt is not None:
        tx, ty = int(target_pt[0]), int(target_pt[1])

        vector_col = (0, 255, 200) if centering.in_deadband else (0, 150, 255)
        cv2.arrowedLine(frame, (cx, cy), (tx, ty), vector_col, 2, tipLength=0.18)

        # Target reticle
        cv2.circle(frame, (tx, ty), 16, COLOR_RETICLE, 2)
        cv2.circle(frame, (tx, ty), 4, COLOR_RETICLE, -1)
        cv2.line(frame, (tx - 24, ty), (tx + 24, ty), COLOR_RETICLE, 1)
        cv2.line(frame, (tx, ty - 24), (tx, ty + 24), COLOR_RETICLE, 1)


def draw_track_visuals(frame: np.ndarray, track: TrackedPerson, is_selected: bool):
    bx1, by1, bx2, by2 = track.body_box

    if track.state == TrackState.CONFIRMED:
        col = COLOR_CONFIRMED
        border_thick = 3 if is_selected else 2
    elif track.state == TrackState.TEMPORARY:
        col = COLOR_TEMPORARY
        border_thick = 3 if is_selected else 2
    else:
        col = COLOR_UNKNOWN
        border_thick = 3 if is_selected else 1

    # Body Box
    cv2.rectangle(frame, (bx1, by1), (bx2, by2), col, border_thick)

    # Face Box (if visible)
    if track.current_face_box is not None:
        fx1, fy1, fx2, fy2 = track.current_face_box
        cv2.rectangle(frame, (fx1, fy1), (fx2, fy2), (0, 255, 255), 2)
        cv2.line(frame, (int((fx1 + fx2) / 2), fy2), (int((bx1 + bx2) / 2), by1), (0, 255, 255), 1)

    # Label text
    if track.is_facing_away:
        label = f"{track.identity} #{track.track_id} [BACK TURNED (ReID {track.appearance_similarity * 100:.0f}%)]"
    else:
        state_tag = "INTRUDER" if track.state == TrackState.UNKNOWN else track.state.value.upper()
        label = f"{track.identity} #{track.track_id} [{state_tag}] {track.confidence * 100:.0f}%"

    (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.46, 1)
    label_y1 = max(0, by1 - lh - 8)
    cv2.rectangle(frame, (bx1, label_y1), (bx1 + lw + 8, by1), col, -1)
    cv2.putText(frame, label, (bx1 + 4, by1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (255, 255, 255), 1, cv2.LINE_AA)

    # Motion Trajectory tail
    if len(track.kalman.history) > 1:
        pts = np.array(track.kalman.history, dtype=np.int32).reshape((-1, 1, 2))
        cv2.polylines(frame, [pts], isClosed=False, color=col, thickness=2)


def main():
    parser = argparse.ArgumentParser(description="Whole-Person Tracking with YuNet, YOLOX, Appearance ReID & Auto-Locking")
    parser.add_argument("--cam", default="internal", help="Camera index ('internal' for computer webcam, 'auto' for external)")
    parser.add_argument("--target", default="Sonia", help="Target identity name to prioritize locking onto in known mode")
    parser.add_argument("--lock-policy", default="unknown", choices=["unknown", "known", "manual"],
                        help="Robotic locking policy: 'unknown' (facility security intruder lock), 'known' (authorized follow), or 'manual'")
    parser.add_argument("--decay-half-life", type=float, default=2.5, help="Confidence decay half-life in seconds")
    parser.add_argument("--match-thresh", type=float, default=0.40, help="ArcFace match distance threshold")
    parser.add_argument("--list-cams", action="store_true", help="List detected cameras and exit")
    parser.add_argument("--headless", action="store_true", help="Run in headless verification mode without GUI")
    parser.add_argument("--max-frames", type=int, default=0, help="Exit after N frames (useful for testing)")
    args = parser.parse_args()

    if args.list_cams:
        print_camera_list()
        return

    # Map lock policy
    policy_map = {
        "unknown": LockPolicy.LOCK_UNKNOWN,
        "known": LockPolicy.LOCK_KNOWN,
        "manual": LockPolicy.MANUAL,
    }
    cur_policy = policy_map[args.lock_policy]

    print("\n" + "=" * 68)
    print("WHOLE-PERSON TRACKER // ROBOTIC AUTO-LOCKING & APPEARANCE ReID")
    print("=" * 68)
    print(f"[*] Locking Policy:     {cur_policy.value}")
    print(f"[*] Target Identity:    {args.target}")
    print(f"[*] Face Detector:      OpenCV YuNet Deep Learning (CNN ONNX)")
    print(f"[*] Body Detector:      Hybrid (YOLOX-Nano ONNX + MediaPipe Pose)")
    print(f"[*] Appearance Model:   Torso Clothing Color Signature (ReID across Back Turns)")
    print(f"[*] Camera Selection:   {args.cam} (Computer Webcam)")
    print("=" * 68)

    tracker = WholePersonFaceTracker(
        target_name=args.target,
        decay_half_life=args.decay_half_life,
        match_thresh=args.match_thresh,
        lock_policy=cur_policy,
    )

    ptz_controller = VirtualPTZController()

    camera = open_camera(args.cam)
    if camera is None or not camera.isOpened():
        print(f"ERROR: Could not open camera '{args.cam}'. Falling back to camera 0...")
        camera = open_camera(0)
        if camera is None or not camera.isOpened():
            print("ERROR: Could not open any camera device.")
            sys.exit(1)

    WINDOW_NAME = "Whole-Person Tracker & Face Association (YuNet + YOLOX + ReID)"

    if not args.headless:
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WINDOW_NAME, 1180, 740)

    print("\nInteractive Hotkeys:")
    print("  [M]     : Cycle Locking Policy (Lock Unknown Intruder <-> Lock Known Member <-> Manual)")
    print("  [H]     : Toggle Head-Turn Mode (simulates facial loss to test Temporary decay)")
    print("  [O]     : Toggle Simulated Occluder Box (tests passing behind obstacles)")
    print("  [P]     : Toggle Virtual PTZ Zoomed Software Gimbal View")
    print("  [F]     : Toggle Fullscreen Mode")
    print("  [U]     : Toggle HUD Overlay On/Off")
    print("  [T]     : Cycle Locked Target Track (in Manual mode)")
    print("  [C]     : Recenter Virtual Gimbal to 0 deg")
    print("  [Q/ESC] : Quit")
    print("=" * 68 + "\n")

    sim_head_turn = False
    sim_occlusion = False
    show_ptz_view = True
    show_hud = True
    is_fullscreen = False

    fps = 0.0
    frame_times = []
    frame_idx = 0

    try:
        while True:
            t0 = time.time()
            ret, frame = camera.read()
            if not ret or frame is None:
                print("End of camera stream or failed frame grab.")
                break

            h, w = frame.shape[:2]
            frame_idx += 1

            # Simulated occlusion box coordinates
            occlusion_box = None
            if sim_occlusion:
                ow = int(w * 0.28)
                oh = int(h * 0.60)
                ox1 = int(w * 0.36)
                oy1 = int(h * 0.15)
                occlusion_box = (ox1, oy1, ox1 + ow, oy1 + oh)

            # Process frame through Whole-Person Face Tracker
            tracks = tracker.process_frame(
                frame=frame,
                current_time=t0,
                suppress_face=sim_head_turn,
                simulated_occlusion_box=occlusion_box,
            )

            # Get selected target based on active lock policy
            target = tracker.get_selected_target()
            target_pt = target.upper_center if target is not None else None

            # Update Virtual PTZ Centering Controller
            centering = ptz_controller.update(
                target_pt=target_pt,
                frame_shape=(h, w),
                current_time=t0,
            )

            # Draw visual tracking annotations
            display_frame = frame.copy()

            for t in tracks:
                is_selected = (target is not None and t.track_id == target.track_id)
                draw_track_visuals(display_frame, t, is_selected)

            draw_centering_reticle(display_frame, target_pt, centering)

            if occlusion_box is not None:
                ox1, oy1, ox2, oy2 = occlusion_box
                cv2.rectangle(display_frame, (ox1, oy1), (ox2, oy2), (40, 40, 45), -1)
                cv2.putText(display_frame, "|| OCCLUDER OBSTACLE ||", (ox1 + 10, oy1 + 35),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 200, 255), 2)

            if show_hud:
                draw_hud_banner(
                    frame=display_frame,
                    target=target,
                    centering=centering,
                    fps=fps,
                    lock_policy=tracker.lock_policy,
                    head_turn_sim=sim_head_turn,
                    occlusion_sim=sim_occlusion,
                    show_ptz=show_ptz_view,
                )

            # Virtual PTZ Viewport in top right
            if show_ptz_view:
                ptz_viewport = ptz_controller.render_virtual_ptz_viewport(frame, out_size=(240, 180))
                pip_h, pip_w = ptz_viewport.shape[:2]
                pip_x = w - pip_w - 15
                pip_y = 15
                display_frame[pip_y:pip_y + pip_h, pip_x:pip_x + pip_w] = ptz_viewport
                cv2.rectangle(display_frame, (pip_x, pip_y), (pip_x + pip_w, pip_y + pip_h), (0, 200, 255), 2)

            # Measure FPS
            t1 = time.time()
            dt = t1 - t0
            frame_times.append(dt)
            if len(frame_times) > 20:
                frame_times.pop(0)
            fps = len(frame_times) / sum(frame_times)

            if not args.headless:
                cv2.imshow(WINDOW_NAME, display_frame)
                key = cv2.waitKey(1) & 0xFF

                if key in (ord('q'), 27):  # Q or ESC
                    break
                elif key == ord('m'):
                    # Cycle locking policy
                    policies = [LockPolicy.LOCK_UNKNOWN, LockPolicy.LOCK_KNOWN, LockPolicy.MANUAL]
                    cur_idx = policies.index(tracker.lock_policy)
                    tracker.lock_policy = policies[(cur_idx + 1) % len(policies)]
                    print(f"[*] Switched Locking Policy to: {tracker.lock_policy.value}")
                elif key == ord('h'):
                    sim_head_turn = not sim_head_turn
                    print(f"[*] Head-Turn Simulation: {'ACTIVE (Face Suppressed)' if sim_head_turn else 'OFF (Face Enabled)'}")
                elif key == ord('o'):
                    sim_occlusion = not sim_occlusion
                    print(f"[*] Occluder Obstacle:     {'ACTIVE' if sim_occlusion else 'OFF'}")
                elif key == ord('p'):
                    show_ptz_view = not show_ptz_view
                elif key == ord('u'):
                    show_hud = not show_hud
                    print(f"[*] HUD Overlay: {'ENABLED' if show_hud else 'DISABLED (Clean View)'}")
                elif key == ord('f'):
                    is_fullscreen = not is_fullscreen
                    prop = cv2.WINDOW_FULLSCREEN if is_fullscreen else cv2.WINDOW_NORMAL
                    cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, prop)
                elif key == ord('c'):
                    ptz_controller.reset()
                    print("[*] Virtual Gimbal Recentered to 0 deg.")
                elif key == ord('t'):
                    if tracks:
                        cur_ids = [t.track_id for t in tracks]
                        if tracker.selected_target_id in cur_ids:
                            idx = (cur_ids.index(tracker.selected_target_id) + 1) % len(cur_ids)
                            tracker.selected_target_id = cur_ids[idx]
                        else:
                            tracker.selected_target_id = cur_ids[0]
                        print(f"[*] Switched to Track #{tracker.selected_target_id}")

            if args.max_frames > 0 and frame_idx >= args.max_frames:
                print(f"[*] Reached max frames limit ({args.max_frames}). Exiting.")
                break

    finally:
        camera.release()
        if not args.headless:
            cv2.destroyAllWindows()
        print("\n[*] Tracker session completed successfully.")


if __name__ == "__main__":
    main()
