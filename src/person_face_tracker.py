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
    fps: float,
    lock_policy: LockPolicy,
    head_turn_sim: bool = False,
    occlusion_sim: bool = False,
):
    """
    Renders a clean, minimal status badge in the top-left corner showing
    only the essential lock policy and active target status.
    """
    if lock_policy == LockPolicy.LOCK_UNKNOWN:
        mode_str = "Auto-Lock Unknown"
        if target is not None and target.state == TrackState.UNKNOWN:
            status_text = f"🚨 LOCKED: [INTRUDER] #{target.track_id}"
            bg_col = (40, 40, 200)
        elif target is not None and target.state == TrackState.CONFIRMED:
            status_text = f"Authorized: [CONFIRMED] {target.identity} ({target.confidence * 100:.0f}%)"
            bg_col = (30, 140, 70)
        elif target is not None and target.state == TrackState.TEMPORARY:
            status_text = f"Authorized: [TEMPORARY] {target.identity} ({target.confidence * 100:.0f}%)"
            bg_col = (25, 120, 200)
        else:
            status_text = "Secure / Monitoring"
            bg_col = (28, 28, 32)
    elif lock_policy == LockPolicy.LOCK_KNOWN:
        mode_str = "Auto-Lock Known"
        if target is not None and target.state == TrackState.CONFIRMED:
            status_text = f"Locked: [CONFIRMED] {target.identity} ({target.confidence * 100:.0f}%)"
            bg_col = (30, 140, 70)
        elif target is not None and target.state == TrackState.TEMPORARY:
            status_text = f"Locked: [TEMPORARY] {target.identity} ({target.confidence * 100:.0f}%)"
            bg_col = (25, 120, 200)
        else:
            status_text = "Searching..."
            bg_col = (28, 28, 32)
    else:
        mode_str = "Manual"
        if target is not None:
            if target.state == TrackState.CONFIRMED:
                status_text = f"Locked: [CONFIRMED] {target.identity} ({target.confidence * 100:.0f}%)"
                bg_col = (30, 140, 70)
            elif target.state == TrackState.TEMPORARY:
                status_text = f"Locked: [TEMPORARY] {target.identity} ({target.confidence * 100:.0f}%)"
                bg_col = (25, 120, 200)
            else:
                status_text = f"Locked: [INTRUDER] #{target.track_id}"
                bg_col = (40, 40, 200)
        else:
            status_text = "No Target"
            bg_col = (28, 28, 32)

    sim_tag = ""
    if head_turn_sim:
        sim_tag += " [Turn Sim]"
    if occlusion_sim:
        sim_tag += " [Occl Sim]"

    info_str = f"[{mode_str}]  {status_text}  |  {fps:.0f} FPS{sim_tag}"

    (tw, th), _ = cv2.getTextSize(info_str, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1)
    px, py = 12, 12
    badge_w, badge_h = tw + 20, th + 14

    # Subtle rounded/translucent background
    roi = frame[py:py + badge_h, px:px + badge_w]
    overlay = roi.copy()
    cv2.rectangle(overlay, (0, 0), (badge_w, badge_h), bg_col, -1)
    cv2.addWeighted(overlay, 0.78, roi, 0.22, 0, roi)
    cv2.rectangle(frame, (px, py), (px + badge_w, py + badge_h), (75, 80, 90), 1)
    cv2.putText(frame, info_str, (px + 10, py + th + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA)


def draw_track_visuals(frame: np.ndarray, track: TrackedPerson, is_selected: bool, lock_policy: LockPolicy = LockPolicy.LOCK_UNKNOWN):
    """
    Renders clean, focused bounding box and minimal label on the target person.
    Focuses cleanly on the face when visible; seamlessly falls back to the body box
    when the person turns away or is partially occluded.
    """
    if track.state == TrackState.CONFIRMED:
        col = COLOR_CONFIRMED
        border_thick = 2
    elif track.state == TrackState.TEMPORARY:
        col = COLOR_TEMPORARY
        border_thick = 2
    else:
        col = COLOR_UNKNOWN
        border_thick = 2

    # Focus bounding box on the face when visible; use body box when face is turned
    if track.current_face_box is not None:
        x1, y1, x2, y2 = track.current_face_box
    else:
        x1, y1, x2, y2 = track.body_box

    # Clean focused rectangle
    cv2.rectangle(frame, (x1, y1), (x2, y2), col, border_thick)

    # Consistent, clean labels for CONFIRMED, TEMPORARY, and INTRUDER
    conf_pct = f"{track.confidence * 100:.0f}%"
    if track.state == TrackState.CONFIRMED:
        tag = f"[CONFIRMED] {track.identity} ({conf_pct})"
    elif track.state == TrackState.TEMPORARY:
        tag = f"[TEMPORARY] {track.identity} ({conf_pct})"
    else:
        tag = f"[INTRUDER] #{track.track_id}"

    label = f"LOCKED: {tag}" if is_selected else tag

    (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.46, 1)
    label_y1 = max(0, y1 - lh - 8)
    cv2.rectangle(frame, (x1, label_y1), (x1 + lw + 8, y1), col, -1)
    cv2.putText(frame, label, (x1 + 4, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (255, 255, 255), 1, cv2.LINE_AA)


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
    show_ptz_view = False
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

            # Update Virtual PTZ Centering Controller (computes pan/tilt internally)
            centering = ptz_controller.update(
                target_pt=target_pt,
                frame_shape=(h, w),
                current_time=t0,
            )

            # Draw visual tracking annotations ONLY on the target person
            display_frame = frame.copy()

            if target is not None:
                draw_track_visuals(display_frame, target, is_selected=True, lock_policy=tracker.lock_policy)

            if occlusion_box is not None:
                ox1, oy1, ox2, oy2 = occlusion_box
                cv2.rectangle(display_frame, (ox1, oy1), (ox2, oy2), (40, 40, 45), -1)
                cv2.putText(display_frame, "|| OCCLUDER OBSTACLE ||", (ox1 + 10, oy1 + 35),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 200, 255), 2)

            if show_hud:
                draw_hud_banner(
                    frame=display_frame,
                    target=target,
                    fps=fps,
                    lock_policy=tracker.lock_policy,
                    head_turn_sim=sim_head_turn,
                    occlusion_sim=sim_occlusion,
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
