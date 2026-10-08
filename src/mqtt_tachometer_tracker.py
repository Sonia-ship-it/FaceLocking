# src/mqtt_tachometer_tracker.py
"""
Face Recognition & Identity Lock-in Wireless Tachometer Stepper Tracker.

Data Flow:
  Webcam / IP Cam
      │
      ▼
  5-Point Landmark Detection + ArcFace Identity Matching
      │
      ▼
  Locked Target Face Centroid (Left / Center / Right)
      │
      ▼
  Horizontal Mapping to Tachometer Needle Degrees (0° to 180°)
      │
      ▼
  Paho MQTT Client (peho) -> Mosquitto Broker -> NodeMCU (ESP8266) -> Stepper Driver -> Tachometer Needle
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Union, Optional

import cv2
import numpy as np
import paho.mqtt.client as mqtt

from .haar_5pt import Haar5ptDetector, align_face_5pt
from .embed import ArcFaceEmbedderONNX
from .camera import open_camera, print_camera_list

DB_JSON = Path("data/db/face_db.json")
DB_NPZ = Path("data/db/face_db.npz")
MATCH_THRESHOLD = 0.40  # Cosine distance cutoff for target identity lock-in


def load_target_embedding(target_name: str) -> tuple[np.ndarray, str]:
    if not DB_JSON.exists() or not DB_NPZ.exists():
        raise RuntimeError(
            f"No face database found. Run `python -m src.enroll --name {target_name}` first."
        )

    with open(DB_JSON, "r") as f:
        meta = json.load(f)

    npz = np.load(DB_NPZ)
    names = meta.get("names", [])
    files = list(npz.files)

    # Build case-insensitive lookup maps
    name_map = {n.lower(): n for n in names}
    file_map = {k.lower(): k for k in files if k != "embeddings"}
    target_lower = target_name.strip().lower()

    resolved_name = target_name
    if target_name in npz.files:
        emb = npz[target_name]
    elif target_lower in file_map:
        resolved_name = file_map[target_lower]
        emb = npz[resolved_name]
    elif target_lower in name_map:
        resolved_name = name_map[target_lower]
        idx = names.index(resolved_name)
        emb = npz["embeddings"][idx]
    elif target_name in names:
        idx = names.index(target_name)
        emb = npz["embeddings"][idx]
    else:
        available = names if names else files
        raise RuntimeError(
            f"'{target_name}' not found in database. Enrolled identities: {available}"
        )

    emb = np.asarray(emb, dtype=np.float32).reshape(-1)
    emb = emb / (np.linalg.norm(emb) + 1e-12)
    return emb, resolved_name


def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(1.0 - np.dot(a, b))


def box_iou(b1: tuple[int, int, int, int], b2: tuple[int, int, int, int]) -> float:
    ix1, iy1 = max(b1[0], b2[0]), max(b1[1], b2[1])
    ix2, iy2 = min(b1[2], b2[2]), min(b1[3], b2[3])
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = float((ix2 - ix1) * (iy2 - iy1))
    a1 = float(max(1, (b1[2] - b1[0]) * (b1[3] - b1[1])))
    a2 = float(max(1, (b2[2] - b2[0]) * (b2[3] - b2[1])))
    return inter / (a1 + a2 - inter)


def main():
    parser = argparse.ArgumentParser(
        description="Wireless Tachometer Stepper Face Tracker (MQTT + NodeMCU)"
    )
    parser.add_argument("--target", "-t", type=str, required=True,
                        help="Enrolled identity name to lock in and track")
    parser.add_argument("--broker", "-b", type=str, default="localhost",
                        help="Mosquitto MQTT Broker address (default: localhost)")
    parser.add_argument("--port", "-p", type=int, default=1883,
                        help="MQTT Broker port (default: 1883)")
    parser.add_argument("--topic", type=str, default="tachometer/angle",
                        help="MQTT topic to publish target angle numbers (default: tachometer/angle)")
    parser.add_argument("--cam", default="auto",
                        help="Camera index or stream URL (default: auto)")
    parser.add_argument("--axis", "-a", type=str, choices=["vertical", "horizontal"], default="vertical",
                        help="Tracking axis: 'vertical' (Up/Down) or 'horizontal' (Left/Right) (default: vertical)")
    parser.add_argument("--invert", action="store_true",
                        help="Invert mapping direction")
    parser.add_argument("--deadzone", type=float, default=0.04,
                        help="Center deadzone ratio (default: 0.04 = +-4 percent)")
    parser.add_argument("--threshold", type=float, default=0.48,
                        help="Cosine distance cutoff for target face matching (default: 0.48)")
    parser.add_argument("--offset", type=float, default=0.0,
                        help="Angle offset shift in degrees (default: 0.0)")
    args = parser.parse_args()

    target_emb, target_name = load_target_embedding(args.target)
    print(f"\n[Identity Lock-in] Target '{target_name}' loaded successfully (input: '{args.target}').")

    # ----------------- PAHO MQTT CLIENT SETUP -----------------
    nodemcu_status = "Waiting for NodeMCU telemetry..."

    def on_connect(client, userdata, flags, rc, properties=None):
        if rc == 0:
            print(f"[MQTT] Connected successfully to Mosquitto Broker @ {args.broker}:{args.port}")
            client.subscribe("tachometer/status")
        else:
            print(f"[MQTT] Connection failed with result code {rc}")

    def on_message(client, userdata, msg):
        nonlocal nodemcu_status
        payload = msg.payload.decode("utf-8", errors="ignore")
        nodemcu_status = f"[NodeMCU] {payload}"

    # Backwards & forwards compatible initialization for Paho-MQTT 1.x and 2.x
    if hasattr(mqtt, "CallbackAPIVersion"):
        mqtt_client = mqtt.Client(
            callback_api_version=mqtt.CallbackAPIVersion.VERSION2,
            client_id="PythonPahoFaceTracker"
        )
    else:
        mqtt_client = mqtt.Client(client_id="PythonPahoFaceTracker")
    mqtt_client.on_connect = on_connect
    mqtt_client.on_message = on_message

    try:
        mqtt_client.connect(args.broker, args.port, keepalive=60)
        mqtt_client.loop_start()
    except Exception as e:
        print(f"[MQTT ERROR] Failed to connect to broker at {args.broker}:{args.port} -> {e}")
        print("Please ensure Mosquitto is running (`Get-Service mosquitto` in PowerShell).")
        return

    # ----------------- CAMERA & DETECTOR SETUP -----------------
    print(f"[Camera] Opening camera source: {args.cam}")
    cap = open_camera(args.cam)
    if not cap.isOpened():
        print(f"[Camera ERROR] Could not open camera {args.cam}")
        mqtt_client.loop_stop()
        return

    det = Haar5ptDetector()
    embedder = ArcFaceEmbedderONNX()

    current_angle = 90.0  # Center needle default
    filtered_angle = 90.0
    last_sent_angle = -1
    last_send_time = 0.0
    invert_direction = args.invert
    track_axis = args.axis.lower()
    last_seen_time = time.time()
    last_locked_box: Optional[tuple[int, int, int, int]] = None
    last_locked_time = 0.0
    match_threshold = float(args.threshold)
    # Relaxed cutoff during head pitch/tilt for a face that is already locked
    retention_threshold = min(0.65, match_threshold + 0.14)
    angle_offset = float(args.offset)
    baseline_y = 0.35  # Natural sitting head level baseline

    print("\n" + "=" * 65)
    print("  TACHOMETER STEPPER FACE TRACKER RUNNING")
    print(f"  Active Axis: {track_axis.upper()} (Default)")
    print(f"  Match Threshold: {match_threshold:.2f} | Tilt Retention: {retention_threshold:.2f}")
    print(f"  Angle Zero Offset: {angle_offset:+.1f}° | Baseline Y: {baseline_y:.2f}")
    print("  Vertical Mapping: Head UP -> Needle LEFT (0°) | Head DOWN -> Needle RIGHT (180°)")
    print("  Horizontal Mapping: Head LEFT -> Needle LEFT (0°) | Head RIGHT -> Needle RIGHT (180°)")
    print("  Hotkeys:")
    print("    [C] : Zero-Calibrate current head position to 0°")
    print("    [ [ ] : Nudge angle offset by -5° / +5°")
    print("    [V] : Toggle tracking axis (Vertical <-> Horizontal)")
    print("    [I] : Invert direction (flip needle response)")
    print("    [0] : Home tachometer needle to 0°")
    print("    [R] : Reset tachometer needle to center (90°)")
    print("    [Q] : Quit")
    print("=" * 65 + "\n")

    raw_angle = 90.0

    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                time.sleep(0.01)
                continue

            H, W = frame.shape[:2]
            center_x = W / 2.0
            center_y = H / 2.0
            faces = det.detect(frame, max_faces=6)

            best_dist = None
            target_face = None
            now = time.time()

            # Identify faces and lock in to requested target
            for f in faces:
                curr_box = (f.x1, f.y1, f.x2, f.y2)
                is_spatially_locked = False

                # Check if this face spatially continues the previously locked target
                if last_locked_box is not None and (now - last_locked_time < 1.8):
                    overlap = box_iou(last_locked_box, curr_box)
                    curr_cx = (f.x1 + f.x2) / 2.0
                    curr_cy = (f.y1 + f.y2) / 2.0
                    prev_cx = (last_locked_box[0] + last_locked_box[2]) / 2.0
                    prev_cy = (last_locked_box[1] + last_locked_box[3]) / 2.0
                    center_dist = float(np.hypot(curr_cx - prev_cx, curr_cy - prev_cy))
                    diag = max(60.0, float(np.hypot(f.x2 - f.x1, f.y2 - f.y1)))

                    if overlap > 0.15 or center_dist < diag * 1.1:
                        is_spatially_locked = True

                dist = 1.0
                aligned, _ = align_face_5pt(frame, f.kps, out_size=(112, 112))
                if aligned is not None and aligned.size:
                    emb_res = embedder.embed(aligned)
                    dist = cosine_distance(emb_res.embedding, target_emb)

                # Tilt tolerance logic:
                # If already holding target lock, allow higher cosine distance caused by pitch distortion
                cutoff = retention_threshold if is_spatially_locked else match_threshold
                is_target = dist < cutoff or (is_spatially_locked and len(faces) == 1 and dist < 0.68)

                color = (0, 255, 0) if is_target else (60, 60, 60)
                cv2.rectangle(frame, (f.x1, f.y1), (f.x2, f.y2), color, 2)
                sim_pct = int(max(0.0, 1.0 - dist) * 100)
                label = f"{target_name} ({sim_pct}%)" if is_target else f"Stranger ({sim_pct}%)"
                cv2.putText(frame, label, (f.x1, max(15, f.y1 - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

                if is_target and (best_dist is None or dist < best_dist):
                    best_dist = dist
                    target_face = f

            direction_label = "SEARCHING TARGET..."

            if target_face is not None:
                last_seen_time = now
                last_locked_box = (target_face.x1, target_face.y1, target_face.x2, target_face.y2)
                last_locked_time = now

                # Sub-pixel landmark centroid: 60% nose tip + 40% eye midpoint
                if target_face.kps is not None and len(target_face.kps) == 5:
                    eye_mid_x = (target_face.kps[0][0] + target_face.kps[1][0]) / 2.0
                    eye_mid_y = (target_face.kps[0][1] + target_face.kps[1][1]) / 2.0
                    nose_x = target_face.kps[2][0]
                    nose_y = target_face.kps[2][1]
                    face_x = 0.6 * nose_x + 0.4 * eye_mid_x
                    face_y = 0.6 * nose_y + 0.4 * eye_mid_y
                else:
                    face_x = (target_face.x1 + target_face.x2) / 2.0
                    face_y = (target_face.y1 + target_face.y2) / 2.0

                if track_axis == "vertical":
                    # 1. Base vertical normalized coordinate: 0.0 (top) to 1.0 (bottom)
                    norm_pos = np.clip(face_y / float(H), 0.0, 1.0)

                    # 2. Head tilt / pitch angle detection:
                    # When tilting UP (chin up): nose tip moves closer to or above eyes -> pitch_tilt is negative
                    # When tilting DOWN (chin down): nose tip drops further below eyes -> pitch_tilt is positive
                    if target_face.kps is not None and len(target_face.kps) == 5:
                        eye_dx = abs(target_face.kps[1][0] - target_face.kps[0][0])
                        eye_span = max(20.0, float(eye_dx))
                        # Typical resting distance ratio between eyes and nose is ~0.36
                        pitch_ratio = float((nose_y - eye_mid_y) / eye_span)
                        pitch_tilt = float((pitch_ratio - 0.36) * 1.8)  # Amplified tilt signal
                    else:
                        pitch_tilt = 0.0

                    # Combined motion: 60% physical translation + 40% head tilt pitch
                    combined_norm = float(np.clip(norm_pos + pitch_tilt, 0.0, 1.0))
                    offset_from_center = combined_norm - baseline_y

                    # 1000-step tracking span: UP = negative, DOWN = positive
                    pixel_offset = int(round(offset_from_center * 1000.0))
                    step_offset = pixel_offset

                    if abs(offset_from_center) < args.deadzone:
                        raw_angle = 90.0
                        direction_label = "CENTER [12:00 LOCKED]"
                    else:
                        # Sensitivity multiplier (2.2x) maps natural head tilts to full 0..180 degree needle sweep
                        scaled_offset = offset_from_center * 2.2
                        if invert_direction:
                            raw_angle = 90.0 - (scaled_offset * 90.0)
                            direction_label = "DOWN [<- 9:00 Left]" if offset_from_center > 0 else "UP [-> 3:00 Right]"
                        else:
                            raw_angle = 90.0 + (scaled_offset * 90.0)
                            direction_label = "DOWN [-> 3:00 Right]" if offset_from_center > 0 else "UP [<- 9:00 Left]"

                    raw_angle = float(np.clip(raw_angle, 0.0, 180.0))
                else:
                    norm_pos = np.clip(face_x / float(W), 0.0, 1.0)
                    offset_from_center = norm_pos - 0.5
                    pitch_tilt = 0.0

                    pixel_offset = int(round(offset_from_center * 1000.0))
                    step_offset = pixel_offset

                    if abs(offset_from_center) < args.deadzone:
                        raw_angle = 90.0
                        direction_label = "CENTER [12:00 LOCKED]"
                    else:
                        if invert_direction:
                            raw_angle = (1.0 - norm_pos) * 180.0
                            direction_label = "RIGHT [<- 9:00 Left]" if norm_pos > 0.5 else "LEFT [-> 3:00 Right]"
                        else:
                            raw_angle = norm_pos * 180.0
                            direction_label = "RIGHT [-> 3:00 Right]" if norm_pos > 0.5 else "LEFT [<- 9:00 Left]"

                    raw_angle = float(np.clip(raw_angle, 0.0, 180.0))

                # Apply zero offset shift
                target_angle = raw_angle + angle_offset
                current_angle = float(np.clip(target_angle, 0.0, 180.0))

                # Visual Reticle on locked target
                cv2.circle(frame, (int(face_x), int(face_y)), 6, (0, 255, 0), -1)
                cv2.circle(frame, (int(face_x), int(face_y)), 22, (0, 255, 255), 2)
                cv2.line(frame, (int(center_x), int(center_y)), (int(face_x), int(face_y)), (0, 255, 255), 2)

            else:
                pixel_offset = 0
                step_offset = 0
                pitch_tilt = 0.0
                if now - last_seen_time > 2.0:
                    direction_label = "TARGET LOST (Holding 90° / 12:00)"
                    raw_angle = 90.0
                    current_angle = 90.0
                    last_locked_box = None
                else:
                    direction_label = "TRACKING (HOLD)"

            # Low-pass filter for butter-smooth tachometer needle movement
            filtered_angle = 0.65 * filtered_angle + 0.35 * current_angle
            angle_int = int(round(filtered_angle))

            # Transmit to MQTT Broker (Only when angle changes or every 250ms heartbeat)
            time_since_send = now - last_send_time
            if (angle_int != last_sent_angle and time_since_send > 0.035) or (time_since_send > 0.3):
                # We send plain number as string (e.g. "90", "135")
                mqtt_client.publish(args.topic, str(angle_int), qos=0)
                last_sent_angle = angle_int
                last_send_time = now

            # ----------------- HUD DISPLAY -----------------
            # Center Crosshairs: Primary tracking axis highlighted in orange, secondary in subtle cyan
            if track_axis == "vertical":
                cv2.line(frame, (0, int(center_y)), (W, int(center_y)), (0, 180, 255), 2)  # Active axis
                cv2.line(frame, (int(center_x), 0), (int(center_x), H), (100, 100, 100), 1)
            else:
                cv2.line(frame, (int(center_x), 0), (int(center_x), H), (0, 180, 255), 2)  # Active axis
                cv2.line(frame, (0, int(center_y)), (W, int(center_y)), (100, 100, 100), 1)

            # Title & Target Info
            axis_badge = f"AXIS: {track_axis.upper()}"
            cv2.putText(frame, f"Lock: {target_name} [{axis_badge}] [{direction_label}]",
                        (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
            cv2.putText(frame, f"Angle: {angle_int} deg (Raw: {int(round(raw_angle))} deg) | Zero Offset: {angle_offset:+.0f} deg",
                        (15, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 255, 255), 2)
            cv2.putText(frame, f"Controls: [C] Set Current as 0 deg | [ / ] Nudge Offset +-5 deg | [0] Home",
                        (15, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (100, 255, 255), 1)
            cv2.putText(frame, f"MQTT: {args.broker}:{args.port} -> {args.topic} | {nodemcu_status}",
                        (15, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1)

            # Tachometer Needle Dial Simulation on Screen
            dial_cx, dial_cy, dial_r = W - 110, 110, 75
            cv2.ellipse(frame, (dial_cx, dial_cy), (dial_r, dial_r), 0, 180, 360, (70, 70, 70), -1)
            cv2.ellipse(frame, (dial_cx, dial_cy), (dial_r, dial_r), 0, 180, 360, (0, 255, 255), 2)

            # Calculate needle endpoint on dial
            rad = np.deg2rad(180 + angle_int)
            needle_x = int(dial_cx + (dial_r - 8) * np.cos(rad))
            needle_y = int(dial_cy + (dial_r - 8) * np.sin(rad))
            cv2.line(frame, (dial_cx, dial_cy), (needle_x, needle_y), (0, 0, 255), 3)
            cv2.circle(frame, (dial_cx, dial_cy), 5, (255, 255, 255), -1)

            # Clock / Dial Indicators: 9 o'clock, 12 o'clock, 3 o'clock
            if track_axis == "vertical":
                cv2.putText(frame, "UP (9:00)", (dial_cx - dial_r - 20, dial_cy + 18),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 255), 1)
                cv2.putText(frame, "MID (12:00)", (dial_cx - 28, dial_cy - dial_r - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 255, 0), 1)
                cv2.putText(frame, "DOWN (3:00)", (dial_cx + dial_r - 20, dial_cy + 18),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 255), 1)
            else:
                cv2.putText(frame, "LEFT (9:00)", (dial_cx - dial_r - 20, dial_cy + 18),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 255), 1)
                cv2.putText(frame, "MID (12:00)", (dial_cx - 28, dial_cy - dial_r - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 255, 0), 1)
                cv2.putText(frame, "RIGHT (3:00)", (dial_cx + dial_r - 20, dial_cy + 18),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 255), 1)

            cv2.imshow("MQTT Wireless Tachometer Face Tracker", frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), 27):
                break
            elif key in (ord('c'), ord('C')):
                # Zero-calibration: whatever raw angle is right now, offset shifts it to 0!
                if target_face is not None:
                    baseline_y = float(combined_norm)
                    angle_offset = -float(raw_angle)
                    print(f"[Calibrate] Zero point calibrated to current head position! Offset: {angle_offset:+.1f} deg | Baseline: {baseline_y:.2f}")
                else:
                    print("[Calibrate] Target face not detected to calibrate zero point.")
            elif key in (ord('['), ord('-')):
                angle_offset -= 5.0
                print(f"[Trim] Angle offset adjusted: {angle_offset:+.1f} deg")
            elif key in (ord(']'), ord('+'), ord('=')):
                angle_offset += 5.0
                print(f"[Trim] Angle offset adjusted: {angle_offset:+.1f} deg")
            elif key in (ord('v'), ord('V')):
                track_axis = "horizontal" if track_axis == "vertical" else "vertical"
                print(f"[Config] Tracking axis switched to: {track_axis.upper()}")
            elif key in (ord('i'), ord('I')):
                invert_direction = not invert_direction
                print(f"[Config] Invert direction toggled: {invert_direction}")
            elif key in (ord('0'),):
                current_angle = 0.0
                filtered_angle = 0.0
                mqtt_client.publish(args.topic, "0", qos=0)
                print("[Config] Homed needle to 0 deg")
            elif key in (ord('r'), ord('R')):
                current_angle = 90.0
                filtered_angle = 90.0
                mqtt_client.publish(args.topic, "90", qos=0)
                print("[Config] Recentered needle to 90 deg")

    finally:
        cap.release()
        cv2.destroyAllWindows()
        mqtt_client.loop_stop()
        mqtt_client.disconnect()
        print("\n[Shutdown] MQTT client disconnected and camera released.")


if __name__ == "__main__":
    main()
