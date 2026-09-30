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


def load_target_embedding(target_name: str) -> np.ndarray:
    if not DB_JSON.exists() or not DB_NPZ.exists():
        raise RuntimeError(
            f"No face database found. Run `python -m src.enroll --name {target_name}` first."
        )

    with open(DB_JSON, "r") as f:
        meta = json.load(f)

    npz = np.load(DB_NPZ)
    if target_name in npz.files:
        emb = npz[target_name]
    else:
        names = meta.get("names", [])
        if target_name not in names:
            available = names if names else list(npz.files)
            raise RuntimeError(
                f"'{target_name}' not found in database. Enrolled identities: {available}"
            )
        idx = names.index(target_name)
        emb = npz["embeddings"][idx]

    emb = np.asarray(emb, dtype=np.float32).reshape(-1)
    emb = emb / (np.linalg.norm(emb) + 1e-12)
    return emb


def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(1.0 - np.dot(a, b))


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
    parser.add_argument("--invert", action="store_true",
                        help="Invert left/right mapping")
    parser.add_argument("--deadzone", type=float, default=0.04,
                        help="Center deadzone ratio (default: 0.04 = +-4 percent)")
    args = parser.parse_args()

    target_name = args.target
    target_emb = load_target_embedding(target_name)
    print(f"\n[Identity Lock-in] Target '{target_name}' loaded successfully.")

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
    last_seen_time = time.time()

    print("\n" + "=" * 60)
    print("  TACHOMETER STEPPER FACE TRACKER RUNNING")
    print("  Needle Mapping: 0 deg (Left) <---> 90 deg (Center) <---> 180 deg (Right)")
    print("  Hotkeys:")
    print("    [I] : Invert direction (toggle left/right mirror)")
    print("    [R] : Reset tachometer needle to center (90 deg)")
    print("    [Q] : Quit")
    print("=" * 60 + "\n")

    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                time.sleep(0.01)
                continue

            H, W = frame.shape[:2]
            center_x = W / 2.0
            faces = det.detect(frame, max_faces=6)

            best_dist = None
            target_face = None
            now = time.time()

            # Identify faces and lock in to requested target
            for f in faces:
                aligned, _ = align_face_5pt(frame, f.kps, out_size=(112, 112))
                if aligned is None or not aligned.size:
                    continue

                emb_res = embedder.embed(aligned)
                dist = cosine_distance(emb_res.embedding, target_emb)

                is_target = dist < MATCH_THRESHOLD
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

                # Sub-pixel landmark centroid: 60% nose tip + 40% eye midpoint
                if target_face.kps is not None and len(target_face.kps) == 5:
                    eye_mid_x = (target_face.kps[0][0] + target_face.kps[1][0]) / 2.0
                    nose_x = target_face.kps[2][0]
                    face_x = 0.6 * nose_x + 0.4 * eye_mid_x
                    face_y = target_face.kps[2][1]
                else:
                    face_x = (target_face.x1 + target_face.x2) / 2.0
                    face_y = (target_face.y1 + target_face.y2) / 2.0

                # Horizontal normalized position from 0.0 (left) to 1.0 (right)
                norm_x = np.clip(face_x / float(W), 0.0, 1.0)

                # Centered offset [-0.5, +0.5]
                offset_from_center = norm_x - 0.5

                # 1000-pixel tracking span mapping:
                # Center = 0 offset, Far-left = -500 px, Far-right = +500 px
                pixel_offset = int(round(offset_from_center * 1000.0))
                # Stepper mapping: 1000 steps span (-500 to +500 steps)
                step_offset = pixel_offset  # 1 px = 1 step precision!

                if abs(offset_from_center) < args.deadzone:
                    target_angle = 90.0
                    direction_label = "CENTER [12:00 LOCKED]"
                else:
                    if invert_direction:
                        # Inverted: Left -> 180°, Right -> 0°
                        target_angle = (1.0 - norm_x) * 180.0
                        direction_label = "RIGHT [-> 3:00]" if norm_x > 0.5 else "LEFT [<- 9:00]"
                    else:
                        # Normal: Left -> 0° (9:00), Center -> 90° (12:00), Right -> 180° (3:00)
                        target_angle = norm_x * 180.0
                        direction_label = "RIGHT [-> 3:00]" if norm_x > 0.5 else "LEFT [<- 9:00]"

                current_angle = float(np.clip(target_angle, 0.0, 180.0))

                # Visual Reticle on locked target
                cv2.circle(frame, (int(face_x), int(face_y)), 6, (0, 255, 0), -1)
                cv2.circle(frame, (int(face_x), int(face_y)), 22, (0, 255, 255), 2)
                cv2.line(frame, (int(center_x), int(H / 2)), (int(face_x), int(face_y)), (0, 255, 255), 2)

            else:
                pixel_offset = 0
                step_offset = 0
                if now - last_seen_time > 2.0:
                    direction_label = "TARGET LOST (Holding 90° / 12:00)"
                    # Optionally recenter to 90° when target is lost for >2 seconds
                    current_angle = 90.0
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
            # Center Vertical Guide Line
            cv2.line(frame, (int(center_x), 0), (int(center_x), H), (0, 180, 255), 1)

            # Title & Target Info
            cv2.putText(frame, f"Identity Lock: {target_name} [{direction_label}]",
                        (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"Angle: {angle_int} deg | Offset: {pixel_offset:+d} px | Step: {step_offset:+d} steps",
                        (15, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
            cv2.putText(frame, f"Precision: 1000px = 1000 steps (1px = 1 step = 0.18 deg)",
                        (15, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (100, 255, 255), 1)
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
            cv2.putText(frame, "9:00 (-500)", (dial_cx - dial_r - 20, dial_cy + 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 255), 1)
            cv2.putText(frame, "12:00 (0)", (dial_cx - 25, dial_cy - dial_r - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 255, 0), 1)
            cv2.putText(frame, "3:00 (+500)", (dial_cx + dial_r - 15, dial_cy + 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 255), 1)

            cv2.imshow("MQTT Wireless Tachometer Face Tracker", frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), 27):
                break
            elif key in (ord('i'), ord('I')):
                invert_direction = not invert_direction
                print(f"[Config] Invert direction toggled: {invert_direction}")
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
