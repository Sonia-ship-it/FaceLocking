# src/track_target.py
"""
High-Accuracy Direct 1-to-1 Horizontal Servo Tracker.

Features:
  - Exact angular mapping (FOV-calibrated horizontal face position to servo angle)
  - 5-point facial landmark sub-pixel centroid tracking (eye/nose midpoint)
  - Ultra-responsive closed-loop servo drive (35Hz update rate, low-latency serial)
  - Real-time interactive controls & calibration:
      [I] : Invert panning direction
      [R] : Recenter servo to 90°
      [1] : Instant 1-to-1 Direct Tracking Mode
      [2] : Smooth Cinematic Cameraman Mode
      [+] / [-] : Fine-tune tracking sensitivity / gain
      [SPACE] : Pause / Resume tracking
      [Q] : Quit
"""
from __future__ import annotations

import argparse
import json
import socket
import time
from pathlib import Path
from typing import Union, Optional

import cv2
import numpy as np

from .haar_5pt import Haar5ptDetector, align_face_5pt
from .embed import ArcFaceEmbedderONNX
from .camera import open_camera, print_camera_list

DB_JSON = Path("data/db/face_db.json")
DB_NPZ = Path("data/db/face_db.npz")

MATCH_THRESHOLD = 0.40
DEFAULT_CAMERA_FOV = 65.0  # Typical USB webcam horizontal field of view in degrees


def load_target_embedding(target_name: str) -> np.ndarray:
    if not DB_JSON.exists() or not DB_NPZ.exists():
        raise RuntimeError(
            f"No database found. Run `python -m src.enroll --name {target_name}` first."
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
                f"'{target_name}' not found in database. Available identities: {available}"
            )
        idx = names.index(target_name)
        emb = npz["embeddings"][idx]

    emb = np.asarray(emb, dtype=np.float32).reshape(-1)
    emb = emb / (np.linalg.norm(emb) + 1e-12)
    return emb


def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    sim = float(np.dot(a, b))
    return 1.0 - sim


def main(
    target_name: str,
    cam_index: Union[int, str] = "auto",
    serial_port: Optional[str] = None,
    baud_rate: int = 115200,
    ip: Optional[str] = None,
    udp_port: int = 8888,
    invert: bool = False,
    fov: float = DEFAULT_CAMERA_FOV,
):
    target_emb = load_target_embedding(target_name)
    print(f"\n[Target Tracker] Loaded enrolled target: '{target_name}'")

    sock = None
    if ip:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        print(f"[ESP8266 WiFi] Streaming UDP angle commands to {ip}:{udp_port}")

    ser = None
    if serial_port:
        try:
            import serial
            ser = serial.Serial(serial_port, baud_rate, timeout=0.01)
            ser.setDTR(False)
            ser.setRTS(False)
            time.sleep(2.0)  # Wait for ESP8266 reset
            ser.reset_input_buffer()
            ser.reset_output_buffer()
            print(f"[ESP8266 Serial] Connected on {serial_port} @ {baud_rate} baud.")
        except Exception as e:
            print(f"[ESP8266 Serial] Warning: {e}")

    det = Haar5ptDetector(min_size=(65, 65), smooth_alpha=0.85, debug=False)
    embedder = ArcFaceEmbedderONNX(debug=False)

    # Open physical camera
    cap = open_camera(cam_index)

    print("\n" + "=" * 65)
    print(f"  HIGH-ACCURACY 1-TO-1 HORIZONTAL TRACKER: '{target_name}'")
    print("=" * 65)
    print("Controls:")
    print("  'I'       : Invert Pan Direction (Flip if motor pans opposite)")
    print("  'M'       : Toggle Mode (1 = Camera on Servo, 2 = Fixed Camera Pointer)")
    print("  'R'       : Recenter Servo (90 deg)")
    print("  '1'       : Instant Direct 1-to-1 Mode (Max accuracy)")
    print("  '2'       : Smooth Cinematic Mode")
    print("  '+' / '-' : Fine-tune sensitivity")
    print("  SPACE     : Pause / Resume Tracking")
    print("  'Q'       : Quit")
    print("=" * 65 + "\n")

    servo_angle = 90.0
    filtered_angle = 90.0
    last_sent_angle = 90
    hardware_mode = 1  # 1 = Camera Mounted on Servo, 2 = Fixed Camera (Desk Pointer)
    tracking_enabled = True
    invert_direction = invert
    gain = 1.0
    tracking_mode = 1  # 1 = Instant 1-to-1, 2 = Smooth Cinematic
    half_fov = fov / 2.0

    last_command_time = time.time()
    last_print_time = time.time()
    last_seen_time = 0.0
    prev_angle_deg = 0.0

    # Initialize servo to 90 degrees center
    if ser:
        ser.write(b"90\n")
        ser.flush()

    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            break

        H, W = frame.shape[:2]
        center_x = W / 2.0
        center_y = H / 2.0

        faces = det.detect(frame, max_faces=6)

        best_dist = None
        target_box = None
        now = time.time()

        for f in faces:
            aligned, _ = align_face_5pt(frame, f.kps, out_size=(112, 112))
            if aligned is None or not aligned.size:
                continue

            result = embedder.embed(aligned)
            dist = cosine_distance(result.embedding, target_emb)

            is_target = dist < MATCH_THRESHOLD
            color = (0, 255, 0) if is_target else (75, 75, 75)

            cv2.rectangle(frame, (f.x1, f.y1), (f.x2, f.y2), color, 2)
            label = f"{target_name} ({1.0 - dist:.2f})" if is_target else f"Stranger ({1.0 - dist:.2f})"
            cv2.putText(frame, label, (f.x1, f.y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

            if is_target and (best_dist is None or dist < best_dist):
                best_dist = dist
                target_box = f

        direction_label = "SEARCHING..."
        offset_px = 0.0
        angle_error_deg = 0.0

        if target_box is not None:
            last_seen_time = now

            # High-precision facial anchor: Weighted average of nose tip and eye midpoint
            if target_box.kps is not None and len(target_box.kps) == 5:
                eye_mid_x = (target_box.kps[0][0] + target_box.kps[1][0]) / 2.0
                nose_x = target_box.kps[2][0]
                face_center_x = 0.6 * nose_x + 0.4 * eye_mid_x
                face_center_y = target_box.kps[2][1]
            else:
                face_center_x = (target_box.x1 + target_box.x2) / 2.0
                face_center_y = (target_box.y1 + target_box.y2) / 2.0

            # Pixel offset from optical center
            offset_px = face_center_x - center_x
            
            # Normalized horizontal error [-1.0, +1.0]
            norm_error = offset_px / center_x

            # Exact angular offset in degrees based on camera FOV
            angle_error_deg = norm_error * half_fov

            if tracking_enabled:
                # Tracking Mode Logic:
                # hardware_mode 1: Mounted Camera (Pan base - closed-loop centering)
                # hardware_mode 2: Fixed Camera (Camera on monitor, servo points at face)
                if hardware_mode == 2:
                    # Absolute direct mapping: Face angle maps directly to servo pointing angle
                    sign = -1.0 if invert_direction else 1.0
                    target_angle = 90.0 + (sign * angle_error_deg * gain)
                    servo_angle = float(np.clip(target_angle, 15.0, 165.0))
                    
                    if abs(norm_error) <= 0.06:
                        direction_label = "CENTERED [LOCKED]"
                    else:
                        direction_label = "POINTING RIGHT ->" if (target_angle > 90) else "<- POINTING LEFT"

                else:
                    # Mounted Camera Mode: Smooth closed-loop recentering with comfortable deadzone
                    # Deadzone of 7% (approx +-22px) keeps servo completely still when facing camera
                    deadzone = 0.07 if (tracking_mode == 1) else 0.10

                    if abs(norm_error) > deadzone:
                        # Smooth proportional adjustment
                        err_mag = abs(norm_error) - deadzone
                        # Max step speed 1.8 deg/frame to prevent blurring and overshooting
                        step = np.clip(err_mag * 18.0 * gain, 0.3, 1.8)
                        step = -step if (norm_error < 0) else step

                        if invert_direction:
                            step = -step

                        servo_angle = float(np.clip(servo_angle + step, 15.0, 165.0))
                        direction_label = "TRACKING RIGHT ->" if (step > 0) else "<- TRACKING LEFT"
                    else:
                        # Inside deadzone: Keep servo stable and still!
                        direction_label = "CENTERED [STABLE]"

                prev_angle_deg = angle_error_deg

            # Visual HUD Target Reticle
            cv2.circle(frame, (int(face_center_x), int(face_center_y)), 5, (0, 255, 0), -1)
            cv2.circle(frame, (int(face_center_x), int(face_center_y)), 18, (0, 255, 255), 1)
            cv2.line(frame, (int(center_x), int(center_y)), (int(face_center_x), int(face_center_y)), (0, 255, 255), 2)

        else:
            if now - last_seen_time > 1.2:
                direction_label = "TARGET LOST"
            else:
                direction_label = "HOLDING POSITION"

        # Low-pass filter for smooth motion (alpha = 0.60 direct, 0.30 cinematic)
        alpha = 0.60 if (tracking_mode == 1) else 0.30
        filtered_angle = (1.0 - alpha) * filtered_angle + alpha * servo_angle

        # Anti-jitter drive to ESP8266: Only transmit when angle integer changes by >= 1 deg
        new_int_angle = int(round(np.clip(filtered_angle, 15.0, 165.0)))
        if (ser or sock) and tracking_enabled:
            angle_delta = abs(new_int_angle - last_sent_angle)
            time_since_send = now - last_command_time

            # Transmit if angle changed, or at least every 0.5s heartbeat
            if (angle_delta >= 1 and time_since_send > 0.035) or (time_since_send > 0.5):
                cmd_bytes = f"{new_int_angle}\n".encode("ascii")

                if ser:
                    ser.write(cmd_bytes)
                    ser.flush()

                if sock and ip:
                    try:
                        sock.sendto(cmd_bytes, (ip, udp_port))
                    except Exception:
                        pass

                last_sent_angle = new_int_angle
                last_command_time = now

        if now - last_print_time > 0.25:
            hw_str = "MOUNTED CAM" if (hardware_mode == 1) else "FIXED CAM POINTER"
            err_str = f"{angle_error_deg:+5.1f} deg ({offset_px:+4.0f}px)" if target_box is not None else "        N/A       "
            dest_str = f"Port: {serial_port}" if serial_port else ("IP: " + ip if ip else "No Link")
            print(f"[Tracker] '{target_name}' [{direction_label:<18s}] Error: {err_str} | Servo: {new_int_angle:3d} deg | [{hw_str}]")
            last_print_time = now

        # Center crosshairs
        cv2.line(frame, (int(center_x), 0), (int(center_x), H), (0, 180, 255), 1)
        cv2.line(frame, (0, int(center_y)), (W, int(center_y)), (60, 60, 60), 1)

        # On-screen HUD
        hw_str = "MOUNTED CAM" if (hardware_mode == 1) else "FIXED POINTER"
        mode_str = "1-to-1 DIRECT" if (tracking_mode == 1) else "SMOOTH CINEMATIC"
        inv_str = "INVERTED" if invert_direction else "NORMAL"

        cv2.putText(frame, f"Target: {target_name} | [{direction_label}]", (15, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
        
        hud_line2 = f"Servo: {new_int_angle} deg | Offset: {angle_error_deg:+.1f} deg | Mode: {mode_str} [1/2]"
        cv2.putText(frame, hud_line2, (15, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 255, 255), 2)

        hud_line3 = f"Rig: [{hw_str} - Key 'M'] | Dir: {inv_str} ['I'] | Gain: {gain:.1f}x [+/-]"
        cv2.putText(frame, hud_line3, (15, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (200, 200, 200), 1)

        if not tracking_enabled:
            cv2.putText(frame, "[PAUSED - Press SPACE to Resume]", (15, 110),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        # Angle Gauge Bar
        gauge_x, gauge_y, gauge_w, gauge_h = 15, H - 25, 240, 14
        cv2.rectangle(frame, (gauge_x, gauge_y), (gauge_x + gauge_w, gauge_y + gauge_h), (80, 80, 80), 1)
        fill_w = int(((new_int_angle - 15.0) / 150.0) * gauge_w)
        cv2.rectangle(frame, (gauge_x, gauge_y), (gauge_x + fill_w, gauge_y + gauge_h), (0, 230, 255), -1)
        cv2.putText(frame, "15 deg", (gauge_x, gauge_y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (180, 180, 180), 1)
        cv2.putText(frame, "165 deg", (gauge_x + gauge_w - 40, gauge_y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (180, 180, 180), 1)

        cv2.imshow("1-to-1 High-Accuracy Face Tracker", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("m"):
            hardware_mode = 2 if (hardware_mode == 1) else 1
            mode_name = "MOUNTED CAM (Pan-Base Recentering)" if (hardware_mode == 1) else "FIXED CAM (Direct Pointer to Face)"
            print(f"\n[Hardware Mode] Switched to: {mode_name}\n")
        elif key == ord("i"):
            invert_direction = not invert_direction
            print(f"\n[Config] Pan Direction: {'INVERTED' if invert_direction else 'NORMAL'}\n")
        elif key == ord("r"):
            servo_angle = 90.0
            filtered_angle = 90.0
            if ser:
                ser.write(b"90\n")
                ser.flush()
            if sock and ip:
                try:
                    sock.sendto(b"90\n", (ip, udp_port))
                except Exception:
                    pass
            print("\n[Config] Servo reset to center (90 deg)\n")
        elif key == ord("1"):
            tracking_mode = 1
            print("\n[Mode] Switched to: INSTANT 1-TO-1 DIRECT TRACKING\n")
        elif key == ord("2"):
            tracking_mode = 2
            print("\n[Mode] Switched to: SMOOTH CINEMATIC MODE\n")
        elif key in (ord("+"), ord("=")):
            gain = min(gain + 0.15, 3.0)
            print(f"[Config] Gain/Sensitivity: {gain:.2f}x")
        elif key in (ord("-"), ord("_")):
            gain = max(gain - 0.15, 0.3)
            print(f"[Config] Gain/Sensitivity: {gain:.2f}x")
        elif key == ord(" "):
            tracking_enabled = not tracking_enabled
            print(f"[Config] Tracking: {'RESUMED' if tracking_enabled else 'PAUSED'}")

    # Return to center on exit
    if ser:
        ser.write(b"90\n")
        ser.flush()
        ser.close()
    if sock and ip:
        try:
            sock.sendto(b"90\n", (ip, udp_port))
            sock.close()
        except Exception:
            pass

    cap.release()
    cv2.destroyAllWindows()
    print("\nTracker exited cleanly. Servo at center (90 deg).")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="High-Accuracy 1-to-1 Horizontal Face Tracker")
    parser.add_argument("--target", required=False, default=None, help="Enrolled target person name")
    parser.add_argument("--cam", default="auto", help="Camera index, 'auto' for physical external camera, or stream URL")
    parser.add_argument("--port", type=str, default=None, help="Serial port (e.g. COM13)")
    parser.add_argument("--ip", type=str, default=None, help="ESP8266 WiFi IP address (e.g. 192.168.4.1 or 192.168.1.100)")
    parser.add_argument("--udp-port", type=int, default=8888, help="ESP8266 UDP port (default: 8888)")
    parser.add_argument("--invert", action="store_true", help="Invert servo panning direction")
    parser.add_argument("--fov", type=float, default=DEFAULT_CAMERA_FOV, help="Camera horizontal FOV in degrees (default: 65.0)")
    parser.add_argument("--list-cams", action="store_true", help="List detected cameras and exit")
    args = parser.parse_args()

    if args.list_cams:
        print_camera_list()
    elif not args.target:
        parser.error("the following arguments are required: --target (unless --list-cams is specified)")
    else:
        main(args.target, args.cam, serial_port=args.port, ip=args.ip, udp_port=args.udp_port, invert=args.invert, fov=args.fov)