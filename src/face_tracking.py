# src/face_tracking.py
"""
Face Tracking with Identity Lock, Smile, Blink, and Position Detection (Part 2).
Locks onto a single enrolled identity, tracks continuity between ArcFace checks,
detects Eye Aspect Ratio (EAR), blink count, sustained eye closure, smile hysteresis,
and normalized horizontal/vertical position errors.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from enum import Enum, auto
from pathlib import Path
from typing import Optional, Union, Tuple, List
import cv2
import numpy as np

from src.align import align_face_5pt
from src.face_signals import FaceSignalExtractor
from src.recognize import (
    ArcFaceEmbedderONNX,
    FaceDBMatcher,
    HaarFaceMesh5pt,
    load_db_npz,
)

# Optional helper to open physical external camera if available
try:
    from src.camera import open_camera
except ImportError:
    open_camera = None


class LockState(Enum):
    SEARCHING = auto()
    LOCKED = auto()
    LOST = auto()


def iou(a, b) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    area_a = max(1, (ax2 - ax1) * (ay2 - ay1))
    area_b = max(1, (bx2 - bx1) * (by2 - by1))
    return inter / float(area_a + area_b - inter)


def center(box) -> np.ndarray:
    x1, y1, x2, y2 = box
    return np.array([(x1 + x2) / 2.0, (y1 + y2) / 2.0], dtype=np.float32)


@dataclass
class TrackingSignal:
    error_x: float
    error_y: float
    horizontal: str
    vertical: str


class LockedFaceTracker:
    def __init__(
        self,
        target_name: str,
        detector,
        embedder,
        matcher,
        verify_every: int = 10,
        lost_timeout: int = 24,
        ema_alpha: float = 0.30,
        dead_zone: float = 0.07,
    ):
        self.target_name = target_name
        self.detector = detector
        self.embedder = embedder
        self.matcher = matcher
        self.verify_every = verify_every
        self.lost_timeout = lost_timeout
        self.ema_alpha = ema_alpha
        self.dead_zone = dead_zone
        self.state = LockState.SEARCHING
        self.last_box = None
        self.smooth_center = None
        self.lost_frames = 0
        self.frame_index = 0

    @staticmethod
    def box(face) -> Tuple[int, int, int, int]:
        return (face.x1, face.y1, face.x2, face.y2)

    def identity(self, frame, face):
        aligned, _ = align_face_5pt(frame, face.kps, out_size=(112, 112))
        if aligned is None or not aligned.size:
            from src.recognize import MatchResult
            return MatchResult(name="Unknown", distance=1.0, similarity=0.0, accepted=False)
        return self.matcher.match(self.embedder.embed(aligned))

    def target_is_verified(self, frame, face) -> bool:
        match = self.identity(frame, face)
        if self.target_name.lower() in ("any", "auto", "*"):
            return match.accepted
        return match.accepted and match.name.lower() == self.target_name.lower()

    def acquire(self, frame, faces):
        best = None
        best_similarity = -1.0
        for face in faces:
            match = self.identity(frame, face)
            is_target = match.accepted and (
                self.target_name.lower() in ("any", "auto", "*")
                or match.name.lower() == self.target_name.lower()
            )
            if is_target and match.similarity > best_similarity:
                best, best_similarity = face, match.similarity
                if self.target_name.lower() in ("any", "auto", "*"):
                    self.target_name = match.name
        return best

    def associate(self, faces):
        if self.last_box is None or not faces:
            return None
        last_center = center(self.last_box)
        last_diag = max(
            float(
                np.linalg.norm(
                    np.array(
                        [
                            self.last_box[2] - self.last_box[0],
                            self.last_box[3] - self.last_box[1],
                        ],
                        dtype=np.float32,
                    )
                )
            ),
            1.0,
        )
        ranked = []
        for face in faces:
            b = self.box(face)
            overlap = iou(self.last_box, b)
            displacement = float(np.linalg.norm(center(b) - last_center) / last_diag)
            score = overlap - 0.35 * displacement
            ranked.append((score, face))

        score, candidate = max(ranked, key=lambda item: item[0])
        return candidate if score > -0.30 else None

    def update(self, frame):
        self.frame_index += 1
        faces = self.detector.detect(frame, max_faces=8)

        if self.state == LockState.SEARCHING:
            candidate = self.acquire(frame, faces)
        else:
            candidate = self.associate(faces)

        if (
            candidate is not None
            and (
                self.state == LockState.LOST
                or self.frame_index % self.verify_every == 0
            )
            and not self.target_is_verified(frame, candidate)
        ):
            candidate = None

        if candidate is None:
            self.lost_frames += 1
            if self.last_box is not None:
                self.state = LockState.LOST
            if self.lost_frames > self.lost_timeout:
                self.state = LockState.SEARCHING
                self.last_box = None
                self.smooth_center = None
            return None, None, faces

        self.state = LockState.LOCKED
        self.lost_frames = 0
        self.last_box = self.box(candidate)
        raw_center = center(self.last_box)

        if self.smooth_center is None:
            self.smooth_center = raw_center
        else:
            a = self.ema_alpha
            self.smooth_center = a * raw_center + (1.0 - a) * self.smooth_center

        return candidate, self.position_signal(frame.shape), faces

    def position_signal(self, shape) -> TrackingSignal:
        height, width = shape[:2]
        ex = float((self.smooth_center[0] - width / 2.0) / (width / 2.0))
        ey = float((self.smooth_center[1] - height / 2.0) / (height / 2.0))
        horizontal = "CENTER"
        vertical = "CENTER"
        if ex < -self.dead_zone:
            horizontal = "LEFT"
        elif ex > self.dead_zone:
            horizontal = "RIGHT"
        if ey < -self.dead_zone:
            vertical = "UP"
        elif ey > self.dead_zone:
            vertical = "DOWN"
        return TrackingSignal(ex, ey, horizontal, vertical)


def draw_label(frame, text, xy, color, scale=0.62, thickness=2):
    cv2.putText(
        frame,
        text,
        xy,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        (0, 0, 0),
        thickness + 2,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        text,
        xy,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        thickness,
        cv2.LINE_AA,
    )


def open_capture_device(cam_arg: Union[int, str]):
    if open_camera is not None and str(cam_arg).lower() in ("auto", "external"):
        return open_camera("auto")
    try:
        idx = int(cam_arg)
        # Try DirectShow first on Windows
        cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
        if not cap.isOpened():
            cap = cv2.VideoCapture(idx)
        return cap
    except ValueError:
        return cv2.VideoCapture(cam_arg)


def main():
    parser = argparse.ArgumentParser(
        description="Face Tracking with Identity Lock, Eye Open/Closed, Blink, and Position Detection (Part 2)"
    )
    parser.add_argument(
        "--target",
        default="darius",
        help="Enrolled identity to lock (e.g. 'darius', 'Sonia', or 'any'). Default: 'darius'",
    )
    parser.add_argument(
        "--camera",
        default="auto",
        help="Camera index (e.g. 0, 1), 'auto' for physical external camera, or stream URL",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.34,
        help="ArcFace cosine distance threshold for identity acceptance (default: 0.34)",
    )
    parser.add_argument(
        "--smile-on",
        type=float,
        default=0.43,
        help="Threshold ratio to trigger SMILE state (default: 0.43)",
    )
    parser.add_argument(
        "--smile-off",
        type=float,
        default=0.39,
        help="Threshold ratio to return to NEUTRAL state (default: 0.39)",
    )
    parser.add_argument(
        "--ear-threshold",
        type=float,
        default=0.21,
        help="Eye Aspect Ratio threshold for eye closure (default: 0.21)",
    )
    args = parser.parse_args()

    db_path = Path("data/db/face_db.npz")
    db = load_db_npz(db_path)

    enrolled_names = list(db.keys())
    print("\n" + "=" * 65)
    print("   PART 2: FACE TRACKING WITH IDENTITY LOCK & EYE SIGNALS")
    print("=" * 65)
    print(f"Target identity to lock: '{args.target}'")
    print(f"Enrolled identities in database: {enrolled_names if enrolled_names else '[None]'}")
    print("Controls:")
    print("  'C' : Auto-calibrate smile & eye thresholds using your resting face")
    print("  'R' : Reset blink count to 0")
    print("  'Q' : Quit")
    print("-----------------------------------------------------------------")

    if args.target.lower() not in [n.lower() for n in enrolled_names] and args.target.lower() not in ("any", "auto"):
        print(f"\n[Notice] Target '{args.target}' is not yet enrolled in {db_path}.")
        if enrolled_names:
            print(f"   Available enrolled identity: {enrolled_names}")
            print(f"   To track an enrolled person, run with: --target \"{enrolled_names[0]}\"")
        print(f"   To enroll '{args.target}', run: python -m src.enroll --name {args.target}")
        print(f"   The system will search for '{args.target}'.\n")

    detector = HaarFaceMesh5pt(min_size=(70, 70), debug=False)
    embedder = ArcFaceEmbedderONNX(
        model_path="models/embedder_arcface.onnx",
        input_size=(112, 112),
        debug=False,
    )
    matcher = FaceDBMatcher(db, dist_thresh=args.threshold)
    tracker = LockedFaceTracker(
        target_name=args.target,
        detector=detector,
        embedder=embedder,
        matcher=matcher,
    )
    signals = FaceSignalExtractor(
        ear_threshold=args.ear_threshold,
        smile_on=args.smile_on,
        smile_off=args.smile_off,
    )

    cap = open_capture_device(args.camera)
    if not cap.isOpened():
        # Fallback to index 0
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open camera '{args.camera}' or default camera 0.")

    import time
    blink_total = 0
    blink_flash_until = 0.0
    calib_text = ""
    calib_until = 0.0
    latest_face_state = None
    print("Camera initialized. Press 'Q' to quit.\n")

    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                break

            locked_face, position, all_faces = tracker.update(frame)
            view = frame.copy()

            # State header
            is_locked = locked_face is not None
            state_text = f"{tracker.state.name}: {tracker.target_name}"
            state_color = (0, 220, 0) if is_locked else (0, 140, 255)
            draw_label(view, state_text, (12, 30), state_color, 0.75, 2)

            # Draw other visible faces (distractors / unverified faces)
            for f in all_faces:
                fb = tracker.box(f)
                if not is_locked or fb != tracker.box(locked_face):
                    f_match = tracker.identity(frame, f)
                    distractor_label = f"{f_match.name}" if f_match.accepted else "Stranger/Other"
                    cv2.rectangle(view, (f.x1, f.y1), (f.x2, f.y2), (80, 80, 80), 1)
                    draw_label(view, distractor_label, (f.x1, max(20, f.y1 - 6)), (140, 140, 140), 0.48, 1)

            latest_face_state = None

            # Locked target processing
            if is_locked and locked_face is not None:
                box = tracker.box(locked_face)
                x1, y1, x2, y2 = box
                cv2.rectangle(view, (x1, y1), (x2, y2), (255, 170, 0), 3)

                face_state = signals.analyze(frame, box)
                latest_face_state = face_state

                if face_state is not None:
                    if face_state.blink:
                        blink_total += 1
                        blink_flash_until = time.time() + 0.45

                    expression = "SMILE" if face_state.smiling else "NEUTRAL"
                    expr_color = (0, 255, 255) if face_state.smiling else (200, 200, 200)
                    eye_text = "EYES CLOSED" if face_state.eyes_closed else "EYES OPEN"
                    eye_color = (0, 0, 255) if face_state.eyes_closed else (0, 255, 0)

                    # Display expression above target box
                    draw_label(view, expression, (x1, max(50, y1 - 48)), expr_color, 0.68, 2)

                    # Display eye state and accumulated blink count
                    draw_label(
                        view,
                        f"{eye_text} | blinks: {blink_total}",
                        (x1, max(75, y1 - 22)),
                        eye_color,
                        0.65,
                    )

                    # Prominent visual blink indicator when blink event triggers
                    if time.time() < blink_flash_until:
                        draw_label(view, "BLINK!", (x1, y2 + 25), (0, 255, 0), 0.85, 2)

                    # Display numeric EAR and smile metrics at bottom left
                    draw_label(
                        view,
                        f"EAR={face_state.ear:.3f} (thresh={signals.ear_threshold:.3f}) | smile={face_state.smile_score:.3f} [on:{signals.smile_on:.2f} off:{signals.smile_off:.2f}] ('C'=calib)",
                        (12, view.shape[0] - 18),
                        (255, 255, 255),
                        0.46,
                    )
                    # Display normalized error and direction labels
                    draw_label(
                        view,
                        f"H={position.horizontal} V={position.vertical}  error=({position.error_x:+.2f}, {position.error_y:+.2f})",
                        (12, 60),
                        (255, 170, 0),
                        0.60,
                    )
                else:
                    signals.reset()
            else:
                signals.reset()

            # Center dead-zone visualization
            h, w = view.shape[:2]
            dz = tracker.dead_zone
            cv2.rectangle(
                view,
                (int(w * (0.5 - dz / 2)), int(h * (0.5 - dz / 2))),
                (int(w * (0.5 + dz / 2)), int(h * (0.5 + dz / 2))),
                (120, 120, 120),
                1,
            )
            cv2.putText(
                view,
                "CENTER DEAD ZONE",
                (int(w * (0.5 - dz / 2)), int(h * (0.5 - dz / 2)) - 6),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.38,
                (120, 120, 120),
                1,
                cv2.LINE_AA,
            )

            # Display calibration notification banner if active
            if time.time() < calib_until and calib_text:
                draw_label(view, calib_text, (12, 90), (0, 255, 0), 0.65, 2)

            cv2.imshow("Locked Face Tracking (Part 2)", view)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("r"):
                blink_total = 0
                print("[Tracker] Reset blink count to 0.")
            elif key == ord("c"):
                if latest_face_state is not None:
                    on, off, ear_th = signals.calibrate(latest_face_state.smile_score, latest_face_state.ear)
                    calib_text = f"CALIBRATED: Neutral={latest_face_state.smile_score:.3f} | Open EAR={latest_face_state.ear:.3f} (thresh={ear_th:.3f})"
                    calib_until = time.time() + 4.0
                    print(f"[Tracker] {calib_text}")
                else:
                    print("[Tracker] Cannot calibrate: target face not detected in frame.")

    finally:
        cap.release()
        signals.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
