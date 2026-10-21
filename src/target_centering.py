# src/target_centering.py
"""
Target Centering Logic & Virtual PTZ (Pan-Tilt-Zoom) Gimbal Simulator.
Simulates a physical closed-loop 2-axis camera gimbal without requiring physical hardware components.
Provides normalized error metrics, PID/proportional control, deadband stabilization,
and dynamic virtual PTZ viewport cropping.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, Optional
import cv2
import numpy as np


@dataclass
class CenteringSignal:
    """Telemetry signal describing target offset relative to optical center."""
    raw_dx: float           # Pixels offset (target_x - center_x)
    raw_dy: float           # Pixels offset (target_y - center_y)
    norm_error_x: float     # Normalized error in [-1.0, 1.0]
    norm_error_y: float     # Normalized error in [-1.0, 1.0]
    in_deadband: bool       # True if target is within deadband threshold
    pan_angle_deg: float    # Virtual pan angle in degrees [-60, +60]
    tilt_angle_deg: float   # Virtual tilt angle in degrees [-30, +30]
    pan_velocity: float     # Instantaneous pan speed (deg/s)
    tilt_velocity: float    # Instantaneous tilt speed (deg/s)


class VirtualPTZController:
    """
    Virtual Closed-Loop Gimbal / Camera Centering Controller.
    Computes camera centering corrections and maintains a smooth virtual PTZ viewport.
    """

    def __init__(
        self,
        fov_horizontal_deg: float = 65.0,
        fov_vertical_deg: float = 45.0,
        deadband_ratio: float = 0.06,  # 6% deadband to eliminate jitter
        kp_pan: float = 45.0,          # Proportional gain (deg/s per unit error)
        kd_pan: float = 12.0,          # Derivative gain for damping
        kp_tilt: float = 35.0,
        kd_tilt: float = 8.0,
        max_pan_speed: float = 60.0,   # Max slew rate (deg/s)
        max_tilt_speed: float = 40.0,
        pan_limits: Tuple[float, float] = (-60.0, 60.0),
        tilt_limits: Tuple[float, float] = (-30.0, 30.0),
        virtual_zoom: float = 1.40,    # Zoom factor for virtual PTZ viewport
    ):
        self.fov_h = fov_horizontal_deg
        self.fov_v = fov_vertical_deg
        self.deadband = deadband_ratio
        self.kp_pan = kp_pan
        self.kd_pan = kd_pan
        self.kp_tilt = kp_tilt
        self.kd_tilt = kd_tilt
        self.max_pan_speed = max_pan_speed
        self.max_tilt_speed = max_tilt_speed
        self.pan_limits = pan_limits
        self.tilt_limits = tilt_limits
        self.virtual_zoom = max(1.0, virtual_zoom)

        # Internal state
        self.pan_angle = 0.0   # Current virtual pan angle (degrees)
        self.tilt_angle = 0.0  # Current virtual tilt angle (degrees)
        self.prev_error_x = 0.0
        self.prev_error_y = 0.0
        self.last_update_time: Optional[float] = None
        self.pan_velocity = 0.0
        self.tilt_velocity = 0.0

    def reset(self):
        """Reset gimbal to home center position (0 deg pan, 0 deg tilt)."""
        self.pan_angle = 0.0
        self.tilt_angle = 0.0
        self.prev_error_x = 0.0
        self.prev_error_y = 0.0
        self.last_update_time = None
        self.pan_velocity = 0.0
        self.tilt_velocity = 0.0

    def update(
        self,
        target_pt: Optional[Tuple[float, float]],
        frame_shape: Tuple[int, int],
        current_time: float,
    ) -> CenteringSignal:
        """
        Updates controller with the target's current (x, y) point in the raw camera frame.
        If target_pt is None (target temporarily unobserved), velocity decays smoothly.
        """
        h, w = frame_shape[:2]
        center_x = w / 2.0
        center_y = h / 2.0

        dt = 0.033  # Default nominal dt ~30 FPS
        if self.last_update_time is not None:
            measured_dt = current_time - self.last_update_time
            if 0.001 < measured_dt < 0.5:
                dt = measured_dt
        self.last_update_time = current_time

        if target_pt is None:
            # Dampen velocities toward zero
            self.pan_velocity *= 0.85
            self.tilt_velocity *= 0.85
            self.pan_angle = np.clip(self.pan_angle + self.pan_velocity * dt, *self.pan_limits)
            self.tilt_angle = np.clip(self.tilt_angle + self.tilt_velocity * dt, *self.tilt_limits)
            return CenteringSignal(
                raw_dx=0.0,
                raw_dy=0.0,
                norm_error_x=0.0,
                norm_error_y=0.0,
                in_deadband=True,
                pan_angle_deg=float(self.pan_angle),
                tilt_angle_deg=float(self.tilt_angle),
                pan_velocity=float(self.pan_velocity),
                tilt_velocity=float(self.tilt_velocity),
            )

        tx, ty = target_pt
        raw_dx = tx - center_x
        raw_dy = ty - center_y

        norm_ex = raw_dx / center_x  # [-1.0, 1.0]
        norm_ey = raw_dy / center_y  # [-1.0, 1.0]

        # Check deadband
        rad_dist = float(np.hypot(norm_ex, norm_ey))
        in_deadband = bool(rad_dist < self.deadband)

        if in_deadband:
            eff_ex = 0.0
            eff_ey = 0.0
        else:
            # Continuous error with deadband subtraction to prevent step discontinuity
            scale = (rad_dist - self.deadband) / rad_dist
            eff_ex = norm_ex * scale
            eff_ey = norm_ey * scale

        # Derivative term
        d_ex = (eff_ex - self.prev_error_x) / dt
        d_ey = (eff_ey - self.prev_error_y) / dt
        self.prev_error_x = eff_ex
        self.prev_error_y = eff_ey

        # Control law: pan follows horizontal offset, tilt follows vertical offset
        cmd_pan_vel = self.kp_pan * eff_ex + self.kd_pan * d_ex
        cmd_tilt_vel = self.kp_tilt * eff_ey + self.kd_tilt * d_ey

        # Slew-rate clamping
        self.pan_velocity = np.clip(cmd_pan_vel, -self.max_pan_speed, self.max_pan_speed)
        self.tilt_velocity = np.clip(cmd_tilt_vel, -self.max_tilt_speed, self.max_tilt_speed)

        # Integrate angles
        self.pan_angle = np.clip(self.pan_angle + self.pan_velocity * dt, *self.pan_limits)
        self.tilt_angle = np.clip(self.tilt_angle + self.tilt_velocity * dt, *self.tilt_limits)

        return CenteringSignal(
            raw_dx=float(raw_dx),
            raw_dy=float(raw_dy),
            norm_error_x=float(norm_ex),
            norm_error_y=float(norm_ey),
            in_deadband=in_deadband,
            pan_angle_deg=float(self.pan_angle),
            tilt_angle_deg=float(self.tilt_angle),
            pan_velocity=float(self.pan_velocity),
            tilt_velocity=float(self.tilt_velocity),
        )

    def render_virtual_ptz_viewport(
        self,
        frame: np.ndarray,
        out_size: Tuple[int, int] = (480, 360),
    ) -> np.ndarray:
        """
        Renders the simulated physical pan-tilt camera's cropped, zoomed viewport.
        As the virtual gimbal pans and tilts, the viewport window smoothly tracks the target.
        """
        h, w = frame.shape[:2]
        crop_w = int(w / self.virtual_zoom)
        crop_h = int(h / self.virtual_zoom)

        # Map pan/tilt angles to pixel offsets in wide frame
        # pan_angle in [-60, +60] maps to available horizontal panning range
        max_shift_x = (w - crop_w) / 2.0
        max_shift_y = (h - crop_h) / 2.0

        norm_pan = self.pan_angle / max(1e-5, self.pan_limits[1])
        norm_tilt = self.tilt_angle / max(1e-5, self.tilt_limits[1])

        shift_x = norm_pan * max_shift_x
        shift_y = norm_tilt * max_shift_y

        center_x = (w / 2.0) + shift_x
        center_y = (h / 2.0) + shift_y

        x1 = int(np.clip(center_x - crop_w / 2.0, 0, w - crop_w))
        y1 = int(np.clip(center_y - crop_h / 2.0, 0, h - crop_h))
        x2 = x1 + crop_w
        y2 = y1 + crop_h

        cropped = frame[y1:y2, x1:x2]
        if cropped.size == 0:
            return np.zeros((out_size[1], out_size[0], 3), dtype=np.uint8)

        resized = cv2.resize(cropped, out_size, interpolation=cv2.INTER_LINEAR)

        # Draw subtle virtual PTZ crosshairs and border
        ch, cw = out_size[1], out_size[0]
        cx, cy = cw // 2, ch // 2
        cv2.line(resized, (cx - 15, cy), (cx + 15, cy), (0, 255, 255), 1)
        cv2.line(resized, (cx, cy - 15), (cx, cy + 15), (0, 255, 255), 1)
        cv2.circle(resized, (cx, cy), 18, (0, 255, 255), 1)
        cv2.rectangle(resized, (0, 0), (cw - 1, ch - 1), (0, 200, 255), 2)

        # Telemetry banner on virtual viewport
        cv2.putText(
            resized,
            f"VIRTUAL PTZ [Pan: {self.pan_angle:+.1f} deg | Tilt: {self.tilt_angle:+.1f} deg]",
            (10, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 255, 255),
            1,
            cv2.LINE_AA,
        )

        return resized
