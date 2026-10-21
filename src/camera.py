# src/camera.py
"""
Camera utility module with automatic external physical camera selection.
Ensures external physical USB cameras are prioritized over PC built-in webcams and virtual cameras.
"""
from __future__ import annotations

import argparse
import sys
import time
from typing import Optional, Union, List, Dict, Tuple
import cv2

# Keywords identifying PC internal/integrated webcams
INTERNAL_CAMERA_KEYWORDS = [
    "integrated",
    "internal",
    "built-in",
    "builtin",
    "facetime",
    "front camera",
    "rear camera",
    "laptop camera",
]

# Keywords identifying virtual/software camera devices
VIRTUAL_CAMERA_KEYWORDS = [
    "virtual",
    "obs",
    "eshare",
    "screen",
    "droidcam",
    "iriun",
    "manycam",
    "vysor",
    "snap camera",
    "unity",
    "fake",
]


def list_available_cameras(max_probe: int = 6) -> List[Dict[str, Union[int, str, bool]]]:
    """
    Enumerate connected cameras on the system.
    Uses DirectShow (pygrabber) on Windows if available, with OpenCV probing fallback.
    """
    cameras = []
    device_names = []

    try:
        from pygrabber.dshow_graph import FilterGraph
        graph = FilterGraph()
        device_names = graph.get_input_devices()
    except Exception:
        device_names = []

    if device_names:
        for idx, name in enumerate(device_names):
            name_lower = name.lower()
            is_internal = any(k in name_lower for k in INTERNAL_CAMERA_KEYWORDS)
            is_virtual = any(k in name_lower for k in VIRTUAL_CAMERA_KEYWORDS)
            is_external_physical = (not is_internal) and (not is_virtual)

            cam_type = "Physical External" if is_external_physical else ("Internal PC Camera" if is_internal else "Virtual Camera")

            cameras.append({
                "index": idx,
                "name": name,
                "type": cam_type,
                "is_external_physical": is_external_physical,
                "is_internal": is_internal,
                "is_virtual": is_virtual,
            })
    else:
        # Fallback: probe OpenCV indices directly
        for idx in range(max_probe):
            backend = cv2.CAP_DSHOW if sys.platform.startswith("win") else cv2.CAP_ANY
            cap = cv2.VideoCapture(idx, backend)
            if cap.isOpened():
                ret, _ = cap.read()
                if ret:
                    # On basic probe, index 0 is typically built-in, index > 0 is typically external
                    is_external = (idx > 0)
                    cameras.append({
                        "index": idx,
                        "name": f"Camera {idx}" + (" (External/Secondary)" if is_external else " (Default/Internal)"),
                        "type": "Physical External" if is_external else "Default / Internal",
                        "is_external_physical": is_external,
                        "is_internal": (idx == 0),
                        "is_virtual": False,
                    })
                cap.release()

    return cameras


def get_physical_camera_index(preferred_name: Optional[str] = None) -> Tuple[int, str]:
    """
    Automatically select the physical external camera.
    Skips internal PC cameras and virtual camera devices.
    Returns (camera_index, camera_name).
    """
    cams = list_available_cameras()

    if not cams:
        return 0, "Default Camera (0)"

    # If preferred name given, search for it
    if preferred_name:
        for c in cams:
            if preferred_name.lower() in str(c["name"]).lower():
                return int(c["index"]), str(c["name"])

    # 1. Look for physical external cameras
    external_cams = [c for c in cams if c["is_external_physical"]]
    if external_cams:
        selected = external_cams[0]
        return int(selected["index"]), str(selected["name"])

    # 2. Look for non-internal cameras
    non_internal = [c for c in cams if not c["is_internal"]]
    if non_internal:
        selected = non_internal[0]
        return int(selected["index"]), str(selected["name"])

    # 3. Fallback to first available
    return int(cams[0]["index"]), str(cams[0]["name"])


def open_camera(
    cam_source: Union[int, str, None] = "auto",
    width: Optional[int] = None,
    height: Optional[int] = None,
    fps: Optional[int] = None,
) -> cv2.VideoCapture:
    """
    Open a camera video capture stream.
    
    If cam_source is "auto", None, or "external", automatically selects
    the physical external camera rather than the PC's built-in webcam.
    """
    cam_name = "Camera"
    target_source: Union[int, str] = 0

    s_lower = str(cam_source).lower() if cam_source is not None else "auto"

    if s_lower in ("internal", "builtin", "laptop", "pc", "webcam", "integrated", "0"):
        # Explicit internal laptop webcam requested
        cams = list_available_cameras()
        internal_cams = [c for c in cams if c.get("is_internal")]
        if internal_cams:
            target_source = int(internal_cams[0]["index"])
            cam_name = str(internal_cams[0]["name"])
        else:
            target_source = 0
            cam_name = "Internal PC Webcam (0)"
    elif s_lower in ("auto", "external", "physical", "usb", "ext"):
        idx, name = get_physical_camera_index()
        target_source = idx
        cam_name = name
    else:
        # Check if integer or numeric string
        try:
            target_source = int(cam_source)
            for c in list_available_cameras():
                if c["index"] == target_source:
                    cam_name = str(c["name"])
                    break
            else:
                cam_name = f"Camera index {target_source}"
        except ValueError:
            target_source = str(cam_source)
            cam_name = f"Stream: {target_source}"

    print(f"[Camera] Connecting to: '{cam_name}' (Source: {target_source})")

    # Use DirectShow on Windows for numeric camera indices for direct device mapping
    backend = cv2.CAP_DSHOW if (isinstance(target_source, int) and sys.platform.startswith("win")) else cv2.CAP_ANY
    cap = cv2.VideoCapture(target_source, backend)

    if not cap.isOpened():
        # Retry with default backend if DSHOW failed
        if backend != cv2.CAP_ANY:
            cap = cv2.VideoCapture(target_source)

    if not cap.isOpened():
        raise RuntimeError(
            f"Failed to open camera source {target_source} ('{cam_name}'). "
            f"Run with '--list-cams' to inspect connected devices."
        )

    if width:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    if height:
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    if fps:
        cap.set(cv2.CAP_PROP_FPS, fps)

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[Camera] Active: '{cam_name}' [{actual_w}x{actual_h}]")

    return cap


def print_camera_list():
    """Print a clean CLI table of all detected camera devices."""
    print("\n" + "=" * 65)
    print(f"{'INDEX':<7} | {'TYPE':<20} | {'DEVICE NAME'}")
    print("=" * 65)
    cams = list_available_cameras()
    if not cams:
        print("  No cameras detected.")
    for c in cams:
        marker = " -> [SELECTED PHYSICAL]" if c["is_external_physical"] else ""
        print(f"  {c['index']:<5} | {c['type']:<20} | {c['name']}{marker}")
    print("=" * 65 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Camera test tool and physical camera inspector")
    parser.add_argument("--cam", default="auto", help="Camera index, 'auto' for physical external camera, or stream URL")
    parser.add_argument("--list-cams", action="store_true", help="List all detected camera devices and exit")
    parser.add_argument("--width", type=int, default=None, help="Desired frame width")
    parser.add_argument("--height", type=int, default=None, help="Desired frame height")
    args = parser.parse_args()

    if args.list_cams:
        print_camera_list()
        return

    cap = open_camera(args.cam, width=args.width, height=args.height)

    print("Camera feed running. Press 'Q' to exit.")
    prev_time = time.time()

    while True:
        success, frame = cap.read()
        if not success or frame is None:
            print("ERROR: Could not read frame from camera.")
            break

        current_time = time.time()
        fps = 1.0 / max(current_time - prev_time, 1e-6)
        prev_time = current_time

        cv2.putText(
            frame,
            f"FPS: {fps:.1f}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
        )

        cv2.imshow("Camera View (Physical Camera)", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Camera stopped.")


if __name__ == "__main__":
    main()