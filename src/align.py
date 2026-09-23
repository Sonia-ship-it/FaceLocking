# src/align.py
"""
5-Point Facial Landmark Alignment for ArcFace ONNX.
Computes similarity transform to warp detected 5 keypoints
(left eye, right eye, nose tip, left mouth corner, right mouth corner)
onto canonical ArcFace 112x112 coordinates.
"""
from __future__ import annotations

from typing import Tuple, Optional
import cv2
import numpy as np

# Canonical ArcFace 112x112 reference landmark positions
ARCFACE_REF_5PTS = np.array(
    [
        [38.2946, 51.6963],  # Left Eye
        [73.5318, 51.5014],  # Right Eye
        [56.0252, 71.7366],  # Nose Tip
        [41.5493, 92.3655],  # Left Mouth Corner
        [70.7266, 92.2041],  # Right Mouth Corner
    ],
    dtype=np.float32,
)


def align_face_5pt(
    img: np.ndarray,
    kps: np.ndarray,
    out_size: Tuple[int, int] = (112, 112),
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Computes similarity transform (rotation + scale + translation)
    to warp detected 5 keypoints onto canonical ArcFace coordinates.
    """
    if kps is None or len(kps) != 5:
        return None, None

    src_pts = np.asarray(kps, dtype=np.float32)
    dst_pts = ARCFACE_REF_5PTS.copy()

    if out_size != (112, 112):
        scale_x = out_size[0] / 112.0
        scale_y = out_size[1] / 112.0
        dst_pts[:, 0] *= scale_x
        dst_pts[:, 1] *= scale_y

    M, _ = cv2.estimateAffinePartial2D(src_pts, dst_pts, method=cv2.LMEDS)
    if M is None:
        return None, None

    warped = cv2.warpAffine(
        img,
        M,
        (int(out_size[0]), int(out_size[1])),
        borderValue=0.0,
        flags=cv2.INTER_LINEAR,
    )
    return warped, M
