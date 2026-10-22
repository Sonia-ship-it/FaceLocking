# src/yunet_detector.py
"""
Modern Deep Learning Face Detector using OpenCV's YuNet (CNN ONNX).
Replaces legacy Haar Cascade with high-accuracy, ultra-fast (<10ms) CPU face detection
and direct 5-point facial landmark extraction (eyes, nose, mouth corners)
for ArcFace alignment and identity recognition.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Optional
import cv2
import numpy as np

from src.haar_5pt import FaceBox5pt, align_face_5pt

DEFAULT_YUNET_PATH = Path("models/face_detection_yunet.onnx")


class YuNetFaceDetector:
    """
    OpenCV FaceDetectorYN Wrapper.
    Runs deep learning face detection on CPU with 5-point landmark regression.
    """

    def __init__(
        self,
        model_path: str | Path = DEFAULT_YUNET_PATH,
        conf_threshold: float = 0.65,
        nms_threshold: float = 0.30,
        top_k: int = 5000,
    ):
        self.model_path = Path(model_path)
        self.conf_threshold = conf_threshold
        self.nms_threshold = nms_threshold
        self.top_k = top_k
        self.detector: Optional[cv2.FaceDetectorYN] = None
        self.current_input_size: Tuple[int, int] = (320, 320)

        if not self.model_path.exists():
            print(f"[YuNet] Warning: Model file {self.model_path} not found. Attempting download...")
            self._download_model()

        if self.model_path.exists():
            try:
                self.detector = cv2.FaceDetectorYN.create(
                    model=str(self.model_path),
                    config="",
                    input_size=self.current_input_size,
                    score_threshold=self.conf_threshold,
                    nms_threshold=self.nms_threshold,
                    top_k=self.top_k,
                )
                print(f"[YuNet] Initialized modern FaceDetectorYN from: {self.model_path}")
            except Exception as e:
                print(f"[YuNet] Initialization failed: {e}")
                self.detector = None
        else:
            print("[YuNet] Could not find or download YuNet ONNX model.")

    def _download_model(self):
        """Auto-download official OpenCV Zoo YuNet ONNX model if missing."""
        import urllib.request
        url = "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=20) as resp, open(self.model_path, "wb") as f:
                f.write(resp.read())
            print(f"[YuNet] Downloaded official YuNet model to: {self.model_path}")
        except Exception as e:
            print(f"[YuNet] Download failed: {e}")

    def detect(self, frame: np.ndarray, max_faces: int = 5) -> List[FaceBox5pt]:
        """
        Detect faces and extract 5 canonical keypoints:
          [left_eye, right_eye, nose_tip, left_mouth, right_mouth]
        """
        if self.detector is None:
            return []

        h, w = frame.shape[:2]
        if (w, h) != self.current_input_size:
            self.detector.setInputSize((w, h))
            self.current_input_size = (w, h)

        _, raw_faces = self.detector.detect(frame)
        if raw_faces is None or len(raw_faces) == 0:
            return []

        results: List[FaceBox5pt] = []

        # Sort by confidence score descending
        sorted_faces = sorted(raw_faces, key=lambda f: f[14], reverse=True)

        for row in sorted_faces[:max_faces]:
            conf = float(row[14])
            if conf < self.conf_threshold:
                continue

            bx = int(row[0])
            by = int(row[1])
            bw = int(row[2])
            bh = int(row[3])

            x1 = max(0, bx)
            y1 = max(0, by)
            x2 = min(w, bx + bw)
            y2 = min(h, by + bh)

            # 5 Landmarks: right eye, left eye, nose, right mouth, left mouth
            # Note: in OpenCV YuNet:
            #   row[4:6]: subject's right eye (viewer's left)
            #   row[6:8]: subject's left eye (viewer's right)
            #   row[8:10]: nose tip
            #   row[10:12]: subject's right mouth corner
            #   row[12:14]: subject's left mouth corner
            eye_r = [float(row[4]), float(row[5])]
            eye_l = [float(row[6]), float(row[7])]
            nose = [float(row[8]), float(row[9])]
            mouth_r = [float(row[10]), float(row[11])]
            mouth_l = [float(row[12]), float(row[13])]

            kps = np.array([eye_r, eye_l, nose, mouth_r, mouth_l], dtype=np.float32)

            # Ensure canonical left-to-right eye order (viewer's perspective)
            if kps[0, 0] > kps[1, 0]:
                kps[[0, 1]] = kps[[1, 0]]
            # Ensure canonical left-to-right mouth order (viewer's perspective)
            if kps[3, 0] > kps[4, 0]:
                kps[[3, 4]] = kps[[4, 3]]

            results.append(FaceBox5pt(x1, y1, x2, y2, kps))

        return results
