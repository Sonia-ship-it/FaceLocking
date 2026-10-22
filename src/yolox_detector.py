# src/yolox_detector.py
"""
YOLOX Deep Learning Person / Body Detector via ONNX Runtime / OpenCV DNN.
Detects full body and torso bounding boxes on CPU with high inference speed.
Allows continuous person tracking even when face is obscured or turned away.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import List, Tuple, Optional
import cv2
import numpy as np

DEFAULT_YOLOX_PATH = Path("models/yolox_nano.onnx")


class YOLOXPersonDetector:
    """
    YOLOX-Nano ONNX detector for whole-person detection.
    Pre-processes frame to 416x416 with letterboxing and filters COCO Class 0 (Person).
    """

    def __init__(
        self,
        model_path: str | Path = DEFAULT_YOLOX_PATH,
        conf_thresh: float = 0.35,
        nms_thresh: float = 0.45,
        input_size: Tuple[int, int] = (416, 416),
    ):
        self.model_path = Path(model_path)
        self.conf_thresh = conf_thresh
        self.nms_thresh = nms_thresh
        self.input_size = input_size
        self.session = None

        if self.model_path.exists() and self.model_path.stat().st_size > 1000 * 1024:
            try:
                import onnxruntime as ort
                opts = ort.SessionOptions()
                opts.intra_op_num_threads = 2
                self.session = ort.InferenceSession(str(self.model_path), opts, providers=["CPUExecutionProvider"])
                self.input_name = self.session.get_inputs()[0].name
                print(f"[YOLOX] Successfully loaded YOLOX-Nano ONNX model from: {self.model_path}")
            except Exception as e:
                print(f"[YOLOX] Could not load ONNX session: {e}")
                self.session = None
        else:
            self.session = None

    def is_available(self) -> bool:
        return self.session is not None

    def _preprocess(self, img: np.ndarray) -> Tuple[np.ndarray, float, Tuple[int, int]]:
        """Letterbox resize to 416x416 while preserving aspect ratio."""
        target_w, target_h = self.input_size
        h, w = img.shape[:2]

        r = min(target_w / w, target_h / h)
        new_w, new_h = int(w * r), int(h * r)

        resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        padded = np.full((target_h, target_w, 3), 114, dtype=np.uint8)
        dw = (target_w - new_w) // 2
        dh = (target_h - new_h) // 2
        padded[dh:dh + new_h, dw:dw + new_w] = resized

        # Format to NCHW float32
        blob = padded.transpose((2, 0, 1)).astype(np.float32)
        blob = np.ascontiguousarray(blob[None, ...])
        return blob, r, (dw, dh)

    def detect(self, frame: np.ndarray) -> List[Tuple[float, float, float, float, float]]:
        """
        Detects person bounding boxes [x1, y1, x2, y2, confidence].
        Only retains COCO class 0 ('person').
        """
        if self.session is None:
            return []

        h, w = frame.shape[:2]
        blob, r, (dw, dh) = self._preprocess(frame)

        outputs = self.session.run(None, {self.input_name: blob})
        predictions = outputs[0][0]  # Shape: (N, 85)

        boxes = []
        scores = []

        # YOLOX output: [cx, cy, w, h, obj_conf, class_0_conf, class_1_conf, ...]
        for row in predictions:
            obj_conf = row[4]
            if obj_conf < self.conf_thresh:
                continue

            person_conf = row[5]  # Class 0: person
            final_conf = float(obj_conf * person_conf)

            if final_conf >= self.conf_thresh:
                cx, cy, bw, bh = row[0:4]

                # Remove letterbox offset and rescale
                rx1 = (cx - bw / 2.0 - dw) / r
                ry1 = (cy - bh / 2.0 - dh) / r
                rx2 = (cx + bw / 2.0 - dw) / r
                ry2 = (cy + bh / 2.0 - dh) / r

                rx1 = max(0.0, min(float(w), rx1))
                ry1 = max(0.0, min(float(h), ry1))
                rx2 = max(0.0, min(float(w), rx2))
                ry2 = max(0.0, min(float(h), ry2))

                boxes.append([int(rx1), int(ry1), int(rx2 - rx1), int(ry2 - ry1)])
                scores.append(final_conf)

        if not boxes:
            return []

        indices = cv2.dnn.NMSBoxes(boxes, scores, self.conf_thresh, self.nms_thresh)
        results = []
        if len(indices) > 0:
            for idx in indices.flatten():
                bx, by, bw, bh = boxes[idx]
                results.append((float(bx), float(by), float(bx + bw), float(by + bh), float(scores[idx])))

        return results
