# src/appearance_reid.py
"""
Appearance Re-Identification (ReID) and Color Histogram Feature Extractor.
Maintains identity continuity when a person turns their back to the camera or covers their face.
Extracts normalized HSV color distributions and clothing signatures from the upper torso,
enabling continuous tracking across severe head turns, profile walks, and facial absence.
"""
from __future__ import annotations

from typing import Tuple, Optional
import cv2
import numpy as np


class AppearanceReID:
    """
    Torso & Clothing Appearance Feature Extractor.
    Extracts 2D Hue-Saturation color histograms and computes similarity metrics.
    """

    def __init__(
        self,
        h_bins: int = 16,
        s_bins: int = 8,
        ema_alpha: float = 0.20,
    ):
        self.h_bins = h_bins
        self.s_bins = s_bins
        self.ema_alpha = ema_alpha

    def extract_torso_signature(
        self,
        frame: np.ndarray,
        body_box: Tuple[int, int, int, int],
    ) -> Optional[np.ndarray]:
        """
        Extracts normalized clothing/torso color histogram from the person's upper body.
        Target region: Top 25% to 75% vertically, middle 60% horizontally (clothing core).
        """
        fh, fw = frame.shape[:2]
        bx1, by1, bx2, by2 = body_box

        bw = bx2 - bx1
        bh = by2 - by1

        if bw < 20 or bh < 30:
            return None

        # Torso bounding box: focus on chest/clothing region, avoid background edges
        tx1 = max(0, int(bx1 + 0.20 * bw))
        tx2 = min(fw, int(bx2 - 0.20 * bw))
        ty1 = max(0, int(by1 + 0.25 * bh))
        ty2 = min(fh, int(by1 + 0.70 * bh))

        if tx2 <= tx1 or ty2 <= ty1:
            return None

        torso_crop = frame[ty1:ty2, tx1:tx2]
        if torso_crop.size == 0:
            return None

        # Convert to HSV color space
        hsv = cv2.cvtColor(torso_crop, cv2.COLOR_BGR2HSV)

        # Mask out extreme darks (shadows) and extreme over-saturation/glare
        mask = cv2.inRange(hsv, np.array([0, 25, 25]), np.array([180, 255, 240]))

        # Compute 2D Hue-Saturation histogram
        hist = cv2.calcHist(
            [hsv],
            channels=[0, 1],
            mask=mask,
            histSize=[self.h_bins, self.s_bins],
            ranges=[0, 180, 0, 256],
        )

        # L1 normalize histogram
        hist_sum = hist.sum()
        if hist_sum > 0:
            hist = hist / hist_sum
        else:
            return None

        return hist.astype(np.float32).flatten()

    @staticmethod
    def compute_similarity(sig_a: Optional[np.ndarray], sig_b: Optional[np.ndarray]) -> float:
        """
        Calculates Bhattacharyya affinity between two clothing color signatures.
        Returns similarity score in [0.0, 1.0]. Higher is better.
        """
        if sig_a is None or sig_b is None:
            return 0.0

        # Bhattacharyya coefficient: sum(sqrt(p * q))
        bhatt_coeff = np.sum(np.sqrt(np.clip(sig_a * sig_b, 0.0, None)))
        return float(np.clip(bhatt_coeff, 0.0, 1.0))

    def update_signature_ema(
        self,
        current_sig: Optional[np.ndarray],
        new_sig: Optional[np.ndarray],
    ) -> Optional[np.ndarray]:
        """Smoothly updates appearance signature over time to accommodate slight lighting changes."""
        if current_sig is None:
            return new_sig
        if new_sig is None:
            return current_sig

        updated = (1.0 - self.ema_alpha) * current_sig + self.ema_alpha * new_sig
        norm = updated.sum()
        if norm > 0:
            updated = updated / norm
        return updated.astype(np.float32)
