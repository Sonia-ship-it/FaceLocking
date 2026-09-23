# src/recognize.py
"""
Part 1 Recognition Module.
Contains ArcFace ONNX embedder, 5-point face detector, database loader, and matcher.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Union, Optional
import numpy as np

from .embed import ArcFaceEmbedderONNX, EmbeddingResult
from .haar_5pt import Haar5ptDetector, FaceBox5pt

# HaarFaceMesh5pt is the 5-point landmark detector class from Part 1
HaarFaceMesh5pt = Haar5ptDetector


@dataclass
class MatchResult:
    name: str
    distance: float
    similarity: float
    accepted: bool


def load_db_npz(db_path: Union[str, Path]) -> Dict[str, np.ndarray]:
    """
    Loads enrolled identity embeddings from a .npz database file.
    Returns a dictionary mapping identity names to L2-normalized (512,) embeddings.
    """
    p = Path(db_path)
    if not p.exists():
        return {}

    npz = np.load(p)
    db: Dict[str, np.ndarray] = {}

    for k in npz.files:
        vec = np.asarray(npz[k], dtype=np.float32)
        if vec.ndim > 1:
            # If multiple embeddings were stored, use mean template
            vec = np.mean(vec, axis=0)
        norm = np.linalg.norm(vec) + 1e-12
        db[k] = (vec / norm).astype(np.float32)

    return db


class FaceDBMatcher:
    """
    Matches query face embeddings against an enrolled database.
    Calculates cosine distance (1.0 - cosine_similarity).
    """

    def __init__(self, db: Dict[str, np.ndarray], dist_thresh: float = 0.34):
        self.db = db
        self.dist_thresh = dist_thresh

    def match(self, query: Union[EmbeddingResult, np.ndarray]) -> MatchResult:
        if isinstance(query, EmbeddingResult):
            q_emb = query.embedding
        else:
            q_emb = np.asarray(query, dtype=np.float32).reshape(-1)

        norm = np.linalg.norm(q_emb) + 1e-12
        q_emb = q_emb / norm

        if not self.db:
            return MatchResult(name="Unknown", distance=1.0, similarity=0.0, accepted=False)

        best_name = "Unknown"
        best_dist = float("inf")

        for name, emb in self.db.items():
            sim = float(np.dot(q_emb, emb))
            dist = max(0.0, 1.0 - sim)
            if dist < best_dist:
                best_dist = dist
                best_name = name

        similarity = max(0.0, 1.0 - best_dist)
        accepted = best_dist <= self.dist_thresh

        return MatchResult(
            name=best_name if accepted else "Unknown",
            distance=best_dist,
            similarity=similarity,
            accepted=accepted,
        )
