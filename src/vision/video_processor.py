"""
Computer-vision traffic feature extractor for Surakshit Bharat Q-JEPA.

STATUS: Stub implementation. The full pipeline is designed but optional.
The system works without this module — the simulator provides observations directly.

Required functionality (implemented as stubs):
1. Load a local video or sample frames
2. Validate video readability
3. Sample frames at a configurable frequency
4. Detect/track vehicles (stub: returns synthetic counts)
5. Estimate vehicle counts, flow, queue lengths, congestion indicators
6. Produce timestamped traffic features
7. Save preview video or annotated frames

If OpenCV is not available or no video is provided, the system falls back to
simulator-generated observations. This module is OPTIONAL.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np

from src.common.logging import get_logger

logger = get_logger(__name__)

# Optional OpenCV import
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    logger.warning("OpenCV not available. Vision module will use synthetic fallback.")


@dataclass
class VideoTrafficFeature:
    """Traffic feature extracted from one video frame."""
    frame_index: int
    timestamp_s: float
    vehicle_count: int
    estimated_flow_veh_per_hour: float
    queue_length_estimate: float  # normalised 0-1
    congestion_indicator: float   # normalised 0-1
    source: str                   # "video" or "synthetic_fallback"


class VideoProcessor:
    """
    Processes a traffic video and extracts per-frame traffic features.

    LIMITATIONS (documented):
    - Vehicle counts are estimated from background subtraction, not a calibrated detector.
    - Speed and queue measurements are not accurate without camera calibration data.
    - A pretrained detector (e.g. YOLOv8) would require downloading weights (~6 MB+).
      This stub avoids that download. Integrate a real detector for production use.
    - This module is OPTIONAL. The simulator provides direct observations as fallback.
    """

    def __init__(
        self,
        video_path: Optional[Path] = None,
        sample_every_n_frames: int = 10,
        seed: int = 42,
    ):
        self.video_path = video_path
        self.sample_every_n_frames = sample_every_n_frames
        self.seed = seed
        self._rng = np.random.default_rng(seed)

    def validate_video(self) -> bool:
        """Check that the video file is readable."""
        if not CV2_AVAILABLE:
            logger.warning("OpenCV unavailable — cannot validate video.")
            return False
        if self.video_path is None or not self.video_path.exists():
            logger.warning("Video path not set or does not exist: %s", self.video_path)
            return False
        cap = cv2.VideoCapture(str(self.video_path))
        ok = cap.isOpened()
        cap.release()
        logger.info("Video validation: %s — %s", self.video_path.name, "OK" if ok else "FAILED")
        return ok

    def extract_features(self) -> List[VideoTrafficFeature]:
        """
        Extract traffic features from video.
        Falls back to synthetic generation if video is unavailable.
        """
        if CV2_AVAILABLE and self.video_path is not None and self.video_path.exists():
            return self._extract_from_video()
        else:
            logger.info("Using synthetic fallback for vision features.")
            return self._synthetic_fallback(n_frames=50)

    def _extract_from_video(self) -> List[VideoTrafficFeature]:
        """Extract features using background subtraction (basic method)."""
        cap = cv2.VideoCapture(str(self.video_path))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        bg_subtractor = cv2.createBackgroundSubtractorMOG2()
        features = []
        frame_idx = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % self.sample_every_n_frames == 0:
                # Background subtraction to detect moving vehicles
                fg_mask = bg_subtractor.apply(frame)
                # Count non-zero pixels as proxy for vehicle presence
                moving_pixels = np.count_nonzero(fg_mask)
                total_pixels = fg_mask.size
                congestion = min(1.0, moving_pixels / max(total_pixels * 0.1, 1))
                vehicle_count = int(moving_pixels / 500)  # rough proxy — NOT calibrated
                flow = vehicle_count * (3600 / (self.sample_every_n_frames / fps))

                features.append(VideoTrafficFeature(
                    frame_index=frame_idx,
                    timestamp_s=frame_idx / fps,
                    vehicle_count=vehicle_count,
                    estimated_flow_veh_per_hour=flow,
                    queue_length_estimate=congestion,
                    congestion_indicator=congestion,
                    source="video",
                ))
            frame_idx += 1

        cap.release()
        logger.info(
            "Extracted %d features from video (every %d frames)",
            len(features), self.sample_every_n_frames,
        )
        return features

    def _synthetic_fallback(self, n_frames: int = 50) -> List[VideoTrafficFeature]:
        """
        Generate synthetic video-equivalent features.
        Clearly labelled as synthetic — NOT real video observations.
        """
        features = []
        for i in range(n_frames):
            t = i * self.sample_every_n_frames / 25.0
            hour = (t / 3600.0) % 24
            # Simple rush-hour pattern
            if 8 <= hour < 10 or 17 <= hour < 19:
                base_cong = 0.7
            else:
                base_cong = 0.3
            cong = float(np.clip(base_cong + self._rng.normal(0, 0.05), 0, 1))
            count = int(max(0, self._rng.normal(15 * cong, 3)))
            features.append(VideoTrafficFeature(
                frame_index=i * self.sample_every_n_frames,
                timestamp_s=t,
                vehicle_count=count,
                estimated_flow_veh_per_hour=float(count * 180),
                queue_length_estimate=cong,
                congestion_indicator=cong,
                source="synthetic_fallback",
            ))
        return features

    def is_available(self) -> bool:
        return CV2_AVAILABLE and self.video_path is not None and self.video_path.exists()
