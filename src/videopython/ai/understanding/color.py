from __future__ import annotations

import cv2
import numpy as np
from sklearn.cluster import KMeans  # type: ignore[import-untyped]

from videopython.base.description import ColorHistogram


class ColorAnalyzer:
    """Analyzes color features from video frames."""

    def __init__(self, num_dominant_colors: int = 5):
        """Initialize the color analyzer.

        Args:
            num_dominant_colors: Number of dominant colors to extract (default: 5)
        """
        self.num_dominant_colors = num_dominant_colors

    def extract_color_features(self, frame: np.ndarray, include_full_histogram: bool = False) -> ColorHistogram:
        """Extract color features from a frame.

        Args:
            frame: Frame as numpy array (H, W, 3) in RGB format
            include_full_histogram: Whether to include full HSV histogram in result

        Returns:
            ColorHistogram object with extracted features
        """
        # Convert RGB to HSV for better color analysis
        hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)

        # Calculate average HSV values
        avg_hue = float(np.mean(hsv[:, :, 0]))
        avg_saturation = float(np.mean(hsv[:, :, 1]))
        avg_value = float(np.mean(hsv[:, :, 2]))

        # Extract dominant colors
        dominant_colors = self._extract_dominant_colors(frame)

        # Optionally calculate full histogram
        hsv_histogram = None
        if include_full_histogram:
            hsv_histogram = self._calculate_hsv_histogram(hsv)

        return ColorHistogram(
            dominant_colors=dominant_colors,
            avg_hue=avg_hue,
            avg_saturation=avg_saturation,
            avg_value=avg_value,
            hsv_histogram=hsv_histogram,
        )

    def _extract_dominant_colors(self, frame: np.ndarray) -> list[tuple[int, int, int]]:
        """Extract dominant colors from a frame using K-means clustering.

        Args:
            frame: Frame as numpy array (H, W, 3) in RGB format

        Returns:
            List of dominant colors as RGB tuples (0-255)
        """
        # Reshape frame to list of pixels
        pixels: np.ndarray = frame.reshape(-1, 3)

        # Sample pixels if image is very large (for performance)
        if len(pixels) > 10000:
            indices = np.random.choice(len(pixels), 10000, replace=False)
            pixels = pixels[indices]

        # Use K-means to find dominant colors
        kmeans = KMeans(n_clusters=self.num_dominant_colors, random_state=42, n_init=10)
        kmeans.fit(pixels)

        # Get cluster centers (dominant colors) and sort by frequency
        colors = kmeans.cluster_centers_.astype(int)
        labels = kmeans.labels_
        label_counts = np.bincount(labels)

        # Sort colors by frequency (most common first)
        sorted_indices = np.argsort(-label_counts)
        dominant_colors: list[tuple[int, int, int]] = [
            (int(colors[i][0]), int(colors[i][1]), int(colors[i][2])) for i in sorted_indices
        ]

        return dominant_colors

    def _calculate_hsv_histogram(self, hsv: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Calculate HSV histograms for all channels.

        Args:
            hsv: HSV image as numpy array (H, W, 3)

        Returns:
            Tuple of (hue_hist, saturation_hist, value_hist)
        """
        h_hist = cv2.calcHist([hsv], [0], None, [50], [0, 180])
        s_hist = cv2.calcHist([hsv], [1], None, [60], [0, 256])
        v_hist = cv2.calcHist([hsv], [2], None, [60], [0, 256])

        cv2.normalize(h_hist, h_hist, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
        cv2.normalize(s_hist, s_hist, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
        cv2.normalize(v_hist, v_hist, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)

        return (h_hist, s_hist, v_hist)

    def calculate_histogram_difference(self, frame1: np.ndarray, frame2: np.ndarray) -> float:
        """Calculate histogram difference between two frames.

        This is useful for scene detection and change detection.

        Args:
            frame1: First frame (H, W, 3) in RGB format
            frame2: Second frame (H, W, 3) in RGB format

        Returns:
            Difference score between 0.0 (identical) and 1.0 (completely different)
        """
        # Convert RGB to HSV for better color comparison
        hsv1 = cv2.cvtColor(frame1, cv2.COLOR_RGB2HSV)
        hsv2 = cv2.cvtColor(frame2, cv2.COLOR_RGB2HSV)

        # Calculate histograms
        hist1 = self._calculate_hsv_histogram(hsv1)
        hist2 = self._calculate_hsv_histogram(hsv2)

        # Compare histograms using correlation
        correlations = [cv2.compareHist(h1, h2, cv2.HISTCMP_CORREL) for h1, h2 in zip(hist1, hist2)]
        avg_correlation = sum(correlations) / len(correlations)

        # Convert correlation (1.0 = similar) to difference (0.0 = similar)
        difference = 1.0 - avg_correlation

        return difference
