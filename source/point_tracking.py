from typing import List

import cv
import feature_estimation
import torch
import torch.nn.functional as F


class PointTracker:
    def __init__(self, distance_threshold: float = 5):
        self._distance_threshold = distance_threshold
        self._tracked_points: torch.tensor = None

    def tracked_points(self) -> torch.tensor:
        return self._tracked_points

    def track_points(self, video: torch.tensor, feature_estimator: feature_estimation.FeatureEstimator) -> List[torch.tensor]:
        # Compute Glottal Area Waveform from glottis segmentations
        gaw: torch.tensor = feature_estimator.glottalAreaWaveform()

        # Smooth 1D tensor
        gaw = cv.gaussian_smooth_1d(gaw, kernel_size=5, sigma=2.0)

        maxima_indices, values = cv.find_local_maxima_1d(gaw)
        
        # Find laser points in frames, where glottis is minimal
        laserpoints_when_glottis_closed: List[torch.tensor] = [feature_estimator.laserpointPositions[maxima_index] for maxima_index in maxima_indices]

        # Compute temporal nearest neighbors in frames of closed glottis
        nearest_neighbors = cv.find_nearest_neighbors_consecutive(laserpoints_when_glottis_closed)

        # Interpolate from neighbors
        per_frame_point_position_estimates: torch.tensor = cv.interpolate_from_neighbors(maxima_indices, laserpoints_when_glottis_closed, nearest_neighbors)

        # Extract windows from position estimates
        per_point_crops: torch.tensor = cv.extract_windows(video, per_frame_point_position_estimates)

        classified_points: torch.tensor = cv.classify_points(per_point_crops)
        
        point_predictions = None
        labels = None
        num_point_per_frame = None

        def filter_points(a,b):
            return 0


        subpixel_points: torch.tensor = cv.compute_subpixel_points(point_predictions, labels, video, num_points_per_frame)

        # Filter points, that do not lie on vocal folds here.
        subpixel_points = filter_points(subpixel_points, feature_estimator.glottisSegmentations())
        subpixel_points = filter_points(subpixel_points, feature_estimator.vocalfoldSegmentations())
        self._tracked_points = subpixel_points

        return self._tracked_points, 0, -1


class SiliconePointTracker(PointTracker):
    def __init__(self, distance_threshold: float = 5):
        super().__init__(distance_threshold)

    def track_points(self):
        # TODO: Implement me!
        return self._tracked_points