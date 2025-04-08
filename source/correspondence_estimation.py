from typing import List

import torch


class CorrespondenceEstimator:
    def __init__(self):
        self._correspondences: List[torch.tensor] = None

    def correspondences(self) -> List[torch.tensor]:
        return self._point_labels

    def compute_correspondences(camera, laser, points2d: List[torch.tensor], frame_index: int) -> None:
        # return labels in the form of points2d
        return points2d