import re
from typing import List

import cv
import feature_estimation
import kornia
import NeuralSegmentation
import torch
import torch.nn.functional as F


# From https://discuss.pytorch.org/t/how-to-do-a-unravel-index-in-pytorch-just-like-in-numpy/12987/3
# Can't use pytorchs own, since this project started with Pytorch 1.x :(
# At some point, we should upgrade this.
def unravel_index(index, shape):
    out = []
    for dim in reversed(shape):
        out.append(index % dim)
        index = index // dim
    return tuple(reversed(out))


class PointTracker:
    def __init__(self, distance_threshold: float = 5):
        self._distance_threshold = distance_threshold
        self._tracked_points: torch.tensor = None

        self._point_classificator = NeuralSegmentation.BinaryKernel3Classificator()
        self._point_classificator.load_state_dict(
            torch.load(
                "assets/binary_specularity_classificator.pth.tar",
                map_location=torch.device("cpu"),
            )
        )
        self._point_classificator.cuda()
        self._point_classificator.eval()

    def draw_points_on_image(self, frame, points) -> torch.tensor:
        empty_frame: torch.tensor = torch.zeros_like(frame)

        mask = ~torch.isnan(points).any(dim=-1)
        cleaned_points = points[mask]
        x = cleaned_points[:, 1].long()
        y = cleaned_points[:, 0].long()

        empty_frame[y, x] = 1
        kornia.morphology.dilation(empty_frame.unsqueeze(0).unsqueeze(0).float(), torch.ones((5, 5), device=frame.device)).squeeze()
        copied_image = frame.clone()
        copied_image[empty_frame != 0] = 0

        return copied_image


    def tracked_points(self) -> torch.tensor:
        return self._tracked_points

    def track_points(self, video: torch.tensor, feature_estimator: feature_estimation.FeatureEstimator) -> List[torch.tensor]:
        # Compute Glottal Area Waveform from glottis segmentations
        gaw: torch.tensor = feature_estimator.glottalAreaWaveform()

        # Smooth 1D tensor
        gaw = cv.gaussian_smooth_1d(gaw, kernel_size=5, sigma=2.0)

        maxima_indices, values = cv.find_local_minima_1d(gaw)
        
        # Find laser points in frames, where glottis is minimal
        laserpoints_when_glottis_closed: List[torch.tensor] = [feature_estimator.laserpointPositions()[maxima_index].float() for maxima_index in maxima_indices]

        # Compute temporal nearest neighbors in frames of closed glottis
        nearest_neighbors = cv.compute_point_estimates_from_nearest_neighbors(laserpoints_when_glottis_closed)

        # Interpolate from neighbors
        glottal_maxima_list: List[int] = maxima_indices.tolist()
        glottal_maxima_list.insert(0, 0)
        glottal_maxima_list.append(video.shape[0])
        
        # Copy first and last point positions
        nearest_neighbors = torch.concat([nearest_neighbors[:1], nearest_neighbors])
        nearest_neighbors = torch.concat([nearest_neighbors, nearest_neighbors[-1:]])

        per_frame_point_position_estimates: torch.tensor = cv.interpolate_from_neighbors(glottal_maxima_list, nearest_neighbors)
        per_frame_point_position_estimates = per_frame_point_position_estimates[:, :, [1, 0]]

        A, B, C = per_frame_point_position_estimates.shape
        flattened: torch.tensor = per_frame_point_position_estimates.clone().reshape(-1, C)
        batch_indices = torch.arange(0, A, device=per_frame_point_position_estimates.device).repeat_interleave(B).reshape(-1, 1)
        indexed_point_positions = torch.concat([batch_indices, flattened], dim=1)

        # Extract windows from position estimates
        crops, y_windows, x_windows = cv.extract_windows_from_batch(video, indexed_point_positions)
        per_crop_max = crops.amax([-1, -2], keepdim=True)
        per_crop_min = crops.amin([-1, -2], keepdim=True)

        normalized_crops = (crops - per_crop_min) / (per_crop_max - per_crop_min)

        # Use 2-layered CNN to classify points
        prediction = self._point_classificator(normalized_crops[:, None, :, :])
        classifications = (torch.sigmoid(prediction) > 0.5) * 1
        classifications = classifications.reshape(per_frame_point_position_estimates.shape[0], per_frame_point_position_estimates.shape[1])


        point_predictions = per_frame_point_position_estimates
            # 0.1 Reshape points, classes and crops into per frame segments, such that we can easily extract a timeseries.
        # I.e. shape is after this: NUM_POINTS x NUM_FRAMES x ...
        point_predictions = point_predictions.permute(1, 0, 2)
        y_windows = y_windows.reshape(
            video.shape[0], classifications.shape[1], crops.shape[-2], crops.shape[-1]
        ).permute(1, 0, 2, 3)
        x_windows = x_windows.reshape(
            video.shape[0], classifications.shape[1], crops.shape[-2], crops.shape[-1]
        ).permute(1, 0, 2, 3)
        labels = classifications.permute(1, 0)
        crops = crops.reshape(
            video.shape[0], classifications.shape[1], crops.shape[-2], crops.shape[-1]
        ).permute(1, 0, 2, 3)

        specular_duration = 5
        # Iterate over every point and class as well as their respective crops
        optimized_points = torch.zeros_like(point_predictions) * torch.nan
        optimized_points_on_crops = torch.zeros_like(point_predictions) * torch.nan
        for points_index, (points, label, crop) in enumerate(
            zip(point_predictions, labels, crops)
        ):

            # Here it now gets super hacky.
            # Convert label array to a string
            labelstring = "".join(map(str, label.squeeze().tolist()))
            # Replace 0s with V for visible
            compute_string = labelstring.replace("1", "V")

            # This regex looks for occurences of VXV, where X may be any mix of specularity or unidentifiable classifications but at most of length 5.
            # If this is given, we will replace VXV by VIV, where X is replaced by that many Is.#
            # Is indicate that we want to interpolate in these values.
            compute_string = re.sub(
                r"(V)([0]+)(V)",
                lambda match: match.group(1) + "I" * len(match.group(2)) + match.group(3),
                compute_string,
            )
            compute_string = re.sub(
                r"(V)([0]+)(V)",
                lambda match: match.group(1) + "I" * len(match.group(2)) + match.group(3),
                compute_string,
            )

            # Finally, every part that couldn't be identified will be labeled as E for error.
            compute_string = compute_string.replace("0", "E")
            compute_string = compute_string.replace("1", "E")
            compute_string = compute_string.replace("2", "E")
            #print(points_index, compute_string)

            if points_index == 100:
                a = 1

            # Compute sub-pixel position for each point labeled as visible (V)
            for frame_index, label in enumerate(compute_string):
                if label != "V":
                    continue

                normalized_crop = crop[frame_index]
                normalized_crop = (normalized_crop - normalized_crop.min()) / (
                    normalized_crop.max() - normalized_crop.min()
                )

                # Find local maximum in 5x5 crop
                local_maximum = unravel_index(
                    torch.argmax(normalized_crop[1:-1, 1:-1]), [5, 5]
                )

                # Add one again, since we removed the border from the local maximum lookup
                x0, y0 = local_maximum[1] + 1, local_maximum[0] + 1

                # Get 3x3 subwindow from crop, where the local maximum is centered.
                neighborhood = 1
                x_min = max(0, x0 - neighborhood)
                x_max = min(normalized_crop.shape[1], x0 + neighborhood + 1)
                y_min = max(0, y0 - neighborhood)
                y_max = min(normalized_crop.shape[0], y0 + neighborhood + 1)

                sub_image = normalized_crop[y_min:y_max, x_min:x_max]
                sub_image = (sub_image - sub_image.min()) / (
                    sub_image.max() - sub_image.min()
                )

                centroids = cv.moment_method(
                    sub_image.unsqueeze(0)
                ).squeeze()

                refined_x = (
                    x_windows[points_index, frame_index, 0, 0] + centroids[0] + x0 - 1
                ).item()
                refined_y = (
                    y_windows[points_index, frame_index, 0, 0] + centroids[1] + y0 - 1
                ).item()

                on_crop_x = (x0 + centroids[0] - 1).item()
                on_crop_y = (y0 + centroids[1] - 1).item()

                optimized_points[points_index, frame_index] = torch.tensor(
                    [refined_x, refined_y]
                )
                optimized_points_on_crops[points_index, frame_index] = torch.tensor(
                    [on_crop_x, on_crop_y]
                )

            # Interpolate inbetween two points
            for frame_index, label in enumerate(compute_string):
                if label != "I":
                    continue

                prev_v_index = compute_string.rfind("V", 0, frame_index)
                next_v_index = compute_string.find("V", frame_index + 1)

                lerp_alpha = (frame_index - prev_v_index) / (next_v_index - prev_v_index)
                point_a = optimized_points[points_index, prev_v_index]
                point_b = optimized_points[points_index, next_v_index]
                lerped_point = cv.lerp(point_a, point_b, lerp_alpha)

                optimized_points[points_index, frame_index] = lerped_point
        
        return optimized_points[:, :, [1, 0]].permute(1, 0, 2)


class SiliconePointTracker(PointTracker):
    def __init__(self, distance_threshold: float = 5):
        super().__init__(distance_threshold)

    def track_points(self):
        # TODO: Implement me!
        return self._tracked_points