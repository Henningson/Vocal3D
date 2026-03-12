import re
from typing import List

import cv
import feature_estimation
import kornia
import NeuralSegmentation
import torch
import torch.nn.functional as F
from tqdm import tqdm


# From https://discuss.pytorch.org/t/how-to-do-a-unravel-index-in-pytorch-just-like-in-numpy/12987/3
# Can't use pytorchs own, since this project started with Pytorch 1.x :(
# At some point, we should upgrade this.
def unravel_index(index, shape):
    out = []
    for dim in reversed(shape):
        out.append(index % dim)
        index = torch.div(index, dim, rounding_mode='floor')
    return tuple(reversed(out))



class PointTrackerBase:
    def __init__(self, distance_threshold: float = 5, min_point_intensity: int = 50, device="cpu"):
        self._distance_threshold = distance_threshold
        self._tracked_points: torch.tensor = None
        self._min_point_intensity = min_point_intensity

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
        return torch.zeros(1)


class InvivoPointTracker(PointTrackerBase):
    def __init__(self, distance_threshold: float = 3.0, min_point_intensity: int = 50, device="cpu"):
        super().__init__(distance_threshold, min_point_intensity, device)
        self._point_classificator = NeuralSegmentation.BinaryKernel3Classificator()
        self._point_classificator.load_state_dict(
            torch.load(
                "assets/binary_specularity_classificator.pth.tar",
                map_location=torch.device("cpu"),
            )
        )
        self._point_classificator.eval()

    def track_points(self, video: torch.tensor, feature_estimator: feature_estimation.FeatureEstimator) -> List[torch.tensor]:
        # Move point classificator to device of video
        self._point_classificator.to(video.device)
        feature_estimator.to(video.device)

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
        crops, y_windows, x_windows = cv.extract_windows_from_batch(video, indexed_point_positions, device=video.device)
        per_crop_max = crops.amax([-1, -2], keepdim=True)
        per_crop_min = crops.amin([-1, -2], keepdim=True)

        normalized_crops = (crops - per_crop_min) / (per_crop_max - per_crop_min)

        # Use 3-layered CNN to classify points
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
        for points_index, (points, label, crop) in tqdm(enumerate(
            zip(point_predictions, labels, crops)
        )):

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
        

        # convert points to frame x num_points x 2 [Y,X]
        optimized_points = optimized_points[:, :, [1, 0]].permute(1, 0, 2)


        # Filter points that fall into the glottal region
        optimized_points = filter_points_on_glottis(optimized_points, feature_estimator.glottisSegmentations())

        optimized_points = filter_points_by_intensity(optimized_points, video, intensity_threshold=self._min_point_intensity)

        return optimized_points


def smooth_points2(points: torch.tensor) -> torch.tensor:
    # Define a Gaussian kernel
    def gaussian_kernel(size, sigma):
        x = torch.arange(-size // 2 + 1, size // 2 + 1)
        kernel = torch.exp(-0.5 * (x / sigma) ** 2)
        kernel /= kernel.sum()  # Normalize
        return kernel

    kernel_size = 5
    sigma = 1.0
    kernel = gaussian_kernel(kernel_size, sigma).view(1, 1, -1)
    kernel = kernel.to(points.device)

    points = points.permute(1, 0, 2)
    for index, point_over_time in enumerate(points):
        padded_point = F.pad(
            point_over_time.permute(1, 0), (2, 2), mode="constant"
        ).permute(1, 0)
        smoothed_points = F.conv1d(
            padded_point.permute(1, 0).unsqueeze(0),
            kernel.float().repeat(point_over_time.shape[1], 1, 1),
            groups=2,
        ).squeeze().permute(1, 0)
        points[index] = smoothed_points

    return points.permute(1, 0, 2)



class InvivoPointTrackerNew(PointTrackerBase):
    def __init__(self, distance_threshold: float = 3.0, min_point_intensity: int = 50, device="cpu"):
        super().__init__(distance_threshold, min_point_intensity, device)
        self._point_classificator = NeuralSegmentation.BinaryKernel3Classificator()
        self._point_classificator.load_state_dict(
            torch.load(
                "assets/binary_specularity_classificator.pth.tar",
                map_location=torch.device("cpu"),
            )
        )
        self._point_classificator.eval()

    def track_points(self, video: torch.tensor, feature_estimator: feature_estimation.FeatureEstimator) -> List[torch.tensor]:
        # Move point classificator to device of video
        self._point_classificator.to(video.device)
        feature_estimator.to(video.device)

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
        crops, y_windows, x_windows = cv.extract_windows_from_batch(video, indexed_point_positions, device=video.device)
        per_crop_max = crops.amax([-1, -2], keepdim=True)
        per_crop_min = crops.amin([-1, -2], keepdim=True)

        normalized_crops = (crops - per_crop_min) / (per_crop_max - per_crop_min)

        # Use 3-layered CNN to classify points
        prediction = self._point_classificator(normalized_crops[:, None, :, :])
        classifications = (torch.sigmoid(prediction) > 0.5) * 1
        classifications = classifications.reshape(per_frame_point_position_estimates.shape[0], per_frame_point_position_estimates.shape[1])


        point_predictions = per_frame_point_position_estimates
            # 0.1 Reshape points, classes and crops into per frame segments, such that we can easily extract a timeseries.
        # I.e. shape is after this: NUM_POINTS x NUM_FRAMES x ...
        point_predictions = point_predictions.permute(1, 0, 2)
        y_windows = y_windows.reshape(
            video.shape[0], classifications.shape[1], crops.shape[-2], crops.shape[-1]
        ).permute(1, 0, 2, 3)[:, :, 0, 0]
        x_windows = x_windows.reshape(
            video.shape[0], classifications.shape[1], crops.shape[-2], crops.shape[-1]
        ).permute(1, 0, 2, 3)[:, :, 0, 0]
        labels = classifications.permute(1, 0)
        crops = crops.reshape(
            video.shape[0], classifications.shape[1], crops.shape[-2], crops.shape[-1]
        ).permute(1, 0, 2, 3)
        # normalized_crops = normalized_crops.reshape(
        #     video.shape[0], classifications.shape[1], crops.shape[-2], crops.shape[-1]
        # ).permute(1, 0, 2, 3)

        specular_duration = 5
        # Iterate over every point and class as well as their respective crops
        

        # Compute masks that define how points should be computed next.
        # Labels that were classified as 1 should get directly computed
        # Labels that are 0 should be interpolated but only if they are inbetween 1s and the sequence of 0s is not longer than 5
        # We compute this with a row-based template matching approach
        kernels = [
            torch.tensor([1, 0, 1], device=point_predictions.device),
            torch.tensor([1, 0, 0, 1], device=point_predictions.device),
            torch.tensor([1, 0, 0, 0, 1], device=point_predictions.device),
            torch.tensor([1, 0, 0, 0, 0, 1], device=point_predictions.device),
            torch.tensor([1, 0, 0, 0, 0, 0, 1], device=point_predictions.device),
        ]

        interpolation_mask = torch.zeros_like(labels)
        direct_mask = labels

        for kernel in kernels:
            interpolation_mask += cv.match_pattern_unfold(labels, kernel)
        interpolation_mask = (interpolation_mask > 0) * 1

        # Remove ends of interpolation masks, as we want to compute the pixel there directly
        interpolation_edges = (direct_mask & interpolation_mask)
        interpolation_mask = interpolation_mask - interpolation_edges

        # Get local maxima inside 5x5 window
        P, F , _, _ = crops.shape
        PF, H, W = normalized_crops[:, 1:-1, 1:-1].shape
        crops_flattened = normalized_crops[:, 1:-1, 1:-1].reshape(P * F, -1)

        max_vals, max_indices = crops_flattened.max(dim=-1)

        # Recover h, w indices from flattened index
        h_indices = torch.div(max_indices, W, rounding_mode='trunc') + 1  # +1 to offset removed border
        w_indices = (max_indices % W) + 1

        # Get 3x3 subwindows around local maxima
        neighborhood = 1
        x_min = torch.maximum(torch.tensor([0], device=w_indices.device), w_indices - neighborhood)
        y_min = torch.maximum(torch.tensor([0], device=w_indices.device), h_indices - neighborhood)

        window_size = neighborhood + 2
        # Assume x is (B, C, H, W)
        x_flat = normalized_crops#.view(PF, window_size, window_size)

        # Relative coordinate grid (H_win, W_win)
        dy, dx = torch.meshgrid(
            torch.arange(window_size, device=crops.device),
            torch.arange(window_size, device=crops.device),
            indexing='ij'  # ensures (row, col) order
        )

        # Broadcast dy, dx to (B*C, H_win, W_win)
        ys = y_min[..., None, None] + dy     # (B*C, H_win, W_win)
        xs = x_min[..., None, None] + dx     # (B*C, H_win, W_win)

        # Batch indices (B*C, H_win, W_win)
        batch_idx = torch.arange(PF, device=crops.device)[..., None, None].expand(-1, window_size, window_size)

        # Extract windows (B*C, H_win, W_win)
        windows_flat = normalized_crops[batch_idx, ys, xs]

        # Reshape back to (B, C, H_win, W_win)
        sub_windows = windows_flat.view(PF, window_size, window_size)
        
        # Compute sub-pixel accurate points using the moment method
        # We could also utilize guo's method here obviously.
        subpixel_points = cv.batched_centroid_method(sub_windows).squeeze().reshape(video.shape[0], classifications.shape[1], 2).permute(1, 0, 2)

        # Add sub-window offsets from original frame, and subsubwindow offsets
        subpixel_points += torch.concat([x_windows[..., None], y_windows[..., None]], dim=-1).float() + torch.stack([x_min, y_min], dim=-1).reshape(video.shape[0], classifications.shape[1], 2).permute(1, 0, 2)

        # Recall that the subpixel points now also include points that we know should not have been computed
        # I.e. the points that we found out that need to be interpolated defined in the interpolation mask
        # Unfortunately, to resolve this we will now need a nested loop to find points for interpolation
        for p in range(P):
            points_over_time = subpixel_points[p]
            interpolants_over_time = interpolation_edges[p].nonzero().squeeze()

            interpolated_points = torch.zeros_like(points_over_time)
            interpolants_over_time = interpolation_edges[p].nonzero().squeeze()

            for j in range(interpolants_over_time.shape[0] - 1):


                first_lerp_index = interpolants_over_time[j]
                second_lerp_index = interpolants_over_time[j + 1]
                
                # Compute mid-point such that we only compute necessary interpolation masks
                mid_point = ((second_lerp_index + first_lerp_index) / 2).floor().int()
                if not interpolation_mask[p, mid_point]:
                    continue

                interval_length = second_lerp_index - first_lerp_index + 1
                first_lerp_point = points_over_time[first_lerp_index]
                second_lerp_point = points_over_time[second_lerp_index]

                first_lerp_point = first_lerp_point.unsqueeze(0).repeat(interval_length, 1)
                second_lerp_point = second_lerp_point.unsqueeze(0).repeat(interval_length, 1)
                alphas = torch.arange(0, interval_length, device=subpixel_points.device) / interval_length

                lerped_points = cv.lerp(first_lerp_point, second_lerp_point, alphas[..., None])
                subpixel_points[p, first_lerp_index:second_lerp_index+1] = lerped_points



        # smooth points
        subpixel_points = smooth_points2(subpixel_points.permute(1, 0, 2)).permute(1, 0, 2)

        # Mask out wrongly lerped points
        # Mask out points that needed to be lerped in subpixel_points
        subpixel_points *= (interpolation_mask | direct_mask)[..., None]

        # Add masked points to subpixel_points
        optimized_points = subpixel_points[:, :, [1, 0]].permute(1, 0, 2)

        #optimized_points = smooth_points2(optimized_points)

        # Filter points that fall into the glottal region
        optimized_points = filter_points_on_glottis(optimized_points.cpu(), feature_estimator.glottisSegmentations().cpu())

        # And lie below a certain intensity
        optimized_points = filter_points_by_intensity(optimized_points.cpu(), video.cpu(), intensity_threshold=self._min_point_intensity)

        return optimized_points






class InvivoPointTrackerNewNew(PointTrackerBase):
    def __init__(
        self,
        distance_threshold: float = 3.0,
        min_point_intensity: int = 50,
        device="cpu",
    ):
        super().__init__(distance_threshold, min_point_intensity, device)
        self._point_classificator = NeuralSegmentation.BinaryKernel3Classificator()
        self._point_classificator.load_state_dict(
            torch.load(
                "assets/binary_specularity_classificator.pth.tar",
                map_location=torch.device("cpu"),
            )
        )
        self._point_classificator.eval()

    def track_points(
        self,
        video: torch.tensor,
        feature_estimator: feature_estimation.FeatureEstimator,
    ) -> List[torch.tensor]:
        # Move point classificator to device of video
        self._point_classificator.to(video.device)
        feature_estimator.to(video.device)

        with torch.no_grad():

            # Synchronize before starting timer
            torch.cuda.synchronize()

            # Start timer
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)

            start_event.record()

            # Compute Glottal Area Waveform from glottis segmentations
            gaw: torch.tensor = feature_estimator.glottalAreaWaveform()

            # Smooth 1D tensor
            gaw = cv.gaussian_smooth_1d(gaw, kernel_size=5, sigma=2.0)

            maxima_indices, values = cv.find_local_minima_1d(gaw)

            # Find laser points in frames, where glottis is minimal
            laserpoints_when_glottis_closed: List[torch.tensor] = [
                feature_estimator.laserpointPositions()[maxima_index].float()
                for maxima_index in maxima_indices
            ]

            # Compute temporal nearest neighbors in frames of closed glottis
            nearest_neighbors = cv.compute_point_estimates_from_nearest_neighbors(
                laserpoints_when_glottis_closed
            )

            # Interpolate from neighbors
            glottal_maxima_list: List[int] = maxima_indices.tolist()
            glottal_maxima_list.insert(0, 0)
            glottal_maxima_list.append(video.shape[0])

            # Copy first and last point positions
            nearest_neighbors = torch.concat([nearest_neighbors[:1], nearest_neighbors])
            nearest_neighbors = torch.concat(
                [nearest_neighbors, nearest_neighbors[-1:]]
            )

            per_frame_point_position_estimates: torch.tensor = (
                cv.interpolate_from_neighbors(glottal_maxima_list, nearest_neighbors)
            )
            per_frame_point_position_estimates = per_frame_point_position_estimates[
                :, :, [1, 0]
            ]

            A, B, C = per_frame_point_position_estimates.shape
            flattened: torch.tensor = (
                per_frame_point_position_estimates.clone().reshape(-1, C)
            )
            batch_indices = (
                torch.arange(0, A, device=per_frame_point_position_estimates.device)
                .repeat_interleave(B)
                .reshape(-1, 1)
            )
            indexed_point_positions = torch.concat([batch_indices, flattened], dim=1)

            # Extract windows from position estimates
            crops, y_windows, x_windows = cv.extract_windows_from_batch(
                video, indexed_point_positions, device=video.device
            )
            per_crop_max = crops.amax([-1, -2], keepdim=True)
            per_crop_min = crops.amin([-1, -2], keepdim=True)

            normalized_crops = (crops - per_crop_min) / (per_crop_max - per_crop_min)

            # Use 3-layered CNN to classify points
            prediction = self._point_classificator(normalized_crops[:, None, :, :])
            classifications = (torch.sigmoid(prediction) > 0.3) * 1
            classifications = classifications.reshape(
                per_frame_point_position_estimates.shape[0],
                per_frame_point_position_estimates.shape[1],
            )

            point_predictions = per_frame_point_position_estimates
            # 0.1 Reshape points, classes and crops into per frame segments, such that we can easily extract a timeseries.
            # I.e. shape is after this: NUM_POINTS x NUM_FRAMES x ...
            point_predictions = point_predictions.permute(1, 0, 2)
            y_windows = y_windows.reshape(
                video.shape[0],
                classifications.shape[1],
                crops.shape[-2],
                crops.shape[-1],
            ).permute(1, 0, 2, 3)[:, :, 0, 0]
            x_windows = x_windows.reshape(
                video.shape[0],
                classifications.shape[1],
                crops.shape[-2],
                crops.shape[-1],
            ).permute(1, 0, 2, 3)[:, :, 0, 0]
            labels = classifications.permute(1, 0)
            crops = crops.reshape(
                video.shape[0],
                classifications.shape[1],
                crops.shape[-2],
                crops.shape[-1],
            ).permute(1, 0, 2, 3)
            # normalized_crops = normalized_crops.reshape(
            #     video.shape[0], classifications.shape[1], crops.shape[-2], crops.shape[-1]
            # ).permute(1, 0, 2, 3)

            specular_duration = 5
            # Iterate over every point and class as well as their respective crops

            # Compute masks that define how points should be computed next.
            # Labels that were classified as 1 should get directly computed
            # Labels that are 0 should be interpolated but only if they are inbetween 1s and the sequence of 0s is not longer than 5
            # We compute this with a row-based template matching approach
            kernels = [
                torch.tensor([1, 0, 1], device=point_predictions.device),
                torch.tensor([1, 0, 0, 1], device=point_predictions.device),
                torch.tensor([1, 0, 0, 0, 1], device=point_predictions.device),
                torch.tensor([1, 0, 0, 0, 0, 1], device=point_predictions.device),
                torch.tensor([1, 0, 0, 0, 0, 0, 1], device=point_predictions.device),
            ]

            interpolation_mask = torch.zeros_like(labels)
            direct_mask = labels

            for kernel in kernels:
                interpolation_mask += cv.match_pattern_unfold(labels, kernel)
            interpolation_mask = (interpolation_mask > 0) * 1

            # Remove ends of interpolation masks, as we want to compute the pixel there directly
            interpolation_edges = direct_mask & interpolation_mask
            interpolation_mask = interpolation_mask - interpolation_edges

            # Get local maxima inside 5x5 window
            P, F, _, _ = crops.shape
            PF, H, W = normalized_crops[:, 1:-1, 1:-1].shape
            crops_flattened = normalized_crops[:, 1:-1, 1:-1].reshape(P * F, -1)

            max_vals, max_indices = crops_flattened.max(dim=-1)

            # Recover h, w indices from flattened index
            h_indices = (
                torch.div(max_indices, W, rounding_mode="trunc") + 1
            )  # +1 to offset removed border
            w_indices = (max_indices % W) + 1

            # Get 3x3 subwindows around local maxima
            neighborhood = 1
            x_min = torch.maximum(
                torch.tensor([0], device=w_indices.device), w_indices - neighborhood
            )
            y_min = torch.maximum(
                torch.tensor([0], device=w_indices.device), h_indices - neighborhood
            )

            window_size = neighborhood + 2
            # Assume x is (B, C, H, W)
            x_flat = normalized_crops  # .view(PF, window_size, window_size)

            # Relative coordinate grid (H_win, W_win)
            dy, dx = torch.meshgrid(
                torch.arange(window_size, device=crops.device),
                torch.arange(window_size, device=crops.device),
                indexing="ij",  # ensures (row, col) order
            )

            # Broadcast dy, dx to (B*C, H_win, W_win)
            ys = y_min[..., None, None] + dy  # (B*C, H_win, W_win)
            xs = x_min[..., None, None] + dx  # (B*C, H_win, W_win)

            # Batch indices (B*C, H_win, W_win)
            batch_idx = torch.arange(PF, device=crops.device)[..., None, None].expand(
                -1, window_size, window_size
            )

            # Extract windows (B*C, H_win, W_win)
            windows_flat = normalized_crops[batch_idx, ys, xs]

            # Reshape back to (B, C, H_win, W_win)
            sub_windows = windows_flat.view(PF, window_size, window_size)

            # Compute sub-pixel accurate points using the moment method
            # We could also utilize guo's method here obviously.
            subpixel_points = (
                cv.batched_centroid_method(sub_windows)
                .squeeze()
                .reshape(video.shape[0], classifications.shape[1], 2)
                .permute(1, 0, 2)
            )

            # Add sub-window offsets from original frame, and subsubwindow offsets
            subpixel_points += torch.concat(
                [x_windows[..., None], y_windows[..., None]], dim=-1
            ).float() + torch.stack([x_min, y_min], dim=-1).reshape(
                video.shape[0], classifications.shape[1], 2
            ).permute(
                1, 0, 2
            )

            # Recall that the subpixel points now also include points that we know should not have been computed
            # I.e. the points that we found out that need to be interpolated defined in the interpolation mask
            # Unfortunately, to resolve this we will now need a nested loop to find points for interpolation
            for p in range(P):
                points_over_time = subpixel_points[p]
                interpolants_over_time = interpolation_edges[p].nonzero().squeeze()

                interpolated_points = torch.zeros_like(points_over_time)
                interpolants_over_time = interpolation_edges[p].nonzero().squeeze()

                for j in range(interpolants_over_time.shape[0] - 1):

                    first_lerp_index = interpolants_over_time[j]
                    second_lerp_index = interpolants_over_time[j + 1]

                    # Compute mid-point such that we only compute necessary interpolation masks
                    mid_point = (
                        ((second_lerp_index + first_lerp_index) / 2).floor().int()
                    )
                    if not interpolation_mask[p, mid_point]:
                        continue

                    interval_length = second_lerp_index - first_lerp_index + 1
                    first_lerp_point = points_over_time[first_lerp_index]
                    second_lerp_point = points_over_time[second_lerp_index]

                    first_lerp_point = first_lerp_point.unsqueeze(0).repeat(
                        interval_length, 1
                    )
                    second_lerp_point = second_lerp_point.unsqueeze(0).repeat(
                        interval_length, 1
                    )
                    alphas = (
                        torch.arange(0, interval_length, device=subpixel_points.device)
                        / interval_length
                    )

                    lerped_points = cv.lerp(
                        first_lerp_point, second_lerp_point, alphas[..., None]
                    )
                    subpixel_points[p, first_lerp_index : second_lerp_index + 1] = (
                        lerped_points
                    )

            subpixel_points = smooth_points2(subpixel_points.permute(1, 0, 2)).permute(
                1, 0, 2
            )
            # Mask out wrongly lerped points
            # Mask out points that needed to be lerped in subpixel_points
            subpixel_points *= (interpolation_mask | direct_mask)[..., None]

            # Add masked points to subpixel_points
            optimized_points = subpixel_points[:, :, [1, 0]].permute(1, 0, 2)
            # optimized_points = smooth_points(optimized_points)

            end_event.record()
            torch.cuda.synchronize()
            elapsed_time_ms = start_event.elapsed_time(end_event)
            print(f"Elapsed time: {elapsed_time_ms:.3f} ms for video {video.shape}")

            # Filter points that fall into the glottal region
            optimized_points = filter_points_on_glottis(
                optimized_points.cpu(), feature_estimator.glottisSegmentations().cpu()
            )

            optimized_points = filter_points_on_glottis(
                optimized_points.cpu(),
                (
                    1
                    - (
                        feature_estimator.glottisSegmentations()
                        | feature_estimator.vocalfoldSegmentations()
                    ).cpu()
                ),
            )

            # # And lie below a certain intensity
            # optimized_points = filter_points_by_intensity(
            #     optimized_points.cpu(),
            #     video.cpu(),
            #     intensity_threshold=self._min_point_intensity,
            # )

            return optimized_points






# TODO: Point Trackert base class I guess.
class SiliconePointTracker(PointTrackerBase):
    def __init__(self, distance_threshold: float = 1.5, min_point_intensity: int = 30, device="cpu"):
        super().__init__(distance_threshold, min_point_intensity, device)

    def track_points(self, video: torch.tensor, feature_estimator: feature_estimation.FeatureEstimator) -> List[torch.tensor]:
        # Get closed glottis frames from gaw.
        gaw: torch.tensor = feature_estimator.glottalAreaWaveform()
        gaw = cv.gaussian_smooth_1d(gaw, kernel_size=5, sigma=2.0)
        minima_indices, values = cv.find_local_minima_1d(gaw)
        minima_indices = minima_indices[values < gaw.median()]
        minima_indices = minima_indices.tolist()

        # Find laser points in frames, where glottis is minimal
        laserpoints_when_glottis_closed: List[torch.tensor] = [feature_estimator.laserpointPositions()[minima_index].float() for minima_index in minima_indices]

        # Compute temporal nearest neighbors in frames of closed glottis
        nearest_neighbors = cv.compute_point_estimates_from_nearest_neighbors(laserpoints_when_glottis_closed)

        # Interpolate from neighbors
        glottal_maxima_list: List[int] = minima_indices
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
        crops, y_windows, x_windows = cv.extract_windows_from_batch(video, indexed_point_positions, device=video.device)
        per_crop_max = crops.amax([-1, -2], keepdim=True)
        per_crop_min = crops.amin([-1, -2], keepdim=True)

        normalized_crops = (crops - per_crop_min) / (per_crop_max - per_crop_min)

        point_predictions = per_frame_point_position_estimates
            # 0.1 Reshape points, classes and crops into per frame segments, such that we can easily extract a timeseries.
        # I.e. shape is after this: NUM_POINTS x NUM_FRAMES x ...
        point_predictions = point_predictions.permute(1, 0, 2)
        y_windows = y_windows.reshape(
            video.shape[0], per_frame_point_position_estimates.shape[1], crops.shape[-2], crops.shape[-1]
        ).permute(1, 0, 2, 3)
        x_windows = x_windows.reshape(
            video.shape[0], per_frame_point_position_estimates.shape[1], crops.shape[-2], crops.shape[-1]
        ).permute(1, 0, 2, 3)
        crops = crops.reshape(
            video.shape[0], per_frame_point_position_estimates.shape[1], crops.shape[-2], crops.shape[-1]
        ).permute(1, 0, 2, 3)

        # Iterate over every point and class as well as their respective crops
        optimized_points = torch.zeros_like(point_predictions) * torch.nan
        optimized_points_on_crops = torch.zeros_like(point_predictions) * torch.nan
        for points_index, (points, crop) in tqdm(enumerate(
            zip(point_predictions, crops)
        )):

            # Compute sub-pixel position for each point labeled as visible (V)
            for frame_index in range(video.shape[0]):
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
        

        # convert points to frame x num_points x 2 [Y,X]
        optimized_points = optimized_points[:, :, [1, 0]].permute(1, 0, 2)

        # Filter points that fall into the glottal region
        optimized_points = filter_points_by_distance(optimized_points, self._distance_threshold)
        optimized_points = filter_points_on_glottis(optimized_points, feature_estimator.glottisSegmentations())
        optimized_points = filter_points_by_intensity(optimized_points, video, intensity_threshold=self._min_point_intensity)

        return optimized_points



def smooth_points(points: torch.tensor) -> torch.tensor:
    # We know that only values are nan, that lie at the border of the frames.
    # So we can easily convolve with a gaussian kernel

    # Define a Gaussian kernel
    def gaussian_kernel(size, sigma):
        x = torch.arange(-size // 2 + 1, size // 2 + 1)
        kernel = torch.exp(-0.5 * (x / sigma) ** 2)
        kernel /= kernel.sum()  # Normalize
        return kernel

    kernel_size = 5
    sigma = 1.0
    kernel = gaussian_kernel(kernel_size, sigma).view(1, 1, -1)
    kernel = kernel.to(points.device)

    for point_over_time in points:
        is_not_nan = ~torch.isnan(point_over_time[:, 0])
        non_nan_points = point_over_time[is_not_nan]

        if len(non_nan_points) < 10:
            continue

        a = non_nan_points.permute(1, 0).unsqueeze(1)
        padded_points = torch.nn.functional.pad(
            a, (kernel_size // 2, kernel_size // 2), "replicate"
        )
        smoothed_points = F.conv1d(padded_points, kernel.float(), padding=0)
        smoothed_points = smoothed_points.squeeze(1).permute(1, 0)
        point_over_time[is_not_nan] = smoothed_points

    return points


def filter_points_by_distance(point_predictions: torch.tensor, maximum_distance) -> torch.tensor:
    """
    points: Tensor of shape [FRAMES, NUM_POINTS, 2]
    threshold: float, maximum allowed distance between consecutive frame points

    Returns a tensor of the same shape, with points that move too far set to NaN in later frames.
    """
    points = point_predictions.clone()
    N, M, _ = points.shape

    # Go in forward direction
    for t in range(1, N):
        # Compute distances between corresponding points in frame t and t-1
        dist = torch.norm(point_predictions[t] - point_predictions[t - 1], dim=1)  # [M]
        dist = torch.nan_to_num(dist, 10000.0)
        
        # Mask: True if point moved too far
        invalid_mask = dist > maximum_distance
        
        # Set those points to NaN in frame t
        points[t, invalid_mask] = float('nan')

    # Go in backward direction.
    # We could do this in a single for loop
    for t in reversed(range(N - 1)):
        # Compute distances between corresponding points in frame t and t+1
        dist = torch.norm(point_predictions[t] - point_predictions[t + 1], dim=1)  # [M]
        dist = torch.nan_to_num(dist, 10000.0)

        # Mask where the distance exceeds the threshold
        invalid_mask = dist > maximum_distance

        # Set to NaN in frame t (the earlier one)
        points[t, invalid_mask] = float('nan')

    return points    


def filter_points_on_glottis(
    point_predictions: torch.tensor, vocalfold_segmentations: torch.tensor, dilate_by: int = 2) -> torch.tensor:
    
    if dilate_by > 0:
        vocalfold_segmentations = vocalfold_segmentations.clone()
        kernel = torch.ones((1, 1, 2*dilate_by + 1, 2*dilate_by + 1,), device=vocalfold_segmentations.device)
        vocalfold_segmentations = kornia.morphology.dilation(vocalfold_segmentations.unsqueeze(0).float(), torch.ones((2*dilate_by + 1, 2*dilate_by + 1,)).to(vocalfold_segmentations.device))
        vocalfold_segmentations = vocalfold_segmentations.to(torch.uint8).squeeze()


    # Convert all nans to 0
    filtered_points = torch.nan_to_num(point_predictions, 0)

    # Floor points and cast such that we have pixel coordinates
    point_indices = torch.floor(filtered_points).long()
        
    for frame_index, (points_in_frame, segmentation) in enumerate(
        zip(point_indices, vocalfold_segmentations)
    ):
        hits = segmentation[points_in_frame[:, 0], points_in_frame[:, 1]]
        hits = (hits == 0) * 1
        filtered_points[frame_index] *= hits[:, None]

    filtered_points[filtered_points == 0] = torch.nan

    return filtered_points



def filter_points_by_intensity(
    point_predictions: torch.tensor, video: torch.tensor, intensity_threshold: int = 20) -> torch.tensor:
    # Convert all nans to 0
    filtered_points = torch.nan_to_num(point_predictions, 0)

    # Floor points and cast such that we have pixel coordinates
    point_indices = torch.floor(filtered_points).long()

    for frame_index, (points_in_frame, frame) in enumerate(
        zip(point_indices, video)
    ):
        hits = frame[points_in_frame[:, 0], points_in_frame[:, 1]]
        hits = (hits > intensity_threshold) * 1
        filtered_points[frame_index] *= hits[:, None]

    filtered_points[filtered_points == 0] = torch.nan

    return filtered_points