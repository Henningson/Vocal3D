import re
from typing import List

import kornia
import torch
import torch.nn.functional as F


def lerp(v0, v1, t):
    return (1 - t) * v0 + t * v1

def get_segmentation_bounds(mask):
    """
    Get the bounding box of a binary segmentation mask using torch.nonzero().
    
    Args:
        mask (torch.Tensor): Binary mask of shape (H, W)
        
    Returns:
        (min_y, min_x), (max_y, max_x): Tuple of top-left and bottom-right coords
    """
    nonzero_indices = torch.nonzero(mask, as_tuple=False)
    
    if nonzero_indices.size(0) == 0:
        return None  # or raise an exception if preferred
    
    min_yx = torch.min(nonzero_indices, dim=0).values
    max_yx = torch.max(nonzero_indices, dim=0).values

    return tuple(min_yx.tolist()), tuple(max_yx.tolist())


def gaussian_smooth_1d(data, kernel_size=5, sigma=1.0):
    # Create a 1D Gaussian kernel
    half_size = kernel_size // 2
    x = torch.arange(-half_size, half_size + 1, dtype=torch.float32, device=data.device)
    gauss_kernel = torch.exp(-0.5 * (x / sigma)**2)
    gauss_kernel /= gauss_kernel.sum()  # Normalize

    # Reshape for conv1d: (out_channels, in_channels, kernel_size)
    gauss_kernel = gauss_kernel.view(1, 1, -1)

    # Add batch and channel dimensions: (batch_size=1, channels=1, width=n)
    data = data.view(1, 1, -1)

    # Pad the input to keep size consistent (reflect avoids edge effects)
    padding = kernel_size // 2
    smoothed = F.conv1d(data.float(), gauss_kernel, padding='same')

    return smoothed.view(-1)  # Remove batch and channel dimensions


def find_local_maxima_1d(tensor: torch.tensor):
    # Make sure it's a 1D tensor
    assert tensor.ndim == 1, "Input must be a 1D tensor"

    # Compare each element to its left and right neighbors
    left = tensor[:-2]
    center = tensor[1:-1]
    right = tensor[2:]

    # A local max is greater than both neighbors
    maxima_mask = (center > left) & (center > right)

    # Indices of the local maxima (offset by +1 because we sliced)
    maxima_indices = torch.nonzero(maxima_mask).squeeze() + 1

    return maxima_indices, tensor[maxima_indices]


def find_local_minima_1d(tensor: torch.tensor):
    # Make sure it's a 1D tensor
    assert tensor.ndim == 1, "Input must be a 1D tensor"

    # Compare each element to its left and right neighbors
    left = tensor[:-2]
    center = tensor[1:-1]
    right = tensor[2:]

    # A local max is greater than both neighbors
    minima_mask = (center < left) & (center < right)

    # Indices of the local maxima (offset by +1 because we sliced)
    minima_indices = torch.nonzero(minima_mask).squeeze() + 1

    return minima_indices, tensor[minima_indices]

def find_local_maxima_2d(tensor: torch.tensor) -> torch.tensor:
    return torch.tensor()


def pairwise_distances(x, y):
    # x: (n, 2), y: (m, 2)
    # Returns: (n, m) matrix of distances
    x_sq = (x ** 2).sum(dim=1, keepdim=True)      # (n, 1)
    y_sq = (y ** 2).sum(dim=1).unsqueeze(0)       # (1, m)
    xy = x @ y.t()                                # (n, m)
    dists = x_sq - 2 * xy + y_sq
    return torch.sqrt(torch.clamp(dists, min=1e-12))


def compute_point_estimates_from_nearest_neighbors(closed_glottis_points: List[torch.tensor]):
    """
    For a list of 2D tensors [A, B, C, ...], finds nearest neighbors from A to B, B to C, etc.
    Returns a list of (indices, distances) tuples.
    Each indices[i] is a LongTensor of size (len(tensors[i]),) giving the index of the nearest neighbor in tensors[i+1]
    """


    distance_threshold = 3*3
    max_points: int = 0
    max_points_index: int = 0
    # Find frame with maximum amount of points
    for index, points in enumerate(closed_glottis_points):
        non_nan_points: int = (~torch.isnan(points).any(dim=-1)).sum()
        if max_points < non_nan_points:
            max_points = non_nan_points
            max_points_index = index

    final_point_tensor: torch.tensor = torch.ones([len(closed_glottis_points), max_points, 2], device=closed_glottis_points[0].device) * torch.nan
    results = []

    # Copy points of frame with most point instances
    index = 0
    for point in closed_glottis_points[max_points_index]:
        if torch.isnan(point.any()):
            continue

        final_point_tensor[max_points_index, index] = point
        index += 1

    # Start forward search
    for i in range(max_points_index, final_point_tensor.shape[0] - 1, 1):
        for index, point in enumerate(final_point_tensor[i]):
            # Given the frame with the maximal amount of points, search for corresponding points in future frames that are closer than a specific threshold.

            point_dists = (point.unsqueeze(0) - closed_glottis_points[i+1]).pow(2).sum(dim=-1)
            min_dist = point_dists.min()
            min_index = point_dists.argmin()

            if min_dist < distance_threshold: # 5*5px since I remove the sqrt from the euclidean distance computation.
                final_point_tensor[i+1, index] = closed_glottis_points[i+1][min_index]
            else:
                # If nothing could be found, copy previous point
                final_point_tensor[i+1, index] = final_point_tensor[i, index]

    # Start backward search
    for i in range(max_points_index, 0, -1):
        for index, point in enumerate(final_point_tensor[i]):
            # Given the frame with the maximal amount of points, search for corresponding points in future frames that are closer than a specific threshold.

            point_dists = (point.unsqueeze(0) - closed_glottis_points[i-1]).pow(2).sum(dim=-1)
            min_dist = point_dists.min()
            min_index = point_dists.argmin()

            if min_dist < distance_threshold: # 5*5px since I remove the sqrt from the euclidean distance computation.
                final_point_tensor[i-1, index] = closed_glottis_points[i-1][min_index]
            else:
                # If nothing could be found, copy previous point
                final_point_tensor[i-1, index] = final_point_tensor[i, index]


    # Remove rows with nans
    keep_vec = ~torch.isnan(final_point_tensor).all(dim=0).all(dim=1)

    final_point_tensor = final_point_tensor[:, keep_vec]

    return final_point_tensor


def fill_nans_in_point_timeseries(neighbors_over_time: torch.tensor) -> torch.tensor:
    a = 1
    return None

def interpolate_from_neighbors(indices: List[int], neighbors_over_time: torch.tensor) -> torch.tensor:
    new_tensor: torch.tensor = torch.zeros((indices[-1], neighbors_over_time.shape[1], 2), device=neighbors_over_time.device)
    current_frame: int = 0

    for i in range(neighbors_over_time.shape[0] - 1):
        amount_of_frames_to_interpolate: int = indices[i+1] - indices[i]
        for j in range(amount_of_frames_to_interpolate):
            lerp_factor = j / amount_of_frames_to_interpolate
            lerped_points = lerp(neighbors_over_time[i], neighbors_over_time[i+1], lerp_factor)
            new_tensor[current_frame] = lerped_points

            current_frame += 1

    return new_tensor




def moment_method(images: torch.tensor) -> torch.tensor:
    """
    Computes the subpixel-accurate centroid of a 2D Gaussian distribution
    in pixel space using the moment method.
    """

    # Get the pixel gridrows = torch.arange(size[0])
    cols = torch.arange(images.shape[-1], device=images.device)
    rows = torch.arange(images.shape[-2], device=images.device)

    # Create a grid
    y, x = torch.meshgrid(rows, cols, indexing="ij")

    # Total intensity (0th moment)
    total_intensity = torch.sum(images, dim=(-2, -1), keepdims=True)

    # First moments for x and y
    x_moment = torch.sum(x * images, dim=(-2, -1), keepdims=True)
    y_moment = torch.sum(y * images, dim=(-2, -1), keepdims=True)

    # Subpixel centroid
    x_centroids = x_moment / total_intensity
    y_centroids = y_moment / total_intensity

    return torch.concat([x_centroids + 0.5, y_centroids + 0.5], dim=-1)[:, 0, :]


def fill_nan_border_values(points: torch.tensor) -> torch.tensor:
    for point_index, point_over_time in enumerate(points):
        nan_count_start = 0
        nan_count_end = 0

        nan_mask = torch.isnan(point_over_time[:, 0])

        for val in nan_mask:
            # Count nans at beginning of sequence
            if val:
                nan_count_start += 1
            else:
                break

        for val in nan_mask.flip(0):
            # Count nans at end of sequence
            if val:
                nan_count_end += 1
            else:
                break

        if (
            nan_count_start == nan_count_end
            and nan_count_start == point_over_time.shape[0]
        ):
            continue

        if nan_count_end != 0:
            point_over_time = point_over_time[nan_count_start:-nan_count_end]
        else:
            point_over_time = point_over_time[nan_count_start:]

        point_over_time = torch.nn.functional.pad(
            point_over_time.permute(1, 0), (nan_count_start, nan_count_end), "replicate"
        ).permute(1, 0)
        points[point_index] = point_over_time

    return points


def fill_nan_border_values_2d(points: torch.tensor) -> torch.tensor:
    # Points should be in FRAMELENGTH x HEIGHT x WIDTH x 2
    FRAMES, HEIGHT, WIDTH, DIMENSIONS = points.shape
    for y in range(HEIGHT):
        for x in range(WIDTH):
            nan_count_start = 0
            nan_count_end = 0

            point_over_time = points[:, y, x, :]
            nan_mask = torch.isnan(point_over_time[:, 0])

            for val in nan_mask:
                # Count nans at beginning of sequence
                if val:
                    nan_count_start += 1
                else:
                    break

            for val in nan_mask.flip(0):
                # Count nans at end of sequence
                if val:
                    nan_count_end += 1
                else:
                    break

            if nan_count_start == nan_count_end and nan_count_start == FRAMES:
                continue

            if nan_count_end != 0:
                point_over_time = point_over_time[nan_count_start:-nan_count_end]
            else:
                point_over_time = point_over_time[nan_count_start:]

            point_over_time = torch.nn.functional.pad(
                point_over_time.permute(1, 0),
                (nan_count_start, nan_count_end),
                "replicate",
            ).permute(1, 0)
            points[:, y, x, :] = point_over_time

    return points


def interpolate_nans_2d(points: torch.tensor) -> torch.tensor:
    # Points should be in FRAMELENGTH x HEIGHT x WIDTH x 2
    FRAMES, HEIGHT, WIDTH, DIMENSIONS = points.shape
    for y in range(HEIGHT):
        for x in range(WIDTH):
            point_over_time = points[:, y, x, :]
            nan_mask = torch.isnan(point_over_time[:, 0]) * 1
            compute_string = "".join(map(str, nan_mask.squeeze().tolist()))
            # Replace 0s with V for visible
            if nan_mask.sum() == FRAMES:
                continue

            for frame_index, label in enumerate(compute_string):
                if label == "0":
                    continue

                prev_v_index = compute_string.rfind("0", 0, frame_index)
                next_v_index = compute_string.find("0", frame_index + 1)

                lerp_alpha = (frame_index - prev_v_index) / (
                    next_v_index - prev_v_index
                )
                point_a = point_over_time[prev_v_index]
                point_b = point_over_time[next_v_index]
                lerped_point = VFLabel.utils.transforms.lerp(
                    point_a, point_b, lerp_alpha
                )

                points[frame_index, y, x] = lerped_point

    return points


def compute_segmentation_outline(segmentation: torch.tensor, kernel_size=3, border_type="inner") -> List[torch.tensor]:
    """
    Extracts the border of a binary segmentation mask using Kornia.

    Args:
        mask: (B, 1, H, W) tensor of binary masks (0 or 1)
        kernel_size: size of the morphological kernel (should be odd)
        border_type: 'both' for dilation - erosion, 'inner' for mask - erosion

    Returns:
        border: (B, 1, H, W) tensor of borders
    """
    kernel = torch.ones((kernel_size, kernel_size), device=segmentation.device)

    dilated = kornia.morphology.dilation(segmentation.unsqueeze(0).unsqueeze(0).float(), kernel).squeeze()
    eroded = kornia.morphology.erosion(segmentation.unsqueeze(0).unsqueeze(0).float(), kernel).squeeze()

    if border_type == "both":
        border = dilated - eroded
    elif border_type == "inner":
        border = segmentation - eroded
    elif border_type == "outer":
        border = dilated - segmentation
    else:
        raise ValueError("border_type must be 'both', 'inner', or 'outer'")

    # Ensure binary result
    return (border > 0).float()


# Indices [n] Tensor
# Image size integer
# pad is window_size//2
def windows_out_of_bounds(indices, image_size, pad):
    # Move big ol indices
    indices = torch.where(
        indices + pad >= image_size,
        indices + ((image_size - pad) - indices) - 1,
        indices,
    )

    indices = torch.where(indices - pad < 0, indices + (pad - indices), indices)

    return indices


def extract_windows(frame, points, window_size=7):
    x = points[:, 1].long()
    y = points[:, 0].long()

    
    y = windows_out_of_bounds(y, frame.shape[0], window_size // 2)
    x = windows_out_of_bounds(x, frame.shape[1], window_size // 2)

    sub = torch.linspace(-window_size // 2 + 1, window_size // 2, window_size)
    x_sub, y_sub = torch.meshgrid(sub, sub, indexing="xy")


    y_windows = y.unsqueeze(1).unsqueeze(1).repeat(1, window_size, window_size)
    x_windows = x.unsqueeze(1).unsqueeze(1).repeat(1, window_size, window_size)

    y_windows += y_sub.unsqueeze(0).long().to(frame.device)
    x_windows += x_sub.unsqueeze(0).long().to(frame.device)

    crops = frame[y_windows.reshape(-1, window_size**2), x_windows.reshape(-1, window_size**2)]
    return crops.reshape(points.shape[0], window_size, window_size)


# Indices: Tensor of size Nx3, like [[batch, y, x], ..]
# Batch: Image batch of size BATCH x X x Y
# Returns: Tensor of Size N x 3 x 3
def extract_windows_from_batch(batch, indices, window_size=7, device="cuda"):
    # Clean Windows, such that no image boundaries are hit

    batch_index = indices[:, 0].long()
    y = indices[:, 2].long()
    x = indices[:, 1].long()

    y = windows_out_of_bounds(y, batch.shape[1], window_size // 2)
    x = windows_out_of_bounds(x, batch.shape[2], window_size // 2)

    y_windows = y.unsqueeze(1).unsqueeze(1).repeat(1, window_size, window_size)
    x_windows = x.unsqueeze(1).unsqueeze(1).repeat(1, window_size, window_size)

    sub = torch.linspace(-window_size // 2 + 1, window_size // 2, window_size)
    x_sub, y_sub = torch.meshgrid(sub, sub, indexing="xy")

    y_windows += y_sub.unsqueeze(0).long().to(device)
    x_windows += x_sub.unsqueeze(0).long().to(device)

    # Catching windows
    windows = batch[
        batch_index.unsqueeze(-1),
        y_windows.reshape(-1, window_size * window_size),
        x_windows.reshape(-1, window_size * window_size),
    ]

    return windows.reshape(-1, window_size, window_size), y_windows, x_windows