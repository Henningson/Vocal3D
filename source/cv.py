from typing import List

import kornia
import torch
import torch.nn.functional as F


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


def gaussian_smooth_1d(tensor, kernel_size=5, sigma=1.0):
    # Create a 1D Gaussian kernel
    half_size = kernel_size // 2
    x = torch.arange(-half_size, half_size + 1, dtype=torch.float32)
    gauss_kernel = torch.exp(-0.5 * (x / sigma)**2)
    gauss_kernel /= gauss_kernel.sum()  # Normalize

    # Reshape for conv1d: (out_channels, in_channels, kernel_size)
    gauss_kernel = gauss_kernel.view(1, 1, -1)

    # Add batch and channel dimensions: (batch_size=1, channels=1, width=n)
    tensor = tensor.view(1, 1, -1)

    # Pad the input to keep size consistent (reflect avoids edge effects)
    padding = kernel_size // 2
    smoothed = F.conv1d(tensor, gauss_kernel, padding=padding, padding_mode='reflect')

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


def find_nearest_neighbors_consecutive(tensors):
    """
    For a list of 2D tensors [A, B, C, ...], finds nearest neighbors from A to B, B to C, etc.
    Returns a list of (indices, distances) tuples.
    Each indices[i] is a LongTensor of size (len(tensors[i]),) giving the index of the nearest neighbor in tensors[i+1]
    """
    results = []
    for i in range(len(tensors) - 1):
        src = tensors[i]      # shape (n, 2)
        tgt = tensors[i + 1]  # shape (m, 2)
        dists = pairwise_distances(src, tgt)  # shape (n, m)
        nn_dists, nn_indices = torch.min(dists, dim=1)
        results.append((nn_indices, nn_dists))
    return results


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
    kernel = torch.ones((1, 1, kernel_size, kernel_size), device=segmentation.device)

    dilated = kornia.morphology.dilation(segmentation, kernel)
    eroded = kornia.morphology.erosion(segmentation, kernel)

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


# Indices: Tensor of size Nx3, like [[batch, y, x], ..]
# Batch: Image batch of size BATCH x X x Y
# Returns: Tensor of Size N x 3 x 3
def extractWindow(batch, indices, window_size=7, device="cuda"):
    # Clean Windows, such that no image boundaries are hit

    batch_index = indices[:, 0].int()
    y = indices[:, 2].floor().int()
    x = indices[:, 1].floor().int()

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