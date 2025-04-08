import cv
import kornia
import numpy as np
import torch


def get_basis(x, y):
    """Return the fit basis polynomials: 1, x, x^2, ..., xy, x^2y, ... etc."""
    basis = []
    for i in range(3):
        for j in range(3 - i):
            basis.append(x**j * y**i)
    return basis


def get_split_indices(nonzeros_indexing, device="cuda"):
    return (
        (
            nonzeros_indexing
            - torch.concat([torch.zeros(1).to(device), nonzeros_indexing])[0:-1]
        )
        .nonzero()
        .squeeze()
    )


def GuosBatchAnalytic(x, y, z):
    weights = z**2

    A = torch.stack(get_basis(x, y), dim=-1)
    A = (A.transpose(1, 2) * weights.unsqueeze(1)).transpose(1, 2)
    b = torch.log(z) * weights
    ATA = (A.transpose(1, 2) @ A).float()
    c = ATA.inverse() @ A.transpose(1, 2) @ b.unsqueeze(-1)
    c = c.squeeze(-1)
    return poly_to_gauss(c[:, [2, 5]], c[:, [1, 3]], c[:, [0, 4]])


def moment_method_torch(images: torch.tensor) -> torch.tensor:
    """
    Computes the subpixel-accurate centroid of a 2D Gaussian distribution
    in pixel space using the moment method.
    """

    # Get the pixel gridrows = torch.arange(size[0])
    cols, rows = torch.arange(images.shape[-1]), torch.arange(images.shape[-2])

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

    return torch.concatenate([x_centroids + 0.5, y_centroids + 0.5], dim=-1)[:, 0, :]


def moment_method(images: np.array) -> np.array:
    """
    Computes the subpixel-accurate centroid of a 2D Gaussian distribution
    in pixel space using the moment method.
    """

    # Get the pixel grid
    y, x = np.indices(images[0].shape)

    # Total intensity (0th moment)
    total_intensity = np.sum(images, axis=(-2, -1), keepdims=True)

    # First moments for x and y
    x_moment = np.sum(x * images, axis=(-2, -1), keepdims=True)
    y_moment = np.sum(y * images, axis=(-2, -1), keepdims=True)

    # Subpixel centroid
    x_centroids = x_moment / total_intensity
    y_centroids = y_moment / total_intensity

    return np.concatenate([x_centroids, y_centroids], axis=-1)[:, 0, :]


def poly_to_gauss(A, B, C):
    sigma = torch.sqrt(-1 / (2.0 * A))
    mu = B * sigma**2
    height = torch.exp(C + 0.5 * mu**2 / sigma**2)
    return sigma, mu, height


class LSQLocalization:
    def __init__(
        self,
        order=2,
        gauss_window=5,
        local_maxima_window=11,
        heatmapaxis=1,
        threshold=0.3,
        device="cuda",
    ):
        super(LSQLocalization, self).__init__()
        self.order = order

        self.local_maxima_window_size = local_maxima_window
        self.gauss_window_size = gauss_window
        self.pad = self.gauss_window_size // 2

        self.heatmapaxis = heatmapaxis
        self.gauss_blur_sigma = 1.5
        self.threshold = threshold

        self.kernel = torch.ones(
            self.local_maxima_window_size, self.local_maxima_window_size
        ).to(device)
        self.kernel[
            self.local_maxima_window_size // 2, self.local_maxima_window_size // 2
        ] = 0

        sub = torch.linspace(
            -self.gauss_window_size // 2 + 1,
            self.gauss_window_size // 2,
            self.gauss_window_size,
        )
        x_sub, y_sub = torch.meshgrid(sub, sub, indexing="xy")
        self.x_sub = x_sub.unsqueeze(0).to(device)
        self.y_sub = y_sub.unsqueeze(0).to(device)

        self.device = device

    def test_with_image(self, image, x, segmentation=None):
        heat = x[:, self.heatmapaxis, :, :].clone()
        heat = heat.unsqueeze(1)

        # Generate thresholded image
        threshed_heat = (heat > self.threshold) * heat
        threshed_heat = kornia.filters.gaussian_blur2d(
            threshed_heat,
            self.kernel.shape,
            [self.kernel.shape[0] / 4, self.kernel.shape[0] / 4],
        )

        # Use dilation filtering to generate local maxima and squeeze first dimension
        dilated_heat = kornia.morphology.dilation(threshed_heat, self.kernel)
        local_maxima = threshed_heat > dilated_heat
        local_maxima = local_maxima[:, 0, :, :]

        if segmentation is not None:
            local_maxima = local_maxima * segmentation

        # Find local maxima and indices at which we need to split the data
        maxima_indices = local_maxima.nonzero()
        split_indices = get_split_indices(
            maxima_indices[:, 0], device=self.device
        ).tolist()

        # Extract windows around the local maxima
        intensities, y_windows, x_windows = cv.extractWindow(
            image, maxima_indices, self.gauss_window_size, device=self.device
        )

        # Reformat [-2, -1, 0, ..] tensors for x-y-indexing
        reformat_x = self.x_sub.repeat(x_windows.size(0), 1, 1).reshape(
            -1, self.gauss_window_size**2
        )
        reformat_y = self.y_sub.repeat(y_windows.size(0), 1, 1).reshape(
            -1, self.gauss_window_size**2
        )

        # Use Guos Weighted Gaussian Fitting algorithm based on the intensities of the non-thresholded image
        sigma, mu, amplitude = GuosBatchAnalytic(
            reformat_x, reformat_y, intensities.reshape(-1, self.gauss_window_size**2)
        )

        # Add found mus to the initial quantized local maxima
        mu = maxima_indices[:, 1:] + mu[:, [1, 0]]

        # Split the tensors and return lists of sigma, mus, and the amplitudes per batch
        return (
            torch.tensor_split(sigma, split_indices),
            torch.tensor_split(mu, split_indices),
            torch.tensor_split(amplitude, split_indices),
        )

    def test(self, x, segmentation=None):
        heat = x[:, self.heatmapaxis, :, :].clone()
        heat = heat.unsqueeze(1)

        # Generate thresholded image
        threshed_heat = (heat > self.threshold) * heat
        threshed_heat = kornia.filters.gaussian_blur2d(
            threshed_heat,
            self.kernel.shape,
            [self.kernel.shape[0] / 4, self.kernel.shape[0] / 4],
        )

        # Use dilation filtering to generate local maxima and squeeze first dimension
        dilated_heat = kornia.morphology.dilation(threshed_heat, self.kernel)
        local_maxima = threshed_heat > dilated_heat
        local_maxima = local_maxima[:, 0, :, :]

        if segmentation is not None:
            local_maxima = local_maxima * segmentation

        # Find local maxima and indices at which we need to split the data
        maxima_indices = local_maxima.nonzero()
        split_indices = get_split_indices(
            maxima_indices[:, 0], device=self.device
        ).tolist()

        # Extract windows around the local maxima
        intensities, y_windows, x_windows = cv.extractWindow(
            heat[:, 0, :, :], maxima_indices, self.gauss_window_size, device=self.device
        )

        # Reformat [-2, -1, 0, ..] tensors for x-y-indexing
        reformat_x = self.x_sub.repeat(x_windows.size(0), 1, 1).reshape(
            -1, self.gauss_window_size**2
        )
        reformat_y = self.y_sub.repeat(y_windows.size(0), 1, 1).reshape(
            -1, self.gauss_window_size**2
        )

        # Use Guos Weighted Gaussian Fitting algorithm based on the intensities of the non-thresholded image
        sigma, mu, amplitude = GuosBatchAnalytic(
            reformat_x, reformat_y, intensities.reshape(-1, self.gauss_window_size**2)
        )

        # Add found mus to the initial quantized local maxima
        mu = maxima_indices[:, 1:] + mu[:, [1, 0]]

        # Split the tensors and return lists of sigma, mus, and the amplitudes per batch
        return (
            torch.tensor_split(sigma, split_indices),
            torch.tensor_split(mu, split_indices),
            torch.tensor_split(amplitude, split_indices),
        )


if __name__ == "__main__":
    # Create a sample 2D Gaussian distribution
    size = 11
    x = np.linspace(-5, 5, size)
    y = np.linspace(-5, 5, size)
    x_grid, y_grid = np.meshgrid(x, y)
    gaussian = np.exp(-(x_grid**2 + y_grid**2) / (2 * 1.5**2))  # Sigma = 1.5

    # Add a slight offset for testing subpixel accuracy
    gaussian_shifted = np.roll(np.roll(gaussian, 2, axis=0), 3, axis=1)

    gaussian_shifted = np.repeat(np.expand_dims(gaussian_shifted, 0), 5, 0)

    # Find the subpixel centroid
    centroid = moment_method(gaussian_shifted)
    print(f"Subpixel centroid: {centroid}")
