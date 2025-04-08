import math
import os
from typing import List, Tuple

import cv
import kornia
import numpy as np
import point_extraction
import torch
import torch.nn.functional as F


class FeatureEstimator:
    def __init__(self):
        self._glottal_midlines: Tuple[List[torch.tensor], List[torch.tensor]] = None
        self._glottal_outline_images: torch.tensor = None
        self._glottal_outlines: List[torch.tensor] = None

        self._vocalfold_segmentations: torch.tensor = None
        self._glottis_segmentations: torch.tensor = None

        self._laserpoint_segmentations: torch.tensor = None
        self._laserpoint_positions: List[torch.tensor] = None

    def glottalMidlines(self) -> Tuple[List[torch.tensor], List[torch.tensor]]:
        return self._glottal_midlines
    
    def glottalOutlines(self) -> List[torch.tensor]:
        return self._glottal_outlines

    def glottalOutlinesAsImages(self) -> torch.tensor:
        return self._glottal_outline_images

    def vocalfoldSegmentations(self) -> torch.tensor:
        return self._vocalfold_segmentations
    
    def laserpointPositions(self) -> List[torch.tensor]:
        return self._laserpoint_positions
    
    def laserpointSegmentations(self) -> torch.tensor:
        # Should generate the segmentations here on the fly, as to not inhibit the reconstruction process.
        return self._laserpoint_segmentations
    
    def glottisSegmentations(self) -> torch.tensor:
        return self._glottis_segmentations
    
    def glottalAreaWaveform(self) -> torch.tensor:
        return self._glottis_segmentations.sum(dim=(1,2))

    def compute_glottal_midline(self, segmentation: torch.tensor) -> None:
        white_points = torch.argwhere(segmentation > 0)

        y = white_points[:, 1]
        x = white_points[:, 0]

        A = torch.vstack(torch.vstack([x, torch.ones(len(x))]).T)
        m, c = torch.linalg.lstsq(A, y, rcond=None)[0]

        upperPoint = torch.tensor([m * x.min() + c, x.min()])
        lowerPoint = torch.tensor([m * x.max() + c, x.max()])

        return upperPoint, lowerPoint

    def compute_features(self, video: torch.tensor) -> None:
        # TODO: Implement me!
        # Here, we need to compute the glottis segmentations, vocalfold_segmentations, laserpoint_positions, glottal midlines and outlines.
        # If your data is different, you need to override this part here!
        pass


class SiliconeFeatureEstimator(FeatureEstimator):
    def __init__(self):
        super().__init__()

    def compute_local_maxima(self, image, kernelsize=7):
        kernel = torch.ones((kernelsize, kernelsize), dtype=np.uint8)
        kernel[math.floor(kernelsize // 2), math.floor(kernelsize // 2)] = 0.0
        maxima = image > kornia.morphology.dilation(image, kernel)
        return maxima, maxima.nonzero()


    def compute_features(self, video: torch.tensor) -> None:
        self._glottis_segmentations = torch.zeros_like(video)
        self._vocalfold_segmentations = torch.zeros_like(video)
        self._laserpoint_segmentations = torch.zeros_like(video)
        
        self._glottal_outline_images = torch.zeros_like(video)
        self._glottal_outlines = []

        self._laserpoint_positions = []

        self._glottal_midlines = [[], []]


        for index, image in enumerate(video):
            glottis_segmentation = torch.where(image == 0, 1, 0).astype(np.uint8)

            # This is in yx
            top_left, bottom_right = cv.get_segmentation_bounds(glottis_segmentation)
            y_min = top_left[0]
            x_min = top_left[1]

            y_max = bottom_right[0]
            x_max = bottom_right[1]

            glottal_width = x_max - x_min
            glottal_height = y_max - y_min

            mid_point = [x_min + glottal_width // 2, y_min + glottal_height // 2]

            roi_width = 2 * (glottal_height)
            roi_height = int(glottal_height * 1.4)
            
            new_x_min = mid_point[0] - roi_width // 2
            new_y_min = mid_point[1] - roi_height // 2
            new_x_max = mid_point[0] + roi_width // 2
            new_y_max = mid_point[1] + roi_width // 2
            
            new_x_min = 0 if (new_x_min < 0) else new_x_min
            new_y_min = 0 if (new_y_min < 0) else new_y_min
            new_x_max = image.shape[1] - 2 if (new_x_max > image.shape[1] - 1) else new_x_max
            new_y_max = image.shape[0] - 2 if (new_y_max > image.shape[0] - 1) else new_y_max
            
            self._vocalfold_segmentations[index, new_y_min:new_y_max, new_x_min:new_x_max] = 1
            self._glottis_segmentations[index] = glottis_segmentation
            laserpoint_image, laserpoint_positions = self.compute_local_maxima(image * self._vocalfold_segmentations[index])

            self._laserpoint_positions.append(laserpoint_positions)
            self._laserpoint_segmentations = laserpoint_image

            upper_point, lower_point = self.compute_glottal_midline(self._glottis_segmentations[index])
            self._glottal_midlines[0].append(upper_point)
            self._glottal_midlines[1].append(lower_point)

            glottal_outline_image = cv.compute_segmentation_outline(glottis_segmentation)
            self._glottal_outline_images[index] = glottal_outline_image
            self._glottal_outlines.append(torch.nonzero(glottal_outline_image))

        return self._glottis_segmentations, self._glottal_midlines, self._glottal_outlines, self._vocalfold_segmentations, self._laserpoint_positions, self._laserpoint_segmentations, self.glottalAreaWaveform()




class NeuralFeatureEstimator(FeatureEstimator):
    def __init__(self, encoder: str, device: torch.cuda.device):
        super.__init__()

        self.point_localizer = point_extraction.LSQLocalization(local_maxima_window = 7)

        model_path = os.path.join("assets", "models", f"glottis_{encoder}.pth.tar")
        self._model = smp.Unet(
            encoder_name=encoder,  # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
            encoder_weights="imagenet",  # use `imagenet` pre-trained weights for encoder initialization
            in_channels=3,  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
            classes=1,
        ).to(device)
        state_dict = torch.load(model_path, weights_only=True)

        if "optimizer" in state_dict:
            del state_dict["optimizer"]

        self._model.load_state_dict(state_dict)

    def compute_features(self, video: torch.tensor) -> None:
        self._glottis_segmentations = torch.zeros_like(video)
        self._vocalfold_segmentations = torch.zeros_like(video)
        self._laserpoint_segmentations = torch.zeros_like(video)
        
        self._glottal_outline_images = torch.zeros_like(video)
        self._glottal_outlines = []

        self._laserpoint_positions = []

        self._glottal_midlines = [[], []]


        for index, image in enumerate(video):
            image = image.unsqueeze(0).float() / 255
            pred_seg = self._model(image).squeeze()
            softmaxed = torch.softmax(pred_seg, dim=-1)
            labels = softmaxed.argmax(dim=-1)

            self._laserpoint_segmentations[index] = (labels == 3) * 1
            segmentation = torch.where(labels == 3, 2, segmentation)
            vocalfold_segmentation = (labels == 2) * 1
            glottis_segmentation = (labels == 1) * 1

            self._vocalfold_segmentations[index] = vocalfold_segmentation
            self._glottis_segmentations[index] = glottis_segmentation
            laserpoints = self.point_localizer.test(softmaxed)
            self._laserpoint_positions.append(laserpoints)

            upper_point, lower_point = self.compute_glottal_midline(self._glottis_segmentations[index])
            self._glottal_midlines[0].append(upper_point)
            self._glottal_midlines[1].append(lower_point)

            glottal_outline_image = cv.compute_segmentation_outline(glottis_segmentation)
            self._glottal_outline_images[index] = glottal_outline_image
            self._glottal_outlines.append(torch.nonzero(glottal_outline_image))
            
        return self._glottis_segmentations, self._glottal_midlines, self._glottal_outlines, self._vocalfold_segmentations, self._laserpoint_positions, self._laserpoint_segmentations, self.glottalAreaWaveform()




class NeuralFeatureEstimator2(FeatureEstimator):
    def __init__(self, encoder: str, device: torch.cuda.device):
        super.__init__()

        self.point_localizer = point_extraction.LSQLocalization(local_maxima_window = 7, threshold = 0.8)

        model_path = os.path.join("assets", "models", f"glottis_{encoder}.pth.tar")
        self._model = smp.Unet(
            encoder_name=encoder,  # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
            encoder_weights="imagenet",  # use `imagenet` pre-trained weights for encoder initialization
            in_channels=3,  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
            classes=1,
        ).to(device)
        state_dict = torch.load(model_path, weights_only=True)

        if "optimizer" in state_dict:
            del state_dict["optimizer"]

        self._model.load_state_dict(state_dict)

    def compute_features(self, video: torch.tensor) -> None:
        self._glottis_segmentations = torch.zeros_like(video)
        self._vocalfold_segmentations = torch.zeros_like(video)
        self._laserpoint_segmentations = torch.zeros_like(video)
        
        self._glottal_outline_images = torch.zeros_like(video)
        self._glottal_outlines = []

        self._laserpoint_positions = []

        self._glottal_midlines = [[], []]


        for index, image in enumerate(video):
            image = image.unsqueeze(0).float() / 255
            pred_seg = self._model(image).squeeze()
            segmentation_channels = pred_seg[:, :, :-1]
            laserpoint_channel = pred_seg[:, :, -1]
            laserpoint_channel = (laserpoint_channel - laserpoint_channel.min()) / (laserpoint_channel.max() - laserpoint_channel.min())
            softmaxed = torch.softmax(segmentation_channels, dim=-1)
            labels = softmaxed.argmax(dim=-1)

            self._laserpoint_segmentations[index] = (labels == 3) * 1
            segmentation = torch.where(labels == 3, 2, segmentation)
            vocalfold_segmentation = (labels == 2) * 1
            glottis_segmentation = (labels == 1) * 1

            self._vocalfold_segmentations[index] = vocalfold_segmentation
            self._glottis_segmentations[index] = glottis_segmentation
            laserpoints = self.point_localizer.test(softmaxed)
            self._laserpoint_positions.append(laserpoints)

            upper_point, lower_point = self.compute_glottal_midline(self._glottis_segmentations[index])
            self._glottal_midlines[0].append(upper_point)
            self._glottal_midlines[1].append(lower_point)

            glottal_outline_image = cv.compute_segmentation_outline(glottis_segmentation)
            self._glottal_outline_images[index] = glottal_outline_image
            self._glottal_outlines.append(torch.nonzero(glottal_outline_image))

        return self._glottis_segmentations, self._glottal_midlines, self._glottal_outlines, self._vocalfold_segmentations, self._laserpoint_positions, self._laserpoint_segmentations, self.glottalAreaWaveform()