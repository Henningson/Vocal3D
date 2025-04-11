import math
import os
from typing import List, Tuple

import cv
import kornia
import models
import NeuralSegmentation
import numpy as np
import point_extraction
import torch
import torch.nn.functional as F
import torchvision


class FeatureEstimator:
    def __init__(self):
        self._flip_horizontal: bool = False
        self._glottal_midlines: List[Tuple[torch.tensor, torch.tensor]] = None
        self._glottal_outline_images: torch.tensor = None
        self._glottal_outlines: List[torch.tensor] = None

        self._vocalfold_segmentations: torch.tensor = None
        self._glottis_segmentations: torch.tensor = None

        self._laserpoint_segmentations: torch.tensor = None
        self._laserpoint_positions: List[torch.tensor] = None

        self._vocalfold_bounding_box: Tuple[torch.tensor, torch.tensor] = None

    def __len__(self) -> int:
        return self._glottis_segmentations.shape[0] if self._glottis_segmentations is not None else 0

    def glottalMidlines(self) -> Tuple[List[torch.tensor], List[torch.tensor]]:
        return self._glottal_midlines
    
    def glottalOutlines(self) -> List[torch.tensor]:
        return self._glottal_outlines

    def glottalOutlinesAsImages(self) -> torch.tensor:
        return self._glottal_outline_images

    def vocalfoldSegmentations(self) -> torch.tensor:
        return self._vocalfold_segmentations
    
    def vocalfoldBoundingBox(self) -> Tuple[torch.tensor, torch.tensor]:
        return self._vocalfold_bounding_box
    
    def laserpointPositions(self) -> List[torch.tensor]:
        return self._laserpoint_positions
    
    def laserpointSegmentations(self) -> torch.tensor:
        # Should generate the segmentations here on the fly, as to not inhibit the reconstruction process.
        return self._laserpoint_segmentations
    
    def glottisSegmentations(self) -> torch.tensor:
        return self._glottis_segmentations
    
    def glottalAreaWaveform(self) -> torch.tensor:
        return self._glottis_segmentations.sum(dim=(1,2))

    def compute_glottal_midline(self, segmentation: torch.tensor, flipped: bool = False) -> None:

        white_points = segmentation.T.nonzero() if flipped else segmentation.nonzero()
        
        if white_points.numel() == 0:
            return None, None

        y = white_points[:, 1]
        x = white_points[:, 0]

        A = torch.vstack([x, torch.ones(len(x), device=segmentation.device)])
        m, c = torch.linalg.lstsq(A.T, y.float()).solution

        upperPoint = torch.tensor([m * x.min() + c, x.min()], device=segmentation.device) if not flipped else torch.tensor([x.min(), m * x.min() + c], device=segmentation.device)
        lowerPoint = torch.tensor([m * x.max() + c, x.max()], device=segmentation.device) if not flipped else torch.tensor([x.max(), m * x.max() + c], device=segmentation.device)

        if (upperPoint == lowerPoint).all():
            return None, None

        return upperPoint, lowerPoint
    
    def create_image_from_points(self, points2D: torch.tensor) -> torch.tensor:
        # Points2D should be Nx2 with Y,X ordering
        base_image: torch.tensor = torch.zeros_like(self._glottis_segmentations[0])

        mask = ~torch.isnan(points2D).any(dim=-1)
        viable_points = points2D[mask]

        base_image[viable_points[:, 0].floor().long(), viable_points[:, 1].floor().long()] = 1
        return base_image

    def create_feature_images(self) -> torch.tensor:
        if self._glottis_segmentations is None:
            return None
        device = self._glottis_segmentations.device
        color_glottis: torch.tensor = torch.tensor([255, 0, 0], device=device)
        color_vocalfolds: torch.tensor = torch.tensor([0, 255, 0], device=device)
        color_glottal_midline: torch.tensor = torch.tensor([255, 255, 255], device=device)
        color_laserpoints: torch.tensor = torch.tensor([0, 0, 255], device=device)
        color_outline: torch.tensor = torch.tensor([0, 255, 255], device=device)

        feature_images = []
        for index in range(len(self)):
            image = torch.zeros(3, self._glottis_segmentations.shape[1], self._glottis_segmentations.shape[2], device=device)
            image += self._glottis_segmentations[index].unsqueeze(0).repeat(3, 1, 1) * color_glottis.unsqueeze(-1).unsqueeze(-1)
            
            if self._vocalfold_bounding_box is not None:
                rectangle = torch.concat(self._vocalfold_bounding_box).flatten()
                image = kornia.utils.draw_rectangle(image.unsqueeze(0), rectangle.unsqueeze(0).unsqueeze(0), color=color_vocalfolds).squeeze()
            elif self._vocalfold_segmentations is not None and self._vocalfold_bounding_box is None:
                image += self._vocalfold_segmentations[index].unsqueeze(0).repeat(3, 1, 1) * color_vocalfolds.unsqueeze(-1).unsqueeze(-1)

            shaped_laserseg = self._laserpoint_segmentations[index].unsqueeze(0).repeat(3, 1, 1)
            image[shaped_laserseg > 0] = 0
            colored_laserpoints = shaped_laserseg * color_laserpoints.unsqueeze(-1).unsqueeze(-1)
            colored_laserpoints = kornia.morphology.dilation(colored_laserpoints.float().unsqueeze(0), torch.ones([3, 3], device=device)).squeeze()
            image += colored_laserpoints

            shaped_glot_outline = self._glottal_outline_images[index].unsqueeze(0).repeat(3, 1, 1)
            image[shaped_glot_outline > 0] = 0
            image += shaped_glot_outline * color_outline.unsqueeze(-1).unsqueeze(-1)

            if self._glottal_midlines[index][0] is not None:
                try:
                    image = kornia.utils.draw_line(image, self._glottal_midlines[index][0], self._glottal_midlines[index][1], color_glottal_midline).squeeze()
                except:
                    continue
            
            feature_images.append(torchvision.transforms.functional.hflip(image) if self._flip_horizontal else image)

        return torch.stack(feature_images)


    def compute_features(self, video: torch.tensor) -> None:
        # TODO: Implement me!
        # Here, we need to compute the glottis segmentations, vocalfold_segmentations, laserpoint_positions, glottal midlines and outlines.
        # If your data is different, you need to override this part here!
        pass


class SiliconeFeatureEstimator(FeatureEstimator):
    def __init__(self):
        super().__init__()
        self._flip_horizontal: bool = False

    def compute_local_maxima(self, image, kernelsize=7):
        kernel = torch.ones((kernelsize, kernelsize), device=image.device)
        kernel[math.floor(kernelsize // 2), math.floor(kernelsize // 2)] = 0.0
        maxima = (image > kornia.morphology.dilation(image.unsqueeze(0).unsqueeze(0).float(), kernel)).squeeze()
        return maxima, maxima.nonzero()


    def compute_features(self, video: torch.tensor) -> None:
        self._glottis_segmentations = torch.zeros_like(video)
        self._vocalfold_segmentations = torch.zeros_like(video)
        self._laserpoint_segmentations = torch.zeros_like(video)
        
        self._glottal_outline_images = torch.zeros_like(video)
        self._glottal_outlines = []

        self._laserpoint_positions = []

        self._glottal_midlines: List[Tuple[torch.tensor, torch.tensor]] = []


        for index, image in enumerate(video):
            glottis_segmentation = torch.where(image == 0, 1, 0)
            self._glottis_segmentations[index] = glottis_segmentation


        gaw_max_index = self.glottalAreaWaveform().argmax()
        # This is in yx
        top_left, bottom_right = cv.get_segmentation_bounds(self._glottis_segmentations[gaw_max_index])
        y_min = top_left[0]
        x_min = top_left[1]

        y_max = bottom_right[0]
        x_max = bottom_right[1]

        glottal_width = x_max - x_min
        glottal_height = y_max - y_min

        mid_point = [x_min + glottal_width // 2, y_min + glottal_height // 2]

        roi_width = int(glottal_height * 1.2)
        roi_height = int(glottal_height * 2.0)
        
        new_x_min = mid_point[0] - roi_height // 2
        new_y_min = mid_point[1] - roi_width // 2
        new_x_max = mid_point[0] + roi_height // 2
        new_y_max = mid_point[1] + roi_width // 2
        
        new_x_min = 0 if (new_x_min < 0) else new_x_min
        new_y_min = 0 if (new_y_min < 0) else new_y_min
        new_x_max = image.shape[1] - 2 if (new_x_max > image.shape[1] - 1) else new_x_max
        new_y_max = image.shape[0] - 2 if (new_y_max > image.shape[0] - 1) else new_y_max


        self._vocalfold_bounding_box = [torch.tensor([new_x_min, new_y_min], device=video.device), torch.tensor([new_x_max, new_y_max], device=video.device)]
        
        for index, image in enumerate(video):
            self._vocalfold_segmentations[index, new_y_min:new_y_max, new_x_min:new_x_max] = 1
            laserpoint_image, laserpoint_positions = self.compute_local_maxima(image * self._vocalfold_segmentations[index])

            self._laserpoint_positions.append(laserpoint_positions)
            self._laserpoint_segmentations[index] = laserpoint_image

            upper_point, lower_point = self.compute_glottal_midline(self._glottis_segmentations[index], flipped=False)
            self._glottal_midlines.append([upper_point, lower_point])

            glottal_outline_image = cv.compute_segmentation_outline(self._glottis_segmentations[index])
            self._glottal_outline_images[index] = glottal_outline_image
            self._glottal_outlines.append(torch.nonzero(glottal_outline_image))

        return self._glottis_segmentations, self._glottal_midlines, self._glottal_outlines, self._vocalfold_segmentations, self._laserpoint_positions, self._laserpoint_segmentations, self.glottalAreaWaveform()




class NeuralFeatureEstimator(FeatureEstimator):
    def __init__(self, device: torch.cuda.device):
        super().__init__()

        self._flip_horizontal: bool = False
        self.point_localizer = point_extraction.LSQLocalization(heatmapaxis = 3, local_maxima_window = 11, gauss_window = 5)

        self._model = NeuralSegmentation.UNETNew().cuda()
        self._model.load_from_dict(torch.load("assets/MKMS.pth.tar"))

    def compute_features(self, video: torch.tensor) -> None:
        self._glottis_segmentations = torch.zeros_like(video)
        self._vocalfold_segmentations = torch.zeros_like(video)
        self._laserpoint_segmentations = torch.zeros_like(video)
        
        self._glottal_outline_images = torch.zeros_like(video)
        self._glottal_outlines = []

        self._laserpoint_positions = []

        self._glottal_midlines = []


        for index, image in enumerate(video):
            image = image.unsqueeze(0).unsqueeze(0).float() / 255
            image = image.repeat(1, 3, 1, 1)
            pred_seg = self._model(image).squeeze()
            softmaxed = torch.softmax(pred_seg, dim=0)
            labels = pred_seg.argmax(dim=0).byte()

            self._laserpoint_segmentations[index] = (labels == 3) * 1
            labels[labels == 3] = 2
            vocalfold_segmentation = (labels == 2) * 1
            glottis_segmentation = (labels == 1) * 1

            self._vocalfold_segmentations[index] = vocalfold_segmentation
            self._glottis_segmentations[index] = glottis_segmentation
            laserpoints = self.point_localizer.test(softmaxed.unsqueeze(0))[:, [1,2]]
            self._laserpoint_positions.append(laserpoints)

            upper_point, lower_point = self.compute_glottal_midline(self._glottis_segmentations[index])
            self._glottal_midlines.append([upper_point, lower_point])

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