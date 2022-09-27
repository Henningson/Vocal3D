import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
from Segmentator import BaseSegmentator
import numpy as np
import cv2
import math

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True))

    def forward(self, x):
        return self.conv(x)


class UNET(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, features=[64, 128, 256, 512]):
        super(UNET, self).__init__()
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.in_channels = in_channels
        self.out_channels = out_channels

        #Downsampling
        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature))
            in_channels = feature

        #Upsampling
        for feature in reversed(features):
            self.ups.append(nn.ConvTranspose2d(feature*2, feature, kernel_size=2, stride=2))
            self.ups.append(DoubleConv(feature*2, feature))

        self.bottleneck = DoubleConv(features[-1], features[-1]*2)
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)
        
    def forward(self, x):
        skip_connections = []
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx//2]

            if x.shape != skip_connection.shape:
                x = TF.resize(x, size=skip_connection.shape[2:])

            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx+1](concat_skip)

        return self.final_conv(x)



class NeuralSegmentator(BaseSegmentator):
    def __init__(self, images, path="assets/checkpoint.pth.tar"):
        super().__init__(images)

        self.model = UNET(in_channels=1, out_channels=3).to(DEVICE)
        self.model.load_state_dict(torch.load(path)["state_dict"])

        self.laserdotSegmentations = list()

        
    def segmentImage(self, frame):
        segmentation = self.model(frame).argmax(dim=1).detach().cpu().numpy()
        self.laserdotSegmentations.append(np.where(segmentation==1, 255, 0))
        return np.where(segmentation==2, 255, 0)

    def computeLocalMaxima(self, index, kernelsize=7):
        image = self.laserdotSegmentations[index]
        kernel = np.ones((kernelsize, kernelsize), dtype=np.uint8)
        kernel[math.floor(kernelsize//2), math.floor(kernelsize//2)] = 0.0
        maxima = image > cv2.morphologyEx(image, cv2.MORPH_DILATE, kernel)
        maxima = (self.getROIImage() & image) * maxima
        return maxima

    def generateROI(self):
        minX = 0
        maxX = 0
        minY = 0
        maxY = 0

        for laserdotSegmentation in self.laserdotSegmentations:
            ys, xs = np.nonzero(laserdotSegmentation)

            maxY = np.max(ys)
            minY = np.max(ys)
            maxX = np.max(xs)
            minX = np.min(xs)


        return [minX, maxX-minX, minY, maxY-minY]


    def estimateClosedGlottis(self):
        num_pixels = 100000000
        glottis_closed_at_frame = 0

        for count, segmentation in enumerate(self.segmentations):
            num_glottis_pixels = len(segmentation.nonzero()[0])

            if num_pixels > num_glottis_pixels:
                num_pixels = num_glottis_pixels
                glottis_closed_at_frame = count
        
        return glottis_closed_at_frame


    def estimateOpenGlottis(self):
        num_pixels = 0
        
        for count, segmentation in enumerate(self.segmentations):
            num_glottis_pixels = len(segmentation.nonzero()[0])

            if num_pixels < num_glottis_pixels:
                num_pixels = num_glottis_pixels
                glottis_open_at_frame = count
        
        return glottis_open_at_frame

    
    def generateSegmentationData(self):
        for i, image in enumerate(self.images):
            self.segmentations.append(self.segmentImage(image))
            self.glottalOutlines.append(self.computeGlottalOutline(i))
            self.glottalMidlines.append(self.computeGlottalMidline(i))
        

        self.ROI = self.generateROI()
        self.ROIImage = self.generateROIImage()
        self.closedGlottisIndex = self.estimateClosedGlottis()
        self.openedGlottisIndex = self.estimateOpenGlottis()