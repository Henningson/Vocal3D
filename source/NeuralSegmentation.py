import math

import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
from Segmentator import BaseSegmentator

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



class DoubleConvB(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConvB, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True))

    def forward(self, x):
        return self.conv(x)


class Decoder(nn.Module):
    def __init__(self, encoder, out_channels, features):
        super(Decoder, self).__init__()
        self.ups = nn.ModuleList()
        self.encoder = encoder
        self.out_channels=out_channels

        for feature in reversed(features):
            self.ups.append(nn.ConvTranspose2d(feature*2, feature, kernel_size=2, stride=2))
            self.ups.append(DoubleConvB(feature*2, feature))


    def forward(self, x):
        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_connection = self.encoder.skip_connections[idx//2]

            if x.shape != skip_connection.shape:
                x = TF.resize(x, size=skip_connection.shape[2:])

            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx+1](concat_skip)

        return x


class Encoder(nn.Module):
    def __init__(self, in_channels, features):
        super(Encoder, self).__init__()
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.in_channels = in_channels
        
        #Downsampling
        for feature in features:
            self.downs.append(DoubleConvB(in_channels, feature))
            in_channels = feature

    def forward(self, x):
        self.skip_connections = []
        for down in self.downs:
            x = down(x)
            self.skip_connections.append(x)
            x = self.pool(x)

        self.skip_connections = self.skip_connections[::-1]

        return x



class UNETNew(nn.Module):
    def __init__(self, config={'in_channels': 3, 'out_channels': 4, 'features': [32, 64, 128, 256, 512]}, state_dict=None, pretrain=False, device="cuda"):
        super(UNETNew, self).__init__()
        try:
            in_channels = config['in_channels']
        except:
            in_channels = 3

        try: 
            out_channels = config['out_channels']
        except:
            out_channels = 4
            
        features = config['features']

        self.bottleneck_size = features[-1]*2

        self.encoder = Encoder(in_channels, features)
        self.decoder = Decoder(self.encoder, out_channels, features)
        self.bottleneck = DoubleConv(features[-1], self.bottleneck_size)
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

        if state_dict:
            self.load_from_dict(state_dict)

        if pretrain:
            self.encoder.requires_grad_ = False

    def get_statedict(self):
        return {"Encoder": self.encoder.state_dict(),
                "Bottleneck": self.bottleneck.state_dict(),
                "Decoder": self.decoder.state_dict(),
                "LastConv": self.final_conv.state_dict()}

    def load_from_dict(self, dict):
        self.encoder.load_state_dict(dict["Encoder"])
        self.bottleneck.load_state_dict(dict["Bottleneck"])
        self.decoder.load_state_dict(dict["Decoder"])
        
        try:
            self.final_conv.load_state_dict(dict["LastConv"])
        except:
            print("Final conv not initialized.")

    def forward(self, x):
        x = self.encoder(x)
        x = self.bottleneck(x)
        x = self.decoder(x)

        return self.final_conv(x)

class DownConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, last_layer=False):
        super(DownConv, self).__init__()

        if not last_layer:
            self.conv = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size, 1, kernel_size // 2),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size, 1, 0),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            )
        else:
            self.conv = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size, 1, kernel_size // 2),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size, 1, 0),
            )

    def forward(self, x):
        return self.conv(x)
    
class BinaryKernel3Classificator(nn.Module):
    def __init__(self):
        super(BinaryKernel3Classificator, self).__init__()

        self.a = DownConv(1, 128, kernel_size=3)
        self.b = DownConv(128, 64, kernel_size=3)
        self.c = DownConv(64, 1, kernel_size=3, last_layer=True)

    def forward(self, x):
        x = self.a(x)
        x = self.b(x)
        x = self.c(x)

        return x.squeeze()


class Kernel3Classificator(nn.Module):
    def __init__(self):
        super(Kernel3Classificator, self).__init__()

        self.a = DownConv(1, 128, kernel_size=3)
        self.b = DownConv(128, 64, kernel_size=3)
        self.c = DownConv(64, 3, kernel_size=3, last_layer=True)

    def forward(self, x):
        x = self.a(x)
        x = self.b(x)
        x = self.c(x)

        return x.squeeze()


class NeuralSegmentator(BaseSegmentator):
    def __init__(self, images, path="assets/model.pth.tar"):
        super().__init__(images)

        self.model = UNET(in_channels=1, out_channels=3).to(DEVICE)
        self.model.load_state_dict(torch.load(path)["state_dict"])

        self.laserdotSegmentations = list()

        self.generateSegmentationData()


    def class_to_color(self, prediction, class_colors):
        prediction = np.expand_dims(prediction, 1)
        output = np.zeros((prediction.shape[0], 3, prediction.shape[-2], prediction.shape[-1]), dtype=np.uint8)
        for class_idx, color in enumerate(class_colors):
            mask = class_idx == prediction.max(axis=1)[0]
            mask = np.expand_dims(np.expand_dims(mask, 0), 0) # should have shape 1, 1, 100, 100
            curr_color = color.reshape(1, 3, 1, 1)
            segment = mask*curr_color # should have shape 1, 3, 100, 100
            output += segment.astype(np.uint8)

        return output

    def segmentImage(self, frame):
        segmentation = self.model(torch.from_numpy(frame).unsqueeze(0).unsqueeze(0).to(DEVICE).float()).argmax(dim=1).detach().cpu().numpy().squeeze().astype(np.uint8)

        # Find biggest connected component
        _, _, stats, centroids = cv2.connectedComponentsWithStats(segmentation, 4)
        stats = np.array(stats)
        area = stats[:, 4]
        stats = stats[:, :4]
        area = area.tolist()
        stats = stats.tolist()
        sorted_stats = sorted(zip(area, stats))

        glottal_roi = np.zeros(segmentation.shape, np.uint8)
        x, y, w, h = sorted_stats[-2][1]
        glottal_roi[y:y+h, x:x+w] = 1
        filtered_glottis = ((segmentation == 2) * 255 * glottal_roi).astype(np.uint8)

        class_colors = [np.array([0, 0, 0]), np.array([0, 255, 0]), np.array([0, 0, 255])]
        colored = np.moveaxis(self.class_to_color(np.expand_dims(segmentation, 0), class_colors)[0], 0, -1)

        self.laserdotSegmentations.append(((segmentation == 1) * 255).astype(np.uint8))
        return filtered_glottis

    def computeLocalMaxima(self, index, kernelsize=7):
        image = self.laserdotSegmentations[index]
        img_erosion1 = cv2.erode(image, np.ones((3, 3), np.uint8), iterations=1)

        _, _, _, centroids = cv2.connectedComponentsWithStats(img_erosion1, 4)
        filtered_centroids = centroids[1:, :]
        maxima = np.zeros(image.shape, np.uint8)
        maxima[filtered_centroids[:, 1].astype(np.int), filtered_centroids[:, 0].astype(np.int)] = 255
        
        return maxima

    def generateROI(self):
        minX = 0
        maxX = 0
        minY = 0
        maxY = 0

        for laserdotSegmentation in self.laserdotSegmentations:
            ys, xs = np.nonzero(laserdotSegmentation)

            maxY = np.max(ys)
            minY = np.min(ys)
            maxX = np.max(xs)
            minX = np.min(xs)


        return [minX, maxX-minX, minY, maxY-minY]


    def estimateClosedGlottis(self):
        num_pixels = 100000000
        glottis_closed_at_frame = 0

        for count, segmentation in enumerate(self._segmentations):
            num_glottis_pixels = len(segmentation.nonzero()[0])

            if num_pixels > num_glottis_pixels:
                num_pixels = num_glottis_pixels
                glottis_closed_at_frame = count
        
        return glottis_closed_at_frame


    def estimateOpenGlottis(self):
        num_pixels = 0
        
        for count, segmentation in enumerate(self._segmentations):
            num_glottis_pixels = len(segmentation.nonzero()[0])

            if num_pixels < num_glottis_pixels:
                num_pixels = num_glottis_pixels
                glottis_open_at_frame = count
        
        return glottis_open_at_frame

    
    def generateSegmentationData(self):
        for i, image in enumerate(self.images):
            self._segmentations.append(self.segmentImage(image))
            self._glottal_outlines.append(self.computeGlottalOutline(i))
            self._glottal_midlines.append(self.computeGlottalMidline(i))
        
        self.ROI = self.generateROI()
        self.ROIImage = self.generateROIImage()
        
        for i, image in enumerate(self.images):
            self.localMaxima.append(self.computeLocalMaxima(i))
        
        self.closedGlottisIndex = self.estimateClosedGlottis()
        self.openedGlottisIndex = self.estimateOpenGlottis()