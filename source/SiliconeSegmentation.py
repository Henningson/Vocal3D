import numpy as np
import cv2
import math

from Segmentator import BaseSegmentator


class SiliconeSegmentator(BaseSegmentator):
    def __init__(self, images):
        super(SiliconeSegmentator, self).__init__(images)
        self.generateSegmentationData()

    def segmentImage(self, frame):
        return np.where(frame == 0, 255, 0).astype(np.uint8)


    def generateROI(self):
        seg_image = self.segmentations[self.openGlottisIndex]
        
        _, _, stats, centroids = cv2.connectedComponentsWithStats(seg_image, 4, cv2.CV_32S)

        stats = np.array(stats)
        area = stats[:, 4]
        stats = stats[:, :4]

        area = area.tolist()
        stats = stats.tolist()

        sorted_stats = sorted(zip(area, stats))
        sorted_centroids = sorted(zip(area, centroids.astype(np.uint32).tolist()))

        centroid = sorted_centroids[-2][1]
        roiX = sorted_stats[-2][1][0]
        roiY = sorted_stats[-2][1][1]
        roiWidth = sorted_stats[-2][1][2]
        roiHeight = sorted_stats[-2][1][3]

        roiX = roiX+roiWidth//2 - roiHeight
        roiWidth = 2*roiHeight
        roiY = int(roiY - (roiHeight*0.2))
        roiHeight = int(roiHeight + (roiHeight * 0.4))

        if roiY < 0:
            roiY = 0

        if roiX < 0:
            roiX = 0

        if roiY+roiHeight >= self.images[0].shape[0]:
            roiHeight = self.images[0].shape[0] - roiY - 2
        if roiX+roiWidth >= self.images[0].shape[1]:
            roiWidth = self.images[0].shape[1] - roiX - 2

        return [roiX, roiWidth, roiY, roiHeight]

    def estimateClosedGlottis(self):
        num_pixels = len((self.segmentations[0] == 255).nonzero()[0])
        glottis_closed_at_frame = 0

        for count, segmentation in enumerate(self.segmentations):
            num_glottis_pixels = len((segmentation == 255).nonzero()[0])

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


    def computeLocalMaxima(self, index, kernelsize=7):
        image = self.images[index]
        kernel = np.ones((kernelsize, kernelsize), dtype=np.uint8)
        kernel[math.floor(kernelsize//2), math.floor(kernelsize//2)] = 0.0
        maxima = image > cv2.morphologyEx(image, cv2.MORPH_DILATE, kernel)
        roiImage = self.getROIImage()
        maxima = (self.getROIImage() & image) * maxima
        return maxima