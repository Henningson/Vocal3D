import numpy as np
import cv2
import math

from Segmentator import BaseSegmentator
from sklearn.mixture import GaussianMixture

def normalize(arr):
    return (arr - arr.min()) / (arr.max() - arr.min())

class IlluminationFunction:
    def __init__(self, m, b):
        self.m = m
        self.b = b

    def transform_image(self, image):
        reflection_image = (image.astype(np.float64) / (self.m * image.astype(np.float64) + self.b))
        return normalize(reflection_image)


class KocSegmentator(BaseSegmentator):
    def __init__(self, images, useContrastEnhancement=False, numImages=50):
        super(KocSegmentator, self).__init__(images)

        if useContrastEnhancement:
            for i, image in enumerate(self.images):
                self.images[i] = self.enhanceContrast(image)

        self.numImages = numImages

        self.gaussianMixture = None

        self.illum = None

        self.intensityMap = None

        self.generateSegmentationData()

    def segmentImage(self, image):
        reflection_image = self.illum.transform_image(image)

        x, w, y, h = self.getROI()
        reflection_image = reflection_image[y:y+h, x:x+w]
        segmentation_prediction = 255 - self.gaussianMixture.predict(reflection_image.reshape(-1, 1)).reshape(h, w)*255
        zero_image = np.zeros(self.images[0].shape, dtype=np.uint8)
        zero_image[y:y+h, x:x+w] = segmentation_prediction

        return zero_image
        

    def generate(self):
        self.intensityMap = self.generateIntensityMap()

        # We use gaussian instead of median filtering to improve runtime
        self.intensityMap = cv2.blur(self.intensityMap, (9, 9))
        Mx = self.getMaxX(self.intensityMap)
        My = self.getMaxY(self.intensityMap)
        union = np.concatenate([Mx, My])

        horizontalSliceInterval = 1

        mean = np.mean(union)
        std = np.std(union)

        binarized = np.where(self.intensityMap > mean + std, 255, 0).astype(np.uint8)
        
        self.generateROI(binarized)

        x, w, y, h = self.getROI()
        roiImage = self.images[0][y:y+h, x:x+w]

        VSI = roiImage[:, w//2].reshape(-1, 1) #VSI := Vertical Slice Image

        for image in self.images:
            VSI = np.concatenate([VSI, image[y:y+h, x+w//2-horizontalSliceInterval:x+w//2]], axis=1)
            
        temporalMean = np.mean(VSI, axis=1)
        xValues = np.arange(0, len(temporalMean), 1)
        m, b = np.polyfit(xValues, temporalMean, 1)
        self.illum = IlluminationFunction(m, b)

        reflectionVSI = self.illum.transform_image(roiImage)
        reflectionVSI = np.expand_dims(reflectionVSI[:, reflectionVSI.shape[1]//2], -1)

        for image in self.images:
            reflection_image = self.illum.transform_image(image)
            reflectionVSI = np.concatenate([reflectionVSI, reflection_image[y:y + h, x+w//2-horizontalSliceInterval:x+w//2]], axis=1)
        
        self.gaussianMixture = GaussianMixture(n_components=2, random_state=0).fit(reflectionVSI.reshape(-1, 1)) 

        sorted_params = sorted(zip(self.gaussianMixture.means_.flatten().tolist(), self.gaussianMixture.covariances_.flatten().tolist(), self.gaussianMixture.weights_.flatten().tolist()), key=lambda pair: pair[0])
        sorted_means = [means for means, _, _ in sorted_params]
        sorted_covariances = [covs for _, covs, _ in sorted_params]
        sorted_weights = [weights for _, _, weights in sorted_params]

        self.gaussianMixture.weights_ = np.array(sorted_weights)
        self.gaussianMixture.means_ = np.array(sorted_means).reshape(2, 1)
        self.gaussianMixture.covariances_ = np.array(sorted_covariances).reshape(2, 1, 1)


    def getMaxX(self, arr):
        return np.max(arr, axis=1)


    def getMaxY(self, arr):
        return np.max(arr, axis=0)


    def generateIntensityMap(self):
        intensityMap = np.zeros(self.images[0].shape, dtype=np.int32)
        
        for imageA, imageB in zip(self.images[0:self.numImages-1], self.images[1:self.numImages]):
            intensityMap += np.abs(imageA.astype(np.int32) - imageB.astype(np.int32))

        return intensityMap


    def enhanceContrast(self, image):
        flattened_image = image.flatten()
        ten_percent = len(flattened_image) // 10
        indices = np.argpartition(flattened_image, -ten_percent)[-ten_percent:]
        
        intensity_threshold = np.average(flattened_image[indices]) # I assume we use the average of the top 10% of values?
        
        maskA = np.where(image > intensity_threshold, 255, 0).astype(np.uint8)
        maskB = cv2.bitwise_not(maskA)

        maskB = maskB * np.power(image / intensity_threshold, 1.8)

        return maskA+maskB.astype(np.uint8)


    def generateROI(self, seg_image):
        
        # In the paper, they determine the region with the highest average intensity and use this as a starting point for masking
        # We instead use the largest thresholded area.
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

        roiX = roiX+roiWidth//2 - roiWidth * 2
        roiWidth = 4*roiWidth

        if roiY < 0:
            roiY = 0

        if roiX < 0:
            roiX = 0

        if roiY+roiHeight >= self.images[0].shape[0]:
            roiHeight = self.images[0].shape[0] - self.roiY - 2
        if roiX+roiWidth >= self.images[0].shape[1]:
            roiWidth = self.images[0].shape[1] - self.roiX - 2

        self.ROI = [roiX, roiWidth, roiY, roiHeight]

    def getROI(self):
        return self.ROI

    def estimateClosedGlottis(self):
        num_pixels = 50000
        glottis_closed_at_frame = 0
        for count, segmentation in enumerate(self.segmentations):
            num_glottis_pixels = len(segmentation.nonzero()[0])

            if num_pixels > num_glottis_pixels:
                num_pixels = num_glottis_pixels
                glottis_closed_at_frame = count
        
        return glottis_closed_at_frame

    def estimateOpenGlottis(self):
        num_pixels = 0
        glottis_open_at_frame = 0

        for count, segmentation in enumerate(self.segmentations):
            num_glottis_pixels = len(segmentation.nonzero()[0])

            if num_pixels < num_glottis_pixels:
                num_pixels = num_glottis_pixels
                glottis_open_at_frame = count
        
        return glottis_open_at_frame


    def generateSegmentationData(self):
        self.generate()

        for i, image in enumerate(self.images):
            self.segmentations.append(self.segmentImage(image))
            self.glottalOutlines.append(self.computeGlottalOutline(i))
            self.glottalMidlines.append(self.computeGlottalMidline(i))
        
        self.closedGlottisIndex = self.estimateClosedGlottis()
        self.openGlottisIndex = self.estimateOpenGlottis()

        self.ROIImage = self.generateROIImage()

        for i, image in enumerate(self.images):
            self.localMaxima.append(self.computeLocalMaxima(i))



    def computeLocalMaxima(self, index, kernelsize=7):
        image = self.images[index]
        kernel = np.ones((kernelsize, kernelsize), dtype=np.uint8)
        kernel[math.floor(kernelsize//2), math.floor(kernelsize//2)] = 0.0
        maxima = image > cv2.morphologyEx(image, cv2.MORPH_DILATE, kernel)
        roiImage = self.getROIImage()
        maxima = (self.getROIImage() & image) * maxima
        return maxima