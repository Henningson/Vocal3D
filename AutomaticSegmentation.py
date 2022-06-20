import numpy as np
import cv2
import main
import helper
import Camera
import matplotlib.pyplot as plt

from sklearn.mixture import GaussianMixture
import scipy.stats as stats


class IlluminationFunction:
    def __init__(self, m, b):
        self.m = m
        self.b = b

    def transform_image(self, image):
        reflection_image = (image.astype(np.float64) / (self.m * image.astype(np.float64) + self.b))
        return helper.normalize(reflection_image)


class HSVGlottisSegmentator:
    def __init__(self, images, useContrastEnhancement=False):
        self.images = images
        self.num_images = len(images)

        if useContrastEnhancement:
            for i, image in enumerate(self.images):
                self.images[i] = self.enhanceContrast(image)

        self.useContrastEnhancement = useContrastEnhancement

        self.roiX = None
        self.roiY = None
        self.roiWidth = None
        self.roiHeight = None

        self.gaussianMixture = None

        self.illum = None

        self.intensityMap = None

    def segment_image(self, image, useRoi=True):
        reflection_image = self.illum.transform_image(image)

        if useRoi:
            reflection_image = reflection_image[self.roiY:self.roiY+self.roiHeight, self.roiX:self.roiX+self.roiWidth]
            return self.gaussianMixture.predict(reflection_image.reshape(-1, 1)).reshape(self.roiHeight, self.roiWidth)

        return self.gaussianMixture.predict(reflection_image.reshape(-1, 1)).reshape(reflection_image.shape[0], reflection_image.shape[1])

    def generate(self, isSilicone=False, plot=False):
        self.intensityMap = self.generateIntensityMap(plot)
        self.intensityMap = cv2.blur(self.intensityMap, (9, 9)) #We gaussian filter, instead of median filtering
        Mx = self.getMaxX(self.intensityMap)
        My = self.getMaxY(self.intensityMap)
        union = np.concatenate([Mx, My])

        horizontalSliceInterval = 1 if not isSilicone else 45

        mean = np.mean(union)
        std = np.std(union)

        binarized = np.where(self.intensityMap > mean + std, 255, 0).astype(np.uint8)
        
        if not isSilicone:
            self.generateROI(binarized)

        roiImage = self.images[0][self.roiY:self.roiY+self.roiHeight, self.roiX:self.roiX+self.roiWidth]

        VSI = roiImage[:, self.roiWidth//2].reshape(-1, 1) #VSI := Vertical Slice Image

        for image in self.images[1:self.num_images]:
            VSI = np.concatenate([VSI, image[self.roiY:self.roiY + self.roiHeight, self.roiX+self.roiWidth//2-horizontalSliceInterval:self.roiX+self.roiWidth//2]], axis=1)
            
        temporalMean = np.mean(VSI, axis=1)
        xValues = np.arange(0, len(temporalMean), 1)
        m, b = np.polyfit(xValues, temporalMean, 1)
        self.illum = IlluminationFunction(m, b)

        reflectionVSI = self.illum.transform_image(roiImage)
        reflectionVSI = np.expand_dims(reflectionVSI[:, reflectionVSI.shape[1]//2], -1)

        for image in self.images[1:]:
            reflection_image = self.illum.transform_image(image)
            reflectionVSI = np.concatenate([reflectionVSI, reflection_image[self.roiY:self.roiY + self.roiHeight, self.roiX+self.roiWidth//2-horizontalSliceInterval:self.roiX+self.roiWidth//2]], axis=1)
        
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

    def generateIntensityMap(self, plot=False):
        intensityMap = np.zeros(self.images[0].shape, dtype=np.int32)
        
        for imageA, imageB in zip(self.images[0:self.num_images-1], self.images[1:self.num_images]):
            intensityMap += np.abs(imageA.astype(np.int32) - imageB.astype(np.int32))

        norm = helper.normalize(intensityMap)
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

    def generateROI(self, thresholded_image):
        # In the paper, they determine the region with the highest average intensity and use this as a starting point for masking
        # We instead use the largest thresholded area.
        _, _, stats, centroids = cv2.connectedComponentsWithStats(thresholded_image, 4, cv2.CV_32S)

        stats = np.array(stats)
        area = stats[:, 4]
        stats = stats[:, :4]

        area = area.tolist()
        stats = stats.tolist()

        sorted_stats = sorted(zip(area, stats))
        sorted_centroids = sorted(zip(area, centroids.astype(np.uint32).tolist()))

        centroid = sorted_centroids[-2][1]
        self.roiX = sorted_stats[-2][1][0]
        self.roiY = sorted_stats[-2][1][1]
        self.roiWidth = sorted_stats[-2][1][2]
        self.roiHeight = sorted_stats[-2][1][3]

    def getROI(self):
        return self.roiX, self.roiWidth, self.roiY, self.roiHeight

    def estimateClosedGlottis(self):
        num_pixels = 50000
        glottis_closed_at_frame = 0
        for count, image in enumerate(self.images[1:-1]):
            segmentation = self.segment_image(self.illum.transform_image(image))

            num_glottis_pixels = len(np.where(segmentation == 0, 255, 0).nonzero()[0])

            if num_pixels > num_glottis_pixels:
                num_pixels = num_glottis_pixels
                glottis_closed_at_frame = count
        
        return glottis_closed_at_frame

    def getGlottalMidline(self, frame):
        segmentation = self.segment_image(self.illum.transform_image(frame))

        white_points = np.argwhere(segmentation == 0)

        if white_points.size == 0:
            return None, None

        y = white_points[:, 1]
        x = white_points[:, 0]

        A = np.vstack(np.vstack([x, np.ones(len(x))]).T)
        m, c = np.linalg.lstsq(A, y, rcond=None)[0]

        upperPoint = np.array([m*x.min() + c, x.min()])
        lowerPoint = np.array([m*x.max() + c, x.max()])

        return upperPoint, lowerPoint

    def getGlottalOutline(self, frame):
        segmentation = self.segment_image(self.illum.transform_image(frame))

        if np.argwhere(segmentation == 0).size == 0:
            return None

        segmentation = cv2.copyMakeBorder(segmentation, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=1)

        contours, hierarchy = cv2.findContours(segmentation.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        i = 1
        contour_points = list()
        while (i != -1):
            contour_points.append(contours[hierarchy[0][i][0]][:, 0, :])
            i = hierarchy[0][i][0]

        contourArray = None
        if len(contour_points) > 1:
            contourArray = np.concatenate(contour_points, axis=0)
        else:
            contourArray = contour_points[0]
        return contourArray - np.ones(contourArray.shape)
