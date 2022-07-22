import numpy as np
import cv2


class SiliconeVocalfoldSegmentator:
    def __init__(self, images):
        self.images = images

    def segment_image(self, frame):
        return np.where(frame == 0, 255, 0).astype(np.uint8)

    def getGlottalOutline(self, frame):
        segmentation = self.segment_image(frame)
        contours, hierarchy = cv2.findContours(segmentation, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        i = 0
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

    def getGlottalMidline(self, frame, isSegmented=True):
        segmentation = frame
        if not isSegmented:
            segmentation = self.segment_image(frame)

        white_points = np.argwhere(segmentation == 255)

        if white_points.size == 0:
            return None, None

        y = white_points[:, 1]
        x = white_points[:, 0]

        A = np.vstack(np.vstack([x, np.ones(len(x))]).T)
        m, c = np.linalg.lstsq(A, y, rcond=None)[0]

        upperPoint = np.array([m*x.min() + c, x.min()])
        lowerPoint = np.array([m*x.max() + c, x.max()])

        return upperPoint, lowerPoint

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

        self.roiX = self.roiX+self.roiWidth//2 - self.roiHeight
        self.roiWidth = 2*self.roiHeight
        self.roiY = int(self.roiY - (self.roiHeight*0.2))
        self.roiHeight = int(self.roiHeight + (self.roiHeight * 0.4))

        if self.roiY < 0:
            self.roiY = 0

        if self.roiX < 0:
            self.roiX = 0

        if self.roiY+self.roiHeight >= self.images[0].shape[0]:
            self.roiHeight = self.images[0].shape[0] - self.roiY - 2
        if self.roiX+self.roiWidth >= self.images[0].shape[1]:
            self.roiWidth = self.images[0].shape[1] - self.roiX - 2

    def getROI(self):
        return self.roiX, self.roiWidth, self.roiY, self.roiHeight

    def estimateClosedGlottis(self):
        num_pixels = 100000000
        glottis_closed_at_frame = 0
        for count, image in enumerate(self.images[1:-1]):
            segmentation = self.segment_image(image)

            num_glottis_pixels = len(segmentation.nonzero()[0])

            if num_pixels > num_glottis_pixels:
                num_pixels = num_glottis_pixels
                glottis_closed_at_frame = count
        
        return glottis_closed_at_frame

    def estimateMaximallyOpenGlottis(self):
        num_pixels = 0
        
        for count, image in enumerate(self.images[1:-1]):
            segmentation = self.segment_image(image)

            num_glottis_pixels = len(segmentation.nonzero()[0])

            if num_pixels > num_glottis_pixels:
                num_pixels = num_glottis_pixels
                glottis_closed_at_frame = count

    def generate(self):
        self.generateROI(self.segment_image(self.images[self.estimateClosedGlottis()]))
