import numpy as np
import cv2

class BaseSegmentator:
    def __init__(self, images):
        # Images
        self.images = images

        # List of Segmentations
        self.segmentations = list()

        # List of Nx2 points
        self.glottalOutlines = list()

        # List of 2x2 points
        self.glottalMidlines = list()

        # List of extracted local Maxima
        self.localMaxima = list()

        self.closedGlottisIndex = None
        self.openGlottisIndex = None

        # 4-Tuple of [X, Width, Y, Height]
        self.ROI = None
        self.ROIImage = None

    def getImage(self, index):
        return self.images[index]

    def __len__(self):
        return len(self.images)


    #TODO: Implement
    def segmentImage(self, frame):
        return None

    def segmentImageIndex(self, index):
        return self.segmentImage(self.images[index])

    def getSegmentation(self, index):
        return self.segmentations[index]

    def computeGlottalOutline(self, index):
        segmentation = self.segmentations[index]
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

    def getGlottalOutline(self, index):
        return self.glottalOutlines[index]


    def computeGlottalMidline(self, index):
        segmentation = self.segmentations[index]

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

    def getGlottalMidline(self, index):
        return self.glottalMidlines[index]


    #TODO: Implement
    def computeLocalMaxima(self, index):
        return None

    def getLocalMaxima(self, index):
        return self.localMaxima[index]


    #TODO: Implement
    def generateROI(self):
        return None

    def getROI(self):
        return self.ROI


    #TODO: Implement
    def estimateClosedGlottis(self):
        return None

    def getClosedGlottisIndex(self):
        return self.closedGlottisIndex


    #TODO: Implement
    def estimateOpenGlottis(self):
        return None

    def getOpenGlottisIndex(self):
        return self.closedGlottisIndex

    
    def generateROIImage(self):
        roiImage = np.zeros((self.images[0].shape[0], self.images[0].shape[1]), dtype=np.uint8)
        x, w, y, h = self.getROI()
        roiImage[y:y+h, x:x+w] = 255

        return roiImage


    def getROIImage(self):
        return self.ROIImage

    def generateSegmentationData(self):
        for i, image in enumerate(self.images):
            self.segmentations.append(self.segmentImage(image))
            self.glottalOutlines.append(self.computeGlottalOutline(i))
            self.glottalMidlines.append(self.computeGlottalMidline(i))
        
        self.closedGlottisIndex = self.estimateClosedGlottis()
        self.openGlottisIndex = self.estimateOpenGlottis()

        self.ROI = self.generateROI()
        self.ROIImage = self.generateROIImage()

        for i, image in enumerate(self.images):
            self.localMaxima.append(self.computeLocalMaxima(i))
