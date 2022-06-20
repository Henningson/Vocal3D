import cv2

global mouseX
global mouseY
    


class SegmentationClicker:
    def __init__(self, image):
        self.polygon = list()
        self.midlinePolygon = list()
        self.image = image

    def get_ClickpositionSeg(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDBLCLK:
            self.polygon.append([x, y])
            print(self.polygon)
            cv2.circle(self.image, [x, y], 3, 255, -1)

    
    def get_ClickpositionMid(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDBLCLK:
            self.midlinePolygon.append([x, y])
            print(self.midlinePolygon)
            cv2.circle(self.image, [x, y], 3, 255, -1)

    def clickSegmentation(self):
        self.polygon = list()

        cv2.namedWindow("Segmentation Clicker")
        cv2.setMouseCallback("Segmentation Clicker", self.get_ClickpositionSeg)

        while len(self.polygon) < 2:
            cv2.imshow("Segmentation Clicker", self.image)
            k = cv2.waitKey(20) & 0xFF
            if k == 27:
                break

        self.roiX = self.polygon[0][0]
        self.roiY = self.polygon[0][1]
        self.roiWidth = self.polygon[1][0] - self.polygon[0][0]
        self.roiHeight = self.polygon[1][1] - self.polygon[0][1]

    def getROI(self):
        return self.roiX, self.roiWidth, self.roiY, self.roiHeight

    def clickMidline(self):
        self.midlinePolygon = list()

        cv2.namedWindow("Midline Clicker")
        cv2.setMouseCallback("Midline Clicker", self.get_ClickpositionMid)

        while len(self.midlinePolygon) < 2:
            cv2.imshow("Midline Clicker", self.image)
            k = cv2.waitKey(20) & 0xFF
            if k == 27:
                break

        self.midlineA = self.midlinePolygon[0]
        self.midlineB = self.midlinePolygon[1]

    def getROI(self):
        return self.roiX, self.roiWidth, self.roiY, self.roiHeight
        
    def getMidline(self):
        return self.midlineA, self.midlineB