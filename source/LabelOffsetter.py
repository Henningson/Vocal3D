import cv2
import numpy as np

global xDelta
global yDelta
    
class LabelOffsetter:
    def __init__(self, image, gridPixPositions):
        self.image = image
        self.xDelta = 0
        self.yDelta = 0
        self.gridPixPositions = gridPixPositions

    def addLabelToImage(self, image):
        for i in range(len(self.gridPixPositions)):
            cv2.circle(image, np.flip(self.gridPixPositions[i][1]).astype(np.int)*4, 3, (255, 0, 0), -1)
            cv2.putText(image, '{0},{1}'.format(self.gridPixPositions[i][0][0] + self.xDelta, self.gridPixPositions[i][0][1] + self.yDelta), np.flip(self.gridPixPositions[i][1]).astype(np.int)*4, cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        return image


    def label(self):
        while True:
            cv2.imshow("Label Offsetter", self.addLabelToImage(cv2.resize(self.image.copy(), (self.image.shape[1]*4, self.image.shape[0]*4))))
            k = cv2.waitKey(20) & 0xFF

            if k == 119:
                self.yDelta += 1
            
            if k == 97:
                self.xDelta += -1
            
            if k == 115:
                self.yDelta -= 1

            if k == 100:
                self.xDelta += 1

            if k == 27:
                break

        for i in range(len(self.gridPixPositions)):
            self.gridPixPositions[i][0] = self.gridPixPositions[i][0] + np.array([self.xDelta, self.yDelta], dtype=np.int)

        return self.gridPixPositions


if __name__ == "__main__":
    for i in range(10):
        cv2.imshow("Test", np.zeros((256, 256), dtype=np.uint8))
        res = cv2.waitKey(0)
        print('You pressed %d (0x%x), LSB: %d (%s)' % (res, res, res % 256, repr(chr(res%256)) if res%256 < 128 else '?'))