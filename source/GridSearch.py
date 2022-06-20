import numpy as np

from sklearn.neighbors import NearestNeighbors


class PointBasedGridSearch:    
    def __init__(self, maxima, startingPoint, startingIndex, averageMaximaDistance, laserDims):
        self.upVector = np.array([-1.0, 0.0])
        self.downVector = np.array([1.0, 0.0])
        self.rightVector = np.array([0.0, 1.0])
        self.leftVector = np.array([0.0, -1.0])

        self.epsilon = 5.0

        self.laserDims = laserDims

        self.startingPoint = np.array(startingPoint)
        self.startingIndex = np.array(startingIndex)

        self.upIndex = np.array([0, 1], np.int32)
        self.downIndex = np.array([0, -1], np.int32)
        self.leftIndex = np.array([1, 0], np.int32)
        self.rightIndex = np.array([-1, 0], np.int32)

        self.maxima = maxima.astype(np.float32)
        self.averageMaximaDistance = averageMaximaDistance

        self.correspondences = list()

    def searchGrid(self):
        self.findMaxima(self.startingPoint, self.startingIndex, self.upVector, self.upIndex)
        self.findMaxima(self.startingPoint, self.startingIndex, self.leftVector, self.leftIndex)
        self.findMaxima(self.startingPoint, self.startingIndex, self.rightVector, self.rightIndex)
        self.findMaxima(self.startingPoint, self.startingIndex, self.downVector, self.downIndex)
        
        if len(self.correspondences) == 0:
            print("Couldn't find Maxima. Trying again.")
            return

        xMax = np.array(self.correspondences)[:, 0][:, 0].max()
        yMax = np.array(self.correspondences)[:, 0][:, 1].max()

        yDelta = 0
        xDelta = 0

        if xMax >= self.laserDims[0]:
            xDelta = xMax - self.laserDims[0] + 1
        
        if yMax >= self.laserDims[1]:
            yDelta = yMax - self.laserDims[1] + 1

        if xDelta > 0 or yDelta > 0:
            for i in range(len(self.correspondences)):
                self.correspondences[i][0] = self.correspondences[i][0] - np.array([xDelta, yDelta], dtype=np.int32)

        print("Found first Grid Estimate")

    def findMaxima(self, currentPoint, currentPointIndex, directionVector, indexVector):
        if len(self.maxima) == 0:
            return

        newPoint = currentPoint + self.averageMaximaDistance * directionVector

        neighbours = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(self.maxima)
        distances, indices = neighbours.kneighbors(newPoint.reshape(1, -1))

        if (newPoint < 0.0).any():
            return

        if distances[0] > self.epsilon:
            return

        nextPoint = self.maxima[indices[0]].copy()[0]
        nextGridIndex = currentPointIndex.copy() + indexVector
        self.correspondences.append([nextGridIndex, nextPoint])
        self.maxima = np.delete(self.maxima, indices[0], 0).astype(np.float32)
        
        self.findMaxima(nextPoint, nextGridIndex, self.upVector, self.upIndex)
        self.findMaxima(nextPoint, nextGridIndex, self.leftVector, self.leftIndex)
        self.findMaxima(nextPoint, nextGridIndex, self.rightVector, self.rightIndex)
        self.findMaxima(nextPoint, nextGridIndex, self.downVector, self.downIndex)


    def getCorrespondences(self):
        return self.correspondences



class RecursiveGridSearch:
    def __init__(self, maxima, startingPoints, averageMaximaDistance, image):
        self.upVector = np.array([-1.0, 0.0])
        self.downVector = np.array([1.0, 0.0])
        self.rightVector = np.array([0.0, 1.0])
        self.leftVector = np.array([0.0, -1.0])

        self.epsilon = 5.0

        self.upIndex = np.array([0, 1], np.int32)
        self.downIndex = np.array([0, -1], np.int32)
        self.leftIndex = np.array([1, 0], np.int32)
        self.rightIndex = np.array([-1, 0], np.int32)

        self.maxima = maxima.astype(np.float32)
        self.correspondences = startingPoints
        self.averageMaximaDistance = averageMaximaDistance
        self.test_image = image

    def searchGrid(self):
        neighbours = self.correspondences[1:]

        for neighbour in neighbours:
            self.findMaxima(np.expand_dims(neighbour[1], 0), neighbour[0], self.upVector, self.upIndex)
            self.findMaxima(np.expand_dims(neighbour[1], 0), neighbour[0], self.leftVector, self.leftIndex)
            self.findMaxima(np.expand_dims(neighbour[1], 0), neighbour[0], self.rightVector, self.rightIndex)
            self.findMaxima(np.expand_dims(neighbour[1], 0), neighbour[0], self.downVector, self.downIndex)

        print("Established Grid Estimate")

    def findMaxima(self, currentPoint, currentPointIndex, directionVector, indexVector):
        if len(self.maxima) == 0:
            return

        newPoint = currentPoint + self.averageMaximaDistance * directionVector

        neighbours = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(self.maxima)
        distances, indices = neighbours.kneighbors(newPoint)

        if (newPoint < 0.0).any():
            return

        if distances[0] > self.epsilon:
            return

        nextPoint = self.maxima[indices[0]].copy()
        nextGridIndex = currentPointIndex.copy() + indexVector
        self.correspondences.append([nextGridIndex, nextPoint[0]])
        self.maxima = np.delete(self.maxima, indices[0], 0).astype(np.float32)

        self.findMaxima(nextPoint, nextGridIndex, self.upVector, self.upIndex)
        self.findMaxima(nextPoint, nextGridIndex, self.leftVector, self.leftIndex)
        self.findMaxima(nextPoint, nextGridIndex, self.rightVector, self.rightIndex)
        self.findMaxima(nextPoint, nextGridIndex, self.downVector, self.downIndex)


    def getCorrespondences(self):
        return self.correspondences