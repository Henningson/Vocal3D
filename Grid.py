import numpy as np
from sklearn.neighbors import NearestNeighbors

class GridPoint:
    def __init__(self, laserX, laserY, pixelLocation):
        self.above = None
        self.right = None
        self.below = None
        self.left = None

        self.laserX = laserX
        self.laserY = laserY
        self.pixelLocation = pixelLocation


def findAbove(gridPoint, averageDistance, maxima):
    if gridPoint.above:
        return
            
    neighbours = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(maxima)
    distances, indices = neighbours.kneighbors(gridPoint.pixelLocation + np.array([0, 1])*averageDistance)

    
def findBelow(gridPoint, averageDistance, maxima):
    if gridPoint.below:
        return
            
    neighbours = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(maxima)
    distances, indices = neighbours.kneighbors(gridPoint.pixelLocation + np.array([0, 1])*averageDistance)

    
def findRight(gridPoint, averageDistance, maxima):
    if gridPoint.right:
        return
            
    neighbours = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(maxima)
    distances, indices = neighbours.kneighbors(gridPoint.pixelLocation + np.array([0, 1])*averageDistance)


def findLeft(gridPoint, averageDistance, maxima):
    if gridPoint.left:
        return
        
    neighbours = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(maxima)
    distances, indices = neighbours.kneighbors(gridPoint.pixelLocation + np.array([0, 1])*averageDistance)


def buildRecursiveLaserGrid(laserMaximaCorrespondences, regularCrossID, averageDistance):
    return None