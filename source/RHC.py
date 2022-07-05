import cv2
import numpy as np
from sklearn.neighbors import NearestNeighbors

import helper
import DiscreteGradientDescent
from GridSearch import PointBasedGridSearch


def globalAlignment(laserGridStuff, pixelLocations, maxima, laser):
    points = maxima.nonzero()
    maximaTransformed = np.concatenate([[points[0]], [points[1]]]).T
    neighbours = NearestNeighbors(n_neighbors=5, algorithm='ball_tree').fit(maximaTransformed)
    distances, indices = neighbours.kneighbors(maximaTransformed)
        
    points = maxima.nonzero()
    maximaTransformed = np.concatenate([[points[0]], [points[1]]]).T
    maximaPoints = maximaTransformed

    pixelDistance = helper.getAveragePixelDistance(distances)

    grid2DPixLocations = None
    for i in range(10):
        ind = np.random.choice(len(pixelLocations), 1, replace=False)
        start2D = pixelLocations[ind[0]]
        startLaser = laserGridStuff[ind[0]]

        gridSearcher = PointBasedGridSearch(maximaPoints, start2D, startLaser, pixelDistance, laser.getDims())
        gridSearcher.searchGrid()
        grid2DPixLocations = gridSearcher.getCorrespondences()
        grid2DPixLocBeforeRANSAC = grid2DPixLocations        
        if len(grid2DPixLocations) > len(maximaPoints) // 2:
            break

    return grid2DPixLocations


def RHC(laserGridStuff, pixelLocations, maxima, camera, laser):
    grid2DPixLocations = globalAlignment(laserGridStuff, pixelLocations, maxima, laser)   

    N = laser.gridHeight() // 2
    distMin, _ = helper.getPointOnRayFromOrigin(laser.origin(), laser.ray(N)/np.linalg.norm(laser.ray(N)), 40.0)
    distMax, _ = helper.getPointOnRayFromOrigin(laser.origin(), laser.ray(N)/np.linalg.norm(laser.ray(N)), 80.0)

    gridIDs = [x for x, _ in grid2DPixLocations]
    points2D = [x for _, x in grid2DPixLocations]

    DGD = DiscreteGradientDescent.DiscreteGradientDescent(camera, laser, points2D, gridIDs)
    indexUpdateVector = DGD.RANSAC(8, 30)

    grid2DPixLocations = list()
    for i in range(len(gridIDs)):
        grid2DPixLocations.append([gridIDs[i] + indexUpdateVector.astype(np.int64), points2D[i]])

    return grid2DPixLocations