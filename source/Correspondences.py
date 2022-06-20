import numpy as np
from sklearn.neighbors import NearestNeighbors

import helper


def initialize(laser, camera, maxima, image, minInterval, maxInterval):
    maxima = maxima.copy()
    points = maxima.nonzero()
    maximaTransformed = np.concatenate([[points[0]], [points[1]]]).T
    
    locations = list()
    ids = list()

    count = 0
    for x in range(0, laser.gridWidth(), 1):
        for y in range(laser.gridHeight() - 1, -1, -1):
            laserRay = laser.ray(y, x)

            mask = helper.generateMask(np.zeros((image.shape[0], image.shape[1]), np.uint8), camera.intrinsic(), laser.origin(), laserRay, minInterval, maxInterval,  2, 2)
            masked_maxima = maxima * mask
            masked_points = masked_maxima.nonzero()

            if masked_points[0].size <= 0:
                continue

            #Find lowest Y-Coordinate in Masked Maxima
            lowestYindex = np.argmax(masked_points[0])
            maximumY = masked_points[0][lowestYindex]
            maximumX = masked_points[1][lowestYindex]

            ids.append(np.array([y, x]))
            locations.append(np.array([maximumY, maximumX]))

            maxima[maximumY, maximumX] = 0

    return locations, ids


def generateFramewise(images, closedGlottisFrame, correspondenceEstimate, segmentation, distance_threshold = 5.0):
    neighbours = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(np.array(correspondenceEstimate)[:, 1])
    
    for i in range(closedGlottisFrame + 1, len(images)):
        image = images[i]

        maxima = helper.findLocalMaxima(image, 7)
        maxima = (segmentation & image) * maxima
        maxima = maxima.nonzero()
        maxima = np.concatenate([[maxima[0]], [maxima[1]]]).T
        distances, indices = neighbours.kneighbors(maxima)
        
        for k, index in enumerate(indices.squeeze()):
            if distances[k] < distance_threshold and len(correspondenceEstimate[index]) < i + 2:
                correspondenceEstimate[index].append(maxima[k])
        
        # Fill up missing values with nans
        maxFrames = 0
        for k in range(len(correspondenceEstimate)):
            maxFrames =  len(correspondenceEstimate[k]) if maxFrames <= len(correspondenceEstimate[k]) else maxFrames
        
        for k in range(len(correspondenceEstimate)):
            if len(correspondenceEstimate[k]) < maxFrames:
                for _ in range(maxFrames - len(correspondenceEstimate[k])):
                    correspondenceEstimate[k].append(np.array([np.nan, np.nan]))
        #print("Image {0}".format(i))
    return correspondenceEstimate

