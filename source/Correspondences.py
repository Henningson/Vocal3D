import numpy as np
import cv2
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
    xy = np.array([[[-1.0, -1.0], [-1.0, 0.0], [-1.0, 1.0]], [[0.0, -1.0], [0.0, 0.0], [0.0, 1.0]], [[1.0, -1.0], [1.0, 0.0], [1.0, 1.0]]])
    
    for i in range(closedGlottisFrame + 1, len(images)):
        image = images[i]

        maxima = helper.findLocalMaxima(image, 7)
        maxima = (segmentation & image) * maxima
        maxima_vec = maxima.nonzero()
        maxima_vec = np.concatenate([[maxima_vec[0]], [maxima_vec[1]]]).T

        y = (maxima_vec[:, :1] + np.arange(-1, 2))[:, :, None]
        x = (maxima_vec[:, 1:] + np.arange(-1, 2))[:, None, :]

        batched_weights = image[y, x] 

        batched_coords = np.repeat(np.expand_dims(xy, 0), batched_weights.shape[0], axis=0)
        batched_weights = np.expand_dims(batched_weights, -1)

        weightbased_pixel_offsets = (batched_coords * batched_weights).sum(axis=(1,2)) / batched_weights.sum(axis=(1,2))
        maxima_vec = maxima_vec + (np.array([[0.5, 0.5]]) + weightbased_pixel_offsets)

        distances, indices = neighbours.kneighbors(maxima_vec)

        indices = indices.squeeze()
        distances = distances.squeeze()
        
        for k, index in enumerate(indices):
            if distances[k] < distance_threshold and len(correspondenceEstimate[index]) < i + 2:
                correspondenceEstimate[index].append(maxima_vec[k])
        
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

