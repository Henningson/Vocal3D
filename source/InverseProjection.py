import numpy as np
import cv2
import math

import matplotlib.pyplot as plt
import specularity as spc

import AutomaticSegmentation

from mpl_toolkits.mplot3d import Axes3D
from sklearn.neighbors import NearestNeighbors

from Laser import Laser
from Camera import Camera
from GridSearch import RecursiveGridSearch, PointBasedGridSearch
import visualization

import SegmentationClicker
import argparse

import DiscreteGradientDescent

import helper
import Timer

def initializeCorrespondences(laser, camera, maxima, image, minInterval, maxInterval, distMin, distMax):
    points = maxima.nonzero()
    maximaTransformed = np.concatenate([[points[0]], [points[1]]]).T
    neighbours = NearestNeighbors(n_neighbors=5, algorithm='ball_tree').fit(maximaTransformed)
    distances, indices = neighbours.kneighbors(maximaTransformed)

    laserMaximaCorrespondences = list()
    points3D = list()
    
    for count, laserRay in enumerate(laser.rays().tolist()):
        laserRay = np.array(laserRay)

        mask = helper.generateMask(np.zeros((image.shape[0], image.shape[1]), np.uint8), camera.intrinsic(), laser.origin(), laserRay, minInterval, maxInterval,  2, 2)
        masked_maxima = maxima * mask
        masked_points = masked_maxima.nonzero()

        pointDistances = list()
        aPoints = list()
        bPoints = list()

        for local_maximum_y, local_maximum_x in zip(masked_points[0], masked_points[1]):
            cameraRay = camera.getRay(np.array([local_maximum_x, local_maximum_y]))
            pointA, pointB, distance = helper.LineLineIntersection(cameraRay*minInterval, cameraRay*maxInterval, laser.origin() + distMin[0]*laserRay, laser.origin() + distMax[0]*laserRay)
            
            #visualization.plotLaserRaysCameraRayHits(laser.origin(), laserRay, cameraRay, pointA, pointB-pointA)

            worldPos = pointA + ((pointB - pointA) / 2.0)
            points3D.append(worldPos)
            pointDistances.append(distance)
            aPoints.append(pointA)
            bPoints.append(pointB)

        if not pointDistances:
            continue

        min_value = min(pointDistances)
        min_index = pointDistances.index(min_value)
        arrayID = helper.findID(points, masked_points, min_index)

        pointInCross, directions = helper.isCross(maximaTransformed[indices[arrayID][0]], maximaTransformed[indices[arrayID][1]], maximaTransformed[indices[arrayID][2]], maximaTransformed[indices[arrayID][3]], maximaTransformed[indices[arrayID][4]])
        laserMaximaCorrespondences.append([laser.getXYfromN(count), [masked_points[0][min_index], masked_points[1][min_index]], arrayID, indices[arrayID], pointInCross, directions])
    print("Num Points found: {0}".format(len(points3D)))
    #visualization.plotPoints3D(points3D)
    return laserMaximaCorrespondences


def gridBasedCorrespondenceEstimate(correspondences, image, maxima, minInterval, maxInterval, distMin, distMax):
    points = maxima.nonzero()
    maximaTransformed = np.concatenate([[points[0]], [points[1]]]).T
    neighbours = NearestNeighbors(n_neighbors=5, algorithm='ball_tree').fit(maximaTransformed)
    distances, indices = neighbours.kneighbors(maximaTransformed)

    points3D = list()
    grid2DPixLocations = list()
    for correspondence in correspondences:
        laserGridID = correspondence[0]
        point2D = correspondence[1]
        arrayID = correspondence[2]
        neighbours = correspondence[3]
        insideCross = correspondence[4]
        directions = correspondence[5]
        
        points = maxima.nonzero()
        maximaTransformed = np.concatenate([[points[0]], [points[1]]]).T
        maximaPoints = maximaTransformed

        if laserGridID[0] >= laser.gridWidth() or laserGridID[1] >= laser.gridHeight():
            continue

        if not insideCross:
            continue
        
        isRegular, gridCorrespondences = helper.isRegularGrid(point2D, laserGridID, neighbours[1:], directions, maximaTransformed, laserMaximaCorrespondences)
        if not isRegular:
            continue

        pixelDistance = helper.getAveragePixelDistance(distances)

        #Delete first few starting points, as we already found correspondences for those
        for index in neighbours:
            maximaPoints = np.delete(maximaPoints, index, 0)

        #Converting stuff for it to work smoothly
        for gridCorrespondence in gridCorrespondences:
            gridCorrespondence[1] = gridCorrespondence[1].astype(np.float32)
            gridCorrespondence[0] = np.array(gridCorrespondence[0])


        gridSearcher = RecursiveGridSearch(maximaPoints, gridCorrespondences, pixelDistance, cv2.resize(cv2.cvtColor(image, cv2.COLOR_GRAY2BGR), (2048, 1024)))
        gridSearcher.searchGrid()
        grid2DPixLocations = gridSearcher.getCorrespondences()
        points3D = list()

        if (np.array(grid2DPixLocations)[:, 0] < 0).any() or (np.array(grid2DPixLocations)[:, 0] >= laser.gridHeight()).any():
            continue

        for gridPos, pixelPosition in grid2DPixLocations:
            cameraRay = camera.getRay(np.flip(pixelPosition))
            laserRay = laser.ray(gridPos[0], gridPos[1])
            pointA, pointB, distance = helper.LineLineIntersection(cameraRay*minInterval, cameraRay*maxInterval, laser.origin() + distMin[0]*laserRay, laser.origin() + distMax[0]*laserRay)
            
            if distance > 5.0:
                continue

            worldPos = pointA + ((pointB - pointA) / 2.0)
            points3D.append(worldPos)

        break

    print("Found Vocal Folds in interval: [{0}, {1}]".format(minInterval, maxInterval))
    #print(len(points3D))
    #visualization.plotPoints3D(points3D)
    return grid2DPixLocations


def generateFramewiseCorrespondences(images, closedGlottisFrame, correspondenceEstimate, distance_threshold = 5.0):
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


def triangulation(camera, laser, framewiseCorrespondences, minInterval, maxInterval, distMin, distMax):
    points3D = list()
    for k, framewiseCorrespondence in enumerate(framewiseCorrespondences):
        frameWise3D = list()
        for i in range(1, len(framewiseCorrespondence)):
            if np.isnan(framewiseCorrespondence[i]).any():
                frameWise3D.append(np.array([np.nan, np.nan, np.nan]))
                continue

            cameraRay = camera.getRay(np.array([framewiseCorrespondence[i][1], framewiseCorrespondence[i][0]]))
            laserRay = laser.ray(framewiseCorrespondence[0][0], framewiseCorrespondence[0][1])
            pointA, pointB, distance = helper.LineLineIntersection(cameraRay*minInterval, cameraRay*maxInterval, laser.origin() + distMin[0]*laserRay, laser.origin() + distMax[0]*laserRay)

            if distance > 5.0:
                frameWise3D.append(np.array([np.nan, np.nan, np.nan]))
                continue

            frameWise3D.append(pointA + ((pointB - pointA) / 2.0))
        points3D.append(frameWise3D)
        print("Triangulating Image: {0}".format(k))
    visualization.show_3d_triangulation(points3D)
    #visualization.write_images("epipolarConstraints/MK", points3D)

def triangulationMat(camera, laser, framewiseCorrespondences, minInterval, maxInterval, distMin, distMax):
    
    points3D = list()

    gridIDs = np.array(framewiseCorrespondences)[:, 0].astype(np.int32)
    laser2DPositions = np.array(framewiseCorrespondences)[:, 1:]


    for k in range(laser2DPositions.shape[1]):
        positions2D = laser2DPositions[:, k, :]

        cameraRays = camera.getRayMat(np.flip(positions2D, axis=1)) # The error is here!
        linearizedIDs = laser.getNfromXY(gridIDs[:, 0], gridIDs[:, 1])
        laserRays = laser.rays()[linearizedIDs]
        origin = np.expand_dims(laser.origin(), 0)

        aPoints, bPoints, distances = helper.MatLineLineIntersection(cameraRays*minInterval, cameraRays*maxInterval, origin + distMin*laserRays, origin + distMax*laserRays)

        worldPositions = aPoints + (bPoints - aPoints) / 2.0
        
        #Filter NaNs and badly estimated 3-D Positions
        worldPositions = worldPositions[distances < 5.0]

        points3D.append((aPoints + ((bPoints - aPoints) / 2.0)).tolist())

    return points3D


def calc_overlap(images, segmentation):
    count_img = np.zeros(images[0].shape, dtype=np.float32)

    for image in images:
        maxima = helper.findMaxima(image, segmentation)
        maxima = np.where(maxima > 0, 1, 0).astype(np.uint8)
        M = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5), (2, 2))
        dilated_maxima = cv2.dilate(maxima, M)
        count_img += dilated_maxima

    count_img = (AutomaticSegmentation.normalize(count_img)*255).astype(np.uint8)
    cv2.imshow("Overlap", count_img)
    cv2.waitKey(0)


def test(laserGridStuff, pixelLocations, maxima, camera, laser, debugImg=None):
    points = maxima.nonzero()
    maximaTransformed = np.concatenate([[points[0]], [points[1]]]).T
    neighbours = NearestNeighbors(n_neighbors=5, algorithm='ball_tree').fit(maximaTransformed)
    distances, indices = neighbours.kneighbors(maximaTransformed)
        
    points = maxima.nonzero()
    maximaTransformed = np.concatenate([[points[0]], [points[1]]]).T
    maximaPoints = maximaTransformed

    pixelDistance = helper.getAveragePixelDistance(distances)

    ind = np.random.choice(len(pixelLocations), 1, replace=False)
    start2D = pixelLocations[ind[0]]
    startLaser = laserGridStuff[ind[0]]

    gridSearcher = PointBasedGridSearch(maximaPoints, start2D, startLaser, pixelDistance, cv2.resize(cv2.cvtColor(debugImg, cv2.COLOR_GRAY2BGR), (512, 1024)))
    gridSearcher.searchGrid()
    grid2DPixLocations = gridSearcher.getCorrespondences()


    points3D = list()
    sumDistance = 0.0
    for gridPos, pixelPosition in grid2DPixLocations:
        cameraRay = camera.getRay(np.flip(pixelPosition))
        laserRay = laser.ray(gridPos[0], gridPos[1])
        pointA, pointB, distance = helper.LineLineIntersection(cameraRay*minInterval, cameraRay*maxInterval, laser.origin() + distMin[0]*laserRay, laser.origin() + distMax[0]*laserRay)
        sumDistance += distance
        worldPos = pointA + ((pointB - pointA) / 2.0)
        points3D.append(worldPos)

    gridIDs = [x for x, _ in grid2DPixLocations]
    points2D = [x for _, x in grid2DPixLocations]

    bla = DiscreteGradientDescent.DiscreteGradientDescent(camera, laser, points2D, gridIDs)
    indexUpdateVector = bla.RANSAC(5, 50)

    print("Updated 3D Coords")
    points3D = list()
    sumDistance = 0.0
    for gridPos, pixelPosition in grid2DPixLocations:
        cameraRay = camera.getRay(np.flip(pixelPosition))
        gridPos += indexUpdateVector.astype(np.int64)
        laserRay = laser.ray(gridPos[0], gridPos[1])
        pointA, pointB, distance = helper.LineLineIntersection(cameraRay*minInterval, cameraRay*maxInterval, laser.origin() + distMin[0]*laserRay, laser.origin() + distMax[0]*laserRay)
        sumDistance += distance
        worldPos = pointA + ((pointB - pointA) / 2.0)
        points3D.append(worldPos)

    grid2DPixLocations = list()
    for i in range(len(gridIDs)):
        grid2DPixLocations.append([gridIDs[i] + indexUpdateVector.astype(np.int64), points2D[i]])

    return grid2DPixLocations

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="(Semi-)Automatic triangulation of High Speed Video of Vocal Folds")
    parser.add_argument('--calibration_file', '-c', type=str, required=True, help="Path to a calibration .MAT or .JSON File")
    parser.add_argument('--image_path', '-i', type=str, required=True, help="Path to a folder containing the Vocal Fold images.")
    parser.add_argument('--silicone', '-s', default=False, action='store_true', help="Start semiautomatic mode, where a Segmentation is generated by hand")
    parser.add_argument('--verbose', '-v', default=False, action='store_true', help="Show debugging output.")

    args = parser.parse_args()

    plotAndLog = args.verbose
    image_path = args.image_path
    calib_path = args.calibration_file
    isSiliconeVocalfold = args.silicone

    print("Image Path: {0}".format(image_path))
    print("Calibration File {0}".format(calib_path))
    print("Silicone Vocalfold? {0}".format(isSiliconeVocalfold))
    print("Plotting active? {0}".format(plotAndLog))

    camera = Camera(calib_path, "MAT")
    laser = Laser(calib_path, "MAT")
    images = helper.loadImages(image_path, camera.intrinsic(), camera.distortionCoefficients())

    width = images[0].shape[1]
    height = images[0].shape[0]

    segmentation = None
    x, y, w, h = [None, None, None, None]
    segmentator = None

    if isSiliconeVocalfold:
        images = images[6:]
        # Get Estimate where the Vocalfolds are
        segmentator = AutomaticSegmentation.HSVGlottisSegmentator(images[:100])
        sc = SegmentationClicker.SegmentationClicker(images[0])
        sc.clickSegmentation()
        segmentator.roiWidth = sc.roiWidth
        segmentator.roiHeight = sc.roiHeight
        segmentator.roiX = sc.roiX
        segmentator.roiY = sc.roiY

        segmentator.generate(isSilicone=True, plot=plotAndLog)
        x, w, y, h = segmentator.getROI()
    else:
        segmentator = AutomaticSegmentation.HSVGlottisSegmentator(images[:100])
        segmentator.generate(isSilicone=False, plot=plotAndLog)
        x, w, y, h = segmentator.getROI()
        x = x+w//2 - w*2
        w = 4*w


    # Use ROI to generate mask image
    segmentation = np.zeros((height, width), dtype=np.uint8)
    segmentation[y:y+h, x:x+w] = 255

    if plotAndLog:
        cv2.imshow("ROI", images[0][y:y+h, x:x+w])
        cv2.waitKey(0)

    frameOfClosedGlottis = segmentator.estimateClosedGlottis()
    print("Found closed Glottis at Frame: {0}".format(frameOfClosedGlottis))
    totalNumberOfFrames = len(images)

    if plotAndLog:
        cv2.imshow("Closed Glottis Frame", images[frameOfClosedGlottis])
        cv2.waitKey(0)

    vocalfold_image = images[frameOfClosedGlottis]

    maxima = helper.findMaxima(vocalfold_image, segmentation)

    if plotAndLog:
        cv2.imshow("Maxima", maxima)
        cv2.waitKey(0)

    distances, numHits = helper.findOverlap(laser, camera, maxima)

    minInterval = distances[numHits.index(max(numHits))] - 5.0
    maxInterval = distances[numHits.index(max(numHits))] + 5.0
    N = laser.gridHeight() // 2
    distMin, _ = helper.getPointOnRayFromOrigin(laser.origin(), laser.ray(N)/np.linalg.norm(laser.ray(N)), minInterval)
    distMax, _ = helper.getPointOnRayFromOrigin(laser.origin(), laser.ray(N)/np.linalg.norm(laser.ray(N)), maxInterval)

    print("Estimated Vocalfolds in Interval: [{0}-{1}]".format(minInterval, maxInterval))

    laserMaximaCorrespondences = initializeCorrespondences(laser, camera, maxima, vocalfold_image, minInterval, maxInterval, distMin, distMax)

    laserGridIDs = [ids for ids, _, _, _, _, _ in laserMaximaCorrespondences]
    pixelLocations = [loc for _, loc, _, _, _, _ in laserMaximaCorrespondences]
    
    grid2DPixLocations = None
    
    if isSiliconeVocalfold:
        grid2DPixLocations = gridBasedCorrespondenceEstimate(laserMaximaCorrespondences, vocalfold_image, maxima, minInterval, maxInterval, distMin, distMax)
    else:
        grid2DPixLocations = test(laserGridIDs, pixelLocations, maxima, camera, laser, vocalfold_image)

    temporalCorrespondence = generateFramewiseCorrespondences(images, frameOfClosedGlottis, grid2DPixLocations)
    
    triangulationMat(camera, laser, temporalCorrespondence, 40.0, 80.0, 40.0, 90.0)
    
    print("Done.")