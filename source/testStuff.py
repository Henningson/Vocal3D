from matplotlib.pyplot import grid
import numpy as np

import sys
sys.path.append(".")

from VocalfoldHSVSegmentation import vocalfold_segmentation
import SiliconeSegmentator

from Laser import Laser
from Camera import Camera
import argparse

import helper
import chamfer

import BSplineVisualization
import SurfaceReconstruction
import SiliconeSurfaceReconstruction
import scipy

import RHC
import VoronoiRHC
import Triangulation
import Correspondences
import Objects
import Intersections

import matplotlib.pyplot as plt
import cv2

from sklearn.decomposition import PCA
import visualization

from PyQt5.QtWidgets import QApplication
import Viewer


def testGeneratingLaserBeams(camera, laser, local_maximum, minDistance, maxDistance, imageHeight, imageWidth):
    minPlane = Objects.Plane(np.array([[0.0, 0.0, 1.0]]), np.array([[0.0, 0.0, minDistance]]))
    maxPlane = Objects.Plane(np.array([[0.0, 0.0, 1.0]]), np.array([[0.0, 0.0, maxDistance]]))
    ray = Objects.Ray(laser.origin(), laser.rays())

    points3Dmin = laser.origin() + minPlane.rayIntersection(ray) * ray._direction
    points3Dmax = laser.origin() + maxPlane.rayIntersection(ray) * ray._direction

    points2Dmin = helper.project3DPointToImagePlaneMat(points3Dmin, camera.intrinsic())
    points2Dmax = helper.project3DPointToImagePlaneMat(points3Dmax, camera.intrinsic())


    for i in range(points2Dmin.shape[0]):
        image = np.zeros((imageHeight, imageWidth), dtype=np.uint8)
        dist = Intersections.pointLineSegmentDistance(points2Dmin[i][:2], points2Dmax[i][:2], np.flip(local_maximum))
        cv2.line(image, points2Dmin[i].astype(np.int), points2Dmax[i].astype(np.int), color=255, thickness=2)
        cv2.circle(image, np.flip(local_maximum).astype(np.int), radius=2, color=255, thickness=-1)
        print("PointLine Segment Distance, for Beam {0}: {1}".format(laser.getXYfromN(i), dist))
        cv2.imshow("image", 255 - image)
        cv2.waitKey(0)


def findGeneratingLaserbeams(camera, laser, local_maximum, min_distance, max_distance, height, width):
    indexList = []

    minPlane = Objects.Plane(np.array([[0.0, 0.0, 1.0]]), np.array([[0.0, 0.0, min_distance]]))
    maxPlane = Objects.Plane(np.array([[0.0, 0.0, 1.0]]), np.array([[0.0, 0.0, max_distance]]))
    ray = Objects.Ray(laser.origin(), laser.rays())

    points3Dmin = laser.origin() + minPlane.rayIntersection(ray) * ray._direction
    points3Dmax = laser.origin() + maxPlane.rayIntersection(ray) * ray._direction

    points2Dmin = helper.project3DPointToImagePlaneMat(points3Dmin, camera.intrinsic())
    points2Dmax = helper.project3DPointToImagePlaneMat(points3Dmax, camera.intrinsic())


    for i in range(points2Dmin.shape[0]):
        image = np.zeros((height, width), dtype=np.uint8)
        dist = Intersections.pointLineSegmentDistance(points2Dmin[i][:2], points2Dmax[i][:2], np.flip(local_maximum))

        if dist < 5.0:
            indexList.append(laser.getXYfromN(i))

    return indexList


def topologyMeasure(point_cloud):

    centroid = np.sum(point_cloud, axis=0) / point_cloud.shape[0]
    centered_point_cloud = point_cloud - centroid

    #return centered_point_cloud

    pca = PCA(3)
    low_d = pca.fit_transform(centered_point_cloud.T)

    return centered_point_cloud / np.expand_dims(np.linalg.norm(low_d, axis=1), 0)


def generateMisalignings(grid2DPixLocations, camera, laser, height, width):

    random_index = np.random.randint(0, len(grid2DPixLocations))
    listOfViableLaserbeams = findGeneratingLaserbeams(camera, laser, grid2DPixLocations[random_index][1], 40.0, 80.0, height, width)
    optimized_index = grid2DPixLocations[random_index][0]

    gridIDs = np.array([x for x, _ in grid2DPixLocations])
    pixIDs = np.array([y for _, y in grid2DPixLocations])

    print(listOfViableLaserbeams)
    print(optimized_index)
    
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    point_clouds = list()
    for viableIndexing in listOfViableLaserbeams:
        offset = optimized_index - np.array(viableIndexing)
        cameraRays = camera.getRayMat(np.flip(pixIDs, axis=1)) # The error is here!
        linearizedIDs = laser.getNfromXY(gridIDs[:, 0] - offset[0], gridIDs[:, 1] - offset[1])
        
        try:
            laser.rays()[linearizedIDs]
        except:
            continue
        
        laserRays = laser.rays()[linearizedIDs]
        origin = np.expand_dims(laser.origin(), 0)

        aPoints, bPoints, distances = helper.MatLineLineIntersection(cameraRays*40.0, cameraRays*80.0, origin + 0.0*laserRays, origin + 1000.0*laserRays)

        worldPositions = aPoints + (bPoints - aPoints) / 2.0
        pcad_points = topologyMeasure(worldPositions)
        point_clouds.append(pcad_points)

        ax.scatter(pcad_points[:, 0], pcad_points[:, 1], pcad_points[:, 2])
        #np.savetxt("{0}_{1}.xyz".format(offset[0], offset[1]), worldPositions, delimiter=" ", fmt="%s")


    average_chamfer_dist = 0.0
    count = 0
    for point_cloud_a in point_clouds:
        for point_cloud_b in point_clouds:
            if (point_cloud_a == point_cloud_b).all():
                continue
            
            chamfer_dist = chamfer.chamfer_distance(point_cloud_a, point_cloud_b)
            print("Chamfer Distance between scaled Point Clouds: {0}".format(chamfer_dist))
            average_chamfer_dist += chamfer_dist
            count += 1

    print("Average Chamfer Distance: {0}".format(average_chamfer_dist / count))


    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.show()
