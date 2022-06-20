from tkinter import Label
import numpy as np
import cv2
import math

import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D
from sklearn.neighbors import NearestNeighbors

from Laser import Laser
from Camera import Camera
from GridSearch import RecursiveGridSearch, PointBasedGridSearch
import visualization
import Correspondences
import Triangulation

import SegmentationClicker
import argparse

import BSplineVisualization

import DiscreteGradientDescent

import helper
import Timer
import os.path
import LabelOffsetter

import SurfaceReconstruction
import scipy

import RHC

from geomdl import BSpline, utilities
import matplotlib.pyplot as plt
from geomdl.visualization import VisMPL
from matplotlib import cm

if __name__ == "__main__":
    #parser = argparse.ArgumentParser(description="(Semi-)Automatic triangulation of High Speed Video of Vocal Folds")
    #parser.add_argument('--calibration_file', '-c', type=str, required=True, help="Path to a calibration .MAT or .JSON File")
    #parser.add_argument('--image_path', '-i', type=str, required=True, help="Path to a folder containing the Vocal Fold images.")
    #parser.add_argument('--verbose', '-v', default=False, action='store_true', help="Show debugging output.")
    #parser.add_argument('--closed_glottis', '-g', type=int, required=True, help="Frame Number of Closed Glottis")

    #args = parser.parse_args()

    #plotAndLog = args.verbose
    #image_path = args.image_path
    #calib_path = args.calibration_file
    #frameOfClosedGlottis = args.closed_glottis

    image_paths = [#"/media/nu94waro/Seagate Expansion Drive/Promotion/Data/Rekonstruktion_Silikon/50_Kay/Kay_50_-15_M2/png/"#,
                   #"/media/nu94waro/Seagate Expansion Drive/Promotion/Data/Rekonstruktion_Silikon/50_Kay/Kay_50_-10_M2/png/",
                   #"/media/nu94waro/Seagate Expansion Drive/Promotion/Data/Rekonstruktion_Silikon/50_Kay/Kay_50_-5_M2/png/",
                   #"/media/nu94waro/Seagate Expansion Drive/Promotion/Data/Rekonstruktion_Silikon/50_Kay/Kay_50_0_M2/png/"#,
                   #"/media/nu94waro/Seagate Expansion Drive/Promotion/Data/Rekonstruktion_Silikon/50_Kay/Kay_50_5_M2/png/",
                   #"/media/nu94waro/Seagate Expansion Drive/Promotion/Data/Rekonstruktion_Silikon/50_Kay/Kay_50_10_M2/png/",
                   #"/media/nu94waro/Seagate Expansion Drive/Promotion/Data/Rekonstruktion_Silikon/50_Kay/Kay_50_15_M2/png/",
                   "/media/nu94waro/Seagate Expansion Drive/Promotion/Data/Rekonstruktion_Silikon/65_Kay/Kay_65_M2_-15/png/",
                   "/media/nu94waro/Seagate Expansion Drive/Promotion/Data/Rekonstruktion_Silikon/65_Kay/Kay_65_M2_-10/png/",
                   "/media/nu94waro/Seagate Expansion Drive/Promotion/Data/Rekonstruktion_Silikon/65_Kay/Kay_65_M2_-5/png/",
                   "/media/nu94waro/Seagate Expansion Drive/Promotion/Data/Rekonstruktion_Silikon/65_Kay/Kay_65_M2_0/png/",
                   "/media/nu94waro/Seagate Expansion Drive/Promotion/Data/Rekonstruktion_Silikon/65_Kay/Kay_65_M2_5/png/",
                   "/media/nu94waro/Seagate Expansion Drive/Promotion/Data/Rekonstruktion_Silikon/65_Kay/Kay_65_M2_10/png/",
                   "/media/nu94waro/Seagate Expansion Drive/Promotion/Data/Rekonstruktion_Silikon/65_Kay/Kay_65_M2_15/png/"
                   #"/media/nu94waro/Seagate Expansion Drive/Promotion/Data/Rekonstruktion_Silikon/80_Kay/80_Kay_-15_M2_gecklickt/png/",
                   #"/media/nu94waro/Seagate Expansion Drive/Promotion/Data/Rekonstruktion_Silikon/80_Kay/80_Kay_-10_M2_gecklickt/png/",
                   #"/media/nu94waro/Seagate Expansion Drive/Promotion/Data/Rekonstruktion_Silikon/80_Kay/80_kay_-05_M2_gecklickt/png/",
                   #"/media/nu94waro/Seagate Expansion Drive/Promotion/Data/Rekonstruktion_Silikon/80_Kay/80_Kay_0_M2_gecklickt/png/",
                   #"/media/nu94waro/Seagate Expansion Drive/Promotion/Data/Rekonstruktion_Silikon/80_Kay/80_kay_05_M2_gecklickt/png/",
                   #"/media/nu94waro/Seagate Expansion Drive/Promotion/Data/Rekonstruktion_Silikon/80_Kay/80_Kay_10_M2_gecklickt/png/",
                   #"/media/nu94waro/Seagate Expansion Drive/Promotion/Data/Rekonstruktion_Silikon/80_Kay/80_Kay_15_M2_gecklickt/png/"
                   ]

    framesOfClosedGlottis = [38, 3, 29, 27, 9, 13, 18,
                            21, 28, 26, 4, 13, 32, 30]
                             44, 10, 20, 29, 19, 3, 19]
    
    calib_path = "/media/nu94waro/Seagate Expansion Drive/Promotion/Data/Rekonstruktion_Silikon/50_Kay/50_offset_corrected.mat"
    camera = Camera(calib_path, "MAT")
    laser = Laser(calib_path, "MAT")

    roiList = list()
    labelList = list()
    for frameOfClosedGlottis, image_path in zip(framesOfClosedGlottis, image_paths):
        segmentation = np.zeros((height, width), dtype=np.uint8)
        while True:
            images = helper.loadImages(image_path, camera.intrinsic(), camera.distortionCoefficients())
            # Get Estimate where the Vocalfolds are
            sc = SegmentationClicker.SegmentationClicker(images[frameOfClosedGlottis].copy())
            sc.clickSegmentation()
            w = sc.roiWidth
            h = sc.roiHeight
            x = sc.roiX
            y = sc.roiY
            
            height = images[frameOfClosedGlottis].shape[0]
            width = images[frameOfClosedGlottis].shape[1]

            # Use ROI to generate mask image
            segmentation = np.zeros((height, width), dtype=np.uint8)
            segmentation[y:y+h, x:x+w] = 255

            vocalfold_image = images[frameOfClosedGlottis]
            maxima = helper.findMaxima(vocalfold_image, segmentation)

            minInterval = 40.0#distances[numHits.index(max(numHits))] - 5.0
            maxInterval = 80.0#distances[numHits.index(max(numHits))] + 5.0
            N = laser.gridHeight() // 2
            distMin, _ = helper.getPointOnRayFromOrigin(laser.origin(), laser.ray(N)/np.linalg.norm(laser.ray(N)), minInterval)
            distMax, _ = helper.getPointOnRayFromOrigin(laser.origin(), laser.ray(N)/np.linalg.norm(laser.ray(N)), maxInterval)

            pixelLocations, laserGridIDs = Correspondences.initialize(laser, camera, maxima, vocalfold_image, minInterval, maxInterval)
            grid2DPixLocations = RHC.globalAlignment(laserGridIDs, pixelLocations, maxima, laser, vocalfold_image)
            labelOff = LabelOffsetter.LabelOffsetter(vocalfold_image, grid2DPixLocations[:])
            newLabels = labelOff.label()

            okStr = input("Is labelling okay?")
            
            if okStr == "y":
                print("Labelling okay")
                break

        labelList.append(newLabels)
        roiList.append(segmentation)

    
    for frameOfClosedGlottis, image_path, segmentation, gtLabel in zip(framesOfClosedGlottis, image_paths, roiList, labelList):

        entation = np.zeros((height, width), dtype=np.uint8)
        segmentation[y:y+h, x:x+w] = 255

        temporalCorrespondence = Correspondences.generateFramewise(images, frameOfClosedGlottis, grid2DPixLocations, segmentation)
        triangulatedPoints = np.array(Triangulation.triangulationMat(camera, laser, temporalCorrespondence, 40.0, 80.0, 40.0, 90.0))
        triangulatedPoints = triangulatedPoints[:50]

        zSubdivisions = 8
        leftDeformed, rightDeformed, leftPoints, rightPoints = SurfaceReconstruction.controlPointBasedARAP(triangulatedPoints, laser, images, camera, segmentator, frameOfClosedGlottis, zSubdivisions=zSubdivisions)

        optimizedLeft = SurfaceReconstruction.surfaceOptimization(leftDeformed, leftPoints, zSubdivisions=zSubdivisions, iterations=10, lr=0.01)
        optimizedLeft = np.array(optimizedLeft)
        smoothedLeft = scipy.ndimage.uniform_filter(optimizedLeft, size=(7, 1, 1), mode='reflect', cval=0.0, origin=0)

        optimizedRight = SurfaceReconstruction.surfaceOptimization(rightDeformed, rightPoints, zSubdivisions=zSubdivisions, iterations=10, lr=0.01)
        optimizedRight = np.array(optimizedRight)
        smoothedRight = scipy.ndimage.uniform_filter(optimizedRight, size=(7, 1, 1), mode='reflect', cval=0.0, origin=0)
    #
        BSplineVisualization.visualizeBM5(smoothedLeft, smoothedRight, zSubdivisions, leftPoints, rightPoints)