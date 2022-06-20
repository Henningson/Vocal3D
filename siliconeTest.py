from tkinter import Label
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
import os.path
import LabelOffsetter

import SurfaceReconstruction
import scipy

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

    framesOfClosedGlottis = [21, 28, 26, 4, 13, 32, 30]#[27]#[38, 3, 29, 27, 9, 13, 18,
                            #[21, 28, 26, 4, 13, 32, 30]
                            # 44, 10, 20, 29, 19, 3, 19]
    
    calib_path = "/media/nu94waro/Seagate Expansion Drive/Promotion/Data/Rekonstruktion_Silikon/50_Kay/50_offset_corrected.mat"
    camera = Camera(calib_path, "MAT")
    laser = Laser(calib_path, "MAT")

    roiList = list()
    labelList = list()
    for frameOfClosedGlottis, image_path in zip(framesOfClosedGlottis, image_paths):
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

            pixelLocations, laserGridIDs = initializeCorrespondences(laser, camera, maxima, vocalfold_image, minInterval, maxInterval)
            grid2DPixLocations = globalAlignment(laserGridIDs, pixelLocations, maxima, laser, vocalfold_image)
            labelOff = LabelOffsetter.LabelOffsetter(vocalfold_image, grid2DPixLocations[:])
            newLabels = labelOff.label()

            okStr = input("Is labelling okay?")
            
            if okStr == "y":
                print("Labelling okay")
                break

        labelList.append(newLabels)
        roiList.append(np.array([x,y,w,h]))



    roiDecreaseIter = 20
    pxOffset = 5


    MSCompleteError = list()
    GACompleteError = list()
    RHCCompleteError = list()
    ROICompleteList = list()
    for frameOfClosedGlottis, image_path, roi, gtLabel in zip(framesOfClosedGlottis, image_paths, roiList, labelList):
        print("Image Path: {0}".format(image_path))
        print("Calibration File {0}".format(calib_path))

        images = helper.loadImages(image_path, camera.intrinsic(), camera.distortionCoefficients())

        width = images[0].shape[1]
        height = images[0].shape[0]

        countL = 0
        countR = 0
        countU = 0
        countD = 0
        i = 0

        msErrors = list()
        gaErrors = list()
        rhcErrors = list()
        rhc8Errors = list()
        rhc12Errors = list()
        roiList = list()
        while i < roiDecreaseIter:
            x = roi[0]
            y = roi[1]
            w = roi[2]
            h = roi[3]

            x = roi[0] + countL*pxOffset
            y = roi[1] + countU*pxOffset
            w = roi[2] - countR*pxOffset
            h = roi[3] - countD*pxOffset

            # Use ROI to generate mask image
            segmentation = np.zeros((height, width), dtype=np.uint8)
            segmentation[y:y+h, x:x+w] = 255

            vocalfold_image = images[frameOfClosedGlottis].copy()
            maxima = helper.findMaxima(vocalfold_image, segmentation)

            minInterval = 40.0#distances[numHits.index(max(numHits))] - 5.0
            maxInterval = 80.0#distances[numHits.index(max(numHits))] + 5.0
            N = laser.gridHeight() // 2
            distMin, _ = helper.getPointOnRayFromOrigin(laser.origin(), laser.ray(N)/np.linalg.norm(laser.ray(N)), minInterval)
            distMax, _ = helper.getPointOnRayFromOrigin(laser.origin(), laser.ray(N)/np.linalg.norm(laser.ray(N)), maxInterval)

            pixMS, gridMS = initializeCorrespondencesNew(laser, camera, maxima, vocalfold_image, minInterval, maxInterval)
            corresRHC, corresGA = RHC(laserGridIDs, pixelLocations, maxima, camera, laser, None)
            corresRHC8, _ = RHC8(laserGridIDs, pixelLocations, maxima, camera, laser, None)
            corresRHC12, _ = RHC12(laserGridIDs, pixelLocations, maxima, camera, laser, None)


            pixMS = np.array(pixMS)
            gridMS = np.array(gridMS)

            pixGA = np.array(corresGA)[:, 1]
            gridGA = np.array(corresGA)[:, 0]

            pixRHC = np.array(corresRHC)[:, 1]
            gridRHC = np.array(corresRHC)[:, 0]

            pixRHC8 = np.array(corresRHC)[:, 1]
            gridRHC8 = np.array(corresRHC)[:, 0]

            pixRHC12 = np.array(corresRHC)[:, 1]
            gridRHC12 = np.array(corresRHC)[:, 0]

            test_image = cv2.resize(vocalfold_image.copy(), (2048, 1024))
            for pixel, grid in zip(pixRHC.tolist(), gridRHC.tolist()):
                cv2.circle(test_image, np.flip(np.array(pixel)).astype(np.uint32)*4, 3, (255, 0, 0), -1)
                cv2.putText(test_image, '{0},{1}'.format(int(grid[0]), int(grid[1])), np.flip(np.array(pixel)).astype(np.uint32)*4, cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            cv2.imshow("RHC Labelling", test_image)
            cv2.waitKey(0)

            #Find GTLabels, which correspond to the labels at the found pixel positions
            gtNeighbors = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(np.array(gtLabel)[:, 1])

            _, indicesMS = gtNeighbors.kneighbors(pixMS)
            _, indicesGA = gtNeighbors.kneighbors(pixGA)
            _, indicesRHC = gtNeighbors.kneighbors(pixRHC)
            _, indicesRHC8 = gtNeighbors.kneighbors(pixRHC8)
            _, indicesRHC12 = gtNeighbors.kneighbors(pixRHC12)

            compareMS = np.array(gtLabel)[indicesMS.squeeze(), 0]
            compareGA = np.array(gtLabel)[indicesGA.squeeze(), 0]
            compareRHC = np.array(gtLabel)[indicesRHC.squeeze(), 0]
            compareRHC8 = np.array(gtLabel)[indicesRHC8.squeeze(), 0]
            compareRHC12 = np.array(gtLabel)[indicesRHC12.squeeze(), 0]

            MSL1 = np.sum(np.linalg.norm(gridMS - compareMS, ord=1, axis=1)) / gridMS.shape[0]
            GAL1 = np.sum(np.linalg.norm(gridGA - compareGA, ord=1, axis=1)) / gridGA.shape[0]
            RHCL1 = np.sum(np.linalg.norm(gridRHC - compareRHC, ord=1, axis=1)) / gridRHC.shape[0]
            RHC8L1 = np.sum(np.linalg.norm(gridRHC8 - compareRHC, ord=1, axis=1)) / gridRHC8.shape[0]
            RHC12L1 = np.sum(np.linalg.norm(gridRHC12 - compareRHC, ord=1, axis=1)) / gridRHC12.shape[0]
            
            msErrors.append(MSL1)
            gaErrors.append(GAL1)
            rhcErrors.append(RHCL1)
            rhc8Errors.append(RHC8L1)
            rhc12Errors.append(RHC12L1)
            roiList.append(np.array([x, y, w, h]))

            print("ROI: {0} {1} {2} {3}".format(x, y, w, h))
            print("After Mask Sweep: {0}".format(MSL1))
            print("After Global Alignment: {0}".format(GAL1))
            print("After RHC: {0}".format(RHCL1))
            print("After RHC8: {0}".format(RHCL1))
            print("After RHC12: {0}".format(RHCL1))
            
            if i % 4 == 0:
                countL += 1
                countR += 1
            if i % 4 == 1:
                countU += 1
                countD += 1
            if i % 4 == 2:
                countR += 1
            if i % 4 == 3:
                countD += 1
            i += 1

        #csvFile = open("test.csv","a")
        #for ms, ga, rhc, rhc8, rhc12, roi in zip(msErrors, gaErrors, rhcErrors, rhc8Errors, rhc12Errors, roiList): 
        #    csvFile.write("{0} {1} {2} {3} {4} {5} {6}\n".format(image_path.split("/")[-3], ms, ga, rhc, rhc8, rhc12, roi))
        #csvFile.close()