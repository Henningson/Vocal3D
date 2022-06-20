import numpy as np
import cv2
import helper
import Laser
import Camera

class DiscreteGradientDescent:
    def __init__(self, camera, laser, pixelEstimates, gridEstimates):
        self.camera = camera
        self.laser = laser
        self.pixelEstimates = pixelEstimates
        self.gridEstimates = gridEstimates
        self.visited = list()

        self.upVector = np.array([0.0, 1.0])
        self.downVector = np.array([0.0, -1.0])
        self.rightVector = np.array([-1.0, 0.0])
        self.leftVector = np.array([1.0, 0.0])
        self.errors = list()
        self.movementVectors = list()

        self.minimalError =  100000000000000.0
        self.minimalDirection = None

    def calc_error(self, movementVector):
        error = 0.0

        cameraRays = self.camera.getRayMat(np.flip(self.point2DSamples, axis=1)) # The error is here!
        nextGridIDs = self.gridSamples + movementVector.astype(np.int)
        laserRays = self.laser.rays()[self.laser.getNfromXY(nextGridIDs[:, 0], nextGridIDs[:, 1])]
        origin = np.expand_dims(self.laser.origin(), 0)

        aPoints, bPoints, distances = helper.MatLineLineIntersection(cameraRays*0.0, cameraRays*100.0, origin + 0.0*laserRays, origin + 100.0*laserRays)
        reprojections = helper.project3DPointToImagePlaneMat(aPoints + ((bPoints - aPoints) / 2.0), self.camera.intrinsic())
        error += np.sum(np.linalg.norm(reprojections - np.flip(self.point2DSamples, axis=1), axis=1))

        return error*error

    def recurse(self, directionVector, previousError):
        if len(self.visited) > 0 and np.any(np.all(np.expand_dims(directionVector, 0) == np.array(self.visited), axis=1)):
            return
        self.visited.append(directionVector)

        if (self.gridEstimates + directionVector < 0).any() or (self.gridEstimates + directionVector >= self.laser.gridHeight()).any():
            return

        error = self.calc_error(directionVector)

        if previousError < error:
            return

        if error < self.minimalError:
            self.minimalError = error
            self.minimalDirection = directionVector.copy()

        self.recurse(self.upVector + directionVector, error)
        self.recurse(self.downVector + directionVector, error)
        self.recurse(self.rightVector + directionVector, error)
        self.recurse(self.leftVector + directionVector, error)

    def RANSAC(self, numSamples=5, numIterations=5):
        directionVectors = list()
        errors = list()
        for i in range(numIterations):
            self.minimalError = 100000000000000.0
            self.minimalDirection = np.array([0.0, 0.0])
            self.visited = list()

            randomIndices = np.random.choice(len(self.pixelEstimates), numSamples, replace=False)
            self.gridSamples = np.array(self.gridEstimates)[randomIndices]
            self.point2DSamples = np.array(self.pixelEstimates)[randomIndices]

            if np.random.randint(low=0, high=6) == 5:
                self.recurse(np.random.randint(low=-6, high=6, size=2), 100000000000000.0)
            else:
                self.recurse(np.array([0.0, 0.0]), 100000000000000.0)
            directionVectors.append(self.minimalDirection)
            errors.append(self.minimalError)
            #print(str(self.minimalError) + ": " + str(self.minimalDirection))
        
        return directionVectors[errors.index(min(errors))]