import helper
import numpy as np
import visualization


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


def triangulationMatNew(camera, laser, laser_correspondences, points_2d, minInterval, maxInterval, distMin, distMax):
    
    points3D = list()
    
    for points_per_frame in points_2d:
        cameraRays = camera.getRayMat(np.flip(points_per_frame, axis=1))
        linearizedIDs = laser.getNfromXY(laser_correspondences[:, 0], laser_correspondences[:, 1])
        laserRays = laser.rays()[linearizedIDs]
        origin = np.expand_dims(laser.origin(), 0)

        aPoints, bPoints, distances = helper.MatLineLineIntersection(cameraRays*minInterval, cameraRays*maxInterval, origin + distMin*laserRays, origin + distMax*laserRays)

        worldPositions = aPoints + (bPoints - aPoints) / 2.0
        
        #Filter NaNs and badly estimated 3-D Positions
        worldPositions = worldPositions[distances < 5.0]

        points3D.append((aPoints + ((bPoints - aPoints) / 2.0)))

    return points3D

def triangulationMat(camera, laser, framewiseCorrespondences, minInterval, maxInterval, distMin, distMax):
    
    points3D = list()

    gridIDs = np.array(framewiseCorrespondences)[:, 0].astype(np.int32)
    laser2DPositions = np.array(framewiseCorrespondences)[:, 1:]


    for k in range(laser2DPositions.shape[1]):
        positions2D = laser2DPositions[:, k, :]

        cameraRays = camera.getRayMat(np.flip(positions2D, axis=1))
        linearizedIDs = laser.getNfromXY(gridIDs[:, 0], gridIDs[:, 1])
        laserRays = laser.rays()[linearizedIDs]
        origin = np.expand_dims(laser.origin(), 0)

        aPoints, bPoints, distances = helper.MatLineLineIntersection(cameraRays*minInterval, cameraRays*maxInterval, origin + distMin*laserRays, origin + distMax*laserRays)

        worldPositions = aPoints + (bPoints - aPoints) / 2.0
        
        #Filter NaNs and badly estimated 3-D Positions
        worldPositions = worldPositions[distances < 5.0]

        points3D.append((aPoints + ((bPoints - aPoints) / 2.0)).tolist())

    return points3D

