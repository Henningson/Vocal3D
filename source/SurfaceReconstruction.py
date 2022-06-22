import argparse
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage
import torch
from geomdl import BSpline, utilities
from geomdl.visualization import VisMPL
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from pytorch3d.loss import chamfer_distance
from scipy.spatial import Delaunay, KDTree
from torch_nurbs_eval.surf_eval import SurfEval
from tqdm import tqdm

import helper
import M5
import Timer
from Laser import Laser

import ARAP

def rotation_matrix_from_vectors(vec1, vec2):
    """ Find the rotation matrix that aligns vec1 to vec2
    :param vec1: A 3d "source" vector
    :param vec2: A 3d "destination" vector
    :return mat: A transform matrix (3x3) which when applied to vec1, aligns it with vec2.
    """
    a, b = (vec1 / np.linalg.norm(vec1)).reshape(3), (vec2 / np.linalg.norm(vec2)).reshape(3)
    v = np.cross(a, b)
    c = np.dot(a, b)
    s = np.linalg.norm(v)
    kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    rotation_matrix = np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2))
    return rotation_matrix


def alignPointData(triangulatedPoints, laser):
    # First move triangulatedPoints into origin
    centroid = np.expand_dims(np.sum(triangulatedPoints, axis=0) / triangulatedPoints.shape[0], 0)
    alignedPoints = triangulatedPoints - centroid

    #Use SVD to fit plane
    svd = np.linalg.svd(alignedPoints.T)[0][:, -1]

    #Rotate points such that plane normal is aligned to [0, 1, 0]
    rotPlane = rotation_matrix_from_vectors(svd, np.array([0.0, 1.0, 0.0]))

    return alignedPoints, centroid


def findXYZExtent(triangulatedPoints):
    return triangulatedPoints[:, 0].min(), triangulatedPoints[:, 0].max(), triangulatedPoints[:, 1].min(), triangulatedPoints[:, 1].max(), triangulatedPoints[:, 2].min(), triangulatedPoints[:, 2].max()


def splitLeftAndRight(points):
    return points[np.where(points[:, 0] < 0)], points[np.where(points[:, 0] >= 0)]


def extrudeM5(vertices, minZ, maxZ, subdivisions=5):
    numPoints = vertices.shape[0]
    extrude = np.expand_dims(np.linspace(minZ, maxZ, subdivisions).repeat(numPoints), 1)
    repeatedVertices = np.tile(vertices, (subdivisions, 1))
    return np.concatenate([repeatedVertices, extrude], axis=1)


def translateVertices(vertices, translation):
    vertices += translation
    return vertices


def rotateX(mat, degree, deg=True):
    rad = degree
    if deg:
        rad = M5.deg2rad(degree)

    rotation_matrix = np.array([[np.cos(rad), 0, np.sin(rad)],
                                [0, 1, 0],
                                [-np.sin(rad), 0, np.cos(rad)]])
    
    return np.matmul(mat, rotation_matrix)


def rotateY(mat, degree, deg=True):
    rad = degree
    if deg:
        rad = M5.deg2rad(degree)

    rotation_matrix = np.array([[1, 0, 0],
                                [0, np.cos(rad),  -np.sin(rad)],
                                [0, np.sin(rad), np.cos(rad)]])
    
    return np.matmul(mat, rotation_matrix)

def rotateZ(mat, degree, deg=True):
    rad = degree
    if deg:
        rad = M5.deg2rad(degree)

    rotation_matrix = np.array([[np.cos(rad), -np.sin(rad), 0],
                                [np.sin(rad),  np.cos(rad), 0],
                                [          0,            0, 1]])
    
    return np.matmul(mat, rotation_matrix)


def generateARAPAnchors(vertices, points, nPointsU, glottalOutlinePoints, isLeft=True):
    # We want to first set the lower points of our Control Point Set to be fixed
    lower_indices = np.where(vertices[:, 1] == vertices[vertices[:, 1].argmin(), 1])
    lower_fixed = vertices[lower_indices]

    anchors = dict(zip(lower_indices[0].tolist(), lower_fixed.tolist()))

    upper_indices = np.where(vertices[:, 1] == vertices[vertices[:, 1].argmax(), 1])
        
    upper_controlpoints = vertices[upper_indices]


    # Fit upper Surface to points
    for i in range(upper_controlpoints.shape[0]):

        if upper_indices[0][i] % nPointsU == 0:
            anchors[upper_indices[0][i]] = upper_controlpoints[i].tolist()
            continue
        
        if upper_indices[0][i] < nPointsU:
            anchors[upper_indices[0][i]] = upper_controlpoints[i].tolist()
            continue

        if upper_indices[0][i] >= ((vertices.shape[0] // nPointsU) - 1) * nPointsU:
            anchors[upper_indices[0][i]] = upper_controlpoints[i].tolist()
            continue
        
        if points.size == 0:
            continue
        
        nearestNeighborIndex, dist = helper.findNearestNeighbour(upper_controlpoints[i][0::2], points[:, 0::2])
        
        direction = -upper_controlpoints[i] + points[nearestNeighborIndex]
        direction = direction / np.linalg.norm(direction)
        angle = np.arccos(direction.dot(np.array([0.0, 1.0, 0.0]))) * 180.0 / 3.141

        if dist > 3.0:
            continue
        
        if np.sign(upper_controlpoints[i, 0]) != np.sign(points[nearestNeighborIndex, 0]):
            continue

        anchors[upper_indices[0][i]] = points[nearestNeighborIndex].tolist()

    nPointsV = vertices.shape[0] // nPointsU

    # Fit glottal out and midline
    if glottalOutlinePoints.size != 0:
        for i in range(nPointsV):
            for j in range(0, 3):
                controlPointIndex = 4 + j + i*nPointsU
                controlPoint = vertices[controlPointIndex]

                glottalPoints = glottalOutlinePoints + np.array([[0.0, controlPoint[1], 0.0]])
                nnIndex, dist = helper.findNearestNeighbour(controlPoint, glottalPoints)

                if dist > 3.0:
                    anchors[controlPointIndex] = controlPoint.tolist()
                    continue

                direction = -glottalPoints[nnIndex] + controlPoint
                direction = direction / np.linalg.norm(direction)


                axis = np.array([-1.0, 0.0, 0.0]) if direction[0] < 0 else np.array([1.0, 0.0, 0.0])
                angle = np.arccos(direction.dot(axis)) * 180.0 / 3.141

                if abs(angle) > 45.0:
                    anchors[controlPointIndex] = controlPoint.tolist()
                    continue
                
                anchors[controlPointIndex] = glottalPoints[nnIndex]

    constrained_vertices = np.zeros(shape=(vertices.shape[0]), dtype=np.bool)
    constrained_vertices[list(anchors.keys())] = True
    return anchors, constrained_vertices


# Given
# 3D Points of type NumFrames X NumPoints x 3
# Calibrated laser object
def controlPointBasedARAP(triangulatedPoints, laser, images, camera, segmentator, closedGlottisFrameNum, zSubdivisions=10):
    left_M5_list = []
    right_M5_list = []
    left_points_list = []
    right_points_list = []

    ARAP_timer = Timer.Timer()
    ARAP_timer.start()

    minX = 0
    maxX = 0
    minY = 0
    maxY = 0
    minZ = 0
    maxZ = 0
    first = True

    frameCount = closedGlottisFrameNum

    # We'll first do this for every frame, and later optimize it
    for i in range(triangulatedPoints.shape[0]):
        points = triangulatedPoints[i]
        points = points[~np.isnan(points).any(axis=1)]
        if points.shape[0] <= 30:
            continue

        x, _, y, _ = segmentator.getROI()
        glottalOutline = segmentator.getGlottalOutline(images[frameCount])
        if glottalOutline is not None:
            glottalOutline += np.array([x, y])

        upperMidLine, lowerMidLine = segmentator.getGlottalMidline(images[frameCount])
        
        k = 1
        while(upperMidLine is None and lowerMidLine is None):
            upperMidLine, lowerMidLine = segmentator.getGlottalMidline(images[frameCount + k])
            k += 1
        
        cameraRay1 = camera.getRay(np.array([x, y]) + upperMidLine)
        cameraRay2 = camera.getRay(np.array([x, y]) + lowerMidLine)

        frameCount += 1

        #Basic Outlier-Filtering
        tree = KDTree(points)
        outlierIndices = np.where(np.sum(tree.query(points, k=4)[0][:, 1:], axis=1) / 3 < 1.5)
        points = points[outlierIndices]

        centroid = np.expand_dims(np.sum(points, axis=0) / points.shape[0], 0)
        alignedPoints = points - centroid

        planeNormal = np.linalg.svd(alignedPoints.T)[0][:, -1]

        # Z-Coordinate of Plane Normal pointing in wrong direction?
        # Then flip normal
        if planeNormal[2] < 0:
            planeNormal = -planeNormal

        rotPlane = helper.rotateAlign(planeNormal/np.linalg.norm(planeNormal), np.array([0.0, 1.0, 0.0]))

    #ax.plot([svd[0], svd[0]*5], [svd[1], svd[1]*5], [svd[2], svd[2]*5], color="black", )
        alignedPoints = np.matmul(rotPlane, alignedPoints.T).T

        rotatedPlaneNormal = np.matmul(rotPlane, planeNormal)
        xx, yy = np.meshgrid(range(-2, 2), range(-2, 2))
        # calculate corresponding z
        z = (-rotatedPlaneNormal[0] * xx - rotatedPlaneNormal[1] * yy - (-np.array([0.0, 0.0, 0.0]).dot(rotatedPlaneNormal))) * 1. /rotatedPlaneNormal[2]

        hit1, t1 = helper.rayPlaneIntersection(centroid, planeNormal, np.array([0.0, 0.0, 0.0]), cameraRay1)
        hit2, t2 = helper.rayPlaneIntersection(centroid, planeNormal, np.array([0.0, 0.0, 0.0]), cameraRay2)
        
        #pointOnPlane1 = (cameraRay1*t1).squeeze()
        #pointOnPlane2 = (cameraRay2*t2).squeeze()

        pointOnPlane1 = (cameraRay1*t1 - centroid).squeeze()
        pointOnPlane2 = (cameraRay2*t2 - centroid).squeeze()
        
        pointOnPlane1 = np.matmul(rotPlane, pointOnPlane1)
        pointOnPlane2 = np.matmul(rotPlane, pointOnPlane2)

        gmplVec = -pointOnPlane1 + pointOnPlane2
        gmplDirection = gmplVec / np.linalg.norm(gmplVec)
        gmplAngle = np.arccos(gmplDirection.dot(np.array([1.0, 0.0, 0.0])))

        glottalOutlinePoints = None
        if glottalOutline is not None:
            glottalOutlineRays = camera.getRayMat(glottalOutline)
            t = helper.rayPlaneIntersectionMat(centroid, np.expand_dims(planeNormal, 0), np.zeros(glottalOutlineRays.shape), glottalOutlineRays)
            glottalOutlinePoints = t * glottalOutlineRays
            glottalOutlinePoints = glottalOutlinePoints - centroid
            glottalOutlinePoints = np.matmul(rotPlane, glottalOutlinePoints.T).T
            glottalOutlinePoints = rotateX(glottalOutlinePoints, gmplAngle, deg=False)


        alignedPoints = rotateX(alignedPoints, gmplAngle, deg=False)
        pointOnPlane1 = rotateX(pointOnPlane1, gmplAngle, deg=False)
        pointOnPlane2 = rotateX(pointOnPlane2, gmplAngle, deg=False)

        zOffset = pointOnPlane2[2]
        alignedPoints -= np.array([[0.0, 0.0, zOffset]])
        pointOnPlane1 -= np.array([0.0, 0.0, zOffset])
        pointOnPlane2 -= np.array([0.0, 0.0, zOffset])

        if glottalOutline is not None:
            glottalOutlinePoints -= np.array([[0.0, 0.0, zOffset]])

        alignedPoints = rotateX(alignedPoints, -90)
        alignedPoints = rotateZ(alignedPoints, 180)
        pointOnPlane1 = rotateX(pointOnPlane1, -90)
        pointOnPlane2 = rotateX(pointOnPlane2, -90)
        if glottalOutline is not None:
            glottalOutlinePoints = rotateX(glottalOutlinePoints, -90)

        alignedPoints -= np.array([[0.0, alignedPoints[:, 1].min(), 0.0]])

        splitIndicesLeft = np.where(alignedPoints[:, 0] < 0)
        splitIndicesRight = np.where(alignedPoints[:, 0] >= 0)

        aligned = alignedPoints


        leftPoints = aligned[splitIndicesLeft]
        rightPoints = aligned[splitIndicesRight]

        # Find X-Y-Z Extent of Vocalfolds to generate fitting M5 Model
        if first:
            minX, maxX, minY, maxY, minZ, maxZ = findXYZExtent(aligned)
            first = False

        # Generate M5 Model for left and right vocalfold
        absMaxX = max(abs(minX), abs(maxX))
        M5_Right = np.array(M5.M52D(1.0, 2.5, 0.0, absMaxX, isLeft=False).getVertices())
        M5_Left = np.array(M5.M52D(1.0, 2.5, 0.0, absMaxX, isLeft=True).getVertices())

        num_2d = M5_Right.shape[0]


        # Extrude M5 Models here
        M5_Right = extrudeM5(M5_Right, minZ, maxZ, subdivisions=zSubdivisions)
        M5_Left = extrudeM5(M5_Left, minZ, maxZ, subdivisions=zSubdivisions)

        # Generate the Anchors for the ARAP-Deformation
        if leftPoints.size == 0:
            a = 1
            pass
        
        left_anchors = generateARAPAnchors(M5_Left, leftPoints, num_2d, glottalOutlinePoints[np.where(glottalOutlinePoints[:, 0] < 0)], isLeft=True)
        right_anchors = generateARAPAnchors(M5_Right, rightPoints, num_2d, glottalOutlinePoints[np.where(glottalOutlinePoints[:, 0] >= 0)], isLeft=False)

        # Triangulate M5 Models
        left_tris = Delaunay(M5_Left)
        right_tris = Delaunay(M5_Right)

        # Use ARAP to deform Control Points
        left_deform = ARAP.ARAP(M5_Left.T, left_tris.convex_hull, left_anchors.keys(), anchor_weight=10000.0)
        right_deform = ARAP.ARAP(M5_Right.T, right_tris.convex_hull, right_anchors.keys(), anchor_weight=10000.0)

        deformed_left = left_deform(left_anchors, num_iters=1).T
        deformed_right = right_deform(right_anchors, num_iters=1).T

        # Save deformed Control Points in List
        left_M5_list.append(deformed_left)
        right_M5_list.append(deformed_right)
        left_points_list.append(np.concatenate([leftPoints, glottalOutlinePoints[np.where(glottalOutlinePoints[:, 0] < 0)]], axis=0))
        right_points_list.append(np.concatenate([rightPoints, glottalOutlinePoints[np.where(glottalOutlinePoints[:, 0] >= 0)]], axis=0))

        if frameCount >= len(images):
            break

    ARAP_timer.stop()
    print("ARAPing {0} Frames takes: {1}s".format(triangulatedPoints.shape[0], ARAP_timer.getAverage()))

    return left_M5_list, right_M5_list, left_points_list, right_points_list


# Finds the np array with the least amount of points
def findMinimumLen(arrayList):
    min = 10000
    for array in arrayList:
        if array.shape[0] < min:
            min = array.shape[0]

    return min


# Reduce array sizes
def reduceArrays(arrayList, minimum):
    new = list()
    for array in arrayList:
        random_indices = np.random.choice(array.shape[0], size=minimum, replace=False)
        new.append(array[random_indices, :].tolist())

    return new

def surfaceOptimization(control_points, points, zSubdivisions=10, iterations=100, lr=1.0):
    numFrames = len(control_points)
    optimizedControlPoints = []
    avg_loss = 0.0

    control_points = np.array(control_points)
    control_points = control_points.reshape(control_points.shape[0], zSubdivisions, -1, 3)

    points = np.array(points)
    minimum = 1000000000000000
    for i in range(points.shape[0]):
        if minimum > points[i].shape[0]:
            minimum = points[i].shape[0]


    targets = np.array(reduceArrays(points, minimum))

    # Generate first Surface to find UV-Values for closest point on surface for each target Point
    surface = BSpline.Surface()
    surface.degree_u = 3
    surface.degree_v = 3
    surface.ctrlpts2d = control_points[0].tolist()
    surface.knotvector_u = utilities.generate_knot_vector(surface.degree_u, surface.ctrlpts_size_u)
    surface.knotvector_v = utilities.generate_knot_vector(surface.degree_v, surface.ctrlpts_size_v)

    #uvCorrespondences = list()


    timer = Timer.Timer()
    timer.start()

    # Get Framewise Conrol Points and Target Points
    frame_ctrl_pnts = control_points
    targets = targets.astype(np.float32)

    # Preliminaries for actual Optimization
    num_eval_pts_u = len(surface.knotvector_u)
    num_eval_pts_v = len(surface.knotvector_v)

    #frame_ctrl_pnts = np.expand_dims(frame_ctrl_pnts, 0)
    num_ctrl_pts1 = np.array(frame_ctrl_pnts.shape[1])
    num_ctrl_pts2 = np.array(frame_ctrl_pnts.shape[2])

    # Getting everything on the GPU and setting up the Layer for the Surface Evaluation
    target = torch.from_numpy(targets)
    layer = SurfEval(num_ctrl_pts1, num_ctrl_pts2, dimension=3, p=3, q=3, u=torch.FloatTensor(np.array(surface.knotvector_u)), v=torch.FloatTensor(np.array(surface.knotvector_v)), out_dim_u=num_eval_pts_u, out_dim_v=num_eval_pts_v)

    inp_ctrl_pts = torch.FloatTensor(frame_ctrl_pnts.astype(np.float32))
    inp_ctrl_pts = torch.nn.Parameter(inp_ctrl_pts)
    weights = torch.ones(inp_ctrl_pts.shape[0], num_ctrl_pts1, num_ctrl_pts2, 1)

    # Optimization starts here
    opt = torch.optim.Adam(iter([inp_ctrl_pts]), lr=lr)
    #pbar = tqdm(range(iterations))
    
    for j in range(iterations):
        opt.zero_grad()
        out = layer(torch.cat((inp_ctrl_pts, weights), -1))
        #out = torch.diagonal(outi.squeeze(), 0, dim1=0, dim2=1).T
        # out = layer(inp_ctrl_pts)
        #loss = ((target-out)**2).mean()
        chamfer_loss, _ = chamfer_distance(target, out.reshape(out.shape[0], -1, 3))
        loss = chamfer_loss
        loss.backward()
        opt.step()

        if loss.item() < 1e-4:
            break
        
        avg_loss += loss.item()
        #pbar.set_description("Loss %s: %s" % (j+1, loss.item()))
    #optimizedControlPoints.append(inp_ctrl_pts.detach().cpu().numpy().squeeze().reshape(-1, 3))

    timer.stop()
    print("Average Loss of {0} Frames: {1}. Took {2} seconds".format(numFrames, avg_loss/(numFrames*iterations), timer.getAverage()))
    return inp_ctrl_pts.detach().cpu().numpy().squeeze().reshape(inp_ctrl_pts.shape[0], -1, 3)
