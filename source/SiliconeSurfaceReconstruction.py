import torch
import numpy as np
import matplotlib.pyplot as plt
import argparse

import cv2
import M5
import helper
import Timer
import os
import math
import scipy.ndimage
import SurfaceReconstruction
import Camera

from pytorch3d.loss import chamfer_distance
from torch_nurbs_eval.surf_eval import SurfEval
from scipy.spatial import Delaunay, KDTree
from geomdl import BSpline
from geomdl import utilities
from tqdm import tqdm

import matplotlib.pyplot as plt
from geomdl.visualization import VisMPL
from matplotlib import cm

from Laser import Laser

from sklearn.decomposition import PCA
import visualization

import ARAP




from PyQt5.QtWidgets import QApplication
import Viewer
import igl
import os
import time
import threading

# Code from: https://stackoverflow.com/a/59204638
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

    #fig = plt.figure()
    #ax = fig.add_subplot(121, projection='3d')
    #ax.scatter(alignedPoints[:, 0], alignedPoints[:, 1], alignedPoints[:, 2])
    #ax.set_xlabel('X-axis')
    #ax.set_ylabel('Y-axis')
    #ax.set_zlabel('Z-axis')
    #ax2 = fig.add_subplot(122)
    #projectedPoints = np.array([alignedPoints[:, 0], alignedPoints[:, 2]]).T
    #ax2.scatter(projectedPoints[:, 0], projectedPoints[:, 1])
    #line = cv2.fitLine(projectedPoints, cv2.DIST_L2, 0, 0.01, 0.01)
    #xy = line.squeeze()[0:2]
    #vxvy = line.squeeze()[2:4]

    #p1 = xy * -10.0*vxvy
    #p2 = xy * 10.0*vxvy

    #angle = np.arctan2(p2[1] - p1[1], p2[0] - p1[0])

    svd = np.linalg.svd(alignedPoints.T)[0][:, -1]
    rotPlane = rotation_matrix_from_vectors(svd, np.array([0.0, -1.0, 0.0]))

    #ax.plot([svd[0], svd[0]*5], [svd[1], svd[1]*5], [svd[2], svd[2]*5], color="black", )
    alignedPoints = np.matmul(alignedPoints, np.linalg.inv(rotPlane))

    #ax.scatter(alignedPoints[:, 0], alignedPoints[:, 1], alignedPoints[:, 2], color="green")
    #ax.set_xlabel('X-axis')
    #ax.set_ylabel('Y-axis')
    #ax.set_zlabel('Z-axis')
    #plt.show()

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


def generateARAPAnchors(vertices, points):
    # We want to first set the lower points of our Control Point Set to be fixed
    lower_indices = np.where(vertices[:, 1] == vertices[vertices[:, 1].argmin(), 1])
    lower_fixed = vertices[lower_indices]

    anchors = dict(zip(lower_indices[0].tolist(), lower_fixed.tolist()))

    upper_indices = np.where(vertices[:, 1] == vertices[vertices[:, 1].argmax(), 1])
    upper_controlpoints = vertices[upper_indices]

    for i in range(upper_controlpoints.shape[0]):
        nearestNeighborIndex, _ = helper.findNearestNeighbour(upper_controlpoints[i], points)
        anchors[upper_indices[0][i]] = points[nearestNeighborIndex]

    return anchors

def getNeighbours(faces, numPoints):
    neighbours = [set() for i in range(numPoints)]
       #temp = list()
    for face in faces:
        neighbours[face[0]].add(face[1])
        neighbours[face[0]].add(face[2])
        neighbours[face[1]].add(face[0])
        neighbours[face[1]].add(face[2])
        neighbours[face[2]].add(face[0])
        neighbours[face[2]].add(face[1])


    for i in range(len(neighbours)):
        neighbours[i] = list(neighbours[i])
    return neighbours

# Given
# 3D Points of type NumFrames X NumPoints x 3
# Calibrated laser object
def controlPointBasedARAP(triangulatedPoints, images, camera, segmentator, zSubdivisions=5):
    left_M5_list = []
    right_M5_list = []
    left_points_list = []
    right_points_list = []
    combined_points = []
    constrained_vertices_list_left = []
    constrained_vertices_list_right = []
    
    minX = 0
    maxX = 0
    minY = 0
    maxY = 0
    minZ = 0
    maxZ = 0
    first = True
    glottalOutline = None

    M5_Right = None
    M5_Left = None
    num_2d = None
    faces_left = None
    faces_right = None
    anchors_left_list = list()
    anchors_right_list = list()

    arap_c_left_list = list()
    arap_c_right_list = list()

    ARAP_timer = Timer.Timer()
    ARAP_timer.start()
    # We'll first do this for every frame, and later optimize it
    for i in tqdm(range(len(triangulatedPoints))):
        points = triangulatedPoints[i]
        points = points[~np.isnan(points).any(axis=1)]
        if points.shape[0] <= 25:
            continue

        centroid = np.expand_dims(np.sum(points, axis=0) / points.shape[0], 0)
        alignedPoints = points - centroid

        planeNormal = np.linalg.svd(alignedPoints.T)[0][:, -1]

        if planeNormal[2] > 0.0:
            planeNormal = -planeNormal

        #Calculating plane for visualization
        xx, yy = np.meshgrid(range(-2, 2), range(-2, 2))
        # calculate corresponding z
        z = (-planeNormal[0] * xx - planeNormal[1] * yy - (-centroid[0].dot(planeNormal))) * 1. / planeNormal[2]

        # Project Glottal Outline Points into Pointcloud
        glottalOutline = segmentator.getGlottalOutline(images[i])
        glottalCameraRays = camera.getRayMat(glottalOutline)
        t = helper.rayPlaneIntersectionMat(centroid, np.expand_dims(planeNormal, 0), np.zeros(glottalCameraRays.shape), glottalCameraRays) 
        glottalOutlinePoints = t * glottalCameraRays


        # Project Glottal Midline Extrema into Pointcloud
        upperMidLine, lowerMidLine = segmentator.getGlottalMidline(images[i], isSegmented=False)
        gml_ray1 = camera.getRay(upperMidLine)
        gml_ray2 = camera.getRay(lowerMidLine)
        _, t1 = helper.rayPlaneIntersection(centroid[0], planeNormal, np.zeros((3)), gml_ray1) 
        _, t2 = helper.rayPlaneIntersection(centroid[0], planeNormal, np.zeros((3)), gml_ray2) 
        gml_point1 = gml_ray1*t1
        gml_point2 = gml_ray2*t2

        # Get everything into tehe origin
        glottalOutlinePoints = glottalOutlinePoints - centroid
        gml_point1 = np.expand_dims(gml_point1, 0) - centroid
        gml_point2 = np.expand_dims(gml_point2, 0) - centroid

        # Compute rotation matrix, aligning the plane normal to the +Y Axis
        rotPlane = helper.rotateAlign(planeNormal/np.linalg.norm(planeNormal), np.array([0.0, 1.0, 0.0]))

        # Rotate everything corresponding to that rotation matrix
        alignedPoints = np.matmul(rotPlane, alignedPoints.T).T
        glottalOutlinePoints = np.matmul(rotPlane, glottalOutlinePoints.T).T
        gml_point1 = np.matmul(rotPlane, gml_point1.T).T
        gml_point2 = np.matmul(rotPlane, gml_point2.T).T



        # Get the angle between the glottal midline and the Z Axis
        gmplVec = -gml_point1 + gml_point2
        gmplDirection = gmplVec / np.linalg.norm(gmplVec)
        gmplAngle = np.arccos(gmplDirection.dot(np.array([1.0, 0.0, 0.0])))

        # Rotate the Object, such that it is aligned to the Z Axis
        alignedPoints = rotateX(alignedPoints, -gmplAngle, deg=False)
        #visualization.plotPoints3D(alignedPoints)
        gml_point1 = rotateX(gml_point1, -gmplAngle, deg=False)
        gml_point2 = rotateX(gml_point2, -gmplAngle, deg=False)
        glottalOutlinePoints = rotateX(glottalOutlinePoints, -gmplAngle, deg=False)


        # Move everything, such that the glottal midlie lies directly ontop the Z Axus
        zOffset = gml_point1[0, 2]
        alignedPoints -= np.array([[0.0, 0.0, zOffset]])
        gml_point1 -= np.array([[0.0, 0.0, zOffset]])
        gml_point2 -= np.array([[0.0, 0.0, zOffset]])
        glottalOutlinePoints -= np.array([0.0, 0.0, zOffset])

        # Rotate everything around by 90 degrees again
        alignedPoints = rotateX(alignedPoints, -90).astype(np.float)
        gml_point1 = rotateX(gml_point1, -90).astype(np.float)
        gml_point2 = rotateX(gml_point2, -90).astype(np.float)
        glottalOutlinePoints = rotateX(glottalOutlinePoints, -90).astype(np.float)


        # Set Y Values to zero of the glottal outline points
        glottalOutlinePoints[:, 1] = 0.0

        # Move vocal folds down a bit
        alignedPoints -= np.array([[0.0, alignedPoints[:, 1].min()/2.0, 0.0]])

        # Split everything into left and right vocal fold
        splitIndicesLeft = np.where(alignedPoints[:, 0] < 0)
        splitIndicesRight = np.where(alignedPoints[:, 0] >= 0)

        # Save it
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

            num_2d = M5_Left.shape[0]

            # Extrude M5 Models here
            M5_Right = extrudeM5(M5_Right, minZ, maxZ, subdivisions=zSubdivisions)
            M5_Left = extrudeM5(M5_Left, minZ, maxZ, subdivisions=zSubdivisions)

            # Triangulate M5 Models
            faces_left = Delaunay(M5_Left).convex_hull
            faces_right = Delaunay(M5_Right).convex_hull

            faces_left = helper.reorder_faces(M5_Left, faces_left)
            faces_right = helper.reorder_faces(M5_Right, faces_right)
        
            neighbours_left = getNeighbours(faces_left, M5_Left.shape[0])
            neighbours_right = getNeighbours(faces_right, M5_Right.shape[0])

        # Generate the Anchors for the ARAP-Deformation
        #left_anchors = generateARAPAnchors(M5_Left, leftPoints)
        #right_anchors = generateARAPAnchors(M5_Right, rightPoints)
        
        left_anchors, constrained_vertices_left = SurfaceReconstruction.generateARAPAnchors(M5_Left, leftPoints, num_2d, glottalOutlinePoints[np.where(glottalOutlinePoints[:, 0] < 0)], isLeft=True)
        right_anchors, constrained_vertices_right = SurfaceReconstruction.generateARAPAnchors(M5_Right, rightPoints, num_2d, glottalOutlinePoints[np.where(glottalOutlinePoints[:, 0] >= 0)], isLeft=False)

        constrained_vertices_list_left.append(constrained_vertices_left.tolist())
        constrained_vertices_list_right.append(constrained_vertices_right.tolist())

        arap_c_left_list.append(list(left_anchors.items()))
        arap_c_right_list.append(list(right_anchors.items()))

        # Use ARAP to deform Control Points, using best practices here.
        #try:
        #    left_deform = ARAP.ARAP(M5_Left.T, np.array(faces_left), left_anchors.keys(), anchor_weight=10000.0)
        #    right_deform = ARAP.ARAP(M5_Right.T, np.array(faces_right), right_anchors.keys(), anchor_weight=10000.0)

        #    deformed_left = left_deform(left_anchors, num_iters=2).T
        #    deformed_right = right_deform(right_anchors, num_iters=2).T
        #except:
        #    continue
        
        #print("")
        #BSplineVisualization.visualizeSingleFrame(deformed_left, deformed_right, zSubdivisions, leftPoints, rightPoints)

        #deformed_left = FastARAP.deform_async([M5_Left], [faces_left], [list(left_anchors.items())], 10000.0, 1)
        #deformed_right = FastARAP.deform_async([M5_Right], [faces_right], [list(right_anchors.items())], 10000.0, 1)
        #BSplineVisualization.visualizeSingleFrame(np.array(deformed_left), np.array(deformed_right), zSubdivisions, leftPoints, rightPoints)
        

        #fig = plt.figure()
        #ax = fig.add_subplot(1, 1, 1)
        #ax.scatter(leftPoints[:, 0], leftPoints[:, 2], color="blue")
        #ax.scatter(rightPoints[:, 0], rightPoints[:, 2], color="black")
        #ax.scatter(glottalOutlinePoints[np.where(glottalOutlinePoints[:, 0] < 0)][:, 0], glottalOutlinePoints[np.where(glottalOutlinePoints[:, 0] < 0)][:, 2], color="green")
        #ax.scatter(glottalOutlinePoints[np.where(glottalOutlinePoints[:, 0] >= 0)][:, 0], glottalOutlinePoints[np.where(glottalOutlinePoints[:, 0] >= 0)][:, 2], color="red")
        #ax.plot([pointOnPlane1[0], pointOnPlane2[0]], [pointOnPlane1[2], pointOnPlane2[2]], color="gray")
        #plt.show()


        # Save deformed Control Points in List
        #left_M5_list.append(deformed_left)
        #right_M5_list.append(deformed_right)
        #left_points_list.append(np.concatenate([leftPoints, glottalOutlinePoints[np.where(glottalOutlinePoints[:, 0] < 0)]], axis=0))
        #right_points_list.append(np.concatenate([rightPoints, glottalOutlinePoints[np.where(glottalOutlinePoints[:, 0] >= 0)]], axis=0))
        left_points_list.append(leftPoints)
        right_points_list.append(rightPoints)
        combined_points.append(np.concatenate([leftPoints, rightPoints]))

    copied_M5_left = np.expand_dims(np.array(M5_Left), 0).repeat(len(arap_c_left_list), axis=0)
    copied_M5_right = np.expand_dims(np.array(M5_Right), 0).repeat(len(arap_c_left_list), axis=0)
    copied_faces_left = np.expand_dims(np.array(faces_left), 0).repeat(len(arap_c_left_list), axis=0)
    copied_faces_right = np.expand_dims(np.array(faces_right), 0).repeat(len(arap_c_left_list), axis=0)
    copied_neighbours_left = [neighbours_left for i in range(len(arap_c_left_list))]
    copied_neighbours_right = [neighbours_right for i in range(len(arap_c_left_list))]

    left_M5_list = ARAP.deform_multiple(copied_M5_left.tolist(), copied_faces_left.tolist(), arap_c_left_list, constrained_vertices_list_left, copied_neighbours_left, 2, 10000.0)
    right_M5_list = ARAP.deform_multiple(copied_M5_right.tolist(), copied_faces_right.tolist(), arap_c_right_list, constrained_vertices_list_right, copied_neighbours_right, 2, 10000.0)

    ARAP_timer.stop()
    print("ARAPing {0} Frames takes: {1}s".format(len(triangulatedPoints), ARAP_timer.getAverage()))

    return left_M5_list, right_M5_list, left_points_list, right_points_list, combined_points


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

def surfaceOptimization(control_points, points, zSubdivisions=10, iterations=10, lr=0.1):
    numFrames = len(control_points)
    optimizedControlPoints = []
    avg_loss = 0.0

    control_points = np.array(control_points)
    control_points = control_points.reshape(control_points.shape[0], zSubdivisions, -1, 3)

    points = np.array(points)
    minimum = 1000000000000000
    for i in range(points.shape[0]):
        if points[i].shape[0] == 0:
            points[i] = points[i-1]

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
    layer = SurfEval(num_ctrl_pts1, num_ctrl_pts2, dimension=3, p=3, q=3, out_dim_u=num_eval_pts_u, out_dim_v=num_eval_pts_v)

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
        #chamfer_loss, _ = chamfer_distance(target, out.reshape(out.shape[0], -1, 3))
        chamfer_loss, _ = chamfer_distance(target[:, :, 1].unsqueeze(-1), out.reshape(out.shape[0], -1, 3)[:, :, 1].unsqueeze(-1))
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


def getCentroid(points):
    return np.expand_dims((np.sum(points, axis=0) / points.shape[0]), 0)


def getPrincipalComponentAxes(points, normalized=True):
    pca = PCA(n_components=2)
    pca.fit(points)
    pc = pca.transform(points)

    if normalized:
        return pc[0] / np.linalg.norm(pc[0]), pc[1] / np.linalg.norm(pc[1])

    return pc[0], pc[1]





def cmap(t):
    c0 = np.array([0.0002189403691192265, 0.001651004631001012, -0.01948089843709184])
    c1 = np.array([0.1065134194856116, 0.5639564367884091, 3.932712388889277])
    c2 = np.array([11.60249308247187, -3.972853965665698, -15.9423941062914])
    c3 = np.array([-41.70399613139459, 17.43639888205313, 44.35414519872813])
    c4 = np.array([77.162935699427, -33.40235894210092, -81.80730925738993])
    c5 = np.array([-71.31942824499214, 32.62606426397723, 73.20951985803202])
    c6 = np.array([25.13112622477341, -12.24266895238567, -23.07032500287172])

    return c0+t*(c1+t*(c2+t*(c3+t*(c4+t*(c5+t*c6)))))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Surface Reconstruction of Vocal Folds using sparse point samples.")
    parser.add_argument('--calibration_file', '-c', type=str, required=True, help="Path to a calibration .MAT or .JSON File")
    parser.add_argument('--input', '-i', type=str, required=True, help="Path to a file containing the 3d points")
    parser.add_argument('--zcntrl' '-z', type=int, default=8, help="Specify the number of control points in Z Direction")
    parser.add_argument('--learning_rate' '-lr', type=float, default=0.1, help="Specify the learning rate for ADAM-Optimizer")
    parser.add_argument('--iterations' '-it', type=int, default=100, help="Define the number of Iterations for Least Squares Optimization")

    args = parser.parse_args()

    point_path = args.input
    calib_path = args.calibration_file
    zSubdivs = args.zcntrl_z
    learning_rate = args.learning_rate_lr
    iterations = args.iterations_it

    print("3D Point Path: {0}".format(point_path))
    print("Calibration File {0}".format(calib_path))
    print("Number of Subdivisions: {0}".format(zSubdivs))
    print("Learning Rate: {0}".format(learning_rate))
    print("Number of Iterations: {0}".format(iterations))

    camera = Camera.Camera(calib_path)
    laser = Laser(calib_path, "MAT")

    path_start = "/media/nu94waro/Seagate Expansion Drive/Promotion/Data/Rekonstruktion_Silikon/"
    names = [("50_Kay", "Kay_50_-15_M2"), 
            ("50_Kay", "Kay_50_-10_M2"),
            ("50_Kay", "Kay_50_-5_M2"),
            ("50_Kay", "Kay_50_0_M2"),
            ("50_Kay", "Kay_50_5_M2"),
            ("50_Kay", "Kay_50_10_M2"),
            ("50_Kay", "Kay_50_15_M2"),
            ("65_Kay", "Kay_65_M2_-15"),
            ("65_Kay", "Kay_65_M2_-10"),
            ("65_Kay", "Kay_65_M2_-5"),
            ("65_Kay", "Kay_65_M2_0"),
            ("65_Kay", "Kay_65_M2_5"),
            ("65_Kay", "Kay_65_M2_10"),
            ("65_Kay", "Kay_65_M2_15"),
            ("80_Kay", "80_Kay_-15_M2_gecklickt"),
            ("80_Kay", "80_Kay_-10_M2_gecklickt"),
            ("80_Kay", "80_kay_-05_M2_gecklickt"),
            ("80_Kay", "80_Kay_0_M2_gecklickt"),
            ("80_Kay", "80_kay_05_M2_gecklickt"),
            ("80_Kay", "80_Kay_10_M2_gecklickt"),
            ("80_Kay", "80_Kay_15_M2_gecklickt")]
    path_middle = "png/"


    framesOfClosedGlottis = [38, 3, 29, 27, 9, 13, 18,
                             21, 28, 26, 4, 13, 32, 30,
                             44, 10, 20, 29, 19, 3, 19]

    paths = list()
    mat_paths = list()
    for folder, name in names:

        #if folder != "65_Kay":
        #    continue

        image_path = path_start + folder + "/" + name + "/" + path_middle
        mat_path = path_start + folder + "/" + name + "/results/reconstruction/" + name + "_rec.npy"
        paths.append(image_path)
        mat_paths.append(mat_path)


    for mat_path in mat_paths:
        test = np.load(mat_path)
        a = 1

    #image_list = list()
    #print("Loading images")
    #for path in paths:
    #    images = helper.loadImages(path, camera.intrinsic(), camera.distortionCoefficients())
    #    image_list.append(images)


    #visualization.plotPoints3D()

    #midlines = list()
    #for focg, images, name  in zip(framesOfClosedGlottis, image_list, names):
    #    segmentator = SegmentationClicker.SegmentationClicker(images[focg+10])
    #    segmentator.clickMidline()
    #    midlines.append(segmentator.getMidline())


    
    rois = np.load("assets/rois.npy").tolist()
    midlines = np.load("assets/midlines.npy").tolist()
        #segmentator.clickMidline()
    
    for focg, mat_path, path, name, roi, midline  in zip(framesOfClosedGlottis, mat_paths, paths, names, rois, midlines):
        
        if name[0] != "65_Kay":
            continue

        print("Reconstructing: {0}".format(name[1]))

        images = helper.loadImages(path, camera.intrinsic(), camera.distortionCoefficients())

        #width = images[0].shape[1]
        #height = images[0].shape[0]


        #frameOfClosedGlottis = focg

        #x, w, y, h = roi


        # Use ROI to generate mask image
        #segmentation = np.zeros((height, width), dtype=np.uint8)
        #segmentation[y:y+h, x:x+w] = 255

        #vocalfold_image = images[frameOfClosedGlottis]

        #maxima = helper.findMaxima(vocalfold_image, segmentation)

        #cv2.imshow("Maxima", maxima)
        #cv2.waitKey(0)

        #for image in images:
        #    lol = np.where(image == 0, 255, 0)
        #    cv2.imshow("LOL", lol.astype(np.uint8))
        #    cv2.waitKey(0)


        # Changing from N x (Y,X) to N x (X,Y)
        #vectorized_maxima = np.flip(np.stack(maxima.nonzero(), axis=1), axis=1)

        #cf = VoronoiRHC.CorrespondenceFinder(camera, laser, minWorkingDistance=40.0, maxWorkingDistance=100.0, threshold=3.0, debug=maxima)
        #correspondences = []
        #while len(correspondences) == 0:
        #    correspondences = cf.establishCorrespondences(vectorized_maxima)

        #grid2DPixLocations = [[laser.getXYfromN(id), np.flip(pix)] for id, pix in correspondences]

        #temporalCorrespondence = Correspondences.generateFramewise(images, frameOfClosedGlottis, grid2DPixLocations, segmentation)
        #triangulatedPoints = np.array(Triangulation.triangulationMat(camera, laser, temporalCorrespondence, 40.0, 80.0, 40.0, 90.0))[:50, :, :]

        triangulatedPoints = np.load(mat_path)
        triangulatedPoints = triangulatedPoints.reshape(triangulatedPoints.shape[2], triangulatedPoints.shape[1], triangulatedPoints.shape[0]).T
        triangulatedPoints = triangulatedPoints[30:60, :, :]
        images = images[30:60]

        triangulatedPoints = triangulatedPoints.tolist()
        newpoints = list()

        segmentations = list()

        laserdots = list()

        for points, image in zip(triangulatedPoints, images):
            points = np.array(points)
            points2d = camera.project(points)
            segmentation = np.where(image == 0, 255, 0).astype(np.uint8)
            points = points[~np.isnan(points2d).any(axis=1)]
            points2d = points2d[~np.isnan(points2d).any(axis=1)]
            seghits = segmentation[points2d.astype(np.int)[:, 1], points2d.astype(np.int)[:, 0]] > 0
            points = points[~seghits]
            points2d = points2d[~seghits]
            newpoints.append(points)
            segmentations.append(segmentation)

            roi_image = np.zeros((256, 512), dtype=np.uint8)
            x, w, y, h = roi
            roi_image[y:y+h, x:x+w] = 255
            laserpoints = image * roi_image
            maxima = helper.findMaxima(image, roi_image)
            maxima = cv2.dilate(maxima, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
            maxima = np.where(maxima == 0, 0, 255).astype(np.uint8)
            maxima = np.concatenate( [np.expand_dims(np.zeros(maxima.shape, dtype=np.uint8), -1), np.expand_dims(maxima, -1), np.expand_dims(np.zeros(maxima.shape, dtype=np.uint8), -1)], axis=-1, dtype=np.uint8)
            laserdots.append(np.expand_dims(image, -1) | maxima)


        # If everything is working as intended
        # left and rightDeformed are of size N x P x 3,
        # where N is the number of Frames, P is the number of Vertices of the M5 Models
        # and 3 is the dimension of the data
        print("ARAPing")
        leftDeformed, rightDeformed, leftPoints, rightPoints = controlPointBasedARAP(newpoints, images, zSubdivisions=zSubdivs, glottalmidline=midline)
        #BSplineVisualization.visualizeBM5(np.array(leftDeformed), np.array(rightDeformed), zSubdivs, leftPoints, rightPoints, filename=name[1], plot=True)

        optimizedLeft = surfaceOptimization(leftDeformed, leftPoints, zSubdivisions=zSubdivs, iterations=10, lr=0.1)
        optimizedLeft = np.array(optimizedLeft)
        smoothedLeft = scipy.ndimage.uniform_filter(optimizedLeft, size=(5, 1, 1), mode='reflect', cval=0.0, origin=0)


        optimizedRight = surfaceOptimization(rightDeformed, rightPoints, zSubdivisions=zSubdivs, iterations=10, lr=0.1)
        optimizedRight = np.array(optimizedRight)
        smoothedRight = scipy.ndimage.uniform_filter(optimizedRight, size=(5, 1, 1), mode='reflect', cval=0.0, origin=0)


        viewer_app = QApplication(["Vocal3D - Vocal Fold 3D Reconstruction"])
        viewer = Viewer.Viewer(smoothedLeft.shape[0], smoothedLeft.max(axis=1)[:, 1], smoothedRight.max(axis=1)[:, 1], smoothedLeft, smoothedRight, images, segmentations, laserdots, zSubdivs)
        viewer.show()

        # Launch the Qt application
        viewer_app.exec()