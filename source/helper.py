import numpy as np
import cv2
import math
import os

UP = 0
RIGHT = 1
DOWN =  2
LEFT = 3
UNDEFINED = -1


# LineLineIntersection re-implementation by Paul Bourkes Original C-Code
# See http://paulbourke.net/geometry/pointlineplane/lineline.c
# Calculates the shortest path Pa -> Pb between two lines in 3D
# P1 -> P2 specifies Line A
# P3 -> P4 specifies Line B 
def LineLineIntersection(p1, p2, p3, p4, epsilon=1e-4):
    p13 = p1 - p3
    p43 = p4 - p3
    if np.sum(np.abs(p43)) < epsilon:
        return np.nan, np.nan, np.nan

    p21 = p2 - p1
    if np.sum(np.abs(p21)) < epsilon:
        return np.nan, np.nan, np.nan
    
    d1343 = np.dot(p13, p43)
    d4321 = np.dot(p43, p21)
    d1321 = np.dot(p13, p21)
    d4343 = np.dot(p43, p43)
    d2121 = np.dot(p21, p21)

    denom = d2121 * d4343 - d4321 * d4321
    if np.abs(denom) < epsilon:
        return np.nan, np.nan, np.nan
    
    numer = d1343 * d4321 - d1321 * d4343
    mua = numer / denom
    mub = (d1343 + d4321 * mua) / d4343

    pa = p1 + mua * p21
    pb = p3 + mub * p43
    return pa, pb, np.linalg.norm(pb - pa)


def MatLineLineIntersection(p1, p2, p3, p4, epsilon=1e-4):
    if len(p1.shape) != 2:
        p1 = np.expand_dims(p1, 0)
        p2 = np.expand_dims(p2, 0)
        p3 = np.expand_dims(p3, 0)
        p4 = np.expand_dims(p4, 0)

    p13 = p1 - p3
    p43 = p4 - p3
    #p43[np.where((np.abs(p43) < epsilon).all(axis=1))] = [[np.nan, np.nan, np.nan]]


    p21 = p2 - p1
    #p21[np.where((np.abs(p21) < epsilon).all(axis=1))] = [[np.nan, np.nan, np.nan]]

    d1343 = np.sum(p13 * p43, axis=1)
    d4321 = np.sum(p43 * p21, axis=1)
    d1321 = np.sum(p13 * p21, axis=1)
    d4343 = np.sum(p43 * p43, axis=1)
    d2121 = np.sum(p21 * p21, axis=1)

    denom = d2121 * d4343 - d4321 * d4321

    #denom = np.where(denom < epsilon, np.nan, denom)
    
    numer = d1343 * d4321 - d1321 * d4343
    mua = numer / denom
    mub = (d1343 + d4321 * mua) / d4343

    pa = p1 + np.expand_dims(mua, -1) * p21
    pb = p3 + np.expand_dims(mub, -1) * p43
    return pa, pb, np.linalg.norm(pb - pa, axis=1)

def getAveragePixelDistance(distances):
    return np.average(np.average(distances[:, 1:], axis=1))

def getPointOnRayFromOrigin(rayOrigin, rayDirection, distance):
    #RayOrigin and RayDirection are vec3

    p = (2 * np.dot(rayOrigin, rayDirection)) / np.dot(rayDirection, rayDirection)
    q = (np.dot(rayOrigin, rayOrigin) - distance*distance) / np.dot(rayDirection, rayDirection)

    r1 = -(p / 2.0) + np.sqrt((p/2.0) * (p/2.0) - q)
    r2 = -(p / 2.0) - np.sqrt((p/2.0) * (p/2.0) - q)

    return [r1, rayOrigin + r1*rayDirection], [r2, rayOrigin + r2*rayDirection]


def getPointOnRayFromOriginMat(rayOrigin, rayDirection, distance):
    #RayOrigin and RayDirection are vec3

    if len(rayOrigin.shape) == 1:
        rayOrigin = np.expand_dims(rayOrigin, 0)

    p = (2 * np.sum(rayOrigin * rayDirection, axis=1)) / np.sum(rayDirection * rayDirection, axis=1)
    q = (np.sum(rayOrigin * rayOrigin, axis=1) - distance*distance) / np.sum(rayDirection * rayDirection, axis=1)

    r1 = -(p / 2.0) + np.sqrt((p/2.0) * (p/2.0) - q)
    r2 = -(p / 2.0) - np.sqrt((p/2.0) * (p/2.0) - q)

    return [r1, rayOrigin + np.expand_dims(r1, -1)*rayDirection], [r2, rayOrigin + np.expand_dims(r2, -1)*rayDirection]

def projectToImagePlane(distance, laserOrigin, laserVec, cameraMatrix):
    point3d = laserOrigin + distance * (laserVec/np.linalg.norm(laserVec))
    point2d = np.matmul(cameraMatrix, point3d)
    point2d /= point2d[2]

    return point2d, np.linalg.norm(point3d)


def project3DPointToImagePlane(point, cameraMatrix):
    point2d = np.matmul(cameraMatrix, point)
    point2d /= point2d[2]
    return point2d


def project3DPointToImagePlaneMat(points, cameraMatrix):
    points2d = np.matmul(cameraMatrix, points.T).T
    points2d /= np.expand_dims(points2d[:, 2], -1)
    return points2d[:, :2]

def findLocalMaxima(image, kernelsize):
    kernel = np.ones((kernelsize, kernelsize), dtype=np.uint8)
    kernel[math.floor(kernelsize//2), math.floor(kernelsize//2)] = 0.0

    return image > cv2.morphologyEx(image, cv2.MORPH_DILATE, kernel)

def generateMask(image, cameraMatrix, laserOrigin, laserBeam, minDistance, maxDistance, radiusMax, radiusMin):

    _, distanceMin = projectToImagePlane(minDistance, laserOrigin, laserBeam, cameraMatrix)
    _, distanceMax = projectToImagePlane(maxDistance, laserOrigin, laserBeam, cameraMatrix)

    i = minDistance
    while i < maxDistance:
        point2d, distance = projectToImagePlane(i, laserOrigin, laserBeam, cameraMatrix)
        
        radius = radiusMax - ((distance - distanceMin) / (distanceMax - distanceMin)) * (radiusMax - radiusMin)

        cv2.circle(image, (math.floor(point2d[0]), math.floor(point2d[1])), radius=round(radius), color=255, thickness=-1)

        i += 0.5

    return image


def loadVideo(path, camera_matrix, distortion_coefficients):
    images = list()

    cap = cv2.VideoCapture(path)

    if not cap.isOpened():
        print("Error opening video file")
        return None

    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame = cv2.undistort(frame, camera_matrix, distortion_coefficients)
            images.append(frame)
        else:
            break
    
    return images


def loadImages(path, camera_matrix, distortion_coefficients):
    images = list()
    number_files = len(os.listdir(path))

    for i in range(1, number_files):
        images.append(cv2.undistort( cv2.imread(path + '{0:05d}'.format(i) + ".png", 0), camera_matrix, distortion_coefficients ))

    return images


def findMaxima(image, segmentation):
    maxima = findLocalMaxima(image, 7)
    maxima = (segmentation & image) * maxima
    return maxima


def rayPlaneIntersection(planeOrigin, planeNormal, rayOrigin, rayDirection):
    denom = planeNormal.dot(rayDirection)

    if abs(denom) > 0.0001:
        t = (planeOrigin - rayOrigin).dot(planeNormal) / denom

        if t >= 0:
            return True, t

    return False, np.nan


def rayPlaneIntersectionMat(planeOrigin, planeNormal, rayOrigin, rayDirection):
    denom = np.sum(planeNormal * rayDirection, axis=1)

    denom = np.where(np.abs(denom) < 0.000001, np.nan, denom)
    t = np.sum((planeOrigin - rayOrigin) * planeNormal, axis=1) / denom

    return np.expand_dims(t, -1)


def rotateAlign(v1, v2):
    axis = np.cross(v1, v2)

    cosA = np.dot(v1, v2)
    k = 1.0 / (1.0 + cosA)

    return np.array([[axis[0] * axis[0] * k + cosA, axis[1] * axis[0] * k - axis[2], axis[2] * axis[0] * k + axis[1]], [axis[0] * axis[1] * k + axis[2], axis[1] * axis[1] * k + cosA, axis[2] * axis[1] * k - axis[0]], [axis[0] * axis[2] * k - axis[1], axis[1] * axis[2] * k + axis[0], axis[2] * axis[2] * k + cosA]])


def intensityWeightedCentroids(images, correspondences, frameOfClosedGlottis):
    #correspondences[0]: grid, correspondences[1]: pixel

    count = 0
    for correspondence in correspondences:
        pixelValues = correspondence[1]
        image = images[frameOfClosedGlottis + count]

        segment = np.zeros(image.shape)
        segment[pixelValues] = 1

        kernel = np.ones((7, 7), dtype=np.uint8)
        segment = cv2.dilate(image, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7,7)))
        image = image * segment

        # for each pixelValue get ROI around pixelValue
        for pixelValue in pixelValues:
            roiStartY = pixelValue[0] - 3
            roiStartX = pixelValue[1] - 3

            region = image[roiStartY:pixelValue[0]+3, roiStartX:pixelValue[1]+3]
            np.linspace(0, 7)

        # compute weighted average

        count += 1

    #Set Pixels to White, that are defined in correspondence[1]


def midPointMethod(surface, points, iterations = 10):
    delta=0.25

    offset = np.array([[-delta, -delta], [-delta, delta], [delta, -delta], [delta, delta]])
    offset = np.tile(offset, (points.shape[0], 1))

    midpoints = np.array([[0.25, 0.25], [0.25, 0.75], [0.75, 0.25], [0.75, 0.75]])
    midpoints = np.tile(midpoints, (points.shape[0], 1))

    count = 0
    while True:
        offset /= 2.0
        p = np.array(surface.evaluate_list(midpoints.tolist()))
        
        distances = np.linalg.norm(p - points.repeat(4, axis=0), axis=1).reshape(-1, 4)
        indexes = np.argmin(distances, axis=1)

        indexes = np.arange(0, points.shape[0], 1) * 4 + indexes


        if count == iterations:
            return midpoints[indexes]

        midpoints = np.tile(midpoints[indexes], 4).reshape(-1, 2) + offset
        count += 1

    return None



def midPointProjection(surface, points, iterations = 10):
    deltaU = 0.25
    deltaV = 0.1

    offset = np.array([[-deltaU, -deltaV], [-deltaU, deltaV], [deltaU, -deltaV], [deltaU, deltaV]])
    offset = np.tile(offset, (points.shape[0], 1))

    midpoints = np.array([[0.25, 0.1], [0.25, 0.1], [0.75, 0.3], [0.75, 0.3]])
    midpoints = np.tile(midpoints, (points.shape[0], 1))

    count = 0
    while True:
        offset /= 2.0
        p = np.array(surface.evaluate_list(midpoints.tolist()))
        
        distances = np.linalg.norm(p[:, [0, 2]] - points.repeat(4, axis=0)[:, [0, 2]], axis=1).reshape(-1, 4)
        indexes = np.argmin(distances, axis=1)

        indexes = np.arange(0, points.shape[0], 1) * 4 + indexes


        if count == iterations:
            return midpoints[indexes]

        midpoints = np.tile(midpoints[indexes], 4).reshape(-1, 2) + offset
        count += 1

    return None


def calc_overlap(images, segmentation):
    count_img = np.zeros(images[0].shape, dtype=np.float32)

    for image in images:
        maxima = findMaxima(image, segmentation)
        maxima = np.where(maxima > 0, 1, 0).astype(np.uint8)
        M = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5), (2, 2))
        dilated_maxima = cv2.dilate(maxima, M)
        count_img += dilated_maxima

    count_img = (normalize(count_img)*255).astype(np.uint8)
    cv2.imshow("Overlap", count_img)
    cv2.waitKey(0)


#Point 1xN
#Points MxN
#Returns the nearest point in points
def findNearestNeighbour(point, target_points):
    if len(point.shape) == 1:
        point = np.expand_dims(point, 0)
    dist = np.linalg.norm(point.repeat(target_points.shape[0], axis=0) - target_points, axis=1)
    return np.argmin(dist), dist.min()


def normalize(arr):
    return (arr - arr.min()) / (arr.max() - arr.min())


def reorder_faces(vertices, faces):
    verts_centroid = np.sum(vertices, axis=0) / vertices.shape[0]
    new_faces = list()
    for face in faces:
        vert0 = vertices[face[0]]
        vert1 = vertices[face[1]]
        vert2 = vertices[face[2]]

        AB = -vert0 + vert1
        AC = -vert0 + vert2
        normal = np.cross(AB, AC)
        normal = normal / np.linalg.norm(normal)
        
        triangle_center = (vert0 + vert1 + vert2) / 3.0
        direction = -verts_centroid + triangle_center
        direction = direction / np.linalg.norm(direction)

        if np.dot(direction, normal) > 0.0:
            new_faces.append([face[0], face[1], face[2]])
        else:
            new_faces.append([face[2], face[1], face[0]])
    
    return new_faces


def generate_laserdot_images(triangulatedPoints, images, camera, segmentation):
    laserdots = list()
    for points, image in zip(triangulatedPoints, images):
        points = np.array(points)
        points2d = camera.project(points)
        #points2d = points2d[points2d > 0]
        projection = np.zeros(image.shape, dtype=np.uint8)
        points2d = points2d.astype(np.int32)[..., ::-1]
        in_bounds = np.bitwise_and(np.bitwise_and(points2d[:, 0] > 0, points2d[:, 1] > 0), np.bitwise_and(points2d[:, 0] < image.shape[0], points2d[:, 1] < image.shape[1]))
        points2d = points2d[in_bounds, :]
        #print(points2d.shape)
        projection[points2d[:, 0], points2d[:, 1]] = 255
        projection = cv2.dilate(projection, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))

        maxima = findMaxima(image, segmentation)
        maxima = cv2.dilate(maxima, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
        maxima = np.where(maxima == 0, 0, 255).astype(np.uint8)
        #maxima = np.concatenate( [np.expand_dims(np.zeros(image.shape, dtype=np.uint8), -1), maxima, np.expand_dims(np.zeros(image.shape, dtype=np.uint8), -1)], axis=-1, dtype=np.uint8)

        expanded_image = np.repeat(np.expand_dims(image, -1), 3, -1)

        expanded_image[maxima > 0, :] = [255, 0, 0]
        expanded_image[projection > 0, :] = [0, 255, 0]

        laserdots.append(expanded_image)

    return laserdots