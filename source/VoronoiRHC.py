import numpy as np
import Objects
import Intersections
import cv2
import helper
import matplotlib.pyplot as plt

from Graph import VisitableGraph
from scipy.spatial import Delaunay, delaunay_plot_2d

from sklearn.neighbors import NearestNeighbors
import visualization




def depthFirstSearchTest(laser, graph):
    image = np.zeros((laser.gridHeight(), laser.gridWidth(), 3), dtype=np.uint8)
    testRecurse(laser, graph, 0, image)


def testRecurse(laser, graph, id, image):
    image[laser.getXYfromN(id)] = [0, 255, 0]
    print(id)
    graph.visit(id)
    for node in graph.edges(id):
        if not graph.wasVisited(node):
            cv2.imshow("Visited Rays", cv2.resize(image, (500, 500), interpolation = cv2.INTER_NEAREST))
            cv2.waitKey(0)
            testRecurse(laser, graph, node, image)



class CorrespondenceFinder:
    def __init__(self, camera, laser, minWorkingDistance = 40, maxWorkingDistance = 100, threshold = 5.0, debug = None):
        self.camera = camera
        self.laser = laser
        self.minDistancePlane = Objects.Plane(np.array([[0.0, 0.0, -1.0]]), np.array([[0.0, 0.0, minWorkingDistance]]))
        self.maxDistancePlane = Objects.Plane(np.array([[0.0, 0.0, -1.0]]), np.array([[0.0, 0.0, maxWorkingDistance]]))
        self.threshold = threshold
        self.correspondences = list()
    
        self.debugImg = cv2.cvtColor(debug, cv2.COLOR_GRAY2BGR)
        self.debugTemp = self.debugImg.copy()


        self.graph = self.generateLaserGraph()
        #depthFirstSearchTest(self.laser, self.graph)

    def sanityCheck(self, a, b):
        if  a - 1 == b or a + 1 == b or a + self.laser.gridWidth() == b or a + self.laser.gridWidth() + 1 or a - self.laser.gridWidth() - 1 == b or a - self.laser.gridWidth() == b or a - self.laser.gridWidth() + 1 or a + self.laser.gridWidth() - 1 == b:
            return True

        return False

    def generateLaserGraph(self):
        laserPlane = Objects.Plane(self.laser.direction(), np.array([[0.0, 0.0, 40.0]]))
        # Intersect laser rays with plane
        points = self.laser.origin() + Intersections.rayPlane(Objects.Ray(self.laser.origin(), self.laser.rays()), laserPlane) * self.laser.rays()
        points = self.camera.project(points)
        #rotationMat = helper.rotateAlign(self.laser.direction().squeeze(), np.array([0.0, 0.0, 1.0]))
        
        # Orientate points, such that they lie axis aligned somewhere
        #points = np.matmul(points, rotationMat)[:, :2]
        
        # Build delaunay triangulation
        delaunay = Delaunay(points)

        graph = VisitableGraph()

        for i in range(self.laser.rays().shape[0]):
            graph.add_vertex(i)

        for simplex in delaunay.simplices:
            
            if self.sanityCheck(simplex[0], simplex[1]):
                graph.add_edge((simplex[0], simplex[1]))
            
            if self.sanityCheck(simplex[1], simplex[2]):
                graph.add_edge((simplex[1], simplex[2]))
            
            if self.sanityCheck(simplex[0], simplex[2]):
                graph.add_edge((simplex[0], simplex[2]))

        #if self.debugImg is not None:
        #    ax = delaunay_plot_2d(delaunay)
        #    plt.show()

        return graph

    def establishCorrespondences(self, localMaxima):
        # Take random local maximum fron local maxima
        # Local Maxima is of dim Nx2



        idx = np.random.randint(localMaxima.shape[0], size=1)
        pointToCheck = localMaxima[idx, :]
        
        # Build 2d lines, and find closest ones
        minPoints = self.laser.origin() + self.minDistancePlane.rayIntersection(Objects.Ray(self.laser.origin(), self.laser.rays())) * self.laser.rays()
        maxPoints = self.laser.origin() + self.maxDistancePlane.rayIntersection(Objects.Ray(self.laser.origin(), self.laser.rays())) * self.laser.rays()

        projectedMinPoints = helper.project3DPointToImagePlaneMat(minPoints, self.camera.intrinsic())
        projectedMaxPoints = helper.project3DPointToImagePlaneMat(maxPoints, self.camera.intrinsic())

        #projectionImage = visualization.generateProjectionImage(self.camera, self.laser, 60.0, 512, 256)
        #cv2.imshow("Projection Nr 178317813", projectionImage)
        #cv2.waitKey(0)

        laserbeamCandidates = list()

        #test_im2000 = np.zeros((512, 256, 3), dtype=np.uint8)
        #for i in range(localMaxima.shape[0]):
        #    cv2.circle(test_im2000, localMaxima[i].astype(np.int), radius=1, color=(255, 255, 255), thickness=-1)

        #for i in range(projectedMinPoints.shape[0]):
        #    cv2.circle(test_im2000, projectedMinPoints[i].astype(np.int), radius=1, color=(0, 0, 255), thickness=-1)

        #cv2.circle(test_im2000, pointToCheck[0].astype(np.int), radius=1, color=(0, 255, 0), thickness=-1)
        #for i in range(projectedMaxPoints.shape[0]):
        #cv2.imshow("TEST IM2000", test_im2000)
        #cv2.waitKey(0)

        laserbeamCandidates = [i for i in range(projectedMinPoints.shape[0]) if Intersections.pointLineSegmentDistance(projectedMinPoints[i], projectedMaxPoints[i], pointToCheck) < self.threshold]

        correspondenceLists = list()
        # For every beam in C
        for candidate in laserbeamCandidates:
            # calculate Z distance and generate plane at this distance
            self.visitImage = np.zeros((self.laser.gridHeight(), self.laser.gridWidth(), 3), dtype=np.uint8)
            self.local_maxima = localMaxima.copy()
            self.graph.reset()
            self.debugTemp = self.debugImg.copy()

            pa, pb, dist = Intersections.lineLine(Objects.Line(np.array([0.0, 0.0, 0.0]), self.camera.getRay(pointToCheck[0])), Objects.Line(self.laser.origin(), self.laser.origin() + self.laser.ray(candidate)))
            depth = pa + (-pa + pb)/2.0
            #print("Searching at depth: {0}".format(depth[0][2]))
            start_plane = Objects.Plane(np.array([[0.0, 0.0, -1.0]]), depth)


            correspondences = list()
            self.recurse(candidate, start_plane, correspondences)
            correspondenceLists.append(correspondences)
    
        # Sort correspondences by number of correspondences found
        correspondenceLists = sorted(correspondenceLists, key=len, reverse=True)

        # Remove all, but the top 25% of correspondence lists
        correspondenceLists = correspondenceLists[:int(np.ceil(len(correspondenceLists)*0.25))]

        error_list = list()
        for correspondences in correspondenceLists:
            laser_ids = np.array(correspondences)[:, 0].astype(np.int)
            pixel_data = np.stack(np.array(correspondences)[:, 1])
            pa, pb, dist = helper.MatLineLineIntersection(np.array([[0.0, 0.0, 0.0]]), self.camera.getRayMat(pixel_data), self.laser.origin(), self.laser.origin() + self.laser.rays()[laser_ids])
            data3d = pa + (-pa + pb)/2.0
            data2d = helper.project3DPointToImagePlaneMat(data3d, self.camera.intrinsic())
            reprojection_error = np.sum(np.linalg.norm(pixel_data - data2d, axis=1)) / data2d.shape[0]
            error_list.append(reprojection_error)

        return correspondenceLists[np.array(error_list).argmin()]

     
    def recurse(self, laser_candidate, plane, correspondences):
        if self.local_maxima.size == 0:
            return

        self.visitImage[self.laser.getXYfromN(laser_candidate)] = [255, 0, 0]
        
        ray = self.laser.ray(laser_candidate)
        worldPoint = self.laser.origin() + Intersections.rayPlane(Objects.Ray(self.laser.origin(), ray), plane) * ray

        point2d = helper.project3DPointToImagePlaneMat(worldPoint, self.camera.intrinsic())

        neighbors = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(self.local_maxima)
        distances, indices = neighbors.kneighbors(point2d)

        projectionImage = visualization.generateProjectionImage(self.camera, self.laser, plane, 512, 256)        

        #cv2.circle(self.debugTemp, point2d[0].astype(np.int), 2, (255, 255, 255), -1)
        #cv2.imshow("Projection Image", projectionImage)
        #cv2.imshow("Visited Rays", cv2.resize(self.visitImage, (500, 500), interpolation = cv2.INTER_NEAREST))
        #cv2.imshow("Image", self.debugTemp)
        #cv2.waitKey(1)

        line1 = Objects.Line(self.laser.origin(), self.laser.origin() + self.laser.ray(laser_candidate))
        line2 = Objects.Line(np.array([0.0, 0.0, 0.0]), self.camera.getRay(self.local_maxima[indices[0][0]]))
        pa, pb, dist = Intersections.lineLine(line1, line2)
        depth = pa + (-pb + pa)/2.0
        
        plane_direction = (-pa + pb) / np.linalg.norm(-pa + pb, axis=1)
        normal = np.cross(np.cross(plane_direction, plane.normal()), plane_direction)
        new_plane = Objects.Plane(normal, depth)
        #print(normal)

        #print("Dist: {0}".format(dist))

        if distances < self.threshold:
            correspondences.append((laser_candidate, self.local_maxima[indices[0][0]]))
            #print(laser_candidate)
            self.graph.visit(laser_candidate)
            
            cv2.circle(self.debugTemp, point2d[0].astype(np.int), 2, (0, 255, 0), -1)
            cv2.circle(self.debugTemp, self.local_maxima[indices[0, 0]].astype(np.int), 2, (255, 0, 0), -1)
            self.visitImage[self.laser.getXYfromN(laser_candidate)] = [0, 255, 0]

            #cv2.imshow("Visited Rays", cv2.resize(self.visitImage, (500, 500), interpolation = cv2.INTER_NEAREST))
            #cv2.imshow("Image", self.debugTemp)
            #cv2.waitKey(1)
            self.local_maxima = np.delete(self.local_maxima, indices, 0)
            #print(self.local_maxima.size)
        else:
            return


        for node in self.graph.edges(laser_candidate):
            #print(self.graph.edges(laser_candidate))
            if not self.graph.wasVisited(node):
                self.recurse(node, new_plane, correspondences)