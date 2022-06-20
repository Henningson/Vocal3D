import numpy as np
import cv2
import math

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure

from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d

from sklearn.neighbors import NearestNeighbors
import main
import Objects
import helper
from collections import OrderedDict

def visualizeWorld(rays, camera, minDistPlane, maxDistPlane, imageWidth, imageHeight):
    fig = plt.figure(figsize=(10, 10))
    ax = fig.gca(projection='3d')
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    drawRays(rays, ax)
    drawCameraFrustum(camera, imageWidth, imageHeight, ax)
    
    if minDistPlane:
        drawPlane(minDistPlane, ax, "Near Plane")

    if maxDistPlane:
        drawPlane(maxDistPlane, ax, "Far Plane")

    ax.view_init(elev=18, azim=-157)
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = OrderedDict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), loc="upper right")
    return fig

def drawRays(ray, ax):
    for i in range(ray._direction.shape[0]):
        ax.plot([ray._origin[0], ray._origin[0] + ray._direction[i][0]],
                [ray._origin[2], ray._origin[2] + ray._direction[i][2]],
                [ray._origin[1], ray._origin[1] + ray._direction[i][1]],
                color="b", label="Laserbeams")

def drawCameraRay(ax, ray):
    ax.plot([0.0, ray[0]],
            [0.0, ray[2]],
            [0.0, ray[1]], color="y", label="Frustum")

def drawFrustumSide(ax, ray1, ray2):
    ax.plot([ray1[0], ray2[0]],
            [ray1[2], ray2[2]],
            [ray1[1], ray2[1]], color="y")

def drawPlane(plane, ax, label, start=-50.0, end=50.0, subdivs=10):
    xx, yy = np.meshgrid(np.linspace(start, end, subdivs), np.linspace(start, end, subdivs))
    d = -plane.origin()[0].dot(plane.normal()[0])
    zz  = (-plane.normal()[0][0] * xx - plane.normal()[0][1] * yy - d) * 1. / plane.normal()[0][2]
    zz[zz == -np.inf] = 0.0
    zz[zz == np.inf] = 0.0
    
    surf = ax.plot_surface(xx, yy, zz, alpha=0.3, label=label, color="yellow")
    surf._facecolors2d = surf._facecolor3d
    surf._edgecolors2d = surf._edgecolor3d


def drawCameraFrustum(camera, imageHeight, imageWidth, ax):
    cameraRay1 = camera.getRay([0.0, 0.0])*80.0
    cameraRay2 = camera.getRay([imageWidth, 0.0])*80.0
    cameraRay3 = camera.getRay([0.0, imageHeight])*80.0
    cameraRay4 = camera.getRay([imageWidth, imageHeight])*80.0

    drawCameraRay(ax, cameraRay1)
    drawCameraRay(ax, cameraRay2)
    drawCameraRay(ax, cameraRay3)
    drawCameraRay(ax, cameraRay4)

    drawFrustumSide(ax, cameraRay1, cameraRay2)
    drawFrustumSide(ax, cameraRay2, cameraRay4)
    drawFrustumSide(ax, cameraRay4, cameraRay3)
    drawFrustumSide(ax, cameraRay3, cameraRay1)


def vis_camera(camera, laser):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    laserRaysX = list()
    laserRaysY = list()
    laserRaysZ = list()

    cameraRaysX = list()
    cameraRaysY = list()
    cameraRaysZ = list()

    for count, laserRay in enumerate(laser.rays().tolist()):
        x, y = laser.ray(count)

        if x % 3 != 0:
            continue

        if y % 3 != 0:
            continue

        laserRay = np.array(laserRay)
        #laserRay = rotZ(math.pi, laserRay)

        laserRaysX.append(laser.origin()[0])
        laserRaysX.append((laser.origin() + 100.0*laserRay)[0])
        laserRaysY.append(laser.origin()[1])
        laserRaysY.append((laser.origin() + 100.0*laserRay)[1])
        laserRaysZ.append(laser.origin()[2])
        laserRaysZ.append((laser.origin() + 100.0*laserRay)[2])

        sampleAt = [[0,0], [512, 0], [512, 256], [0, 256]]
        colors = ['green', 'purple', 'red', 'blue']

        for color, sample in zip(colors, sampleAt):
            cameraRay = camera.getRay(np.array(sample))
            ax.plot([0.0, 100.0*cameraRay[0]], [0.0, 100.0*cameraRay[1]], [0.0, 100.0*cameraRay[2]], color=color)

    ax.plot(laserRaysX, laserRaysY, laserRaysZ)
    plt.show()

def plotLaserRaysCameraRayHits(laserOrigin, laserRay, cameraRay, hitA, hitDir):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    laserRaysX = list()
    laserRaysY = list()
    laserRaysZ = list()

    cameraRaysX = list()
    cameraRaysY = list()
    cameraRaysZ = list()

    connectingPointX = list()
    connectingPointY = list()
    connectingPointZ = list()
    
    connectingPointX.append(hitA[0])
    connectingPointX.append(hitA[0] + hitDir[0])
    connectingPointY.append(hitA[1])
    connectingPointY.append(hitA[1] + hitDir[1])
    connectingPointZ.append(hitA[2])
    connectingPointZ.append(hitA[2] + hitDir[2])

    laserRaysX.append(laserOrigin[0])
    laserRaysX.append((laserOrigin + 100.0*laserRay)[0])
    laserRaysY.append(laserOrigin[1])
    laserRaysY.append((laserOrigin + 100.0*laserRay)[1])
    laserRaysZ.append(laserOrigin[2])
    laserRaysZ.append((laserOrigin + 100.0*laserRay)[2])

    cameraRaysX.append(0.0)
    cameraRaysX.append(100.0*cameraRay[0])
    cameraRaysY.append(0.0)
    cameraRaysY.append(100.0*cameraRay[1])
    cameraRaysZ.append(0.0)
    cameraRaysZ.append(100.0*cameraRay[2])

    ax.plot(laserRaysX, laserRaysY, laserRaysZ)
    ax.plot(cameraRaysX, cameraRaysY, cameraRaysZ)
    ax.plot(connectingPointX, connectingPointY, connectingPointZ)
    plt.show()

def plot_3d(laser, camera, points3D = None, cameraRays = None, closestRay = None):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    laserRaysX = list()
    laserRaysY = list()
    laserRaysZ = list()

    cameraRaysX = list()
    cameraRaysY = list()
    cameraRaysZ = list()

    for count, laserRay in enumerate(laser.rays().tolist()):
        laserRay = np.array(laserRay)

        laserRaysX.append(laser.origin()[0])
        laserRaysX.append((laser.origin() + 100.0*laserRay)[0])
        laserRaysY.append(laser.origin()[1])
        laserRaysY.append((laser.origin() + 100.0*laserRay)[1])
        laserRaysZ.append(laser.origin()[2])
        laserRaysZ.append((laser.origin() + 100.0*laserRay)[2])
    
    if cameraRays:
        for cameraRay in cameraRays:
            cameraRaysX.append(0.0)
            cameraRaysX.append(100.0*cameraRay[0])
            cameraRaysY.append(0.0)
            cameraRaysY.append(100.0*cameraRay[1])
            cameraRaysZ.append(0.0)
            cameraRaysZ.append(100.0*cameraRay[2])

        if closestRay:
            ax.plot(closestRay[0], closestRay[1], closestRay[2])

    else:
        for y in range(0, 512, 16):
            for x in range(0, 256, 16):
                cameraRay = camera.getRay(np.array([y, x]))
        
                cameraRaysX.append(0.0)
                cameraRaysX.append(100.0*cameraRay[0])
                cameraRaysY.append(0.0)
                cameraRaysY.append(100.0*cameraRay[1])
                cameraRaysZ.append(0.0)
                cameraRaysZ.append(100.0*cameraRay[2])

    if points3D:
        xs = [x[0] for x in points3D]
        ys = [x[1] for x in points3D]
        zs = [x[2] for x in points3D]
        ax.scatter(xs, ys, zs)


    ax.plot(laserRaysX, laserRaysY, laserRaysZ)
    ax.plot(cameraRaysX, cameraRaysY, cameraRaysZ)
    plt.show()


def visualize_laser_grid(height, width, rayOrigin, laserRays, camera_matrix, image=None, radiusMin=1, radiusMax=5, intervalMin=40, intervalMax=100):
    numHits = list()
    distances = list()
    for k in range(intervalMin, intervalMax, 1):
        print("Distance from Laser Origin: {0}".format(k))
        laser_image = np.zeros((height, width), dtype=np.uint8)
        for i in range(laserRays.shape[0]):
            _, distanceMin = main.projectToImagePlane(50, rayOrigin, laserRays[i, :], camera_matrix)
            _, distanceMax = main.projectToImagePlane(80, rayOrigin, laserRays[i, :], camera_matrix)
            
            r1, _ = main.getPointOnRayFromOrigin(rayOrigin, laserRays[i, :], k)

            point2d, distance = main.projectToImagePlane(r1[0], rayOrigin, laserRays[i, :], camera_matrix)
            radius = radiusMax - ((distance - distanceMin) / (distanceMax - distanceMin)) * (radiusMax - radiusMin)
            radius = 1.0 if radius < 1.0 else radius

            cv2.circle(laser_image, (math.floor(point2d[0]), math.floor(point2d[1])), radius=round(radius), color=255, thickness=-1)
        
        numHits.append(len((laser_image & image).nonzero()[0]))
        distances.append(k)
        print("Number of Hits: {0}".format(numHits[-1]))
        cv2.imshow("Laserpoints Sim", laser_image)
        cv2.waitKey(0)
    
    cv2.destroyAllWindows()
    return distances, numHits



def plotPoints3D(points3D):
    print("Plotting 3D Points")
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    xs = list()
    ys = list()
    zs = list()
    count = 0
    for i in range(0, len(points3D)):
        if np.isnan(points3D[i]).any():
            continue
        count += 1
        xs.append(points3D[i][0])
        ys.append(points3D[i][1])
        zs.append(points3D[i][2])

    
    ax.scatter(xs, ys, zs)
    ax.view_init(elev=-142, azim=-40)
    plt.show(block=True)


def write_images(path, points3D):
    frameWriteCount = 0
    for k in range(len(points3D[0])):
        print("Plotting Image: {0}".format(k))
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")

        xs = list()
        ys = list()
        zs = list()
        count = 0
        for i in range(0, len(points3D)):
            if np.isnan(points3D[i][k]).any():
                continue
            count += 1
            xs.append(points3D[i][k][0])
            ys.append(points3D[i][k][1])
            zs.append(points3D[i][k][2])

        #TODO: Can we find a way that doesn't need this random barrier?
        if count <= 50:
            continue
        
        ax.scatter(xs, ys, zs)
        ax.view_init(elev=-142, azim=-40)
        plt.savefig(path + "{0:05d}.png".format(frameWriteCount))
        frameWriteCount += 1


def show_3d_triangulation(points3D):
    frameWriteCount = 0
    for k in range(len(points3D[0])):
        print("Plotting Image: {0}".format(k))
        fig = plt.figure()
        canvas = FigureCanvas(fig)
        ax = fig.add_subplot(111, projection='3d')
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")

        xs = list()
        ys = list()
        zs = list()
        count = 0
        for i in range(0, len(points3D)):
            if np.isnan(points3D[i][k]).any():
                continue
            count += 1
            xs.append(points3D[i][k][0])
            ys.append(points3D[i][k][1])
            zs.append(points3D[i][k][2])

        if count <= 50:
            continue

        ax.scatter(xs, ys, zs)
        ax.view_init(elev=-142, azim=-40)
        canvas.draw()
        width, height = fig.get_size_inches() * fig.get_dpi()
        image = np.frombuffer(canvas.tostring_rgb(), dtype=np.uint8).reshape(int(height), int(width), 3)

        cv2.imshow("Triangulated Vocalfold", image)
        cv2.waitKey(25)


def show_3d_triangulation2(points3D):
    frameWriteCount = 0
    for i in range(len(points3D)):
        print("Plotting Image: {0}".format(i))
        fig = plt.figure()
        canvas = FigureCanvas(fig)
        ax = fig.add_subplot(111, projection='3d')
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")

        xs = list()
        ys = list()
        zs = list()
        count = 0
        for j in range(0, len(points3D[i])):
            if np.isnan(points3D[i][j]).any():
                continue
            count += 1
            xs.append(points3D[i][j][0])
            ys.append(points3D[i][j][1])
            zs.append(points3D[i][j][2])

        if count <= 50:
            continue

        ax.scatter(xs, ys, zs)
        ax.view_init(elev=-142, azim=-40)
        canvas.draw()
        width, height = fig.get_size_inches() * fig.get_dpi()
        image = np.frombuffer(canvas.tostring_rgb(), dtype=np.uint8).reshape(int(height), int(width), 3)

        cv2.imshow("Triangulated Vocalfold", image)
        cv2.waitKey(25)


class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        FancyArrowPatch.__init__(self, (0,0), (0,0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
        self.set_positions((xs[0],ys[0]),(xs[1],ys[1]))
        FancyArrowPatch.draw(self, renderer)


def generateProjectionImage(camera, laser, plane, imageHeight, imageWidth): 

    ray = Objects.Ray(laser.origin(), laser.rays())

    points3D = laser.origin() + plane.rayIntersection(ray) * ray._direction

    points2D = helper.project3DPointToImagePlaneMat(points3D, camera.intrinsic())

    image = np.ones((imageHeight, imageWidth), dtype=np.uint8)*255
    for i in range(points2D.shape[0]):
        cv2.circle(image, points2D[i].astype(np.int), radius=2, color=0, thickness=-1)

    return image



    

def generateEPCLineImage(camera, laser, minDistance, maxDistance, imageHeight, imageWidth): 

    minPlane = Objects.Plane(np.array([[0.0, 0.0, 1.0]]), np.array([[0.0, 0.0, minDistance]]))
    maxPlane = Objects.Plane(np.array([[0.0, 0.0, 1.0]]), np.array([[0.0, 0.0, maxDistance]]))
    ray = Objects.Ray(laser.origin(), laser.rays())

    points3Dmin = laser.origin() + minPlane.rayIntersection(ray) * ray._direction
    points3Dmax = laser.origin() + maxPlane.rayIntersection(ray) * ray._direction

    points2Dmin = helper.project3DPointToImagePlaneMat(points3Dmin, camera.intrinsic())
    points2Dmax = helper.project3DPointToImagePlaneMat(points3Dmax, camera.intrinsic())

    image = np.zeros((imageHeight, imageWidth), dtype=np.uint8)
    for i in range(points2Dmin.shape[0]):
        temp_image = np.zeros((imageHeight, imageWidth), dtype=np.uint8)
        cv2.line(temp_image, points2Dmin[i].astype(np.int), points2Dmax[i].astype(np.int), color=1, thickness=2)
        image += temp_image

    image = image.astype(np.float32)
    image = ((image / image.max()) * 255).astype(np.uint8)

    return image


from PyQt5.QtWidgets import QApplication
import Viewer
import igl
import os
import time
import threading

def MeshViewer(points_left, points_right, zSubdivs):
    import Mesh
    pts_left, pts_right, faces = Mesh.generate_BM5_mesh(points_left, points_right, zSubdivs)
    vertices_left = pts_left[0].reshape((-1, 3))
    vertices_right = pts_right[0].reshape((-1, 3))

    viewer_app = QApplication(["VOCALOID"])
    viewer = Viewer.Viewer()
    viewer.show()
    viewer.set_column_stretch(0, 1)

    # Add a viewer widget to visualize 3D meshes to our viewer window
    viewer_widget, _ = viewer.add_viewer_widget(0, 0)
    viewer_widget.show()
    viewer_widget.link_light_to_camera()

    # Add a screenshot button
    def screenshot_function():
        viewer.save_screenshot(os.path.join("screenshot.png"))

    def reload_shader_func():
        viewer.reload_shaders_()

    viewer.add_ui_button("Take Screenshot", screenshot_function)

    viewer.add_ui_button("Reload Shaders", reload_shader_func)

    # Add a mesh to our viewer widget
    # This requires three steps:
    # - Adding the mesh vertices and faces
    # - Adding a mesh prefab that contains shader attributes and uniform values
    # - Adding an instance of our prefab whose position is defined by a model matrix

    # Here, we use the lambert shader.
    # This shader requires two things:
    # - A uniform value called 'albedo' for the color of the mesh.
    # - An attribute called 'normal' for the mesh normals.
    uniforms = {}
    vertex_attributes_left = {}
    face_attributes_left = {}
    vertex_attributes_right = {}
    face_attributes_right = {}

    #uniforms["albedo"] = np.array([1.0, 0.6, 0.2])
    uniforms["minmax"] = np.array([np.concatenate([pts_left, pts_right]).reshape((-1, 3))[:, 1].min(), np.concatenate([pts_left, pts_right]).reshape((-1, 3))[:, 1].max()])
    #uniforms["k_ambient"] = np.array([0.0, 0.0, 0.0])
    #uniforms["k_specular"] = np.array([1.0, 1.0, 1.0])
    #uniforms["shininess"] = np.array([50.0])

    # If we want flat shading with normals defined per face.
    #face_normals = igl.per_face_normals(vertices.astype(np.float64), faces, np.array([1.0, 1.0, 1.0])).astype(
    #    np.float32
    #)
    #face_attributes["normal"] = face_normals

    # If we want smooth shading with normals defined per vertex.
    vertex_normals_left = igl.per_vertex_normals(vertices_left, faces, igl.PER_VERTEX_NORMALS_WEIGHTING_TYPE_AREA).astype(np.float32)
    vertex_attributes_left['normal'] = vertex_normals_left

    vertex_normals_right = igl.per_vertex_normals(vertices_right, faces, igl.PER_VERTEX_NORMALS_WEIGHTING_TYPE_AREA).astype(np.float32)
    vertex_attributes_right['normal'] = vertex_normals_right

    mesh_index_left = viewer_widget.add_mesh(vertices_left, faces)
    mesh_prefab_index_left = viewer_widget.add_mesh_prefab(
        mesh_index_left,
        "colormap",
        vertex_attributes=vertex_attributes_left,
        face_attributes=face_attributes_left,
        uniforms=uniforms,
    )
    instance_index_left = viewer_widget.add_mesh_instance(
        mesh_prefab_index_left, np.eye(4, dtype="f")
    )
    viewer_widget.add_wireframe(instance_index_left, line_color=np.array([0.1, 0.1, 0.1]))

    mesh_index_right = viewer_widget.add_mesh(vertices_right, faces)
    mesh_prefab_index_right = viewer_widget.add_mesh_prefab(
        mesh_index_right,
        "colormap",
        vertex_attributes=vertex_attributes_right,
        face_attributes=face_attributes_right,
        uniforms=uniforms,
    )
    instance_index_right = viewer_widget.add_mesh_instance(
        mesh_prefab_index_right, np.eye(4, dtype="f")
    )
    viewer_widget.add_wireframe(instance_index_right, line_color=np.array([0.1, 0.1, 0.1]))


    def animate_func():
        #global pts_, vertices_right, faces
        for t in range(pts_left.shape[0]):
            viewer_widget.update_mesh_vertices(mesh_index_left, pts_left[t].reshape((-1, 3)).astype(np.float32))
            viewer_widget.update_mesh_vertices(mesh_index_right, pts_right[t].reshape((-1, 3)).astype(np.float32))
            viewer_widget.update()
            time.sleep(0.04)


    # This animation needs to run in its own thread to not block the rendering on the main thread.
    def animate_func_async():
        animating_thread = threading.Thread(target=animate_func, args=())
        animating_thread.start()


    viewer.add_ui_button("Animate", animate_func_async)

    # Launch the Qt application
    viewer_app.exec()