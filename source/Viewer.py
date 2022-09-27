#import sys
#sys.path.append("source/GUI/")
#sys.path.append("source/")

from PyQt5.QtCore import Qt, pyqtSignal, QTimer, QThread
from PyQt5.QtWidgets import QWidget, QFrame, QVBoxLayout, QHBoxLayout, QGridLayout

import igl
import scipy
import cv2
import numpy as np

from PyIGL_viewer.viewer.viewer_widget import ViewerWidget
from GraphWidget import GraphWidget
from ImageViewerWidget import ImageViewerWidget
from MainMenuWidget import MainMenuWidget
from VideoPlayerWidget import VideoPlayerWidget
from QLines import QHLine, QVLine

import cProfile

import KocSegmentation
import SiliconeSegmentation
import NeuralSegmentation

import Mesh
import Camera
import Laser
import helper
import RHC
import VoronoiRHC
import Correspondences
import Triangulation
import SiliconeSurfaceReconstruction


class Viewer(QWidget):
    close_signal = pyqtSignal()
    screenshot_signal = pyqtSignal(str)
    legend_signal = pyqtSignal(list, list)
    load_shader_signal = pyqtSignal()
    image_changed_signal = pyqtSignal(int)

    def __init__(self):
        super().__init__()
        self.viewer_palette = {
            "viewer_background": "#252526",
            "viewer_widget_border_color": "#555555",
            "menu_background": "#333333",
            "ui_element_background": "#3e3e3e",
            "ui_group_border_color": "#777777",
            "font_color": "#ccccbd"
        }

        self.setAutoFillBackground(True)
        self.setStyleSheet(
            f"background-color: {self.viewer_palette['viewer_background']}; color: {self.viewer_palette['font_color']};" + "QSlider::handle:horizontal{background-color: white;};" + "QLineEdit { background-color: yellow }")

        # SET UP FOR LATER

        self.camera = None
        self.laser = None

        self.images = None
        self.segmentations = None
        self.laserdots = None

        self.left_vf_triangulated = None
        self.right_vf_triangulated = None

        self.plots_set = False
        self.meshes_set = False
        self.images_set = False

        self.obj_ids = {"LeftVF": None, "RightVF": None, "LeftVFOpt": None, "RightVFOpt": None}
        self.point_cloud_mesh_core = None
        self.point_cloud_id = None
        self.point_cloud_offsets = []
        self.point_cloud_elements = []


        self.main_layout = QHBoxLayout(self)

        self.image_widget = ImageViewerWidget(self)
        self.player_widget = VideoPlayerWidget(self)
        self.graph_widget = GraphWidget(self)
        self.menu_widget = MainMenuWidget(self.viewer_palette, self)
        self.viewer_widget = self.add_viewer_widget(0, 0, 1, 1)

        # Compatibility with PyIGL viewer widget
        self.linked_cameras = False 
        
        widg_right = QWidget(self)
        vertical_layout = QVBoxLayout(widg_right)
        vertical_layout.addWidget(self.image_widget)
        vertical_layout.addWidget(QHLine())
        vertical_layout.addWidget(self.graph_widget)

        widg_middle = QWidget(self)
        vertical_layout = QVBoxLayout(widg_middle)
        vertical_layout.addWidget(self.viewer_widget, 90)
        vertical_layout.addWidget(QHLine())
        vertical_layout.addWidget(self.player_widget, 10)

        self.main_layout.addWidget(self.menu_widget, 10)
        self.main_layout.addWidget(QVLine())
        self.main_layout.addWidget(widg_middle, 60)
        self.main_layout.addWidget(QVLine())
        self.main_layout.addWidget(widg_right, 30)


        self.viewer_widget.link_light_to_camera()


        self.menu_widget.ocs_widget.fileOpenedSignal.connect(self.loadData)
        self.menu_widget.button_dict["Segment Images"].clicked.connect(self.segmentImages)
        self.menu_widget.button_dict["Build Correspondences"].clicked.connect(self.buildCorrespondences)
        self.menu_widget.button_dict["Triangulate"].clicked.connect(self.triangulate)
        self.menu_widget.button_dict["Dense Shape Estimation"].clicked.connect(self.denseShapeEstimation)
        self.menu_widget.button_dict["Least Squares Optimization"].clicked.connect(self.lsqOptimization)
        self.menu_widget.button_dict["Automatic Reconstruction"].clicked.connect(self.automaticReconstruction)


        self.timer_thread = QThread(self)
        self.timer_thread.started.connect(self.gen_timer_thread)

        self.image_timer_thread = QThread(self)
        self.image_timer_thread.started.connect(self.gen_image_timer_thread)

        self.player_widget.slider.valueChanged.connect(self.updatePointCloud)

        self.timer_thread.start()
        self.image_timer_thread.start()

        self.loadData("assets/camera_calibration.json", "assets/laser_calibration.json", "assets/example_vid.avi")

    def gen_timer_thread(self):
        timer = QTimer(self.timer_thread)
        timer.timeout.connect(self.player_widget.update_frame_when_playing)
        timer.timeout.connect(self.animate_func)
        timer.setInterval(25)
        timer.start()

    def gen_image_timer_thread(self):
        timer = QTimer(self.image_timer_thread)
        timer.timeout.connect(self.update_images_func)
        timer.setInterval(25)
        timer.start()

    def add_viewer_widget(self, x, y, row_span=1, column_span=1):
        group_layout = QGridLayout()
        group_layout.setSpacing(0)
        group_layout.setContentsMargins(0, 0, 0, 0)
        widget = QFrame(self)
        widget.setLineWidth(2)
        widget.setLayout(group_layout)
        widget.setObjectName("groupFrame")
        widget.setStyleSheet(
            "#groupFrame { border: 1px solid "
            + self.viewer_palette["viewer_widget_border_color"]
            + "; }"
        )

        viewer_widget = ViewerWidget(self)
        viewer_widget.setFocusPolicy(Qt.ClickFocus)
        group_layout.addWidget(viewer_widget)
        #self.main_layout.addWidget(widget, x, y, row_span, column_span)
        viewer_widget.show()
        return viewer_widget
    
    def update_all_viewers(self):
        self.viewer_widget.update()

    def set_background_color(self, color):
        self.viewer_palette["viewer_background"] = color
        self.setStyleSheet(
            f"background-color: {self.viewer_palette['viewer_background']}"
        )

    def keyPressEvent(self, e):
        if e.key() == Qt.Key_Escape:
            sys.stdout.close()
            sys.stderr.close()
            self.close_signal.emit()
            exit()

    def closeEvent(self, event):
        self.close_signal.emit()

    def animate_func(self):
        curr_frame = self.player_widget.getCurrentFrame()
        
        if curr_frame == self.player_widget.slider.maximum():
            return

        self.updateMesh(curr_frame)
        self.updatePlots(curr_frame)

    def update_images_func(self):
        if self.images_set:
            curr_frame = self.player_widget.getCurrentFrame()

            if curr_frame == self.player_widget.slider.maximum():
                return

            self.image_widget.updateImages(self.images[curr_frame], self.segmentations[curr_frame], self.laserdots[curr_frame])

    def updateMesh(self, frameNum):
        if self.meshes_set:
            self.viewer_widget.update_mesh_vertices(self.obj_ids["LeftVF"], self.pts_left[frameNum].reshape((-1, 3)).astype(np.float32))
            self.viewer_widget.update_mesh_vertices(self.obj_ids["RightVF"], self.pts_right[frameNum].reshape((-1, 3)).astype(np.float32))
            self.updatePointCloud(frameNum)
            self.viewer_widget.update()

    def updatePlots(self, frameNum):
        if self.plots_set:
            self.graph_widget.updateLines(frameNum)

    def updatePointCloud(self, frameNum):
        self.point_cloud_mesh_core.offset = self.point_cloud_offsets[self.player_widget.slider.value()]
        self.point_cloud_mesh_core.number_elements = self.point_cloud_elements[self.player_widget.slider.value()]


    def addVocalfoldMeshes(self, points_left, points_right, zSubdivisions):
        self.pts_left, self.pts_right, faces = Mesh.generate_BM5_mesh(points_left, points_right, zSubdivisions)
        vertices_left = self.pts_left[self.player_widget.getCurrentFrame()].reshape((-1, 3))
        vertices_right = self.pts_right[self.player_widget.getCurrentFrame()].reshape((-1, 3))

        if self.obj_ids["RightVF"] is not None:
            self.viewer_widget.remove_mesh(self.obj_ids["RightVF"])

        if self.obj_ids["LeftVF"] is not None:
            self.viewer_widget.remove_mesh(self.obj_ids["LeftVF"])


        self.graph_widget.updateGraphs(np.array(points_right)[:, :, 1].max(axis=1), np.array(points_left)[:, :, 1].max(axis=1), np.array(points_left)[:, :, 1].max(axis=1))
        self.setPlots()

        uniforms = {}
        vertex_attributes_left = {}
        face_attributes_left = {}
        face_attributes_right = {}
        vertex_attributes_right = {}

        uniforms["minmax"] = np.array([np.concatenate([self.pts_left, self.pts_right]).reshape((-1, 3))[:, 1].min(), np.concatenate([self.pts_left, self.pts_right]).reshape((-1, 3))[:, 1].max()])

        vertex_normals_left = igl.per_vertex_normals(vertices_left, faces, igl.PER_VERTEX_NORMALS_WEIGHTING_TYPE_AREA).astype(np.float32)
        vertex_attributes_left['normal'] = vertex_normals_left

        vertex_normals_right = igl.per_vertex_normals(vertices_right, faces, igl.PER_VERTEX_NORMALS_WEIGHTING_TYPE_AREA).astype(np.float32)
        vertex_attributes_right['normal'] = vertex_normals_right
        
        mesh_index_left = self.viewer_widget.add_mesh(vertices_left, faces)
        self.obj_ids["LeftVF"] = self.viewer_widget.add_mesh_prefab(
            mesh_index_left,
            "colormap",
            vertex_attributes=vertex_attributes_left,
            face_attributes=face_attributes_left,
            uniforms=uniforms,
        )
        instance_index_left = self.viewer_widget.add_mesh_instance(
            self.obj_ids["LeftVF"], np.eye(4, dtype="f")
        )
        self.viewer_widget.add_wireframe(instance_index_left, line_color=np.array([0.1, 0.1, 0.1]))

        mesh_index_right = self.viewer_widget.add_mesh(vertices_right, faces)
        self.obj_ids["RightVF"] = self.viewer_widget.add_mesh_prefab(
            mesh_index_right,
            "colormap",
            vertex_attributes=vertex_attributes_right,
            face_attributes=face_attributes_right,
            uniforms=uniforms,
        )
        instance_index_right = self.viewer_widget.add_mesh_instance(
            self.obj_ids["RightVF"], np.eye(4, dtype="f")
        )

        self.viewer_widget.add_wireframe(instance_index_right, line_color=np.array([0.1, 0.1, 0.1]))

    def toggleVisibility(self, instance_id):
        mesh_instance = self.viewer_widget.get_mesh_instance(instance_id)
        mesh_instance.set_visibility(not mesh_instance.get_visibility())
        self.viewer_widget.update()

    def setImages(self):
        self.images_set = True

    def unsetImages(self):
        self.images_set = False

    def setMeshes(self):
        self.meshes_set = True

    def unsetMeshes(self):
        self.meshes_set = False

    def setPlots(self):
        self.plots_set = True

    def unsetPlots(self):
        self.plots_set = False

    def loadData(self, camera_path, laser_path, video_path):
        self.camera = Camera.Camera(camera_path, "JSON")
        self.laser = Laser.Laser(laser_path, "JSON")
        self.images = helper.loadVideo(video_path, self.camera.intrinsic(), self.camera.distortionCoefficients())
        self.segmentations = self.images
        self.laserdots = self.images

        self.player_widget.setSliderRange(0, len(self.images))

        self.images_set = True

    def segmentImages(self):
        if self.menu_widget.getSubmenuValue("Segmentation", "Koc et al"):
            self.segmentator = KocSegmentation.KocSegmentator(self.images)
        elif self.menu_widget.getSubmenuValue("Segmentation", "Neural Segmentation"):
            self.segmentator = NeuralSegmentation.NeuralSegmentator(self.images)
        elif self.menu_widget.getSubmenuValue("Segmentation", "Silicone Segmentation"):
                self.segmentator = SiliconeSegmentation.SiliconeSegmentator(self.images)
        else:
            print("Please choose a Segmentation Algorithm")

        x, w, y, h = self.segmentator.getROI()
        self.roi = self.segmentator.getROIImage()

        segmentations = list()
        laserdots = list()

        for index in range(len(self.segmentator)):
            base_image = self.segmentator.getImage(index).copy()

            segmentation_image = self.segmentator.getSegmentation(index).copy()
            gml_a, gml_b = self.segmentator.getGlottalMidline(index)

            segmentation_image = cv2.cvtColor(segmentation_image, cv2.COLOR_GRAY2BGR)

            cv2.rectangle(segmentation_image, (x, y), (x+w,y+h), color=(255, 0, 0), thickness=2)
            try:
                cv2.line(segmentation_image, gml_a.astype(np.int32), gml_b.astype(np.int32), color=(125, 125, 0), thickness=2)
            except:
                pass
            segmentations.append(cv2.cvtColor(base_image, cv2.COLOR_GRAY2BGR) | segmentation_image)

            laserdot_image = self.segmentator.getLocalMaxima(index).copy()
            laserdot_image = cv2.dilate(laserdot_image, np.ones((3,3)))
            laserdot_image = np.where(laserdot_image > 0, 255, 0).astype(np.uint8)
            laserdot_image = cv2.cvtColor(laserdot_image, cv2.COLOR_GRAY2BGR)
            laserdot_image[:, :, [0, 2]] = 0
            laserdots.append(cv2.cvtColor(base_image, cv2.COLOR_GRAY2BGR) | laserdot_image)
        
        self.segmentations = segmentations
        self.laserdots = laserdots

    def buildCorrespondences(self):
        min_search_space = float(self.menu_widget.getSubmenuValue("RHC", "Minimum Distance"))
        max_search_space = float(self.menu_widget.getSubmenuValue("RHC", "Maximum Distance"))
        thresh = float(self.menu_widget.getSubmenuValue("RHC", "GA Thresh"))
        set_size = int(self.menu_widget.getSubmenuValue("RHC", "Consensus Size"))
        iterations = int(self.menu_widget.getSubmenuValue("RHC", "Iterations"))


        if self.menu_widget.getSubmenuValue("RHC", "Activated"):
            pixelLocations, laserGridIDs = Correspondences.initialize(self.laser, self.camera, self.segmentator, min_search_space, max_search_space)
            self.grid2DPixLocations = RHC.RHC(laserGridIDs, pixelLocations, self.segmentator, self.camera, self.laser, set_size, iterations)
        elif self.menu_widget.getSubmenuValue("Voronoi RHC", "Activated"):
            cf = VoronoiRHC.CorrespondenceFinder(self.camera, self.laser, minWorkingDistance=min_search_space, maxWorkingDistance=max_search_space, threshold=thresh)
            correspondences = []
            vectorized_maxima = np.flip(np.stack(self.maxima.nonzero(), axis=1), axis=1)
            while len(correspondences) == 0:
                correspondences = cf.establishCorrespondences(vectorized_maxima)

            self.grid2DPixLocations = [[self.laser.getXYfromN(id), np.flip(pix)] for id, pix in correspondences]

        pixel_coords = np.array(self.grid2DPixLocations)[:, 1, :].astype(np.int32)
        debug_img = np.zeros(self.segmentator.getImage(0).shape, np.uint8)
        debug_img[pixel_coords[:, 0], pixel_coords[:, 1]] = 255
        debug_img = cv2.dilate(debug_img, np.ones((3,3)))

        base_img = self.images[self.segmentator.getClosedGlottisIndex()].copy()
        base_img = cv2.cvtColor(base_img, cv2.COLOR_GRAY2BGR)
        base_img = base_img | cv2.cvtColor(debug_img, cv2.COLOR_GRAY2BGR)
        
        self.image_widget.updateImage(base_img, self.image_widget.getWidget("Closed Vocal Folds"))
        
    def triangulate(self):
        min_search_space = float(self.menu_widget.getSubmenuValue("RHC", "Minimum Distance"))
        max_search_space = float(self.menu_widget.getSubmenuValue("RHC", "Maximum Distance"))

        temporalCorrespondence = Correspondences.generateFramewise(self.segmentator, self.grid2DPixLocations)
        self.triangulatedPoints = np.array(Triangulation.triangulationMat(self.camera, self.laser, temporalCorrespondence, min_search_space, max_search_space, min_search_space, max_search_space))

        
    def denseShapeEstimation(self):    
        zSubdivisions = int(self.menu_widget.getSubmenuValue("Tensor Product M5", "Z Subdivisions"))
        #glottalmidline = self.segmentator.getGlottalMidline(self.images[self.frameOfClosedGlottis])

        if self.point_cloud_id is not None:
            self.viewer_widget.remove_mesh(self.point_cloud_id)
    
        self.leftDeformed, self.rightDeformed, self.leftPoints, self.rightPoints, self.pointclouds = SiliconeSurfaceReconstruction.controlPointBasedARAP(self.triangulatedPoints, self.camera, self.segmentator, zSubdivisions=zSubdivisions)

        self.point_cloud_offsets = [0]
        self.point_cloud_elements = [self.pointclouds[0].shape[0]]

        for pc in self.pointclouds[1:]:
            num_verts = pc.shape[0]
            self.point_cloud_offsets.append(self.point_cloud_offsets[-1] + num_verts)
            self.point_cloud_elements.append(num_verts)

        super_point_cloud = np.concatenate(self.pointclouds, axis=0)
        self.point_cloud_id = self.viewer_widget.display_point_cloud(super_point_cloud)
        self.viewer_widget.process_mesh_events()
        self.point_cloud_mesh_core = self.viewer_widget.get_mesh(self.point_cloud_id).mesh_core

        self.addVocalfoldMeshes(self.leftDeformed, self.rightDeformed, zSubdivisions)
        self.setMeshes()
        
    def lsqOptimization(self):
        zSubdivisions = int(self.menu_widget.getSubmenuValue("Tensor Product M5", "Z Subdivisions"))
        iterations = int(self.menu_widget.getSubmenuValue("Least Squares Optimization", "Iterations"))
        lr = float(self.menu_widget.getSubmenuValue("Least Squares Optimization", "Learning Rate"))
        window_size = int(self.menu_widget.getSubmenuValue("Temporal Smoothing", "Window Size"))

        self.optimizedLeft = SiliconeSurfaceReconstruction.surfaceOptimization(self.leftDeformed, self.leftPoints, zSubdivisions=zSubdivisions, iterations=iterations, lr=lr)
        self.optimizedLeft = np.array(self.optimizedLeft)
        self.smoothedLeft = scipy.ndimage.uniform_filter(self.optimizedLeft, size=(window_size, 1, 1), mode='reflect', cval=0.0, origin=0)

        self.optimizedRight = SiliconeSurfaceReconstruction.surfaceOptimization(self.rightDeformed, self.rightPoints, zSubdivisions=zSubdivisions, iterations=iterations, lr=lr)
        self.optimizedRight = np.array(self.optimizedRight)
        self.smoothedRight = scipy.ndimage.uniform_filter(self.optimizedRight, size=(window_size, 1, 1), mode='reflect', cval=0.0, origin=0)

        self.addVocalfoldMeshes(self.smoothedLeft, self.smoothedRight, zSubdivisions)
        
    def automaticReconstruction(self):
        self.segmentImages()
        self.buildCorrespondences()
        self.triangulate()
        self.denseShapeEstimation()
        self.lsqOptimization()