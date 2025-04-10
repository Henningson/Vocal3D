# import sys
# sys.path.append("source/GUI/")
# sys.path.append("source/")

import cProfile
import os

import Camera
import correspondence_estimation
import Correspondences
import cv2
import feature_estimation
import helper
import igl
import KocSegmentation
import kornia
import Laser
import Mesh
import NeuralSegmentation
import numpy as np
import point_tracking
import reconstruction_pipeline
import RHC
import scipy
import SiliconeSegmentation
import SiliconeSurfaceReconstruction
import surface_reconstruction
import torch
import Triangulation
import VoronoiRHC
from GraphWidget import GraphWidget
from ImageViewerWidget import ImageViewerWidget
from MainMenuWidget import MainMenuWidget
from PyIGL_viewer.viewer.viewer_widget import ViewerWidget
from PyQt5.QtCore import Qt, QThread, QTimer, pyqtSignal
from PyQt5.QtWidgets import (QFileDialog, QFrame, QGridLayout, QHBoxLayout,
                             QScrollArea, QVBoxLayout, QWidget)
from QLines import QHLine, QVLine
from VideoPlayerWidget import VideoPlayerWidget


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
            "font_color": "#ccccbd",
        }

        self.setAutoFillBackground(True)
        self.setStyleSheet(
            f"background-color: {self.viewer_palette['viewer_background']}; color: {self.viewer_palette['font_color']};"
            + "QSlider::handle:horizontal{background-color: white;};"
            + "QLineEdit { background-color: yellow }"
        )

        # SET UP FOR LATER

        self.camera = None
        self.laser = None

        self.images = None
        self.video = None
        self.segmentations = None
        self.laserdots = None

        self.left_vf_triangulated = None
        self.right_vf_triangulated = None

        self.plots_set = False
        self.meshes_set = False
        self.images_set = False

        self.obj_ids = {
            "LeftVF": None,
            "RightVF": None,
            "LeftVFOpt": None,
            "RightVFOpt": None,
        }
        self.point_cloud_mesh_core = None
        self.point_cloud_id = None
        self.point_cloud_offsets = []
        self.point_cloud_elements = []

        self.main_layout = QHBoxLayout(self)

        self.image_widget = ImageViewerWidget(self)
        self.player_widget = VideoPlayerWidget(self)
        self.graph_widget = GraphWidget(self)

        menu_widget = MainMenuWidget(self.viewer_palette, self)

        scrollable = QScrollArea()
        scrollable.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOn)
        scrollable.setWidgetResizable(True)
        scrollable.setWidget(menu_widget)
        scrollable.setMinimumWidth(400)
        self.menu_widget = scrollable
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

        self.menu_widget.widget().ocs_widget.fileOpenedSignal.connect(self.loadData)
        self.menu_widget.widget().button_dict["Segment Images"].clicked.connect(
            self.segmentImages
        )
        self.menu_widget.widget().button_dict["Build Correspondences"].clicked.connect(
            self.buildCorrespondences
        )
        self.menu_widget.widget().button_dict["Triangulate"].clicked.connect(self.triangulate)
        self.menu_widget.widget().button_dict["Dense Shape Estimation"].clicked.connect(
            self.denseShapeEstimation
        )
        self.menu_widget.widget().button_dict["Least Squares Optimization"].clicked.connect(
            self.lsqOptimization
        )
        self.menu_widget.widget().button_dict["Automatic Reconstruction"].clicked.connect(
            self.automaticReconstruction
        )
        self.menu_widget.widget().button_dict["Save Models"].clicked.connect(self.saveModels)
        self.menu_widget.widget().button_dict["Track Points"].clicked.connect(self.trackPoints)


        self.timer_thread = QThread(self)
        self.timer_thread.started.connect(self.gen_timer_thread)

        self.image_timer_thread = QThread(self)
        self.image_timer_thread.started.connect(self.gen_image_timer_thread)

        self.player_widget.slider.valueChanged.connect(self.updatePointCloud)
        self.player_widget.slider.valueChanged.connect(self.update_images_func)

        self.timer_thread.start()
        self.image_timer_thread.start()

        path = "/media/nu94waro/Windows_C/save/datasets/HLEDataset/dataset"
        
        '''
        self.loadData(
            "assets/camera_calibration.json",
            "assets/laser_calibration.json",
            "assets/example_vid.avi",
        )'''
        
        self.loadData(
            os.path.join(path, "camera_calibration.json"),
            os.path.join(path, "laser_calibration.json"),
            os.path.join(path, "MK/MK.avi"),
        )

        self._reconstruction_pipeline = reconstruction_pipeline.ReconstructionPipeline(
            self.camera, 
            self.laser, 
            feature_estimation.SiliconeFeatureEstimator(), 
            point_tracking.PointTracker(), 
            correspondence_estimation.CorrespondenceEstimator(), 
            surface_reconstruction.SurfaceReconstructor())
        

        glottal_outline_images = torch.from_numpy(np.load("load/glottal_outline_images.npy")).cuda()
        glottis_segmentations = torch.from_numpy(np.load("load/glottis_segmentations.npy",)).cuda()
        vocalfold_segmentations = torch.from_numpy(np.load("load/vocalfold_segmentations.npy",)).cuda()
        laserpoint_segmentations = torch.from_numpy(np.load("load/laserpoint_segmentations.npy",)).cuda()


        glottal_midlines = []
        glottal_outlines = []
        for i in range(475):
            go = torch.from_numpy(np.load(f"load/glottal_outlines{i:05d}.npy")).cuda()
            glottal_outlines.append(go)

            gma = torch.from_numpy(np.load(f"load/glottal_midlines_a{i:05d}.npy")).cuda()
            if (gma == -1).any():
                gma = None

            gmb = torch.from_numpy(np.load(f"load/glottal_midlines_b{i:05d}.npy")).cuda()
            if (gmb == -1).any():
                gmb = None

            glottal_midlines.append([gma, gmb])




        laserpoint_positions = torch.from_numpy(np.load("load/laserpoint_positions.npy")).cuda()

        point_positions = torch.from_numpy(np.load("load/point_positions.npy")).cuda()

        self._reconstruction_pipeline._feature_estimator._glottal_midlines = glottal_midlines
        self._reconstruction_pipeline._glottal_midlines = glottal_midlines

        self._reconstruction_pipeline._feature_estimator._glottal_outline_images = glottal_outline_images

        self._reconstruction_pipeline._feature_estimator._glottal_outlines = glottal_outlines
        self._reconstruction_pipeline._glottal_outlines = glottal_outlines

        self._reconstruction_pipeline._feature_estimator._vocalfold_segmentations = vocalfold_segmentations
        self._reconstruction_pipeline._vocalfold_segmentations = vocalfold_segmentations

        self._reconstruction_pipeline._feature_estimator._laserpoint_segmentations = laserpoint_segmentations
        self._reconstruction_pipeline._laserpoint_estimates = laserpoint_positions

        self._reconstruction_pipeline._feature_estimator._glottis_segmentations = glottis_segmentations
        self._reconstruction_pipeline._glottal_segmentaitons = glottis_segmentations

        self._reconstruction_pipeline._feature_estimator._laserpoint_positions = laserpoint_positions
        self._reconstruction_pipeline._laserpoint_positions = laserpoint_positions

        self._reconstruction_pipeline._optimized_point_positions = point_positions

        feature_images = self._reconstruction_pipeline._feature_estimator.create_feature_images()

        self.segmentations = []
        for feature_image in feature_images:
            self.segmentations.append(feature_image.permute(1, 2, 0).detach().cpu().numpy().astype(np.uint8))

        
        self.graph_widget.updateGraph(
            self._reconstruction_pipeline._feature_estimator.glottalAreaWaveform().tolist(), self.graph_widget.glottal_seg_graph
        )

        self.image_widget.point_viewer.add_points(point_positions.detach().cpu().numpy())
        self.update_images_func()
        

    def update_pipeline(self):
        segmentator = None
        if self.menu_widget.widget().getSubmenuValue("Segmentation", "Koc et al"):
            segmentator = KocSegmentation.KocSegmentator(self.images)
        elif self.menu_widget.widget().getSubmenuValue("Segmentation", "Neural Segmentation"):
            segmentator = NeuralSegmentation.NeuralSegmentator(self.images)
        elif self.menu_widget.widget().getSubmenuValue("Segmentation", "Silicone Segmentation"):
            segmentator = SiliconeSegmentation.SiliconeSegmentator(self.images)
        else:
            print("Please choose a Segmentation Algorithm")


        
        min_search_space = float(
            self.menu_widget.widget().getSubmenuValue("RHC", "Minimum Distance")
        )
        max_search_space = float(
            self.menu_widget.widget().getSubmenuValue("RHC", "Maximum Distance")
        )
        thresh = float(self.menu_widget.widget().getSubmenuValue("RHC", "GA Thresh"))
        set_size = int(self.menu_widget.widget().getSubmenuValue("RHC", "Consensus Size"))
        iterations = int(self.menu_widget.widget().getSubmenuValue("RHC", "Iterations"))
        

    def gen_timer_thread(self):
        timer = QTimer(self.timer_thread)
        timer.timeout.connect(self.player_widget.update_frame_when_playing)
        timer.timeout.connect(self.animate_func)
        timer.setInterval(25)
        timer.start()

    def gen_image_timer_thread(self):
        # timer = QTimer(self.image_timer_thread)
        # timer.timeout.connect(self.update_images_func)
        # timer.setInterval(25)
        # timer.start()
        pass

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
        # self.main_layout.addWidget(widget, x, y, row_span, column_span)
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

        if curr_frame == self.player_widget.slider.maximum() - 1:
            return

        self.updateMesh(curr_frame)
        self.updatePlots(curr_frame)

    def update_images_func(self):
        if self.images_set:
            curr_frame = self.player_widget.getCurrentFrame()

            if curr_frame == self.player_widget.slider.maximum():
                return

            self.image_widget.updateImages(
                self.images[curr_frame],
                self.segmentations[curr_frame],
                curr_frame
            )

            if self.menu_widget.widget().getSubmenuValue("Video Generation", "Generate Video"):
                path = self.menu_widget.widget().getSubmenuValue("Video Generation", "Path")
                path_reco = os.path.join(path, "reco")
                path_video = os.path.join(path, "vid")

                if not os.path.exists(path):
                    os.makedirs(path, exist_ok=True)
                    os.makedirs(path_reco, exist_ok=True)
                    os.makedirs(path_video, exist_ok=True)

                if self.player_widget.isPlaying():
                    curr_frame = self.player_widget.getCurrentFrame()
                    self.viewer_widget.save_screenshot(
                        os.path.join(path_reco, "{:05d}.png".format(curr_frame))
                    )
                    cv2.imwrite(
                        os.path.join(path_video, "{:05d}.png".format(curr_frame)),
                        self.images[curr_frame],
                    )

    def updateMesh(self, frameNum):
        if self.meshes_set:
            self.viewer_widget.update_mesh_vertices(
                self.obj_ids["LeftVF"],
                self.pts_left[frameNum].reshape((-1, 3)).astype(np.float32),
            )
            self.viewer_widget.update_mesh_vertices(
                self.obj_ids["RightVF"],
                self.pts_right[frameNum].reshape((-1, 3)).astype(np.float32),
            )
            # self.updatePointCloud(frameNum)
            self.viewer_widget.update()

    def updatePlots(self, frameNum):
        if self.plots_set:
            self.graph_widget.updateLines(frameNum)

    def updatePointCloud(self, frameNum):
        self.point_cloud_mesh_core.offset = self.point_cloud_offsets[
            self.player_widget.slider.value()
        ]
        self.point_cloud_mesh_core.number_elements = self.point_cloud_elements[
            self.player_widget.slider.value()
        ]

    def addVocalfoldMeshes(self, points_left, points_right, zSubdivisions):
        self.pts_left, self.pts_right, faces_left, faces_right = Mesh.generate_BM5_mesh(
            points_left, points_right, zSubdivisions
        )
        faces = faces_left
        self.faces_left = faces_left.copy()
        self.faces_right = faces_right.copy()
        vertices_left = self.pts_left[self.player_widget.getCurrentFrame()].reshape(
            (-1, 3)
        )
        vertices_right = self.pts_right[self.player_widget.getCurrentFrame()].reshape(
            (-1, 3)
        )

        if self.obj_ids["RightVF"] is not None:
            self.viewer_widget.remove_mesh(self.obj_ids["RightVF"])

        if self.obj_ids["LeftVF"] is not None:
            self.viewer_widget.remove_mesh(self.obj_ids["LeftVF"])

        self.graph_widget.updateGraph(
            np.array(points_left)[:, :, 1].max(axis=1),
            self.graph_widget.height_graph_left,
        )
        self.graph_widget.updateGraph(
            np.array(points_right)[:, :, 1].max(axis=1),
            self.graph_widget.height_graph_right,
        )
        self.setPlots()

        uniforms = {}
        vertex_attributes_left = {}
        face_attributes_left = {}
        face_attributes_right = {}
        vertex_attributes_right = {}

        uniforms["minmax"] = np.array(
            [
                np.concatenate([self.pts_left, self.pts_right])
                .reshape((-1, 3))[:, 1]
                .min(),
                np.concatenate([self.pts_left, self.pts_right])
                .reshape((-1, 3))[:, 1]
                .max(),
            ]
        )

        vertex_normals_left = igl.per_vertex_normals(
            vertices_left, faces, igl.PER_VERTEX_NORMALS_WEIGHTING_TYPE_AREA
        ).astype(np.float32)
        vertex_attributes_left["normal"] = vertex_normals_left

        vertex_normals_right = igl.per_vertex_normals(
            vertices_right, faces, igl.PER_VERTEX_NORMALS_WEIGHTING_TYPE_AREA
        ).astype(np.float32)
        vertex_attributes_right["normal"] = vertex_normals_right

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
        self.viewer_widget.add_wireframe(
            instance_index_left, line_color=np.array([0.1, 0.1, 0.1])
        )

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

        self.viewer_widget.add_wireframe(
            instance_index_right, line_color=np.array([0.1, 0.1, 0.1])
        )

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
        self.images = helper.loadVideo(
            video_path, self.camera.intrinsic(), self.camera.distortionCoefficients()
        )
        self.segmentations = self.images
        self.laserdots = self.images

        self.image_widget.point_viewer.add_video(helper.vid_2_QImage(self.images))

        self.player_widget.setSliderRange(0, len(self.images))
        self.player_widget.stop_video_()

        self.images_set = True
        self.video = torch.from_numpy(np.stack(self.images)).to("cuda")

        self.update_images_func()



    def segmentImages(self):
        segmentator: feature_estimation.FeatureEstimator = None
        if self.menu_widget.widget().getSubmenuValue("Segmentation", "Neural Segmentation"):
            segmentator = feature_estimation.NeuralFeatureEstimator("bla")
        elif self.menu_widget.widget().getSubmenuValue("Segmentation", "Silicone Segmentation"):
            segmentator = feature_estimation.SiliconeFeatureEstimator()
        else:
            print("Please choose a Segmentation Algorithm")

        self._reconstruction_pipeline.set_feature_estimator(segmentator)
        
        
        #segmentator.compute_features(self.video)
        self._reconstruction_pipeline.estimate_features(self.video)

        segmentations = list()
        feature_images = segmentator.create_feature_images()

        for feature_image in feature_images:
            segmentations.append(feature_image.permute(1, 2, 0).detach().cpu().numpy().astype(np.uint8))

        self.graph_widget.updateGraph(
            segmentator.glottalAreaWaveform().tolist(), self.graph_widget.glottal_seg_graph
        )

        self.segmentations = segmentations
        self.laserdots = segmentations

    def trackPoints(self):
        point_positions = self._reconstruction_pipeline.track_points(self.video)
        self.image_widget.point_viewer.add_points(point_positions.detach().cpu().numpy())

    def buildCorrespondences(self):
        min_search_space = float(
            self.menu_widget.widget().getSubmenuValue("RHC", "Minimum Distance")
        )
        max_search_space = float(
            self.menu_widget.widget().getSubmenuValue("RHC", "Maximum Distance")
        )
        thresh = float(self.menu_widget.widget().getSubmenuValue("RHC", "GA Thresh"))
        set_size = int(self.menu_widget.widget().getSubmenuValue("RHC", "Consensus Size"))
        iterations = int(self.menu_widget.widget().getSubmenuValue("RHC", "Iterations"))

        self._reconstruction_pipeline.estimate_correspondences(min_search_space, max_search_space, set_size, iterations)

    def triangulate(self):
        min_search_space = float(
            self.menu_widget.widget().getSubmenuValue("RHC", "Minimum Distance")
        )
        max_search_space = float(
            self.menu_widget.widget().getSubmenuValue("RHC", "Maximum Distance")
        )
        self.points_3d = self._reconstruction_pipeline.triangulation(min_search_space, max_search_space)

    def denseShapeEstimation(self):
        zSubdivisions = int(
            self.menu_widget.widget().getSubmenuValue("Tensor Product M5", "Z Subdivisions")
        )
        # glottalmidline = self.segmentator.getGlottalMidline(self.images[self.frameOfClosedGlottis])

        if self.point_cloud_id is not None:
            self.viewer_widget.remove_mesh(self.point_cloud_id)

        (
            self.leftDeformed,
            self.rightDeformed,
            self.leftPoints,
            self.rightPoints,
            self.pointclouds,
        ) = SiliconeSurfaceReconstruction.controlPointBasedARAP(
            self.points_3d,
            self.camera,
            self._reconstruction_pipeline._feature_estimator,
            zSubdivisions=zSubdivisions,
        )

        self.point_cloud_offsets = [0]
        self.point_cloud_elements = [self.pointclouds[0].shape[0]]

        for pc in self.pointclouds[1:]:
            num_verts = pc.shape[0]
            self.point_cloud_offsets.append(self.point_cloud_offsets[-1] + num_verts)
            self.point_cloud_elements.append(num_verts)

        # super_point_cloud = np.concatenate(self.pointclouds, axis=0)
        # self.point_cloud_id = self.viewer_widget.display_point_cloud(super_point_cloud)
        self.viewer_widget.process_mesh_events()
        # self.point_cloud_mesh_core = self.viewer_widget.get_mesh(
        #    self.point_cloud_id
        # ).mesh_core

        self.addVocalfoldMeshes(self.leftDeformed, self.rightDeformed, zSubdivisions)
        self.setMeshes()

    def lsqOptimization(self):
        zSubdivisions = int(
            self.menu_widget.widget().getSubmenuValue("Tensor Product M5", "Z Subdivisions")
        )
        iterations = int(
            self.menu_widget.widget().getSubmenuValue("Least Squares Optimization", "Iterations")
        )
        lr = float(
            self.menu_widget.widget().getSubmenuValue(
                "Least Squares Optimization", "Learning Rate"
            )
        )
        window_size = int(
            self.menu_widget.widget().getSubmenuValue("Temporal Smoothing", "Window Size")
        )

        self.optimizedLeft = SiliconeSurfaceReconstruction.surfaceOptimization(
            self.leftDeformed,
            self.leftPoints,
            zSubdivisions=zSubdivisions,
            iterations=iterations,
            lr=lr,
        )
        self.optimizedLeft = np.array(self.optimizedLeft)
        self.smoothedLeft = scipy.ndimage.uniform_filter(
            self.optimizedLeft,
            size=(window_size, 1, 1),
            mode="reflect",
            cval=0.0,
            origin=0,
        )

        self.optimizedRight = SiliconeSurfaceReconstruction.surfaceOptimization(
            self.rightDeformed,
            self.rightPoints,
            zSubdivisions=zSubdivisions,
            iterations=iterations,
            lr=lr,
        )
        self.optimizedRight = np.array(self.optimizedRight)
        self.smoothedRight = scipy.ndimage.uniform_filter(
            self.optimizedRight,
            size=(window_size, 1, 1),
            mode="reflect",
            cval=0.0,
            origin=0,
        )

        self.addVocalfoldMeshes(self.smoothedLeft, self.smoothedRight, zSubdivisions)

    def saveModels(self):
        a = self.pts_left.reshape(self.pts_left.shape[0], -1, 3)
        b = self.pts_right.reshape(self.pts_right.shape[0], -1, 3)
        combined_vertices = np.concatenate([a, b], axis=1)

        faces_left = self.faces_left
        faces_right = self.faces_right + a.shape[1]
        combined_faces = np.concatenate([faces_left, faces_right], axis=0)
        dir_path = QFileDialog.getExistingDirectory(caption="Model Destination")

        for i in range(a.shape[0]):
            path = os.path.join(dir_path, "{0:05d}.obj".format(i))
            igl.write_obj(path, combined_vertices[i], combined_faces)

    def automaticReconstruction(self):
        self.segmentImages()
        self.trackPoints()
        self.buildCorrespondences()
        self.triangulate()
        self.denseShapeEstimation()
        self.lsqOptimization()
