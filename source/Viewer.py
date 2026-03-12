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
import matplotlib.pyplot as plt
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
import visualization
import VoronoiRHC
from GraphWidget import GraphWidget
from ImageViewerWidget import ImageViewerWidget
from MainMenuWidget import MainMenuWidget
from PyIGL_viewer.mesh import GlMeshCoreId, GlMeshInstanceId, GlMeshPrefabId
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
            "viewer_background":"#252526", # "#FFFFFF",
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

        self.plots_set = False
        self.meshes_set = False
        self.images_set = False

        self.point_cloud_mesh_core = None
        self.pointcloud_instance_id = None
        self.point_cloud_offsets = []
        self.point_cloud_elements = []

        self.pointclouds_go = None
        self.pointcloud_go_mesh_core = None
        self.pointcloud_go_instance_id = None
        self.pointcloud_go_offsets = []
        self.pointcloud_go_elements = []

        self.controlpoints = None
        self.controlpoints_mesh_core = None
        self.controlpoints_instance_id = None

        self.left_vf_instance = None
        self.right_vf_instance = None

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
        self.menu_widget.widget().button_dict["Compute Features"].clicked.connect(
            self.computeFeatures
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

        self.image_widget.opacity_widget.show_vf_checkbox.stateChanged.connect(self.toggleMeshVisibility)
        self.image_widget.opacity_widget.widget_features["GO"]["checkbox"].stateChanged.connect(self.toggleGoVisibility)
        self.image_widget.opacity_widget.show_controlpoints.stateChanged.connect(self.toggleControlpointVisibility)
        self.image_widget.opacity_widget.show_triangulated_points.stateChanged.connect(self.togglePointVisibility)

        self.menu_widget.widget().button_dict["Apply Cam"].clicked.connect(self.applyCameraParams)
        self.menu_widget.widget().button_dict["Top-Down Cam"].clicked.connect(self.setTopDownCamera)
        self.menu_widget.widget().button_dict["45Deg Cam"].clicked.connect(self.set45DegCamera)

        self.timer_thread.start()
        self.image_timer_thread.start()

        
        # (UN)COMMENT THIS TO LOAD HLE DATA AFTER STARTING
        
        path = "/media/nu94waro/Windows_C/save/datasets/HLEDataset/dataset"
        
        self.menu_widget.widget().ocs_widget.camera_calib_path = os.path.join(path, "camera_calibration.json")
        self.menu_widget.widget().ocs_widget.laser_calib_path = os.path.join(path, "laser_calibration.json")
        self.loadData(
            os.path.join(path, "camera_calibration.json"),
            os.path.join(path, "laser_calibration.json"),
            os.path.join(path, "MK/MK.avi"),
        )

        self._reconstruction_pipeline = reconstruction_pipeline.ReconstructionPipeline(
            self.camera, 
            self.laser, 
            feature_estimation.NeuralFeatureEstimator("cuda"), 
            point_tracking.InvivoPointTracker(), 
            correspondence_estimation.BruteForceEstimator(10, 30, 40, 100), 
            surface_reconstruction.SurfaceReconstructor())
        

        # (UN)COMMENT THIS TO LOAD SILICONE DATA AFTER STARTING

        # path = "assets"
        
        # self.menu_widget.widget().ocs_widget.camera_calib_path = os.path.join(path, "camera_calibration.json")
        # self.menu_widget.widget().ocs_widget.laser_calib_path = os.path.join(path, "laser_calibration.json")
        # self.loadData(
        #     "assets/camera_calibration.json",
        #     "assets/laser_calibration.json",
        #     "assets/example_vid.avi",
        # )

        # self._reconstruction_pipeline = reconstruction_pipeline.ReconstructionPipeline(
        #     self.camera, 
        #     self.laser, 
        #     feature_estimation.SiliconeFeatureEstimator(), 
        #     point_tracking.SiliconePointTracker(), 
        #     correspondence_estimation.BruteForceEstimator(10, 30, 40, 100), 
        #     surface_reconstruction.SurfaceReconstructor())
            


        '''
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

        #self.segmentations = []
        #for feature_image in feature_images:
        #    self.segmentations.append(feature_image.permute(1, 2, 0).detach().cpu().numpy().astype(np.uint8))


        self.graph_widget.updateGraph(
            self._reconstruction_pipeline._feature_estimator.glottalAreaWaveform().tolist(), self.graph_widget.glottal_seg_graph
        )



        gs_qimages = [helper.np_2_QImage((visualization.segmentation_to_color_mask(gs, (255, 0, 0), 255)).detach().cpu().numpy().astype(np.uint8)) for gs in glottis_segmentations]
        vf_qimages = [helper.np_2_QImage((visualization.segmentation_to_color_mask(vf, (0, 255, 0), 255)).detach().cpu().numpy().astype(np.uint8)) for vf in vocalfold_segmentations]
        self.image_widget.point_viewer.add_points(point_positions.detach().cpu().numpy())
        self.image_widget.point_viewer.add_vocalfold_segmentations(vf_qimages)
        self.image_widget.point_viewer.add_glottal_segmentations(gs_qimages)
        self.image_widget.point_viewer.add_glottal_midlines(glottal_midlines)
        self.image_widget.point_viewer.add_glottal_outlines(glottal_outlines)
        '''
        


        
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
            self.menu_widget.widget().getSubmenuValue("Correspondence Estimation", "Minimum Distance")
        )
        max_search_space = float(
            self.menu_widget.widget().getSubmenuValue("Correspondence Estimation", "Maximum Distance")
        )
        thresh = float(self.menu_widget.widget().getSubmenuValue("Correspondence Estimation", "GA Thresh"))
        set_size = int(self.menu_widget.widget().getSubmenuValue("Correspondence Estimation", "Consensus Size"))
        iterations = int(self.menu_widget.widget().getSubmenuValue("Correspondence Estimation", "Iterations"))
        

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

            # Needs refactoring.
            self.image_widget.updateImages(
                None,
                None,
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
                self.left_vf_instance,
                self.pts_left[frameNum].reshape((-1, 3)).astype(np.float32),
            )
            self.viewer_widget.update_mesh_vertices(
                self.right_vf_instance,
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

        self.pointcloud_go_mesh_core.offset = self.pointcloud_go_offsets[
            self.player_widget.slider.value()
        ]
        self.pointcloud_go_mesh_core.number_elements = self.pointcloud_go_elements[
            self.player_widget.slider.value()
        ]

        self.viewer_widget.update_mesh_vertices(
            self.controlpoints_instance_id,
            self.controlpoints[frameNum].reshape((-1, 3)).astype(np.float32),
        )

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

        if self.right_vf_instance is not None:
            self.viewer_widget.remove_mesh_(self.right_vf_instance)

        if self.left_vf_instance is not None:
            self.viewer_widget.remove_mesh_(self.left_vf_instance)

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
        uniforms["albedo"] = np.array([0.8, 0.8, 0.8])

        '''
        vertex_normals_left = igl.per_vertex_normals(
            vertices_left, faces, igl.PER_VERTEX_NORMALS_WEIGHTING_TYPE_AREA
        ).astype(np.float32)
        vertex_attributes_left["normal"] = vertex_normals_left

        vertex_normals_right = igl.per_vertex_normals(
            vertices_right, faces, igl.PER_VERTEX_NORMALS_WEIGHTING_TYPE_AREA
        ).astype(np.float32)
        vertex_attributes_right["normal"] = vertex_normals_right
        '''



        mesh_index = self.viewer_widget.add_mesh(vertices_left, faces_left)
        mesh_prefab_index = self.viewer_widget.add_mesh_prefab(mesh_index, "colormap", uniforms=uniforms)
        self.left_vf_instance = self.viewer_widget.add_mesh_instance(
            mesh_prefab_index, np.eye(4, dtype="f")
        )

        mesh_index = self.viewer_widget.add_mesh(vertices_right, faces_right)
        mesh_prefab_index = self.viewer_widget.add_mesh_prefab(mesh_index, "colormap", uniforms=uniforms)
        self.right_vf_instance = self.viewer_widget.add_mesh_instance(
            mesh_prefab_index, np.eye(4, dtype="f")
        )

        


        '''
        left_vf_instance = self.viewer_widget.display_mesh_new(self.pts_left, self.faces_left, vertex_normals_left, vertex_attributes=vertex_attributes_left, uniforms=uniforms)
        right_vf_instance = self.viewer_widget.display_mesh_new(self.pts_right, self.faces_right, vertex_normals_right, vertex_attributes=vertex_attributes_right, uniforms=uniforms)
        
        left_vf_wf_instance = self.viewer_widget.display_wireframe_(self.pts_left, self.faces_left, np.array([0.1, 0.1, 0.1]))
        right_vf_wf_instance = self.viewer_widget.display_wireframe_(self.pts_right, self.faces_right, np.array([0.1, 0.1, 0.1]))
        '''

        a = 1
        '''
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

        self.viewer_widget.set_mesh_instance_visibility_(instance_index_right, False)
        '''

    def togglePointVisibility(self):
        visibility = self.viewer_widget.get_mesh_instance_visibility(self.pointcloud_instance_id)
        self.viewer_widget.set_mesh_instance_visibility(self.pointcloud_instance_id, not visibility)
        self.viewer_widget.set_mesh_instance_visibility(self.pointcloud_instance_id, not visibility)

    def toggleControlpointVisibility(self):
        visibility = self.viewer_widget.get_mesh_instance_visibility(self.controlpoints_instance_id)
        self.viewer_widget.set_mesh_instance_visibility(self.controlpoints_instance_id, not visibility)
        self.viewer_widget.set_mesh_instance_visibility(self.controlpoints_instance_id, not visibility)

    def toggleGoVisibility(self):
        visibility = self.viewer_widget.get_mesh_instance_visibility(self.pointcloud_go_instance_id)
        self.viewer_widget.set_mesh_instance_visibility(self.pointcloud_go_instance_id, not visibility)
        self.viewer_widget.set_mesh_instance_visibility(self.pointcloud_go_instance_id, not visibility)

    def toggleMeshVisibility(self):
        visibility = self.viewer_widget.get_mesh_instance_visibility(self.left_vf_instance)
        self.viewer_widget.set_mesh_instance_visibility(self.right_vf_instance, not visibility)
        self.viewer_widget.set_mesh_instance_visibility(self.left_vf_instance, not visibility)

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

        self.image_widget.point_viewer.add_video(helper.vid_2_QImage(self.images))

        self.player_widget.setSliderRange(0, len(self.images))
        self.player_widget.stop_video_()

        self.images_set = True
        self.video = torch.from_numpy(np.stack(self.images)).to("cuda")

        self.update_images_func()


    def applyCameraParams(self):
        self.viewer_widget.camera.field_of_view = float(
            self.menu_widget.widget().getSubmenuValue("Camera", "FOV")
        )
        self.viewer_widget.camera.near_plane = float(
            self.menu_widget.widget().getSubmenuValue("Camera", "Near Plane")
        )
        self.viewer_widget.camera.far_plane = float(
            self.menu_widget.widget().getSubmenuValue("Camera", "Far Plane")
        )

        self.viewer_widget.camera.set_position_and_direction(np.array([float(
            self.menu_widget.widget().getSubmenuValue("Camera", "X-Pos")
        ), float(
            self.menu_widget.widget().getSubmenuValue("Camera", "Y-Pos")
        ), float(
            self.menu_widget.widget().getSubmenuValue("Camera", "Z-Pos")
        )]), np.array([float(
            self.menu_widget.widget().getSubmenuValue("Camera", "X-Dir")
        ), float(
            self.menu_widget.widget().getSubmenuValue("Camera", "Y-Dir")
        ), float(
            self.menu_widget.widget().getSubmenuValue("Camera", "Z-Dir")
        )]))
    
    def setTopDownCamera(self):
        self.viewer_widget.camera.set_position_and_direction(np.array([0.0, 8.0, 0.5]), np.array([0.0, -0.999, 0.0001]))
    
    
    def set45DegCamera(self):
        self.viewer_widget.camera.set_position_and_direction(np.array([0.0, 2.0, -8.5]), np.array([0.0, -0.25, 1.0]))
        #self.viewer_widget.camera.current_eye = self.viewer_widget.camera.eye
        #self.viewer_widget.camera.current_target = self.viewer_widget.camera.target


    def computeFeatures(self):
        feature_estimator: feature_estimation.FeatureEstimator = None
        if self.menu_widget.widget().getSubmenuValue("Segmentation", "Neural Segmentation"):
            feature_estimator = feature_estimation.NeuralFeatureEstimator("bla")
        elif self.menu_widget.widget().getSubmenuValue("Segmentation", "Silicone Segmentation"):
            feature_estimator = feature_estimation.SiliconeFeatureEstimator()
        else:
            print("Please choose a Segmentation Algorithm")

        use_cuda = self.menu_widget.widget().getSubmenuValue("CUDA", "Use")

        self._reconstruction_pipeline.set_feature_estimator(feature_estimator)
        self._reconstruction_pipeline.estimate_features(self.video if use_cuda else self.video.cpu())

        self.graph_widget.updateGraph(
            feature_estimator.glottalAreaWaveform().tolist(), self.graph_widget.glottal_seg_graph
        )

        gs_qimages = [helper.np_2_QImage((visualization.segmentation_to_color_mask(gs, (255, 0, 0), 255)).detach().cpu().numpy().astype(np.uint8)) for gs in feature_estimator.glottisSegmentations()]
        vf_qimages = [helper.np_2_QImage((visualization.segmentation_to_color_mask(vf, (0, 255, 0), 255)).detach().cpu().numpy().astype(np.uint8)) for vf in feature_estimator.vocalfoldSegmentations()]
        np_points = [point.detach().cpu().numpy() for point in feature_estimator.laserpointPositions()]
        self.image_widget.point_viewer.add_points(np_points)
        self.image_widget.point_viewer.add_vocalfold_segmentations(vf_qimages)
        self.image_widget.point_viewer.add_glottal_segmentations(gs_qimages)
        self.image_widget.point_viewer.add_glottal_midlines(feature_estimator.glottalMidlines())
        self.image_widget.point_viewer.add_glottal_outlines(feature_estimator.glottalOutlines())

        
        self.update_images_func()
        #self.segmentations = segmentations
        #self.laserdots = segmentations

    def trackPoints(self):
        use_cuda = self.menu_widget.widget().getSubmenuValue("CUDA", "Use")

        point_tracker: point_tracking.PointTrackerBase = None
        if self.menu_widget.widget().getSubmenuValue("Point Tracking", "InvivoSlow"):
            point_tracker = point_tracking.InvivoPointTracker()
        if self.menu_widget.widget().getSubmenuValue("Point Tracking", "InvivoFast"):
            point_tracker = point_tracking.InvivoPointTrackerNewNew()
        elif self.menu_widget.widget().getSubmenuValue("Point Tracking", "Silicone"):
            point_tracker = point_tracking.SiliconePointTracker()
        else:
            print("Please choose a Segmentation Algorithm")

        self._reconstruction_pipeline.set_point_tracker(point_tracker)
        point_positions = self._reconstruction_pipeline.track_points(self.video if use_cuda else self.video.cpu())
        self.image_widget.point_viewer.add_optimized_points(point_positions.detach().cpu().numpy())
        
        self.update_images_func()

    def buildCorrespondences(self):
        min_search_space = float(
            self.menu_widget.widget().getSubmenuValue("Correspondence Estimation", "Minimum Distance")
        )
        max_search_space = float(
            self.menu_widget.widget().getSubmenuValue("Correspondence Estimation", "Maximum Distance")
        )
        thresh = float(self.menu_widget.widget().getSubmenuValue("Correspondence Estimation", "GA Thresh"))
        set_size = int(self.menu_widget.widget().getSubmenuValue("Correspondence Estimation", "Consensus Size"))
        iterations = int(self.menu_widget.widget().getSubmenuValue("Correspondence Estimation", "Iterations"))

        correspondence_estimator = None
        if self.menu_widget.widget().getSubmenuValue("Correspondence Estimation", "RHC"):
            correspondence_estimator = correspondence_estimation.RHCEstimator(set_size, iterations, min_search_space, max_search_space)
        elif self.menu_widget.widget().getSubmenuValue("Correspondence Estimation", "BF_RANSAC"):
            correspondence_estimator = correspondence_estimation.BruteForceEstimator(set_size, iterations, min_search_space, max_search_space)
        else:
            print("No suitable correspondence estimator selected.")
            return
        
        self._reconstruction_pipeline.set_correspondence_matcher(correspondence_estimator)

        import time
        start_time = time.time()
        self._reconstruction_pipeline.estimate_correspondences(min_search_space, max_search_space, set_size, iterations)
        end_time = time.time()
        print(f"Correspondence Estimation took: {(end_time-start_time):.3f}")

    def triangulate(self):
        min_search_space = float(
            self.menu_widget.widget().getSubmenuValue("Correspondence Estimation", "Minimum Distance")
        )
        max_search_space = float(
            self.menu_widget.widget().getSubmenuValue("Correspondence Estimation", "Maximum Distance")
        )
        self.points_3d = self._reconstruction_pipeline.triangulation(min_search_space, max_search_space)

    def denseShapeEstimation(self):
        zSubdivisions = int(
            self.menu_widget.widget().getSubmenuValue("Tensor Product M5", "Z Subdivisions")
        )
        xSubdivisions = int(
            self.menu_widget.widget().getSubmenuValue("Tensor Product M5", "X Subdivisions")
        )

        r_zero = float(self.menu_widget.widget().getSubmenuValue("Tensor Product M5", "R_0"))
        T = float(self.menu_widget.widget().getSubmenuValue("Tensor Product M5", "T"))
        psi = float(self.menu_widget.widget().getSubmenuValue("Tensor Product M5", "psi"))

        ARAP_iterations = int(self.menu_widget.widget().getSubmenuValue("As-Rigid-As-Possible", "Iterations"))
        ARAP_weight = float(self.menu_widget.widget().getSubmenuValue("As-Rigid-As-Possible", "Weight"))
        # glottalmidline = self.segmentator.getGlottalMidline(self.images[self.frameOfClosedGlottis])

        if self.pointcloud_instance_id is not None:
            self.viewer_widget.remove_mesh_(self.pointcloud_instance_id)
            self.pointcloud_instance_id = None
            self.point_cloud_mesh_core = None

        if self.pointcloud_go_instance_id is not None:
            self.viewer_widget.remove_mesh_(self.pointcloud_go_instance_id)
            self.pointcloud_go_instance_id = None
            self.pointcloud_go_mesh_core = None

        if self.controlpoints_instance_id is not None:
            self.viewer_widget.remove_mesh_(self.controlpoints_instance_id)
            self.controlpoints_instance_id = None
            self.controlpoints_mesh_core = None

        (
            self.leftDeformed,
            self.rightDeformed,
            self.leftPoints,
            self.rightPoints,
            self.pointclouds,
            self.pointclouds_go
        ) = SiliconeSurfaceReconstruction.controlPointBasedARAP(
            self.points_3d,
            self.camera,
            self._reconstruction_pipeline._feature_estimator,
            zSubdivisions=zSubdivisions,
            xSubdivisions=xSubdivisions,
            r_zero=r_zero,
            T=T,
            psi=psi,
            ARAP_iterations=ARAP_iterations,
            ARAP_weight=ARAP_weight,
            flip_y=False if self.menu_widget.widget().getSubmenuValue("Point Tracking", "Silicone") else True
        )

        self.point_cloud_offsets = [0]
        self.point_cloud_elements = [self.pointclouds[0].shape[0]]

        for pc in self.pointclouds[1:]:
            num_verts = pc.shape[0]
            self.point_cloud_offsets.append(self.point_cloud_offsets[-1] + num_verts)
            self.point_cloud_elements.append(num_verts)

        super_point_cloud = np.concatenate(self.pointclouds, axis=0)



        vertex_attributes = {}
        if not "vertexColor" in vertex_attributes:
            vertex_attributes["vertexColor"] = np.tile(
                np.array([0.0, 0.0, 1.0], dtype=np.float32), (super_point_cloud.shape[0], 1)
            )
        faces = np.linspace(0, super_point_cloud.shape[0], num=super_point_cloud.shape[0], endpoint=False, dtype=np.int32)[:, np.newaxis]
        core_id = GlMeshCoreId()
        self.viewer_widget.add_mesh_(core_id, super_point_cloud, faces)
        prefab_id = GlMeshPrefabId(core_id)
        self.viewer_widget.add_mesh_prefab_(
            prefab_id,
            shader="per_vertex_color",
            vertex_attributes=vertex_attributes,
            face_attributes={},
            uniforms={},
        )
        self.pointcloud_instance_id = GlMeshInstanceId(prefab_id)
        self.viewer_widget.add_mesh_instance_(self.pointcloud_instance_id, np.eye(4))
        self.point_cloud_mesh_core = self.viewer_widget.get_mesh(self.pointcloud_instance_id).mesh_core





        self.pointcloud_go_offsets = [0]
        self.pointcloud_go_elements = [self.pointclouds_go[0].shape[0]]

        for pc in self.pointclouds_go[1:]:
            num_verts = pc.shape[0]
            self.pointcloud_go_offsets.append(self.pointcloud_go_offsets[-1] + num_verts)
            self.pointcloud_go_elements.append(num_verts)

        super_pointcloud_go = np.concatenate(self.pointclouds_go, axis=0)



        vertex_attributes = {}
        if not "vertexColor" in vertex_attributes:
            vertex_attributes["vertexColor"] = np.tile(
                np.array([0.0, 1.0, 0.0], dtype=np.float32), (super_pointcloud_go.shape[0], 1)
            )
        faces = np.linspace(0, super_pointcloud_go.shape[0], num=super_pointcloud_go.shape[0], endpoint=False, dtype=np.int32)[:, np.newaxis]
        core_id = GlMeshCoreId()
        self.viewer_widget.add_mesh_(core_id, super_pointcloud_go, faces)
        prefab_id = GlMeshPrefabId(core_id)
        self.viewer_widget.add_mesh_prefab_(
            prefab_id,
            shader="per_vertex_color",
            vertex_attributes=vertex_attributes,
            face_attributes={},
            uniforms={},
        )
        self.pointcloud_go_instance_id = GlMeshInstanceId(prefab_id)
        self.viewer_widget.add_mesh_instance_(self.pointcloud_go_instance_id, np.eye(4))
        self.pointcloud_go_mesh_core = self.viewer_widget.get_mesh(self.pointcloud_go_instance_id).mesh_core

        self.controlpoints = np.concatenate([np.array(self.leftDeformed), np.array(self.rightDeformed)], axis=1)
       
        faces = []
        res_u = zSubdivisions
        res_v = len(self.leftDeformed[0]) // res_u
        half_verts = len(self.leftDeformed[0])
        num_verts = 2 * half_verts
        for i in range(res_u - 1):
            for j in range(res_v - 1):
                p0 = i * res_v + j
                p1 = p0 + 1
                p2 = p0 + res_v + 1
                p3 = p0 + res_v
                faces.append([p0, p1, p2, p3])  # a quad face
                faces.append([half_verts + p0, half_verts + p1, half_verts + p2, half_verts + p3])
        faces = np.array(faces)

        core_id = GlMeshCoreId()
        self.viewer_widget.add_mesh_(core_id, self.controlpoints[self.player_widget.getCurrentFrame()], faces)
        prefab_id = GlMeshPrefabId(core_id)
        self.viewer_widget.add_mesh_prefab_(
            prefab_id,
            shader="wireframe",
            vertex_attributes={},
            face_attributes={},
            uniforms={"lineColor": np.array([0.0, 0.0, 1.0])},
        )
        self.controlpoints_instance_id = GlMeshInstanceId(prefab_id)
        self.viewer_widget.add_mesh_instance_(self.controlpoints_instance_id, np.eye(4))
        self.controlpoints_mesh_core = self.viewer_widget.get_mesh(self.controlpoints_instance_id).mesh_core




        '''
        self.point_cloud_id = self.viewer_widget.display_point_cloud(super_point_cloud)
        self.viewer_widget.process_mesh_events()
        self.point_cloud_mesh_core = self.viewer_widget.get_mesh(
           self.point_cloud_id
        ).mesh_core
        '''
        
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
        self.computeFeatures()
        self.trackPoints()
        self.buildCorrespondences()
        self.triangulate()
        self.denseShapeEstimation()
        self.lsqOptimization()
