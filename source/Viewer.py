import sys
import copy
import time
import threading

import sys
sys.path.append(".")
sys.path.append("source/")

import Mesh
import igl
import scipy

from PyIGL_viewer.viewer.viewer_widget import ViewerWidget
from PyIGL_viewer.viewer.ui_widgets import PropertyWidget, LegendWidget
from pyqtgraph.widgets.RawImageWidget import RawImageGLWidget
from PyIGL_viewer.mesh.mesh import GlMeshPrefab, GlMeshPrefabId
from random import randint

import cv2

from PyQt5.QtCore import Qt, pyqtSignal, QTimer, QThread
from PyQt5.QtGui import QImage, QPixmap, QFont
from pyqtgraph import PlotWidget, plot
import pyqtgraph as pg
import numpy as np
from PyQt5.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QFrame,
    QVBoxLayout,
    QHBoxLayout,
    QSlider,
    QGridLayout,
    QPushButton,
    QLabel,
    QMenuBar,
    QMenu,
    QFormLayout,
    QLineEdit,
    QCheckBox,
    QFileDialog
)

import Camera
import Laser
import helper
import SiliconeSegmentator
import RHC
import VoronoiRHC
import Correspondences
import Triangulation
import SiliconeSurfaceReconstruction
from VocalfoldHSVSegmentation import vocalfold_segmentation

pg.setConfigOption('imageAxisOrder', 'row-major')


class QHLine(QFrame):
    def __init__(self):
        super(QHLine, self).__init__()
        self.setFrameShape(QFrame.HLine)
        self.setFrameShadow(QFrame.Sunken)
        self.setStyleSheet("background-color: #999999;")

class QVLine(QFrame):
    def __init__(self):
        super(QVLine, self).__init__()
        self.setFrameShape(QFrame.VLine)
        self.setFrameShadow(QFrame.Sunken)
        self.setStyleSheet("background-color: #999999;")


class VideoPlayerWidget(QWidget):
    def __init__(self, parent=None):
        super(VideoPlayerWidget, self).__init__()
        layout = QVBoxLayout(self)

        self.slider = QSlider(Qt.Horizontal, self)
        self.slider.setMinimum(0)
        self.slider.setRange(0, 100)
        self.slider.setValue(0)
        self.slider.setGeometry(0, 0, 1000, 1000)
        layout.addWidget(self.slider)
        self.setLayout(layout)

        widget_button = QWidget(self)
        layout_button = QHBoxLayout()
        
        self.bPlay = QPushButton("Play")
        self.bPause = QPushButton("Pause")
        self.bStop = QPushButton("Stop")
        self.bReplay = QPushButton("Replay")
        self.bPrevious = QPushButton("Previous Frame")
        self.bNext = QPushButton("Next Frame")

        self.play = False
        
        self.bPlay.clicked.connect(self.play_video_)
        self.bPause.clicked.connect(self.pause_video_)
        self.bStop.clicked.connect(self.stop_video_)
        self.bReplay.clicked.connect(self.replay_video_)
        self.bPrevious.clicked.connect(self.prev_frame_)
        self.bNext.clicked.connect(self.next_frame_)
        
        layout_button.addWidget(self.bPlay)
        layout_button.addWidget(self.bPause)
        layout_button.addWidget(self.bStop)
        layout_button.addWidget(self.bReplay)
        layout_button.addWidget(self.bPrevious)
        layout_button.addWidget(self.bNext)

        widget_button.setLayout(layout_button)
        layout.addWidget(widget_button)

    def play_video_(self):
        self.play = True

    def pause_video_(self):
        self.play = False

    def stop_video_(self):
        self.slider.setValue(0)
        self.pause_video_()

    def replay_video_(self):
        self.slider.setValue(0)
        self.play_video_()

    def update_frame_when_playing(self):
        if self.isPlaying():
            self.next_frame_()

    def next_frame_(self):
        self.slider.setValue(self.slider.value() + 1 % (self.slider.maximum() - 1))

    def prev_frame_(self):
        self.slider.setValue(self.slider.value() - 1 if self.slider.minimum() < self.slider.value() - 1 else self.slider.value())

    def setSliderPosition(self, pos):
        self.slider.setValue(pos)

    def setSliderRange(self, min, max):
        self.slider.setRange(min, max)

    def isPlaying(self):
        return self.play

    def isPaused(self):
        return not self.play

    def getCurrentFrame(self):
        return self.slider.value()

class OpenCloseSaveWidget(QWidget):
    fileOpenedSignal = pyqtSignal(str, str, str)

    def __init__(self, parent=None):
        super(OpenCloseSaveWidget, self).__init__()
        self.setLayout(QHBoxLayout())

        self.addButton("Open", self.open)
        self.addButton("New Video", self.openVideo)
        self.addButton("Load", self.open)
        self.addButton("Save", self.open)

    def addButton(self, title, function):
        button = QPushButton(title)
        button.clicked.connect(function)
        self.layout().addWidget(button)

    def open(self):
        self.camera_calib_path, _ = QFileDialog.getOpenFileName(self, 'Open Camera Calibration file', '', "Camera Calibration Files (*.json *.mat)")
        self.laser_calib_path, _ = QFileDialog.getOpenFileName(self, 'Open Laser Calibration file', '', "Laser Calibration Files (*.json *.mat)")
        self.video_path, _ = QFileDialog.getOpenFileName(self, 'Open Video', '', "Video Files (*.avi *.mp4 *.mkv *.AVI *.MP4)")
        self.fileOpenedSignal.emit(self.camera_calib_path, self.laser_calib_path, self.video_path)

    def openVideo(self):
        self.video_path, _ = QFileDialog.getOpenFileName(self, 'Open Video', '', "Video Files (*.avi *.mp4 *.mkv *.AVI *.MP4)")
        self.fileOpenedSignal.emit(self.camera_calib_path, self.laser_calib_path, self.video_path)

    def loadProject(self):
        project_path = QFileDialog.getOpenFileName(self, 'Open Camera Calibration file', '', "Project Folders (*.json *.mat)")

        # TODO: implement
        # Load Stuff

    def saveProject(self):
        # Save a project
        print("Save a project! And implement me!")
        pass


class MainMenuWidget(QWidget):
    def __init__(self, viewer_palette, parent=None):
        super(MainMenuWidget, self).__init__()
        #self.setStyle(QFrame.Panel | QFrame.Raised)
        self.setStyleSheet(f"background-color: {viewer_palette['menu_background']}")
        self.base_layout = QVBoxLayout()
        self.base_layout.setAlignment(Qt.AlignTop)
        self.setLayout(self.base_layout)

        self.ocs_widget = OpenCloseSaveWidget(self)
        self.base_layout.addWidget(self.ocs_widget)

        self.submenu_dict = {}
        self.button_dict = {}

        self.addSubMenu("Tensor Product M5", [("Z Subdivisions", "field", 8), ("X Subdivisions", "field", 3)])
        self.addSubMenu("Segmentation", [("Koc et al", "checkbox", True), ("Neural Segmentation", "checkbox", False), ("Silicone Segmentation", "checkbox", False)])
        self.addSubMenu("RHC", [("Activated", "checkbox", True), ("Iterations", "field", 30), ("Consensus Size", "field", 8), ("GA Thresh", "field", 5.0), ("Minimum Distance", "field", 40.0), ("Maximum Distance", "field", 80.0)])
        self.addSubMenu("Voronoi RHC", [("Activated", "checkbox", False), ("Threshold", "field", 5)])
        self.addSubMenu("As-Rigid-As-Possible", [("Iterations", "field", 2), ("Weight", "field", 10000)])
        self.addSubMenu("Least Squares Optimization", [("Iterations", "field", 10), ("Learning Rate", "field", 0.1)])
        self.addSubMenu("Temporal Smoothing", [("Window Size", "field", 7)])

        self.base_layout.addWidget(QHLine())
        self.addButton("Segment Images")
        self.addButton("Build Correspondences")
        self.addButton("Triangulate")
        self.addButton("Dense Shape Estimation")
        self.addButton("Least Squares Optimization")
        self.base_layout.addWidget(QHLine())
        self.addButton("Automatic Reconstruction")

    def addSubMenu(self, title, listOfTriplets):
        submenu_widget = SubMenuWidget(title, listOfTriplets, self)
        self.base_layout.addWidget(submenu_widget)
        self.submenu_dict[title] = submenu_widget.get_dict()

    def addButton(self, label):
        button = QPushButton(label)
        self.base_layout.addWidget(button)
        self.button_dict[label] = button

    def getSubmenuValue(self, submenu, key):
        field = self.submenu_dict[submenu][key]
        return field.text() if type(field) is QLineEdit else field.isChecked()


class SubMenuWidget(QWidget):
    def __init__(self, title, listOfTriplets, parent=None):
        super(SubMenuWidget, self).__init__()
        base_layout = QVBoxLayout(self)

        title = QLabel(title)
        title_font = title.font()
        title_font.setBold(True)
        title.setFont(title_font)

        base_layout.addWidget(title)
        
        base_layout.addWidget(QHLine())
        
        self.DICT = {}

        self.subform = QWidget(self)
        self.subform_layout = QFormLayout()
        self.subform.setLayout(self.subform_layout)

        for key, button_type, default_value in listOfTriplets:
            if button_type == "checkbox":
                check = QCheckBox(self.subform)
                check.setChecked(default_value)
                self.subform_layout.addRow(QLabel(key, self.subform), check)
                self.DICT[key] = check
            elif button_type == "field":
                lineedit = QLineEdit(str(default_value), self.subform)
                self.subform_layout.addRow(QLabel(key, self.subform), lineedit)
                self.DICT[key] = lineedit

        base_layout.addWidget(self.subform)

    def get_dict(self):
        return self.DICT


class GraphWidget(QWidget):
    def __init__(self, parent=None):
        super(GraphWidget, self).__init__()
        layout = QVBoxLayout(self)

        self.height_graph_left = pg.PlotWidget()
        self.current_frame_line_height_left = pg.InfiniteLine(pos=0)
        self.height_graph_left.addItem(self.current_frame_line_height_left)

        self.height_graph_right = pg.PlotWidget()
        self.current_frame_line_height_right = pg.InfiniteLine(pos=0)
        self.height_graph_right.addItem(self.current_frame_line_height_right)
        
        self.glottal_seg_graph = pg.PlotWidget()
        self.current_frame_line_glottal_seg = pg.InfiniteLine(pos=0)
        self.glottal_seg_graph.addItem(self.current_frame_line_glottal_seg)
        
        layout.addWidget(self.height_graph_left)
        layout.addWidget(self.height_graph_right)
        layout.addWidget(self.glottal_seg_graph)

    def updateLines(self, val):
        self.current_frame_line_glottal_seg.setPos(val)
        self.current_frame_line_height_right.setPos(val)
        self.current_frame_line_height_left.setPos(val)

    def updateGraph(self, vals, graph):
        x = np.arange(0, len(vals))
        pen = pg.mkPen(color=(255, 125, 15))
        graph.plot(x, vals, pen=pen)

    def updateGraphs(self, a, b, c):
        self.updateGraph(a, self.height_graph_right)
        self.updateGraph(b, self.height_graph_left)
        self.updateGraph(c, self.glottal_seg_graph)


class ImageViewerWidget(QWidget):
    def __init__(self, parent=None):
        super(ImageViewerWidget, self).__init__()
        self.base_layout = QHBoxLayout(self)
        
        self.imageDICT = {}

        self.addImageWidget("Main", (256, 512))
        self.base_layout.addWidget(QVLine())
        self.addImageWidget("Segmentation", (256, 512))
        self.base_layout.addWidget(QVLine())
        self.addImageWidget("Laserdots", (256, 512))
        self.base_layout.addWidget(QVLine())
        self.addImageWidget("Closed Vocal Folds", (256, 512))

    def addImageWidget(self, title, size):
        widg = QWidget(self)
        lay = QVBoxLayout(widg)

        title_widget = QLabel(title)
        title_font = title_widget.font()
        title_font.setBold(True)
        title_widget.setFont(title_font)

        image_widg = QLabel(title)
        image_widg.setFixedSize(size[0], size[1])
        self.imageDICT[title] = image_widg

        lay.addWidget(title_widget, alignment=Qt.AlignmentFlag.AlignCenter)
        lay.addWidget(image_widg)
        self.base_layout.addWidget(widg)

    def updateImages(self, a, b, c):
        # We assume images to be in RGB Format
        self.updateImage(a, self.imageDICT["Main"])
        self.updateImage(b, self.imageDICT["Segmentation"])
        self.updateImage(c, self.imageDICT["Laserdots"])

    def convertImage(self, image):
        # Check if Mono
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

        if image.shape[0] < image.shape[1]:
            image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)

        h, w, ch = image.shape
        bytesPerLine = ch * w
        return QImage(image.data, w, h, bytesPerLine, QImage.Format_BGR888)
        
    def updateImage(self, image, widget):
        widget.setPixmap(QPixmap.fromImage(self.convertImage(image)))

    def getWidget(self, key):
        return self.imageDICT[key]


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
            self.segmentator = vocalfold_segmentation.HSVGlottisSegmentator(self.images[:50])
        elif self.menu_widget.getSubmenuValue("Segmentation", "Neural Segmentation"):
            self.segmentator = None
            # TODO: Implement
        elif self.menu_widget.getSubmenuValue("Segmentation", "Silicone Segmentation"):
                self.segmentator = SiliconeSegmentator.SiliconeVocalfoldSegmentator(self.images[:50])
        else:
            print("Please choose a Segmentation Algorithm")
        
        self.segmentator.generate()

        x, w, y, h = self.segmentator.getROI()

        # Use ROI to generate mask image
        self.roi = np.zeros((self.images[0].shape[0], self.images[0].shape[1]), dtype=np.uint8)
        self.roi[y:y+h, x:x+w] = 255

        self.frameOfClosedGlottis = self.segmentator.estimateClosedGlottis()
        vocalfold_image = self.images[self.frameOfClosedGlottis]

        self.maxima = helper.findMaxima(vocalfold_image, self.roi)

        segmentations = list()
        laserdots = list()

        for image in self.images:
            base_image = image

            segmentation_image = base_image.copy()
            segmentation_image = self.segmentator.segment_image(segmentation_image)
            gml_a, gml_b = self.segmentator.getGlottalMidline(segmentation_image)

            if self.menu_widget.getSubmenuValue("Segmentation", "Koc et al"):
                segmentation_image = self.segmentator.gen_segmentation_image(segmentation_image)

            segmentation_image = cv2.cvtColor(segmentation_image, cv2.COLOR_GRAY2BGR)

            cv2.rectangle(segmentation_image, (x, y), (x+w,y+h), color=(255, 0, 0), thickness=2)
            try:
                cv2.line(segmentation_image, gml_a.astype(np.int32), gml_b.astype(np.int32), color=(125, 125, 0), thickness=2)
            except:
                pass
            segmentations.append(cv2.cvtColor(image, cv2.COLOR_GRAY2BGR) | segmentation_image)

            laserdot_image = helper.findMaxima(image, self.roi)
            laserdot_image = cv2.dilate(laserdot_image, np.ones((3,3)))
            laserdot_image = np.where(laserdot_image > 0, 255, 0).astype(np.uint8)
            laserdot_image = cv2.cvtColor(laserdot_image, cv2.COLOR_GRAY2BGR)
            laserdot_image[:, :, [0, 2]] = 0
            laserdots.append(cv2.cvtColor(image, cv2.COLOR_GRAY2BGR) | laserdot_image)
        
        self.segmentations = segmentations
        self.laserdots = laserdots

    def buildCorrespondences(self):
        min_search_space = float(self.menu_widget.getSubmenuValue("RHC", "Minimum Distance"))
        max_search_space = float(self.menu_widget.getSubmenuValue("RHC", "Maximum Distance"))
        thresh = float(self.menu_widget.getSubmenuValue("RHC", "GA Thresh"))
        set_size = int(self.menu_widget.getSubmenuValue("RHC", "Consensus Size"))
        iterations = int(self.menu_widget.getSubmenuValue("RHC", "Iterations"))

        if self.menu_widget.getSubmenuValue("RHC", "Activated"):
            pixelLocations, laserGridIDs = Correspondences.initialize(self.laser, self.camera, self.maxima, self.images[self.frameOfClosedGlottis], min_search_space, max_search_space)
            self.grid2DPixLocations = RHC.RHC(laserGridIDs, pixelLocations, self.maxima, self.camera, self.laser, set_size, iterations)
        elif self.menu_widget.getSUbmenuValue("Voronoi RHC", "Activated"):
            cf = VoronoiRHC.CorrespondenceFinder(self.camera, self.laser, minWorkingDistance=min_search_space, maxWorkingDistance=max_search_space, threshold=thresh)
            correspondences = []
            vectorized_maxima = np.flip(np.stack(self.maxima.nonzero(), axis=1), axis=1)
            while len(correspondences) == 0:
                correspondences = cf.establishCorrespondences(vectorized_maxima)

            self.grid2DPixLocations = [[self.laser.getXYfromN(id), np.flip(pix)] for id, pix in correspondences]

        pixel_coords = np.array(self.grid2DPixLocations)[:, 1, :].astype(np.int32)
        debug_img = np.zeros(self.images[0].shape, np.uint8)
        debug_img[pixel_coords[:, 0], pixel_coords[:, 1]] = 255
        debug_img = cv2.dilate(debug_img, np.ones((3,3)))

        base_img = self.images[self.frameOfClosedGlottis]
        base_img = cv2.cvtColor(base_img, cv2.COLOR_GRAY2BGR)
        base_img = base_img | cv2.cvtColor(debug_img, cv2.COLOR_GRAY2BGR)
        
        self.image_widget.updateImage(base_img, self.image_widget.getWidget("Closed Vocal Folds"))
        
    def triangulate(self):
        min_search_space = float(self.menu_widget.getSubmenuValue("RHC", "Minimum Distance"))
        max_search_space = float(self.menu_widget.getSubmenuValue("RHC", "Maximum Distance"))

        temporalCorrespondence = Correspondences.generateFramewise(self.images, self.frameOfClosedGlottis, self.grid2DPixLocations, self.roi)
        self.triangulatedPoints = np.array(Triangulation.triangulationMat(self.camera, self.laser, temporalCorrespondence, min_search_space, max_search_space, min_search_space, max_search_space))

        
    def denseShapeEstimation(self):    
        zSubdivisions = int(self.menu_widget.getSubmenuValue("Tensor Product M5", "Z Subdivisions"))
        #glottalmidline = self.segmentator.getGlottalMidline(self.images[self.frameOfClosedGlottis])

        if self.point_cloud_id is not None:
            self.viewer_widget.remove_mesh(self.point_cloud_id)

        self.leftDeformed, self.rightDeformed, self.leftPoints, self.rightPoints, self.pointclouds = SiliconeSurfaceReconstruction.controlPointBasedARAP(self.triangulatedPoints, self.images, self.camera, self.segmentator, zSubdivisions=zSubdivisions)

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
        


if __name__ == "__main__":
    viewer_app = QApplication(["Vocal3D - Vocal Fold 3D Reconstruction"])
    viewer = Viewer()
    viewer.show()

    # Launch the Qt application
    viewer_app.exec()