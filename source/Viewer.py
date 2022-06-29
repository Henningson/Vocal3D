import sys
import copy
import time
import threading
import Mesh
import igl

from PyIGL_viewer.viewer.viewer_widget import ViewerWidget
from PyIGL_viewer.viewer.ui_widgets import PropertyWidget, LegendWidget
from pyqtgraph.widgets.RawImageWidget import RawImageGLWidget
from PyIGL_viewer.mesh.mesh import GlMeshPrefab, GlMeshPrefabId
from random import randint

import cv2

from PyQt5.QtCore import Qt, pyqtSignal, QTimer, QThread
from PyQt5.QtGui import QImage, QPixmap
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
    QLabel
)

pg.setConfigOption('imageAxisOrder', 'row-major')


class Viewer(QMainWindow):
    close_signal = pyqtSignal()
    screenshot_signal = pyqtSignal(str)
    legend_signal = pyqtSignal(list, list)
    load_shader_signal = pyqtSignal()
    image_changed_signal = pyqtSignal(int)

    def __init__(self, numFrames, heightLeft, heightRight, points_left, points_right, images, segmentations, laserdots, zSubdivs):
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
            f"background-color: {self.viewer_palette['viewer_background']}; color: {self.viewer_palette['font_color']};" + "QSlider::handle:horizontal{background-color: white;}")

        self.images = np.array(images)
        self.segmentations = np.array(segmentations)
        self.laserdots = laserdots

        self.main_layout = QGridLayout()
        self.central_widget = QWidget()
        self.central_widget.setLayout(self.main_layout)

        menu_widget = QFrame(self)
        menu_widget.setFrameStyle(QFrame.Panel | QFrame.Raised)
        menu_widget.setStyleSheet(
            f"background-color: {self.viewer_palette['menu_background']}"
        )
        self.menu_layout = QVBoxLayout()
        self.menu_layout.setAlignment(Qt.AlignTop)
        menu_widget.setLayout(self.menu_layout)

        self.line_current_playtime_left = pg.InfiniteLine(pos=0)
        self.line_current_playtime_right = pg.InfiniteLine(pos=0)

        self.image_viewer = self.generate_video_frame_widget()
        self.updateImages(0)

        self.graph_height_left = pg.PlotWidget()
        x = np.arange(0, numFrames)
        pen = pg.mkPen(color=(255, 125, 15))
        self.graph_height_left.plot(x, heightLeft, pen=pen)
        self.graph_height_left.addItem(self.line_current_playtime_left)


        self.graph_height_right = pg.PlotWidget()
        x = np.arange(0, numFrames)
        pen = pg.mkPen(color=(255, 125, 15))
        self.graph_height_right.plot(x, heightRight, pen=pen)
        self.graph_height_right.addItem(self.line_current_playtime_right)

        #self.graph_GAW.removeItem(self.line_current_playtime)

        self.main_layout.addWidget(menu_widget, 0, 0, -1, 1)
        self.main_layout.addWidget(self.image_viewer, 0, 4, 2, 1)
        self.main_layout.addWidget(self.graph_height_left, 2, 4, 1, 1)
        self.main_layout.addWidget(self.graph_height_right, 3, 4, 1, 1)
        #self.main_layout.addWidget(self.graph_height_left, 0, 4, 4, 1)
        #self.main_layout.addWidget(GAW_graph, 3, 3)

        self.pause = False

        self.viewer_widgets = []
        self.linked_cameras = False
        self.menu_properties = {}
        self.current_menu_layout = self.menu_layout

        self.setCentralWidget(self.central_widget)


        # Add a viewer widget to visualize 3D meshes to our viewer window
        self.viewer_widget, _ = self.add_viewer_widget(0, 1, 5, 2)
        self.viewer_widget.link_light_to_camera()

        self.player_widget = self.generate_video_widget(numFrames)
        self.main_layout.addWidget(self.player_widget, 5, 1, 4, 2)

        self.screenshot_signal.connect(self.save_screenshot_)
        self.legend_signal.connect(self.add_ui_legend_)
        #self.add_ui_button("Do nothing button". self.save_screenshot_)

        # Add a mesh to our viewer widget
        # This requires three steps:
        # - Adding the mesh vertices and faces
        # - Adding a mesh prefab that contains shader attributes and uniform values
        # - Adding an instance of our prefab whose position is defined by a model matrix


        self.pts_left, self.pts_right, self.faces = Mesh.generate_BM5_mesh(points_left, points_right, zSubdivs)
        vertices_left = self.pts_left[0].reshape((-1, 3))
        vertices_right = self.pts_right[0].reshape((-1, 3))

        # Here, we use the lambert shader.
        # This shader requires two things:
        # - A uniform value called 'albedo' for the color of the mesh.
        # - An attribute called 'normal' for the mesh normals.
        self.uniforms = {}
        self.vertex_attributes_left = {}
        self.face_attributes_left = {}
        self.vertex_attributes_right = {}
        self.face_attributes_right = {}

        #uniforms["albedo"] = np.array([1.0, 0.6, 0.2])
        self.uniforms["minmax"] = np.array([np.concatenate([self.pts_left, self.pts_right]).reshape((-1, 3))[:, 1].min(), np.concatenate([self.pts_left, self.pts_right]).reshape((-1, 3))[:, 1].max()])
        #uniforms["k_ambient"] = np.array([0.0, 0.0, 0.0])
        #uniforms["k_specular"] = np.array([1.0, 1.0, 1.0])
        #uniforms["shininess"] = np.array([50.0])

        # If we want flat shading with normals defined per face.
        #face_normals = igl.per_face_normals(vertices.astype(np.float64), faces, np.array([1.0, 1.0, 1.0])).astype(
        #    np.float32
        #)
        #face_attributes["normal"] = face_normals

        # If we want smooth shading with normals defined per vertex.
        self.vertex_normals_left = igl.per_vertex_normals(vertices_left, self.faces, igl.PER_VERTEX_NORMALS_WEIGHTING_TYPE_AREA).astype(np.float32)
        self.vertex_attributes_left['normal'] = self.vertex_normals_left

        vertex_normals_right = igl.per_vertex_normals(vertices_right, self.faces, igl.PER_VERTEX_NORMALS_WEIGHTING_TYPE_AREA).astype(np.float32)
        self.vertex_attributes_right['normal'] = vertex_normals_right

        self.mesh_index_left = self.viewer_widget.add_mesh(vertices_left, self.faces)
        self.mesh_prefab_index_left = self.viewer_widget.add_mesh_prefab(
            self.mesh_index_left,
            "colormap",
            vertex_attributes=self.vertex_attributes_left,
            face_attributes=self.face_attributes_left,
            uniforms=self.uniforms,
        )
        self.instance_index_left = self.viewer_widget.add_mesh_instance(
            self.mesh_prefab_index_left, np.eye(4, dtype="f")
        )
        self.viewer_widget.add_wireframe(self.instance_index_left, line_color=np.array([0.1, 0.1, 0.1]))

        self.mesh_index_right = self.viewer_widget.add_mesh(vertices_right, self.faces)
        self.mesh_prefab_index_right = self.viewer_widget.add_mesh_prefab(
            self.mesh_index_right,
            "colormap",
            vertex_attributes=self.vertex_attributes_right,
            face_attributes=self.face_attributes_right,
            uniforms=self.uniforms,
        )
        self.instance_index_right = self.viewer_widget.add_mesh_instance(
            self.mesh_prefab_index_right, np.eye(4, dtype="f")
        )
        self.viewer_widget.add_wireframe(self.instance_index_right, line_color=np.array([0.1, 0.1, 0.1]))

        self.set_column_stretch(1, 2)

        self.timer_thread = QThread(self)
        self.timer_thread.started.connect(self.gen_timer_thread)

        self.image_timer_thread = QThread(self)
        self.image_timer_thread.started.connect(self.gen_image_timer_thread)

        self.playSet = False

        for i in range(10):
            self.add_ui_button("Button " + str(i), self.test)

        #self.image_viewer.moveToThread(self.image_timer_thread)
        #self.viewer_widget.moveToThread(self.timer_thread)

    def gen_timer_thread(self):
        timer = QTimer(self.timer_thread)
        timer.timeout.connect(self.animate_func)
        timer.setInterval(25)
        timer.start()

    def gen_image_timer_thread(self):
        timer = QTimer(self.image_timer_thread)
        timer.timeout.connect(self.update_images_func)
        timer.setInterval(25)
        timer.start()

    def generate_video_widget(self, frames):
        widget_video = QWidget(self)
        base_layout = QVBoxLayout()

        self.slider_frame = QSlider(Qt.Horizontal, widget_video)
        self.slider_frame.setMinimum(0)
        self.slider_frame.setRange(0, frames - 1)
        self.slider_frame.setValue(0)
        self.slider_frame.setGeometry(0, 0, 1000, 1000)
        #self.slider_frame.
        #self.slider_frame.set
        base_layout.addWidget(self.slider_frame)
        widget_video.setLayout(base_layout)

        widget_button = QWidget(widget_video)
        layout_button = QHBoxLayout()
        self.button_play = QPushButton("Play")
        self.button_play.clicked.connect(self.play_video_)
        self.button_pause = QPushButton("Pause")
        self.button_pause.clicked.connect(self.pause_video_)
        self.button_stop = QPushButton("Stop")
        self.button_stop.clicked.connect(self.stop_video_)
        self.button_replay = QPushButton("Replay")
        self.button_replay.clicked.connect(self.replay_video_)
        self.button_previous = QPushButton("Previous Frame")
        self.button_previous.clicked.connect(self.previous_frame_)
        self.button_next = QPushButton("Next Frame")
        self.button_next.clicked.connect(self.next_frame_)
        layout_button.addWidget(self.button_play)
        layout_button.addWidget(self.button_pause)
        layout_button.addWidget(self.button_stop)
        layout_button.addWidget(self.button_replay)
        layout_button.addWidget(self.button_previous)
        layout_button.addWidget(self.button_next)
        widget_button.setLayout(layout_button)
        base_layout.addWidget(widget_button)

        return widget_video

    def generate_video_frame_widget(self):
        widget_video = QWidget(self)
        base_layout = QHBoxLayout()
        
        self.image_main = QLabel(self)
        self.image_segmentation = QLabel(self)
        self.image_laserdots = QLabel(self)

        self.image_main.setFixedSize(256, 512)
        self.image_segmentation.setFixedSize(256, 512)
        self.image_laserdots.setFixedSize(256, 512)

        #self.graphicsview_main.setAspectLocked(True)
        #self.graphicsview_segmentation.setAspectLocked(True)
        #self.graphicsview_laserdots.setAspectLocked(True)

        #self.graphicsview_main.useOpenGL()
        #self.graphicsview_segmentation.useOpenGL()
        #self.graphicsview_laserdots.useOpenGL()

        #self.viewbox_main = pg.ViewBox()
        #self.viewbox_segmentation = pg.ViewBox()
        #self.viewbox_laserdots = pg.ViewBox()
        
        #self.viewbox_main.setAspectLocked(True)
        #self.viewbox_segmentation.setAspectLocked(True)
        #self.viewbox_laserdots.setAspectLocked(True)

        #self.graphicsview_main.setCentralItem(self.viewbox_main)
        #self.graphicsview_segmentation.setCentralItem(self.viewbox_segmentation)
        #self.graphicsview_laserdots.setCentralItem(self.viewbox_laserdots)

        #self.image_main = pg.ImageItem()
        #self.image_segmentation = pg.ImageItem()
        #self.image_laserdots = pg.ImageItem()

        #self.viewbox_main.addItem(self.image_main)
        #self.viewbox_segmentation.addItem(self.image_segmentation)
        #self.viewbox_laserdots.addItem(self.image_laserdots)

        base_layout.addWidget(self.image_main)
        base_layout.addWidget(self.image_segmentation)
        base_layout.addWidget(self.image_laserdots)
        widget_video.setLayout(base_layout)

        #widget_video.setFixedSize(256*3, 512)

        return widget_video

    def test(self):
        print("Yay")

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
        self.main_layout.addWidget(widget, x, y, row_span, column_span)
        viewer_widget.show()
        self.viewer_widgets.append(viewer_widget)
        return viewer_widget, len(self.viewer_widgets) - 1

    def get_viewer_widget(self, index):
        if len(self.viewer_widgets) > index:
            return self.viewer_widgets[index]
        else:
            return None

    def link_all_cameras(self):
        if len(self.viewer_widgets) > 0:
            self.camera = self.viewer_widgets[0].camera
            self.linked_cameras = True
            for widget in self.viewer_widgets:
                widget.camera = self.camera
            self.update_all_viewers()

    def unlink_all_cameras(self):
        if len(self.viewer_widgets) > 0:
            self.linked_cameras = False
            for widget in self.viewer_widgets:
                widget.camera = copy.deepcopy(widget.camera)
            self.update_all_viewers()

    def update_all_viewers(self):
        for widget in self.viewer_widgets:
            widget.update()

    def start_ui_group(self, name):
        group_layout = QVBoxLayout()
        widget = QFrame(self)
        widget.setLineWidth(2)
        widget.setLayout(group_layout)
        widget.setObjectName("groupFrame")
        widget.setStyleSheet(
            "#groupFrame { border: 1px solid "
            + self.viewer_palette["ui_group_border_color"]
            + "; }"
        )
        group_label = QLabel(name, widget)
        group_layout.addWidget(group_label)
        group_layout.setAlignment(group_label, Qt.AlignHCenter)
        group_layout.setContentsMargins(2, 2, 2, 2)
        self.menu_layout.addWidget(widget)
        self.current_menu_layout = group_layout

    def finish_ui_group(self):
        self.current_menu_layout = self.menu_layout

    def add_ui_button(self, text, function, color=None):
        if color == None:
            color = self.viewer_palette["ui_element_background"]
        button = QPushButton(text, self)
        button.clicked.connect(function)
        button.setAutoFillBackground(True)
        button.setStyleSheet(f"background-color: {color}")
        self.current_menu_layout.addWidget(button)
        return button

    def add_ui_property(self, property_name, text, initial_value, read_only=False):
        widget = PropertyWidget(text, initial_value, read_only)
        self.menu_properties[property_name] = widget
        self.current_menu_layout.addWidget(widget)

    def add_ui_legend_(self, names, colors):
        legend_widget = LegendWidget(names, colors)
        self.current_menu_layout.addWidget(legend_widget)

    def add_ui_legend(self, names, colors):
        self.legend_signal.emit(names, colors)

    def set_float_property(self, name, new_value):
        if name in self.menu_properties:
            try:
                self.menu_properties[name].set_value(new_value)
            except ValueError:
                return
        else:
            return

    def get_float_property(self, name):
        if name in self.menu_properties:
            try:
                float_property = float(self.menu_properties[name].value)
                return float_property
            except ValueError:
                return None
        else:
            return None

    def set_column_stretch(self, column, stretch):
        self.main_layout.setColumnStretch(column + 1, stretch)

    def set_row_stretch(self, row, stretch):
        self.main_layout.setRowStretch(row, stretch)

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

    def save_screenshot_(self, path):
        screenshot = QApplication.primaryScreen().grabWindow(
            self.central_widget.winId()
        )
        screenshot.save(path, "png")

    def save_screenshot(self, path):
        self.screenshot_signal.emit(path)

    def play_video_(self):
        if not self.playSet:
            self.playSet = True
            self.timer_thread.start()
            self.image_timer_thread.start()

        if self.pause:
            self.pause = False

    def pause_video_(self):
        self.pause = True

    def stop_video_(self):
        self.pause_video_()
        self.updateSlider(0)
        self.onSliderUpdate()

    def replay_video_(self):
        self.updateSlider(0)
        self.play_video_()

    def pause_video_(self):
        self.pause = True

    def previous_frame_(self):
        if self.pause:
            self.updateSlider(self.slider_frame.value() - 1 if self.slider_frame.value() > self.slider_frame.minimum() else self.slider_frame.value())
            self.onSliderUpdate()
            self.updateImages(self.slider_frame.value())

    def next_frame_(self):
        if self.pause:
            self.updateSlider(self.slider_frame.value() + 1 if self.slider_frame.value() < self.slider_frame.maximum() else self.slider_frame.maximum())
            self.onSliderUpdate()
            self.updateImages(self.slider_frame.value())

    def animate_func(self):
        #global pts_, vertices_right, faces
        if not self.pause:
                new_value = self.slider_frame.value() + 1 if self.slider_frame.value() < self.slider_frame.maximum() - 1 else 0
                self.updateMesh(new_value)
                self.updatePlots(new_value)
                self.updateSlider(new_value)

    def update_images_func(self):
        #global pts_, vertices_right, faces
        if not self.pause:
                self.updateImages(self.slider_frame.value())

    def updateImagesAsync(self, val):
        animating_thread = threading.Thread(target=self.updateImages, args=[val])
        animating_thread.start()

    def updateMesh(self, frameNum):
        self.viewer_widget.update_mesh_vertices(self.mesh_index_left, self.pts_left[frameNum].reshape((-1, 3)).astype(np.float32))
        self.viewer_widget.update_mesh_vertices(self.mesh_index_right, self.pts_right[frameNum].reshape((-1, 3)).astype(np.float32))
        self.viewer_widget.update()

    def updateSlider(self, frameNum):
        self.slider_frame.setValue(frameNum)

    def updatePlots(self, frameNum):
        self.line_current_playtime_left.setPos(frameNum)
        self.line_current_playtime_right.setPos(frameNum)

    def onSliderUpdate(self):
        self.updateMesh(self.slider_frame.value())
        self.updatePlots(self.slider_frame.value())

    def updateImages(self, frameNum):
        image = cv2.cvtColor(cv2.rotate(self.images[frameNum], cv2.ROTATE_90_CLOCKWISE), cv2.COLOR_GRAY2BGR)
        h, w, ch = image.shape
        bytesPerLine = ch * w
        qimage = QImage(image.data, w, h, bytesPerLine, QImage.Format_BGR888)
        #p = image.scaled(256, 512, Qt.KeepAspectRatio)
        
        segmentation = cv2.cvtColor(cv2.rotate(self.segmentations[frameNum], cv2.ROTATE_90_CLOCKWISE), cv2.COLOR_GRAY2BGR)
        segmentation = segmentation | image
        h, w, ch = segmentation.shape
        bytesPerLine = ch * w
        qsegmentation = QImage(segmentation.data, w, h, bytesPerLine, QImage.Format_BGR888)
        #p = image.scaled(256, 512, Qt.KeepAspectRatio)
        
        laserdotim = cv2.rotate(self.laserdots[frameNum], cv2.ROTATE_90_CLOCKWISE)
        h, w, ch = laserdotim.shape
        bytesPerLine = ch * w
        qlaserdotim = QImage(laserdotim.data, w, h, bytesPerLine, QImage.Format_BGR888)

        self.image_main.setPixmap(QPixmap.fromImage(qimage))
        self.image_segmentation.setPixmap(QPixmap.fromImage(qsegmentation))
        self.image_laserdots.setPixmap(QPixmap.fromImage(qlaserdotim))