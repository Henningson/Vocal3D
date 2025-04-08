import sys

import sys
sys.path.append(".")
sys.path.append("source/")

import igl

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
    QWidget,
    QHBoxLayout,
    QVBoxLayout,
    QPushButton
)

class Viewer(QWidget):
    close_signal = pyqtSignal()

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

        self.point_clouds = list()

        self.main_layout = QVBoxLayout(self)
        self.viewer_widget = ViewerWidget(self)
        self.main_layout.addWidget(self.viewer_widget)

        self.viewer_widget.setMinimumWidth(500)
        self.viewer_widget.setMinimumHeight(500)
        

        vertices1, _ = igl.read_triangle_mesh("cow_without_texture.obj")
        vertices2, _ = igl.read_triangle_mesh("bunny.obj")
        vertices3 = vertices2 * 2.0 + np.array([[1.0, 0.0, 0.0]])
        
        self.vertices = np.concatenate([vertices1, vertices2, vertices3], axis=0)
        self.offsets = [0, vertices1.shape[0], vertices1.shape[0] + vertices2.shape[0]]
        self.elements = [vertices1.shape[0], vertices2.shape[0], vertices3.shape[0]]

        self.mesh_num = 0

        self.instance_index = self.viewer_widget.display_point_cloud(self.vertices)

        self.linked_cameras = False        

        btn = QPushButton("Visibility", default=False, autoDefault=False)
        self.main_layout.addWidget(btn)
        btn.clicked.connect(self.toggleVisibility)

        btn = QPushButton("Verts", default=False, autoDefault=False)
        self.main_layout.addWidget(btn)
        btn.clicked.connect(self.toggleVerts)


    def toggleVisibility(self):
        mesh_instance = self.viewer_widget.get_mesh_instance(self.instance_index)
        mesh_instance.set_visibility(not mesh_instance.get_visibility())
        self.viewer_widget.update()


    def toggleVerts(self):
        mesh_instance = self.viewer_widget.get_mesh(self.instance_index)
        self.mesh_num = (self.mesh_num + 1) % len(self.offsets)

        mesh_instance.mesh_core.offset = self.offsets[self.mesh_num]
        mesh_instance.mesh_core.number_elements = self.elements[self.mesh_num]
        self.viewer_widget.update()


if __name__ == "__main__":
    viewer_app = QApplication(["Vocal3D - Vocal Fold 3D Reconstruction"])
    viewer = Viewer()
    viewer.show()

    # Launch the Qt application
    viewer_app.exec()