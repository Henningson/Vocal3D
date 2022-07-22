from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLineEdit, QPushButton

from QLines import QHLine
from OpenCloseSaveWidget import OpenCloseSaveWidget
from SubMenuWidget import SubMenuWidget


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
        self.addSubMenu("Segmentation", [("Koc et al", "checkbox", False), ("Neural Segmentation", "checkbox", False), ("Silicone Segmentation", "checkbox", True)])
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