from OpenCloseSaveWidget import OpenCloseSaveWidget
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QLineEdit, QPushButton, QVBoxLayout, QWidget
from QLines import QHLine
from SubMenuWidget import SubMenuWidget


class MainMenuWidget(QWidget):
    def __init__(self, viewer_palette, parent=None):
        super(MainMenuWidget, self).__init__()
        # self.setStyle(QFrame.Panel | QFrame.Raised)
        self.setStyleSheet(f"background-color: {viewer_palette['menu_background']}")
        self.base_layout = QVBoxLayout()
        self.base_layout.setAlignment(Qt.AlignTop)
        self.setLayout(self.base_layout)

        self.ocs_widget = OpenCloseSaveWidget(self)
        self.base_layout.addWidget(self.ocs_widget)

        self.submenu_dict = {}
        self.button_dict = {}

        self.addSubMenu(
            "Tensor Product M5",
            [
                ("Z Subdivisions", "field", 10), 
                ("X Subdivisions", "field", 4),
                ("R_0", "field", 0.1), # Again from Scherer, for Silicone: 1.0, 2.5 is good
                ("T", "field",   0.3),
                ("psi", "field", 0.0),
            ],
        )
        self.addSubMenu(
            "Segmentation",
            [
                ("Koc et al", "checkbox", False),
                ("Neural Segmentation", "checkbox", True),
                ("Silicone Segmentation", "checkbox", False),
            ],
        )
        self.addSubMenu(
            "Point Tracking",
            [
                ("InvivoSlow", "checkbox", False),
                ("InvivoFast", "checkbox", True),
                ("Silicone", "checkbox", False),
            ],
        )
        self.addSubMenu(
            "Correspondence Estimation",
            [   
                ("RHC", "checkbox", False),
                ("BF_RANSAC", "checkbox", True),
                ("Iterations", "field", 30),
                ("Consensus Size", "field", 8),
                ("GA Thresh", "field", 5.0),
                ("Minimum Distance", "field", 40.0),
                ("Maximum Distance", "field", 80.0),
            ],
        )
        self.addSubMenu(
            "As-Rigid-As-Possible",
            [
             ("Iterations", "field", 2), 
             ("Weight", "field", 10000)],
        )
        self.addSubMenu(
            "CUDA",
            [
                ("Use", "checkbox", True)
            ],
        )
        self.addSubMenu(
            "Least Squares Optimization",
            [("Iterations", "field", 10), ("Learning Rate", "field", 0.1)],
        )
        self.addSubMenu("Temporal Smoothing", [("Window Size", "field", 7)])
        self.addSubMenu(
            "Video Generation",
            [("Generate Video", "checkbox", False), ("Path", "field", "temp")],
        )
        self.addSubMenu("Camera",
            [("Near Plane", "field", 0.01),
             ("Far Plane", "field", 100.0),
             ("FOV", "field", 60.0),
             ("X-Pos", "field", 0),
             ("Y-Pos", "field", 2.0),
             ("Z-Pos", "field", 0),
             ("X-Dir", "field", 0.0),
             ("Y-Dir", "field", -1.0),
             ("Z-Dir", "field", 0.0)])
        
        self.base_layout.addWidget(QHLine())
        self.addButton("Apply Cam")
        self.addButton("Top-Down Cam")
        self.addButton("45Deg Cam")
        self.base_layout.addWidget(QHLine())
        self.addButton("Compute Features")
        self.addButton("Track Points")
        self.addButton("Build Correspondences")
        self.addButton("Triangulate")
        self.addButton("Dense Shape Estimation")
        self.addButton("Least Squares Optimization")
        self.addButton("Temporal Smoothing")
        self.base_layout.addWidget(QHLine())
        self.addButton("Automatic Reconstruction")
        self.addButton("Save Models")

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
