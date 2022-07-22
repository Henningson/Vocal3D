import sys
sys.path.append("source/")
sys.path.append("source/GUI/")

from PyQt5.QtWidgets import QApplication
import Viewer

if __name__ == "__main__":
    viewer_app = QApplication(["Vocal3D - Vocal Fold 3D Reconstruction"])
    viewer = Viewer.Viewer()
    viewer.show()

    # Launch the Qt application
    viewer_app.exec()