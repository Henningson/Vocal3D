from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import QWidget, QHBoxLayout, QPushButton, QFileDialog

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