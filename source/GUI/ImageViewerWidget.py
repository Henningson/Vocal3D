import cv2
from cropAndSubpixelPoints import PointViewer
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QHBoxLayout, QLabel, QVBoxLayout, QWidget
from QLines import QVLine


class ImageViewerWidget(QWidget):
    def __init__(self, parent=None):
        super(ImageViewerWidget, self).__init__()
        self.base_layout = QHBoxLayout(self)
        
        self.imageDICT = {}
        self.point_viewer = PointViewer(None, None)

        self.addImageWidget("Main", (256, 512))
        self.base_layout.addWidget(QVLine())
        self.addImageWidget("Segmentation", (256, 512))
        self.base_layout.addWidget(QVLine())
        self.base_layout.addWidget(self.point_viewer)

    def addImageWidget(self, title, size):
        widg = QWidget(self)
        lay = QVBoxLayout(widg)

        title_widget = QLabel(title)
        title_font = title_widget.font()
        title_font.setBold(True)
        title_widget.setFont(title_font)

        image_widg = QLabel(title)
        self.imageDICT[title] = image_widg

        lay.addWidget(title_widget, alignment=Qt.AlignmentFlag.AlignCenter)
        lay.addWidget(image_widg)
        self.base_layout.addWidget(widg)

    def updateImages(self, a, b, current_frame):
        # We assume images to be in RGB Format
        self.updateImage(a, self.imageDICT["Main"])
        self.updateImage(b, self.imageDICT["Segmentation"])
        self.point_viewer.change_frame(current_frame)
        self.point_viewer.redraw()

    def convertImage(self, image):
        # Check if Mono
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

        if image.shape[0] < image.shape[1]:
            image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)

        h, w, ch = image.shape
        bytesPerLine = ch * w
        return QImage(image.copy().data, w, h, bytesPerLine, QImage.Format_BGR888)
        
    def updateImage(self, image, widget):
        widget.setPixmap(QPixmap.fromImage(self.convertImage(image)))

    def getWidget(self, key):
        return self.imageDICT[key]