from typing import List

import zoomable
from PyQt5 import QtCore
from PyQt5.QtGui import QImage
from PyQt5.QtWidgets import QMenu


class ZoomableVideo(zoomable.Zoomable):

    signal_increment_frame = QtCore.pyqtSignal(bool)
    signal_decrement_frame = QtCore.pyqtSignal(bool)

    def __init__(
        self,
        images: List[QImage] = None,
        parent=None,
    ):
        super(ZoomableVideo, self).__init__(parent)
        self.images = images

        if self.images:
            self.set_image(self.images[0])

        self._current_frame: int = 0
        self._num_frames: int = len(self.images) if self.images else 0

    def redraw(self) -> None:
        if self.images:
            self.set_image(self.images[self._current_frame])

    def change_frame(self, frame: int) -> None:
        if frame < 0 or frame >= self._num_frames:
            return

        self._current_frame = frame
        self.redraw()

    def add_video(self, video: List[QImage]) -> None:
        self.images = video
        self._num_frames = len(self.images)
        self._image_width = video[0].width()
        self._image_height = video[0].height()

    def contextMenuEvent(self, event) -> None:
        """
        Opens a context menu with options for zooming in and out.

        :param event: The QContextMenuEvent containing information about the context menu event.
        :type event: QContextMenuEvent
        """
        menu = QMenu()
        menu.addAction("Zoom in\tMouseWheel Up", self.zoomIn)
        menu.addAction("Zoom out\tMouseWheel Down", self.zoomOut)
        menu.addAction("Frame forward\tRight Arrow", self.frame_forward)
        menu.addAction("Frame backward\tLeft Arrow", self.frame_backward)
        menu.addAction("Reset Zoom", self.zoomReset)
        menu.exec_(event.globalPos())

    def frame_forward(self):
        if self._current_frame + 1 < 0 or self._current_frame + 1 >= self._num_frames:
            return
        self.change_frame(self._current_frame + 1)
        self.signal_increment_frame.emit(True)

    def frame_backward(self):
        if self._current_frame - 1 < 0 or self._current_frame - 1 >= self._num_frames:
            return
        self.change_frame(self._current_frame - 1)
        self.signal_decrement_frame.emit(True)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.fit_view()

    def keyPressEvent(self, event):
        if event.key() == QtCore.Qt.Key_Right:
            self.frame_forward()
        elif event.key() == QtCore.Qt.Key_Left:
            self.frame_backward()
