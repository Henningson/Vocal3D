from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QWidget, QHBoxLayout, QVBoxLayout, QSlider, QPushButton

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
