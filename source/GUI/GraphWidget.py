from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QWidget, QVBoxLayout
import pyqtgraph as pg
import numpy as np

pg.setConfigOption('imageAxisOrder', 'row-major')

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