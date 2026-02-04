from PyQt6.QtWidgets import QWidget, QVBoxLayout, QGraphicsView, QGraphicsScene
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QPainter, QWheelEvent

from ..models.graph_model import GraphScene

class GraphWidget(QWidget):
    nodeSelected = pyqtSignal(dict) # Emits node data

    def __init__(self, agent_worker):
        super().__init__()
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)

        # Scene & View
        self.scene = GraphScene()
        self.view = ZoomableGraphicsView(self.scene)
        self.layout.addWidget(self.view)

        # Connect Worker
        self.agent_worker = agent_worker
        self.agent_worker.thoughtCreated.connect(self.scene.add_node)
        self.agent_worker.linkCreated.connect(self.scene.add_edge)
        self.agent_worker.thoughtUpdated.connect(self.scene.update_node)

        # Selection
        self.scene.selectionChanged.connect(self._on_selection_change)

    def _on_selection_change(self):
        items = self.scene.selectedItems()
        if items:
            # Assuming the first item is the one we care about
            item = items[0]
            if hasattr(item, "node_data"):
                self.nodeSelected.emit(item.node_data)


class ZoomableGraphicsView(QGraphicsView):
    def __init__(self, scene):
        super().__init__(scene)
        self.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setBackgroundBrush(Qt.GlobalColor.black) # Fallback if CSS fails

    def wheelEvent(self, event: QWheelEvent):
        zoom_in_factor = 1.15
        zoom_out_factor = 1 / zoom_in_factor

        # Zoom
        if event.angleDelta().y() > 0:
            zoom_factor = zoom_in_factor
        else:
            zoom_factor = zoom_out_factor

        self.scale(zoom_factor, zoom_factor)
