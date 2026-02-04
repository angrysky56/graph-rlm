from PyQt6.QtWidgets import QGraphicsScene, QGraphicsItem, QGraphicsRectItem, QGraphicsPathItem
from PyQt6.QtCore import Qt, QRectF, QPointF
from PyQt6.QtGui import QBrush, QPen, QColor, QPainterPath, QFont, QPainter

class GraphScene(QGraphicsScene):
    def __init__(self):
        super().__init__()
        self.nodes = {} # id -> NodeItem
        self.edges = []
        self.setBackgroundBrush(QBrush(QColor("#121212")))
        self.setSceneRect(-5000, -5000, 10000, 10000)

    def add_node(self, node_data: dict):
        nid = node_data.get("id")
        if nid in self.nodes:
            self.update_node(node_data)
            return

        item = NodeItem(node_data)

        # Simple auto-layout: Spiral or Grid
        # For now, let's just place them randomly or in a grid based on count
        count = len(self.nodes)
        x = (count % 10) * 250
        y = (count // 10) * 200

        # If parent exists, place near parent?
        # We need edge info for that, which might come later.

        item.setPos(x, y)
        self.addItem(item)
        self.nodes[nid] = item

    def update_node(self, node_data: dict):
        nid = node_data.get("id")
        if nid in self.nodes:
            self.nodes[nid].update_data(node_data)

    def add_edge(self, link_data: dict):
        source_id = link_data.get("source")
        target_id = link_data.get("target")

        if source_id in self.nodes and target_id in self.nodes:
            source_item = self.nodes[source_id]
            target_item = self.nodes[target_id]

            # Improved Layout Heuristic: Place child relative to parent
            # If target node is at (0,0) [likely unpositioned default], move it
            if target_item.pos() == QPointF(0,0) or target_item.pos() == source_item.pos():
                # Get number of existing children for this source to spread them out
                # This requires tracking structure, simpler to just random jitter for now
                import random
                offset_x = (random.random() - 0.5) * 400
                target_item.setPos(source_item.pos() + QPointF(offset_x, 250))

            edge = EdgeItem(source_item, target_item)
            self.addItem(edge)
            self.edges.append(edge)

class NodeItem(QGraphicsItem):
    def __init__(self, node_data: dict):
        super().__init__()
        self.node_data = node_data
        self.setFlags(QGraphicsItem.GraphicsItemFlag.ItemIsMovable | QGraphicsItem.GraphicsItemFlag.ItemIsSelectable | QGraphicsItem.GraphicsItemFlag.ItemSendsGeometryChanges)
        self.width = 220
        self.height = 120

        self.color_map = {
            "active": QColor("#007acc"),
            "running": QColor("#007acc"),
            "success": QColor("#228b22"),
            "failed": QColor("#d32f2f"),
            "error": QColor("#d32f2f"),
            "reflexion": QColor("#ff9800"),
            "pending": QColor("#555555")
        }

    def itemChange(self, change, value):
        if change == QGraphicsItem.GraphicsItemChange.ItemPositionChange:
            # Force update of scene to redraw connected edges
            # This is a bit expensive if many edges, but okay for MVP
            if self.scene():
                self.scene().update()
        return super().itemChange(change, value)

    def update_data(self, data: dict):
        self.node_data.update(data)
        self.update()

    def boundingRect(self) -> QRectF:
        return QRectF(0, 0, self.width, self.height)

    def paint(self, painter: QPainter, option, widget):
        # LOD check
        lod = option.levelOfDetailFromTransform(painter.worldTransform())

        status = self.node_data.get("status", "pending")
        bg_color = self.color_map.get(status, QColor("#555555"))

        # Selection Highlight
        if self.isSelected():
            painter.setPen(QPen(QColor("#ffffff"), 3))
        else:
            painter.setPen(QPen(QColor("#000000"), 1))

        # Draw Box
        painter.setBrush(QBrush(bg_color))
        painter.drawRoundedRect(0, 0, self.width, self.height, 8, 8)

        # Draw Content
        if lod > 0.4:
            # Status Bar
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QBrush(QColor(0,0,0, 60)))
            painter.drawRoundedRect(0, 0, self.width, 25, 8, 8)
            # Fix bottom corners of header
            painter.drawRect(0, 15, self.width, 10)

            # Status Text
            painter.setPen(QPen(QColor("#eeeeee")))
            font_title = QFont("Segoe UI", 9, QFont.Weight.Bold)
            painter.setFont(font_title)
            painter.drawText(QRectF(10, 0, self.width-20, 25), Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter, f"ID: {self.node_data.get('id', '')[:8]}")

            # Main Text
            font_body = QFont("Segoe UI", 9)
            painter.setFont(font_body)
            label = self.node_data.get("label", "") or self.node_data.get("prompt", "")

            # If zoomed out a bit, truncate more
            limit = 200
            if lod < 0.7: limit = 80

            if len(label) > limit:
                label = label[:limit] + "..."

            rect = QRectF(10, 30, self.width - 20, self.height - 35)
            painter.drawText(rect, Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop | Qt.TextFlag.TextWordWrap, label)

        else:
            # Low LOD: Just a colored box with a symbol
            pass

class EdgeItem(QGraphicsPathItem):
    def __init__(self, source: NodeItem, target: NodeItem):
        super().__init__()
        self.source = source
        self.target = target
        self.setZValue(-1) # Behind nodes
        self.pen = QPen(QColor("#666666"), 2)
        self.pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        self.setPen(self.pen)
        self.update_path()

    def update_path(self):
        start = self.source.pos() + QPointF(self.source.width/2, self.source.height)
        end = self.target.pos() + QPointF(self.target.width/2, 0)

        path = QPainterPath()
        path.moveTo(start)

        # Cubic Bezier for smooth flow
        dist_y = end.y() - start.y()
        ctrl_offset = max(50, dist_y * 0.5)

        ctrl1 = start + QPointF(0, ctrl_offset)
        ctrl2 = end - QPointF(0, ctrl_offset)
        path.cubicTo(ctrl1, ctrl2, end)

        self.setPath(path)

    def paint(self, painter, option, widget):
        # Lazy update on paint ensures connected edges follow nodes
        self.update_path()
        super().paint(painter, option, widget)
