from PyQt6.QtWidgets import QGraphicsScene, QGraphicsItem, QGraphicsPathItem, QGraphicsDropShadowEffect
from PyQt6.QtCore import Qt, QRectF, QPointF, QTimer
from PyQt6.QtGui import QBrush, QPen, QColor, QPainterPath, QFont, QPainter

from .layout_engine import ForceDirectedLayout

class GraphScene(QGraphicsScene):
    def __init__(self):
        super().__init__()
        self.nodes = {} # id -> NodeItem
        self.edges = []
        # Dark Cyberpunk Background
        self.setBackgroundBrush(QBrush(QColor("#0a0a0a")))
        self.setSceneRect(-5000, -5000, 10000, 10000)

        # Physics
        self.layout_engine = ForceDirectedLayout()
        self.physics_enabled = True
        self.physics_timer = QTimer()
        self.physics_timer.timeout.connect(self.tick)
        self.physics_timer.start(16) # ~60 FPS

    def set_physics_enabled(self, enabled: bool):
        self.physics_enabled = enabled
        if enabled:
            self.physics_timer.start(16)
        else:
            self.physics_timer.stop()

    def tick(self):
        """Simple Force-Directed Layout Step"""
        if self.physics_enabled:
            self.layout_engine.compute(self.nodes, self.edges)

    def add_node(self, node_data: dict):
        nid = node_data.get("id")
        if nid in self.nodes:
            self.update_node(node_data)
            return

        item = NodeItem(node_data)

        # Simple auto-layout: Spiral or Grid (Fallback)
        count = len(self.nodes)
        x = (count % 10) * 250
        y = (count // 10) * 200

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

            # Improved Layout Heuristic: Place child relative to parent if at 0,0
            if target_item.pos() == QPointF(0,0) or target_item.pos() == source_item.pos():
                import random
                offset_x = (random.random() - 0.5) * 400
                target_item.setPos(source_item.pos() + QPointF(offset_x, 250))

            edge = EdgeItem(source_item, target_item)
            self.addItem(edge)
            self.edges.append(edge)

            # Register edge with nodes for efficient updates
            source_item.add_edge(edge)
            target_item.add_edge(edge)

class NodeItem(QGraphicsItem):
    def __init__(self, node_data: dict):
        super().__init__()
        self.node_data = node_data
        self.setFlags(QGraphicsItem.GraphicsItemFlag.ItemIsMovable | QGraphicsItem.GraphicsItemFlag.ItemIsSelectable | QGraphicsItem.GraphicsItemFlag.ItemSendsGeometryChanges)
        self.width = 240
        self.height = 140
        self._edges = []

        # Cyberpunk / Neon Palette
        self.color_map = {
            "active": QColor("#00ffcc"),    # Cyan
            "running": QColor("#00ffcc"),   # Cyan
            "success": QColor("#00ff66"),   # Neon Green
            "failed": QColor("#ff0055"),    # Neon Red/Pink
            "error": QColor("#ff0055"),     # Neon Red/Pink
            "reflexion": QColor("#ffcc00"), # Neon Orange/Gold
            "pending": QColor("#444444")    # Dark Grey
        }

        self.emoji_map = {
            "active": "⚙️",
            "running": "🧠",
            "success": "✅",
            "failed": "❌",
            "error": "💥",
            "reflexion": "💡",
            "pending": "⏳"
        }

        # Glow Effect
        self.glow = QGraphicsDropShadowEffect()
        self.glow.setBlurRadius(20)
        self.glow.setOffset(0, 0)
        self.setGraphicsEffect(self.glow)
        self.update_glow()

    def add_edge(self, edge):
        self._edges.append(edge)

    def itemChange(self, change, value):
        if change == QGraphicsItem.GraphicsItemChange.ItemPositionHasChanged:
            # Update connected edges efficiently
            for edge in self._edges:
                edge.update_path()
        return super().itemChange(change, value)

    def update_data(self, data: dict):
        self.node_data.update(data)
        self.update_glow()
        self.update()

    def update_glow(self):
        status = self.node_data.get("status", "pending")
        color = self.color_map.get(status, QColor("#444444"))
        priority = self.node_data.get("priority", "medium")

        # Glow intensity based on priority/status
        if status in ["running", "active"]:
            self.glow.setColor(color)
            self.glow.setBlurRadius(30 if priority == "high" else 20)
            self.glow.setEnabled(True)
        elif status == "failed":
            self.glow.setColor(color)
            self.glow.setBlurRadius(25)
            self.glow.setEnabled(True)
        elif priority == "high":
            self.glow.setColor(color)
            self.glow.setBlurRadius(15)
            self.glow.setEnabled(True)
        else:
            self.glow.setEnabled(False)

    def boundingRect(self) -> QRectF:
        return QRectF(-5, -5, self.width + 10, self.height + 10)

    def paint(self, painter: QPainter, option, widget):
        # LOD check
        lod = option.levelOfDetailFromTransform(painter.worldTransform())
        status = self.node_data.get("status", "pending")
        base_color = self.color_map.get(status, QColor("#444444"))

        # Determine Visual Style
        # Background: Semi-transparent dark fill
        painter.setBrush(QBrush(QColor(20, 20, 20, 220)))

        # Border: Colored neon line
        pen = QPen(base_color, 2)
        if self.isSelected():
            pen.setColor(QColor("#ffffff"))
            pen.setWidth(3)
        painter.setPen(pen)

        # Draw Box
        painter.drawRoundedRect(0, 0, self.width, self.height, 10, 10)

        # Draw Header Background
        header_height = 30
        header_path = QPainterPath()
        header_path.addRoundedRect(0, 0, self.width, header_height, 10, 10)
        # Clip bottom to make it flat
        painter.setClipRect(0, 0, self.width, header_height)
        painter.fillPath(header_path, QBrush(base_color.darker(150)))
        painter.setClipping(False)

        # --- LOD 0: Very Far (Symbols only) ---
        if lod < 0.3:
            painter.setFont(QFont("Segoe UI Emoji", 40))
            emoji = self.emoji_map.get(status, "❓")
            painter.drawText(QRectF(0, 0, self.width, self.height), Qt.AlignmentFlag.AlignCenter, emoji)
            return

        # --- LOD 1: Mid Range (Header + Emoji) ---

        # Draw Emoji Icon
        painter.setFont(QFont("Segoe UI Emoji", 14))
        emoji = self.emoji_map.get(status, "")
        painter.setPen(QPen(QColor("#eeeeee")))
        painter.drawText(QRectF(10, 0, 30, header_height), Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter, emoji)

        # Draw ID / Title
        font_title = QFont("Segoe UI", 10, QFont.Weight.Bold)
        painter.setFont(font_title)
        nid = self.node_data.get("id", "Unknown")
        painter.drawText(QRectF(40, 0, self.width-50, header_height), Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter, f"{nid}")

        # Priority Indicator (Right side of header)
        priority = self.node_data.get("priority", "medium")
        p_color = {"high": "#ff0055", "medium": "#ffcc00", "low": "#00ffcc"}.get(priority, "#aaaaaa")
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QBrush(QColor(p_color)))
        painter.drawEllipse(self.width - 20, 8, 12, 12)


        # --- LOD 2: Close Up (Content) ---
        if lod >= 0.6:
            painter.setPen(QPen(QColor("#dddddd")))
            font_body = QFont("Segoe UI", 9)
            painter.setFont(font_body)

            label = self.node_data.get("label", "") or self.node_data.get("prompt", "")

            # Content Area
            rect = QRectF(10, header_height + 5, self.width - 20, self.height - header_height - 10)
            painter.drawText(rect, Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop | Qt.TextFlag.TextWordWrap, label)

            # Draw "Recency" bar at bottom if very recent?
            # (Skipped for now to keep it clean)

class EdgeItem(QGraphicsPathItem):
    def __init__(self, source: NodeItem, target: NodeItem):
        super().__init__()
        self.source = source
        self.target = target
        self.setZValue(-1) # Behind nodes

        # Cyberpunk Edge Style
        self.pen = QPen(QColor("#444444"), 2)
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

        # Dynamic coloring based on source status
        status = self.source.node_data.get("status", "pending")
        if status in ["running", "active", "reflexion"]:
             # Active flow
             self.pen.setColor(QColor("#00ffcc"))
             self.pen.setWidth(3)
        elif status == "failed":
             self.pen.setColor(QColor("#ff0055"))
             self.pen.setWidth(2)
        else:
             self.pen.setColor(QColor("#444444"))
             self.pen.setWidth(2)
        self.setPen(self.pen)

    def paint(self, painter, option, widget):
        # We manually call update_path from NodeItem events, so standard paint is fine
        super().paint(painter, option, widget)
