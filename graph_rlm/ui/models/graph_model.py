from PyQt6.QtWidgets import QGraphicsScene, QGraphicsItem, QGraphicsPathItem, QGraphicsDropShadowEffect
from PyQt6.QtCore import Qt, QRectF, QPointF, QTimer
from PyQt6.QtGui import QBrush, QPen, QColor, QPainterPath, QFont, QPainter, QLinearGradient

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
        """Simple Force-Directed Layout Step & Animation Tick"""
        if self.physics_enabled:
            self.layout_engine.compute(self.nodes, self.edges)

        # Animate nodes (pulsing effects)
        for node in self.nodes.values():
            node.tick()

    def add_node(self, node_data: dict):
        nid = node_data.get("id")
        if nid in self.nodes:
            self.update_node(node_data)
            return

        item = NodeItem(node_data)

        # Check for explicit coordinates
        if "x" in node_data and "y" in node_data:
            item.setPos(float(node_data["x"]), float(node_data["y"]))
        else:
            # Fallback layout
            count = len(self.nodes)
            x = (count % 10) * 350 # Increased spacing
            y = (count // 10) * 250
            item.setPos(x, y)

        self.addItem(item)
        self.nodes[nid] = item

        # Explicit update to trigger redraw
        self.update()

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

            edge = EdgeItem(source_item, target_item)
            self.addItem(edge)
            self.edges.append(edge)

            source_item.add_edge(edge)
            target_item.add_edge(edge)

            self.update()

class NodeItem(QGraphicsItem):
    def __init__(self, node_data: dict):
        super().__init__()
        self.node_data = node_data
        self.setFlags(QGraphicsItem.GraphicsItemFlag.ItemIsMovable | QGraphicsItem.GraphicsItemFlag.ItemIsSelectable | QGraphicsItem.GraphicsItemFlag.ItemSendsGeometryChanges)
        self.width = 300 # Increased width
        self.height = 180 # Increased height
        self._edges = []

        # Animation State
        self.pulse_phase = 0.0
        self.animating = False

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
        self.update_visual_state()

    def add_edge(self, edge):
        self._edges.append(edge)

    def itemChange(self, change, value):
        if change == QGraphicsItem.GraphicsItemChange.ItemPositionHasChanged:
            for edge in self._edges:
                edge.update_path()
        return super().itemChange(change, value)

    def update_data(self, data: dict):
        self.node_data.update(data)
        self.update_visual_state()
        self.update()

    def update_visual_state(self):
        status = self.node_data.get("status", "pending")
        color = self.color_map.get(status, QColor("#444444"))
        priority = self.node_data.get("priority", "medium")

        self.animating = status in ["running", "active", "reflexion"]

        if self.animating:
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

    def tick(self):
        if self.animating:
            self.pulse_phase += 0.1
            if self.pulse_phase > 6.28: # 2*PI
                self.pulse_phase = 0.0
            self.update() # Trigger repaint for animation

    def boundingRect(self) -> QRectF:
        return QRectF(-5, -5, self.width + 10, self.height + 10)

    def paint(self, painter: QPainter, option, widget):
        status = self.node_data.get("status", "pending")
        base_color = self.color_map.get(status, QColor("#444444"))

        # Calculate Pulse Alpha
        alpha_pulse = 0
        if self.animating:
             # Sine wave oscillation for alpha: 20 to 60
             import math
             alpha_pulse = int(40 + 20 * math.sin(self.pulse_phase))

        # --- Background (Gradient) ---
        grad = QLinearGradient(0, 0, 0, self.height)
        bg_color = QColor(20, 20, 20, 230)
        grad.setColorAt(0, bg_color)
        grad.setColorAt(1, QColor(10, 10, 10, 250))
        painter.setBrush(QBrush(grad))

        # --- Border ---
        pen = QPen(base_color, 2)
        if self.isSelected():
            pen.setColor(QColor("#ffffff"))
            pen.setWidth(3)
        elif self.animating:
            # Pulsing border color
            pulse_color = QColor(base_color)
            pulse_color.setAlpha(150 + alpha_pulse)
            pen.setColor(pulse_color)

        painter.setPen(pen)
        painter.drawRoundedRect(0, 0, self.width, self.height, 8, 8)

        # --- Header ---
        header_height = 35
        header_path = QPainterPath()
        header_path.addRoundedRect(0, 0, self.width, header_height, 8, 8)
        # Clip to top corners only for rounded rect feel?
        # Actually standard rounded rect fill is fine but we need to clip bottom to be flat if we want distinct header
        # Let's just draw a separate rounded rect for header and clip it

        painter.save()
        painter.setClipRect(0, 0, self.width, header_height)
        header_color = base_color.darker(150)
        header_color.setAlpha(200)
        painter.fillPath(header_path, QBrush(header_color))
        painter.restore()

        # Separator Line
        painter.setPen(QPen(base_color, 1))
        painter.drawLine(0, header_height, self.width, header_height)

        # --- Content Rendering with LOD ---
        lod = option.levelOfDetailFromTransform(painter.worldTransform())

        # 1. Header Text (ID & Emoji) - Always visible if LOD > 0.2
        if lod > 0.2:
            painter.setFont(QFont("Segoe UI Emoji", 12))
            emoji = self.emoji_map.get(status, "")

            # Draw Status Icon
            icon_rect = QRectF(10, 0, 30, header_height)
            painter.setPen(QPen(QColor("#eeeeee")))
            painter.drawText(icon_rect, Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter, emoji)

            # Draw ID
            font_title = QFont("Segoe UI", 10, QFont.Weight.Bold)
            painter.setFont(font_title)
            nid = self.node_data.get("id", "Unknown")
            title_rect = QRectF(40, 0, self.width - 50, header_height)
            painter.drawText(title_rect, Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter, f"{nid}")

        # 2. Body (Prompt/Label) - Visible if LOD > 0.4
        if lod > 0.4:
            painter.setPen(QPen(QColor("#dddddd")))
            font_body = QFont("Segoe UI", 10)
            painter.setFont(font_body)

            label = self.node_data.get("label", "") or self.node_data.get("prompt", "")
            # Truncate if too long? TextWordWrap handles wrapping.

            # Area for Body
            body_rect = QRectF(10, header_height + 10, self.width - 20, 60)
            painter.drawText(body_rect, Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop | Qt.TextFlag.TextWordWrap, label)

        # 3. Footer (Result/Status Detail) - Visible if LOD > 0.5
        if lod > 0.5:
            result = self.node_data.get("result", "")
            if result:
                # Separator
                painter.setPen(QPen(QColor("#444444"), 1, Qt.PenStyle.DashLine))
                painter.drawLine(10, header_height + 75, self.width - 10, header_height + 75)

                # Result Text
                painter.setPen(QPen(QColor("#aaaaaa")))
                font_footer = QFont("Consolas", 9)
                painter.setFont(font_footer)

                footer_rect = QRectF(10, header_height + 80, self.width - 20, self.height - header_height - 90)
                painter.drawText(footer_rect, Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop | Qt.TextFlag.TextWordWrap, result)

            # If active, draw a small "Processing..." indicator
            elif self.animating:
                painter.setPen(QPen(self.color_map["active"]))
                font_mini = QFont("Segoe UI", 8, QFont.Weight.Bold)
                painter.setFont(font_mini)
                painter.drawText(QRectF(10, self.height - 20, self.width-20, 20), Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignBottom, "PROCESSING >>")

class EdgeItem(QGraphicsPathItem):
    def __init__(self, source: NodeItem, target: NodeItem):
        super().__init__()
        self.source = source
        self.target = target
        self.setZValue(-1) # Behind nodes
        self.pen = QPen(QColor("#444444"), 2)
        self.pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        self.setPen(self.pen)
        self.update_path()

    def update_path(self):
        start = self.source.pos() + QPointF(self.source.width/2, self.source.height)
        end = self.target.pos() + QPointF(self.target.width/2, 0)

        path = QPainterPath()
        path.moveTo(start)

        dist_y = end.y() - start.y()
        ctrl_offset = max(50, dist_y * 0.5)

        ctrl1 = start + QPointF(0, ctrl_offset)
        ctrl2 = end - QPointF(0, ctrl_offset)
        path.cubicTo(ctrl1, ctrl2, end)

        self.setPath(path)

        status = self.source.node_data.get("status", "pending")
        if status in ["running", "active", "reflexion"]:
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
        super().paint(painter, option, widget)
