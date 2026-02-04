from PyQt6.QtCore import QPointF

class ForceDirectedLayout:
    def __init__(self):
        # Constants
        self.k = 300.0 # Ideal spring length
        self.repulsion = 500000.0
        self.attraction = 0.05
        self.damping = 0.85
        self.max_velocity = 20.0 # Limit speed

    def compute(self, nodes: dict, edges: list):
        """
        Computes the forces for a single tick of the layout simulation and updates positions.

        Args:
            nodes: Dict mapping node_id to NodeItem (which has .pos(), .width, .height)
            edges: List of EdgeItem (which has .source, .target)
        """
        if not nodes: return

        # Initialize forces
        forces = {nid: QPointF(0, 0) for nid in nodes}

        # 1. Repulsion (All nodes repel each other)
        node_items = list(nodes.values())
        for i, n1 in enumerate(node_items):
            p1 = n1.pos() + QPointF(n1.width/2, n1.height/2)
            for n2 in node_items[i+1:]:
                p2 = n2.pos() + QPointF(n2.width/2, n2.height/2)
                vec = p1 - p2
                dist_sq = vec.x()**2 + vec.y()**2
                if dist_sq < 1: dist_sq = 1

                # F = k / d^2
                force = vec * (self.repulsion / dist_sq)

                forces[n1.node_data["id"]] += force
                forces[n2.node_data["id"]] -= force

        # 2. Attraction (Edges pull connected nodes)
        for edge in edges:
            n1 = edge.source
            n2 = edge.target
            p1 = n1.pos() + QPointF(n1.width/2, n1.height/2)
            p2 = n2.pos() + QPointF(n2.width/2, n2.height/2)

            vec = p2 - p1
            dist = (vec.x()**2 + vec.y()**2)**0.5
            if dist < 1: dist = 1

            # Hooke's Law: F = k * (dist - ideal)
            force = vec * ((dist - self.k) * self.attraction)

            forces[n1.node_data["id"]] += force
            forces[n2.node_data["id"]] -= force

        # 3. Apply Forces
        for nid, item in nodes.items():
            if item.isUnderMouse() and item.isSelected():
                continue # Don't move if user is grabbing it

            f = forces[nid]

            # Limit force to prevent explosion
            if f.manhattanLength() > 100:
                f *= (100 / f.manhattanLength())

            # Update pos
            new_pos = item.pos() + f * 0.1
            item.setPos(new_pos)
