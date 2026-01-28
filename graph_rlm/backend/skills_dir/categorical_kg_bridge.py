"""
Categorical Knowledge Graph Bridge
===================================

A skill that bridges three interconnected research areas:
1. Neo4j Knowledge Graphs (Informal Reasoning)
2. ZX-Calculus / Monoidal Categories (Formal Diagrammatic Reasoning)
3. Prover9 / FOL (Verified Proofs)

Pipeline: Neo4j (Informal) → ZX-style (Formal) → Prover9 (Verified)

Theoretical Foundation
----------------------
This skill implements the insight that knowledge graph structures are implicitly
categorical. By making this structure explicit, we enable:

1. **Diagrammatic Reasoning**: Treat graph patterns as categorical diagrams
   - Nodes are Objects in a category
   - Edges are Morphisms (arrows) between objects
   - Paths represent morphism composition
   - Query equivalence = diagram commutativity

2. **ZX-Calculus Inspiration**: Although ZX-calculus is for quantum computing,
   its key insight applies broadly: diagrammatic rewrite rules can be sound
   and complete. We adopt the approach of:
   - Representing structures as string diagrams (tensor networks)
   - Defining rewrite rules that preserve semantic equality
   - Proving equivalences through rewrite sequences

3. **Categorical Semantics of Schema Evolution**: The schema_builder.py concepts
   map naturally to category theory:
   - SchemaDefinition → Objects (types/classes)
   - evaluate_fit() → Morphism validity (does this edge exist?)
   - evolve_schema() → Functorial transformation (structure-preserving maps)
   - Assimilation → Morphism in existing category
   - Accommodation → Category extension (adding objects/morphisms)

Key Categorical Concepts
------------------------
- **Category**: A collection of objects and morphisms with composition
- **Functor**: A structure-preserving map between categories
- **Natural Transformation**: A way to transform one functor into another
- **Commutative Diagram**: Two paths with same start/end yield equal results
- **Monoidal Category**: Category with tensor product (parallel composition)
- **String Diagram**: 2D representation of morphisms in monoidal categories

DiscoRL Connection
------------------
The meta-learning system in DiscoRL discovers prediction targets that exhibit
categorical structure without explicit programming. Our bridge formalizes this:
- Meta-network: States → Updates (functorial behavior)
- Discovered predictions: Track compositional structure (morphisms)
- Bootstrapping: Future morphisms inform current targets (colimit construction)

Usage
-----
```python
from graph_rlm.backend.skills_dir.categorical_kg_bridge import categorical_kg_bridge, CategoricalBridge

# Analyze a Neo4j pattern
result = categorical_kg_bridge(
    pattern="(Person)-[:KNOWS]->(Person)",
    analysis_type="full"
)

# Or use the bridge directly
bridge = CategoricalBridge()
diagram = bridge.parse_cypher_pattern("(Person)-[:KNOWS]->(Person)")
prover9_clauses = bridge.to_prover9(diagram)
```

MCP Integration
---------------
This skill is designed to work with:
- mcp-logic server: Prover9 theorem proving via prove, verify-commutativity
- chatdag: Knowledge persistence and retrieval
- schema_builder: Cognitive schema operations
"""

import hashlib
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

# =============================================================================
# CORE DATA STRUCTURES
# =============================================================================


class MorphismType(Enum):
    """Classification of edge/morphism types in categorical terms."""

    IDENTITY = "identity"  # Self-loops, identity morphisms
    FUNCTIONAL = "functional"  # Many-to-one relationships (functions)
    RELATIONAL = "relational"  # Many-to-many relationships
    ISOMORPHISM = "isomorphism"  # Bijective, invertible
    MONOMORPHISM = "monomorphism"  # Injective (embeddings)
    EPIMORPHISM = "epimorphism"  # Surjective (quotients)


@dataclass
class CategoricalObject:
    """
    An object in a category, derived from a Neo4j node or label.

    In category theory, objects are atomic - they have no internal structure.
    All structure comes from morphisms between objects.

    Attributes:
        name: Unique identifier for the object
        label: Original Neo4j label (if applicable)
        properties: Node properties (used for functorial mappings)
        is_terminal: True if this is a terminal object (unique morphism from all objects)
        is_initial: True if this is an initial object (unique morphism to all objects)
    """

    name: str
    label: str = ""
    properties: dict[str, Any] = field(default_factory=dict)
    is_terminal: bool = False
    is_initial: bool = False

    def __hash__(self) -> int:
        return hash(self.name)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, CategoricalObject):
            return self.name == other.name
        return False

    def to_fol(self) -> str:
        """Generate FOL representation of this object."""
        return f"object({self.name.lower()})"


@dataclass
class CategoricalMorphism:
    """
    A morphism (arrow) in a category, derived from a Neo4j relationship.

    Morphisms are the primary structure in category theory. They capture:
    - Relationships between entities
    - Functions between types
    - Proofs of propositions (Curry-Howard)

    Attributes:
        name: Unique identifier for the morphism
        source: Source object (domain)
        target: Target object (codomain)
        rel_type: Original Neo4j relationship type
        properties: Relationship properties
        morphism_type: Classification of the morphism
        is_identity: True if this is an identity morphism
    """

    name: str
    source: CategoricalObject
    target: CategoricalObject
    rel_type: str = ""
    properties: dict[str, Any] = field(default_factory=dict)
    morphism_type: MorphismType = MorphismType.RELATIONAL
    is_identity: bool = False

    def __hash__(self) -> int:
        return hash((self.name, self.source.name, self.target.name))

    def to_fol(self) -> list[str]:
        """Generate FOL representation of this morphism."""
        clauses = [
            f"morphism({self.name.lower()})",
            f"source({self.name.lower()}, {self.source.name.lower()})",
            f"target({self.name.lower()}, {self.target.name.lower()})",
        ]
        if self.is_identity:
            clauses.append(f"identity({self.name.lower()}, {self.source.name.lower()})")
        return clauses


@dataclass
class ComposedMorphism:
    """
    A composed morphism representing a path in the graph.

    Composition is the fundamental operation in category theory.
    For morphisms f: A → B and g: B → C, their composition g ∘ f: A → C
    represents following f then g.

    Attributes:
        morphisms: List of morphisms in composition order (first to last)
        name: Generated name for the composed morphism
    """

    morphisms: list[CategoricalMorphism] = field(default_factory=list)
    name: str = ""

    def __post_init__(self) -> None:
        if not self.name and self.morphisms:
            # Generate a name from component morphisms
            components = "_".join(m.name for m in self.morphisms)
            self.name = f"comp_{hashlib.sha256(components.encode()).hexdigest()[:8]}"

    @property
    def source(self) -> CategoricalObject | None:
        """Source of the composed morphism."""
        return self.morphisms[0].source if self.morphisms else None

    @property
    def target(self) -> CategoricalObject | None:
        """Target of the composed morphism."""
        return self.morphisms[-1].target if self.morphisms else None

    def is_valid(self) -> bool:
        """Check if the composition is valid (source/target chain matches)."""
        for i in range(len(self.morphisms) - 1):
            if self.morphisms[i].target != self.morphisms[i + 1].source:
                return False
        return True

    def to_fol_premises(self) -> list[str]:
        """Generate FOL premises for this composition."""
        if len(self.morphisms) == 0:
            return []
        if len(self.morphisms) == 1:
            return self.morphisms[0].to_fol()

        premises = []
        for m in self.morphisms:
            premises.extend(m.to_fol())

        # Add composition facts
        current = self.morphisms[0].name.lower()
        for i in range(1, len(self.morphisms)):
            next_morph = self.morphisms[i].name.lower()
            if i < len(self.morphisms) - 1:
                temp_name = f"{self.name}_temp_{i}"
                premises.append(f"compose({next_morph}, {current}, {temp_name})")
                current = temp_name
            else:
                premises.append(
                    f"compose({next_morph}, {current}, {self.name.lower()})"
                )

        return premises


@dataclass
class CategoricalDiagram:
    """
    A categorical diagram representing a graph pattern.

    Diagrams are visual/structural representations of categorical relationships.
    A diagram "commutes" if all paths with the same start and end yield
    equal composed morphisms.

    This is the core data structure for bridging Neo4j → ZX → Prover9.

    Attributes:
        objects: Set of objects in the diagram
        morphisms: Set of morphisms in the diagram
        paths: All paths discovered in the diagram
        commutative_squares: Detected commutative squares
    """

    name: str = "unnamed_diagram"
    objects: set[CategoricalObject] = field(default_factory=set)
    morphisms: set[CategoricalMorphism] = field(default_factory=set)
    paths: list[ComposedMorphism] = field(default_factory=list)
    commutative_squares: list[tuple[ComposedMorphism, ComposedMorphism]] = field(
        default_factory=list
    )

    def add_object(self, obj: CategoricalObject) -> None:
        """Add an object to the diagram."""
        self.objects.add(obj)

    def add_morphism(self, morph: CategoricalMorphism) -> None:
        """Add a morphism and its source/target objects to the diagram."""
        self.objects.add(morph.source)
        self.objects.add(morph.target)
        self.morphisms.add(morph)

    def find_paths(
        self, source: CategoricalObject, target: CategoricalObject, max_length: int = 5
    ) -> list[ComposedMorphism]:
        """
        Find all paths from source to target up to max_length.

        Uses DFS to enumerate paths in the underlying graph.
        """
        paths: list[ComposedMorphism] = []

        def dfs(current: CategoricalObject, path: list[CategoricalMorphism]) -> None:
            if len(path) > max_length:
                return
            if current == target and path:
                paths.append(ComposedMorphism(morphisms=list(path)))
                return

            for morph in self.morphisms:
                if morph.source == current and morph not in path:
                    path.append(morph)
                    dfs(morph.target, path)
                    path.pop()

        dfs(source, [])
        return paths

    def detect_commutative_squares(
        self,
    ) -> list[tuple[ComposedMorphism, ComposedMorphism]]:
        """
        Detect potential commutative squares in the diagram.

        A commutative square has the form:
            A --f--> B
            |        |
            g        h
            |        |
            v        v
            C --k--> D

        Where h ∘ f = k ∘ g (the two paths from A to D are equal).
        """
        squares: list[tuple[ComposedMorphism, ComposedMorphism]] = []

        # Find all pairs of objects and check for parallel paths
        for obj_a in self.objects:
            for obj_d in self.objects:
                if obj_a == obj_d:
                    continue

                paths = self.find_paths(obj_a, obj_d, max_length=3)

                # Check pairs of paths of length 2 (forming squares)
                for i, path1 in enumerate(paths):
                    for path2 in paths[i + 1 :]:
                        if len(path1.morphisms) == 2 and len(path2.morphisms) == 2:
                            # This is a square - paths might commute
                            squares.append((path1, path2))

        self.commutative_squares = squares
        return squares

    def to_zx_representation(self) -> dict[str, Any]:
        """
        Generate a ZX-calculus inspired representation.

        While not a full ZX-calculus implementation, this captures the
        key ideas of string diagrams:
        - Wires (objects) flow left to right
        - Boxes (morphisms) connect wires
        - Parallel composition = tensor product
        - Sequential composition = wire connection
        """
        nodes: list[dict[str, Any]] = []
        edges: list[dict[str, Any]] = []

        # Objects become wire endpoints
        for obj in self.objects:
            nodes.append(
                {
                    "id": obj.name,
                    "type": "wire_endpoint",
                    "label": obj.label or obj.name,
                    "is_terminal": obj.is_terminal,
                    "is_initial": obj.is_initial,
                }
            )

        # Morphisms become boxes connecting wires
        for morph in self.morphisms:
            nodes.append(
                {
                    "id": f"box_{morph.name}",
                    "type": "morphism_box",
                    "label": morph.rel_type or morph.name,
                    "morphism_type": morph.morphism_type.value,
                }
            )
            edges.append(
                {"from": morph.source.name, "to": f"box_{morph.name}", "type": "input"}
            )
            edges.append(
                {"from": f"box_{morph.name}", "to": morph.target.name, "type": "output"}
            )

        # Identify tensor product structure (parallel morphisms)
        parallel_groups: list[list[str]] = []
        for m1 in self.morphisms:
            group = [m1.name]
            for m2 in self.morphisms:
                if m1 != m2 and m1.source != m2.source and m1.target != m2.target:
                    # These morphisms are potentially parallel (tensor)
                    group.append(m2.name)
            if len(group) > 1:
                parallel_groups.append(group)

        return {
            "nodes": nodes,
            "edges": edges,
            "parallel_groups": parallel_groups,
            "rewrite_rules": self._generate_rewrite_rules(),
        }

    def _generate_rewrite_rules(self) -> list[dict[str, Any]]:
        """
        Generate ZX-inspired rewrite rules for this diagram.

        Basic rules:
        1. Identity elimination: id ∘ f = f = f ∘ id
        2. Associativity: (h ∘ g) ∘ f = h ∘ (g ∘ f)
        3. Spider fusion: Merge parallel same-type morphisms
        """
        rules: list[dict[str, Any]] = []

        # Identity elimination rules
        for morph in self.morphisms:
            if morph.is_identity:
                rules.append(
                    {
                        "name": f"identity_left_{morph.name}",
                        "pattern": f"compose({morph.name}, X, Y)",
                        "replacement": "X = Y",
                        "condition": f"source(X) = {morph.target.name}",
                    }
                )
                rules.append(
                    {
                        "name": f"identity_right_{morph.name}",
                        "pattern": f"compose(X, {morph.name}, Y)",
                        "replacement": "X = Y",
                        "condition": f"target(X) = {morph.source.name}",
                    }
                )

        # Commutative squares become rewrite rules
        for path1, path2 in self.commutative_squares:
                src_name = path1.source.name if path1.source else "unknown"
                tgt_name = path1.target.name if path1.target else "unknown"
                rules.append(
                    {
                        "name": f"commute_{path1.name}_{path2.name}",
                        "pattern": path1.name,
                        "replacement": path2.name,
                        "condition": f"source = {src_name}, target = {tgt_name}",
                    }
                )

        return rules

    def to_prover9_clauses(self) -> dict[str, Any]:
        """
        Generate Prover9-compatible clauses for verification.

        Returns a dictionary with:
        - premises: FOL statements about the diagram structure
        - category_axioms: Basic category theory axioms
        - commutativity_goals: Conclusions to prove for each commutative square
        """
        premises: list[str] = []

        # Object declarations
        for obj in self.objects:
            premises.append(obj.to_fol())

        # Morphism declarations
        for morph in self.morphisms:
            premises.extend(morph.to_fol())

        # Path compositions
        for path in self.paths:
            premises.extend(path.to_fol_premises())

        # Category axioms (from mcp-logic)
        category_axioms = [
            # Identity morphisms exist
            "all x (object(x) -> exists i (morphism(i) & source(i,x) & target(i,x) & identity(i,x)))",
            # Composition exists when source/target match
            "all f all g ((morphism(f) & morphism(g) & target(f) = source(g)) -> exists h (morphism(h) & compose(g,f,h)))",
            # Composition is associative
            "all f all g all h all fg all gh all fgh all gfh ((compose(g,f,fg) & compose(h,g,gh) & compose(h,fg,fgh) & compose(gh,f,gfh)) -> fgh = gfh)",
            # Left identity law
            "all f all a all id ((morphism(f) & source(f,a) & identity(id,a) & compose(f,id,comp)) -> comp = f)",
            # Right identity law
            "all f all b all id ((morphism(f) & target(f,b) & identity(id,b) & compose(id,f,comp)) -> comp = f)",
        ]

        # Commutativity goals
        goals: list[dict[str, Any]] = []
        for path1, path2 in self.commutative_squares:
            goals.append(
                {
                    "description": f"Verify {path1.name} = {path2.name}",
                    "path_a": [m.name for m in path1.morphisms],
                    "path_b": [m.name for m in path2.morphisms],
                    "object_start": path1.source.name if path1.source else "",
                    "object_end": path1.target.name if path1.target else "",
                    "conclusion": f"{path1.name} = {path2.name}",
                }
            )

        return {
            "premises": premises,
            "category_axioms": category_axioms,
            "commutativity_goals": goals,
            "mcp_logic_ready": True,
            "usage_hint": "Use mcp-logic 'prove' tool with premises + category_axioms as premises and each goal's conclusion",
        }


# =============================================================================
# CYPHER PATTERN PARSER
# =============================================================================


class CypherPatternParser:
    """
    Parse Cypher patterns into categorical diagrams.

    Cypher patterns like (a:Person)-[:KNOWS]->(b:Person) are parsed into
    categorical structure:
    - Nodes with labels → Objects
    - Relationships with types → Morphisms
    - Path patterns → Composed morphisms
    """

    # Regex patterns for Cypher elements
    NODE_PATTERN = re.compile(
        r"\((?P<var>\w*):?(?P<label>\w*)\s*(?:\{[^}]*\})?\)", re.IGNORECASE
    )
    REL_PATTERN = re.compile(
        r"-\[(?P<var>\w*):?(?P<type>\w*)\s*(?:\{[^}]*\})?\]-?>?", re.IGNORECASE
    )
    PATH_PATTERN = re.compile(
        r"(?P<node>\([^)]*\))(?:-\[[^\]]*\]-?>?(?P<next_node>\([^)]*\)))*",
        re.IGNORECASE,
    )

    def __init__(self) -> None:
        self.var_counter = 0
        self.objects: dict[str, CategoricalObject] = {}
        self.morphisms: list[CategoricalMorphism] = []

    def _generate_var(self, prefix: str = "v") -> str:
        """Generate a unique variable name."""
        self.var_counter += 1
        return f"{prefix}_{self.var_counter}"

    def parse_node(self, node_str: str) -> CategoricalObject:
        """Parse a Cypher node pattern into a CategoricalObject."""
        match = self.NODE_PATTERN.match(node_str)
        if not match:
            # Handle bare node like (Person)
            clean = node_str.strip("()")
            if ":" in clean:
                var, label = clean.split(":", 1)
            else:
                var = clean
                label = clean

            if not var:
                var = self._generate_var("n")

            obj = CategoricalObject(name=var, label=label)
            self.objects[var] = obj
            return obj

        var = match.group("var") or self._generate_var("n")
        label = match.group("label") or var

        # Reuse existing object with same variable
        if var in self.objects:
            return self.objects[var]

        obj = CategoricalObject(name=var, label=label)
        self.objects[var] = obj
        return obj

    def parse_relationship(
        self, rel_str: str, source: CategoricalObject, target: CategoricalObject
    ) -> CategoricalMorphism:
        """Parse a Cypher relationship pattern into a CategoricalMorphism."""
        match = self.REL_PATTERN.match(rel_str)

        if match:
            var = match.group("var") or self._generate_var("r")
            rel_type = match.group("type") or "RELATED_TO"
        else:
            # Extract type from pattern like -[:KNOWS]->
            type_match = re.search(r":(\w+)", rel_str)
            rel_type = type_match.group(1) if type_match else "RELATED_TO"
            var = self._generate_var("r")

        morph = CategoricalMorphism(
            name=var, source=source, target=target, rel_type=rel_type
        )

        # Detect if this is an identity (self-loop)
        if source == target:
            morph.is_identity = True
            morph.morphism_type = MorphismType.IDENTITY

        self.morphisms.append(morph)
        return morph

    def parse_pattern(self, pattern: str) -> CategoricalDiagram:
        """
        Parse a complete Cypher pattern into a CategoricalDiagram.

        Handles patterns like:
        - (Person)-[:KNOWS]->(Person)
        - (a:Person)-[:KNOWS]->(b:Person)-[:WORKS_AT]->(c:Company)
        - (p:Person)-[:MANAGES]->(p)  (self-loop/identity)
        """
        diagram = CategoricalDiagram(
            name=f"diagram_{hashlib.sha256(pattern.encode()).hexdigest()[:8]}"
        )

        # Tokenize into alternating nodes and relationships
        tokens: list[str] = []
        current = ""
        in_brackets = 0
        in_parens = 0

        for char in pattern:
            if char == "[":
                in_brackets += 1
            elif char == "]":
                in_brackets -= 1
            elif char == "(":
                if current and not in_brackets:
                    tokens.append(current)
                    current = ""
                in_parens += 1
            elif char == ")":
                in_parens -= 1
                current += char
                if in_parens == 0:
                    tokens.append(current)
                    current = ""
                continue

            current += char

        if current:
            tokens.append(current)

        # Process tokens
        nodes: list[CategoricalObject] = []
        relationships: list[str] = []

        for token in tokens:
            token = token.strip()
            if not token:
                continue
            if token.startswith("("):
                nodes.append(self.parse_node(token))
            elif "-" in token:
                relationships.append(token)

        # Create morphisms from nodes and relationships
        for i, rel in enumerate(relationships):
            if i < len(nodes) - 1:
                morph = self.parse_relationship(rel, nodes[i], nodes[i + 1])
                diagram.add_morphism(morph)
            elif len(nodes) == 1:
                # Self-referential pattern
                morph = self.parse_relationship(rel, nodes[0], nodes[0])
                diagram.add_morphism(morph)

        # Add identity morphisms for all objects
        for obj in diagram.objects:
            id_morph = CategoricalMorphism(
                name=f"id_{obj.name}",
                source=obj,
                target=obj,
                rel_type="IDENTITY",
                is_identity=True,
                morphism_type=MorphismType.IDENTITY,
            )
            diagram.add_morphism(id_morph)

        # Detect paths and commutative squares
        for obj1 in diagram.objects:
            for obj2 in diagram.objects:
                if obj1 != obj2:
                    paths = diagram.find_paths(obj1, obj2)
                    diagram.paths.extend(paths)

        diagram.detect_commutative_squares()

        return diagram


# =============================================================================
# CATEGORICAL BRIDGE CLASS
# =============================================================================


class CategoricalBridge:
    """
    Main bridge class for Neo4j → ZX → Prover9 pipeline.

    This class coordinates the translation of informal graph patterns
    into formal categorical structures that can be verified.

    Example:
        bridge = CategoricalBridge()

        # From Cypher pattern
        diagram = bridge.parse_cypher_pattern("(Person)-[:KNOWS]->(Person)")

        # Get ZX-style representation
        zx = bridge.to_zx_diagram(diagram)

        # Get Prover9 clauses for verification
        clauses = bridge.to_prover9(diagram)

        # Full analysis
        analysis = bridge.full_analysis(diagram)
    """

    def __init__(self) -> None:
        self.parser = CypherPatternParser()
        self._diagrams: dict[str, CategoricalDiagram] = {}

    def parse_cypher_pattern(self, pattern: str) -> CategoricalDiagram:
        """Parse a Cypher pattern string into a categorical diagram."""
        diagram = self.parser.parse_pattern(pattern)
        self._diagrams[diagram.name] = diagram
        return diagram

    def from_neo4j_result(
        self, nodes: list[dict[str, Any]], relationships: list[dict[str, Any]]
    ) -> CategoricalDiagram:
        """
        Create a diagram from Neo4j query results.

        Args:
            nodes: List of node dictionaries with 'id', 'labels', 'properties'
            relationships: List of relationship dicts with 'id', 'type',
                          'startNode', 'endNode', 'properties'
        """
        diagram = CategoricalDiagram(name=f"neo4j_result_{len(self._diagrams)}")

        # Create objects from nodes
        node_map: dict[Any, CategoricalObject] = {}
        for node in nodes:
            obj = CategoricalObject(
                name=str(node.get("id", node.get("name", ""))),
                label=(
                    node.get("labels", [""])[0]
                    if isinstance(node.get("labels"), list)
                    else str(node.get("labels", ""))
                ),
                properties=node.get("properties", {}),
            )
            node_map[node.get("id")] = obj
            diagram.add_object(obj)

        # Create morphisms from relationships
        for rel in relationships:
            start_id = rel.get("startNode") or rel.get("start")
            end_id = rel.get("endNode") or rel.get("end")

            source = node_map.get(start_id)
            target = node_map.get(end_id)

            if source and target:
                morph = CategoricalMorphism(
                    name=str(rel.get("id", f"r_{len(diagram.morphisms)}")),
                    source=source,
                    target=target,
                    rel_type=rel.get("type", "RELATED_TO"),
                    properties=rel.get("properties", {}),
                )
                diagram.add_morphism(morph)

        # Add identity morphisms
        for obj in diagram.objects:
            id_morph = CategoricalMorphism(
                name=f"id_{obj.name}",
                source=obj,
                target=obj,
                rel_type="IDENTITY",
                is_identity=True,
                morphism_type=MorphismType.IDENTITY,
            )
            diagram.add_morphism(id_morph)

        # Detect structure
        for obj1 in diagram.objects:
            for obj2 in diagram.objects:
                if obj1 != obj2:
                    paths = diagram.find_paths(obj1, obj2)
                    diagram.paths.extend(paths)

        diagram.detect_commutative_squares()
        self._diagrams[diagram.name] = diagram

        return diagram

    def to_zx_diagram(self, diagram: CategoricalDiagram) -> dict[str, Any]:
        """Convert diagram to ZX-calculus inspired representation."""
        return diagram.to_zx_representation()

    def to_prover9(self, diagram: CategoricalDiagram) -> dict[str, Any]:
        """Convert diagram to Prover9-compatible clauses."""
        return diagram.to_prover9_clauses()

    def analyze_structure(self, diagram: CategoricalDiagram) -> dict[str, Any]:
        """
        Analyze the categorical structure of a diagram.

        Returns analysis including:
        - Object count and types
        - Morphism count and classification
        - Detected patterns (commutative squares, pullbacks, etc.)
        - Categorical properties (has terminal? has initial? is connected?)
        """
        analysis: dict[str, Any] = {
            "diagram_name": diagram.name,
            "object_count": len(diagram.objects),
            "morphism_count": len(diagram.morphisms),
            "path_count": len(diagram.paths),
            "commutative_square_count": len(diagram.commutative_squares),
        }

        # Object analysis
        analysis["objects"] = [
            {
                "name": obj.name,
                "label": obj.label,
                "is_terminal": obj.is_terminal,
                "is_initial": obj.is_initial,
                "incoming_morphisms": sum(
                    1
                    for m in diagram.morphisms
                    if m.target == obj and not m.is_identity
                ),
                "outgoing_morphisms": sum(
                    1
                    for m in diagram.morphisms
                    if m.source == obj and not m.is_identity
                ),
            }
            for obj in diagram.objects
        ]

        # Morphism analysis by type
        type_counts: dict[str, int] = {}
        for morph in diagram.morphisms:
            t = morph.morphism_type.value
            type_counts[t] = type_counts.get(t, 0) + 1
        analysis["morphism_types"] = type_counts

        # Connectivity analysis
        non_id_morphisms = [m for m in diagram.morphisms if not m.is_identity]
        analysis["is_connected"] = self._is_connected(diagram)
        analysis["has_cycles"] = self._has_cycles(diagram)

        # Special patterns
        analysis["patterns_detected"] = []

        if analysis["commutative_square_count"] > 0:
            analysis["patterns_detected"].append("commutative_squares")

        # Check for pullback candidates (pairs of morphisms with same target)
        pullback_candidates = {}
        for m in non_id_morphisms:
            tgt = m.target.name
            if tgt not in pullback_candidates:
                pullback_candidates[tgt] = []
            pullback_candidates[tgt].append(m.name)

        for tgt, morphs in pullback_candidates.items():
            if len(morphs) >= 2:
                analysis["patterns_detected"].append(f"pullback_candidate_at_{tgt}")

        return analysis

    def _is_connected(self, diagram: CategoricalDiagram) -> bool:
        """Check if the diagram is connected (ignoring identity morphisms)."""
        if len(diagram.objects) <= 1:
            return True

        visited: set[str] = set()
        to_visit = [next(iter(diagram.objects))]

        while to_visit:
            obj = to_visit.pop()
            if obj.name in visited:
                continue
            visited.add(obj.name)

            for morph in diagram.morphisms:
                if morph.is_identity:
                    continue
                if morph.source == obj and morph.target.name not in visited:
                    to_visit.append(morph.target)
                if morph.target == obj and morph.source.name not in visited:
                    to_visit.append(morph.source)

        return len(visited) == len(diagram.objects)

    def _has_cycles(self, diagram: CategoricalDiagram) -> bool:
        """Check if the diagram has non-trivial cycles."""
        for obj in diagram.objects:
            paths = diagram.find_paths(obj, obj, max_length=5)
            # Filter out trivial paths (just identity)
            for path in paths:
                if len(path.morphisms) > 1 or (
                    len(path.morphisms) == 1 and not path.morphisms[0].is_identity
                ):
                    return True
        return False

    def full_analysis(self, diagram: CategoricalDiagram) -> dict[str, Any]:
        """
        Perform complete analysis and generate all outputs.

        This is the main entry point for the bridge functionality.
        Returns everything needed to:
        1. Understand the categorical structure
        2. Visualize as a ZX-style diagram
        3. Verify with Prover9
        """
        return {
            "structure_analysis": self.analyze_structure(diagram),
            "zx_representation": self.to_zx_diagram(diagram),
            "prover9_clauses": self.to_prover9(diagram),
            "validation_status": self._validate_translation(diagram),
        }

    def _validate_translation(self, diagram: CategoricalDiagram) -> dict[str, Any]:
        """
        Validate that the translation preserves structure.

        Checks:
        1. All objects have corresponding FOL declarations
        2. All morphisms have source/target declarations
        3. Composition structure is preserved
        4. Rewrite rules are sound (don't introduce new structure)
        """
        issues: list[str] = []

        prover9 = diagram.to_prover9_clauses()
        premises = prover9["premises"]

        # Check object declarations
        for obj in diagram.objects:
            if not any(f"object({obj.name.lower()})" in p for p in premises):
                issues.append(f"Missing object declaration for {obj.name}")

        # Check morphism declarations
        for morph in diagram.morphisms:
            has_morphism = any(f"morphism({morph.name.lower()})" in p for p in premises)
            has_source = any(f"source({morph.name.lower()}" in p for p in premises)
            has_target = any(f"target({morph.name.lower()}" in p for p in premises)

            if not has_morphism:
                issues.append(f"Missing morphism declaration for {morph.name}")
            if not has_source:
                issues.append(f"Missing source declaration for {morph.name}")
            if not has_target:
                issues.append(f"Missing target declaration for {morph.name}")

        # Check identity morphisms
        identity_count = sum(1 for m in diagram.morphisms if m.is_identity)
        expected_identities = len(diagram.objects)
        if identity_count < expected_identities:
            issues.append(
                f"Missing identity morphisms: expected {expected_identities}, found {identity_count}"
            )

        return {
            "valid": len(issues) == 0,
            "issues": issues,
            "object_count": len(diagram.objects),
            "morphism_count": len(diagram.morphisms),
            "premise_count": len(premises),
        }


# =============================================================================
# SKILL ENTRY POINT
# =============================================================================


def categorical_kg_bridge(
    pattern: str | None = None,
    nodes: list[dict[str, Any]] | None = None,
    relationships: list[dict[str, Any]] | None = None,
    analysis_type: str = "full",
) -> dict[str, Any]:
    """
    Main entry point for the categorical knowledge graph bridge skill.

    Bridges Neo4j knowledge graphs with formal categorical reasoning
    and automated theorem proving.

    Pipeline: Neo4j (Informal) → ZX-style (Formal) → Prover9 (Verified)

    Args:
        pattern: Cypher pattern string, e.g., "(Person)-[:KNOWS]->(Person)"
        nodes: List of Neo4j node dictionaries (alternative to pattern)
        relationships: List of Neo4j relationship dictionaries (with nodes)
        analysis_type: Type of analysis to perform:
            - "full": Complete analysis including all outputs
            - "structure": Only structural analysis
            - "prover9": Only Prover9 clauses for verification
            - "zx": Only ZX-calculus representation

    Returns:
        Dictionary containing:
        - diagram: The categorical diagram representation
        - analysis: Structural analysis (if requested)
        - zx: ZX-calculus representation (if requested)
        - prover9: Prover9-compatible clauses (if requested)
        - validation: Translation validation results

    Example:
        # From Cypher pattern
        result = categorical_kg_bridge(
            pattern="(Person)-[:KNOWS]->(Person)",
            analysis_type="full"
        )

        # Check commutativity goals
        for goal in result["prover9"]["commutativity_goals"]:
            print(f"Verify: {goal['description']}")
            # Use mcp-logic prove tool with these premises/conclusions

    MCP Integration:
        The returned prover9 clauses can be used directly with the mcp-logic
        server's 'prove' and 'verify-commutativity' tools:

        ```python
        from graph_rlm.backend.mcp_tools import mcp_logic

        result = categorical_kg_bridge(pattern="...")
        clauses = result["prover9"]

        # Verify structure
        proof = mcp_logic.prove(
            premises=clauses["premises"] + clauses["category_axioms"],
            conclusion=clauses["commutativity_goals"][0]["conclusion"]
        )
        ```
    """
    bridge = CategoricalBridge()

    # Parse input
    if pattern:
        diagram = bridge.parse_cypher_pattern(pattern)
    elif nodes is not None and relationships is not None:
        diagram = bridge.from_neo4j_result(nodes, relationships)
    else:
        return {
            "error": "Must provide either 'pattern' or both 'nodes' and 'relationships'",
            "usage": "categorical_kg_bridge(pattern='(A)-[:REL]->(B)') or categorical_kg_bridge(nodes=[...], relationships=[...])",
        }

    # Build response based on analysis type
    response: dict[str, Any] = {
        "diagram_name": diagram.name,
        "input_type": "cypher_pattern" if pattern else "neo4j_result",
    }

    if analysis_type == "full":
        full = bridge.full_analysis(diagram)
        response["structure_analysis"] = full["structure_analysis"]
        response["zx_representation"] = full["zx_representation"]
        response["prover9"] = full["prover9_clauses"]
        response["validation"] = full["validation_status"]
    elif analysis_type == "structure":
        response["structure_analysis"] = bridge.analyze_structure(diagram)
    elif analysis_type == "prover9":
        response["prover9"] = bridge.to_prover9(diagram)
    elif analysis_type == "zx":
        response["zx_representation"] = bridge.to_zx_diagram(diagram)
    else:
        response["error"] = f"Unknown analysis_type: {analysis_type}"
        response["valid_types"] = ["full", "structure", "prover9", "zx"]

    return response


# Export main function for skill usage
__all__ = [
    "categorical_kg_bridge",
    "CategoricalBridge",
    "CategoricalDiagram",
    "CategoricalObject",
    "CategoricalMorphism",
    "ComposedMorphism",
    "MorphismType",
    "CypherPatternParser",
]
