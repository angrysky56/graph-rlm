"""
Schema-Category Integration
===========================

Bridges schema_builder.py with categorical_kg_bridge.py to make the
categorical semantics of schema evolution explicit.

Key Insight: Schema operations are categorical operations in disguise:

| Schema Operation     | Categorical Interpretation                    |
|---------------------|----------------------------------------------|
| SchemaDefinition    | Object in the category of schemas            |
| instantiate()       | Morphism from Schema to Instance             |
| evaluate_fit()      | Morphism existence check + validity          |
| evolve_schema()     | Functorial transformation                    |
| Assimilation        | Morphism within existing category            |
| Accommodation       | Category extension (new objects/morphisms)   |
| Inheritance         | Pullback along parent morphism               |

This module provides:
1. CategoricalSchemaView - View any SchemaBuilder as a category
2. SchemaEvolutionFunctor - Track schema changes as functor applications
3. FitMorphism - Represent evaluate_fit results as morphism properties

Usage:
    from schema_category_integration import CategoricalSchemaView

    builder = SchemaBuilder()
    builder.bootstrap_meta_ontology()

    view = CategoricalSchemaView(builder)
    diagram = view.to_categorical_diagram()
    prover9 = view.to_prover9_clauses()
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from schema_builder import SchemaBuilder

from .categorical_kg_bridge import (
    CategoricalBridge,
    CategoricalDiagram,
    CategoricalMorphism,
    CategoricalObject,
    MorphismType,
)


@dataclass
class FitMorphism:
    """
    Represents the result of evaluate_fit() as a categorical morphism.

    In categorical terms, fitting an instance to a schema is checking
    whether a morphism exists from the instance to the schema.

    - fit_score: How "close" the morphism is to being valid (0-1)
    - status: "assimilation" = morphism exists, "accommodation" = need extension
    - missing_slots: Parts of the morphism that are undefined
    - violations: Morphism constraint violations

    A perfect fit (score=1.0, status="assimilation") means the morphism
    is well-defined and satisfies all constraints.
    """

    source_data: dict[str, Any]
    target_schema: str
    fit_score: float
    status: str  # "assimilation" or "accommodation"
    missing_slots: list[str] = field(default_factory=list)
    violations: list[str] = field(default_factory=list)
    essential_missing: bool = False

    def is_valid_morphism(self) -> bool:
        """A valid morphism exists iff we can assimilate without essential gaps."""
        return self.status == "assimilation" and not self.essential_missing

    def to_categorical_morphism(
        self, source_obj: CategoricalObject, target_obj: CategoricalObject
    ) -> CategoricalMorphism:
        """Convert to a categorical morphism with appropriate classification."""
        if self.is_valid_morphism():
            morph_type = MorphismType.MONOMORPHISM  # Embedding into schema
        else:
            morph_type = MorphismType.RELATIONAL  # Partial/invalid

        return CategoricalMorphism(
            name=f"fit_{source_obj.name}_to_{target_obj.name}",
            source=source_obj,
            target=target_obj,
            rel_type="FITS_TO",
            morphism_type=morph_type,
            properties={
                "fit_score": self.fit_score,
                "status": self.status,
                "missing": self.missing_slots,
                "violations": self.violations,
            },
        )


class CategoricalSchemaView:
    """
    View a SchemaBuilder as a category.

    Objects: SchemaDefinitions (classes/types)
    Morphisms:
        - Inheritance (parent → child schemas)
        - Instantiation (schema → instance placeholders)
        - Fit morphisms (data → schema)

    This makes the implicit categorical structure of schema_builder explicit,
    enabling formal verification of schema operations.
    """

    def __init__(self, builder: SchemaBuilder) -> None:
        """
        Create a categorical view of a SchemaBuilder.

        Args:
            builder: A SchemaBuilder instance (with schemas registered)
        """
        self.builder = builder
        self._diagram: CategoricalDiagram | None = None
        self._bridge = CategoricalBridge()

    def to_categorical_diagram(self) -> CategoricalDiagram:
        """
        Convert the entire schema registry to a categorical diagram.

        Returns a diagram where:
        - Each schema is an object
        - Parent-child relationships are morphisms (inheritance)
        - Identity morphisms exist for all schemas
        """
        diagram = CategoricalDiagram(name="schema_category")

        # Create objects for each schema
        schema_objects: dict[str, CategoricalObject] = {}
        for name, schema in self.builder.registry.items():
            obj = CategoricalObject(
                name=name,
                label=name,
                properties={
                    "slots": list(schema.slots.keys()),
                    "slot_count": len(schema.slots),
                    "has_parent": schema.parent is not None,
                    "child_count": len(schema.children),
                },
            )
            schema_objects[name] = obj
            diagram.add_object(obj)

        # Create morphisms for inheritance relationships
        for name, schema in self.builder.registry.items():
            if schema.parent and schema.parent in schema_objects:
                # Inheritance morphism: parent → child
                # This is an epimorphism (child specializes parent)
                morph = CategoricalMorphism(
                    name=f"inherit_{schema.parent}_to_{name}",
                    source=schema_objects[schema.parent],
                    target=schema_objects[name],
                    rel_type="INHERITS_FROM",
                    morphism_type=MorphismType.EPIMORPHISM,
                    properties={
                        "inherited_slots": list(
                            self.builder.registry[schema.parent].slots.keys()
                        ),
                        "new_slots": [
                            s
                            for s in schema.slots.keys()
                            if s not in self.builder.registry[schema.parent].slots
                        ],
                    },
                )
                diagram.add_morphism(morph)

        # Add identity morphisms
        for name, obj in schema_objects.items():
            id_morph = CategoricalMorphism(
                name=f"id_{name}",
                source=obj,
                target=obj,
                rel_type="IDENTITY",
                is_identity=True,
                morphism_type=MorphismType.IDENTITY,
            )
            diagram.add_morphism(id_morph)

        # Detect paths and commutative structure
        for obj1 in diagram.objects:
            for obj2 in diagram.objects:
                if obj1 != obj2:
                    paths = diagram.find_paths(obj1, obj2)
                    diagram.paths.extend(paths)

        diagram.detect_commutative_squares()
        self._diagram = diagram

        return diagram

    def analyze_fit(self, schema_name: str, data: dict[str, Any]) -> FitMorphism:
        """
        Analyze a data-to-schema fit as a categorical morphism.

        This wraps evaluate_fit() and interprets the result categorically.
        """
        fit_result = self.builder.evaluate_fit(schema_name, data)

        return FitMorphism(
            source_data=data,
            target_schema=schema_name,
            fit_score=fit_result["fit_score"],
            status=fit_result["status"],
            missing_slots=fit_result["missing_slots"],
            violations=fit_result["violations"],
            essential_missing=fit_result["essential_missing"],
        )

    def to_prover9_clauses(self) -> dict[str, Any]:
        """
        Generate Prover9 clauses for the schema category.

        This enables formal verification of:
        - Inheritance transitivity
        - Schema compatibility
        - Fit morphism validity
        """
        if self._diagram is None:
            self.to_categorical_diagram()

        return self._diagram.to_prover9_clauses()

    def verify_inheritance_transitivity(self) -> dict[str, Any]:
        """
        Generate a proof goal for inheritance transitivity.

        If A inherits from B and B inherits from C, then A should
        have all of C's slots (transitivity of inheritance).
        """
        # Find chains of length 2
        chains: list[tuple[str, str, str]] = []

        for name, schema in self.builder.registry.items():
            if schema.parent:
                parent = self.builder.registry.get(schema.parent)
                if parent and parent.parent:
                    chains.append((parent.parent, schema.parent, name))

        proof_goals = []
        for grandparent, parent, child in chains:
            proof_goals.append(
                {
                    "description": f"Inheritance transitivity: {grandparent} → {parent} → {child}",
                    "conclusion": f"inherits_from({child}, {grandparent})",
                    "chain": [grandparent, parent, child],
                }
            )

        return {"chains_found": len(chains), "proof_goals": proof_goals}


class SchemaEvolutionFunctor:
    """
    Track schema evolution as a functor between categories.

    When evolve_schema() is called, it creates a functor F: C → C'
    where:
    - C is the category before evolution
    - C' is the category after evolution
    - F maps old objects to new objects
    - F maps old morphisms to new morphisms

    Accommodation = F adds new structure (not strictly structure-preserving)
    Assimilation = F is the identity functor (structure preserved exactly)

    This enables us to formally track what changes during schema evolution
    and verify that the changes are consistent.
    """

    def __init__(self, before: CategoricalDiagram, after: CategoricalDiagram) -> None:
        """
        Create a functor tracking schema evolution.

        Args:
            before: Diagram of schemas before evolution
            after: Diagram of schemas after evolution
        """
        self.before = before
        self.after = after
        self._object_map: dict[str, str] = {}
        self._morphism_map: dict[str, str] = {}
        self._new_objects: set[str] = set()
        self._new_morphisms: set[str] = set()

    def compute_mapping(self) -> None:
        """Compute the functor mapping between categories."""
        before_obj_names = {obj.name for obj in self.before.objects}
        after_obj_names = {obj.name for obj in self.after.objects}

        # Objects that exist in both = preserved by functor
        for name in before_obj_names:
            if name in after_obj_names:
                self._object_map[name] = name  # Identity on preserved objects

        # New objects = accommodation added these
        self._new_objects = after_obj_names - before_obj_names

        # Same for morphisms
        before_morph_names = {m.name for m in self.before.morphisms}
        after_morph_names = {m.name for m in self.after.morphisms}

        for name in before_morph_names:
            if name in after_morph_names:
                self._morphism_map[name] = name

        self._new_morphisms = after_morph_names - before_morph_names

    def is_identity_functor(self) -> bool:
        """Check if this is the identity functor (pure assimilation)."""
        return len(self._new_objects) == 0 and len(self._new_morphisms) == 0

    def summary(self) -> dict[str, Any]:
        """Get a summary of the evolution functor."""
        self.compute_mapping()

        return {
            "is_assimilation": self.is_identity_functor(),
            "preserved_objects": len(self._object_map),
            "new_objects": list(self._new_objects),
            "preserved_morphisms": len(self._morphism_map),
            "new_morphisms": list(self._new_morphisms),
            "functor_type": "identity" if self.is_identity_functor() else "extension",
        }


def demonstrate_integration() -> dict[str, Any]:
    """
    Demonstrate the schema-category integration.

    Returns results from:
    1. Building a schema hierarchy
    2. Converting to categorical diagram
    3. Generating Prover9 clauses
    4. Testing fit morphisms
    """
    # Import here to avoid circular imports
    from schema_builder import SchemaBuilder

    # Build schemas
    builder = SchemaBuilder()
    builder.bootstrap_meta_ontology()

    # Create categorical view
    view = CategoricalSchemaView(builder)
    diagram = view.to_categorical_diagram()

    # Generate Prover9 clauses
    prover9 = view.to_prover9_clauses()

    # Test a fit morphism
    test_data = {
        "concepts": ["Person", "Animal"],
        "relationships": ["IS_A"],
        "purpose": "Domain ontology",
    }
    fit = view.analyze_fit("Ontology", test_data)

    # Check inheritance transitivity
    transitivity = view.verify_inheritance_transitivity()

    return {
        "diagram_name": diagram.name,
        "object_count": len(diagram.objects),
        "morphism_count": len(diagram.morphisms),
        "inheritance_chains": transitivity["chains_found"],
        "sample_fit": {
            "valid_morphism": fit.is_valid_morphism(),
            "fit_score": fit.fit_score,
            "status": fit.status,
        },
        "prover9_premise_count": len(prover9["premises"]),
        "prover9_ready": prover9["mcp_logic_ready"],
    }


# Export for skill usage
__all__ = [
    "CategoricalSchemaView",
    "SchemaEvolutionFunctor",
    "FitMorphism",
    "demonstrate_integration",
]
