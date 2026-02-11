"""
Schema Builder Skill.

Implements the 'Schema of Schema' concept for cognitive framing,
supporting creation, instantiation, inference, and evolution of domain
knowledge frameworks within the RLM environment.
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger("graph_rlm.skills.schema_builder")


@dataclass
class SchemaDefinition:
    """Represents a cognitive framework (Class) with slots, defaults, and constraints."""

    concept_name: str
    slots: Dict[str, Any] = field(default_factory=dict)
    defaults: Dict[str, Any] = field(default_factory=dict)
    constraints: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    parent: Optional[str] = None
    children: List[str] = field(default_factory=list)


@dataclass
class SchemaInstance:
    """Represents a specific instance of a schema (Object)."""

    schema_name: str
    data: Dict[str, Any]
    inferred_data: Dict[str, Any] = field(default_factory=dict)
    fit_score: float = 0.0
    status: str = "unprocessed"


class SchemaBuilder:
    """
    Implements the 'Schema of Schema' concept.
    Capabilities: create_schema, instantiate, infer, evaluate_fit, evolve_schema.
    """

    def __init__(self) -> None:
        """Initializes the Schema registry."""
        self.registry: Dict[str, SchemaDefinition] = {}

    def create_schema(
        self,
        concept_name: str,
        slots: Dict[str, Any],
        defaults: Optional[Dict[str, Any]] = None,
        constraints: Optional[Dict[str, Dict[str, Any]]] = None,
        parent: Optional[str] = None,
    ) -> SchemaDefinition:
        """
        Creates a new schema with inheritance support.

        Args:
            concept_name: Mandatory name for the new concept.
            slots: Dictionary of properties and their types/weights.
            defaults: Default values to fill gaps during instantiation.
            constraints: Limits (min/max/regex) for specific slots.
            parent: Optional parent schema to inherit from.

        Returns:
            The created SchemaDefinition.
        """
        final_slots = {}
        final_defaults = {}
        final_constraints = {}

        if parent and parent in self.registry:
            p = self.registry[parent]
            final_slots.update(p.slots)
            final_defaults.update(p.defaults)
            final_constraints.update(p.constraints)
            if concept_name not in p.children:
                p.children.append(concept_name)

        final_slots.update(slots)
        if defaults:
            final_defaults.update(defaults)
        if constraints:
            final_constraints.update(constraints)

        schema = SchemaDefinition(
            concept_name=concept_name,
            slots=final_slots,
            defaults=final_defaults,
            constraints=final_constraints,
            parent=parent,
        )
        self.registry[concept_name] = schema
        return schema

    def instantiate(self, schema_name: str, data: Dict[str, Any]) -> SchemaInstance:
        """
        Match data to a schema (Recognition).

        Args:
            schema_name: Name of the schema to use for instantiation.
            data: The raw data to wrap.

        Returns:
            A SchemaInstance object.
        """
        if schema_name not in self.registry:
            raise ValueError(f"Schema '{schema_name}' not found in registry.")
        return SchemaInstance(schema_name=schema_name, data=data)

    def infer(self, instance: SchemaInstance) -> SchemaInstance:
        """
        Fill gaps in data using schema default values (Gap Filling).

        Args:
            instance: The SchemaInstance to enrich.

        Returns:
            The enriched SchemaInstance with inferred_data populated.
        """
        schema = self.registry[instance.schema_name]
        inferred = instance.data.copy()
        for slot in schema.slots:
            if slot not in inferred or inferred[slot] is None:
                if slot in schema.defaults:
                    inferred[slot] = schema.defaults[slot]
        instance.inferred_data = inferred
        return instance

    def evaluate_fit(self, schema_name: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Return match score and identify Assimilation vs Accommodation.

        Supports weighted slots for salience:
        - Essential properties (weight ~0.9): Missing crashes score
        - Accidental properties (weight ~0.1): Missing barely affects score

        Slot format can be:
        - Simple: {"wings": "bool"} -> default weight 0.5
        - Weighted: {"wings": {"type": "bool", "weight": 0.9}}

        Args:
            schema_name: The schema to check against.
            data: The data to evaluate.

        Returns:
            A dictionary containing fit_score, status, and violation details.
        """
        if schema_name not in self.registry:
            return {"error": f"Schema {schema_name} not found"}

        schema = self.registry[schema_name]
        missing_slots = []
        violations = []
        weighted_match = 0.0
        total_weight = 0.0

        for slot, slot_def in schema.slots.items():
            # Extract weight (default 0.5 if not specified)
            if isinstance(slot_def, dict) and "weight" in slot_def:
                weight = float(slot_def["weight"])
            else:
                weight = 0.5  # Default: neutral importance

            total_weight += weight

            if slot in data:
                weighted_match += weight
                val = data[slot]
                if slot in schema.constraints:
                    c = schema.constraints[slot]
                    if "min" in c:
                        try:
                            if val < c["min"]:
                                violations.append(f"{slot} < {c['min']}")
                                weighted_match -= weight * 0.5
                        except TypeError:
                            logger.warning(
                                "Min constraint skipped: %s not comparable to %s",
                                val,
                                c["min"],
                            )
                    if "max" in c:
                        try:
                            if val > c["max"]:
                                violations.append(f"{slot} > {c['max']}")
                                weighted_match -= weight * 0.5
                        except TypeError:
                            logger.warning(
                                "Max constraint skipped: %s not comparable to %s",
                                val,
                                c["max"],
                            )
            else:
                missing_slots.append(slot)

        fit_score = weighted_match / total_weight if total_weight > 0 else 1.0
        fit_score = max(0.0, min(1.0, fit_score))  # Clamp to [0, 1]

        # Check if any essential slot (weight >= 0.8) is missing
        essential_missing = any(
            (
                isinstance(schema.slots.get(s), dict)
                and float(schema.slots[s].get("weight", 0.5)) >= 0.8
            )
            for s in missing_slots
        )

        # Status logic
        if essential_missing:
            status = "accommodation"  # Essential property missing = must restructure
        elif not violations and fit_score >= 0.7:
            status = "assimilation"
        else:
            status = "accommodation"

        return {
            "fit_score": round(fit_score, 3),
            "status": status,
            "missing_slots": missing_slots,
            "violations": violations,
            "essential_missing": essential_missing,
        }

    def evolve_schema(self, schema_name: str, new_data: Dict[str, Any]) -> str:
        """
        Update schema via assimilation or accommodation based on new data.

        Args:
            schema_name: The schema to evolve.
            new_data: The new information to incorporate.

        Returns:
            A string describing the evolution outcome.
        """
        eval_res = self.evaluate_fit(schema_name, new_data)
        schema = self.registry[schema_name]

        if eval_res["status"] == "accommodation":
            added = []
            for k, v in new_data.items():
                if k not in schema.slots:
                    schema.slots[k] = type(v).__name__
                    added.append(k)
            return f"Accommodated: Added slots {added}"
        return "Assimilated: No structural changes."

    def bootstrap_meta_ontology(self) -> None:
        """Initialize the meta-ontology that guides all domain schema creation."""
        # Root: The concept of Ontology itself
        self.create_schema(
            "Ontology",
            slots={
                "concepts": "list",
                "relationships": "list",
                "axioms": "list",
                "vocabulary": "dict",
                "purpose": "str",
                "version": "str",
            },
            defaults={
                "version": "1.0.0",
                "concepts": [],
                "relationships": [],
                "axioms": [],
            },
        )
        # Core types
        self.create_schema(
            "Class", parent="Ontology", slots={"label": "str", "description": "str"}
        )
        self.create_schema(
            "Property", parent="Ontology", slots={"domain": "str", "range": "str"}
        )
        self.create_schema(
            "Individual", parent="Ontology", slots={"type": "str", "values": "dict"}
        )
        self.create_schema(
            "Axiom", parent="Ontology", slots={"expression": "str", "logic_type": "str"}
        )
        # Specialized types
        self.create_schema(
            "ObjectProperty",
            parent="Property",
            slots={"inverse_of": "str", "is_transitive": "bool"},
        )
        self.create_schema("DataProperty", parent="Property", slots={"datatype": "str"})
        self.create_schema(
            "SubclassOf", parent="Axiom", slots={"sub": "str", "super": "str"}
        )
        self.create_schema(
            "DisjointWith", parent="Axiom", slots={"class_a": "str", "class_b": "str"}
        )

    def measure_compression(
        self, schema_name: str, instance_count: int = 1
    ) -> Dict[str, Any]:
        """
        Measure how efficiently a schema compresses information.

        Args:
            schema_name: Name of schema to measure.
            instance_count: How many instances this schema represents.

        Returns:
            A dictionary with slot_count, compression_ratio, and efficiency.
        """
        schema = self.registry.get(schema_name)
        if not schema:
            return {"error": f"Schema {schema_name} not found"}

        slot_count = len(schema.slots)
        # Compression = instances covered per slot (higher = more general)
        compression_ratio = instance_count / slot_count if slot_count > 0 else 0

        # Efficiency based on defaults
        default_coverage = len(schema.defaults) / slot_count if slot_count > 0 else 0

        return {
            "schema": schema_name,
            "slot_count": slot_count,
            "instance_count": instance_count,
            "compression_ratio": round(compression_ratio, 2),
            "default_coverage": round(default_coverage, 2),
            "efficiency": round((compression_ratio + default_coverage) / 2, 2),
        }


def schema_builder() -> SchemaBuilder:
    """Entry point to create a new SchemaBuilder instance."""
    return SchemaBuilder()
