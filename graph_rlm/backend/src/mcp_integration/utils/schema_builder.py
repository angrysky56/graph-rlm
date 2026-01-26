from dataclasses import dataclass, field
from typing import Any


@dataclass
class SchemaDefinition:
    """Represents a cognitive framework (Class) with slots, defaults, and constraints."""

    concept_name: str
    slots: dict[str, Any] = field(default_factory=dict)
    defaults: dict[str, Any] = field(default_factory=dict)
    constraints: dict[str, dict[str, Any]] = field(default_factory=dict)
    parent: str | None = None
    children: list[str] = field(default_factory=list)


@dataclass
class SchemaInstance:
    """Represents a specific instance of a schema (Object)."""

    schema_name: str
    data: dict[str, Any]
    inferred_data: dict[str, Any] = field(default_factory=dict)
    fit_score: float = 0.0
    status: str = "unprocessed"


class SchemaBuilder:
    """
    Implements the 'Schema of Schema' concept.
    Capabilities: create_schema, instantiate, infer, evaluate_fit, evolve_schema.
    """

    def __init__(self):
        self.registry: dict[str, SchemaDefinition] = {}

    def create_schema(
        self,
        concept_name: str,
        slots: dict[str, Any],
        defaults: dict[str, Any] | None = None,
        constraints: dict[str, dict[str, Any]] | None = None,
        parent: str | None = None,
    ) -> SchemaDefinition:
        """Creates a new schema with inheritance support."""
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

    def instantiate(self, schema_name: str, data: dict[str, Any]) -> SchemaInstance:
        """Match data to a schema (Recognition)."""
        if schema_name not in self.registry:
            raise ValueError(f"Schema {schema_name} not found.")
        return SchemaInstance(schema_name=schema_name, data=data)

    def infer(self, instance: SchemaInstance) -> SchemaInstance:
        """Fill gaps in data using schema default values (Gap Filling)."""
        schema = self.registry[instance.schema_name]
        inferred = instance.data.copy()
        for slot in schema.slots:
            if slot not in inferred or inferred[slot] is None:
                if slot in schema.defaults:
                    inferred[slot] = schema.defaults[slot]
        instance.inferred_data = inferred
        return instance

    def evaluate_fit(
        self, schema_name: str, data: dict[str, object]
    ) -> dict[str, object]:
        """
        Return match score and identify Assimilation vs Accommodation.

        Supports weighted slots for salience:
        - Essential properties (weight ~0.9): Missing crashes score
        - Accidental properties (weight ~0.1): Missing barely affects score

        Slot format can be:
        - Simple: {"wings": "bool"} → default weight 0.5
        - Weighted: {"wings": {"type": "bool", "weight": 0.9}}
        """
        schema = self.registry[schema_name]
        missing_slots = []
        violations = []
        weighted_match = 0.0
        total_weight = 0.0

        for slot, slot_def in schema.slots.items():
            # Extract weight (default 0.5 if not specified)
            if isinstance(slot_def, dict) and "weight" in slot_def:
                weight = slot_def["weight"]
            else:
                weight = 0.5  # Default: neutral importance

            total_weight += weight

            if slot in data:
                weighted_match += weight
                val = data[slot]
                if slot in schema.constraints:
                    c = schema.constraints[slot]
                    if "min" in c and val < c["min"]:
                        violations.append(f"{slot} < {c['min']}")
                        weighted_match -= weight * 0.5  # Penalty
                    if "max" in c and val > c["max"]:
                        violations.append(f"{slot} > {c['max']}")
                        weighted_match -= weight * 0.5  # Penalty
            else:
                missing_slots.append(slot)
                # High-weight missing slot is a bigger problem

        fit_score = weighted_match / total_weight if total_weight > 0 else 1.0
        fit_score = max(0.0, min(1.0, fit_score))  # Clamp to [0, 1]

        # Check if any essential slot (weight >= 0.8) is missing
        essential_missing = any(
            (
                isinstance(schema.slots.get(s), dict)
                and schema.slots[s].get("weight", 0.5) >= 0.8
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

    def evolve_schema(self, schema_name: str, new_data: dict[str, Any]) -> str:
        """Update schema via assimilation or accommodation based on new data."""
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

    def create_domain_concept(
        self,
        name: str,
        concept_type: str,
        slots: dict | None = None,
        defaults: dict | None = None,
    ) -> "SchemaDefinition":
        """
        Create a new domain concept using the meta-ontology as template.

        Args:
            name: Name of the new concept (e.g., "Person", "hasAge")
            concept_type: Meta-ontology type ("Class", "ObjectProperty", "Axiom", etc.)
            slots: Additional domain-specific slots
            defaults: Default values for slots

        Returns:
            The created schema, inheriting from the meta-ontology type.
        """
        if concept_type not in self.registry:
            raise ValueError(
                f"Unknown concept type: {concept_type}. "
                f"Call bootstrap_meta_ontology() first or use: {list(self.registry.keys())}"
            )
        return self.create_schema(
            name, parent=concept_type, slots=slots or {}, defaults=defaults or {}
        )

    def measure_compression(self, schema_name: str, instance_count: int = 1) -> dict:
        """
        Measure how efficiently a schema compresses information.

        High compression = few slots represent many instances.
        Low compression = too many slots for too few instances.

        Args:
            schema_name: Name of schema to measure
            instance_count: How many instances this schema represents

        Returns:
            dict with slot_count, compression_ratio, efficiency
        """
        schema = self.registry.get(schema_name)
        if not schema:
            return {"error": f"Schema {schema_name} not found"}

        slot_count = len(schema.slots)
        # Compression = instances covered per slot (higher = more general)
        compression_ratio = instance_count / slot_count if slot_count > 0 else 0

        # Efficiency based on defaults (more defaults = less info needed per instance)
        default_coverage = len(schema.defaults) / slot_count if slot_count > 0 else 0

        return {
            "schema": schema_name,
            "slot_count": slot_count,
            "instance_count": instance_count,
            "compression_ratio": round(compression_ratio, 2),
            "default_coverage": round(default_coverage, 2),
            "efficiency": round((compression_ratio + default_coverage) / 2, 2),
        }


def schema_builder():
    return SchemaBuilder()
