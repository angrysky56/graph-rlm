"""
Skills persistence system - enables AI to save and reuse learned patterns.

Implements Anthropic's "skills accumulation" pattern where agents can:
1. Write code to solve a problem
2. Save that code as a reusable skill
3. Import and reuse skills in future tasks

This creates a growing library of higher-level capabilities.
Refactored to use FalkorDB.
"""

import ast
import json
from pathlib import Path
from typing import Any, cast, Optional

from graph_rlm.backend.src.core.db import db
from graph_rlm.backend.src.core.logger import get_logger

logger = get_logger("graph_rlm.skills")


class SkillsManager:
    """
    Manages a directory of reusable skills (Python functions) in FalkorDB.
    """

    def __init__(self, skills_dir: Path) -> None:
        """
        Initialize skills manager.
        Args:
            skills_dir: Directory containing skill files (local cache of the source)
        """
        self.db = db
        self.skills_dir = skills_dir
        self.skills_dir.mkdir(parents=True, exist_ok=True)
        # Ensure __init__ exists for import
        (self.skills_dir / "__init__.py").touch(exist_ok=True)
        # We can sync on start if we assume disk is source of truth?
        # For now, we trust DB, but if empty, we might load from disk.
        self.sync_from_disk()

    def sync_from_disk(self) -> None:
        """
        Sync skills from disk to database.
        Scans *.py files and MERGES them into the Graph.
        """
        count = 0
        for file_path in self.skills_dir.glob("*.py"):
            if file_path.name == "__init__.py":
                continue

            try:
                code = file_path.read_text(encoding="utf-8")
                # Parse
                tree = ast.parse(code)
                func_def = next(
                    (node for node in ast.walk(tree) if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))),
                    None,
                )
                if not func_def:
                    continue

                name = file_path.stem
                function_name = func_def.name
                description = ast.get_docstring(func_def) or ""

                # Upsert into Graph
                cypher = """
                MERGE (s:Skill {name: $name})
                SET s.code = $code,
                    s.description = $desc,
                    s.function_name = $func,
                    s.updated_at = timestamp()
                """
                # Versioning is implicit: latest is what's in the node.
                # If we want history, we'd create linked list of :VERSION nodes.
                # Keeping it simple for MVP: One active version.

                self.db.query(cypher, {
                    "name": name,
                    "code": code,
                    "desc": description,
                    "func": function_name
                })
                count += 1
            except Exception as e:
                logger.error(f"Failed to sync skill {file_path.name}: {e}")

        if count > 0:
            logger.info(f"Synced {count} skills from disk to FalkorDB.")

    def save_skill(
        self,
        name: str,
        code: str,
        description: str | None = None,
        tags: list[str] | None = None,
    ) -> str:
        """
        Save a skill function to the skills library.
        """
        try:
            tree = ast.parse(code)
            func_def = next(
                (node for node in ast.walk(tree) if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))),
                None,
            )
            if func_def is None:
                raise ValueError("Code must contain a function definition")
            function_name = func_def.name
        except SyntaxError as e:
            raise ValueError(f"Invalid Python syntax: {e}")

        # Update Graph
        cypher = """
        MERGE (s:Skill {name: $name})
        SET s.code = $code,
            s.description = $desc,
            s.function_name = $func,
            s.tags = $tags,
            s.version = COALESCE(s.version, 0) + 1,
            s.updated_at = timestamp()
        RETURN s.version
        """
        res = self.db.query(cypher, {
            "name": name,
            "code": code,
            "desc": description or "",
            "func": function_name,
            "tags": tags or []
        })

        # Write to disk
        try:
            skill_file = self.skills_dir / f"{name}.py"
            skill_file.write_text(code, encoding="utf-8")
        except Exception as e:
            logger.error(f"Failed to write skill to disk: {e}")

        return name

    def list_skills(self) -> dict[str, dict[str, Any]]:
        """
        List all available skills with metadata.
        """
        cypher = "MATCH (s:Skill) RETURN s"
        results = self.db.query(cypher) or []

        skills = {}
        for row in results:
            if not row:
                continue
            # Handle list vs dict return from client
            node = row[0] if isinstance(row, list) else row.get('s')
            if not node:
                 continue

            props = node.properties if hasattr(node, 'properties') else node
            if not isinstance(props, dict):
                 continue

            skills[props.get('name', 'unknown')] = {
                "description": props.get('description'),
                "tags": props.get('tags', []),
                "function_name": props.get('function_name'),
                "version": props.get('version', 1)
            }
        return skills

    def get_skill(self, name: str) -> dict[str, Any] | None:
        """
        Get the code and metadata for a specific skill.
        """
        cypher = "MATCH (s:Skill {name: $name}) RETURN s"
        results = self.db.query(cypher, {"name": name})

        if not results:
            return None

        row = results[0]
        if not row:
            return None

        node = row[0] if isinstance(row, list) else row.get('s')
        if not node:
            return None

        props = node.properties if hasattr(node, 'properties') else node
        if not isinstance(props, dict):
             return None

        return {
            "name": props.get('name'),
            "code": props.get('code'),
            "description": props.get('description'),
            "function_name": props.get('function_name'),
            "tags": props.get('tags', []),
            "version": props.get('version', 1),
            "latest": True
        }

    def get_import_statement(self, name: str) -> str:
        """
        Get the Python import statement for a skill.
        Ensures the file exists on disk first.
        """
        skill = self.get_skill(name)
        if not skill:
            raise ValueError(f"Skill '{name}' not found")

        # Write to disk to ensure importable
        skill_file = self.skills_dir / f"{name}.py"
        if not skill_file.exists() or skill_file.read_text() != skill["code"]:
             skill_file.write_text(skill["code"])

        return f"from skills_dir.{name} import {skill['function_name']}"


# Global skills manager instance
_global_skills_manager: SkillsManager | None = None

def get_skills_manager() -> SkillsManager:
    """
    Get or create the global skills manager instance.
    """
    global _global_skills_manager

    if _global_skills_manager is None:
        # Resolve skills directory relative to backend root or workspace
        # Let's put it in backend/src/skills_dir to be importable as a module if we add __init__?
        # Or better, a dedicated skills_dir alongside src?
        # User repo has graph_rlm/backend/skills_dir already in previous implementations?
        # Let's use: graph_rlm/backend/skills_cache

        backend_root = Path(__file__).parent.parent.parent # mcp_integration -> src -> backend -> root
        skills_dir = backend_root / "skills_dir"

        _global_skills_manager = SkillsManager(skills_dir)

    return _global_skills_manager
