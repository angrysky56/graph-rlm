"""
Skills persistence system - enables AI to save and reuse learned patterns.

Implements Anthropic's "skills accumulation" pattern where agents can:
1. Write code to solve a problem
2. Save that code as a reusable skill
3. Import and reuse skills in future tasks

This creates a growing library of higher-level capabilities.
"""

import ast
import json
from pathlib import Path
from typing import Any, cast

from .database import DatabaseManager, get_db_manager


class SkillsManager:
    """
    Manages a directory of reusable skills (Python functions) in a database.
    """

    def __init__(self, db_manager: DatabaseManager, skills_dir: Path) -> None:
        """
        Initialize skills manager.

        Args:
            db_manager: Database manager instance
            skills_dir: Directory containing skill files
        """
        self.db = db_manager
        self.skills_dir = skills_dir
        self.skills_dir.mkdir(parents=True, exist_ok=True)
        self.sync_from_disk()

    def sync_from_disk(self) -> None:
        """
        Sync skills from disk to database.

        Scans the skills directory for Python files, parses them,
        and updates the database index.
        """
        import logging

        logger = logging.getLogger(__name__)

        # Ensure __init__.py exists
        init_file = self.skills_dir / "__init__.py"
        if not init_file.exists():
            init_file.touch()

        count = 0
        for file_path in self.skills_dir.glob("*.py"):
            if file_path.name == "__init__.py":
                continue

            try:
                code = file_path.read_text(encoding="utf-8")

                # Parse to validate and extract metadata
                tree = ast.parse(code)
                func_def = next(
                    (
                        node
                        for node in ast.walk(tree)
                        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                    ),
                    None,
                )

                if not func_def:
                    logger.warning(f"Skipping {file_path.name}: No function found")
                    continue

                name = file_path.stem
                function_name = func_def.name
                description = ast.get_docstring(func_def)

                # Check if skill exists and if code changed
                cursor = self.db.execute(
                    "SELECT code, version FROM skills WHERE name = ? AND latest = TRUE",
                    (name,),
                )
                row = cursor.fetchone()

                if row:
                    if row["code"].strip() == code.strip():
                        continue  # No change
                    version = row["version"] + 1
                else:
                    version = 1

                # Update/Insert
                self.db.execute(
                    "UPDATE skills SET latest = FALSE WHERE name = ?", (name,)
                )

                self.db.execute(
                    """
                    INSERT INTO skills (name, version, code, description, tags, function_name, latest)
                    VALUES (?, ?, ?, ?, ?, ?, TRUE)
                    """,
                    (
                        name,
                        version,
                        code,
                        description,
                        "[]",
                        function_name,
                    ),  # TODO: Extract tags?
                )
                count += 1

            except Exception as e:
                logger.error(f"Failed to sync skill {file_path.name}: {e}")

        if count > 0:
            self.db.commit()
            logger.info(f"Synced {count} skills from disk.")

    def save_skill(
        self,
        name: str,
        code: str,
        description: str | None = None,
        tags: list[str] | None = None,
    ) -> int:
        """
        Save a skill function to the skills library.

        Args:
            name: Name for the skill
            code: Python function code
            description: Optional description of what the skill does
            tags: Optional tags for categorization

        Returns:
            The ID of the saved skill
        """
        try:
            tree = ast.parse(code)
            func_def = next(
                (
                    node
                    for node in ast.walk(tree)
                    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                ),
                None,
            )
            if func_def is None:
                raise ValueError("Code must contain a function definition")
            function_name = func_def.name
        except SyntaxError as e:
            raise ValueError(f"Invalid Python syntax: {e}")

        cursor = self.db.execute(
            "SELECT MAX(version) FROM skills WHERE name = ?", (name,)
        )
        max_version = cursor.fetchone()[0]
        new_version = (max_version or 0) + 1

        self.db.execute("UPDATE skills SET latest = FALSE WHERE name = ?", (name,))

        tags_json = json.dumps(tags or [])
        cursor = self.db.execute(
            """
            INSERT INTO skills (name, version, code, description, tags, function_name, latest)
            VALUES (?, ?, ?, ?, ?, ?, TRUE)
            """,
            (name, new_version, code, description, tags_json, function_name),
        )
        self.db.commit()

        # Write to disk immediately ensuring source of truth matches
        try:
            skill_file = self.skills_dir / f"{name}.py"
            skill_file.write_text(code, encoding="utf-8")
        except Exception as e:
            # Log warning but don't fail the DB transaction?
            # Actually, we should probably treat disk failure as critical for transparency
            import logging

            logger = logging.getLogger(__name__)
            logger.error(f"Failed to write skill to disk: {e}")

        return cast(int, cursor.lastrowid)

    def list_skills(self) -> dict[str, dict[str, Any]]:
        """
        List all available skills with metadata.

        Returns:
            Dictionary mapping skill names to their metadata
        """
        cursor = self.db.execute(
            "SELECT name, description, tags, function_name, version FROM skills WHERE latest = TRUE"
        )
        skills = {}
        for row in cursor.fetchall():
            skills[row["name"]] = {
                "description": row["description"],
                "tags": json.loads(row["tags"]),
                "function_name": row["function_name"],
                "version": row["version"],
            }
        return skills

    def get_skill(self, name: str, version: int | None = None) -> dict[str, Any] | None:
        """
        Get the code and metadata for a specific skill.

        Args:
            name: Skill name
            version: Optional skill version (defaults to latest)

        Returns:
            Dictionary with skill data or None if not found
        """
        if version:
            cursor = self.db.execute(
                "SELECT * FROM skills WHERE name = ? AND version = ?", (name, version)
            )
        else:
            cursor = self.db.execute(
                "SELECT * FROM skills WHERE name = ? AND latest = TRUE", (name,)
            )

        row = cursor.fetchone()
        if row:
            return {
                "id": row["id"],
                "name": row["name"],
                "version": row["version"],
                "code": row["code"],
                "description": row["description"],
                "tags": json.loads(row["tags"]),
                "function_name": row["function_name"],
                "latest": row["latest"],
            }
        return None

    def search_skills(self, query: str) -> dict[str, dict[str, Any]]:
        """
        Search skills by name, description, or tags.

        Args:
            query: Search query

        Returns:
            Dictionary of matching skills
        """
        query_lower = f"%{query.lower()}%"
        cursor = self.db.execute(
            """
            SELECT name, description, tags, function_name, version FROM skills
            WHERE latest = TRUE AND (LOWER(name) LIKE ? OR LOWER(description) LIKE ? OR LOWER(tags) LIKE ?)
            """,
            (query_lower, query_lower, query_lower),
        )
        skills = {}
        for row in cursor.fetchall():
            skills[row["name"]] = {
                "description": row["description"],
                "tags": json.loads(row["tags"]),
                "function_name": row["function_name"],
                "version": row["version"],
            }
        return skills

    def delete_skill(self, name: str, version: int | None = None) -> None:
        """
        Delete a skill from the library.

        Args:
            name: Skill name
            version: Optional version to delete (deletes all versions if None)
        """
        if version:
            self.db.execute(
                "DELETE FROM skills WHERE name = ? AND version = ?", (name, version)
            )
        else:
            self.db.execute("DELETE FROM skills WHERE name = ?", (name,))
        self.db.commit()

    def get_import_statement(self, name: str, version: int | None = None) -> str:
        """
        Get the Python import statement for a skill.

        Args:
            name: Skill name
            version: Optional skill version

        Returns:
            Import statement string
        """
        skill = self.get_skill(name, version)
        if not skill:
            raise ValueError(f"Skill '{name}' not found")

        skills_dir = self.skills_dir
        skills_dir.mkdir(exist_ok=True)
        (skills_dir / "__init__.py").touch(exist_ok=True)

        skill_file = skills_dir / f"{name}.py"
        skill_file.write_text(skill["code"])

        return f"from skills.{name} import {skill['function_name']}"


# Global skills manager instance
_global_skills_manager: SkillsManager | None = None


def get_skills_manager(db_path: str | Path | None = None) -> SkillsManager:
    """
    Get or create the global skills manager instance.

    Args:
        db_path: Optional path to the database file

    Returns:
        Shared SkillsManager instance
    """
    global _global_skills_manager

    if _global_skills_manager is None:
        import os

        db_manager = get_db_manager(db_path)

        # Resolve skills directory relative to this file
        # graph_rlm/backend/src/mcp_integration/skills.py -> .../backend/skills_dir
        kb_dir = Path(__file__).parent.parent.parent
        skills_dir = kb_dir / "skills_dir"

        _global_skills_manager = SkillsManager(db_manager, skills_dir)

    return _global_skills_manager
