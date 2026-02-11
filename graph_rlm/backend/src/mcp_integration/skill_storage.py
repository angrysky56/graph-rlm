"""
Skills persistence system - enables AI to save and reuse learned patterns.

Implements Anthropic's "skills accumulation" pattern where agents can:
1. Write code to solve a problem
2. Save that code as a reusable skill
3. Import and reuse skills in future tasks

This creates a growing library of higher-level capabilities.
Refactored to use FalkorDB.

Axiom Archival Policy:
Problematic or redundant axioms should be moved to the '_disabled' subdirectory
within the axioms_dir. The system's sync logic is non-recursive and will
automatically ignore these files.
"""

import ast
import re
from pathlib import Path
from typing import Any, Dict, List

from graph_rlm.backend.src.core.db import db
from graph_rlm.backend.src.core.llm import llm
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
        # NOTE: sync_from_disk() removed from __init__ to avoid loop conflicts.
        # It should be called explicitly via await sync_from_disk() during startup.

    async def sync_from_disk(self) -> None:
        """
        Sync skills from disk to database.
        Scans *.py files AND directories with SKILL.md.
        """
        count = 0

        # 1. Scan for Python Skills (*.py)
        # We iterate everything in the root
        for item in self.skills_dir.iterdir():
            if item.name == "__init__.py" or item.name.startswith("__"):
                continue

            # Case A: Python File (Standard Skill)
            if item.is_file() and item.suffix == ".py":
                if await self._sync_python_skill(item):
                    count += 1

            # Case B: Folder-Based Skill (Instructional/Complex)
            elif item.is_dir():
                skill_md = item / "SKILL.md"
                if skill_md.exists():
                    if await self._sync_instructional_skill(item, skill_md):
                        count += 1

        if count > 0:
            logger.info("Synced %d skills from disk to FalkorDB.", count)

    async def _sync_python_skill(self, file_path: Path) -> bool:
        try:
            code = file_path.read_text(encoding="utf-8")
            # Parse
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
                return False

            name = file_path.stem
            function_name = func_def.name
            description = ast.get_docstring(func_def) or ""

            # Extract tags from docstring or comments
            tags = []
            # Heuristic 1: Look for "Tags: tag1, tag2" in docstring
            tag_match = re.search(r"Tags:\s*([^\n]+)", description, re.IGNORECASE)
            if tag_match:
                tags = [t.strip() for t in tag_match.group(1).split(",")]

            # Heuristic 2: Look for "# Tags: tag1, tag2" in the code comments
            comment_match = re.search(r"#\s*Tags:\s*([^\n]+)", code, re.IGNORECASE)
            if comment_match:
                comment_tags = [t.strip() for t in comment_match.group(1).split(",")]
                tags.extend([t for t in comment_tags if t not in tags])

            # Heuristic 3: Infer from filename if starts with axiom_
            if not tags and name.startswith("axiom_"):
                if "physics" in name.lower():
                    tags.append("physics")
                elif "coding" in name.lower():
                    tags.append("coding")
                elif "math" in name.lower() or "logic" in name.lower():
                    tags.append("math")
                # If still no tags, we leave it empty (SheafMonitor will treat as general for general tasks)

            # Generate embedding for semantic search
            text_to_embed = f"{name}: {description}" if description else name
            # Properly await the async embedding call
            try:
                vec = await llm.get_embedding(text_to_embed)
            except Exception as e:  # pylint: disable=broad-except
                logger.warning("Failed to generate embedding for skill %s: %s", name, e)
                vec = None

            # Upsert into Graph (with health tracking fields)
            cypher = """
            MERGE (s:Skill {name: $name})
            ON CREATE SET s.error_count = 0, s.success_count = 0
            SET s.code = $code,
                s.description = $desc,
                s.function_name = $func,
                s.tags = $tags,
                s.type = 'python',
                s.updated_at = timestamp()
            """
            if vec:
                cypher += ", s.embedding = vecf32($vec)"

            self.db.query(
                cypher,
                {
                    "name": name,
                    "code": code,
                    "desc": description,
                    "func": function_name,
                    "tags": tags,
                    "vec": vec,
                },
            )
            return True
        except Exception as e:  # pylint: disable=broad-except
            logger.error("Failed to sync python skill %s: %s", file_path.name, e)
            return False

    async def _sync_instructional_skill(self, dir_path: Path, md_path: Path) -> bool:
        try:
            content = md_path.read_text(encoding="utf-8")

            # Simple Frontmatter Parser
            # We look for leading --- ... ---
            match = re.match(r"^---\s*\n(.*?)\n---\s*\n", content, re.DOTALL)
            frontmatter = {}
            if match:
                fm_text = match.group(1)
                # Simple key: value parsing
                for line in fm_text.splitlines():
                    if ":" in line:
                        key, val = line.split(":", 1)
                        # Fix for list parsing in frontmatter
                        val_str = val.strip()
                        if val_str.startswith("[") and val_str.endswith("]"):
                            frontmatter[key.strip()] = [
                                v.strip() for v in val_str[1:-1].split(",") if v.strip()
                            ]
                        else:
                            frontmatter[key.strip()] = val_str

            name = frontmatter.get("name") or dir_path.name
            description = frontmatter.get("description") or "Instructional Skill"
            tags = frontmatter.get("tags") or []

            # Upsert into Graph
            # We store the FULL markdown content as 'code' so the agent reads the manual
            cypher = """
            MERGE (s:Skill {name: $name})
            SET s.code = $code,
                s.description = $desc,
                s.function_name = $func,
                s.tags = $tags,
                s.type = 'instructional',
                s.updated_at = timestamp()
            """
            self.db.query(
                cypher,
                {
                    "name": name,
                    "code": content,
                    "desc": description,
                    "func": "__instruction__",  # Marker
                    "tags": tags,
                },
            )
            return True
        except Exception as e:  # pylint: disable=broad-except
            logger.error("Failed to sync instructional skill %s: %s", dir_path.name, e)
            return False

    async def install_skill(self, source: str) -> str:
        """
        Install a skill from a remote source (Git URL or HTTP URL).
        mimics 'repo2skill' functionality.
        """
        import subprocess

        # 1. Handle Git URL
        if source.endswith(".git") or "github.com" in source:
            try:
                skill_name = source.split("/")[-1].replace(".git", "")
                skill_path = self.skills_dir / skill_name

                if skill_path.exists():
                    logger.warning("Skill %s already exists. Updating...", skill_name)
                    # Perform git pull
                    subprocess.run(
                        ["git", "-C", str(skill_path), "pull"],
                        check=True,
                        capture_output=True,
                    )
                else:
                    logger.info("Cloning skill from %s...", source)
                    subprocess.run(
                        ["git", "clone", source, str(skill_path)],
                        check=True,
                        capture_output=True,
                    )

                # Check for SKILL.md
                skill_md = skill_path / "SKILL.md"
                if not skill_md.exists():
                    # Attempt to auto-generate or use README
                    readme = skill_path / "README.md"
                    if readme.exists():
                        content = readme.read_text()
                        # Create minimal SKILL.md
                        (skill_path / "SKILL.md").write_text(
                            f"---\nname: {skill_name}\ndescription: Auto-imported skill from {source}\n---\n\n{content}"
                        )

                # Sync
                await self.sync_from_disk()
                return skill_name
            except Exception as e:
                logger.error("Failed to install skill from git: %s", e)
                raise

        # 2. Handle Single File URL (assumed to be raw content or downloadable)
        else:
            try:
                skill_name = source.split("/")[-1].replace(".py", "")
                skill_file = self.skills_dir / f"{skill_name}.py"

                if skill_file.exists():
                    logger.warning(
                        "Skill file %s already exists. Overwriting...", skill_name
                    )

                logger.info("Downloading skill from %s...", source)
                # Use curl to download
                subprocess.run(
                    ["curl", "-L", "-o", str(skill_file), source],
                    check=True,
                    capture_output=True,
                )

                # Validate it's python
                if skill_file.read_text().strip():
                    return (
                        f"Skill '{skill_name}' downloaded successfully to {skill_file}"
                    )

                skill_file.unlink()
                raise ValueError("Downloaded file is empty")

            except Exception as e:
                logger.error("Failed to download skill file: %s", e)
                raise RuntimeError(f"Skill download failed: {e}") from e

    async def save_skill(
        self,
        name: str,
        code: str,
        description: str | None = None,
        tags: list[str] | None = None,
    ) -> str:
        """
        Save a skill function to the skills library.
        """
        # Sanitize and limit name length for OS compatibility
        name = re.sub(r"[^a-zA-Z0-9_]", "_", name)[:100]
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
            raise ValueError(f"Invalid Python syntax: {e}") from e

        # Generate embedding
        try:
            text_to_embed = f"{name}: {description}" if description else name
            vec = await llm.get_embedding(text_to_embed)
        except Exception as e:  # pylint: disable=broad-except
            logger.warning("Failed to generate embedding for skill %s: %s", name, e)
            vec = None

        # Update Graph
        cypher = """
        MERGE (s:Skill {name: $name})
        SET s.code = $code,
            s.description = $desc,
            s.function_name = $func,
            s.tags = $tags,
            s.version = COALESCE(s.version, 0) + 1,
            s.updated_at = timestamp()
        """
        if vec:
            cypher += ", s.embedding = vecf32($vec)"

        cypher += " RETURN s.version"

        self.db.query(
            cypher,
            {
                "name": name,
                "code": code,
                "desc": description or "",
                "func": function_name,
                "tags": tags or [],
                "vec": vec,
            },
        )

        # Write to disk
        try:
            skill_file = self.skills_dir / f"{name}.py"
            skill_file.write_text(code, encoding="utf-8")
        except Exception as e:  # pylint: disable=broad-except
            logger.error("Failed to write skill to disk: %s", e)

        return name

    async def save_instructional_skill(
        self,
        name: str,
        instructions: str,
        description: str | None = None,
        tags: list[str] | None = None,
    ) -> str:
        """
        Save an instructional (folder-based) skill.
        Creates a directory with SKILL.md.
        """
        # Sanitize and limit name length
        name = re.sub(r"[^a-zA-Z0-9_]", "_", name)[:100]
        skill_dir = self.skills_dir / name
        skill_dir.mkdir(parents=True, exist_ok=True)

        md_file = skill_dir / "SKILL.md"

        # Construct Frontmatter
        desc = description or "Instructional Skill"
        tag_str = ", ".join(tags) if tags else ""

        content = f"""---
name: {name}
description: {desc}
tags: [{tag_str}]
type: instructional
---

{instructions}
"""
        md_file.write_text(content, encoding="utf-8")

        # Sync to DB
        await self._sync_instructional_skill(skill_dir, md_file)

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
            node = row[0] if isinstance(row, list) else row.get("s")
            if not node:
                continue

            props = node.properties if hasattr(node, "properties") else node
            if not isinstance(props, dict):
                continue

            skills[props.get("name", "unknown")] = {
                "description": props.get("description"),
                "tags": props.get("tags", []),
                "function_name": props.get("function_name"),
                "version": props.get("version", 1),
                "type": props.get("type", "python"),
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

        node = row[0] if isinstance(row, list) else row.get("s")
        if not node:
            return None

        props = node.properties if hasattr(node, "properties") else node
        if not isinstance(props, dict):
            return None

        return {
            "name": props.get("name"),
            "code": props.get("code"),
            "description": props.get("description"),
            "function_name": props.get("function_name"),
            "tags": props.get("tags", []),
            "version": props.get("version", 1),
            "type": props.get("type", "python"),
            "latest": True,
        }

    def get_import_statement(self, name: str) -> str:
        """
        Get the Python import statement for a skill.
        Ensures the file exists on disk first.
        """
        skill = self.get_skill(name)
        if not skill:
            raise ValueError(f"Skill '{name}' not found")

        if skill.get("type", "python") == "python":
            # Write to disk to ensure importable
            skill_file = self.skills_dir / f"{name}.py"
            if not skill_file.exists() or skill_file.read_text() != skill["code"]:
                skill_file.write_text(skill["code"])

            return f"from skills.{name} import {skill['function_name']}"

        return f"# Instructional Skill: {name} (See SKILL.md)"

    async def find_similar_skills(
        self, query: str, limit: int = 3
    ) -> List[Dict[str, Any]]:
        """
        Fuzzy search for skills based on semantic similarity.
        """
        vec = await llm.get_embedding(query)
        if not vec:
            return []

        params = {"vec": vec, "limit": limit}
        cypher = (
            f"CALL db.idx.vector.queryNodes('Skill', 'embedding', {limit}, vecf32($vec)) "
            "YIELD node, score RETURN node, score"
        )

        results = self.db.query(cypher, params)
        final = []
        for row in results:
            node = row.get("node")
            score = row.get("score")
            if node:
                props = node.properties if hasattr(node, "properties") else node
                final.append({**props, "score": score})
        return final

    def record_skill_success(self, skill_name: str) -> None:
        """Increment success counter for a skill."""
        cypher = """
        MATCH (s:Skill {name: $name})
        SET s.success_count = coalesce(s.success_count, 0) + 1,
            s.last_success = timestamp()
        """
        self.db.query(cypher, {"name": skill_name})

    def record_skill_error(self, skill_name: str, error_msg: str | None = None) -> None:
        """Increment error counter for a skill and optionally log the error."""
        cypher = """
        MATCH (s:Skill {name: $name})
        SET s.error_count = coalesce(s.error_count, 0) + 1,
            s.last_error = $error,
            s.last_error_at = timestamp()
        """
        self.db.query(cypher, {"name": skill_name, "error": error_msg or ""})

    def get_skill_health(self, skill_name: str) -> Dict[str, Any] | None:
        """Get health metrics for a skill."""
        cypher = """
        MATCH (s:Skill {name: $name})
        RETURN s.success_count as success_count,
               s.error_count as error_count,
               s.last_error as last_error
        """
        results = self.db.query(cypher, {"name": skill_name})
        if results:
            row = results[0]
            total = (row.get("success_count") or 0) + (row.get("error_count") or 0)
            return {
                "success_count": row.get("success_count") or 0,
                "error_count": row.get("error_count") or 0,
                "success_rate": (
                    ((row.get("success_count") or 0) / total * 100)
                    if total > 0
                    else 0.0
                ),
                "last_error": row.get("last_error"),
            }
        return None


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

        # Resolve skills directory relative to repo root
        if "graph_rlm" in str(Path(__file__).absolute()):
            # If we are inside the package, go up to repo root
            # mcp_integration -> src -> backend -> graph_rlm -> repo_root
            repo_root = Path(__file__).parent.parent.parent.parent.parent
        else:
            # Fallback
            repo_root = Path.cwd()

        skills_dir = repo_root / "skills"
        if not skills_dir.exists():
            # Try alternative location if we are in a different context
            skills_dir = Path("skills").absolute()

        _global_skills_manager = SkillsManager(skills_dir)

    return _global_skills_manager


# =============================================================================
# AXIOMS MANAGER - Separate from Skills for auto-generated validators
# =============================================================================


class AxiomsManager:
    """
    Manages a directory of auto-generated axioms (Python validators) in FalkorDB.
    Axioms are distinct from Skills - they are created by the Dreamer's Tier 3 consolidation.
    """

    def __init__(self, axioms_dir: Path) -> None:
        self.db = db
        self.axioms_dir = axioms_dir
        self.axioms_dir.mkdir(parents=True, exist_ok=True)
        (self.axioms_dir / "__init__.py").touch(exist_ok=True)
        # NOTE: sync_from_disk() removed from __init__ to avoid loop conflicts.

    async def sync_from_disk(self) -> None:
        """Sync axiom files from disk to database with :Axiom label."""
        count = 0
        for item in self.axioms_dir.iterdir():
            if item.name == "__init__.py" or item.name.startswith("__"):
                continue
            if item.is_file() and item.suffix == ".py":
                if await self._sync_axiom(item):
                    count += 1
        if count > 0:
            logger.info("Synced %d axioms from disk to FalkorDB.", count)

    async def _sync_axiom(self, file_path: Path) -> bool:
        try:
            code = file_path.read_text(encoding="utf-8")
            if "\x00" in code:
                logger.warning(
                    "Skipping axiom %s: Contains null bytes (DB incompatible).",
                    file_path.name,
                )
                return False
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
                return False

            name = file_path.stem
            function_name = func_def.name
            description = ast.get_docstring(func_def) or ""

            # Extract tags from filename
            tags = []
            if "physics" in name.lower():
                tags.append("physics")
            elif "fluid" in name.lower():
                tags.append("fluid_dynamics")
            elif "math" in name.lower() or "logic" in name.lower():
                tags.append("math")

            # Extract axiom_type from docstring
            # Heuristic: "Axiom Type: solver" or "Type: advisor"
            axiom_type = "validator"
            type_match = re.search(
                r"(?:Axiom\s+)?Type:\s*([\w_]+)", description, re.IGNORECASE
            )
            if type_match:
                axiom_type = type_match.group(1).lower()

            # Upsert into Graph with :Axiom label
            cypher = """
            MERGE (a:Axiom {name: $name})
            SET a.code = $code,
                a.description = $desc,
                a.function_name = $func,
                a.tags = $tags,
                a.axiom_type = $type,
                a.updated_at = timestamp()
            """
            self.db.query(
                cypher,
                {
                    "name": name,
                    "code": code,
                    "desc": description,
                    "func": function_name,
                    "tags": tags,
                    "type": axiom_type,
                },
            )
            return True
        except (ValueError, KeyError, RuntimeError) as e:
            logger.error("Axiom logic error during sync of %s: %s", file_path.name, e)
            return False
        except Exception as e:  # pylint: disable=broad-except
            logger.error("Unexpected error during axiom sync %s: %s", file_path.name, e)
            return False

    async def save_axiom(
        self,
        name: str,
        code: str,
        description: str | None = None,
        tags: list[str] | None = None,
        axiom_type: str = "validator",
        healing_code: str | None = None,
    ) -> str:
        """Save an axiom to the axioms library."""
        name = re.sub(r"[^a-zA-Z0-9_]", "_", name)[:100]
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
            raise ValueError(f"Invalid Python syntax: {e}") from e

        # Generate embedding
        try:
            text_to_embed = f"{name}: {description}" if description else name
            vec = await llm.get_embedding(text_to_embed)
        except Exception as e:  # pylint: disable=broad-except
            logger.warning("Failed to generate embedding for axiom %s: %s", name, e)
            vec = None

        cypher = """
        MERGE (a:Axiom {name: $name})
        SET a.code = $code,
            a.description = $desc,
            a.function_name = $func,
            a.tags = $tags,
            a.axiom_type = $type,
            a.healing_code = $healing,
            a.version = COALESCE(a.version, 0) + 1,
            a.updated_at = timestamp()
        """
        if vec:
            cypher += ", a.embedding = vecf32($vec)"

        cypher += " RETURN a.version"

        self.db.query(
            cypher,
            {
                "name": name,
                "code": code,
                "desc": description or "",
                "func": function_name,
                "tags": tags or ["general"],
                "type": axiom_type,
                "healing": healing_code,
                "vec": vec,
            },
        )

        # Write to disk
        try:
            axiom_file = self.axioms_dir / f"{name}.py"
            # Secondary Safety Net: Ensure final newline if LLM missed it
            if not code.endswith("\n"):
                code += "\n"
            axiom_file.write_text(code, encoding="utf-8")
        except Exception as e:  # pylint: disable=broad-except
            logger.error("Failed to write axiom to disk: %s", e)

        return name

    def list_axioms(self) -> dict[str, dict[str, Any]]:
        """List all available axioms with metadata."""
        cypher = "MATCH (a:Axiom) RETURN a"
        results = self.db.query(cypher) or []
        axioms = {}
        for row in results:
            if not row:
                continue
            node = row[0] if isinstance(row, list) else row.get("a")
            if not node:
                continue
            props = node.properties if hasattr(node, "properties") else node
            if not isinstance(props, dict):
                continue
            axioms[props.get("name", "unknown")] = {
                "description": props.get("description"),
                "tags": props.get("tags", []),
                "function_name": props.get("function_name"),
                "axiom_type": props.get("axiom_type", "validator"),
            }
        return axioms

    def get_axiom(self, name: str) -> dict[str, Any] | None:
        """Get the code and metadata for a specific axiom."""
        cypher = "MATCH (a:Axiom {name: $name}) RETURN a"
        results = self.db.query(cypher, {"name": name})
        if not results:
            return None
        row = results[0]
        node = row[0] if isinstance(row, list) else row.get("a")
        if not node:
            return None
        props = node.properties if hasattr(node, "properties") else node
        if not isinstance(props, dict):
            return None

        return props

    async def find_similar_axioms(
        self, query: str, limit: int = 3
    ) -> List[Dict[str, Any]]:
        """
        Fuzzy search for axioms based on semantic similarity.
        """
        vec = await llm.get_embedding(query)
        if not vec:
            return []

        params = {"vec": vec, "limit": limit}
        cypher = (
            f"CALL db.idx.vector.queryNodes('Axiom', 'embedding', {limit}, vecf32($vec)) "
            "YIELD node, score RETURN node, score"
        )

        results = self.db.query(cypher, params)
        final = []
        for row in results:
            node = row.get("node")
            score = row.get("score")
            if node:
                props = node.properties if hasattr(node, "properties") else node
                final.append({**props, "score": score})
        return final

    async def find_healing_axiom(
        self, violation_description: str, limit: int = 1
    ) -> List[Dict[str, Any]]:
        """
        Semantic search for an axiom that can 'heal' or fix a specific violation.
        Looks for axioms of type 'solver' or 'advisor' related to the error.
        """
        vec = await llm.get_embedding(violation_description)
        if not vec:
            return []

        params = {"vec": vec, "limit": limit}
        cypher = (
            "CALL db.idx.vector.queryNodes('Axiom', 'embedding', $limit, vecf32($vec)) "
            "YIELD node, score "
            "WHERE node.healing_code IS NOT NULL "
            "OR node.axiom_type IN ['solver', 'advisor'] "
            "RETURN node, score"
        )

        results = self.db.query(cypher, params)
        final = []
        for row in results:
            node = row.get("node")
            score = row.get("score")
            if node:
                props = node.properties if hasattr(node, "properties") else node
                final.append({**props, "score": score})
        return final

    async def disable_axiom(self, name: str) -> bool:
        """
        Move an axiom to the _disabled directory and remove from DB.
        """
        try:
            axiom_file = self.axioms_dir / f"{name}.py"
            disabled_dir = self.axioms_dir / "_disabled"
            disabled_dir.mkdir(parents=True, exist_ok=True)

            if axiom_file.exists():
                target_path = disabled_dir / axiom_file.name
                # Use shutil or just rename
                import shutil

                shutil.move(str(axiom_file), str(target_path))
                logger.info("Axiom '%s' moved to _disabled.", name)

            # Remove from DB
            self.db.query("MATCH (a:Axiom {name: $name}) DELETE a", {"name": name})
            logger.info("Axiom '%s' removed from database.", name)
            return True
        except Exception as e:
            logger.error("Failed to disable axiom '%s': %s", name, e)
            return False


# Global axioms manager instance
_global_axioms_manager: AxiomsManager | None = None


def get_axioms_manager() -> AxiomsManager:
    """Get or create the global axioms manager instance."""
    global _global_axioms_manager

    if _global_axioms_manager is None:
        backend_root = Path(__file__).parent.parent.parent
        axioms_dir = backend_root / "axioms_dir"
        _global_axioms_manager = AxiomsManager(axioms_dir)

    return _global_axioms_manager


# =============================================================================
# MCP PSEUDO-SERVER INTEGRATION
# =============================================================================

TOOLS = [
    {
        "name": "run_skill",
        "description": "Execute a registered skill with optional arguments.",
        "input_schema": {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "Name of the skill to execute",
                },
                "args": {
                    "type": "object",
                    "description": "Dictionary of arguments for the skill",
                },
            },
            "required": ["name"],
        },
    },
    {
        "name": "save_skill",
        "description": "Save a Python code block as a persistent skill.",
        "input_schema": {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Name for the skill"},
                "code": {"type": "string", "description": "Python source code"},
                "description": {
                    "type": "string",
                    "description": "Optional description",
                },
                "tags": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Optional tags",
                },
            },
            "required": ["name", "code"],
        },
    },
    {
        "name": "list_skills",
        "description": "List all available skills with metadata.",
        "input_schema": {"type": "object", "properties": {}},
    },
]


async def run_skill(name: str, args: dict | None = None, **_kwargs) -> Any:
    """MCP Wrapper for executing a skill."""
    from .skill_harness import execute_skill

    return await execute_skill(name, args or {})


async def save_skill(
    name: str,
    code: str,
    description: str | None = None,
    tags: list[str] | None = None,
    **_kwargs,
) -> str:
    """MCP Wrapper for saving a skill."""
    mgr = get_skills_manager()
    return await mgr.save_skill(name, code, description, tags)


async def list_skills(**_kwargs) -> dict[str, dict[str, Any]]:
    """MCP Wrapper for listing all skills."""
    mgr = get_skills_manager()
    return mgr.list_skills()
