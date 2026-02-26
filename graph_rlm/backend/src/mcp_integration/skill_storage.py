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
"""

import ast
import re
import shutil
import subprocess
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from graph_rlm.backend.src.core.db import db
from graph_rlm.backend.src.core.llm import llm
from graph_rlm.backend.src.core.logger import get_logger

logger = get_logger("graph_rlm.skills")


def _spec_name(raw: str) -> str:
    """Sanitize a raw name to Agent Skills spec format.

    Rules (from https://agentskills.io/specification.md):
    - 1-64 characters
    - Lowercase alphanumeric and hyphens only
    - Must not start or end with hyphen
    - No consecutive hyphens
    - Must match parent directory name
    """
    name = re.sub(r"[^a-z0-9-]", "-", raw.lower()).strip("-")
    name = re.sub(r"-{2,}", "-", name)  # collapse consecutive hyphens
    return name[:64] or "unnamed-skill"


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
        Removes any skills from DB that no longer exist on disk.
        """
        count = 0
        seen_names = set()

        # 1. Scan for Python Skills (*.py)
        for item in self.skills_dir.iterdir():
            if item.name == "__init__.py" or item.name.startswith("__"):
                continue

            # Case A: Python File (Standard Skill)
            if item.is_file() and item.suffix == ".py":
                synced_name = await self._sync_python_skill(item)
                if synced_name:
                    count += 1
                    seen_names.add(synced_name)

            # Case B: Folder-Based Skill (Instructional/Complex)
            elif item.is_dir():
                skill_md = item / "SKILL.md"
                # Check for spec-compliant script in scripts/ dir
                module_safe = item.name.replace("-", "_")
                scripts_dir = item / "scripts"
                py_script = scripts_dir / f"{module_safe}.py"

                if not py_script.exists() and scripts_dir.exists():
                    # Fallback: get the first .py in scripts/
                    for f in scripts_dir.iterdir():
                        if f.suffix == ".py" and not f.name.startswith("__"):
                            py_script = f
                            break

                if py_script.exists():
                    # Executable folder skill - handle as python skill
                    # BUT we might want to use metadata from SKILL.md if present
                    synced_name = await self._sync_python_skill(
                        py_script, name_override=_spec_name(item.name)
                    )
                    if synced_name:
                        count += 1
                        seen_names.add(synced_name)
                elif skill_md.exists():
                    synced_name = await self._sync_instructional_skill(item, skill_md)
                    if synced_name:
                        count += 1
                        seen_names.add(synced_name)

        # 2. Cleanup Stale Skills
        all_skills = self.list_skills()
        for name in list(all_skills.keys()):
            if name not in seen_names:
                logger.info("🗑️ Removing stale skill from DB: %s", name)
                self.db.query("MATCH (s:Skill {name: $name}) DELETE s", {"name": name})

        total = len(self.list_skills())
        logger.info(
            "Sync complete: %d items updated. Database now contains %d skills.",
            count,
            total,
        )

    async def _sync_python_skill(
        self, file_path: Path, name_override: str | None = None
    ) -> str | None:
        try:
            code = file_path.read_text(encoding="utf-8").strip()
            name = name_override or _spec_name(file_path.stem)

            # Optimization: Check if content changed before parsing/embedding
            existing = self.get_skill(name)
            if existing and existing.get("code", "").strip() == code:
                # Still check if embedding exists
                res = self.db.query(
                    "MATCH (s:Skill {name: $name}) RETURN s.embedding IS NOT NULL as has_vec",
                    {"name": name},
                )
                if res and res[0].get("has_vec"):
                    return name  # Skip redundant work but return name

            # Parse with stricter warning checks
            with warnings.catch_warnings(record=True):
                warnings.simplefilter("error", SyntaxWarning)
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
                logger.warning(
                    "Skipping %s: No function definition found.", file_path.name
                )
                return None

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
                # If still no tags, we leave it empty
                # (SheafMonitor will treat as general for general tasks)

            # Generate embedding for semantic search
            text_to_embed = f"{name}: {description}" if description else name
            try:
                vec = await llm.get_embedding(text_to_embed)
                if vec is None:
                    logger.warning(
                        "Embedding service returned None for skill %s. Semantic search will be unavailable.",
                        name,
                    )
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
            return name
        except Exception as e:  # pylint: disable=broad-except
            logger.error("Failed to sync python skill %s: %s", file_path.name, e)
            return None

    async def _sync_instructional_skill(
        self, dir_path: Path, md_path: Path
    ) -> str | None:
        try:
            content = md_path.read_text(encoding="utf-8").strip()
            name = dir_path.name

            # Optimization
            existing = self.get_skill(name)
            if (
                existing
                and existing.get("code", "").strip() == content
                and existing.get("type") == "instructional"
            ):
                return name

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

            name = _spec_name(frontmatter.get("name") or dir_path.name)
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
            return name
        except Exception as e:  # pylint: disable=broad-except
            logger.error("Failed to sync instructional skill %s: %s", dir_path.name, e)
            return None

    async def install_skill(self, source: str) -> str:
        """
        Install a skill from a remote source (Git URL or HTTP URL).
        mimics 'repo2skill' functionality.
        """
        # Validate that we have the necessary tools
        git_path = shutil.which("git")
        curl_path = shutil.which("curl")

        # 1. Handle Git URL
        if source.endswith(".git") or "github.com" in source:
            if not git_path:
                raise RuntimeError("Git executable not found in PATH.")

            try:
                skill_name = source.split("/")[-1].replace(".git", "")
                skill_path = self.skills_dir / skill_name

                if skill_path.exists():
                    logger.warning("Skill %s already exists. Updating...", skill_name)
                    # Perform git pull
                    subprocess.run(
                        [git_path, "-C", str(skill_path), "pull"],
                        check=True,
                        capture_output=True,
                    )
                else:
                    logger.info("Cloning skill from %s...", source)
                    subprocess.run(
                        [git_path, "clone", source, str(skill_path)],
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
                            f"---\nname: {skill_name}\n"
                            f"description: Auto-imported skill from {source}\n"
                            "---\n\n"
                            f"{content}"
                        )

                # Sync
                await self.sync_from_disk()
                return skill_name
            except Exception as e:
                logger.error("Failed to install skill from git: %s", e)
                raise

        # 2. Handle Single File URL (assumed to be raw content or downloadable)
        else:
            if not curl_path:
                raise RuntimeError("Curl executable not found in PATH.")

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
                    [curl_path, "-L", "-o", str(skill_file), source],
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

        Creates an Agent Skills spec-compliant directory:
            {name}/
            ├── SKILL.md          # Frontmatter + instructions
            └── scripts/
                └── {name}.py     # Executable code
        """
        name = _spec_name(name)
        code = code.strip()
        try:
            # 1. Parse with strict warning handling
            with warnings.catch_warnings(record=True):
                warnings.simplefilter("error", SyntaxWarning)
                tree = ast.parse(code)

            # 2. Ensure function definition exists
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

        except (SyntaxError, SyntaxWarning) as e:
            raise ValueError(f"Invalid Python syntax or warning: {e}") from e

        # Generate embedding
        desc = description or f"Python skill: {name}"
        try:
            text_to_embed = f"{name}: {desc}"
            vec = await llm.get_embedding(text_to_embed)
            if vec is None:
                logger.warning(
                    "Embedding service returned None for saved skill %s.", name
                )
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
            s.type = 'python',
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
                "desc": desc,
                "func": function_name,
                "tags": tags or [],
                "vec": vec,
            },
        )

        # Write Agent Skills spec-compliant directory
        try:
            skill_dir = self.skills_dir / name
            skill_dir.mkdir(parents=True, exist_ok=True)

            # scripts/ subdirectory
            scripts_dir = skill_dir / "scripts"
            scripts_dir.mkdir(exist_ok=True)
            script_file = scripts_dir / f"{name.replace('-', '_')}.py"
            script_file.write_text(code, encoding="utf-8")

            # Also write a flat importable .py at the skill dir level
            # so the harness can import it as skills.{name}.{name}
            init_file = skill_dir / "__init__.py"
            # Re-export the function from scripts/
            module_safe = name.replace("-", "_")
            init_file.write_text(
                f'"""Auto-generated: re-exports skill function."""\n'
                f"from .scripts.{module_safe} import {function_name}\n",
                encoding="utf-8",
            )

            # SKILL.md with spec-compliant frontmatter
            tag_str = ", ".join(f'"{t}"' for t in (tags or []))
            skill_md = skill_dir / "SKILL.md"
            skill_md.write_text(
                f"---\n"
                f"name: {name}\n"
                f"description: {desc}\n"
                + (
                    f"metadata:\n  tags: [{tag_str}]\n  type: skill\n  origin: agent\n"
                    if tags
                    else "metadata:\n  type: skill\n  origin: agent\n"
                )
                + f"---\n\n"
                f"# {name}\n\n"
                f"{desc}\n\n"
                f"## Usage\n\n"
                f"Run this skill's entry function `{function_name}` from `scripts/{name.replace('-', '_')}.py`.\n",
                encoding="utf-8",
            )
            logger.info("Skill '%s' saved as Agent Skills directory.", name)
        except Exception as e:  # pylint: disable=broad-except
            logger.error("Failed to write skill directory to disk: %s", e)

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
        Creates an Agent Skills spec-compliant directory with SKILL.md.
        """
        name = _spec_name(name)
        instructions = instructions.strip()
        skill_dir = self.skills_dir / name
        skill_dir.mkdir(parents=True, exist_ok=True)

        md_file = skill_dir / "SKILL.md"

        # Construct spec-compliant frontmatter
        desc = description or "Instructional skill"
        metadata_lines = ["metadata:", "  type: instructional", "  origin: agent"]
        if tags:
            tag_str = ", ".join(f'"{t}"' for t in tags)
            metadata_lines.append(f"  tags: [{tag_str}]")
        metadata_block = "\n".join(metadata_lines)

        content = (
            f"---\n"
            f"name: {name}\n"
            f"description: {desc}\n"
            f"{metadata_block}\n"
            f"---\n\n"
            f"{instructions}\n"
        )
        md_file.write_text(content, encoding="utf-8")

        # Sync to DB
        await self._sync_instructional_skill(skill_dir, md_file)
        logger.info("Instructional skill '%s' saved (Agent Skills spec).", name)

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


@dataclass
class _ManagerState:
    """Internal state container for singleton managers."""

    skills_manager: Optional["SkillsManager"] = None
    axioms_manager: Optional["AxiomsManager"] = None


_state = _ManagerState()


def _resolve_system_path(relative_to_backend: str) -> Path:
    """
    Resolves a path relative to graph_rlm/backend/.
    Ensures consistent resolution regardless of whether running from source or installed.
    """
    file_path = Path(__file__).absolute()
    if "graph_rlm" in str(file_path):
        # We are inside the package structure
        # .../graph_rlm/backend/src/mcp_integration/skill_storage.py
        # Go up 3 levels to get to graph_rlm/backend/
        backend_root = file_path.parent.parent.parent
    else:
        # Fallback to current working directory or relative to src
        backend_root = file_path.parent.parent.parent

    target_dir = backend_root / relative_to_backend
    target_dir.mkdir(parents=True, exist_ok=True)
    return target_dir


def get_skills_manager() -> SkillsManager:
    """
    Get or create the global skills manager instance.
    """
    if _state.skills_manager is None:
        skills_dir = _resolve_system_path("skills")
        _state.skills_manager = SkillsManager(skills_dir)

    return _state.skills_manager


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
        # Ensure __init__ exists
        (self.axioms_dir / "__init__.py").touch(exist_ok=True)
        # NOTE: sync_from_disk() removed from __init__ to avoid loop conflicts.

    async def sync_from_disk(self) -> None:
        """Sync axiom files from disk to database with :Axiom label.

        Handles both legacy flat .py files and new spec-compliant directories.
        """
        count = 0
        seen_names = set()

        # Group items by spec_name to prevent double-processing flat vs directory
        items_by_name: Dict[str, List[Path]] = {}
        for item in self.axioms_dir.iterdir():
            if (
                item.name == "__init__.py"
                or item.name.startswith("__")
                or item.name == "_disabled"
            ):
                continue

            name = _spec_name(item.stem if item.is_file() else item.name)
            if name not in items_by_name:
                items_by_name[name] = []
            items_by_name[name].append(item)

        for name, items in items_by_name.items():
            # Prioritize directory over flat file
            target_item = next((i for i in items if i.is_dir()), items[0])

            # Case A: Legacy flat .py file
            if target_item.is_file() and target_item.suffix == ".py":
                if await self._sync_axiom(target_item):
                    count += 1
                seen_names.add(name)

            # Case B: Spec-compliant directory with SKILL.md
            elif target_item.is_dir():
                skill_md = target_item / "SKILL.md"
                scripts_dir = target_item / "scripts"
                if skill_md.exists():
                    # Find the .py script inside scripts/
                    py_file = None
                    if scripts_dir.exists():
                        for f in scripts_dir.iterdir():
                            if f.suffix == ".py" and not f.name.startswith("__"):
                                py_file = f
                                break
                    if py_file:
                        if await self._sync_axiom(py_file):
                            count += 1
                        seen_names.add(name)
                    else:
                        # Instructional axiom (no code)
                        seen_names.add(name)

        # Cleanup Stale Axioms
        all_axioms = self.list_axioms()
        for name in list(all_axioms.keys()):
            if name not in seen_names:
                logger.info("🗑️ Removing stale axiom from DB: %s", name)
                self.db.query("MATCH (a:Axiom {name: $name}) DELETE a", {"name": name})

        total = len(self.list_axioms())
        logger.info(
            "Sync complete: %d items updated. Database now contains %d axioms.",
            count,
            total,
        )

    async def _sync_axiom(self, file_path: Path) -> bool:
        try:
            code = file_path.read_text(encoding="utf-8").strip()
            name = _spec_name(file_path.stem)

            # Optimization: Check if content changed
            existing = self.get_axiom(name)
            if existing and existing.get("code", "").strip() == code:
                # Still check if embedding exists
                res = self.db.query(
                    "MATCH (a:Axiom {name: $name}) RETURN a.embedding IS NOT NULL as has_vec",
                    {"name": name},
                )
                if res and res[0].get("has_vec"):
                    return False

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

            # Extract tags from docstring (function and module level)
            module_doc = ast.get_docstring(tree) or ""
            if (
                "Tags: system_utility" in description
                or "Tags: system_utility" in module_doc
            ):
                tags.append("system_utility")

            # Extract axiom_type from docstring
            # Extract axiom_type from docstring
            # Heuristic: "Axiom Type: solver" or "Type: advisor"
            axiom_type = "validator"
            type_match = re.search(
                r"(?:Axiom\s+)?Type:\s*([\w_]+)", description, re.IGNORECASE
            )
            if type_match:
                axiom_type = type_match.group(1).lower()

            # Generate embedding for semantic search
            text_to_embed = f"{name}: {description}" if description else name
            try:
                vec = await llm.get_embedding(text_to_embed)
            except Exception as e:  # pylint: disable=broad-except
                logger.warning("Failed to generate embedding for axiom %s: %s", name, e)
                vec = None

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
            if vec:
                cypher += ", a.embedding = vecf32($vec)"

            self.db.query(
                cypher,
                {
                    "name": name,
                    "code": code,
                    "desc": description,
                    "func": function_name,
                    "tags": tags,
                    "type": axiom_type,
                    "vec": vec,
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
        markdown_body: str | None = None,
        session_id: str | None = None,
        root_session_id: str | None = None,
    ) -> str:
        """Save an axiom to the axioms library.

        Creates an Agent Skills spec-compliant directory:
            {name}/
            ├── SKILL.md          # Frontmatter with metadata.type = axiom_type
            └── scripts/
                └── {name}.py     # Executable code
        """
        name = _spec_name(name)
        code = code.strip()
        try:
            # 1. Parse with strict warning handling
            with warnings.catch_warnings(record=True):
                warnings.simplefilter("error", SyntaxWarning)
                tree = ast.parse(code)

            # 2. Ensure function definition exists
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

        except (SyntaxError, SyntaxWarning) as e:
            raise ValueError(f"Invalid Python syntax or warning: {e}") from e

        # 2. Generate embedding
        desc = description or f"Axiom ({axiom_type}): {name}"
        try:
            text_to_embed = f"{name}: {desc}"
            vec = await llm.get_embedding(text_to_embed)
            if vec is None:
                logger.warning(
                    "Embedding service returned None for saved axiom %s.", name
                )
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
            a.session_id = $session_id,
            a.root_session_id = $root_id,
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
                "desc": desc,
                "func": function_name,
                "tags": tags or ["general"],
                "type": axiom_type,
                "healing": healing_code,
                "session_id": session_id,
                "root_id": root_session_id,
                "vec": vec,
            },
        )

        # Write Agent Skills spec-compliant directory
        try:
            axiom_dir = self.axioms_dir / name
            axiom_dir.mkdir(parents=True, exist_ok=True)

            # scripts/ subdirectory
            scripts_dir = axiom_dir / "scripts"
            scripts_dir.mkdir(exist_ok=True)
            module_safe = name.replace("-", "_")
            script_file = scripts_dir / f"{module_safe}.py"
            script_file.write_text(code, encoding="utf-8")

            # __init__.py for importability
            init_file = axiom_dir / "__init__.py"
            init_file.write_text(
                f'"""Auto-generated: re-exports axiom function."""\n'
                f"from .scripts.{module_safe} import {function_name}\n",
                encoding="utf-8",
            )

            # SKILL.md with spec frontmatter
            tag_str = ", ".join(f'"{t}"' for t in (tags or ["general"]))

            # Default body if none provided
            final_markdown_body = markdown_body or (
                f"# {name}\n\n"
                f"{desc}\n\n"
                f"## Usage\n\n"
                f"Entry function: `{function_name}` in `scripts/{module_safe}.py`.\n"
            )

            skill_md = axiom_dir / "SKILL.md"
            skill_md.write_text(
                f"---\n"
                f"name: {name}\n"
                f"description: {desc}\n"
                f"metadata:\n"
                f"  type: {axiom_type}\n"
                f"  origin: dreamer\n"
                f"  tags: [{tag_str}]\n"
                + (f"  session-id: {session_id}\n" if session_id else "")
                + f"---\n\n"
                f"{final_markdown_body}",
                encoding="utf-8",
            )
            logger.info(
                "Axiom '%s' saved as Agent Skills directory (type=%s).",
                name,
                axiom_type,
            )
        except Exception as e:  # pylint: disable=broad-except
            logger.error("Failed to write axiom directory to disk: %s", e)

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

    def get_system_axioms(self) -> List[Dict[str, Any]]:
        """
        Get all axioms tagged as 'system_utility'.
        These are the foundational axioms loaded by default.
        """
        cypher = "MATCH (a:Axiom) WHERE 'system_utility' IN a.tags RETURN a"
        results = self.db.query(cypher) or []
        system_axioms = []
        for row in results:
            if not row:
                continue
            node = row[0] if isinstance(row, list) else row.get("a")
            if not node:
                continue
            props = node.properties if hasattr(node, "properties") else node
            if not isinstance(props, dict):
                continue
            system_axioms.append(props)
        return system_axioms

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
            # 1. Identify physical location (Directory, Hyphenated file, or Legacy underscore file)
            axiom_dir = self.axioms_dir / name
            axiom_file = self.axioms_dir / f"{name}.py"
            legacy_file = self.axioms_dir / f"{name.replace('-', '_')}.py"

            disabled_dir = self.axioms_dir / "_disabled"
            disabled_dir.mkdir(parents=True, exist_ok=True)

            target_path = None
            if axiom_dir.is_dir():
                target_path = axiom_dir
            elif axiom_file.exists():
                target_path = axiom_file
            elif legacy_file.exists():
                target_path = legacy_file

            if target_path:
                destination = disabled_dir / target_path.name
                # Cleanup destination if it already exists to avoid move errors
                if destination.exists():
                    if destination.is_dir():
                        shutil.rmtree(str(destination))
                    else:
                        destination.unlink()

                shutil.move(str(target_path), str(destination))
                logger.info("Axiom '%s' physically moved to _disabled.", name)

            # 2. Remove from DB
            self.db.query("MATCH (a:Axiom {name: $name}) DELETE a", {"name": name})
            logger.info("Axiom '%s' removed from database.", name)
            return True
        except (OSError, RuntimeError) as e:
            logger.error("Failed to disable axiom '%s': %s", name, e)
            return False
        except Exception as e:  # pylint: disable=broad-except
            logger.error("Unexpected error disabling axiom '%s': %s", name, e)
            return False


def get_axioms_manager() -> AxiomsManager:
    """Get or create the global axioms manager instance."""
    if _state.axioms_manager is None:
        axioms_dir = _resolve_system_path("axioms_dir")
        _state.axioms_manager = AxiomsManager(axioms_dir)

    return _state.axioms_manager


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
    # pylint: disable=import-outside-toplevel
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
