"""Module to generate a text-based tree representation of the project structure."""

import os
import shutil
import subprocess


def generate_tree(startpath: str, output_file: str):
    """
    Generates a text-based tree representation of the project structure.
    Respects .gitignore if the directory is a git repository.

    Args:
        startpath: The root directory to start walking from.
        output_file: The path to the file where the structure will be saved.
    """
    ignore_dirs = {
        ".git",
        ".venv",
        "__pycache__",
        "node_modules",
        ".pytest_cache",
        ".ruff_cache",
        "dist",
        "build",
        "coverage",
        ".idea",
        ".vscode",
    }

    # Check if we're in a git repository
    git_cmd = shutil.which("git")
    has_git = False
    if git_cmd:
        try:
            subprocess.run(
                [git_cmd, "rev-parse", "--is-inside-work-tree"],
                cwd=startpath,
                capture_output=True,
                check=True,
            )
            has_git = True
        except (subprocess.CalledProcessError, FileNotFoundError):
            pass

    def filter_git_ignored(items, root, is_dir):
        """Returns items that are not ignored by git."""
        if not items or not has_git or not git_cmd:
            return items
        try:
            rel_root = os.path.relpath(root, startpath)
            paths_to_check = []
            for item in items:
                # Construct relative path for git check-ignore
                path = item if rel_root == "." else os.path.join(rel_root, item)
                if is_dir:
                    path += "/"
                paths_to_check.append(path)

            res = subprocess.run(
                [git_cmd, "check-ignore", "--stdin"],
                cwd=startpath,
                input="\n".join(paths_to_check),
                text=True,
                capture_output=True,
                check=False,
            )
            ignored_paths = set(res.stdout.splitlines())
            return [
                i
                for i, p in zip(items, paths_to_check, strict=True)
                if p not in ignored_paths
            ]
        except (subprocess.SubprocessError, OSError):
            return items

    with open(output_file, "w", encoding="utf-8") as f:
        f.write(f"Project Structure for: {os.path.abspath(startpath)}\n")
        f.write("=" * 50 + "\n\n")

        for root, dirs, files in os.walk(startpath):
            # 1. Prune hardcoded ignored directories
            dirs[:] = [d for d in dirs if d not in ignore_dirs]

            # 2. Prune git-ignored directories and files
            if has_git:
                dirs[:] = filter_git_ignored(dirs, root, is_dir=True)
                files = filter_git_ignored(files, root, is_dir=False)
            else:
                # Fallback: Skip hidden files if not in a git repo
                files = [file for file in files if not file.startswith(".")]

            level = root.replace(startpath, "").count(os.sep)
            indent = "    " * level
            f.write(f"{indent}{os.path.basename(root) or startpath}/\n")

            subindent = "    " * (level + 1)
            for file in sorted(files):
                f.write(f"{subindent}{file}\n")


if __name__ == "__main__":
    generate_tree(".", "project_structure.txt")
    print("Project structure saved to project_structure.txt")
