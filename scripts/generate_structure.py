import os

def generate_tree(startpath, output_file):
    ignore_dirs = {'.git', '.venv', '__pycache__', 'node_modules', '.pytest_cache', '.ruff_cache', 'dist', 'build', 'coverage', '.idea', '.vscode'}

    with open(output_file, 'w') as f:
        f.write(f"Project Structure for: {os.path.abspath(startpath)}\n")
        f.write("=" * 50 + "\n\n")

        for root, dirs, files in os.walk(startpath):
            # Modify dirs in-place to exclude ignored directories
            dirs[:] = [d for d in dirs if d not in ignore_dirs]

            level = root.replace(startpath, '').count(os.sep)
            indent = '    ' * level
            f.write(f"{indent}{os.path.basename(root)}/\n")

            subindent = '    ' * (level + 1)
            for file in files:
                if file.startswith('.'): # Skip hidden files
                    continue
                f.write(f"{subindent}{file}\n")

if __name__ == "__main__":
    generate_tree('.', 'project_structure.txt')
    print("Project structure saved to project_structure.txt")
