import ast
import os
import traceback
import warnings


def check_file(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        source = f.read()

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always", SyntaxWarning)
        try:
            compile(source, filepath, "exec")
        except SyntaxError:
            pass  # We are looking for warnings not errors

        if w:
            print(f"\nFile: {filepath}")
            for warning in w:
                print(f"  Line {warning.lineno}: {warning.message}")


def main():
    target_dirs = ["graph_rlm/backend/axioms_dir", "graph_rlm/backend/skills"]

    current_dir = os.getcwd()

    for relative_dir in target_dirs:
        full_path = os.path.join(current_dir, relative_dir)
        if not os.path.isdir(full_path):
            print(f"Skipping {relative_dir} (not found)")
            continue

        print(f"Scanning {relative_dir}...")
        for root, dirs, files in os.walk(full_path):
            for file in files:
                if file.endswith(".py"):
                    check_file(os.path.join(root, file))


if __name__ == "__main__":
    main()
