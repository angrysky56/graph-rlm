import importlib
import os
import pkgutil
import sys

# Add current directory to path so it can find 'graph_rlm' package
sys.path.insert(0, os.getcwd())

def import_all(package_name):
    print(f"Scanning package: {package_name}")
    try:
        package = importlib.import_module(package_name)
    except ImportError as e:
        print(f"Could not import root package {package_name}: {e}")
        return

    # Use recursive walk
    path = package.__path__
    prefix = package.__name__ + "."

    for _, name, is_pkg in pkgutil.walk_packages(path, prefix):
        try:
            # print(f"Importing {name}...")
            importlib.import_module(name)
        except Exception as e:
            # print(f"Failed to import {name}: {e}")
            pass

if __name__ == "__main__":
    # graph_rlm should be importable if current dir is in path
    import_all("graph_rlm.backend")
