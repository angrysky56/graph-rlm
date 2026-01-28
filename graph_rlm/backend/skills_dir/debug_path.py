
def debug_path():
    """Debugs the python path and environment."""
    import os
    import sys
    print(f"SYS.PATH: {sys.path}")
    print(f"CWD: {os.getcwd()}")
    print(f"PYTHONPATH: {os.environ.get('PYTHONPATH')}")
    try:
        print(f"LS /knowledge_base: {os.listdir('/knowledge_base')}")
        print(f"LS /knowledge_base/skills: {os.listdir('/knowledge_base/skills')}")
    except Exception as e:
        print(f"LS Error: {e}")
