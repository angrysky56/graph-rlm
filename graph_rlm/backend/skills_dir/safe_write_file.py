def safe_write_file(path, content):
    from pathlib import Path

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        f.write(content)
    return f"Successfully wrote to {path}"
