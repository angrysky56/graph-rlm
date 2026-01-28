def docker_safe_write(filename: str, content: str, subdir: str = "outputs") -> dict:
    """
    Write a file to the knowledge base using Docker-aware paths.
    
    Automatically detects if running in Docker and uses appropriate paths:
    - Docker: /knowledge_base/{subdir}/{filename}
    - Local: ./knowledge_base/{subdir}/{filename}
    
    Args:
        filename: Name of the file to write (e.g., "report.md")
        content: Content to write to the file
        subdir: Subdirectory within knowledge_base (default: "outputs")
    
    Returns:
        dict with 'success', 'path', and 'message' keys
    """
    from pathlib import Path
    import os
    
    try:
        # Detect if we're in Docker or local environment
        if Path("/knowledge_base").exists():
            # Inside Docker container
            base_path = Path("/knowledge_base")
            env_name = "Docker"
        else:
            # Local execution - use project-relative path
            kb_dir = os.getenv("MCP_KNOWLEDGE_BASE_DIR", "./knowledge_base")
            base_path = Path(kb_dir)
            env_name = "Local"
        
        # Create the target directory
        target_dir = base_path / subdir
        target_dir.mkdir(parents=True, exist_ok=True)
        
        # Write the file
        file_path = target_dir / filename
        file_path.write_text(content)
        
        # Return the path as it appears in the container (for logging)
        display_path = f"/{subdir}/{filename}" if env_name == "Docker" else str(file_path)
        
        return {
            "success": True,
            "path": str(file_path),
            "display_path": display_path,
            "environment": env_name,
            "message": f"File written successfully to {display_path} ({env_name})"
        }
        
    except Exception as e:
        return {
            "success": False,
            "path": None,
            "message": f"Failed to write file: {str(e)}"
        }
