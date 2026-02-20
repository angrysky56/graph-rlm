#!/usr/bin/env python3
"""
Utility script to regenerate MCP tool wrappers from mcp_servers.json.
Ensures the Python wrappers in mcp_tools/ are in sync with the server configurations.
"""

import logging
import sys
from pathlib import Path

# Add backend to path for imports
backend_path = Path(__file__).parent.parent.parent.parent.absolute()
if str(backend_path) not in sys.path:
    sys.path.insert(0, str(backend_path))

from graph_rlm.backend.src.mcp_integration.generator import generate_from_config

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def main():
    config_path = Path("mcp_servers.json")
    output_dir = Path("graph_rlm/backend/mcp_tools")

    logger.info("Starting MCP tool regeneration...")
    try:
        generate_from_config(config_path, output_dir=output_dir)
        logger.info("✅ Tools regenerated successfully in %s", output_dir)
    except Exception as e:
        logger.error("❌ Regeneration failed: %s", e)
        sys.exit(1)


if __name__ == "__main__":
    main()
