from typing import Dict, Optional, List
from .core import PythonREPL
import logging

logger = logging.getLogger("graph_rlm.repl.manager")

class REPLManager:
    """
    Manages multiple PythonREPL instances.
    """
    def __init__(self):
        self._repls: Dict[str, PythonREPL] = {}

    def create_repl(self, repl_id: Optional[str] = None) -> str:
        """
        Create a new REPL instance.

        Args:
            repl_id: Optional ID for the REPL. If not provided, one will be generated.

        Returns:
            The ID of the created REPL.
        """
        repl = PythonREPL(repl_id)
        self._repls[repl.repl_id] = repl
        logger.info(f"Created REPL session: {repl.repl_id}")
        return repl.repl_id

    def get_repl(self, repl_id: str) -> Optional[PythonREPL]:
        """
        Get a REPL instance by ID.
        """
        return self._repls.get(repl_id)

    def delete_repl(self, repl_id: str) -> bool:
        """
        Delete a REPL instance.

        Returns:
            True if deleted, False if not found.
        """
        if repl_id in self._repls:
            del self._repls[repl_id]
            return True
        return False

    def list_repls(self) -> List[str]:
        """
        List all active REPL IDs.
        """
        return list(self._repls.keys())

    def exists(self, repl_id: str) -> bool:
        """Check if a REPL exists."""
        return repl_id in self._repls
