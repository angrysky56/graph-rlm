"""
Database management for MCP-Coordinator.

Handles SQLite connection, schema creation, and session management.
"""

import sqlite3
from pathlib import Path
from typing import Any


class DatabaseManager:
    """Manages the SQLite database for skills."""

    def __init__(self, db_path: str | Path = "./workspace/mcp_coordinator.db") -> None:
        """
        Initialize the database manager.

        Args:
            db_path: Path to the SQLite database file
        """
        self.db_path = Path(db_path).resolve()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.connection: sqlite3.Connection | None = None

    def connect(self) -> sqlite3.Connection:
        """Establish a connection to the database."""
        if self.connection is None:
            self.connection = sqlite3.connect(self.db_path, check_same_thread=False)
            self.connection.row_factory = sqlite3.Row
        return self.connection

    def close(self) -> None:
        """Close the database connection."""
        if self.connection:
            self.connection.close()
            self.connection = None

    def execute(self, query: str, params: tuple[Any, ...] = ()) -> sqlite3.Cursor:
        """Execute a SQL query."""
        conn = self.connect()
        cursor = conn.cursor()
        cursor.execute(query, params)
        return cursor

    def commit(self) -> None:
        """Commit the current transaction."""
        conn = self.connect()
        conn.commit()

    def create_tables(self) -> None:
        """Create the necessary tables if they don't exist."""
        create_skills_table = """
        CREATE TABLE IF NOT EXISTS skills (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            version INTEGER NOT NULL,
            code TEXT NOT NULL,
            description TEXT,
            tags TEXT,
            function_name TEXT,
            latest BOOLEAN DEFAULT TRUE,
            UNIQUE(name, version)
        );
        """
        self.execute(create_skills_table)

        # SECURITY NOTE: tool_logs stores ALL tool arguments in plaintext.
        # This may include API keys or secrets passed to tools.
        # Keep this database private. If exposing the coordinator to public
        # access, sanitize or disable logging, or clean the database regularly.
        create_tool_logs_table = """
        CREATE TABLE IF NOT EXISTS tool_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            tool_call_id TEXT,
            server TEXT NOT NULL,
            tool TEXT NOT NULL,
            arguments TEXT,  -- May contain sensitive data!
            result TEXT,
            duration REAL,
            success BOOLEAN,
            error TEXT
        );
        """
        self.execute(create_tool_logs_table)
        self.commit()

    def __enter__(self) -> "DatabaseManager":
        """Context manager entry."""
        self.connect()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Context manager exit."""
        self.close()


# Global database manager instance
_global_db_manager: DatabaseManager | None = None


def get_db_manager(db_path: str | Path | None = None) -> DatabaseManager:
    """
    Get or create the global database manager instance.

    Args:
        db_path: Optional path to the database file

    Returns:
        Shared DatabaseManager instance
    """
    global _global_db_manager

    if _global_db_manager is None:
        if db_path is None:
            db_path = "./workspace/mcp_coordinator.db"
        _global_db_manager = DatabaseManager(db_path)
        _global_db_manager.create_tables()

    return _global_db_manager
