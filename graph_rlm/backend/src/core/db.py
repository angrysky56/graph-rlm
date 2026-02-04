# Shim for backward compatibility
from .database import db, GraphClientFacade, client

__all__ = ["db", "GraphClientFacade", "client"]
