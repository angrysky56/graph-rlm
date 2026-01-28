"""
Skills Library.

This package contains reusable skills for the Agent.
Skills are loaded dynamically via the SkillsManager (AST parsing) or imported explicitly by the Agent.
We do NOT eagerly import everything here to prevent crashes from missing dependencies in individual skills.
"""
