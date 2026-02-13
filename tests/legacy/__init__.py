"""Legacy tests package for Graph-RLM.

Contains tests using traditional unittest.mock patterns (not pytest-mock fixtures).
These tests focus on isolated unit testing with direct mock configuration.
"""

import unittest.mock

# Re-export commonly used mocking utilities for backwards compatibility
Mock = unittest.mock.Mock
MagicMock = unittest.mock.MagicMock
patch = unittest.mock.patch
patch.object = unittest.mock.patch.object
AsyncMock = unittest.mock.AsyncMock
