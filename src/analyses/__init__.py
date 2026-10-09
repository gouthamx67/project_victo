"""
Deterministic safety analyses.

Importing this package loads all built-in analysis modules so their
decorated analysis functions are registered with the central registry.
"""

from src.analyses import core
from src.analyses import reactions
from src.analyses import reporting
from src.analyses import trends
__all__ = [
    "core",
    "reactions",
    "trends",
]