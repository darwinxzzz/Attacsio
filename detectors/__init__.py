"""
detectors/__init__.py

This file indicates that 'detectors' is a Python package. It also re-exports
the GameManager class, so you can import it directly from 'detectors'.
"""

from .gamification import GameManager

__all__ = [
    "GameManager",
]
