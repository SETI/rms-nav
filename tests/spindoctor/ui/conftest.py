"""Pytest configuration for UI tests (PyQt6).

``QT_QPA_PLATFORM`` must be set before any ``PyQt6`` import so headless Linux
hosts and CI runners do not require a display server.
"""

import os

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
