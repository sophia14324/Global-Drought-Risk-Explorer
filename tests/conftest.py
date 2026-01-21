"""Pytest configuration to ensure project root is on the Python path.

This file modifies ``sys.path`` at test collection time so that the
project's top‑level ``src`` package and its subpackages are importable.

Without this, tests that import ``src`` may fail with
``ModuleNotFoundError`` if the project root is not automatically added
to ``sys.path`` by the test runner.
"""

import os
import sys

# Determine the absolute path to the project root (one level up from the
# ``tests`` directory) and add it to the front of ``sys.path``.
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
