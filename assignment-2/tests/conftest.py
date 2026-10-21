"""Make the assignment root, data/ and results/ importable from the tests."""

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for path in (ROOT, os.path.join(ROOT, "data"), os.path.join(ROOT, "results")):
    if path not in sys.path:
        sys.path.insert(0, path)
