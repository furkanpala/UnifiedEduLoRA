"""
pytest config: make the unifiedfl/ package importable like the runtime scripts
do (they sys.path.insert(0, str(Path(__file__).parent)) on the unifiedfl dir).
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))           # so `import experiments...` works
sys.path.insert(0, str(REPO / "unifiedfl"))  # so `from evaluation.metrics import ...` works
