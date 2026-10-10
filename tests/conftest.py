"""Allow `pytest RWR/tests` from either the repository root or RWR/."""
import sys
from pathlib import Path

RWR_ROOT = Path(__file__).resolve().parents[1]
if str(RWR_ROOT) not in sys.path:
    sys.path.insert(0, str(RWR_ROOT))
