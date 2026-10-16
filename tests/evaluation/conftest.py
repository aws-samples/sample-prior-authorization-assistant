import sys
from pathlib import Path

# The evaluation package lives at the repository root, outside src/
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
