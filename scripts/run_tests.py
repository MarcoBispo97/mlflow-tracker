"""Runs pytest with the project's virtual environment (used by the pre-commit hook).

Git hooks do not activate the virtual environment, so this script finds
``.venv`` next to the project and runs ``python -m pytest`` with it. If there is
no ``.venv`` (e.g. on CI), it uses the current interpreter.
"""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

for candidate in (ROOT / ".venv" / "Scripts" / "python.exe", ROOT / ".venv" / "bin" / "python"):
    if candidate.exists():
        python = candidate
        break
else:
    python = Path(sys.executable)

sys.exit(subprocess.call([str(python), "-m", "pytest"], cwd=ROOT))
