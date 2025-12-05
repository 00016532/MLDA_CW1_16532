from __future__ import annotations
import sys
from pathlib import Path

def add_project_root_to_path(file: str) -> Path:
    p = Path(file).resolve()
    # walk up until we find 'src' directory
    cur = p.parent
    for _ in range(6):
        if (cur / "src").exists():
            root = cur
            if str(root) not in sys.path:
                sys.path.insert(0, str(root))
            return root
        cur = cur.parent
    # fallback
    root = Path(__file__).resolve().parents[1]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    return root
