"""Constitution P2/P12: hsi_core imports only the standard library (minus network modules),
numpy, pydantic, yaml and itself."""
from __future__ import annotations

import ast
import sys
from pathlib import Path

import hsi_core

ALLOWED_THIRD_PARTY = {"numpy", "pydantic", "yaml", "hsi_core"}
FORBIDDEN_STDLIB = {"socket", "ssl", "http", "urllib", "ftplib", "smtplib", "poplib", "imaplib",
                    "xmlrpc", "asyncio", "telnetlib", "socketserver", "webbrowser"}


def _imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names |= {a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            names.add(node.module.split(".")[0])
    return names


def test_core_imports_are_allowed():
    root = Path(hsi_core.__file__).parent
    bad: dict[str, set[str]] = {}
    for f in root.rglob("*.py"):
        for name in _imports(f):
            third_party = name not in sys.stdlib_module_names and name != "__future__"
            if name in FORBIDDEN_STDLIB or (third_party and name not in ALLOWED_THIRD_PARTY):
                bad.setdefault(str(f.relative_to(root)), set()).add(name)
    assert not bad, f"forbidden imports in hsi_core: {bad}"
