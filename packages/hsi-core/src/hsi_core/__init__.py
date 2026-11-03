"""HSI core engine.

The core is domain-agnostic. It speaks only the canonical types in `hsi_core.model`
and reaches plugins only through the protocols in `hsi_core.interfaces`.
It must never import network, LLM or vendor libraries (enforced by tests/test_core_imports.py).
"""

__version__ = "0.1.0"
