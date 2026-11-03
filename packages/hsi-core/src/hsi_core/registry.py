"""Plugin registry. Plugins are discovered through the `hsi.plugins` entry-point group.

Entry-point names have the form `<family>:<name>`, e.g. `vector_store:faiss`.
"""
from __future__ import annotations

from importlib.metadata import entry_points
from typing import Any

FAMILIES = frozenset({"encoder", "vector_store", "metadata_store", "connector", "parser",
                      "mention_extractor", "reranker", "nli", "reader", "identity", "redactor", "surface"})
GROUP = "hsi.plugins"


class PluginError(RuntimeError):
    pass


def available(family: str | None = None) -> dict[str, list[str]]:
    found: dict[str, list[str]] = {}
    for ep in entry_points(group=GROUP):
        fam, _, name = ep.name.partition(":")
        if family is None or fam == family:
            found.setdefault(fam, []).append(name)
    return {k: sorted(v) for k, v in found.items()}


def load(family: str, name: str, **options: Any) -> Any:
    """Instantiate plugin `family:name` with `options`. Fails with a clear message if unknown."""
    if family not in FAMILIES:
        raise PluginError(f"unknown plugin family {family!r}; expected one of {sorted(FAMILIES)}")
    key = f"{family}:{name}"
    matches = [ep for ep in entry_points(group=GROUP) if ep.name == key]
    if not matches:
        have = available(family).get(family, [])
        raise PluginError(f"no plugin {key!r} installed; installed {family} plugins: {have or 'none'}")
    if len(matches) > 1:
        raise PluginError(f"plugin {key!r} is registered by more than one package")
    factory = matches[0].load()
    return factory(**options)
