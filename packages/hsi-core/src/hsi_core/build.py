"""Graph-build settings and inputs shared by the index builder, the service and the harness (A5)."""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict

from hsi_core import registry
from hsi_core.config import PluginSpec
from hsi_core.graph import GraphConfig, entity_key
from hsi_core.graph_stream import build_compact_graph
from hsi_core.transitions import LinearTransition, make_edge_scorer


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class GraphSection(_Strict):
    fan_out: int
    hub_degree_cap: int
    bidirectional: bool
    co_mention: bool
    weight_multi_token: float
    weight_single_token: float
    reverse_factor: float
    co_mention_multi_token: float
    co_mention_single_token: float
    hub_mode: Literal["drop", "facets"] = "drop"      # decision 0014
    facet_size: int | None = None
    facet_iterations: int | None = None
    mention_extractor: PluginSpec | None = None
    edge_scorer: str | None = None          # learned index-time edge selection model (decision 0010)

    def to_core(self) -> GraphConfig:
        return GraphConfig(**self.model_dump(exclude={"mention_extractor", "edge_scorer"}))


def graph_inputs(section: GraphSection, passages: list, vector_of=None) -> tuple[Any, Any]:
    """Mentions and the optional learned edge scorer, shared by both graph builders."""
    scorer = None
    if section.edge_scorer is not None:
        if vector_of is None:
            raise ValueError("a learned edge scorer needs passage vectors")
        scorer = make_edge_scorer(LinearTransition.load(section.edge_scorer), vector_of)
    mentions = None
    if section.mention_extractor is not None:
        extractor = registry.load("mention_extractor", section.mention_extractor.plugin,
                                  **section.mention_extractor.options)
        extractor.fit(p.text for p in passages)
        mentions = {p.id: {entity_key(s): sal for s, sal in extractor.extract(p.text)} for p in passages}
    return mentions, scorer


def build_compact(section: GraphSection, passages: list, ids: list[str], vecs: Any, vector_of=None) -> Any:
    """Streaming construction straight to CSR (decision 0013); same edges as `build_graph`."""
    mentions, scorer = graph_inputs(section, passages, vector_of)
    return build_compact_graph(passages, ids, vecs, section.to_core(), mentions=mentions, edge_scorer=scorer)
