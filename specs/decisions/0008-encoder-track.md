# Decision 0008: the encoder is part of the engine

**Date:** 2026-11-02 · **Status:** accepted for development

## Context
No configuration so far outperforms published systems. On MuSiQue, dense retrieval with our
bge-small (33M parameters) is ~14 points of gold-passage recall@5 below the 7B encoder used by
HippoRAG 2, and our walk adds less on top of its dense baseline than HippoRAG 2 adds on top of its own.

## Decision: two levers, both measured on the development split
1. **Encoder ladder** (no new method): bge-small → bge-base (110M) → bge-large (335M), all through the
   ONNX plugin, same graph and walks. Measures how much of the gap is encoder size.
2. **Graph-smoothed passage vectors** (new, index time, no LLM, no re-encoding):

       v'_p = normalize((1 − γ)·v_p + γ · Σ_{n ∈ N(p)} w(p, n)·v_n / Σ w(p, n))     repeated `steps` times

   Hop-2/3 passages absorb part of their neighbours' meaning, so they become findable from the
   question. Cost O(steps · N · m) at index time; zero at query time.

Lever 3 (later, if 1–2 are not enough): fine-tune the encoder contrastively on MuSiQue train gold
passages, no LLM.
