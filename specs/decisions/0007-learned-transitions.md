# Decision 0007: learned walk transitions (no LLM)

**Date:** 2026-11-02 · **Status:** accepted for development · **Affects:** C7 walks, C5 (composer role)

## Context
Two development sweeps (60 variants) plateau at ~17% full-chain / ~54% gold-passage recall@10 on
MuSiQue-train 3–4 hop questions. The walk adds nothing at @5 (the seeds fill the top 5), and it gains
less over its own dense baseline than HippoRAG 2 reports over its dense baseline. Fixed rules choose
the next hop from edge weight and similarity to the *question*, but the next hop depends on the
*current passage* and its bridge entity.

## Decision
Transitions become a log-linear model learned from gold chains:

    P(u → v | q) ∝ exp(θ · z(φ(q, u, v)))      z = feature standardization stored with θ

φ (computed by one shared core function for training and inference): cos(q, v), cos(u, v),
cos(q, u), edge weight, log df(bridge), bridge appears in the question, edge is a co-mention,
mention salience in u.

Training: listwise softmax over the out-edges of each gold passage u, the positive being the gold next
passage; L2 regularization. Training questions: MuSiQue **train**, 2,000 questions with seed 1,
excluding the development sample (seed 0). θ, feature means and deviations are written to a JSON
file referenced from config (constitution P13).

## Evaluation (development only)
On the development sample: learned transitions vs the best rule-based walk (β = 2, 5 seeds), at @5
and @10, full-chain and gold-passage recall. The held-out test stays untouched until a configuration
clearly beats the rule-based walk on development.
