# Decision 0006: rank walk results by arrived mass, with a few kept seeds

**Date:** 2026-11-02 · **Status:** accepted for development (MuSiQue train split) · **Affects:** C7 walks

## Evidence (development sweep `79c3a6ba0fb6de57`, MuSiQue train, 1,000 questions)
With 10 seeds every PPR variant equals dense retrieval exactly (deep full-chain R@10 4.9%): restart
mass makes the seeds outrank everything the walk reaches, so with a 10-passage budget the walk adds
nothing. With 5 seeds it only competes for the remaining 5 slots. Question-awareness helped a little
(β = 2: 17.3 vs 15.4, not significant); β ≥ 20 hurt.

## Decision
Add `rank_by: arrived`: rank passages by r − α·s (mass that arrived through edges); place the top
`keep_seeds` dense seeds first and fill the rest of the budget by arrived mass. `rank_by: total` keeps
the old behaviour. Both are explicit config values; nothing is defaulted in code.

## Next sweep (development only)
β ∈ {0, 2} × seeds ∈ {5, 10, 20} × {total; arrived with keep_seeds ∈ {0, 3, 5}}, α = 0.5.
