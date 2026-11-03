# Gate 0′ decision (RQ1′, shortcut hops)

**Status:** DECIDED — **failed** on both datasets · **Date:** 2026-11-02

| Dataset | Config | Rule checks passed | Key numbers (full-chain R@10, deep slice) |
|---|---|---|---|
| MuSiQue (1,000 fresh, seed 1) | `689fe20491c6fce4` | 1 of 4 | shortcuts 5.8, passage hops 4.9 (2 rounds) / 3.6 (4 rounds), PPR 9.4 |
| 2Wiki (1,000 fresh, seed 1) | `d1d9a3aa96169c49` | 1 of 4 | shortcuts 2.6, passage hops 6.6 / 0.4, **PPR 67.7** |

Lessons: beam-style hopping commits to paths early and loses deep chains; an untrained query update
drifts (4 rounds worse than 2); PPR over the LLM-free graph is by far the strongest system on 2Wiki.
Next: decision 0004.
