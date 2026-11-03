# Scaling benchmark `b1ebcde92429e26e`

200 musique questions; encoder `onnx:BAAI/bge-small-en-v1.5:512:cls`; no LLM calls.

| Passages | Edges | Bytes/edge | Graph build s | Peak RSS GB | Pushes/query p50 (max) | Vectors read p50 | Seed search p50 ms | Push p50 ms | Global p50 ms | GP@5 dense | push | global |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 10,000 | 577,117 | 24 | 9 | 1.2 | 653 (1055) | 5532 | 1.7 | 74.2 | 121.2 | 55.9 | 65.8 | 65.8 |
| 140,000 | 8,712,756 | 24 | 247 | 3.2 | 682 (1042) | 11770 | 13.3 | 96.8 |  | 44.8 | 46.3 |  |
