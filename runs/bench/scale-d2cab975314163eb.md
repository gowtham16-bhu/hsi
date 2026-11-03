# Scaling benchmark `d2cab975314163eb`

200 musique questions; encoder `onnx:BAAI/bge-small-en-v1.5:512:cls`; no LLM calls.

| Passages | Edges | Bytes/edge | Graph build s | Peak RSS GB | Pushes/query p50 (max) | Vectors read p50 | Seed search p50 ms | Push p50 ms | Global p50 ms | GP@5 dense | push | global |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 10,000 | 302,254 | 24 | 10 | 1.2 | 521 (865) | 3306 | 18.6 | 55.4 | 79.0 | 55.9 | 64.8 | 64.8 |
| 40,000 | 1,245,013 | 24 | 48 | 1.6 | 558 (844) | 4782 | 6.9 | 57.9 | 296.9 | 48.5 | 55.3 | 55.3 |
| 140,000 | 4,422,354 | 24 | 260 | 3.1 | 554 (871) | 5416 | 13.7 | 65.4 |  | 44.8 | 45.1 |  |
