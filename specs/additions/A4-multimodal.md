# A4 — Addition: Multimodal Ingestion

Audio, video, images, GIFs, charts and diagrams enter the core as text passages with a pointer back to the exact place in the media. The core algorithm does not change. Planned for product v1.1, after Gate 2. Not part of the thesis claims.

## 1. Extractor plugins
A new plugin family, `Extractor.extract(bytes, mime) -> list[ExtractedSegment]`, runs before the parser stage. All extractors run locally.

| Media | Extractor | Segment | Media reference |
|---|---|---|---|
| Audio | speech-to-text + speaker detection | transcript split by speaker turns | file, start–end time, speaker |
| Video | speech-to-text on audio + keyframes at scene changes + OCR on frames | time segment of speech and on-screen text | time range, frame |
| Image, screenshot, scan | layout-aware OCR | text block | image region |
| GIF | frames, near-duplicates dropped, OCR | as short video | frame |
| Chart image | chart-to-table extraction + OCR of axes and legend | structured facts + short summary | chart region |
| Diagram source (draw.io, SVG, Mermaid, Visio) | direct parse of nodes and edges | exact structured facts | diagram element |
| Diagram image | OCR + arrow detection | facts with lower confidence | image region |
| Graph data (graph databases, RDF) | direct import | structured facts | node or edge ID |

## 2. Core additions (C1 only)
- `MediaRef = (uri, media_type, time_range?, region?, frame?)` on `Passage` and `StructuredFact`.
- `extraction_confidence ∈ [0, 1]` on `Passage` and `StructuredFact`. It multiplies into edge weights, so `log P(c)` falls for chains through uncertain extractions.

No other core change. Verification (C9) checks quotes against the extracted text, and surfaces link evidence to the exact time or region.

## 3. Rules
- **Zero LLM calls at index time still holds by default.** Speech-to-text, OCR and chart extraction use small local models, not LLMs. Image captioning with a vision-language model is an optional plugin, off by default, on-prem only, and recorded in the index metadata when used.
- **Secrets.** OCR and transcript text pass through the redactor before indexing (A3, S4).
- **Privacy.** Each connector sets retention for recordings and images. Media keep the ACL of their source.

## 4. Acceptance
- Every extracted passage carries a valid `MediaRef`, and the surface opens the referenced time or region.
- A chain through a low-confidence extraction has a lower `log P(c)` than the same chain through clean text (unit test).
- With captioning disabled, the index records zero generative-model calls.
