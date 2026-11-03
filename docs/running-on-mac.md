# Running the experiments on an Apple Silicon Mac

The embedding model and the answer reader are much faster on the Mac's GPU (`mps`) than on a few cloud
CPUs. The configs in [`configs/mac/`](../configs/mac/) are the decision 0017 configs with only the device
changed (see the amendment in [`specs/decisions/0017-comparison-encoder-answers.md`](../specs/decisions/0017-comparison-encoder-answers.md)).

## 1. Set up (once)

Requirements: macOS on Apple Silicon (M1 or later), Python 3.11+, Docker Desktop or PostgreSQL 16.

```bash
git clone https://github.com/gowtham16-bhu/hsi.git ~/Projects/hsi
```

```bash
cd ~/Projects/hsi && python3 -m venv .venv && . .venv/bin/activate
```

```bash
pip install -e packages/hsi-core -e packages/hsi-plugins-basic -e packages/hsi-plugins-faiss \
            -e packages/hsi-plugins-postgres -e packages/hsi-plugins-onnx -e packages/hsi-plugins-st \
            -e packages/hsi-harness -e packages/hsi-connector-kit -e packages/hsi-plugins-osv \
            -e packages/hsi-service transformers pytest ruff import-linter
```

Check that PyTorch sees the GPU (it should print `True`):

```bash
python -c "import torch; print(torch.backends.mps.is_available())"
```

Start PostgreSQL and download the data:

```bash
eval "$(scripts/dev-postgres.sh)"
```

```bash
scripts/get-datasets.sh
```

## 2. Run

Per-question results of vector search and HSI on both test sets (needed by the answer test):

```bash
hsi-gate0 --config configs/mac/2wiki-hsi-per-question.yaml
```

```bash
hsi-gate0 --config configs/mac/musique-hsi-per-question.yaml
```

E3, answer quality (a local reader writes answers from each system's top 5 passages):

```bash
hsi-answer-eval --config configs/mac/answers-2wiki.yaml
```

```bash
hsi-answer-eval --config configs/mac/answers-musique.yaml
```

E2, the stronger encoder at scale:

```bash
hsi-bench-scale --config configs/mac/scale-large.yaml
```

Results are written to `runs/compare-mac/` and `runs/bench/`. Commit them, or send them back, and they
will be recorded in decision 0017.

## Notes
- The first run downloads the models from Hugging Face: `BAAI/bge-large-en-v1.5` (about 1.3 GB) and
  `Qwen/Qwen2.5-1.5B-Instruct` (about 3 GB).
- On `mps` the vectors can differ from the CPU ONNX run in the last decimal places, so recall may differ
  slightly from the cloud final test. Reports record the device, and results are labeled by where they ran.
