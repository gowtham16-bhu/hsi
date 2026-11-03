"""Convert a HippoRAG 2 corpus file into the JSONL format of connector:jsonl (all documents public).

    python scripts/make-demo-jsonl.py data/hipporag2/2wikimultihopqa_corpus.json data/demo/2wiki-corpus.jsonl
"""
import json
import sys
from pathlib import Path

src, dst = Path(sys.argv[1]), Path(sys.argv[2])
corpus = json.loads(src.read_text(encoding="utf-8"))
dst.parent.mkdir(parents=True, exist_ok=True)
with dst.open("w", encoding="utf-8") as f:
    for i, p in enumerate(corpus):
        f.write(json.dumps({"id": f"p{i}", "title": p["title"], "text": p["text"], "acl": {"public": True}}) + "\n")
print(f"wrote {len(corpus)} documents to {dst}")
