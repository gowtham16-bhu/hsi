"""Command line: `hsi-gate0 --config configs/gate0/musique.yaml`."""
from __future__ import annotations

import argparse
import json
import sys

from hsi_harness.gate0 import load_config, run


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="hsi-gate0",
                                 description="Run a retrieval experiment from a config file (Gate 0, Gate 0′).")
    ap.add_argument("--config", required=True, help="YAML config file")
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    report = run(cfg)
    print(json.dumps({"decision": report["decision"], "paths": report["paths"],
                      "run_id": report.get("run_id")}, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
