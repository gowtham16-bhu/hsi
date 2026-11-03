"""Command line: `hsi-index build --config index.yaml` and `hsi-serve --config serve.yaml --transport rest`."""
from __future__ import annotations

import argparse
import json
import sys

from hsi_core.config import load_yaml


def index_main(argv: list[str] | None = None) -> int:
    from hsi_service.index_build import IndexConfig, build
    ap = argparse.ArgumentParser(prog="hsi-index", description="Build an HSI index from connectors (no LLM).")
    ap.add_argument("command", choices=["build"])
    ap.add_argument("--config", required=True)
    args = ap.parse_args(argv)
    manifest = build(IndexConfig.model_validate(load_yaml(args.config)))
    print(json.dumps({k: manifest[k] for k in ("index_id", "documents", "passages", "graph", "build_seconds")},
                     indent=2, default=str))
    return 0


def serve_main(argv: list[str] | None = None) -> int:
    from hsi_service.service import RetrievalService, ServeConfig
    ap = argparse.ArgumentParser(prog="hsi-serve", description="Serve an HSI index over REST or MCP.")
    ap.add_argument("--config", required=True)
    ap.add_argument("--transport", choices=["rest", "mcp-stdio", "mcp-http"], required=True)
    ap.add_argument("--host", required=True)
    ap.add_argument("--port", type=int, required=True)
    args = ap.parse_args(argv)
    service = RetrievalService(ServeConfig.model_validate(load_yaml(args.config)))
    if args.transport == "rest":
        import uvicorn

        from hsi_service.api import create_app
        uvicorn.run(create_app(service), host=args.host, port=args.port)
    else:
        from hsi_service.mcp_server import create_mcp
        mcp = create_mcp(service)
        mcp.settings.host, mcp.settings.port = args.host, args.port
        mcp.run(transport="stdio" if args.transport == "mcp-stdio" else "streamable-http")
    return 0


if __name__ == "__main__":
    sys.exit(serve_main())
