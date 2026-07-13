"""Build the study-material vector index.

Run: ``uv run python -m scripts.build_index [--dataset all]``
Rebuilds embeddings for all materials and writes the catalog hash so the index only
rebuilds when the material set changes.
"""

import argparse

from src.retrieval.indexer import build_index
from src.utils.config import load_config
from src.utils.data_loader import load_dataset
from src.utils.logging_config import setup_logging


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="build_index")
    ap.add_argument("--dataset", default="all", choices=["sample", "synthetic", "all"])
    args = ap.parse_args(argv)
    setup_logging("INFO")
    cfg = load_config()
    repo = load_dataset(args.dataset)
    n = build_index(repo.materials(), cfg)
    print(f"indexed {n} materials from dataset '{args.dataset}'")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
