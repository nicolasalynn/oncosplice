#!/usr/bin/env python
"""Upload oncosplice model weights to the Hugging Face Hub.

Stages every model's weight files into the repo layout that
``oncosplice.weights`` expects (one sub-folder per model) and pushes them to
``--repo-id`` (default: the package's ``HF_REPO_ID``).

Sources are resolved from the locally-installed upstream packages / checkouts:

  pangolin         <site-packages>/pangolin/models/final.{j}.{i}.3   (40 files)
  openspliceai     <site-packages>/geney/models/openspliceai-mane/10000nt/*.pt
  spliceai_pytorch <site-packages>/geney/models/spliceai-original/10000nt/*.pt
  spliceai_keras   <site-packages>/spliceai/models/*.h5
  spliceformer     $SPLICEFORMER_REPO_DIR/Results/PyTorch_Models/transformer_encoder_45k_171022_*

Authenticate first (``huggingface-cli login`` or ``HF_TOKEN``), then e.g.::

    python scripts/upload_weights.py --public
    python scripts/upload_weights.py --models pangolin spliceformer --dry-run
"""
from __future__ import annotations

import argparse
import glob
import os
import sys
from pathlib import Path
from typing import Callable, Dict, List

# Default repo id mirrors oncosplice.weights.HF_REPO_ID.
try:
    from oncosplice.weights import HF_REPO_ID as _DEFAULT_REPO
except Exception:
    _DEFAULT_REPO = "nicolynnvila/oncosplice-weights"


def _site_packages() -> Path:
    import numpy  # any installed dep
    return Path(numpy.__file__).resolve().parent.parent


def _spliceformer_repo() -> Path:
    return Path(
        os.environ.get(
            "SPLICEFORMER_REPO_DIR",
            str(Path.home() / "Documents/phd/libraries/Spliceformer"),
        )
    )


def _glob(pattern: str) -> List[Path]:
    return sorted(Path(p) for p in glob.glob(pattern))


# model_name -> (sub-folder, callable returning the list of source files)
SOURCES: Dict[str, Callable[[], List[Path]]] = {
    "pangolin":         lambda: _glob(str(_site_packages() / "pangolin/models/final.*.3")),
    "openspliceai":     lambda: _glob(str(_site_packages() / "geney/models/openspliceai-mane/10000nt/*.pt")),
    "spliceai_pytorch": lambda: _glob(str(_site_packages() / "geney/models/spliceai-original/10000nt/*.pt")),
    "spliceai_keras":   lambda: _glob(str(_site_packages() / "spliceai/models/*.h5")),
    "spliceformer":     lambda: _glob(str(_spliceformer_repo() / "Results/PyTorch_Models/transformer_encoder_45k_171022_*")),
}

# Sanity-check expected counts so a half-installed source is caught early.
EXPECTED = {
    "pangolin": 40,
    "openspliceai": 5,
    "spliceai_pytorch": 5,
    "spliceai_keras": 5,
    "spliceformer": 10,
}

REPO_README = """\
---
license: other
tags:
  - rna-splicing
  - splice-site-prediction
  - bioinformatics
---

# oncosplice model weights

Weight files for the splice-site engines used by
[oncosplice](https://github.com/nicolasalynn/oncosplice). One sub-folder per
model; oncosplice downloads them on first use.

## Provenance & licensing of the trained weights

| sub-folder | model | weights © / license |
|---|---|---|
| `openspliceai/` | OpenSpliceAI (MANE) | OpenSpliceAI authors — see upstream |
| `spliceai_pytorch/` | SpliceAI weights (Keras→PyTorch) | Jaganathan et al. 2019 (Illumina) |
| `spliceai_keras/` | SpliceAI original Keras `.h5` | Jaganathan et al. 2019 (Illumina) |
| `pangolin/` | Pangolin 40-model ensemble | Zeng & Li 2022 — **GPL-3.0** |
| `spliceformer/` | Spliceformer 45k base ensemble | Jónsson et al. 2024 — MIT |

These weights are redistributed for **non-commercial research use** under their
respective original terms. The Pangolin weights are GPL-3.0 (© Zeng & Li 2022);
the oncosplice architecture code that runs them is an independent MIT
re-implementation. The Spliceformer base ensemble (`transformer_encoder_45k_171022_*`)
is the general-purpose model — the blood-finetuned variants are intentionally
not included.
"""


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Upload oncosplice weights to the HF Hub.")
    p.add_argument("--repo-id", default=_DEFAULT_REPO)
    vis = p.add_mutually_exclusive_group()
    vis.add_argument("--private", dest="private", action="store_true", default=None,
                     help="Create the repo private.")
    vis.add_argument("--public", dest="private", action="store_false",
                     help="Create the repo public (needed for pip-install auto-download).")
    p.add_argument("--models", nargs="*", default=list(SOURCES),
                   help="Subset of models to upload (default: all).")
    p.add_argument("--dry-run", action="store_true", help="Resolve + validate sources, upload nothing.")
    args = p.parse_args(argv)

    if args.private is None and not args.dry_run:
        p.error("choose --public or --private (public is needed for auto-download).")

    # Resolve + validate every requested model's sources first.
    plan: Dict[str, List[Path]] = {}
    problems = []
    for name in args.models:
        if name not in SOURCES:
            p.error(f"unknown model {name!r}; valid: {list(SOURCES)}")
        files = SOURCES[name]()
        plan[name] = files
        exp = EXPECTED.get(name)
        flag = "" if (exp is None or len(files) == exp) else f"  ⚠ expected {exp}"
        print(f"  {name:18s} {len(files):3d} file(s){flag}")
        if not files:
            problems.append(f"{name}: no source files found")
        elif exp is not None and len(files) != exp:
            problems.append(f"{name}: found {len(files)}, expected {exp}")
    if problems:
        print("\nSource problems:\n  - " + "\n  - ".join(problems), file=sys.stderr)
        if not args.dry_run:
            return 2

    if args.dry_run:
        print("\n[dry-run] nothing uploaded.")
        return 0

    from huggingface_hub import CommitOperationAdd, HfApi
    api = HfApi()
    print(f"\nCreating repo {args.repo_id} (private={args.private})…")
    api.create_repo(args.repo_id, repo_type="model", private=args.private, exist_ok=True)

    # Model card / README at the repo root.
    api.upload_file(
        path_or_fileobj=REPO_README.encode(), path_in_repo="README.md",
        repo_id=args.repo_id, repo_type="model", commit_message="Add model card",
    )

    for name, files in plan.items():
        ops = [
            CommitOperationAdd(path_in_repo=f"{name}/{f.name}", path_or_fileobj=str(f))
            for f in files
        ]
        print(f"Uploading {name}: {len(ops)} file(s)…")
        api.create_commit(
            repo_id=args.repo_id, repo_type="model", operations=ops,
            commit_message=f"Upload {name} weights ({len(ops)} files)",
        )
    print(f"\nDone → https://huggingface.co/{args.repo_id}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
