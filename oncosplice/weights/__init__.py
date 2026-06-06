"""Centralised model-weight management for the oncosplice predictors.

Weights are distributed from a single Hugging Face Hub repository,
:data:`HF_REPO_ID`, with one sub-folder per model
(``openspliceai/``, ``spliceai_pytorch/``, ``pangolin/``, ``spliceformer/`` …).
They are downloaded into a user-cache and resolved from there at runtime.

Every adapter resolves its weight directory through :func:`resolve_dir`
(cheap, no network) and, at model-load time, through :func:`ensure_dir`
(resolves, and auto-downloads on a miss unless disabled). :func:`resolve_dir`
checks, in order:

1. ``$ONCOSPLICE_WEIGHTS_DIR/<model_name>/``  — explicit override.
2. ``~/.oncosplice/weights/<model_name>/``     — user-cache (populated by the
   ``oncosplice-download-weights`` CLI or by auto-download).
3. ``<package>/weights/<model_name>/``          — bundled with the wheel
   (only used for very small vendored models).

Auto-download is on by default (mirroring the Hugging Face / torch.hub
experience); set ``ONCOSPLICE_AUTO_DOWNLOAD=0`` to require an explicit
``oncosplice-download-weights`` step instead (useful for offline / CI).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

#: Hugging Face Hub repo hosting every model's weights (one sub-folder each).
HF_REPO_ID = os.environ.get("ONCOSPLICE_HF_REPO", "nicolynnvila/oncosplice-weights")

# Package-bundled weights (only viable for very small vendored models).
_PKG_WEIGHTS_DIR = Path(__file__).parent

# User cache.
_USER_WEIGHTS_DIR = Path.home() / ".oncosplice" / "weights"


def manifest_path() -> Path:
    """Path to the JSON manifest of model metadata (sub-folder, license, …)."""
    return _PKG_WEIGHTS_DIR / "manifest.json"


def load_manifest() -> Dict[str, dict]:
    """Read manifest.json. Returns ``{}`` if not present.

    Keys starting with ``_`` (comments) are filtered out.
    """
    p = manifest_path()
    if not p.exists():
        return {}
    try:
        raw = json.loads(p.read_text())
    except Exception:
        return {}
    return {k: v for k, v in raw.items() if not k.startswith("_")}


def _subfolder(model_name: str) -> str:
    """The repo sub-folder for a model (defaults to the model name)."""
    entry = load_manifest().get(model_name, {})
    return entry.get("subfolder", model_name)


def _auto_download_enabled() -> bool:
    return os.environ.get("ONCOSPLICE_AUTO_DOWNLOAD", "1").strip().lower() not in (
        "0", "false", "no", "off", "",
    )


def resolve_dir(model_name: str, upstream_fallback: Optional[Path] = None) -> Optional[Path]:
    """Resolve the weight directory for ``model_name`` without any network I/O.

    Returns the first existing, non-empty path, or ``upstream_fallback`` if
    nothing else is present. Returns ``None`` if no weights are found anywhere.
    """
    override = os.environ.get("ONCOSPLICE_WEIGHTS_DIR")
    candidates: List[Path] = []
    if override:
        candidates.append(Path(override) / model_name)
    candidates.append(_USER_WEIGHTS_DIR / model_name)
    candidates.append(_PKG_WEIGHTS_DIR / model_name)
    if upstream_fallback is not None:
        candidates.append(upstream_fallback)
    for c in candidates:
        if c.exists() and any(c.iterdir() if c.is_dir() else [True]):
            return c
    return None


def ensure_dir(model_name: str, *, auto_download: Optional[bool] = None) -> Optional[Path]:
    """Resolve the weight directory, downloading from the Hub on a miss.

    Call this at model-load time (not in ``is_available``): a miss triggers a
    download unless auto-download is disabled. ``auto_download=None`` (default)
    consults ``ONCOSPLICE_AUTO_DOWNLOAD``.
    """
    d = resolve_dir(model_name)
    if d is not None:
        return d
    do_download = _auto_download_enabled() if auto_download is None else auto_download
    if do_download and model_name in load_manifest():
        print(
            f"[oncosplice] weights for {model_name!r} not found locally; "
            f"downloading from {HF_REPO_ID} (set ONCOSPLICE_AUTO_DOWNLOAD=0 to disable)…"
        )
        download_model(model_name)
        return resolve_dir(model_name)
    return None


# ---------------------------------------------------------------------------
# Downloader
# ---------------------------------------------------------------------------

def download_model(model_name: str, *, force: bool = False, verbose: bool = True) -> Path:
    """Download a model's weights from the Hub into the user cache.

    Files land in ``~/.oncosplice/weights/<model_name>/`` (the repo sub-folder
    is mapped to the cache sub-folder of the same name).
    """
    manifest = load_manifest()
    if model_name not in manifest:
        raise KeyError(
            f"Unknown model {model_name!r}. Known models: {sorted(manifest)}. "
            f"Add an entry to {manifest_path()} or place weights manually in "
            f"{_USER_WEIGHTS_DIR / model_name}/"
        )
    try:
        from huggingface_hub import snapshot_download
    except ImportError as e:
        raise RuntimeError(
            "huggingface_hub is required to download weights "
            "(`pip install huggingface_hub`)."
        ) from e

    repo_id = manifest[model_name].get("repo_id", HF_REPO_ID)
    subfolder = _subfolder(model_name)
    target_root = _USER_WEIGHTS_DIR
    target_root.mkdir(parents=True, exist_ok=True)

    if verbose:
        print(f"  ↓ {repo_id}:{subfolder}/  →  {target_root / model_name}")

    # Pull just this model's sub-folder, copying real files (not symlinks) into
    # the cache so the layout matches resolve_dir's expectations.
    snapshot_download(
        repo_id=repo_id,
        allow_patterns=[f"{subfolder}/*"],
        local_dir=str(target_root),
        force_download=force,
    )
    # If the repo sub-folder name differs from our cache name, normalise it.
    src = target_root / subfolder
    dst = target_root / model_name
    if src != dst and src.exists():
        dst.mkdir(parents=True, exist_ok=True)
        for f in src.iterdir():
            f.rename(dst / f.name)
        try:
            src.rmdir()
        except OSError:
            pass
    return dst


def download_all(*, force: bool = False, verbose: bool = True) -> Dict[str, Path]:
    """Download every model listed in the manifest."""
    manifest = load_manifest()
    if not manifest:
        raise RuntimeError(
            f"manifest.json is empty. Populate {manifest_path()} before "
            f"running download_all()."
        )
    out: Dict[str, Path] = {}
    for name in manifest:
        if verbose:
            print(f"[{name}]")
        out[name] = download_model(name, force=force, verbose=verbose)
    return out


def status() -> Dict[str, dict]:
    """Return per-model availability: where weights resolve from + file count."""
    manifest = load_manifest()
    rows = {}
    for name in manifest or ("openspliceai", "spliceai_pytorch", "pangolin", "spliceformer"):
        d = resolve_dir(name)
        rows[name] = {
            "resolved": str(d) if d else None,
            "exists":   bool(d and d.exists()),
            "n_files":  sum(1 for _ in d.iterdir()) if (d and d.is_dir() and d.exists()) else 0,
        }
    return rows


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _cli(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(
        prog="oncosplice-download-weights",
        description=f"Download model weights for the oncosplice splicing engines "
                    f"(from {HF_REPO_ID}).",
    )
    sub = p.add_subparsers(dest="cmd", required=False)
    p_dl = sub.add_parser("download", help="Download one or all models.")
    p_dl.add_argument("model", nargs="?", default="all",
                      help="Model name (openspliceai | spliceai_pytorch | "
                           "spliceai_keras | pangolin | spliceformer | all). Default: all.")
    p_dl.add_argument("--force", action="store_true", help="Re-download even if present.")
    sub.add_parser("status", help="Show where each model resolves from.")
    args = p.parse_args(argv)

    if args.cmd in (None, "download"):
        target = getattr(args, "model", "all")
        force = getattr(args, "force", False)
        if target == "all":
            download_all(force=force)
        else:
            download_model(target, force=force)
        return 0
    if args.cmd == "status":
        rows = status()
        for name, info in rows.items():
            marker = "✓" if info["exists"] else "✗"
            print(f"  {marker} {name:18s} {info.get('resolved') or '(not found)'} "
                  f"[{info['n_files']} files]")
        return 0
    p.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(_cli())
