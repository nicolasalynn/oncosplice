"""Centralised model-weight management for the oncosplice predictors.

Every adapter resolves its weight directory through :func:`resolve_dir`,
which checks (in order):

1. ``$ONCOSPLICE_WEIGHTS_DIR/<model_name>/``  — explicit override.
2. ``~/.oncosplice/weights/<model_name>/``     — user-cache (populated by the
   ``oncosplice-download-weights`` CLI).
3. ``<package>/weights/<model_name>/``          — bundled with the wheel.
4. Falls back to the upstream package's resource_filename location
   (``spliceai/models/``, ``openspliceai/models/``, ``pangolin/models/``).

The user-cache scheme is the recommended one for distribution: PyPI's
60 MB-per-file limit makes shipping the full ~500 MB of model weights in the
wheel impractical, but ``pip install oncosplice && oncosplice-download-weights``
gives the same one-command experience.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.request
import zipfile
from pathlib import Path
from typing import Dict, List, Optional


# Package-bundled weights (if anyone wants to vendor a small model)
_PKG_WEIGHTS_DIR = Path(__file__).parent

# User cache
_USER_WEIGHTS_DIR = Path.home() / ".oncosplice" / "weights"


def manifest_path() -> Path:
    """Path to the JSON manifest of (model_name → download URL + filenames)."""
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


def resolve_dir(model_name: str, upstream_fallback: Optional[Path] = None) -> Optional[Path]:
    """Resolve the weight directory for ``model_name``.

    Returns the first path that exists, or ``upstream_fallback`` if nothing
    else is present. Returns ``None`` if no weights can be found anywhere.
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


# ---------------------------------------------------------------------------
# Downloader
# ---------------------------------------------------------------------------

def download_model(model_name: str, *, force: bool = False, verbose: bool = True) -> Path:
    """Download model weights to ``~/.oncosplice/weights/<model_name>/``.

    Reads URL + filenames from ``manifest.json``. If the manifest doesn't
    mention this model, the function raises with a helpful message.
    """
    manifest = load_manifest()
    if model_name not in manifest:
        raise KeyError(
            f"No download instructions for {model_name!r} in manifest. "
            f"Known models: {list(manifest)}. Add an entry to "
            f"{manifest_path()} or place weights manually in "
            f"{_USER_WEIGHTS_DIR / model_name}/"
        )
    entry = manifest[model_name]
    target_dir = _USER_WEIGHTS_DIR / model_name
    target_dir.mkdir(parents=True, exist_ok=True)

    files: List[dict] = entry.get("files", [])
    for f in files:
        url = f["url"]
        fname = f.get("filename") or Path(url).name
        dst = target_dir / fname
        if dst.exists() and not force:
            if verbose:
                print(f"  ✓ {fname} (already present)")
            continue
        if verbose:
            print(f"  ↓ {fname} ← {url}")
        with urllib.request.urlopen(url) as r, open(dst, "wb") as out:
            out.write(r.read())
        if fname.endswith(".zip") and entry.get("auto_unzip", True):
            with zipfile.ZipFile(dst, "r") as zf:
                zf.extractall(target_dir)
            dst.unlink()
    return target_dir


def download_all(*, force: bool = False, verbose: bool = True) -> Dict[str, Path]:
    """Download every model listed in the manifest."""
    manifest = load_manifest()
    if not manifest:
        raise RuntimeError(
            f"manifest.json is empty. Populate {manifest_path()} with model "
            f"entries before running download_all()."
        )
    out: Dict[str, Path] = {}
    for name in manifest:
        if verbose:
            print(f"[{name}]")
        out[name] = download_model(name, force=force, verbose=verbose)
    return out


def status() -> Dict[str, dict]:
    """Return per-model availability: where weights resolve from + size on disk."""
    manifest = load_manifest()
    rows = {}
    for name in manifest or ("openspliceai", "spliceai_keras", "pangolin", "spliceformer"):
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
    p = argparse.ArgumentParser(prog="oncosplice-download-weights",
                                description="Download model weights for the oncosplice splicing engines.")
    sub = p.add_subparsers(dest="cmd", required=False)
    p_dl = sub.add_parser("download", help="Download one or all models.")
    p_dl.add_argument("model", nargs="?", default="all",
                      help="Model name (openspliceai | spliceai_keras | pangolin | spliceformer | all). Default: all.")
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
