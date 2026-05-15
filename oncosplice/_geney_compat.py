"""Internal compatibility shim for the optional ``geney`` dependency.

Most of oncosplice — single-/pair-/N-variant classification, ``scan()``,
``classify_dataframe()`` — runs without ``geney``. Only the protein-library
(alternative-isoform enumeration + Oncosplice protein-divergence score) path
requires geney. This module gates the geney-only symbols so the rest of the
package imports cleanly even when geney is absent.

What's still re-exported here:

- ``Mutation``, ``MutationalEvent``  — variant event objects (only used by
  ``Variant.to_event()`` / the legacy DataFrame conversion helpers).
- ``SpliceSimulator``, ``TranscriptLibrary``, ``Oncosplice`` — protein-library
  machinery, lazily-imported by ``OncospliceEngine._score_isoforms``.

``select_transcript`` is implemented locally (no geney dependency) because the
geney version has a bug in its fallback path that drops ~10% of genes whose
primary isoform doesn't contain the variant position.
"""
from __future__ import annotations


def _missing(name: str):
    """Return a placeholder that raises if accessed at runtime."""
    class _MissingGeney:
        def __init__(self, *a, **kw):
            raise ImportError(
                f"`{name}` requires the optional `geney` package. "
                "Install with `pip install oncosplice[protein]`."
            )
    _MissingGeney.__name__ = name
    return _MissingGeney


# ----- Variant event objects (geney.variants) ---------------------------------
try:
    from geney.variants import Mutation, MutationalEvent
except ImportError:
    Mutation = _missing("Mutation")
    MutationalEvent = _missing("MutationalEvent")

# ----- Oncosplice protein-divergence class ------------------------------------
try:
    from geney.oncosplice import Oncosplice
except ImportError:
    try:
        from geney.oncosplice.scoring import Oncosplice
    except ImportError:
        Oncosplice = _missing("Oncosplice")

# ----- TranscriptLibrary + SpliceSimulator ------------------------------------
try:
    from geney.splicing import SpliceSimulator, TranscriptLibrary
except ImportError:
    try:
        from geney.transcripts import TranscriptLibrary
        from geney.splice_graph import SpliceSimulator
    except ImportError:
        TranscriptLibrary = _missing("TranscriptLibrary")
        SpliceSimulator = _missing("SpliceSimulator")

# ----- splice-engine device setter -------------------------------------------
try:
    from geney.splicing.engines import set_splicing_device
except ImportError:
    try:
        from geney.engines import set_splicing_device
    except ImportError:
        def set_splicing_device(device):
            """No-op when geney isn't installed."""
            return None


# ─── select_transcript (LOCAL implementation, no geney needed) ───────────────
# geney's own implementation has a bug — its step-3 fallback iterates raw dict
# entries from ``gene.transcripts`` instead of materialised Transcript objects,
# so the bounds-containment check raises AttributeError on every alternative
# isoform and the function silently returns None. That drops major genes like
# ABL1/ABL2 (variant outside primary isoform's bounds but inside others).
# We override with a correct, self-contained implementation.

def select_transcript(gene, position: int, preferred_transcript_id=None):
    """Return a transcript of ``gene`` whose genomic bounds contain ``position``.

    Order tried: (1) ``preferred_transcript_id`` if given, (2) ``primary_transcript``,
    (3) every other transcript. Pre-mRNA windows are used for splicing prediction
    so intronic positions are valid — we only need a transcript that *spans* the
    variant, not one whose exons contain it.
    """
    def _bounds(t):
        s = getattr(t, "transcript_start", None)
        e = getattr(t, "transcript_end", None)
        if s is None or e is None:
            return None
        return (min(s, e), max(s, e))

    def _contains(t, pos):
        b = _bounds(t)
        return b is not None and b[0] <= pos <= b[1]

    if preferred_transcript_id:
        try:
            t = gene.transcript(preferred_transcript_id)
            if _contains(t, position):
                return t
        except Exception:
            pass

    try:
        prim_id = getattr(gene, "primary_transcript", None)
        if prim_id:
            t = gene.transcript(prim_id)
            if _contains(t, position):
                return t
    except Exception:
        pass

    try:
        for tid in list(getattr(gene, "transcripts", {})):
            try:
                t = gene.transcript(tid)
            except Exception:
                continue
            if _contains(t, position):
                return t
    except Exception:
        pass

    return None


__all__ = [
    "Mutation", "MutationalEvent",
    "Oncosplice",
    "TranscriptLibrary", "SpliceSimulator",
    "set_splicing_device",
    "select_transcript",
]
