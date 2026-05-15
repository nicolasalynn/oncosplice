"""Internal compatibility shim — geney has two coexisting layouts in the wild.

The legacy installed wheel uses a flat layout (``geney/pipelines.py``,
``geney/transcripts.py``, …) while the modular development checkout uses a
package layout (``geney/pipelines/__init__.py``, ``geney/splicing/__init__.py``,
…).  This module imports the symbols we need from whichever layout is active,
so the rest of oncosplice doesn't have to care.
"""
from __future__ import annotations

# ----- MutationalEvent / Mutation ---------------------------------------------
try:
    from geney.variants import Mutation, MutationalEvent  # both layouts
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "geney.variants is required (install geney from the modular or "
        "flat-layout package)."
    ) from e

# ----- Oncosplice protein-divergence class -----------------------------------
try:
    from geney.oncosplice import Oncosplice  # modular layout (re-exports)
except ImportError:
    try:
        from geney.oncosplice.scoring import Oncosplice  # alt path
    except ImportError:
        from geney.oncosplice import Oncosplice  # flat layout single-file

# ----- TranscriptLibrary + SpliceSimulator ------------------------------------
try:
    from geney.splicing import TranscriptLibrary, SpliceSimulator
except ImportError:  # flat layout
    from geney.transcripts import TranscriptLibrary  # type: ignore
    from geney.splice_graph import SpliceSimulator   # type: ignore

# ----- splice-engine setters --------------------------------------------------
try:
    from geney.splicing.engines import set_splicing_device  # modular
except ImportError:  # flat: function may live in geney.engines
    try:
        from geney.engines import set_splicing_device  # type: ignore
    except ImportError:
        def set_splicing_device(device):  # type: ignore
            """No-op fallback when the installed geney version doesn't expose
            a device-setter."""
            return None

# ----- select_transcript -----------------------------------------------------
# The geney implementation has a bug: its step-3 fallback iterates raw dict
# entries from ``gene.transcripts`` instead of materialised Transcript objects,
# so the bounds-containment check raises AttributeError on every alternative
# isoform and the function silently returns None. That dropped major genes like
# ABL1/ABL2 (variant outside the primary isoform's bounds but inside other
# isoforms) and projected to ~80K lost TCGA pairs. We override with a correct,
# self-contained implementation: check primary first, then every transcript by
# its ``transcript_start``/``transcript_end`` bounds.

def select_transcript(gene, position: int, preferred_transcript_id=None):
    """Return any transcript of ``gene`` whose genomic bounds contain ``position``.

    Order tried: (1) ``preferred_transcript_id`` if given, (2) ``primary_transcript``,
    (3) every other transcript in ``gene.transcripts``. Pre-mRNA windows are
    used for splicing prediction so intronic positions are valid — we only need
    a transcript that *spans* the variant, not one whose exons contain it.
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

    # (1) explicit preference
    if preferred_transcript_id:
        try:
            t = gene.transcript(preferred_transcript_id)
            if _contains(t, position):
                return t
        except Exception:
            pass

    # (2) primary / MANE transcript
    try:
        prim_id = getattr(gene, "primary_transcript", None)
        if prim_id:
            t = gene.transcript(prim_id)
            if _contains(t, position):
                return t
    except Exception:
        pass

    # (3) sweep every transcript by ID — materialise each via ``gene.transcript(tid)``
    try:
        tids = list(getattr(gene, "transcripts", {}))
        for tid in tids:
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
