"""Pangolin (Zeng & Li 2022) splice-site predictor adapter.

Delegates to the proven ``geney.engines.pangolin_predict_probs`` implementation,
which handles the 40-model ensemble loading, one-hot encoding, multi-tissue
channel aggregation, and per-base probability extraction.

Channel layout (per single model output, shape ``(1, 12, L)``):

- ``[0, 3, 6, 9]``  — site usage per tissue (heart, liver, brain, testis)
- ``[1, 4, 7, 10]`` — acceptor gain per tissue
- ``[2, 5, 8, 11]`` — donor gain per tissue

We aggregate across tissues with ``max`` (matches geney's convention) and
return ``(acceptor, donor)`` arrays sliced to the biological region.
"""
from __future__ import annotations

from typing import List, Sequence

import numpy as np

from .base import SplicingPrediction, SplicingPredictor


class Pangolin(SplicingPredictor):
    """Pangolin (multi-tissue, 40-model ensemble) adapter."""

    name = "pangolin"
    _CONTEXT = 5000  # 5,000 bp on each side

    def __init__(self, device: str | None = None):
        self._device_override = device

    @property
    def context_length(self) -> int:
        return self._CONTEXT

    def is_available(self) -> bool:
        try:
            import pangolin.model  # noqa: F401
            import torch  # noqa: F401
            return True
        except ImportError:
            return False

    # ------------------------------------------------------------------
    def predict_one(self, sequence: str) -> SplicingPrediction:
        """Run Pangolin via geney's proven inference path.

        Returns arrays of length ``len(sequence) - 2 * context_length`` —
        the biological centre with the context portion stripped.
        """
        # geney 1.x had this in geney.engines; 2.x moved it under
        # geney.splicing.engines. Try both.
        try:
            from geney.engines import pangolin_predict_probs
        except ImportError:
            from geney.splicing.engines import pangolin_predict_probs

        # geney returns (donor_probs, acceptor_probs) as Python lists matching
        # the full input length.
        donor, acceptor = pangolin_predict_probs(sequence)
        donor = np.asarray(donor, dtype=np.float32)
        acceptor = np.asarray(acceptor, dtype=np.float32)

        cl = self._CONTEXT
        if donor.shape[0] == len(sequence):
            donor = donor[cl:-cl]
            acceptor = acceptor[cl:-cl]
        elif donor.shape[0] != len(sequence) - 2 * cl:
            raise RuntimeError(
                f"Pangolin output length {donor.shape[0]} matches neither "
                f"len(seq)={len(sequence)} nor len(seq)-2*cl={len(sequence)-2*cl}"
            )
        return SplicingPrediction(acceptor=acceptor, donor=donor)

    def predict_batch(self, sequences: Sequence[str]) -> List[SplicingPrediction]:
        # geney's function processes one sequence at a time; loop here.
        return [self.predict_one(s) for s in sequences]
