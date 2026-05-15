"""Abstract base for splice-site predictor adapters.

Every concrete adapter (SpliceAI-Keras, OpenSpliceAI, Pangolin, Spliceformer, …)
implements one interface: take a nucleotide sequence, return per-base
acceptor & donor probabilities. The caller handles padding, mutation
application, and per-context bookkeeping — predictors are stateless wrt the
biology and only know how to encode a sequence and run the model.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np


@dataclass(frozen=True)
class SplicingPrediction:
    """Output of one predictor call.

    Arrays are 1-D and aligned position-for-position with the **biological**
    portion of the input — i.e. the central (`len(sequence) - 2*context_length`)
    positions. Caller maps array indices back to genomic coordinates.
    """
    acceptor: np.ndarray
    donor:    np.ndarray

    def __post_init__(self):
        if self.acceptor.shape != self.donor.shape:
            raise ValueError("acceptor and donor must have the same shape")

    @property
    def length(self) -> int:
        return int(self.acceptor.shape[0])


class SplicingPredictor(ABC):
    """Uniform interface for every splice-site model adapter.

    Concrete subclasses must implement :meth:`predict_one`. They may override
    :meth:`predict_batch` to provide a more efficient batched path; the
    default implementation just loops.

    Public callers should use :meth:`predict` / :meth:`predict_many` which
    handle padding and validation.
    """

    #: Short identifier for the engine (used by the factory + logging).
    name: str = "base"

    @property
    @abstractmethod
    def context_length(self) -> int:
        """Half-context (in bp) the model needs on each side of the region
        of interest. Caller is responsible for padding the input sequence
        with at least this many ``N``s on each side; the predictor returns
        probabilities for the *unpadded* central region only.
        """

    @abstractmethod
    def is_available(self) -> bool:
        """Return ``True`` iff the underlying model / weights are importable
        and runnable on this machine. Should not actually load the model.
        """

    @abstractmethod
    def predict_one(self, sequence: str) -> SplicingPrediction:
        """Run the model on one padded sequence.

        ``sequence`` is uppercase A/C/G/T/N, length ≥ ``2*context_length + 1``.
        Returns acceptor & donor probability arrays of length
        ``len(sequence) - 2*context_length``.
        """

    def predict_batch(self, sequences: Sequence[str]) -> List[SplicingPrediction]:
        """Run the model on many sequences. Default: loop ``predict_one``.

        Subclasses with a real batched path (e.g. OpenSpliceAI on GPU) should
        override this for ~10–50× speedups.
        """
        return [self.predict_one(s) for s in sequences]

    def predict(self, sequence: str) -> SplicingPrediction:
        """Public single-sequence entry point — validates inputs then calls
        :meth:`predict_one`.
        """
        self._validate_seq(sequence)
        return self.predict_one(sequence)

    def predict_many(self, sequences: Sequence[str]) -> List[SplicingPrediction]:
        """Public batched entry point — validates inputs then calls
        :meth:`predict_batch`.
        """
        for s in sequences:
            self._validate_seq(s)
        return self.predict_batch(sequences)

    def _validate_seq(self, seq: str) -> None:
        cl = self.context_length
        if len(seq) < 2 * cl + 1:
            raise ValueError(
                f"sequence length ({len(seq)}) is below the minimum required "
                f"by {self.name} ({2*cl + 1}). Pad with 'N' on each side."
            )

    # Convenience: predictors are usually constructed implicitly via the
    # factory, so a useful repr makes debugging easier.
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r}, context_length={self.context_length})"
