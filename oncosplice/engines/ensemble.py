"""Ensemble adapter — averages predictions across multiple :class:`SplicingPredictor` s.

Use case: pool SpliceAI + Pangolin (or any other combination) into a single
predictor so the rest of the pipeline doesn't care. All component predictors
must share the same ``context_length`` — the ensemble will validate this.
"""
from __future__ import annotations

from typing import List, Sequence

import numpy as np

from .base import SplicingPrediction, SplicingPredictor


class EnsemblePredictor(SplicingPredictor):
    """Average-of-models ensemble of :class:`SplicingPredictor`s."""

    def __init__(self, predictors: Sequence[SplicingPredictor], name: str | None = None):
        if not predictors:
            raise ValueError("EnsemblePredictor needs at least one component.")
        cls = {p.context_length for p in predictors}
        if len(cls) != 1:
            raise ValueError(
                f"All ensemble members must share context_length; got {sorted(cls)}."
            )
        self._predictors = list(predictors)
        self.name = name or "ensemble:" + ",".join(p.name for p in predictors)
        self._cl = next(iter(cls))

    @property
    def context_length(self) -> int:
        return self._cl

    def is_available(self) -> bool:
        return all(p.is_available() for p in self._predictors)

    def predict_one(self, sequence: str) -> SplicingPrediction:
        preds = [p.predict(sequence) for p in self._predictors]
        acc = np.mean([p.acceptor for p in preds], axis=0)
        don = np.mean([p.donor    for p in preds], axis=0)
        return SplicingPrediction(acceptor=acc, donor=don)

    def predict_batch(self, sequences: Sequence[str]) -> List[SplicingPrediction]:
        if not sequences:
            return []
        per_model = [p.predict_many(sequences) for p in self._predictors]
        out: List[SplicingPrediction] = []
        for i in range(len(sequences)):
            acc = np.mean([per_model[k][i].acceptor for k in range(len(self._predictors))], axis=0)
            don = np.mean([per_model[k][i].donor    for k in range(len(self._predictors))], axis=0)
            out.append(SplicingPrediction(acceptor=acc, donor=don))
        return out
