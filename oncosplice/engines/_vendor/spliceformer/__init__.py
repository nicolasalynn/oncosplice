"""Vendored Spliceformer model definition.

Source : https://github.com/benniatli/Spliceformer  (Code/src/)
Paper  : Jónsson et al., "A transformer-based model for splice-site
         prediction", Communications Biology (2024).
         https://doi.org/10.1038/s42003-024-07298-9
License : MIT — Copyright (c) 2024 Benedikt Atli Jónsson (see LICENSE in this
          directory).

``model.py`` and ``weight_init.py`` are copied verbatim from the upstream
repository (only this provenance header is added) so oncosplice can run the
released Spliceformer weights without a separate source checkout.
"""
from .model import SpliceFormer

__all__ = ["SpliceFormer"]
