"""Self-contained-engine guards.

Two things these tests pin down after moving Pangolin/Spliceformer off external
checkouts:

1. The vendored (MIT) Spliceformer model imports and runs a forward pass with no
   weights and no cloned repo.
2. The independent (MIT) Pangolin re-implementation is numerically identical to
   the established `geney` aggregation — *when* the upstream `pangolin` package
   and its weights happen to be installed (otherwise skipped).
"""
from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")


def test_vendored_spliceformer_forward_runs():
    pytest.importorskip("einops")
    from oncosplice.engines._vendor.spliceformer import SpliceFormer

    cl = 200
    model = SpliceFormer(CL_max=cl, determenistic=True).eval()
    length = cl + 64
    x = torch.zeros(1, 4, length)
    x[0, 0, :] = 1.0
    with torch.no_grad():
        out, *_ = model(x)
    assert out.shape == (1, 3, length - cl)
    # 3-way softmax over (none, acceptor, donor) sums to 1 per position.
    assert torch.allclose(out.sum(1), torch.ones(1, length - cl), atol=1e-4)


def _upstream_pangolin_dir():
    try:
        from pkg_resources import resource_filename
        from pathlib import Path
        d = Path(resource_filename("pangolin", "models"))
        return d if d.exists() and any(d.glob("final.*.3")) else None
    except Exception:
        return None


@pytest.mark.skipif(
    _upstream_pangolin_dir() is None,
    reason="upstream pangolin package/weights not installed",
)
def test_pangolin_reimpl_matches_geney():
    geney_engines = pytest.importorskip("geney.engines")

    from oncosplice.engines import get_predictor

    cl = 5000
    rng = np.random.default_rng(7)
    seq = "".join(rng.choice(list("ACGT"), size=2 * cl + 101))

    pred = get_predictor("pangolin", weights_dir=str(_upstream_pangolin_dir())).predict(seq)
    don_ref, acc_ref = geney_engines.pangolin_predict_probs(seq)

    assert np.allclose(pred.acceptor, np.asarray(acc_ref), atol=1e-5)
    assert np.allclose(pred.donor, np.asarray(don_ref), atol=1e-5)
