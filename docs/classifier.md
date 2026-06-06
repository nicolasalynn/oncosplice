# Classifier

The classifier sits in `oncosplice.scoring.epistasis` and operates per splice
site. For each site it has four probability columns:

| Column | Meaning |
|---|---|
| `ref` | WT splice-site probability |
| `m1`  | probability under mutation 1 alone |
| `m2`  | probability under mutation 2 alone |
| `ev`  | probability under the joint event |

From these it derives three deltas — `d1 = m1 − ref`, `d2 = m2 − ref`,
`de = ev − ref` — and a residual `de − (d1 + d2)` (signed excess over the
additive expectation).

!!! info "Why not classify on `|residual|` alone?"
    Splice-site probabilities saturate near 0 and 1, so a large `|residual|`
    is often a boundary artifact rather than biology. The rules below anchor
    on the *direction* and *magnitude* of `d1`, `d2`, `de` relative to the
    reference, with explicit guards against the saturation cases.

## The rules

| Class | Rule | Mechanism |
|---|---|---|
| **rescue** | worst single ≥ 0.30 in magnitude; joint ≤ 0.20 from WT; joint at least 0.15 closer to WT than worst; joint on the same side of WT as worst (tiny overshoots permitted). | One single substantially perturbs the splice site; the joint restores it. |
| **synergistic — flip** | worst and joint on opposite sides of WT; \|de\| ≥ 0.15; joint differs from EVERY single by ≥ 0.15. | Singles push one way, joint produces an emergent opposite effect. |
| **synergistic — emergent at edge** | ref ≤ 0.10 or ref ≥ 0.90; singles barely move; joint \|Δ\| ≥ 0.10. | Site was pegged at "off" or "on", singles don't budge it, joint creates a logit-scale meaningful change. |
| **synergistic — super-additive** | \|de\| > \|worst\|; \|de\| > \|additive\|; (\|de\| − \|additive\|) > 0.25 × \|worst\|. | Joint clearly exceeds both the strongest single and the additive prediction. |
| **compounding** | both \|d\| ≥ 0.20; \|de\| > \|worst\|; \|de\| > \|additive\|; (\|de\| − \|additive\|) ≤ 0.25 × \|worst\|. | Both singles contribute meaningfully; joint is the stacked (≈ additive) effect. |
| **non-epistatic** | else. | Saturation, dominance, redundant disruption, or noise. |

Pair-level call: descending priority **synergistic > rescue > compounding >
non-epistatic** over the per-site classifications.

## Why no "antagonistic" bucket?

Cases that look antagonistic under a strict `|residual|` rule almost always
turn out to be saturation artifacts (joint pegged at 1.0 when the additive
prediction is > 1.0). They are correctly reported as **non-epistatic** —
the rule set deliberately doesn't try to manufacture an antagonism call out
of those.

## Direct programmatic access

```python
from oncosplice.scoring.epistasis import (
    classify_pair_residuals,
    compute_site_residuals,
)

df = compute_site_residuals(ref, m1, m2, ev)   # adds d1, d2, de, residual
calls = classify_pair_residuals(df)            # per-site + pair-level label
```

See [API reference](api/scoring.md) for full signatures.
