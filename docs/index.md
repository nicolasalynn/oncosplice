# oncosplice

> Given two (or more) mutations in the same gene, classify how their joint
> effect on splicing differs from the additive prediction — and which mechanism
> (rescue, synergy, compounding) drives it.

**oncosplice** is a sequence-level pipeline for splicing-epistasis analysis of
single-, double-, and N-variant constructs. It runs a splice-site predictor
under each variant context, computes per-site residuals against the additive
expectation, and applies a three-bucket mechanistic classifier.

## Install

```bash
pip install oncosplice[spliceai_pytorch]
```

See [Quickstart](quickstart.md) for a 30-second example, or jump into the
[classifier](classifier.md) to understand the mechanism rules.

## Citation

If you use **oncosplice** in published work, please cite:

> Lynn, N. & Tuller, T. *Detecting and understanding meaningful cancerous
> mutations based on computational models of mRNA splicing.* **npj Systems
> Biology and Applications** (2024).

The splicing-epistasis layer (multi-variant classifier, TCGA-scale runner) is
described in a manuscript in preparation; please cite this repository in the
meantime:

> Lynn, N. *oncosplice: a sequence-level splicing-epistasis pipeline.*
> github.com/nicolasalynn/oncosplice
