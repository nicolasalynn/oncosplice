"""Quickstart — single variant + double variant.

Run from a Python ≥ 3.10 environment with `oncosplice` installed:

    python examples/quickstart.py
"""
from oncosplice import OncospliceEngine

eng = OncospliceEngine(splicing_engine="spliceai")
print(eng)

# ---- Single variant -------------------------------------------------------
print("\n=== Single variant: KRAS Q61K (12:25227344 A>T) ===")
single = eng.analyze_single("KRAS:12:25227344:A:T")
print(single.summary())
print(single.missplicing.to_dataframe())
print(single.isoforms[["isoform_id", "prevalence", "splicing_changes",
                       "oncosplice_score"]].head())

# ---- Double variant: the canonical Osimertinib-resistance KRAS pair -------
# G60G synonymous (12:25227343 G>T) + Q61K (12:25227344 A>T) — antagonistic
# splice rescue described in Roper et al. 2020.
print("\n=== Double variant: KRAS G60G + Q61K ===")
pair = eng.analyze_pair(
    "KRAS:12:25227343:G:T",
    "KRAS:12:25227344:A:T",
)
print({**pair.summary()})
print("\nEpistatic splice sites only:")
print(pair.epistatic_sites())

# Plot the headline figure
fig, _ = pair.plot_residuals()
fig.savefig("kras_g60g_q61k_residuals.png", dpi=150, bbox_inches="tight")
print("Wrote kras_g60g_q61k_residuals.png")

# Composite summary
fig2, _ = pair.plot_summary()
fig2.savefig("kras_g60g_q61k_summary.png", dpi=150, bbox_inches="tight")
print("Wrote kras_g60g_q61k_summary.png")
