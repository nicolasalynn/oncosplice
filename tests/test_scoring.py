"""Unit tests for the pure-numerical scoring layer.

These tests do not touch the splicing engines or seqmat — they exercise the
math given hand-built site_table inputs.
"""
import numpy as np
import pandas as pd
import pytest

from oncosplice.scoring.splicing import (
    extract_site_table, classify_missplicing, missplicing_to_dict,
    max_splicing_delta, MissplicingEvent,
)
from oncosplice.scoring.epistasis import (
    compute_site_residuals, classify_pair, summarize_residuals,
)


def make_wide_splicing_df():
    """Build a synthetic splicing_results DataFrame with all four contexts.

    Schema mirrors what geney's adjoin_splicing_outcomes produces:
    columns are MultiIndex (site_type, metric).
    """
    positions = [100, 105, 200, 205]
    cols = pd.MultiIndex.from_tuples([
        ("donors",    "annotated"),
        ("donors",    "ref_prob"),
        ("donors",    "mut1_prob"),
        ("donors",    "mut2_prob"),
        ("donors",    "event_prob"),
        ("acceptors", "annotated"),
        ("acceptors", "ref_prob"),
        ("acceptors", "mut1_prob"),
        ("acceptors", "mut2_prob"),
        ("acceptors", "event_prob"),
    ])
    data = np.array([
        # pos 100: annotated donor that survives
        [True,  0.95, 0.92, 0.94, 0.91,   False, 0.0, 0.0, 0.0, 0.0],
        # pos 105: cryptic donor discovered jointly (synergy)
        [False, 0.05, 0.10, 0.12, 0.85,   False, 0.0, 0.0, 0.0, 0.0],
        # pos 200: annotated acceptor — antagonism (each mut breaks it, joint rescues)
        [False, 0.0, 0.0, 0.0, 0.0,        True,  0.90, 0.30, 0.30, 0.85],
        # pos 205: cryptic acceptor — additive (no epistasis)
        [False, 0.0, 0.0, 0.0, 0.0,        False, 0.05, 0.30, 0.30, 0.55],
    ])
    return pd.DataFrame(data, index=positions, columns=cols)


def test_extract_site_table_yields_long_format():
    df = make_wide_splicing_df()
    long = extract_site_table(df, contexts=["ref", "mut1", "mut2", "event"])
    expected_rows = 4 * 2 * 4   # 4 positions * 2 site_types * 4 contexts
    assert len(long) == expected_rows
    assert set(long.columns) == {"position", "site_type", "context", "prob", "annotated"}
    # ref, mut1, mut2, event are all represented
    assert set(long.context.unique()) == {"ref", "mut1", "mut2", "event"}


def test_classify_missplicing_finds_discovered_donor_and_missed_acceptor():
    df = make_wide_splicing_df()
    long = extract_site_table(df, contexts=["ref", "event"])
    events = classify_missplicing(long, context="event", threshold=0.25)
    types = {(e.position, e.event_type) for e in events}
    # pos 105 should be a discovered_donor (cryptic gain ≥ 0.25)
    assert (105, "discovered_donor") in types
    # pos 200 (annotated acceptor) should NOT be missed in event (rescued)
    # pos 205 (cryptic acceptor) should be discovered in event
    assert (205, "discovered_acceptor") in types


def test_compute_site_residuals_signs_and_classifications():
    df = make_wide_splicing_df()
    long = extract_site_table(df, contexts=["ref", "mut1", "mut2", "event"])
    res = compute_site_residuals(long, threshold=0.25)

    # Pos 105: cryptic donor that *appears* in joint context only.
    # ref=0.05, mut1=0.10, mut2=0.12, event=0.85
    # delta_event=0.80, expected_delta=0.12 ⇒ |0.80|>|0.12| ⇒ synergistic
    row_105 = res[(res.position == 105) & (res.site_type == "donor")].iloc[0]
    assert row_105.classification == "synergistic"
    assert abs(row_105.residual) > 0.25

    # Pos 200: rescue at the annotated acceptor.
    # ref=0.90, mut1=0.30, mut2=0.30, event=0.85 — both singles strongly disrupt
    # (d1=d2=-0.60), but joint stays near WT (de=-0.05). Mutual-cancellation rescue.
    row_200 = res[(res.position == 200) & (res.site_type == "acceptor")].iloc[0]
    assert row_200.classification == "rescue"

    # Pos 205: additive (cv1 + cv2 ≈ joint), should be non-epistatic.
    row_205 = res[(res.position == 205) & (res.site_type == "acceptor")].iloc[0]
    assert row_205.classification == "non-epistatic"
    assert abs(row_205.residual) < 0.25


def test_classify_pair_and_summary():
    df = make_wide_splicing_df()
    long = extract_site_table(df, contexts=["ref", "mut1", "mut2", "event"])
    res = compute_site_residuals(long, threshold=0.25)
    pair_cls = classify_pair(res)
    # 3-bucket classifier: synergistic > rescue > compounding > non-epistatic.
    assert pair_cls in {"synergistic", "rescue", "compounding"}
    summary = summarize_residuals(res)
    assert summary["n_sites"] == len(res)
    assert summary["pair_classification"] == pair_cls
    assert summary["max_abs_residual"] > 0.25


def test_max_splicing_delta_picks_largest_signed_change():
    df = make_wide_splicing_df()
    long = extract_site_table(df, contexts=["ref", "event"])
    delta = max_splicing_delta(long, context="event")
    # Largest absolute change is the donor at 105: 0.85 - 0.05 = +0.80
    assert pytest.approx(delta, abs=1e-6) == 0.80


def test_missplicing_to_dict_round_trip():
    df = make_wide_splicing_df()
    long = extract_site_table(df, contexts=["ref", "event"])
    events = classify_missplicing(long, context="event", threshold=0.25)
    bucketed = missplicing_to_dict(events)
    expected_keys = {"missed_donors", "missed_acceptors",
                     "discovered_donors", "discovered_acceptors"}
    assert set(bucketed.keys()) == expected_keys
    n_total = sum(len(v) for v in bucketed.values())
    assert n_total == len(events)
