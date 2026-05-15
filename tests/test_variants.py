"""Tests for variant parsing and pair canonicalization."""
import pytest
from oncosplice.variants import Variant, VariantPair


def test_variant_from_id_canonicalizes():
    v = Variant.from_id("kras:12:25227343:g:t")
    assert v.gene == "kras"   # gene case is preserved (only alleles upper-cased)
    assert v.chrom == "12"
    assert v.pos == 25227343
    assert v.ref == "G"
    assert v.alt == "T"
    assert v.is_snv


def test_variant_indel_detection():
    v = Variant.from_id("CEP83:12:94370026:AGTTC:-")
    assert not v.is_snv
    assert v.span == 5


def test_variant_pair_orders_by_position():
    p = VariantPair.from_ids("KRAS:12:25227344:A:T", "KRAS:12:25227343:G:T")
    assert p.mut1.pos == 25227343
    assert p.mut2.pos == 25227344
    assert p.distance == 1


def test_variant_pair_from_combined_id():
    p = VariantPair.from_ids("KRAS:12:25227343:G:T|KRAS:12:25227344:A:T")
    assert p.mut1.pos == 25227343
    assert p.mut2.pos == 25227344


def test_variant_pair_rejects_cross_gene():
    with pytest.raises(ValueError, match="same a gene|share a gene"):
        VariantPair.from_ids("KRAS:12:25227343:G:T", "TP53:17:7674220:C:T")


def test_variant_pair_epistasis_id_canonical():
    p1 = VariantPair.from_ids("KRAS:12:25227343:G:T", "KRAS:12:25227344:A:T")
    p2 = VariantPair.from_ids("KRAS:12:25227344:A:T", "KRAS:12:25227343:G:T")
    assert p1.epistasis_id == p2.epistasis_id   # canonical order
