"""
Variant and VariantPair — thin, validated wrappers over geney.variants.

Why a wrapper:
- ``oncosplice`` users never need to know the geney parser exists.
- Pair-level helpers (canonical id ordering, distance, gene-consistency check)
  live in one place.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Tuple

# geney is imported lazily by `to_event()` only — keeping the top-level
# import out of variants.py means `oncosplice.engines.get_predictor()` works
# without geney installed (predictors themselves don't need it).


def _normalize(mut_id: str) -> str:
    """Canonicalize a single mutation id (uppercase alleles, strip whitespace)."""
    parts = mut_id.strip().split(":")
    if len(parts) != 5:
        raise ValueError(
            f"Expected GENE:CHROM:POS:REF:ALT, got {mut_id!r}"
        )
    gene, chrom, pos, ref, alt = parts
    return f"{gene}:{chrom}:{int(pos)}:{ref.upper()}:{alt.upper()}"


@dataclass(frozen=True)
class Variant:
    """Single mutation, canonicalized."""

    mut_id: str
    gene: str
    chrom: str
    pos: int
    ref: str
    alt: str

    @classmethod
    def from_id(cls, mut_id: str) -> "Variant":
        """Parse a canonical GENE:CHROM:POS:REF:ALT string.

        Pure-python — no geney import needed. ``to_event()`` lazy-imports
        ``MutationalEvent`` only when the caller actually wants one.
        """
        canonical = _normalize(mut_id)
        gene, chrom, pos, ref, alt = canonical.split(":")
        return cls(canonical, gene, chrom, int(pos), ref, alt)

    @property
    def is_snv(self) -> bool:
        return (
            len(self.ref) == 1 and len(self.alt) == 1
            and self.ref != "-" and self.alt != "-"
        )

    @property
    def span(self) -> int:
        ref_len = 0 if self.ref == "-" else len(self.ref)
        alt_len = 0 if self.alt == "-" else len(self.alt)
        return max(ref_len, alt_len, 1)

    def to_event(self):
        from ._geney_compat import MutationalEvent
        return MutationalEvent(self.mut_id)

    def __str__(self) -> str:
        return self.mut_id


@dataclass(frozen=True)
class VariantPair:
    """Ordered pair of two single variants in the same gene."""

    mut1: Variant
    mut2: Variant

    @classmethod
    def from_ids(cls, mut1_id: str, mut2_id: str | None = None) -> "VariantPair":
        # Accept either two ids, or one combined "id1|id2" string.
        if mut2_id is None:
            if "|" not in mut1_id:
                raise ValueError(
                    "Pass two mut_ids or one 'mut1|mut2' string."
                )
            mut1_id, mut2_id = mut1_id.split("|", 1)
        v1 = Variant.from_id(mut1_id)
        v2 = Variant.from_id(mut2_id)
        if v1.gene != v2.gene:
            raise ValueError(
                f"Variants must share a gene; got {v1.gene} vs {v2.gene}."
            )
        if v1.chrom != v2.chrom:
            raise ValueError(
                f"Variants must share a chromosome; got {v1.chrom} vs {v2.chrom}."
            )
        # Canonical order: by genomic position, ties broken by allele.
        if (v1.pos, v1.ref, v1.alt) > (v2.pos, v2.ref, v2.alt):
            v1, v2 = v2, v1
        return cls(v1, v2)

    @property
    def gene(self) -> str:
        return self.mut1.gene

    @property
    def epistasis_id(self) -> str:
        return f"{self.mut1.mut_id}|{self.mut2.mut_id}"

    @property
    def distance(self) -> int:
        """Inter-variant distance in nt."""
        return abs(self.mut2.pos - self.mut1.pos)

    @property
    def central_position(self) -> int:
        return (self.mut1.pos + self.mut2.pos) // 2

    def to_event(self):
        from ._geney_compat import MutationalEvent
        return MutationalEvent(self.epistasis_id)

    def __str__(self) -> str:
        return self.epistasis_id


def parse_pair_ids(items: Iterable[str | Tuple[str, str]]) -> List[VariantPair]:
    """Convenience: parse a list of pair ids (either '|'-joined or tuples)."""
    out: List[VariantPair] = []
    for item in items:
        if isinstance(item, tuple):
            out.append(VariantPair.from_ids(item[0], item[1]))
        else:
            out.append(VariantPair.from_ids(item))
    return out
