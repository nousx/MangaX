from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Literal

RegionChangeKind = Literal["reset", "updated", "inserted", "removed"]


@dataclass(frozen=True)
class RegionChange:
    """One change notification for region data: kind decides the smallest refresh the view has to do.

    - updated:  the content of the regions in indices changed; the items are refreshed in place
    - inserted: new regions were inserted at the positions in indices
    - removed:  the regions at the positions in indices were deleted
    - reset:    a document-level change (image switch, clear, import, global render parameters); everything is rebuilt
    """

    kind: RegionChangeKind
    indices: tuple[int, ...] = ()
    fields: tuple[str, ...] = ()
    source: str = ""

    @classmethod
    def reset(cls, *, source: str = "") -> "RegionChange":
        return cls("reset", source=source)

    @classmethod
    def updated(
        cls,
        indices: Iterable[int],
        *,
        fields: Iterable[str] | None = None,
        source: str = "",
    ) -> "RegionChange":
        return cls("updated", _normalize_indices(indices), _normalize_fields(fields), source)

    @classmethod
    def inserted(cls, indices: Iterable[int], *, source: str = "") -> "RegionChange":
        return cls("inserted", _normalize_indices(indices), source=source)

    @classmethod
    def removed(cls, indices: Iterable[int], *, source: str = "") -> "RegionChange":
        return cls("removed", _normalize_indices(indices), source=source)


def _normalize_indices(indices: Iterable[int]) -> tuple[int, ...]:
    return tuple(sorted({int(index) for index in indices}))


def _normalize_fields(fields: Iterable[str] | None) -> tuple[str, ...]:
    if fields is None:
        return ()
    return tuple(sorted({str(field) for field in fields if field is not None}))
