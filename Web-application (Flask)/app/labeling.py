"""Work out which rows of the labelling grid were changed by the user."""
from __future__ import annotations

import pandas as pd


def clean(value):
    """NaN, None and blank strings -> None (the database rejects NaN); strings are stripped."""
    if value is None:
        return None
    if isinstance(value, str):
        value = value.strip()
        return value or None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return value


def changed_labels(original: pd.DataFrame, edited: pd.DataFrame) -> list[tuple[int, str | None, str | None]]:
    """Return (id, actual, notes) for every row whose `actual` or `notes` differs.

    Rows are matched by `id`, not by position.
    """
    if original.empty or edited.empty:
        return []
    before = original.assign(id=original["id"].astype(int)).set_index("id")
    changes = []
    for _, row in edited.iterrows():
        pid = int(row["id"])
        if pid not in before.index:
            continue
        new_actual, new_notes = clean(row.get("actual")), clean(row.get("notes"))
        old_actual, old_notes = clean(before.at[pid, "actual"]), clean(before.at[pid, "notes"])
        if new_actual != old_actual or new_notes != old_notes:
            changes.append((pid, new_actual, new_notes))
    return changes
