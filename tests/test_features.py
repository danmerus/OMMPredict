import itertools
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from converting_service import convert
from features import (
    ELLIPSOID_COEF, FEATURE_COLUMNS, build_features, genotype_code, normalize_baseline, to_target,
)
from labeling import changed_labels

DEFAULT_BASELINE = Path(__file__).resolve().parents[1] / "Web-application (Flask)" / "app" / "data" / "baseline_examples.csv"

GENO_FEATURES = ["vegfa634gg", "vegfa634c", "tp53gg", "vegfa936cc", "kitlg80441cc"]
ALL_GENOTYPES = list(itertools.product(["CC", "GC", "GG"], ["CC", "GC", "GG"], ["CC", "CT", "TT"], ["CC", "CT", "TT"]))


def patient(**overrides):
    row = dict(relapse=1, periods=5.0, mecho=7.5, first_symptom=3.0, emergency_birth=0, fsh=7.0,
               vleft=10.0, vright=12.0, vegfa634="GC", tp53="CC", vegfa936="CT", kitlg80441="TT")
    row.update(overrides)
    return row


def test_genotype_letters_match_legacy_convert():
    rows = [patient(vegfa634=a, tp53=b, vegfa936=c, kitlg80441=d) for a, b, c, d in ALL_GENOTYPES]
    X = build_features(pd.DataFrame(rows))
    expected = np.array([convert(*g) for g in ALL_GENOTYPES])
    np.testing.assert_array_equal(X[GENO_FEATURES].values, expected)


def test_all_genotype_combinations_give_distinct_features():
    rows = [patient(vegfa634=a, tp53=b, vegfa936=c, kitlg80441=d) for a, b, c, d in ALL_GENOTYPES]
    X = build_features(pd.DataFrame(rows))
    assert len(X[GENO_FEATURES].drop_duplicates()) == 81


def test_db_row_and_baseline_row_give_identical_features():
    """Same patient stored the app's way (cm³, letters) and the baseline CSV's way
    (L×W×H in mm, 0/1/2 codes) must produce the same model input."""
    db_row = patient()
    lwh_left, lwh_right = db_row["vleft"] * 1000 / ELLIPSOID_COEF, db_row["vright"] * 1000 / ELLIPSOID_COEF
    csv_row = dict(db_row, vleft=lwh_left, vright=lwh_right,
                   vegfa634=1.0, tp53=2.0, vegfa936=1.0, kitlg80441=2.0, actual=1.0)
    from_db = build_features(pd.DataFrame([db_row]))
    from_csv = build_features(normalize_baseline(pd.DataFrame([csv_row])))
    np.testing.assert_allclose(from_db.values, from_csv.values)


def test_volume_is_used_in_cm3_as_entered():
    X = build_features(pd.DataFrame([patient(vleft=10.0, vright=4.5)]))
    assert X.loc[0, "vleft_cm3"] == 10.0 and X.loc[0, "vright_cm3"] == 4.5


@pytest.mark.parametrize("value", ["XX", "", "  ", None, np.nan, 3, -1, "abc", 1.5])
def test_unknown_genotype_is_nan_not_zero(value):
    assert math.isnan(genotype_code("vegfa634", value))


def test_genotype_accepts_reversed_letters_and_codes():
    assert genotype_code("vegfa634", "CG") == genotype_code("vegfa634", "GC") == 1.0
    assert genotype_code("kitlg80441", "tc") == 1.0
    assert genotype_code("tp53", 2.0) == genotype_code("tp53", "2") == 2.0


def test_to_target():
    s = pd.Series(["Высокий", "Низкий", " высокий ", 1.0, 0, "1", None, "maybe", 2])
    out = to_target(s).tolist()
    assert out[:6] == [1.0, 0.0, 1.0, 1.0, 0.0, 1.0]
    assert all(math.isnan(v) for v in out[6:])


def test_baseline_csv_loads_with_real_genotypes_and_cm3_volumes():
    base = normalize_baseline(pd.read_csv(DEFAULT_BASELINE))
    X = build_features(base)
    assert list(X.columns) == FEATURE_COLUMNS
    assert not X.isna().any().any()
    for col in GENO_FEATURES:
        assert X[col].nunique() > 1, f"{col} is constant"
    assert 1 < X["vleft_cm3"].median() < 50
    assert set(base["target"].unique()) == {0.0, 1.0}


# ---- labelling ---------------------------------------------------------

def _grid(rows):
    return pd.DataFrame(rows, columns=["id", "patient_card", "date_research", "outcome", "actual", "notes"], dtype=object)


def test_changed_labels_detects_only_real_changes():
    before = _grid([[5, "a", "d", "Низкий", None, None],
                    [7, "b", "d", "Высокий", None, np.nan],
                    [9, "c", "d", "Низкий", None, None]])
    after = before.copy()
    after.loc[1, "actual"] = "Высокий"          # labelled
    after.loc[2, "notes"] = "  "                 # whitespace only -> no change
    assert changed_labels(before, after) == [(7, "Высокий", None)]


def test_changed_labels_matches_rows_by_id_and_never_sends_nan():
    before = _grid([[5, "a", "d", "Низкий", None, np.nan], [7, "b", "d", "Высокий", None, None]])
    after = before.iloc[::-1].copy()             # different order
    after.loc[after["id"] == 5, ["actual", "notes"]] = ["Низкий", " ok "]
    changes = changed_labels(before, after)
    assert changes == [(5, "Низкий", "ok")]
    for _, actual, notes in changes:
        assert not (isinstance(notes, float) and math.isnan(notes))


def test_changed_labels_empty():
    assert changed_labels(_grid([]), _grid([])) == []
