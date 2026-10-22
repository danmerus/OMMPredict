import copy
from types import SimpleNamespace

import pytest

from features import FEATURE_VERSION
from model_registry import (
    cv_auc, describe, feature_version_of, is_compatible, models_table, set_current_model,
)

NEW_A = dict(version="2026-10-22T07-00-00Z-aaaaaa", params={"depth": 3, "feature_version": FEATURE_VERSION},
             metrics={"auc": 0.75}, samples_used=69, is_current=False)
NEW_B = dict(version="2026-10-23T07-00-00Z-bbbbbb", params='{"depth": 6, "feature_version": %d}' % FEATURE_VERSION,
             metrics='{"auc": 0.61}', samples_used=70, is_current=True)
OLD = dict(version="2025-08-10T10-00-00Z", params={"iterations": 600}, metrics={"auc": 0.697},
           samples_used=69, is_current=False)


class FakeRegistry:
    """Minimal stand-in for sb.table("model_registry") queries used by set_current_model."""

    def __init__(self, rows):
        self.rows = copy.deepcopy(rows)

    def table(self, name):
        assert name == "model_registry"
        return _Query(self.rows)

    def current(self):
        return sorted(r["version"] for r in self.rows if r["is_current"])


class _Query:
    def __init__(self, rows):
        self.rows, self.filters, self.payload = rows, [], None

    def select(self, *_):
        return self

    def update(self, payload):
        self.payload = payload
        return self

    def eq(self, col, val):
        self.filters.append(lambda r: r.get(col) == val)
        return self

    def neq(self, col, val):
        self.filters.append(lambda r: r.get(col) != val)
        return self

    def limit(self, _):
        return self

    def execute(self):
        hit = [r for r in self.rows if all(f(r) for f in self.filters)]
        if self.payload is not None:
            for r in hit:
                r.update(self.payload)
        return SimpleNamespace(data=copy.deepcopy(hit))


def test_params_and_metrics_as_dict_json_or_none():
    assert feature_version_of(NEW_A) == FEATURE_VERSION
    assert feature_version_of(NEW_B) == FEATURE_VERSION          # JSON string
    assert feature_version_of(dict(version="x", params=None)) is None
    assert feature_version_of(dict(version="x", params="not json")) is None
    assert cv_auc(NEW_B) == pytest.approx(0.61)


def test_old_models_are_incompatible_and_have_no_comparable_auc():
    assert is_compatible(NEW_A) and is_compatible(NEW_B)
    assert not is_compatible(OLD)
    assert cv_auc(OLD) is None
    assert "старый формат" in describe(OLD)


def test_describe():
    assert describe(NEW_A) == "2026-10-22T07-00-00Z-aaaaaa · AUC (CV) 0.75 · 69 примеров"


def test_models_table_newest_first():
    t = models_table([OLD, NEW_A, NEW_B])
    assert t["Версия"].tolist() == [NEW_B["version"], NEW_A["version"], OLD["version"]]
    assert t["Текущая"].tolist() == [True, False, False]
    assert t["Совместима"].tolist() == [True, True, False]
    assert t["depth"].tolist()[:2] == [6, 3]


def test_switch_moves_current_flag():
    reg = FakeRegistry([OLD, NEW_A, NEW_B])
    set_current_model(reg, NEW_A["version"])
    assert reg.current() == [NEW_A["version"]]


@pytest.mark.parametrize("version", ["2099-01-01T00-00-00Z-typo00", OLD["version"]])
def test_switch_to_unknown_or_old_model_raises_and_changes_nothing(version):
    reg = FakeRegistry([OLD, NEW_A, NEW_B])
    with pytest.raises(ValueError):
        set_current_model(reg, version)
    assert reg.current() == [NEW_B["version"]]
