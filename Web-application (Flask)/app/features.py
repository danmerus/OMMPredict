"""Feature engineering shared by training and the calculator.

Both paths must turn patient data into model inputs the same way, so both call
build_features() from here.

Canonical units (what the calculator asks for and what the `predictions` table
stores): ovary volumes in cm³, genotypes as letters ("GG", "GC", "CC", ...).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# Bump whenever the meaning of a feature changes. Each registered model records
# the version it was trained with, and the calculator refuses other versions.
FEATURE_VERSION = 2

FEATURE_COLUMNS = [
    "relapse", "vegfa634gg", "vegfa634c", "periods", "tp53gg", "mecho",
    "vegfa936cc", "first_symptom", "kitlg80441cc", "emergency_birth",
    "vleft_cm3", "fsh", "vright_cm3",
]

NUMERIC_INPUTS = ["relapse", "periods", "mecho", "first_symptom", "emergency_birth", "fsh", "vleft", "vright"]
GENOTYPE_INPUTS = ["vegfa634", "tp53", "vegfa936", "kitlg80441"]

# Genotype letters -> code. Same numbers as converting_service.convert().
# The baseline CSV already stores these codes (checked against the letters
# printed in the research notebook for the same patients).
GENOTYPE_CODES = {
    "vegfa634": {"GG": 0, "GC": 1, "CC": 2},
    "tp53": {"GG": 0, "GC": 1, "CC": 2},
    "vegfa936": {"CC": 0, "CT": 1, "TT": 2},
    "kitlg80441": {"CC": 0, "CT": 1, "TT": 2},
}

# The baseline CSV stores ovary size as length × width × height in mm.
# Ellipsoid volume = L·W·H · π/6 (≈ 0.523); mm³ -> cm³ divides by 1000.
ELLIPSOID_COEF = 0.523

_TARGET_LABELS = {"высокий": 1, "низкий": 0, "high": 1, "low": 0}


def genotype_code(gene: str, value) -> float:
    """Genotype letters ("GC" or "CG") or an existing 0/1/2 code -> code. Anything else -> NaN."""
    if value is None:
        return np.nan
    if isinstance(value, str):
        s = value.strip().upper()
        codes = GENOTYPE_CODES[gene]
        if s in codes:
            return float(codes[s])
        if s[::-1] in codes:
            return float(codes[s[::-1]])
        try:
            value = float(s)
        except ValueError:
            return np.nan
    try:
        v = float(value)
    except (TypeError, ValueError):
        return np.nan
    return v if v in (0.0, 1.0, 2.0) else np.nan


def lwh_mm_to_cm3(values) -> pd.Series:
    """L×W×H product in mm -> ellipsoid volume in cm³."""
    return pd.to_numeric(values, errors="coerce") * ELLIPSOID_COEF / 1000.0


def to_target(values: pd.Series) -> pd.Series:
    """'Высокий'/'Низкий' (or 1/0) -> 1/0. Anything else -> NaN."""
    def one(v):
        if v is None:
            return np.nan
        if isinstance(v, str):
            s = v.strip().lower()
            if s in _TARGET_LABELS:
                return float(_TARGET_LABELS[s])
            try:
                v = float(s)
            except ValueError:
                return np.nan
        try:
            f = float(v)
        except (TypeError, ValueError):
            return np.nan
        return f if f in (0.0, 1.0) else np.nan
    return pd.Series([one(v) for v in values], index=values.index, dtype=float)


def normalize_baseline(df: pd.DataFrame) -> pd.DataFrame:
    """Baseline CSV -> canonical units (volumes in cm³) with a numeric `target`."""
    df = df.copy()
    for col in ("vleft", "vright"):
        if col in df.columns:
            df[col] = lwh_mm_to_cm3(df[col])
    label_col = "target" if "target" in df.columns else "actual"
    df["target"] = to_target(df[label_col]) if label_col in df.columns else np.nan
    return df


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Rows in canonical units -> model inputs, in FEATURE_COLUMNS order.

    Missing or unrecognised values become NaN (never a silent 0).
    """
    def num(col):
        if col not in df.columns:
            return pd.Series(np.nan, index=df.index)
        return pd.to_numeric(df[col], errors="coerce")

    def geno(gene):
        if gene not in df.columns:
            return pd.Series(np.nan, index=df.index)
        return df[gene].map(lambda v: genotype_code(gene, v)).astype(float)

    vegfa634 = geno("vegfa634")
    out = pd.DataFrame({
        "relapse": num("relapse"),
        "vegfa634gg": vegfa634,
        "vegfa634c": (vegfa634 > 0).astype(float).where(vegfa634.notna()),  # has a C allele
        "periods": num("periods"),
        "tp53gg": geno("tp53"),
        "mecho": num("mecho"),
        "vegfa936cc": geno("vegfa936"),
        "first_symptom": num("first_symptom"),
        "kitlg80441cc": geno("kitlg80441"),
        "emergency_birth": num("emergency_birth"),
        "vleft_cm3": num("vleft"),
        "fsh": num("fsh"),
        "vright_cm3": num("vright"),
    }, index=df.index)
    return out[FEATURE_COLUMNS].astype(float)
