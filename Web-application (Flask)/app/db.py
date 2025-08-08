import sqlite3
from pathlib import Path

DB_PATH = Path("ommpredict.db")

def get_conn():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False, isolation_level=None)
    conn.execute("PRAGMA journal_mode=WAL;")  # better for concurrent reads
    return conn

def init_db(conn):
    conn.execute("""
    CREATE TABLE IF NOT EXISTS predictions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        created_at   TEXT DEFAULT (datetime('now')),
        patient_card TEXT,
        date_research TEXT,
        relapse INTEGER,
        periods REAL,
        mecho REAL,
        first_symptom REAL,
        emergency_birth INTEGER,
        fsh REAL,
        vleft REAL,
        vright REAL,
        vegfa634 TEXT,
        tp53 TEXT,
        vegfa936 TEXT,
        kitlg80441 TEXT,
        outcome TEXT
    );
    """)

def insert_prediction(conn, row: dict):
    cols = ["patient_card","date_research","relapse","periods","mecho",
            "first_symptom","emergency_birth","fsh","vleft","vright",
            "vegfa634","tp53","vegfa936","kitlg80441","outcome"]
    placeholders = ",".join("?" for _ in cols)
    conn.execute(
        f"INSERT INTO predictions ({','.join(cols)}) VALUES ({placeholders})",
        tuple(row[c] for c in cols)
    )

def read_predictions(conn, limit=1000):
    import pandas as pd
    return pd.read_sql_query(
        "SELECT * FROM predictions ORDER BY id DESC LIMIT ?",
        conn, params=(limit,)
    )
