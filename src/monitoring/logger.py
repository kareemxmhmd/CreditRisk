"""
Audit logging engine for credit decisions into SQLite database.
"""

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, List, Optional
import pandas as pd
from src.config import DB_PATH, MODEL_VERSION


class DecisionLogger:
    """
    Persists real-time application inputs, risk predictions, decisions, and reason codes
    for compliance audit and drift monitoring.
    """

    def __init__(self, db_path: Path = DB_PATH):
        self.db_path = db_path
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        return sqlite3.connect(str(self.db_path))

    def _init_db(self):
        """Create decisions table if not exists."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS credit_decisions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    application_id TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    model_version TEXT NOT NULL,
                    probability_of_default REAL NOT NULL,
                    decision TEXT NOT NULL,
                    risk_tier TEXT NOT NULL,
                    recommended_interest_rate REAL NOT NULL,
                    reason_codes_json TEXT NOT NULL,
                    latency_ms REAL NOT NULL,
                    raw_features_json TEXT NOT NULL
                )
            """)
            conn.commit()

    def log_decision(
        self,
        application_id: str,
        probability_of_default: float,
        decision: str,
        risk_tier: str,
        recommended_interest_rate: float,
        reason_codes: List[Dict[str, Any]],
        raw_features: Dict[str, Any],
        latency_ms: float = 0.0,
        model_version: str = MODEL_VERSION,
    ) -> int:
        """Log a single decision to the database."""
        timestamp = datetime.now(timezone.utc).isoformat()
        reason_codes_str = json.dumps(reason_codes)
        raw_features_str = json.dumps(raw_features)

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO credit_decisions (
                    application_id, timestamp, model_version, probability_of_default,
                    decision, risk_tier, recommended_interest_rate, reason_codes_json,
                    latency_ms, raw_features_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                application_id, timestamp, model_version, float(probability_of_default),
                decision, risk_tier, float(recommended_interest_rate), reason_codes_str,
                float(latency_ms), raw_features_str
            ))
            conn.commit()
            return cursor.lastrowid

    def get_recent_decisions(self, limit: int = 100) -> pd.DataFrame:
        """Retrieve recent decisions as a DataFrame."""
        with self._get_connection() as conn:
            df = pd.read_sql_query(
                "SELECT * FROM credit_decisions ORDER BY id DESC LIMIT ?",
                conn,
                params=(limit,)
            )
        return df

    def get_all_logged_features(self) -> pd.DataFrame:
        """Extract all logged features into a tabular DataFrame for drift detection."""
        with self._get_connection() as conn:
            df = pd.read_sql_query("SELECT id, timestamp, raw_features_json, probability_of_default FROM credit_decisions", conn)
        
        if df.empty:
            return pd.DataFrame()

        rows = []
        for _, row in df.iterrows():
            features = json.loads(row["raw_features_json"])
            features["logged_pd"] = row["probability_of_default"]
            features["logged_id"] = row["id"]
            rows.append(features)
        return pd.DataFrame(rows)
