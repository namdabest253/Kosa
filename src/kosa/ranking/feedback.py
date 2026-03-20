"""Human feedback logging for hypotheses.

Day 1 requirement: log thumbs-up/thumbs-down on every hypothesis.
No feedback loop yet — just logging. Analysis comes in Phase 1.5+.

Storage: SQLite for simplicity and queryability. Separate from Neo4j
because feedback is metadata about the system, not graph structure.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

logger = logging.getLogger(__name__)

DEFAULT_DB_PATH = "results/feedback.db"


@dataclass
class FeedbackEntry:
    """A single human feedback entry for a hypothesis."""

    hypothesis_id: str
    rating: int  # +1 (thumbs up), -1 (thumbs down), 0 (skip)
    hypothesis_text: str
    graph_path: str
    activation_score: float
    elo_rating: float = 0.0
    composite_score: float = 0.0
    comment: str = ""
    rater_id: str = ""
    timestamp: str = ""


class FeedbackStore:
    """SQLite-based feedback storage."""

    def __init__(self, db_path: str = DEFAULT_DB_PATH):
        self.db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self):
        """Create the feedback table if it doesn't exist."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS feedback (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    hypothesis_id TEXT NOT NULL,
                    rating INTEGER NOT NULL,
                    hypothesis_text TEXT NOT NULL,
                    graph_path TEXT NOT NULL,
                    activation_score REAL NOT NULL,
                    elo_rating REAL DEFAULT 0,
                    composite_score REAL DEFAULT 0,
                    comment TEXT DEFAULT '',
                    rater_id TEXT DEFAULT '',
                    timestamp TEXT NOT NULL,
                    created_at TEXT NOT NULL DEFAULT (datetime('now'))
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_feedback_hypothesis
                ON feedback(hypothesis_id)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_feedback_rating
                ON feedback(rating)
            """)

    def log_feedback(self, entry: FeedbackEntry) -> int:
        """Log a feedback entry. Returns the row ID."""
        if not entry.timestamp:
            entry.timestamp = datetime.now(UTC).isoformat()

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                """
                INSERT INTO feedback (
                    hypothesis_id, rating, hypothesis_text, graph_path,
                    activation_score, elo_rating, composite_score,
                    comment, rater_id, timestamp
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    entry.hypothesis_id,
                    entry.rating,
                    entry.hypothesis_text,
                    entry.graph_path,
                    entry.activation_score,
                    entry.elo_rating,
                    entry.composite_score,
                    entry.comment,
                    entry.rater_id,
                    entry.timestamp,
                ),
            )
            return cursor.lastrowid

    def get_feedback(self, hypothesis_id: str) -> list[FeedbackEntry]:
        """Get all feedback for a hypothesis."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT * FROM feedback WHERE hypothesis_id = ? ORDER BY timestamp",
                (hypothesis_id,),
            ).fetchall()

        return [_row_to_entry(row) for row in rows]

    def get_all_feedback(
        self,
        rating_filter: int | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[FeedbackEntry]:
        """Get all feedback, optionally filtered by rating."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            if rating_filter is not None:
                rows = conn.execute(
                    "SELECT * FROM feedback WHERE rating = ? "
                    "ORDER BY timestamp DESC LIMIT ? OFFSET ?",
                    (rating_filter, limit, offset),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM feedback ORDER BY timestamp DESC LIMIT ? OFFSET ?",
                    (limit, offset),
                ).fetchall()

        return [_row_to_entry(row) for row in rows]

    def get_stats(self) -> dict[str, int]:
        """Get feedback statistics."""
        with sqlite3.connect(self.db_path) as conn:
            total = conn.execute("SELECT COUNT(*) FROM feedback").fetchone()[0]
            thumbs_up = conn.execute("SELECT COUNT(*) FROM feedback WHERE rating = 1").fetchone()[0]
            thumbs_down = conn.execute(
                "SELECT COUNT(*) FROM feedback WHERE rating = -1"
            ).fetchone()[0]
            skipped = conn.execute("SELECT COUNT(*) FROM feedback WHERE rating = 0").fetchone()[0]
            unique_hypotheses = conn.execute(
                "SELECT COUNT(DISTINCT hypothesis_id) FROM feedback"
            ).fetchone()[0]

        return {
            "total": total,
            "thumbs_up": thumbs_up,
            "thumbs_down": thumbs_down,
            "skipped": skipped,
            "unique_hypotheses": unique_hypotheses,
            "approval_rate": thumbs_up / max(1, thumbs_up + thumbs_down),
        }

    def export_json(self, path: str) -> int:
        """Export all feedback to JSON. Returns count exported."""
        entries = self.get_all_feedback(limit=999999)
        data = [
            {
                "hypothesis_id": e.hypothesis_id,
                "rating": e.rating,
                "hypothesis_text": e.hypothesis_text,
                "graph_path": e.graph_path,
                "activation_score": e.activation_score,
                "elo_rating": e.elo_rating,
                "composite_score": e.composite_score,
                "comment": e.comment,
                "rater_id": e.rater_id,
                "timestamp": e.timestamp,
            }
            for e in entries
        ]
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        return len(data)


def _row_to_entry(row: sqlite3.Row) -> FeedbackEntry:
    """Convert a SQLite row to a FeedbackEntry."""
    return FeedbackEntry(
        hypothesis_id=row["hypothesis_id"],
        rating=row["rating"],
        hypothesis_text=row["hypothesis_text"],
        graph_path=row["graph_path"],
        activation_score=row["activation_score"],
        elo_rating=row["elo_rating"],
        composite_score=row["composite_score"],
        comment=row["comment"],
        rater_id=row["rater_id"],
        timestamp=row["timestamp"],
    )
