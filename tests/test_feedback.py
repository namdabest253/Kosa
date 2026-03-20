"""Tests for human feedback logging."""

import os
import tempfile

import pytest

from kosa.ranking.feedback import FeedbackEntry, FeedbackStore


@pytest.fixture
def feedback_store():
    """Create a temporary feedback store."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    store = FeedbackStore(db_path)
    yield store
    os.unlink(db_path)


def _make_entry(
    hyp_id: str = "hyp_001",
    rating: int = 1,
) -> FeedbackEntry:
    return FeedbackEntry(
        hypothesis_id=hyp_id,
        rating=rating,
        hypothesis_text="Test hypothesis",
        graph_path="A -[MITIGATES]-> B",
        activation_score=0.85,
        elo_rating=1550.0,
        composite_score=0.7,
        comment="Looks promising",
        rater_id="user1",
    )


class TestFeedbackStore:
    def test_log_and_retrieve(self, feedback_store):
        entry = _make_entry()
        row_id = feedback_store.log_feedback(entry)
        assert row_id > 0

        retrieved = feedback_store.get_feedback("hyp_001")
        assert len(retrieved) == 1
        assert retrieved[0].hypothesis_id == "hyp_001"
        assert retrieved[0].rating == 1
        assert retrieved[0].hypothesis_text == "Test hypothesis"

    def test_multiple_feedback_same_hypothesis(self, feedback_store):
        feedback_store.log_feedback(_make_entry("hyp_001", rating=1))
        feedback_store.log_feedback(_make_entry("hyp_001", rating=-1))

        retrieved = feedback_store.get_feedback("hyp_001")
        assert len(retrieved) == 2

    def test_get_all_feedback(self, feedback_store):
        feedback_store.log_feedback(_make_entry("hyp_001", rating=1))
        feedback_store.log_feedback(_make_entry("hyp_002", rating=-1))
        feedback_store.log_feedback(_make_entry("hyp_003", rating=1))

        all_entries = feedback_store.get_all_feedback()
        assert len(all_entries) == 3

    def test_filter_by_rating(self, feedback_store):
        feedback_store.log_feedback(_make_entry("hyp_001", rating=1))
        feedback_store.log_feedback(_make_entry("hyp_002", rating=-1))
        feedback_store.log_feedback(_make_entry("hyp_003", rating=1))

        thumbs_up = feedback_store.get_all_feedback(rating_filter=1)
        assert len(thumbs_up) == 2

        thumbs_down = feedback_store.get_all_feedback(rating_filter=-1)
        assert len(thumbs_down) == 1

    def test_stats(self, feedback_store):
        feedback_store.log_feedback(_make_entry("hyp_001", rating=1))
        feedback_store.log_feedback(_make_entry("hyp_002", rating=-1))
        feedback_store.log_feedback(_make_entry("hyp_003", rating=1))
        feedback_store.log_feedback(_make_entry("hyp_003", rating=0))

        stats = feedback_store.get_stats()
        assert stats["total"] == 4
        assert stats["thumbs_up"] == 2
        assert stats["thumbs_down"] == 1
        assert stats["skipped"] == 1
        assert stats["unique_hypotheses"] == 3
        assert abs(stats["approval_rate"] - 2 / 3) < 0.01

    def test_export_json(self, feedback_store):
        feedback_store.log_feedback(_make_entry("hyp_001", rating=1))
        feedback_store.log_feedback(_make_entry("hyp_002", rating=-1))

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            export_path = f.name

        try:
            count = feedback_store.export_json(export_path)
            assert count == 2

            import json

            with open(export_path) as f:
                data = json.load(f)
            assert len(data) == 2
            assert data[0]["hypothesis_id"] in {"hyp_001", "hyp_002"}
        finally:
            os.unlink(export_path)

    def test_pagination(self, feedback_store):
        for i in range(5):
            feedback_store.log_feedback(_make_entry(f"hyp_{i:03d}"))

        page1 = feedback_store.get_all_feedback(limit=2, offset=0)
        page2 = feedback_store.get_all_feedback(limit=2, offset=2)
        assert len(page1) == 2
        assert len(page2) == 2
        assert page1[0].hypothesis_id != page2[0].hypothesis_id

    def test_empty_store(self, feedback_store):
        assert feedback_store.get_all_feedback() == []
        stats = feedback_store.get_stats()
        assert stats["total"] == 0
