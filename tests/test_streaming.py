"""Tests for streaming ingestion module."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from kosa.ingestion.streaming import (
    IngestionJob,
    fetch_arxiv_by_date,
    get_job,
    merge_citations_incremental,
    merge_papers_incremental,
    read_watermark,
    register_job,
    write_watermark,
)


class TestIngestionJobRegistry:
    def test_register_and_get(self):
        job = IngestionJob(job_id="test-1")
        register_job(job)
        assert get_job("test-1") is job

    def test_get_missing(self):
        assert get_job("nonexistent") is None

    def test_to_dict(self):
        job = IngestionJob(
            job_id="test-2",
            status="completed",
            papers_fetched=10,
            papers_extracted=8,
        )
        d = job.to_dict()
        assert d["job_id"] == "test-2"
        assert d["status"] == "completed"
        assert d["papers_fetched"] == 10
        assert d["papers_extracted"] == 8

    def test_to_dict_truncates_errors(self):
        job = IngestionJob(job_id="test-3", errors=[f"err-{i}" for i in range(20)])
        d = job.to_dict()
        assert len(d["errors"]) == 10  # last 10 only


class TestWatermark:
    def test_read_write(self, tmp_path, monkeypatch):
        wm_path = tmp_path / "watermark.json"
        monkeypatch.setattr("kosa.ingestion.streaming.WATERMARK_PATH", wm_path)

        assert read_watermark() is None
        write_watermark("2025-03-15")
        assert read_watermark() == "2025-03-15"
        write_watermark("2025-03-20")
        assert read_watermark() == "2025-03-20"


class TestFetchArxivByDate:
    def test_basic_fetch(self):
        xml_response = """<?xml version="1.0" encoding="UTF-8"?>
        <feed xmlns="http://www.w3.org/2005/Atom">
          <entry>
            <id>http://arxiv.org/abs/2501.00001v1</id>
            <title>Test Paper Title</title>
            <summary>This is a test abstract.</summary>
            <published>2025-01-01T00:00:00Z</published>
            <author><name>Test Author</name></author>
          </entry>
        </feed>"""

        mock_resp = MagicMock()
        mock_resp.text = xml_response
        mock_resp.raise_for_status = MagicMock()

        with patch("kosa.ingestion.streaming.httpx.get", return_value=mock_resp):
            results = fetch_arxiv_by_date("2025-01-01", "2025-01-31", max_results=10)

        assert len(results) == 1
        assert results[0]["arxiv_id"] == "2501.00001"
        assert results[0]["title"] == "Test Paper Title"
        assert results[0]["authors"] == ["Test Author"]
        assert results[0]["published"] == "2025-01-01"

    def test_empty_response(self):
        xml_response = """<?xml version="1.0" encoding="UTF-8"?>
        <feed xmlns="http://www.w3.org/2005/Atom">
        </feed>"""

        mock_resp = MagicMock()
        mock_resp.text = xml_response
        mock_resp.raise_for_status = MagicMock()

        with patch("kosa.ingestion.streaming.httpx.get", return_value=mock_resp):
            results = fetch_arxiv_by_date("2025-01-01", "2025-01-31")

        assert results == []


class TestMergePapersIncremental:
    def test_merge_creates_nodes(self):
        from kosa.ingestion.extract import (
            PaperExtractionResult,
        )
        from kosa.ingestion.pipeline import PaperMetadata

        mock_session = MagicMock()
        mock_driver = MagicMock()
        mock_driver.session.return_value.__enter__ = MagicMock(return_value=mock_session)
        mock_driver.session.return_value.__exit__ = MagicMock(return_value=False)

        paper = PaperMetadata(
            arxiv_id="2501.00001",
            title="Test",
            abstract="Abstract",
            authors=["Author"],
            year=2025,
            venue=None,
            significance=0.5,
        )

        extraction = PaperExtractionResult(
            arxiv_id="2501.00001",
            title="Test",
            abstract="Abstract",
            authors=["Author"],
            year=2025,
            venue=None,
            entities=[],
            relations=[],
        )

        counts = merge_papers_incremental(mock_driver, [paper], [extraction])
        assert counts["papers"] == 1
        assert mock_session.run.called


class TestMergeCitationsIncremental:
    def test_merge_creates_edges_and_stubs(self):
        mock_session = MagicMock()
        mock_driver = MagicMock()
        mock_driver.session.return_value.__enter__ = MagicMock(return_value=mock_session)
        mock_driver.session.return_value.__exit__ = MagicMock(return_value=False)

        # Simulate stub creation
        mock_result = MagicMock()
        mock_result.single.return_value = {"is_stub": True}
        mock_session.run.return_value = mock_result

        citations = {"2501.00001": ["2401.12345", "2301.99999"]}
        counts = merge_citations_incremental(mock_driver, citations)

        assert counts["edges"] == 2
        # Each citation triggers 2 calls: MERGE stub + MERGE edge
        assert mock_session.run.call_count == 4
