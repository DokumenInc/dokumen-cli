"""Tests for unified output artifact schemas (Phase 7)."""
import os
import pytest
from dokumen.output_schemas import (
    OutputArtifact, BrowserArtifact, ReportArtifact, TestOutputResult,
    _infer_content_type, _to_unified,
)


class TestOutputArtifactModel:
    """Tests for unified OutputArtifact base model."""

    def test_output_artifact_size_bytes_optional_none(self):
        """size_bytes defaults to None, not 0."""
        artifact = OutputArtifact(filename="f.txt", path="f.txt")
        assert artifact.size_bytes is None

    def test_browser_artifact_inherits_output_artifact(self):
        """BrowserArtifact inherits from OutputArtifact."""
        assert issubclass(BrowserArtifact, OutputArtifact)

    def test_report_artifact_inherits_output_artifact(self):
        """ReportArtifact inherits from OutputArtifact."""
        assert issubclass(ReportArtifact, OutputArtifact)

    def test_browser_artifact_default_source_is_browser(self):
        """BrowserArtifact defaults source to 'browser'."""
        ba = BrowserArtifact(type="video", path="v.webm", filename="v.webm")
        assert ba.source == "browser"

    def test_report_artifact_default_source_is_report(self):
        """ReportArtifact defaults source to 'report'."""
        ra = ReportArtifact(path="report.md", filename="report.md")
        assert ra.source == "report"


class TestAllArtifactsProperty:
    """Tests for TestOutputResult.all_artifacts property."""

    def test_all_artifacts_returns_output_when_source_present(self):
        """When output_artifacts have source field, return them directly."""
        artifacts = [
            OutputArtifact(filename="v.webm", path="recordings/v.webm",
                          content_type="video/webm", source="browser"),
            OutputArtifact(filename="report.md", path="report.md",
                          content_type="text/markdown", source="report", content="# Report"),
        ]
        result = TestOutputResult(
            name="t", status="passed", duration_ms=100,
            output_artifacts=artifacts,
        )
        all_a = result.all_artifacts
        assert len(all_a) == 2
        assert all_a[0].source == "browser"
        assert all_a[1].source == "report"

    def test_all_artifacts_merges_all_three_when_no_source(self):
        """When no source field, merges browser + report + output artifacts."""
        result = TestOutputResult(
            name="t", status="passed", duration_ms=100,
            browser_artifacts=[
                BrowserArtifact(type="video", path="v.webm", filename="v.webm"),
            ],
            report_artifacts=[
                ReportArtifact(path="r.md", filename="r.md", content="# R"),
            ],
            output_artifacts=[
                OutputArtifact(filename="f.bin", path="f.bin", size_bytes=10),
            ],
        )
        all_a = result.all_artifacts
        assert len(all_a) == 3
        sources = [a.source for a in all_a]
        assert "browser" in sources
        assert "report" in sources
        assert "output" in sources

    def test_all_artifacts_dedup_by_path(self):
        """Same path in multiple fields appears only once."""
        result = TestOutputResult(
            name="t", status="passed", duration_ms=100,
            browser_artifacts=[
                BrowserArtifact(type="screenshot", path="img.png", filename="img.png"),
            ],
            output_artifacts=[
                OutputArtifact(filename="img.png", path="img.png",
                              content_type="image/png"),
            ],
        )
        all_a = result.all_artifacts
        paths = [a.path for a in all_a]
        assert paths.count("img.png") == 1

    def test_all_artifacts_empty_when_all_none(self):
        """Returns empty list when all artifact fields are None."""
        result = TestOutputResult(name="t", status="passed", duration_ms=100)
        assert result.all_artifacts == []

    def test_all_artifacts_legacy_browser_gets_mime(self):
        """BrowserArtifact .webm gets video/webm, .png gets image/png."""
        result = TestOutputResult(
            name="t", status="passed", duration_ms=100,
            browser_artifacts=[
                BrowserArtifact(type="video", path="v.webm", filename="v.webm"),
                BrowserArtifact(type="screenshot", path="s.png", filename="s.png"),
            ],
        )
        all_a = result.all_artifacts
        ct_map = {a.filename: a.content_type for a in all_a}
        assert ct_map["v.webm"] == "video/webm"
        assert ct_map["s.png"] == "image/png"

    def test_all_artifacts_legacy_report_gets_markdown_mime(self):
        """ReportArtifact always gets text/markdown."""
        result = TestOutputResult(
            name="t", status="passed", duration_ms=100,
            report_artifacts=[
                ReportArtifact(path="r.md", filename="r.md", content="# Report"),
            ],
        )
        all_a = result.all_artifacts
        assert all_a[0].content_type == "text/markdown"

    def test_all_artifacts_legacy_returns_output_artifact_instances(self):
        """All merged items are OutputArtifact type."""
        result = TestOutputResult(
            name="t", status="passed", duration_ms=100,
            browser_artifacts=[
                BrowserArtifact(type="video", path="v.webm", filename="v.webm"),
            ],
            report_artifacts=[
                ReportArtifact(path="r.md", filename="r.md"),
            ],
        )
        for a in result.all_artifacts:
            assert isinstance(a, OutputArtifact)


class TestInferContentType:
    """Tests for _infer_content_type helper."""

    def test_webm_maps_to_video(self):
        assert _infer_content_type("file.webm") == "video/webm"

    def test_mp4_maps_to_video(self):
        assert _infer_content_type("file.mp4") == "video/mp4"

    def test_png_maps_to_image(self):
        assert _infer_content_type("file.png") == "image/png"

    def test_unknown_maps_to_octet_stream(self):
        assert _infer_content_type("file.xyz") == "application/octet-stream"
