"""
Tests for browser artifacts feature.

TDD tests for browser recording and screenshot artifacts:
- BrowserArtifact model validation
- Artifact collection from recordings directory
- Artifact inclusion in results.json
"""
import json
import os
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from dokumen.output_schemas import BrowserArtifact, TestOutputResult


# =============================================================================
# BrowserArtifact Model Tests
# =============================================================================

class TestBrowserArtifactModel:
    """Tests for BrowserArtifact Pydantic model."""

    def test_video_artifact_valid(self):
        """Video artifact with all fields is valid."""
        artifact = BrowserArtifact(
            type="video",
            path="video-1.webm",
            filename="video-1.webm",
            size_bytes=1024000
        )
        assert artifact.type == "video"
        assert artifact.path == "video-1.webm"
        assert artifact.filename == "video-1.webm"
        assert artifact.size_bytes == 1024000

    def test_screenshot_artifact_valid(self):
        """Screenshot artifact with all fields is valid."""
        artifact = BrowserArtifact(
            type="screenshot",
            path="screenshots/page-20240124-120000.png",
            filename="page-20240124-120000.png",
            size_bytes=52400
        )
        assert artifact.type == "screenshot"
        assert artifact.path == "screenshots/page-20240124-120000.png"
        assert artifact.filename == "page-20240124-120000.png"

    def test_size_bytes_optional(self):
        """size_bytes field is optional."""
        artifact = BrowserArtifact(
            type="video",
            path="video-1.webm",
            filename="video-1.webm"
        )
        assert artifact.size_bytes is None

    def test_type_must_be_video_or_screenshot(self):
        """Type field only accepts 'video' or 'screenshot'."""
        # Valid types
        BrowserArtifact(type="video", path="v.webm", filename="v.webm")
        BrowserArtifact(type="screenshot", path="s.png", filename="s.png")

        # Invalid type should raise validation error
        with pytest.raises(Exception):  # Pydantic ValidationError
            BrowserArtifact(type="audio", path="a.mp3", filename="a.mp3")

    def test_artifact_serializes_to_dict(self):
        """Artifact can be serialized to dict for JSON output."""
        artifact = BrowserArtifact(
            type="video",
            path="video-1.webm",
            filename="video-1.webm",
            size_bytes=1024
        )
        data = artifact.model_dump()
        assert data["type"] == "video"
        assert data["path"] == "video-1.webm"
        assert data["filename"] == "video-1.webm"
        assert data["size_bytes"] == 1024
        # Inherited fields from OutputArtifact base
        assert data["source"] == "browser"
        assert data["content_type"] == "application/octet-stream"
        assert data["download_url"] is None
        assert data["content"] is None


# =============================================================================
# TestOutputResult Browser Artifacts Tests
# =============================================================================

class TestTestOutputResultBrowserArtifacts:
    """Tests for browser_artifacts field in TestOutputResult."""

    def test_browser_artifacts_field_exists(self):
        """TestOutputResult has browser_artifacts field."""
        result = TestOutputResult(
            name="test-with-browser",
            status="passed",
            duration_ms=5000,
            browser_artifacts=None
        )
        assert hasattr(result, 'browser_artifacts')

    def test_browser_artifacts_can_be_list(self):
        """browser_artifacts accepts a list of BrowserArtifact."""
        artifacts = [
            BrowserArtifact(type="video", path="video-1.webm", filename="video-1.webm"),
            BrowserArtifact(type="screenshot", path="screenshots/page.png", filename="page.png"),
        ]
        result = TestOutputResult(
            name="test-with-browser",
            status="passed",
            duration_ms=5000,
            browser_artifacts=artifacts
        )
        assert len(result.browser_artifacts) == 2
        assert result.browser_artifacts[0].type == "video"
        assert result.browser_artifacts[1].type == "screenshot"

    def test_browser_artifacts_serializes_in_json(self):
        """browser_artifacts is included in JSON serialization."""
        artifacts = [
            BrowserArtifact(type="video", path="video-1.webm", filename="video-1.webm", size_bytes=1024),
        ]
        result = TestOutputResult(
            name="test-with-browser",
            status="passed",
            duration_ms=5000,
            browser_artifacts=artifacts
        )
        data = result.model_dump()
        assert "browser_artifacts" in data
        assert len(data["browser_artifacts"]) == 1
        assert data["browser_artifacts"][0]["type"] == "video"

    def test_browser_artifacts_null_when_not_set(self):
        """browser_artifacts is null/None when not provided."""
        result = TestOutputResult(
            name="test-without-browser",
            status="passed",
            duration_ms=5000
        )
        assert result.browser_artifacts is None

    def test_browser_artifacts_empty_list_valid(self):
        """browser_artifacts can be an empty list."""
        result = TestOutputResult(
            name="test-browser-no-recordings",
            status="passed",
            duration_ms=5000,
            browser_artifacts=[]
        )
        assert result.browser_artifacts == []


# =============================================================================
# Artifact Collection Tests (for TestObject)
# =============================================================================

class TestArtifactCollection:
    """Tests for artifact collection from recordings directory."""

    @pytest.fixture
    def recordings_dir(self, tmp_path):
        """Create a mock recordings directory with artifacts."""
        test_dir = tmp_path / ".dokumen-cache" / "recordings" / "app-login-test"
        test_dir.mkdir(parents=True)

        # Create video file
        video_file = test_dir / "video-1.webm"
        video_file.write_bytes(b"mock video content" * 100)

        # Create screenshots subdirectory
        screenshots_dir = test_dir / "screenshots"
        screenshots_dir.mkdir()

        # Create screenshot files
        (screenshots_dir / "page-20240124-120000.png").write_bytes(b"mock png content")
        (screenshots_dir / "page-20240124-120001.png").write_bytes(b"mock png content")

        return test_dir

    def test_collect_artifacts_finds_webm_videos(self, recordings_dir):
        """Collector finds .webm video files."""
        from dokumen.test_object import collect_browser_artifacts

        artifacts = collect_browser_artifacts(str(recordings_dir))
        videos = [a for a in artifacts if a["type"] == "video"]

        assert len(videos) == 1
        assert videos[0]["filename"] == "video-1.webm"
        assert videos[0]["path"] == "video-1.webm"

    def test_collect_artifacts_finds_screenshots(self, recordings_dir):
        """Collector finds screenshot images in subdirectories."""
        from dokumen.test_object import collect_browser_artifacts

        artifacts = collect_browser_artifacts(str(recordings_dir))
        screenshots = [a for a in artifacts if a["type"] == "screenshot"]

        assert len(screenshots) == 2
        # Paths should be relative to output_dir
        paths = [s["path"] for s in screenshots]
        assert "screenshots/page-20240124-120000.png" in paths
        assert "screenshots/page-20240124-120001.png" in paths

    def test_collect_artifacts_returns_size_bytes(self, recordings_dir):
        """Collector includes file sizes."""
        from dokumen.test_object import collect_browser_artifacts

        artifacts = collect_browser_artifacts(str(recordings_dir))

        for artifact in artifacts:
            assert "size_bytes" in artifact
            assert artifact["size_bytes"] > 0

    def test_collect_artifacts_empty_for_missing_dir(self, tmp_path):
        """Returns empty list if directory doesn't exist."""
        from dokumen.test_object import collect_browser_artifacts

        artifacts = collect_browser_artifacts(str(tmp_path / "nonexistent"))

        assert artifacts == []

    def test_collect_artifacts_empty_for_no_media(self, tmp_path):
        """Returns empty list if no video/image files found."""
        from dokumen.test_object import collect_browser_artifacts

        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        (empty_dir / "notes.txt").write_text("not a media file")

        artifacts = collect_browser_artifacts(str(empty_dir))

        assert artifacts == []

    def test_collect_artifacts_finds_mp4_videos(self, tmp_path):
        """Collector finds .mp4 video files."""
        from dokumen.test_object import collect_browser_artifacts

        test_dir = tmp_path / "recordings"
        test_dir.mkdir()
        (test_dir / "video.mp4").write_bytes(b"mock mp4")

        artifacts = collect_browser_artifacts(str(test_dir))
        videos = [a for a in artifacts if a["type"] == "video"]

        assert len(videos) == 1
        assert videos[0]["filename"] == "video.mp4"

    def test_collect_artifacts_finds_jpg_screenshots(self, tmp_path):
        """Collector finds .jpg/.jpeg screenshot files."""
        from dokumen.test_object import collect_browser_artifacts

        test_dir = tmp_path / "recordings"
        test_dir.mkdir()
        (test_dir / "shot1.jpg").write_bytes(b"mock jpg")
        (test_dir / "shot2.jpeg").write_bytes(b"mock jpeg")

        artifacts = collect_browser_artifacts(str(test_dir))
        screenshots = [a for a in artifacts if a["type"] == "screenshot"]

        assert len(screenshots) == 2


# =============================================================================
# Directory Cleanup Tests (prevent duplicate videos)
# =============================================================================

class TestOutputDirectoryCleanup:
    """Tests for clearing output directory before test runs to prevent duplicate artifacts."""

    def test_clear_output_dir_removes_stale_videos(self, tmp_path):
        """clear_output_dir removes existing videos from previous runs."""
        from dokumen.test_object import clear_output_dir

        # Create directory with pre-existing videos (from previous run)
        test_dir = tmp_path / ".dokumen-cache" / "recordings" / "app-login-test"
        test_dir.mkdir(parents=True)
        (test_dir / "old-video.webm").write_bytes(b"old video from previous run")
        (test_dir / "another-old.webm").write_bytes(b"another old video")

        # Clear the directory
        clear_output_dir(str(test_dir))

        # Directory should be empty or not exist
        assert not test_dir.exists() or len(list(test_dir.iterdir())) == 0

    def test_clear_output_dir_handles_nonexistent_dir(self, tmp_path):
        """clear_output_dir handles gracefully when directory doesn't exist."""
        from dokumen.test_object import clear_output_dir

        nonexistent = tmp_path / "does" / "not" / "exist"

        # Should not raise
        clear_output_dir(str(nonexistent))

    def test_clear_output_dir_removes_screenshots_subdir(self, tmp_path):
        """clear_output_dir removes screenshots subdirectory too."""
        from dokumen.test_object import clear_output_dir

        test_dir = tmp_path / ".dokumen-cache" / "recordings" / "app-login-test"
        screenshots_dir = test_dir / "screenshots"
        screenshots_dir.mkdir(parents=True)
        (test_dir / "video.webm").write_bytes(b"video")
        (screenshots_dir / "page.png").write_bytes(b"screenshot")

        clear_output_dir(str(test_dir))

        assert not test_dir.exists()

    def test_clear_output_dir_only_affects_target_directory(self, tmp_path):
        """clear_output_dir only removes the specific test directory, not siblings."""
        from dokumen.test_object import clear_output_dir

        recordings_base = tmp_path / ".dokumen-cache" / "recordings"

        # Create two test directories
        test1_dir = recordings_base / "test-1"
        test2_dir = recordings_base / "test-2"
        test1_dir.mkdir(parents=True)
        test2_dir.mkdir(parents=True)
        (test1_dir / "video.webm").write_bytes(b"test1 video")
        (test2_dir / "video.webm").write_bytes(b"test2 video")

        # Clear only test-1
        clear_output_dir(str(test1_dir))

        # test-1 should be gone, test-2 should remain
        assert not test1_dir.exists()
        assert test2_dir.exists()
        assert (test2_dir / "video.webm").exists()
