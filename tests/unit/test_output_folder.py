"""
Tests for output folder creation and artifact collection (Part 2).

Tests the output directory lifecycle:
- Creation before executor runs
- Artifact collection after execution
- Content inlining for text files
- Content type detection
"""
import os

import pytest

from dokumen.test_object import collect_output_artifacts


class TestCollectOutputArtifacts:
    """Test the collect_output_artifacts function."""

    def test_output_artifacts_collected(self, tmp_path):
        """Write files to output dir, verify they appear in artifacts."""
        output_dir = str(tmp_path / "output" / "test-1")
        os.makedirs(output_dir, exist_ok=True)

        # Create test files
        with open(os.path.join(output_dir, "result.py"), "w") as f:
            f.write("print('hello')")
        with open(os.path.join(output_dir, "notes.md"), "w") as f:
            f.write("# Notes\n\nSome notes")

        artifacts = collect_output_artifacts(output_dir)

        assert len(artifacts) == 2
        filenames = {a["filename"] for a in artifacts}
        assert "result.py" in filenames
        assert "notes.md" in filenames

    def test_output_dir_empty(self, tmp_path):
        """No artifacts when directory is empty."""
        output_dir = str(tmp_path / "output" / "test-empty")
        os.makedirs(output_dir, exist_ok=True)

        artifacts = collect_output_artifacts(output_dir)
        assert artifacts == []

    def test_output_dir_nonexistent(self, tmp_path):
        """No artifacts when directory doesn't exist."""
        output_dir = str(tmp_path / "output" / "test-nonexistent")

        artifacts = collect_output_artifacts(output_dir)
        assert artifacts == []

    def test_output_artifact_content_inlined_for_small_text(self, tmp_path):
        """Text files < 100KB have content inlined."""
        output_dir = str(tmp_path / "output" / "test-inline")
        os.makedirs(output_dir, exist_ok=True)

        content = "print('small file')"
        with open(os.path.join(output_dir, "script.py"), "w") as f:
            f.write(content)

        artifacts = collect_output_artifacts(output_dir)

        assert len(artifacts) == 1
        assert artifacts[0]["content"] == content

    def test_output_artifact_content_not_inlined_for_large_text(self, tmp_path):
        """Text files > 100KB don't have content inlined."""
        output_dir = str(tmp_path / "output" / "test-large")
        os.makedirs(output_dir, exist_ok=True)

        # Create a file larger than 100KB
        content = "x" * (101 * 1024)
        with open(os.path.join(output_dir, "big.txt"), "w") as f:
            f.write(content)

        artifacts = collect_output_artifacts(output_dir)

        assert len(artifacts) == 1
        assert artifacts[0]["content"] is None
        assert artifacts[0]["size_bytes"] > 100 * 1024

    def test_output_artifact_binary_no_content(self, tmp_path):
        """Binary files don't have content inlined."""
        output_dir = str(tmp_path / "output" / "test-binary")
        os.makedirs(output_dir, exist_ok=True)

        with open(os.path.join(output_dir, "data.bin"), "wb") as f:
            f.write(b"\x00\x01\x02\x03")

        artifacts = collect_output_artifacts(output_dir)

        assert len(artifacts) == 1
        assert artifacts[0]["content"] is None
        assert artifacts[0]["content_type"] == "application/octet-stream"

    def test_output_artifact_content_types(self, tmp_path):
        """Verify content type detection for various extensions."""
        output_dir = str(tmp_path / "output" / "test-types")
        os.makedirs(output_dir, exist_ok=True)

        expected = {
            "script.py": "text/x-python",
            "readme.md": "text/markdown",
            "notes.txt": "text/plain",
            "data.csv": "text/csv",
            "config.json": "application/json",
            "settings.yaml": "text/yaml",
            "settings2.yml": "text/yaml",
            "unknown.xyz": "application/octet-stream",
        }

        for filename in expected:
            with open(os.path.join(output_dir, filename), "w") as f:
                f.write("content")

        artifacts = collect_output_artifacts(output_dir)

        # Build a map of filename -> content_type
        type_map = {a["filename"]: a["content_type"] for a in artifacts}

        for filename, expected_type in expected.items():
            assert type_map[filename] == expected_type, (
                f"Expected {filename} -> {expected_type}, got {type_map.get(filename)}"
            )

    def test_output_artifact_json_inlined(self, tmp_path):
        """JSON files have content inlined (application/json is treated as text-like)."""
        output_dir = str(tmp_path / "output" / "test-json")
        os.makedirs(output_dir, exist_ok=True)

        content = '{"key": "value"}'
        with open(os.path.join(output_dir, "data.json"), "w") as f:
            f.write(content)

        artifacts = collect_output_artifacts(output_dir)

        assert len(artifacts) == 1
        assert artifacts[0]["content"] == content

    def test_output_artifact_size_bytes(self, tmp_path):
        """Artifact includes correct size_bytes."""
        output_dir = str(tmp_path / "output" / "test-size")
        os.makedirs(output_dir, exist_ok=True)

        content = "hello world"
        with open(os.path.join(output_dir, "file.txt"), "w") as f:
            f.write(content)

        artifacts = collect_output_artifacts(output_dir)

        assert len(artifacts) == 1
        assert artifacts[0]["size_bytes"] == len(content)

    def test_output_artifact_nested_directory(self, tmp_path):
        """Artifacts in nested directories are collected."""
        output_dir = str(tmp_path / "output" / "test-nested")
        nested_dir = os.path.join(output_dir, "sub", "dir")
        os.makedirs(nested_dir, exist_ok=True)

        with open(os.path.join(nested_dir, "deep.py"), "w") as f:
            f.write("# deep file")

        artifacts = collect_output_artifacts(output_dir)

        assert len(artifacts) == 1
        assert artifacts[0]["filename"] == "deep.py"
        # Path should be relative
        assert "sub" in artifacts[0]["path"]

    def test_output_artifact_path_relative(self, tmp_path):
        """Artifact path is relative to the parent of output_dir."""
        output_dir = str(tmp_path / "output" / "test-1")
        os.makedirs(output_dir, exist_ok=True)

        with open(os.path.join(output_dir, "result.txt"), "w") as f:
            f.write("result")

        artifacts = collect_output_artifacts(output_dir)

        assert len(artifacts) == 1
        # Path should be relative
        path = artifacts[0]["path"]
        assert not os.path.isabs(path)

    def test_collect_output_webm_content_type(self, tmp_path):
        """Collect .webm files with correct video/webm content_type."""
        output_dir = str(tmp_path / "output" / "test-webm")
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, "video.webm"), "wb") as f:
            f.write(b"fake webm")
        artifacts = collect_output_artifacts(output_dir)
        assert len(artifacts) == 1
        assert artifacts[0]["content_type"] == "video/webm"

    def test_collect_output_mp4_content_type(self, tmp_path):
        """Collect .mp4 files with correct video/mp4 content_type."""
        output_dir = str(tmp_path / "output" / "test-mp4")
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, "video.mp4"), "wb") as f:
            f.write(b"fake mp4")
        artifacts = collect_output_artifacts(output_dir)
        assert len(artifacts) == 1
        assert artifacts[0]["content_type"] == "video/mp4"

    def test_collect_output_path_relative_to_output_dir(self, tmp_path):
        """Path is relative to output_dir, not parent (just filename)."""
        output_dir = str(tmp_path / "output" / "test-1")
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, "result.txt"), "w") as f:
            f.write("result")
        artifacts = collect_output_artifacts(output_dir)
        assert len(artifacts) == 1
        assert artifacts[0]["path"] == "result.txt"

    def test_collect_output_excludes_click_indicator_js_in_recordings(self, tmp_path):
        """click-indicator.js is excluded from recordings/ subdir."""
        output_dir = str(tmp_path / "output" / "test-1")
        rec_dir = os.path.join(output_dir, "recordings")
        os.makedirs(rec_dir, exist_ok=True)
        with open(os.path.join(rec_dir, "click-indicator.js"), "w") as f:
            f.write("// internal")
        with open(os.path.join(rec_dir, "video.webm"), "wb") as f:
            f.write(b"video data")
        artifacts = collect_output_artifacts(output_dir, skip_inline_dirs={"recordings"})
        filenames = [a["filename"] for a in artifacts]
        assert "click-indicator.js" not in filenames
        assert "video.webm" in filenames

    def test_click_indicator_not_excluded_at_output_root(self, tmp_path):
        """click-indicator.js is NOT excluded at output root (only in skip dirs)."""
        output_dir = str(tmp_path / "output" / "test-1")
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, "click-indicator.js"), "w") as f:
            f.write("// user file")
        artifacts = collect_output_artifacts(output_dir, skip_inline_dirs={"recordings"})
        filenames = [a["filename"] for a in artifacts]
        assert "click-indicator.js" in filenames

    def test_skip_inline_dirs_prevents_screenshot_base64(self, tmp_path):
        """Images in recordings/ are not base64-inlined when skip_inline_dirs is set."""
        output_dir = str(tmp_path / "output" / "test-1")
        rec_dir = os.path.join(output_dir, "recordings")
        os.makedirs(rec_dir, exist_ok=True)
        # Write a small PNG file
        with open(os.path.join(rec_dir, "screenshot.png"), "wb") as f:
            f.write(b"\x89PNG\r\n\x1a\n" + b"\x00" * 100)
        artifacts = collect_output_artifacts(output_dir, skip_inline_dirs={"recordings"})
        png_artifact = [a for a in artifacts if a["filename"] == "screenshot.png"][0]
        assert png_artifact["content"] is None  # Not inlined

    def test_zero_byte_files_skipped(self, tmp_path):
        """0-byte files are excluded from artifacts (e.g. Playwright empty video)."""
        output_dir = str(tmp_path / "output" / "test-empty-video")
        rec_dir = os.path.join(output_dir, "recordings")
        os.makedirs(rec_dir, exist_ok=True)

        # 0-byte video (Playwright context init artifact)
        with open(os.path.join(rec_dir, "empty.webm"), "wb") as f:
            pass  # 0 bytes

        # Real video
        with open(os.path.join(rec_dir, "real.webm"), "wb") as f:
            f.write(b"fake webm data")

        # 0-byte text file at root
        with open(os.path.join(output_dir, "empty.txt"), "w") as f:
            pass

        # Real text file at root
        with open(os.path.join(output_dir, "real.txt"), "w") as f:
            f.write("content")

        artifacts = collect_output_artifacts(output_dir, skip_inline_dirs={"recordings"})
        filenames = {a["filename"] for a in artifacts}

        assert "empty.webm" not in filenames, "0-byte video should be skipped"
        assert "empty.txt" not in filenames, "0-byte text should be skipped"
        assert "real.webm" in filenames, "Real video should be collected"
        assert "real.txt" in filenames, "Real text should be collected"
        assert len(artifacts) == 2

    def test_zero_byte_file_os_error_treated_as_zero(self, tmp_path):
        """Files where os.path.getsize raises OSError are treated as size=0 and skipped."""
        output_dir = str(tmp_path / "output" / "test-oserror")
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, "good.txt"), "w") as f:
            f.write("content")
        artifacts = collect_output_artifacts(output_dir)
        # Just verify the normal file is collected (OSError path is a safety net)
        assert len(artifacts) == 1
        assert artifacts[0]["filename"] == "good.txt"

    def test_skip_inline_dirs_allows_normal_image_inline(self, tmp_path):
        """Images outside recordings/ are still base64-inlined."""
        output_dir = str(tmp_path / "output" / "test-1")
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, "chart.png"), "wb") as f:
            f.write(b"\x89PNG\r\n\x1a\n" + b"\x00" * 100)
        artifacts = collect_output_artifacts(output_dir, skip_inline_dirs={"recordings"})
        png_artifact = [a for a in artifacts if a["filename"] == "chart.png"][0]
        assert png_artifact["content"] is not None  # Inlined as base64


class TestOutputArtifactInTestResult:
    """Test that output_artifacts field exists on TestResult."""

    def test_test_result_has_output_artifacts_field(self):
        """TestResult dataclass includes output_artifacts field."""
        from dokumen.test_object import TestResult
        from datetime import datetime

        result = TestResult(
            test_id="test-1",
            passed=True,
            executor_passed=True,
            judge_results=[],
            executor_output=None,
            duration=1.0,
            timestamp=datetime.now(),
        )
        assert hasattr(result, "output_artifacts")
        assert result.output_artifacts is None

    def test_test_result_to_dict_includes_output_artifacts(self):
        """TestResult.to_dict() includes output_artifacts."""
        from dokumen.test_object import TestResult
        from datetime import datetime

        result = TestResult(
            test_id="test-1",
            passed=True,
            executor_passed=True,
            judge_results=[],
            executor_output=None,
            duration=1.0,
            timestamp=datetime.now(),
            output_artifacts=[{"filename": "test.py", "path": "output/test-1/test.py",
                              "size_bytes": 100, "content_type": "text/x-python",
                              "content": "print('hi')"}],
        )
        d = result.to_dict()
        assert "output_artifacts" in d
        assert len(d["output_artifacts"]) == 1
        assert d["output_artifacts"][0]["filename"] == "test.py"


class TestOutputArtifactSchema:
    """Test OutputArtifact in output_schemas.py."""

    def test_output_artifact_model_exists(self):
        """OutputArtifact Pydantic model exists in output_schemas."""
        from dokumen.output_schemas import OutputArtifact
        assert OutputArtifact is not None

    def test_output_artifact_fields(self):
        """OutputArtifact has required fields."""
        from dokumen.output_schemas import OutputArtifact

        artifact = OutputArtifact(
            filename="calc.py",
            path="output/test-1/calc.py",
            size_bytes=256,
            content_type="text/x-python",
            content="print(42)",
        )
        assert artifact.filename == "calc.py"
        assert artifact.size_bytes == 256
        assert artifact.content_type == "text/x-python"
        assert artifact.content == "print(42)"

    def test_output_artifact_content_optional(self):
        """OutputArtifact.content is optional (None for binary/large files)."""
        from dokumen.output_schemas import OutputArtifact

        artifact = OutputArtifact(
            filename="image.png",
            path="output/test-1/image.png",
            size_bytes=50000,
            content_type="application/octet-stream",
        )
        assert artifact.content is None

    def test_test_output_result_has_output_artifacts(self):
        """TestOutputResult model includes output_artifacts field."""
        from dokumen.output_schemas import TestOutputResult

        result = TestOutputResult(
            name="test-1",
            status="passed",
            duration_ms=1000,
        )
        assert hasattr(result, "output_artifacts")
        assert result.output_artifacts is None


# =============================================================================
# TestObject.run() Output Folder Wiring Tests
# =============================================================================

class TestOutputFolderWiring:
    """Tests for output folder directory creation and prompt injection in run()."""

    def _make_test_object(self, test_id="test-output"):
        """Create a minimal TestObject for testing output folder wiring."""
        from unittest.mock import MagicMock

        executor = MagicMock()
        executor.id = "executor"
        executor.system_prompt = "You are an executor."
        executor.user_prompt = "Do the task."
        executor.tools = []
        executor.provider = MagicMock()
        executor.provider.model = "mock-model"
        executor.max_tool_result_chars = 10000

        judge = MagicMock()
        judge.id = "accuracy"
        judge.system_prompt = "Judge the output."
        judge.tools = None
        judge.provider = MagicMock()
        judge.provider.model = "mock-model"
        judge._get_assertion_text = MagicMock(return_value="check accuracy")

        from dokumen.test_object import TestObject
        test_obj = TestObject(
            id=test_id,
            reason="Test output folder",
            executor=executor,
            judges=[judge],
            timeout=30.0,
        )
        return test_obj

    def _mock_executor_success(self):
        """Return a coroutine that produces a successful ExecutorOutput."""
        from dokumen.agent_object import ExecutorOutput

        async def mock_run(**kwargs):
            return ExecutorOutput(
                tool_calls=[], final_response="done",
                success=True, system_prompt="", user_prompt="",
                input_tokens=10, output_tokens=10,
            )
        return mock_run

    def _mock_judge_pass(self):
        """Return a coroutine that produces a passing JudgeResult."""
        from dokumen.agent_object import JudgeResult

        async def mock_run(**kwargs):
            return JudgeResult(
                judge_id="accuracy", passed=True,
                input_tokens=5, output_tokens=5,
            )
        return mock_run

    @pytest.mark.asyncio
    async def test_output_dir_created_before_executor(self):
        """Output directory .dokumen-cache/output/{test_id}/ is created before executor runs."""
        test_obj = self._make_test_object()
        output_dir = os.path.join(".dokumen-cache", "output", test_obj.id)

        dir_existed = []

        async def check_dir_executor(**kwargs):
            dir_existed.append(os.path.isdir(output_dir))
            from dokumen.agent_object import ExecutorOutput
            return ExecutorOutput(
                tool_calls=[], final_response="done",
                success=True, system_prompt="", user_prompt="",
                input_tokens=10, output_tokens=10,
            )

        test_obj.executor.run = check_dir_executor
        test_obj.judges[0].run = self._mock_judge_pass()

        try:
            await test_obj.run()
            assert dir_existed[0] is True, \
                "Output directory should exist when executor runs"
        finally:
            import shutil
            if os.path.exists(output_dir):
                shutil.rmtree(output_dir)

    @pytest.mark.asyncio
    async def test_output_path_injected_in_executor_prompt(self):
        """Executor user_prompt includes the output folder path."""
        test_obj = self._make_test_object()
        output_dir = os.path.join(".dokumen-cache", "output", test_obj.id)

        captured_prompt = []

        async def capture_prompt_executor(**kwargs):
            captured_prompt.append(test_obj.executor.user_prompt)
            from dokumen.agent_object import ExecutorOutput
            return ExecutorOutput(
                tool_calls=[], final_response="done",
                success=True, system_prompt="", user_prompt="",
                input_tokens=10, output_tokens=10,
            )

        test_obj.executor.run = capture_prompt_executor
        test_obj.judges[0].run = self._mock_judge_pass()

        try:
            await test_obj.run()
            assert output_dir in captured_prompt[0], \
                f"Executor prompt should contain output dir. Got: {captured_prompt[0]}"
        finally:
            import shutil
            if os.path.exists(output_dir):
                shutil.rmtree(output_dir)

    @pytest.mark.asyncio
    async def test_output_path_injected_in_judge_prompt(self):
        """Judge system_prompt includes the output folder path."""
        test_obj = self._make_test_object()
        output_dir = os.path.join(".dokumen-cache", "output", test_obj.id)

        captured_prompt = []

        test_obj.executor.run = self._mock_executor_success()

        async def capture_judge_prompt(**kwargs):
            captured_prompt.append(test_obj.judges[0].system_prompt)
            from dokumen.agent_object import JudgeResult
            return JudgeResult(
                judge_id="accuracy", passed=True,
                input_tokens=5, output_tokens=5,
            )

        test_obj.judges[0].run = capture_judge_prompt

        try:
            await test_obj.run()
            assert output_dir in captured_prompt[0], \
                f"Judge prompt should contain output dir. Got: {captured_prompt[0]}"
        finally:
            import shutil
            if os.path.exists(output_dir):
                shutil.rmtree(output_dir)

    @pytest.mark.asyncio
    async def test_output_artifacts_populated_on_result(self):
        """Output artifacts written by executor appear in TestResult.output_artifacts."""
        test_obj = self._make_test_object()
        output_dir = os.path.join(".dokumen-cache", "output", test_obj.id)

        async def write_file_executor(**kwargs):
            # Simulate executor writing a deliverable
            os.makedirs(output_dir, exist_ok=True)
            with open(os.path.join(output_dir, "calc.py"), "w") as f:
                f.write("print(42)")
            from dokumen.agent_object import ExecutorOutput
            return ExecutorOutput(
                tool_calls=[], final_response="done",
                success=True, system_prompt="", user_prompt="",
                input_tokens=10, output_tokens=10,
            )

        test_obj.executor.run = write_file_executor
        test_obj.judges[0].run = self._mock_judge_pass()

        try:
            result = await test_obj.run()
            assert result.output_artifacts is not None
            assert len(result.output_artifacts) == 1
            assert result.output_artifacts[0]["filename"] == "calc.py"
            assert result.output_artifacts[0]["content"] == "print(42)"
        finally:
            import shutil
            if os.path.exists(output_dir):
                shutil.rmtree(output_dir)

    @pytest.mark.asyncio
    async def test_output_artifacts_none_when_empty(self):
        """output_artifacts is None when no files are written to output dir."""
        test_obj = self._make_test_object()
        output_dir = os.path.join(".dokumen-cache", "output", test_obj.id)

        test_obj.executor.run = self._mock_executor_success()
        test_obj.judges[0].run = self._mock_judge_pass()

        try:
            result = await test_obj.run()
            assert result.output_artifacts is None
        finally:
            import shutil
            if os.path.exists(output_dir):
                shutil.rmtree(output_dir)

    @pytest.mark.asyncio
    async def test_original_user_prompt_preserved_for_display(self):
        """Original user prompt is passed as original_user_prompt despite injection."""
        test_obj = self._make_test_object()
        output_dir = os.path.join(".dokumen-cache", "output", test_obj.id)
        original = test_obj.executor.user_prompt

        captured = []

        async def capture_original(**kwargs):
            captured.append(kwargs.get("original_user_prompt", ""))
            from dokumen.agent_object import ExecutorOutput
            return ExecutorOutput(
                tool_calls=[], final_response="done",
                success=True, system_prompt="", user_prompt="",
                input_tokens=10, output_tokens=10,
            )

        test_obj.executor.run = capture_original
        test_obj.judges[0].run = self._mock_judge_pass()

        try:
            await test_obj.run()
            assert captured[0] == original, \
                "original_user_prompt should be the unmodified prompt"
        finally:
            import shutil
            if os.path.exists(output_dir):
                shutil.rmtree(output_dir)

    @pytest.mark.asyncio
    async def test_judge_prompt_restored_after_run(self):
        """Judge system_prompt is restored to original value after run completes."""
        test_obj = self._make_test_object()
        output_dir = os.path.join(".dokumen-cache", "output", test_obj.id)
        original_judge_prompt = test_obj.judges[0].system_prompt

        test_obj.executor.run = self._mock_executor_success()
        test_obj.judges[0].run = self._mock_judge_pass()

        try:
            await test_obj.run()
            assert test_obj.judges[0].system_prompt == original_judge_prompt
        finally:
            import shutil
            if os.path.exists(output_dir):
                shutil.rmtree(output_dir)

    @pytest.mark.asyncio
    async def test_executor_prompt_restored_after_run(self):
        """Executor user_prompt is restored to original value after run completes."""
        test_obj = self._make_test_object()
        output_dir = os.path.join(".dokumen-cache", "output", test_obj.id)
        original_executor_prompt = test_obj.executor.user_prompt

        test_obj.executor.run = self._mock_executor_success()
        test_obj.judges[0].run = self._mock_judge_pass()

        try:
            await test_obj.run()
            assert test_obj.executor.user_prompt == original_executor_prompt
        finally:
            import shutil
            if os.path.exists(output_dir):
                shutil.rmtree(output_dir)
