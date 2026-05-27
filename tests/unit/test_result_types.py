"""Tests for the canonical result types: ExecutorResult and JudgeVerdict."""

from datetime import datetime
from dokumen.sdk.types import ExecutorResult, JudgeVerdict


class TestExecutorResultFields:
    """Test that ExecutorResult has all required fields."""

    def test_executor_result_minimal_construction(self):
        """ExecutorResult can be created with only required fields."""
        result = ExecutorResult(
            success=True,
            final_response="The document is accurate.",
        )
        assert result.success is True
        assert result.final_response == "The document is accurate."
        assert result.error is None
        assert result.tool_calls == []
        assert result.conversation_log == []
        assert result.input_tokens == 0
        assert result.output_tokens == 0
        assert result.cache_creation_tokens == 0
        assert result.cache_read_tokens == 0
        assert result.system_prompt == ""
        assert result.user_prompt == ""
        assert result.original_user_prompt == ""
        assert result.duration_ms == 0

    def test_executor_result_full_construction(self):
        """ExecutorResult can be created with all fields populated."""
        result = ExecutorResult(
            success=True,
            final_response="Response text",
            tool_calls=[{"tool_name": "read_file", "parameters": {}, "result": "ok"}],
            input_tokens=100,
            output_tokens=50,
            cache_creation_tokens=10,
            cache_read_tokens=5,
            conversation_log=[{"role": "user", "content": "test"}],
            system_prompt="You are a validator.",
            user_prompt="Read docs/api.md",
            original_user_prompt="Read docs/api.md",
            duration_ms=1500,
            error=None,
        )
        assert result.success is True
        assert result.input_tokens == 100
        assert result.output_tokens == 50
        assert result.cache_creation_tokens == 10
        assert result.cache_read_tokens == 5
        assert len(result.tool_calls) == 1
        assert result.duration_ms == 1500
        assert result.system_prompt == "You are a validator."

    def test_executor_result_to_dict(self):
        """to_dict() produces expected dict shape."""
        result = ExecutorResult(
            success=False,
            final_response="Error occurred",
            error="Timeout",
            tool_calls=[{"tool_name": "glob", "parameters": {"pattern": "*.md"}, "result": "found"}],
            input_tokens=200,
            output_tokens=100,
            cache_creation_tokens=20,
            cache_read_tokens=10,
            conversation_log=[],
            system_prompt="sys",
            user_prompt="usr",
            original_user_prompt="orig",
            duration_ms=5000,
        )
        d = result.to_dict()
        assert d["success"] is False
        assert d["final_response"] == "Error occurred"
        assert d["error"] == "Timeout"
        assert d["system_prompt"] == "sys"
        assert d["user_prompt"] == "usr"
        assert d["original_user_prompt"] == "orig"
        assert d["input_tokens"] == 200
        assert d["output_tokens"] == 100
        assert d["cache_creation_tokens"] == 20
        assert d["cache_read_tokens"] == 10
        assert len(d["tool_calls"]) == 1

    def test_executor_result_error_case(self):
        """ExecutorResult correctly represents an error state."""
        result = ExecutorResult(
            success=False,
            final_response="",
            error="Executor timed out after 60s",
            duration_ms=60000,
        )
        assert result.success is False
        assert result.error == "Executor timed out after 60s"
        assert result.duration_ms == 60000


class TestJudgeVerdictFields:
    """Test that JudgeVerdict has all required fields."""

    def test_judge_verdict_minimal_construction(self):
        """JudgeVerdict can be created with only required fields."""
        verdict = JudgeVerdict(
            judge_id="accuracy",
            passed=True,
        )
        assert verdict.judge_id == "accuracy"
        assert verdict.passed is True
        assert verdict.failure_reason is None
        assert verdict.confidence is None
        assert verdict.response is None
        assert verdict.tool_calls == []
        assert verdict.assertion_text is None
        assert verdict.input_tokens == 0
        assert verdict.output_tokens == 0
        assert verdict.cache_creation_tokens == 0
        assert verdict.cache_read_tokens == 0
        assert verdict.conversation_log == []
        assert verdict.error is None

    def test_judge_verdict_full_construction(self):
        """JudgeVerdict can be created with all fields populated."""
        verdict = JudgeVerdict(
            judge_id="completeness",
            passed=False,
            failure_reason="Missing section on auth methods",
            confidence=0.85,
            response='{"verdict": "FAIL", "confidence": 0.85, "reason": "..."}',
            tool_calls=[{"tool_name": "read_file", "parameters": {}, "result": "ok"}],
            assertion_text="Must mention OAuth and API keys",
            input_tokens=300,
            output_tokens=150,
            cache_creation_tokens=30,
            cache_read_tokens=15,
            conversation_log=[{"role": "assistant", "content": "..."}],
            error=False,
        )
        assert verdict.judge_id == "completeness"
        assert verdict.passed is False
        assert verdict.failure_reason == "Missing section on auth methods"
        assert verdict.confidence == 0.85
        assert verdict.input_tokens == 300
        assert verdict.error is False

    def test_judge_verdict_to_dict(self):
        """to_dict() produces expected dict shape."""
        verdict = JudgeVerdict(
            judge_id="accuracy",
            passed=True,
            confidence=0.95,
            response="PASS",
            assertion_text="Check auth methods",
            input_tokens=100,
            output_tokens=50,
            cache_creation_tokens=10,
            cache_read_tokens=5,
        )
        d = verdict.to_dict()
        assert d["judge_id"] == "accuracy"
        assert d["passed"] is True
        assert d["confidence"] == 0.95
        assert d["assertion_text"] == "Check auth methods"
        assert d["input_tokens"] == 100
        assert d["output_tokens"] == 50
        assert d["cache_creation_tokens"] == 10
        assert d["cache_read_tokens"] == 5
        assert d["error"] is None

    def test_judge_verdict_error_case(self):
        """JudgeVerdict correctly represents timeout/error state."""
        verdict = JudgeVerdict(
            judge_id="timeout-judge",
            passed=False,
            failure_reason="Judge timed out after 30s",
            error=True,
        )
        assert verdict.passed is False
        assert verdict.error is True
        assert verdict.failure_reason == "Judge timed out after 30s"


class TestBackwardCompatAliases:
    """Test that old type names still work as aliases."""

    def test_executor_output_alias(self):
        """ExecutorOutput should be an alias for ExecutorResult."""
        from dokumen.agent_object import ExecutorOutput
        assert ExecutorOutput is ExecutorResult

    def test_judge_result_alias(self):
        """JudgeResult should be an alias for JudgeVerdict."""
        from dokumen.agent_object import JudgeResult
        assert JudgeResult is JudgeVerdict

    def test_sdk_executor_result_alias(self):
        """SdkExecutorResult should be an alias for ExecutorResult."""
        from dokumen.sdk.types import SdkExecutorResult
        assert SdkExecutorResult is ExecutorResult

    def test_sdk_judge_result_alias(self):
        """SdkJudgeResult should be an alias for JudgeVerdict."""
        from dokumen.sdk.types import SdkJudgeResult
        assert SdkJudgeResult is JudgeVerdict


class TestExecutorResultDefaults:
    """Test that all optional fields have correct defaults."""

    def test_defaults(self):
        """All optional fields default correctly."""
        result = ExecutorResult(success=True, final_response="ok")
        assert result.tool_calls == []
        assert result.conversation_log == []
        assert result.system_prompt == ""
        assert result.user_prompt == ""
        assert result.original_user_prompt == ""
        assert result.input_tokens == 0
        assert result.output_tokens == 0
        assert result.cache_creation_tokens == 0
        assert result.cache_read_tokens == 0
        assert result.duration_ms == 0
        assert result.error is None


class TestJudgeVerdictDefaults:
    """Test that all optional fields have correct defaults."""

    def test_defaults(self):
        """All optional fields default correctly."""
        verdict = JudgeVerdict(judge_id="test", passed=True)
        assert verdict.failure_reason is None
        assert verdict.confidence is None
        assert verdict.response is None
        assert verdict.tool_calls == []
        assert verdict.assertion_text is None
        assert verdict.input_tokens == 0
        assert verdict.output_tokens == 0
        assert verdict.cache_creation_tokens == 0
        assert verdict.cache_read_tokens == 0
        assert verdict.conversation_log == []
        assert verdict.error is None
