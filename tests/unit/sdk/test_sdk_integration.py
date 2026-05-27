"""End-to-end integration tests for the SDK agent path.

Tests the full flow: scaffold -> loader -> SDK agents -> wrapper -> ExecutorOutput/JudgeResult.
Now that the legacy path is removed, these tests verify the single SDK execution path.
"""

import json
from pathlib import Path
from typing import Any, List, Optional

import pytest
import yaml

from claude_agent_sdk import ClaudeAgentOptions, ResultMessage

from dokumen.agent_object import AgentType, ExecutorOutput, JudgeResult
from dokumen.sdk.agent_wrapper import SdkExecutorWrapper, SdkJudgeWrapper
from dokumen.sdk.executor import ExecutorAgent
from dokumen.sdk.judge import JudgeAgent
from dokumen.sdk.query_runner import MockQueryRunner
from dokumen.sdk.testing import (
    make_assistant,
    make_executor_simple,
    make_executor_with_tools,
    make_init,
    make_judge_fail,
    make_judge_pass,
    make_result,
    make_tool_result,
)
from dokumen.sdk.types import SdkExecutorResult, SdkJudgeResult


def _write_scaffold(tmp_path: Path, **overrides) -> Path:
    """Write a test scaffold to tmp_path."""
    scaffold = {
        "name": "integration-test",
        "reason": "E2E SDK integration",
        "files": [{"path": "docs/test.md"}],
        "executor": {
            "system_prompt": "You are a doc validator.",
            "user_prompt": "Check the docs for accuracy.",
            "tools": ["read_file"],
        },
        "judges": [
            {
                "name": "accuracy",
                "system_prompt": 'Evaluate accuracy. Return JSON: {"verdict": "PASS", "confidence": 0.9, "reason": "..."}',
            }
        ],
    }
    scaffold.update(overrides)
    path = tmp_path / "integration-test.test.yaml"
    path.write_text(yaml.dump(scaffold))
    return path


def _setup_project(tmp_path: Path) -> None:
    """Create minimal project structure for loader."""
    (tmp_path / "docs").mkdir(exist_ok=True)
    (tmp_path / "docs" / "test.md").write_text("# Test\nThis is a test doc.")


class TestSdkExecutorToWrapperIntegration:
    async def test_executor_produces_compatible_output(self):
        """ExecutorAgent -> SdkExecutorWrapper -> ExecutorOutput round-trip."""
        messages = make_executor_with_tools(
            tool_sequence=[
                ("read_file", {"path": "docs/test.md"}, "# Test\nThis is a test doc."),
            ],
            final_text="The documentation is correct and complete.",
        )
        runner = MockQueryRunner(messages)

        executor = ExecutorAgent(
            id="e2e-executor",
            system_prompt="You validate docs.",
            user_prompt="Check docs/test.md for accuracy.",
            sdk_tools=["Read"],
            query_runner=runner,
        )
        wrapper = SdkExecutorWrapper(
            executor,
            system_prompt="You validate docs.",
            user_prompt="Check docs/test.md for accuracy.",
        )

        result = await wrapper.run(original_user_prompt="Check docs/test.md for accuracy.")

        assert isinstance(result, ExecutorOutput)
        assert result.success is True
        assert result.final_response == "The documentation is correct and complete."
        assert result.system_prompt == "You validate docs."
        assert result.user_prompt == "Check docs/test.md for accuracy."
        assert result.original_user_prompt == "Check docs/test.md for accuracy."
        assert result.input_tokens >= 0
        assert result.output_tokens >= 0

    async def test_executor_error_produces_compatible_output(self):
        """ExecutorAgent error -> SdkExecutorWrapper -> ExecutorOutput with error."""
        messages = [
            make_init(),
            make_assistant("An error occurred during execution."),
            make_result("An error occurred during execution.", is_error=True),
        ]
        runner = MockQueryRunner(messages)

        executor = ExecutorAgent(
            id="e2e-error-exec",
            system_prompt="sys",
            user_prompt="usr",
            sdk_tools=[],
            query_runner=runner,
        )
        wrapper = SdkExecutorWrapper(executor, system_prompt="sys", user_prompt="usr")

        result = await wrapper.run()

        assert isinstance(result, ExecutorOutput)
        assert result.success is False
        assert result.error is not None


class TestSdkJudgeToWrapperIntegration:
    async def test_judge_pass_produces_compatible_result(self):
        """JudgeAgent PASS -> SdkJudgeWrapper -> JudgeResult with passed=True."""
        runner = MockQueryRunner(
            make_judge_pass(confidence=0.92, reason="Docs are accurate and complete.")
        )

        judge = JudgeAgent(
            id="e2e-accuracy",
            system_prompt="Evaluate accuracy.",
            user_prompt="Is it correct?",
            sdk_tools=[],
            query_runner=runner,
        )
        wrapper = SdkJudgeWrapper(
            judge, assertion_text="accuracy", system_prompt="Evaluate accuracy."
        )

        executor_output = ExecutorOutput(
            tool_calls=[],
            final_response="The docs are correct.",
            success=True,
            system_prompt="sys",
            user_prompt="usr",
            input_tokens=100,
            output_tokens=50,
        )

        result = await wrapper.run(executor_output=executor_output)

        assert isinstance(result, JudgeResult)
        assert result.passed is True
        assert result.confidence == 0.92
        assert result.failure_reason is None
        assert result.judge_id == "e2e-accuracy"
        assert result.assertion_text == "accuracy"

    async def test_judge_fail_produces_compatible_result(self):
        """JudgeAgent FAIL -> SdkJudgeWrapper -> JudgeResult with passed=False."""
        runner = MockQueryRunner(
            make_judge_fail(reason="Missing API endpoint details.", confidence=0.75)
        )

        judge = JudgeAgent(
            id="e2e-completeness",
            system_prompt="Evaluate completeness.",
            user_prompt="Is it complete?",
            sdk_tools=[],
            query_runner=runner,
        )
        wrapper = SdkJudgeWrapper(
            judge, assertion_text="completeness", system_prompt="Evaluate completeness."
        )

        executor_output = ExecutorOutput(
            tool_calls=[],
            final_response="The docs list some endpoints.",
            success=True,
            system_prompt="sys",
            user_prompt="usr",
            input_tokens=100,
            output_tokens=50,
        )

        result = await wrapper.run(executor_output=executor_output)

        assert isinstance(result, JudgeResult)
        assert result.passed is False
        assert result.failure_reason == "Missing API endpoint details."
        assert result.confidence == 0.75


class TestFullExecutorJudgePipeline:
    async def test_executor_then_judge_pass(self):
        """Full pipeline: executor runs -> judge evaluates -> PASS."""
        exec_messages = make_executor_with_tools(
            tool_sequence=[
                ("read_file", {"path": "docs/api.md"}, "# API\nGET /users\nPOST /users"),
            ],
            final_text="The API documentation correctly lists GET /users and POST /users.",
        )
        exec_runner = MockQueryRunner(exec_messages)
        executor = ExecutorAgent(
            id="pipeline-exec",
            system_prompt="Read the API docs.",
            user_prompt="What endpoints exist?",
            sdk_tools=["Read"],
            query_runner=exec_runner,
        )
        exec_wrapper = SdkExecutorWrapper(
            executor,
            system_prompt="Read the API docs.",
            user_prompt="What endpoints exist?",
        )

        executor_output = await exec_wrapper.run(original_user_prompt="What endpoints exist?")
        assert executor_output.success is True

        judge_runner = MockQueryRunner(
            make_judge_pass(confidence=0.95, reason="All endpoints correctly listed.")
        )
        judge = JudgeAgent(
            id="pipeline-judge",
            system_prompt="Evaluate if endpoints are correct.",
            user_prompt="Are all endpoints listed?",
            sdk_tools=[],
            query_runner=judge_runner,
        )
        judge_wrapper = SdkJudgeWrapper(
            judge,
            assertion_text="endpoint-accuracy",
            system_prompt="Evaluate if endpoints are correct.",
        )

        judge_result = await judge_wrapper.run(executor_output=executor_output)

        assert isinstance(judge_result, JudgeResult)
        assert judge_result.passed is True
        assert judge_result.confidence == 0.95
        assert judge_result.assertion_text == "endpoint-accuracy"

    async def test_executor_then_judge_fail(self):
        """Full pipeline: executor runs -> judge evaluates -> FAIL."""
        exec_runner = MockQueryRunner(
            make_executor_simple("The docs mention GET /users only.")
        )
        executor = ExecutorAgent(
            id="pipeline-exec-fail",
            system_prompt="Read the docs.",
            user_prompt="List all endpoints.",
            sdk_tools=[],
            query_runner=exec_runner,
        )
        exec_wrapper = SdkExecutorWrapper(
            executor,
            system_prompt="Read the docs.",
            user_prompt="List all endpoints.",
        )

        executor_output = await exec_wrapper.run()
        assert executor_output.success is True

        judge_runner = MockQueryRunner(
            make_judge_fail(
                reason="Missing POST /users endpoint.",
                confidence=0.85,
            )
        )
        judge = JudgeAgent(
            id="pipeline-judge-fail",
            system_prompt="Evaluate completeness.",
            user_prompt="Are all endpoints listed?",
            sdk_tools=[],
            query_runner=judge_runner,
        )
        judge_wrapper = SdkJudgeWrapper(
            judge,
            assertion_text="completeness",
            system_prompt="Evaluate completeness.",
        )

        judge_result = await judge_wrapper.run(executor_output=executor_output)

        assert judge_result.passed is False
        assert judge_result.failure_reason == "Missing POST /users endpoint."


class TestLoaderSdkIntegration:
    def test_load_scaffold_creates_sdk_agents(self, tmp_path):
        """Loader creates SDK wrapper agents."""
        from dokumen.loader import load_scaffold

        _setup_project(tmp_path)
        scaffold_path = _write_scaffold(tmp_path)

        test_obj = load_scaffold(
            str(scaffold_path),
            project_root=str(tmp_path),
        )

        # Executor is a wrapper
        assert test_obj.executor.__class__.__name__ == "SdkExecutorWrapper"
        assert test_obj.executor.agent_type == AgentType.EXECUTOR
        assert test_obj.executor.system_prompt != ""
        assert test_obj.executor.user_prompt == "Check the docs for accuracy."

        # Judge is a wrapper
        assert len(test_obj.judges) == 1
        judge = test_obj.judges[0]
        assert judge.__class__.__name__ == "SdkJudgeWrapper"
        assert judge.agent_type == AgentType.JUDGE
        assert judge.include_executor_output is True

    def test_load_scaffold_produces_correct_test_id(self, tmp_path):
        """Loader produces correct test_id from scaffold name."""
        from dokumen.loader import load_scaffold

        _setup_project(tmp_path)
        scaffold_path = _write_scaffold(tmp_path)

        test_obj = load_scaffold(
            str(scaffold_path),
            project_root=str(tmp_path),
        )

        assert test_obj.id == "integration-test"


class TestFullPipelineWithToolCalls:
    async def test_judge_sees_executor_tool_calls_and_prompts(self):
        """End-to-end: executor tool calls and prompts appear in judge context."""
        # 1. Build executor with tool calls
        exec_messages = make_executor_with_tools(
            tool_sequence=[
                ("read_file", {"path": "docs/api.md"}, "# API Reference\nGET /users"),
                ("glob", {"pattern": "*.md"}, "docs/api.md\ndocs/guide.md"),
            ],
            final_text="Found 2 documentation files with API endpoints.",
        )
        exec_runner = MockQueryRunner(exec_messages)
        executor = ExecutorAgent(
            id="e2e-tc-executor",
            system_prompt="You are a doc validator.",
            user_prompt="Read all markdown files and summarize.",
            sdk_tools=["Read", "Glob"],
            query_runner=exec_runner,
        )
        exec_wrapper = SdkExecutorWrapper(
            executor,
            system_prompt="You are a doc validator.",
            user_prompt="Read all markdown files and summarize.",
        )

        # 2. Run executor, verify tool_calls have results (Bug 1 fix)
        exec_result = await exec_wrapper.run()
        assert len(exec_result.tool_calls) == 2
        assert exec_result.tool_calls[0]["tool_result"] is not None
        assert exec_result.tool_calls[1]["tool_result"] is not None

        # 3. Verify conversation_log has "tool" role entries (Bug 1 fix)
        tool_entries = [e for e in exec_result.conversation_log if e.get("role") == "tool"]
        assert len(tool_entries) == 2

        # 4. Run judge with executor output + prompts
        judge_runner = MockQueryRunner(make_judge_pass(confidence=0.9, reason="All correct"))
        judge = JudgeAgent(
            id="e2e-tc-accuracy",
            system_prompt="Evaluate completeness.",
            user_prompt="Check if all docs were read.",
            include_executor_output=True,
            sdk_tools=[],
            query_runner=judge_runner,
        )
        judge_wrapper = SdkJudgeWrapper(
            judge,
            assertion_text="accuracy",
            system_prompt="Evaluate completeness.",
        )

        judge_result = await judge_wrapper.run(
            executor_output=exec_result,
            executor_system_prompt="You are a doc validator.",
            executor_user_prompt="Read all markdown files and summarize.",
        )

        assert judge_result.passed is True

        # 5. Verify judge context contains all sections
        sent_prompt = judge_runner.calls[0].prompt
        assert "## Executor Task" in sent_prompt
        assert "You are a doc validator." in sent_prompt
        assert "## Executor Output" in sent_prompt
        assert "Found 2 documentation files" in sent_prompt
        assert "## Executor Tool Calls" in sent_prompt
        assert "read_file" in sent_prompt
        assert "glob" in sent_prompt
        assert "## Evaluation Criteria" in sent_prompt
