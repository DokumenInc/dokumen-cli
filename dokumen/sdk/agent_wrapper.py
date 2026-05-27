"""
SDK Agent Wrappers — adapters for test_object.py.

These wrappers implement the interface that test_object.py expects,
delegating to the SDK executor/judge agents internally. Since the SDK
agents now produce canonical ExecutorResult/JudgeVerdict types directly,
these wrappers simply pass through the results with minimal adjustment
(adding system_prompt/user_prompt/original_user_prompt to the executor
result, and assertion_text to the judge result).
"""

import logging
from typing import Any, Dict, List, Optional

from ..agent_object import AgentType
from .executor import ExecutorAgent
from .judge import JudgeAgent
from .types import ExecutorResult, JudgeVerdict

logger = logging.getLogger(__name__)


class _SdkProviderStub:
    """Minimal stub satisfying test_suite/test_object provider attribute access.

    test_suite.py and test_object.py access executor.provider.model for logging.
    This stub provides that without needing a real LLM provider.
    """

    def __init__(self, model: Optional[str] = None):
        self.model = model or "sdk"


class _SdkToolStub:
    """Minimal stub satisfying test_suite/test_object tool iteration.

    test_suite.py iterates executor.tools for .name (logging/provenance).
    """

    def __init__(self, name: str):
        self.name = name


class SdkExecutorWrapper:
    """Wraps ExecutorAgent to match the interface test_object.py expects.

    test_object.py calls:
        result = await executor.run(
            timeout=..., on_tool_call=...,
            on_conversation_message=..., original_user_prompt=...
        )
    and expects ExecutorResult back.
    """

    def __init__(self, executor: ExecutorAgent, system_prompt: str, user_prompt: str):
        self.id = executor.id
        self.agent_type = AgentType.EXECUTOR
        self.system_prompt = system_prompt
        self.user_prompt = user_prompt
        self._executor = executor
        # Expose provider stub for test_suite.py model logging
        model = getattr(executor._options, 'model', None)
        self.provider = _SdkProviderStub(model)
        # Expose tools as stubs for provenance tracking
        sdk_tools = getattr(executor._options, 'allowed_tools', []) or []
        self.tools = [_SdkToolStub(name) for name in sdk_tools]
        # test_object.py accesses max_tool_result_chars for explore agent
        self.max_tool_result_chars = 50000

    async def run(
        self,
        timeout: Optional[float] = None,
        on_tool_call: Optional[Any] = None,
        on_conversation_message: Optional[Any] = None,
        original_user_prompt: str = "",
        **kwargs,
    ) -> ExecutorResult:
        """Run the SDK executor and return an ExecutorResult."""
        logger.info(
            "SdkExecutorWrapper.run starting",
            extra={
                "agent_id": self.id,
                "timeout_override": timeout,
            },
        )

        # Override timeout if provided
        if timeout is not None:
            self._executor.timeout = timeout

        result = await self._executor.run()

        # Attach prompt metadata (executor agent doesn't know about prompts)
        result.system_prompt = self.system_prompt
        result.user_prompt = self.user_prompt
        result.original_user_prompt = original_user_prompt or self.user_prompt

        return result


class SdkJudgeWrapper:
    """Wraps JudgeAgent to match the interface test_object.py expects.

    test_object.py calls:
        result = await judge.run(executor_output=executor_result)
    and expects JudgeVerdict back.
    """

    def __init__(
        self,
        judge: JudgeAgent,
        assertion_text: str = "",
        system_prompt: str = "",
    ):
        self.id = judge.id
        self.agent_type = AgentType.JUDGE
        self.system_prompt = system_prompt
        self._judge = judge
        self._assertion_text = assertion_text
        # Expose provider stub for test_object.py model logging
        model = getattr(judge._options, 'model', None)
        self.provider = _SdkProviderStub(model)
        # Expose for compatibility
        self.tools = []
        self.include_executor_output = judge.include_executor_output
        self.timeout = judge.timeout

    async def run(
        self,
        executor_output: Optional[ExecutorResult] = None,
        timeout: Optional[float] = None,
        on_tool_call: Optional[Any] = None,
        on_conversation_message: Optional[Any] = None,
        executor_system_prompt: str = "",
        executor_user_prompt: str = "",
        **kwargs,
    ) -> JudgeVerdict:
        """Run the SDK judge and return a JudgeVerdict."""
        logger.info(
            "SdkJudgeWrapper.run starting",
            extra={
                "judge_id": self.id,
                "has_executor_output": executor_output is not None,
                "has_executor_prompts": bool(executor_system_prompt or executor_user_prompt),
            },
        )

        # Override timeout if provided
        if timeout is not None:
            self._judge.timeout = timeout

        # Pass the executor result directly to the judge
        # (judge.run expects ExecutorResult)
        if executor_output is None:
            executor_output = ExecutorResult(
                success=False,
                final_response="",
            )

        result = await self._judge.run(
            executor_output,
            executor_system_prompt=executor_system_prompt,
            executor_user_prompt=executor_user_prompt,
        )

        # Attach assertion text metadata
        result.assertion_text = self._assertion_text

        return result

    def _get_assertion_text(self) -> str:
        """Return the assertion text for this judge (used in error fallback)."""
        return self._assertion_text
