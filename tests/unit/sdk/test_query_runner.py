"""Tests for dokumen.sdk.query_runner — MockQueryRunner and MockCall."""

import pytest

from claude_agent_sdk import ClaudeAgentOptions

from dokumen.sdk.query_runner import MockCall, MockQueryRunner, QueryRunner, SDKQueryRunner
from dokumen.sdk.testing import make_assistant, make_init, make_result


class TestMockQueryRunnerProtocol:
    def test_mock_runner_is_not_protocol_instance(self):
        """MockQueryRunner satisfies the QueryRunner protocol structurally."""
        # Protocol check: MockQueryRunner has async run() method
        runner = MockQueryRunner([])
        assert hasattr(runner, "run")
        assert callable(runner.run)

    def test_sdk_runner_has_run_method(self):
        """SDKQueryRunner has an async run method."""
        runner = SDKQueryRunner()
        assert hasattr(runner, "run")
        assert callable(runner.run)


class TestMockQueryRunnerInit:
    def test_empty_messages(self):
        """Runner initialized with empty message list."""
        runner = MockQueryRunner([])
        assert runner._messages == []
        assert runner.calls == []

    def test_messages_stored(self):
        """Runner stores provided messages."""
        msgs = [make_init(), make_assistant("hi"), make_result("hi")]
        runner = MockQueryRunner(msgs)
        assert len(runner._messages) == 3


class TestMockQueryRunnerRun:
    async def test_yields_all_messages(self):
        """run() yields all pre-configured messages."""
        msgs = [make_init(), make_assistant("hello"), make_result("hello")]
        runner = MockQueryRunner(msgs)

        collected = []
        async for msg in runner.run("test prompt"):
            collected.append(msg)

        assert len(collected) == 3
        assert collected[0] is msgs[0]
        assert collected[1] is msgs[1]
        assert collected[2] is msgs[2]

    async def test_records_call(self):
        """run() records the call with prompt and options."""
        runner = MockQueryRunner([make_init(), make_result("ok")])
        opts = ClaudeAgentOptions(system_prompt="sys")

        async for _ in runner.run("my prompt", opts):
            pass

        assert len(runner.calls) == 1
        assert runner.calls[0].prompt == "my prompt"
        assert runner.calls[0].options is opts

    async def test_records_multiple_calls(self):
        """Multiple run() calls are all recorded."""
        runner = MockQueryRunner([make_result("ok")])

        async for _ in runner.run("first"):
            pass
        async for _ in runner.run("second"):
            pass

        assert len(runner.calls) == 2
        assert runner.calls[0].prompt == "first"
        assert runner.calls[1].prompt == "second"

    async def test_none_options_recorded(self):
        """run() with no options records None."""
        runner = MockQueryRunner([make_result("ok")])

        async for _ in runner.run("prompt"):
            pass

        assert runner.calls[0].options is None

    async def test_empty_messages_yields_nothing(self):
        """run() with empty messages yields nothing."""
        runner = MockQueryRunner([])

        collected = []
        async for msg in runner.run("prompt"):
            collected.append(msg)

        assert collected == []
        assert len(runner.calls) == 1


class TestMockCall:
    def test_mock_call_fields(self):
        """MockCall stores prompt and options."""
        opts = ClaudeAgentOptions(system_prompt="test")
        call = MockCall(prompt="hello", options=opts)

        assert call.prompt == "hello"
        assert call.options is opts

    def test_mock_call_none_options(self):
        """MockCall allows None options."""
        call = MockCall(prompt="hello", options=None)
        assert call.options is None


class TestSDKQueryRunnerTyping:
    async def test_mock_runner_accepts_async_iterable_prompt(self):
        msgs = [make_result("ok")]
        runner = MockQueryRunner(msgs)

        async def prompt_gen():
            yield {"type": "user", "message": {"role": "user", "content": "hello"}}

        collected = []
        async for msg in runner.run(prompt_gen()):
            collected.append(msg)

        assert len(collected) == 1
        assert runner.calls[0].prompt is not None
