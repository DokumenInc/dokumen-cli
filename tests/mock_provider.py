"""
Test-only MockProvider for tests that still need the legacy Provider interface.

This replaces the production MockProvider that was removed as part of
the legacy AgentObject/Provider path cleanup (issue #599).
Used by explore agent tests and other tests that need a mock LLM provider.
"""
from typing import Any, Dict, List, Optional

from dokumen.agent_object import Provider


class MockProvider(Provider):
    """Mock LLM provider for testing. Test-only, not shipped in the package."""

    def __init__(self, responses: List[Dict[str, Any]] = None, model: str = "test-model"):
        self.responses = responses or []
        self.calls: List[Dict[str, Any]] = []
        self.call_index = 0
        self.model = model

    async def complete(
        self,
        messages: List[Dict[str, str]],
        tools: Optional[List[Dict]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        self.calls.append({
            "messages": messages,
            "tools": tools,
            "kwargs": kwargs
        })
        if self.responses and self.call_index < len(self.responses):
            response = self.responses[self.call_index]
            self.call_index += 1
            return response
        return {"content": "Mock response", "tool_calls": []}

    def get_call_count(self) -> int:
        return len(self.calls)

    def get_last_call(self) -> Optional[Dict[str, Any]]:
        return self.calls[-1] if self.calls else None

    @staticmethod
    def create_judge_pass_response(confidence: float = 0.95) -> Dict[str, Any]:
        return {
            "content": f'{{"verdict": "PASS", "confidence": {confidence}, "reason": "All criteria met"}}'
        }

    @staticmethod
    def create_judge_fail_response(reason: str, confidence: float = 0.8) -> Dict[str, Any]:
        return {
            "content": f'{{"verdict": "FAIL", "confidence": {confidence}, "reason": "{reason}"}}'
        }
