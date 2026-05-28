"""Shared prompt preparation helpers for executor-like stages."""

import hashlib
from typing import Optional

from ..pipeline import PipelineContext


def prompt_hash(prompt: Optional[str]) -> str:
    """Compute a short SHA256 hash of a prompt for observability logging."""
    if prompt is None or not isinstance(prompt, str):
        return "none"
    return hashlib.sha256(prompt.encode()).hexdigest()[:12]


def prepare_agent_prompts(ctx: PipelineContext, executor_role: str, logger) -> None:
    """Apply shared executor/judge prompt mutations and prompt observability logs.

    ExecutorStage and CoordinatorStage both produce executor output that judges
    consume. Keeping prompt injection here prevents those two paths from
    drifting when output-folder or prompt-hash behavior changes.
    """
    if not ctx.original_user_prompt:
        ctx.original_user_prompt = ctx.executor.user_prompt

    output_folder_instruction = (
        f"\n\n---\nOUTPUT FOLDER: {ctx.output_dir}\n"
        "Write any deliverables, calculations, scripts, or evidence "
        "files to this folder.\n"
        "Files here will be visible to the user after the test completes."
    )
    ctx.executor.user_prompt = ctx.executor.user_prompt + output_folder_instruction

    judge_output_instruction = (
        f"\n\nOUTPUT FOLDER: {ctx.output_dir}\n"
        "You may write analysis or evidence files here. "
        "Files will be visible to the user."
    )
    for judge in ctx.judges:
        ctx.original_judge_prompts[judge.id] = judge.system_prompt
        judge.system_prompt = judge.system_prompt + judge_output_instruction

    exec_sys = ctx.executor.system_prompt if isinstance(ctx.executor.system_prompt, str) else None
    exec_usr = ctx.executor.user_prompt if isinstance(ctx.executor.user_prompt, str) else None
    logger.info(
        "agent.prompt_applied",
        test_id=ctx.test_id,
        role=executor_role,
        system_prompt_hash=prompt_hash(exec_sys),
        user_prompt_hash=prompt_hash(exec_usr),
        system_prompt_len=len(exec_sys) if exec_sys else 0,
        user_prompt_len=len(exec_usr) if exec_usr else 0,
    )
    for judge in ctx.judges:
        judge_sys = judge.system_prompt if isinstance(judge.system_prompt, str) else None
        logger.info(
            "agent.prompt_applied",
            test_id=ctx.test_id,
            role="judge",
            judge_name=judge.id,
            system_prompt_hash=prompt_hash(judge_sys),
            system_prompt_len=len(judge_sys) if judge_sys else 0,
        )


def ensure_final_response_from_conversation(executor_output, logger, test_id: str) -> bool:
    """Fill a missing final response from assistant text in the conversation log."""
    if not executor_output or getattr(executor_output, "final_response", None):
        return False

    conversation_log = getattr(executor_output, "conversation_log", None) or []
    text_chunks = []
    for item in conversation_log:
        if not isinstance(item, dict):
            continue
        if item.get("role") != "assistant":
            continue
        content = item.get("content")
        if isinstance(content, str) and content.strip():
            text_chunks.append(content.strip())

    if not text_chunks:
        return False

    executor_output.final_response = "\n\n".join(text_chunks)
    logger.warning(
        "stage.executor.reconstructed_response",
        test_id=test_id,
        chunk_count=len(text_chunks),
    )
    return True
