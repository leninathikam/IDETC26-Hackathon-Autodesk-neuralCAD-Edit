"""Independent visual / instruction critic."""

from __future__ import annotations

from typing import Any, Optional

from groundedcad.agents.schemas import Critique, EditSpec, ExecutionResult, GroundedIntent
from groundedcad.llm.base import LLMClient
from groundedcad.verify.agent import llm_verification, run_verification_agent


def deterministic_critique(
    *,
    before: dict[str, Any],
    execution: ExecutionResult,
    edit_spec: Optional[EditSpec] = None,
    intent: Optional[GroundedIntent] = None,
) -> Critique:
    return run_verification_agent(
        before=before,
        execution=execution,
        edit_spec=edit_spec,
        intent=intent,
        use_llm=False,
    )


def llm_critique(
    client: LLMClient,
    *,
    instruction: str,
    intent: GroundedIntent,
    edit_spec: EditSpec,
    execution: ExecutionResult,
    base_critique: Critique,
    before: Optional[dict[str, Any]] = None,
) -> Critique:
    decision = base_critique.verification
    if decision is None:
        return run_verification_agent(
            before=before or {},
            execution=execution,
            edit_spec=edit_spec,
            intent=intent,
            instruction=instruction,
            client=client,
            use_llm=True,
        )
    return llm_verification(
        client,
        instruction=instruction,
        intent=intent,
        edit_spec=edit_spec,
        execution=execution,
        before=before or execution.geometry_summary or {},
        base=base_critique,
        decision=decision,
    )
