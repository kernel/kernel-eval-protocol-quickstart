"""
Agent Auth Evaluation Test for Eval Protocol.

Evaluates VLM agents on login/register form discovery tasks using:
- Kernel browser pools for browser execution
- WebJudge for trajectory scoring

Usage:
    # Run the evaluation
    pytest test_agent_auth.py -vs

    # With custom settings via environment
    EP_MAX_ROWS=5 pytest test_agent_auth.py -vs

Environment Variables:
    KERNEL_API_KEY: Required for Kernel browser API
    FIREWORKS_API_KEY: Required for VLM rollout inference
    OPENAI_API_KEY: Required for WebJudge (LLM judge)
"""

import os
from pathlib import Path
from typing import Any, Dict, List

from dotenv import load_dotenv

from eval_protocol.models import EvaluateResult, EvaluationRow, InputMetadata
from eval_protocol.pytest import evaluation_test

from kernel_browser_rollout_processor import (
    KernelBrowserRolloutProcessor,
    decode_screenshots,
)

from core.reward_models.webjudge import Trajectory, WebJudge
from agent_auth.actions import AGENT_AUTH_ACTIONS
from agent_auth.config import (
    AGENT_AUTH_EVALUATION_CRITERIA,
    get_agent_auth_system_prompt,
)

# Load environment variables
load_dotenv()

# WebJudge singleton
_webjudge: WebJudge | None = None
MAX_DATASET_ROWS = int(os.getenv("EP_MAX_ROWS", "4"))


def get_webjudge(model: str = "gpt-5-mini") -> WebJudge:
    """Get or create WebJudge instance using OpenAI directly."""
    global _webjudge
    if _webjudge is None:
        _webjudge = WebJudge(
            model=model,
            api_key=os.environ.get("OPENAI_API_KEY"),
            base_url="https://api.openai.com/v1",
            evaluation_criteria=AGENT_AUTH_EVALUATION_CRITERIA,
        )
    return _webjudge


def agent_auth_dataset_adapter(rows: List[Dict[str, Any]]) -> List[EvaluationRow]:
    """
    Convert tasks.jsonl rows to EvaluationRow objects.

    Expected input format:
        {"id": "...", "initial_url": "https://...", "task": "Navigate to..."}
    """
    converted: List[EvaluationRow] = []

    for r in rows:
        converted.append(
            EvaluationRow(
                messages=[],  # Will be populated during rollout
                input_metadata=InputMetadata(
                    row_id=r.get("id"),
                    dataset_info={
                        "initial_url": r.get("initial_url"),
                        "task": r.get("task"),
                    },
                ),
            )
        )

    return converted



@evaluation_test(
    input_dataset=[str(Path(__file__).parent / "tasks.jsonl")],
    dataset_adapter=agent_auth_dataset_adapter,
    completion_params=[
        {
            "model": "accounts/fireworks/models/qwen3-vl-30b-a3b-thinking",
        },
    ],
    rollout_processor=KernelBrowserRolloutProcessor(
        pool_name="eval-browser-pool",
        max_steps=10,
        image_max_size=512,
        system_prompt=get_agent_auth_system_prompt(),
        extra_actions=AGENT_AUTH_ACTIONS,
    ),
    passed_threshold=0.5,
    max_dataset_rows=MAX_DATASET_ROWS,
    num_runs=1,
    max_concurrent_rollouts=16,
    max_concurrent_evaluations=16,
    mode="pointwise",
)
async def test_agent_auth_evaluation(row: EvaluationRow) -> EvaluationRow:
    """
    Evaluate agent auth task using WebJudge.

    The KernelBrowserRolloutProcessor has already run the browser episode.
    This function evaluates the trajectory with WebJudge.

    Scoring:
    - 1.0: Agent successfully navigated to auth page and identified input fields
    - 0.0: Agent failed to complete the task
    """
    extra = row.execution_metadata.extra or {}

    # Check for rollout errors
    if extra.get("error"):
        row.evaluation_result = EvaluateResult(
            score=0.0,
            reason=f"Rollout error: {extra['error']}",
            is_score_valid=False,
        )
        return row

    # Get trajectory data
    screenshots_b64 = extra.get("screenshots_b64", [])
    
    row.execution_metadata.extra["screenshots_b64"] = []  # TODO: reduce size, preserved in row.messages # pyright: ignore[reportOptionalSubscript]

    action_history = extra.get("action_history", [])
    task = extra.get("task", "")
    task_id = extra.get("task_id", "unknown")
    initial_url = extra.get("initial_url")
    final_url = extra.get("final_url")

    # Check for empty trajectory
    if not screenshots_b64:
        row.evaluation_result = EvaluateResult(
            score=0.0,
            reason="No screenshots captured",
            is_score_valid=False,
        )
        return row

    # Decode screenshots from base64
    screenshots = decode_screenshots(screenshots_b64)

    # Build trajectory for WebJudge
    trajectory = Trajectory(
        task_id=task_id,
        task=task,
        action_history=action_history,
        screenshots=screenshots,
        initial_url=initial_url,
        final_url=final_url,
        metadata={
            "termination_reason": extra.get("termination_reason"),
            "terminal_action": extra.get("terminal_action"),
            "steps_completed": extra.get("steps_completed"),
        },
    )

    # Evaluate with WebJudge
    try:
        webjudge = get_webjudge()
        result = await webjudge.evaluate(trajectory)

        row.evaluation_result = EvaluateResult(
            score=result.score,
            reason=result.reasoning if result.reasoning else "",
            is_score_valid=True,
            trajectory_info=extra,
        )

    except Exception as e:
        row.evaluation_result = EvaluateResult(
            score=0.0,
            reason=f"WebJudge error: {str(e)}",
            is_score_valid=False,
            trajectory_info=extra,
        )

    return row
