"""
Kernel Browser Rollout Processor for Eval Protocol.

Runs multi-step browser episodes using Kernel browser pools.
This enables evaluating VLM agents on computer use tasks within the EP framework.

Usage:
    from kernel_browser_rollout_processor import KernelBrowserRolloutProcessor
    
    @evaluation_test(
        rollout_processor=KernelBrowserRolloutProcessor(pool_name="eval-pool"),
        ...
    )
    def test_my_eval(row: EvaluationRow) -> EvaluationRow:
        ...
"""

import asyncio
import base64
import io
import logging
import os

from PIL import Image

from eval_protocol.models import EvaluationRow, Message, Status
from eval_protocol.pytest.rollout_processor import RolloutProcessor
from eval_protocol.pytest.types import RolloutProcessorConfig

# From kernel-tinker-rl (vendored core package)
from kernel import Kernel
from core.agent import AgentConfig, QwenAgent
from core.agent_loop import run_agent_loop
from core.browser import KernelBrowserAdapter

logger = logging.getLogger(__name__)


def encode_screenshots(images: list[Image.Image]) -> list[str]:
    """Encode PIL Images to base64 strings for JSON storage."""
    encoded = []
    for img in images:
        buffer = io.BytesIO()
        img.save(buffer, format="PNG")
        encoded.append(base64.b64encode(buffer.getvalue()).decode("utf-8"))
    return encoded


def decode_screenshots(encoded: list[str]) -> list[Image.Image]:
    """Decode base64 strings back to PIL Images."""
    images = []
    for b64 in encoded:
        buffer = io.BytesIO(base64.b64decode(b64))
        images.append(Image.open(buffer))
    return images


class KernelBrowserRolloutProcessor(RolloutProcessor):
    """
    Rollout processor that runs multi-step browser episodes using Kernel.

    Each rollout:
    1. Acquires browser from Kernel pool
    2. Navigates to initial URL
    3. Runs VLM agent loop (screenshot → predict → execute → repeat)
    4. Captures trajectory (screenshots, action_history)
    5. Releases browser
    6. Returns EvaluationRow with trajectory data in execution_metadata.extra

    The evaluation function can then use WebJudge to score the trajectory.

    Args:
        pool_name: Name of the Kernel browser pool to use
        max_steps: Maximum number of agent steps per episode
        image_max_size: Max dimension for screenshot resizing
        acquire_timeout_seconds: Timeout for acquiring browser from pool
        system_prompt: Custom system prompt (defaults to agent_auth prompt)
        extra_actions: Additional action types to support
        base_url: API base URL for model inference
    """

    def __init__(
        self,
        pool_name: str = "eval-browser-pool",
        max_steps: int = 10,
        image_max_size: int = 512,
        acquire_timeout_seconds: int = 60,
        system_prompt: str | None = None,
        extra_actions: list | None = None,
        base_url: str = "https://api.fireworks.ai/inference/v1",
    ):
        self.pool_name = pool_name
        self.max_steps = max_steps
        self.image_max_size = image_max_size
        self.acquire_timeout_seconds = acquire_timeout_seconds
        self.system_prompt = system_prompt
        self.extra_actions = extra_actions or []
        self.base_url = base_url
        self.api_key = os.environ.get("FIREWORKS_API_KEY")
        self._kernel: Kernel | None = None

    def setup(self) -> None:
        """Initialize Kernel client."""
        self._kernel = Kernel()
        logger.info(f"Kernel client initialized, using pool: {self.pool_name}")

    def __call__(
        self,
        rows: list[EvaluationRow],
        config: RolloutProcessorConfig,
    ) -> list[asyncio.Task[EvaluationRow]]:
        """Process evaluation rows and return async tasks."""
        semaphore = config.semaphore

        async def _sem_wrapper(row: EvaluationRow) -> EvaluationRow:
            async with semaphore:
                return await self._process_row(row, config)

        return [asyncio.create_task(_sem_wrapper(row)) for row in rows]

    async def _process_row(
        self,
        row: EvaluationRow,
        config: RolloutProcessorConfig,
    ) -> EvaluationRow:
        """
        Run a single browser episode and return trajectory.

        The trajectory data is stored in row.execution_metadata.extra for
        the evaluation function to use with WebJudge.
        """
        if self._kernel is None:
            self._kernel = Kernel()

        # Extract task info from row
        dataset_info = row.input_metadata.dataset_info or {}
        initial_url = dataset_info.get("initial_url", "")
        task = dataset_info.get("task", "")
        task_id = row.input_metadata.row_id or "unknown"

        # Get model from completion_params
        completion_params = config.completion_params or {}
        model = completion_params.get("model", "qwen/qwen3-vl-8b-instruct")
        temperature = completion_params.get("temperature", 0.0)
        max_tokens = completion_params.get("max_tokens", 512)

        # Episode state
        adapter: KernelBrowserAdapter | None = None
        error: str | None = None
        final_url: str | None = None

        try:
            # Acquire browser from pool
            browser = self._kernel.browser_pools.acquire(
                self.pool_name,
                acquire_timeout_seconds=self.acquire_timeout_seconds,
            )
            adapter = KernelBrowserAdapter(self._kernel, browser, reset_on_init=True)

            # Start heartbeat to keep browser alive during VLM inference
            adapter.start_heartbeat_sync(task_label=f"{task_id}")

            # Navigate to initial URL
            initial_screenshot = adapter.navigate(initial_url)

            # Create agent
            agent_config = AgentConfig(
                model=model,
                base_url=self.base_url,
                api_key=self.api_key,
                system_prompt=self.system_prompt,
                extra_actions=self.extra_actions,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            agent = QwenAgent(config=agent_config)

            # Run the multi-turn agent loop
            # This runs in a thread pool since it has blocking VLM calls
            loop_result = await asyncio.to_thread(
                run_agent_loop,
                agent=agent,
                adapter=adapter,
                task=task,
                initial_screenshot=initial_screenshot,
                max_steps=self.max_steps,
                image_max_size=self.image_max_size,
            )

            # Get final URL
            final_url = adapter.get_current_url()

            # Get preserved message history from agent (includes tool_calls)
            agent_messages = agent.get_messages()
            row.messages = [
                Message(
                    role=m["role"],
                    content=m.get("content"),
                    tool_calls=m.get("tool_calls"),
                )
                for m in agent_messages
            ]

            # Store trajectory data in extra (JSON-serializable)
            if row.execution_metadata.extra is None:
                row.execution_metadata.extra = {}

            row.execution_metadata.extra.update(
                {
                    # For WebJudge evaluation (base64 encoded for JSON serialization)
                    "screenshots_b64": encode_screenshots(loop_result.screenshots),
                    "action_history": loop_result.action_history,
                    # Model responses (raw LLM outputs)
                    "model_responses": agent.state.responses,
                    # Task info (needed for Trajectory construction)
                    "task": task,
                    "task_id": task_id,
                    "initial_url": initial_url,
                    "final_url": final_url,
                    # Episode metadata
                    "termination_reason": loop_result.termination_reason,
                    "terminal_action": loop_result.terminal_action,
                    "steps_completed": loop_result.steps_completed,
                    "error": loop_result.error,
                    # Step-by-step details (for debugging)
                    "step_results": [
                        {
                            "step": sr.step,
                            "action_desc": sr.action_desc,
                            "predict_time": sr.predict_time,
                            "exec_time": sr.exec_time,
                            "total_time": sr.total_time,
                            "is_terminal": sr.is_terminal,
                            "error": sr.error,
                        }
                        for sr in loop_result.step_results
                    ],
                }
            )

            # Set rollout status
            if loop_result.error:
                row.rollout_status = Status.error(loop_result.error)
            else:
                row.rollout_status = Status.rollout_finished()

        except Exception as e:
            error = str(e)
            logger.error(f"Rollout failed for {task_id}: {error}")

            if row.execution_metadata.extra is None:
                row.execution_metadata.extra = {}

            row.execution_metadata.extra.update(
                {
                    "task": task,
                    "task_id": task_id,
                    "initial_url": initial_url,
                    "error": error,
                    "screenshots_b64": [],
                    "action_history": [],
                    "model_responses": [],
                    "termination_reason": "error",
                    "terminal_action": None,
                    "steps_completed": 0,
                }
            )
            row.rollout_status = Status.error(error)

        finally:
            # Release browser back to pool
            if adapter is not None:
                try:
                    adapter.stop_heartbeat_sync()
                    reuse = error is None and not adapter._should_not_reuse
                    self._kernel.browser_pools.release(
                        self.pool_name,
                        session_id=adapter.session_id,
                        reuse=reuse,
                    )
                except Exception as e:
                    logger.warning(f"Failed to release browser: {e}")

        return row

