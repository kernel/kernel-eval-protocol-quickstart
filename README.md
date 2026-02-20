# Kernel Eval Protocol

[Eval Protocol](https://github.com/eval-protocol) (EP) is an open solution for doing reinforcement learning fine-tuning on existing agents — across any language, container, or framework. This quickstart uses it to evaluate and fine-tune VLM browser agents using [Kernel](https://onkernel.com) serverless browsers and [Fireworks](https://fireworks.ai) for VLM inference.

## Quickstart

Requires Python 3.10+.

1. **Clone and install**

   ```bash
   git clone https://github.com/kernel/kernel-eval-protocol-quickstart.git
   cd kernel-eval-protocol-quickstart
   python -m venv .venv
   source .venv/bin/activate
   uv pip install -r requirements.txt
   ```

2. **Set API keys**

   Copy `.env.example` to `.env` and fill in the three keys: **Kernel** (serverless browser), **Fireworks** (VLM inference), **OpenAI** (WebJudge scoring).

   ```bash
   cp .env.example .env
   # Edit .env with your keys
   ```

3. **Create a browser pool**

   Browsers must stay alive during VLM inference, so use a long inactivity timeout. The default concurrency uses up to 16 rollouts, so a pool of 20 is a good fit.

   ```bash
   kernel pools create eval-browser-pool --size 20 --timeout 900
   ```

4. **Start the local monitoring server**

   In a separate terminal, start the Eval Protocol UI so you can monitor runs in real-time:

   ```bash
   source .venv/bin/activate
   .venv/bin/ep logs
   ```

   Keep this running -- when you kick off pytest in the next step, open `http://localhost:8000` to watch progress, view live results, and explore the pivot/table views that pytest prints to the console.

5. **Run the evaluation**

   ```bash
   pytest test_agent_auth.py -vs
   ```

   By default, the test runs `4` rows. Override with `EP_MAX_ROWS`:
   - Fewer rows: `EP_MAX_ROWS=3 pytest test_agent_auth.py -vs`
   - More rows: `EP_MAX_ROWS=20 pytest test_agent_auth.py -vs`

## What Happens When You Run It

The included dataset (`tasks.jsonl`) contains 469 **Agent Auth** tasks. Each Agent Auth task asks the agent to navigate to a website, find its login or registration page, and identify the required input fields -- without typing credentials or submitting any forms. For each task:

- **KernelBrowserRolloutProcessor** acquires a browser from the pool, navigates to the task URL, runs the VLM agent loop (screenshot → predict → execute → repeat), captures the trajectory, then releases the browser.
- The test function scores each trajectory with **WebJudge** (LLM-as-judge) against the evaluation rubric in `agent_auth/config.py`.
- Results are reported by pytest / Eval Protocol.

```
┌─────────────────────────────────────────────────────────────────┐
│                     Eval Protocol                               │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  @evaluation_test(...)                                   │   │
│  │  async def test_agent_auth(row):                         │   │
│  │      trajectory = get_trajectory(row)                    │   │
│  │      score = webjudge.evaluate(trajectory)               │   │
│  └──────────────────────────────────────────────────────────┘   │
│                            │                                    │
│                            ▼                                    │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  KernelBrowserRolloutProcessor                           │   │
│  │    1. Acquire browser from Kernel pool                   │   │
│  │    2. Navigate to initial URL                            │   │
│  │    3. Run agent loop (screenshot → predict → execute)    │   │
│  │    4. Capture trajectory, release browser                │   │
│  └──────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
                             │
                             ▼
              ┌─────────────────────────────┐
              │      Kernel Browser Pool    │
              │  ┌─────┐ ┌─────┐ ┌─────┐    │
              │  │ 🌐  │ │ 🌐  │ │ 🌐  │    │
              │  └─────┘ └─────┘ └─────┘    │
              └─────────────────────────────┘
```

## Training with RFT

RFT produces a smaller model trained specifically on the browser-agent actions that work for your tasks, so you can run cheaper inference without losing task performance. Create a reinforcement fine-tuning job from evaluation results:

### Evaluation → RFT lifecycle

1. Run `pytest test_agent_auth.py -vs` to generate Eval Protocol results from your task dataset.
2. Eval scoring uses `AGENT_AUTH_EVALUATION_CRITERIA` (via WebJudge) to produce the success/failure signal.
3. Run `ep create rft ...` to build the training dataset from those evaluation results and start an RFT job.
4. After training completes, evaluate the new model again with the same `test_agent_auth.py` flow.

```bash
ep create rft --base-model accounts/fireworks/models/qwen3-vl-8b-instruct --chunk-size 50 --max-context-length 32768 --batch-size 32768 --epochs 4
```

### Using the RFT model

After the RFT job completes, you get a new model ID (e.g. from Fireworks). To evaluate that model instead of the default, set it in `test_agent_auth.py` in the `@evaluation_test` decorator:

```python
completion_params=[
    {"model": "accounts/fireworks/models/your-rft-model-id"},
],
```

Then run the evaluation as usual: `pytest test_agent_auth.py -vs`.

## Writing Custom Evaluations

Use `KernelBrowserRolloutProcessor` with your own dataset and scorer:

```python
from eval_protocol.pytest import evaluation_test
from eval_protocol.models import EvaluateResult
from kernel_browser_rollout_processor import (
    KernelBrowserRolloutProcessor,
    decode_screenshots,
)
from core.reward_models.webjudge import Trajectory, WebJudge
from agent_auth.actions import AGENT_AUTH_ACTIONS
from agent_auth.config import get_agent_auth_system_prompt

@evaluation_test(
    input_dataset=["your_tasks.jsonl"],
    rollout_processor=KernelBrowserRolloutProcessor(
        pool_name="your-pool",
        max_steps=15,
        system_prompt=get_agent_auth_system_prompt(),
        extra_actions=AGENT_AUTH_ACTIONS,
    ),
    completion_params=[{"model": "accounts/fireworks/models/qwen3-vl-30b-a3b-thinking"}],
)
async def test_your_evaluation(row):
    extra = row.execution_metadata.extra
    screenshots = decode_screenshots(extra["screenshots_b64"])
    actions = extra["action_history"]
    messages = row.messages
    score = your_scorer(screenshots, actions)
    row.evaluation_result = EvaluateResult(score=score, reason="...")
    return row
```

## Project Structure

```
kernel-eval-protocol-quickstart/
├── core/
│   ├── agent.py
│   ├── agent_loop.py
│   ├── browser.py
│   ├── actions.py
│   ├── prompts.py
│   ├── tracking.py
│   ├── utils.py
│   └── reward_models/
│       ├── base.py
│       └── webjudge.py
├── agent_auth/
│   ├── actions.py
│   └── config.py
├── kernel_browser_rollout_processor.py
├── test_agent_auth.py
├── tasks.jsonl
├── requirements.txt
├── pytest.ini
└── README.md
```

## Related

- [Eval Protocol](https://github.com/eval-protocol) — Pytest-based LLM evaluation framework
- [Fireworks](https://fireworks.ai) — VLM inference (e.g. Qwen3-VL)
- [Kernel](https://onkernel.com) — Browser-as-a-service
