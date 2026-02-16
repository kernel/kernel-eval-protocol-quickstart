# Kernel Eval Protocol

Eval Protocol integration for evaluating and fine-tuning VLM browser agents using [Kernel](https://onkernel.com) serverless browsers and [Fireworks](https://fireworks.ai) for VLM inference.

## Quickstart

1. **Clone and install**

   ```bash
   git clone <this-repo> kernel-eval-protocol-quickstart
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

4. **Run the evaluation**

   ```bash
   pytest test_agent_auth.py -vs
   ```

   To run on fewer tasks (e.g. 5 rows): `EP_MAX_ROWS=5 pytest test_agent_auth.py -vs`

## What Happens When You Run It

Eval Protocol reads `tasks.jsonl` (hundreds of browser tasks). For each task:

- **KernelBrowserRolloutProcessor** acquires a browser from the pool, navigates to the task URL, runs the VLM agent loop (screenshot → predict → execute → repeat), captures the trajectory, then releases the browser.
- The test function scores each trajectory with **WebJudge** (LLM-as-judge).
- Results are reported by pytest / Eval Protocol.

```
┌─────────────────────────────────────────────────────────────────┐
│                     Eval Protocol                               │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  @evaluation_test(...)                                    │  │
│  │  async def test_agent_auth(row):                         │  │
│  │      trajectory = get_trajectory(row)                     │  │
│  │      score = webjudge.evaluate(trajectory)               │  │
│  └──────────────────────────────────────────────────────────┘  │
│                            │                                    │
│                            ▼                                    │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  KernelBrowserRolloutProcessor                            │  │
│  │    1. Acquire browser from Kernel pool                    │  │
│  │    2. Navigate to initial URL                             │  │
│  │    3. Run agent loop (screenshot → predict → execute)     │  │
│  │    4. Capture trajectory, release browser                │  │
│  └──────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                             │
                             ▼
              ┌─────────────────────────────┐
              │      Kernel Browser Pool     │
              │  ┌─────┐ ┌─────┐ ┌─────┐    │
              │  │ 🌐  │ │ 🌐  │ │ 🌐  │    │
              │  └─────┘ └─────┘ └─────┘    │
              └─────────────────────────────┘
```

## Training with RFT

RFT produces a smaller model trained specifically on the browser-agent actions that work for your tasks, so you can run cheaper inference without losing task performance. Create a reinforcement fine-tuning job from evaluation results:

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

- [Eval Protocol](https://evalprotocol.com) — Pytest-based LLM evaluation framework
- [Fireworks](https://fireworks.ai) — VLM inference (e.g. Qwen3-VL)
- [Kernel](https://onkernel.com) — Browser-as-a-service
