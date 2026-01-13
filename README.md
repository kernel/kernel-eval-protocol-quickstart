# Kernel Eval Protocol

Eval Protocol integration for evaluating VLM browser agents using [Kernel](https://onkernel.com) browser pools.

This package bridges [kernel-tinker-rl](https://github.com/onkernel/kernel-tinker-rl) with [Eval Protocol](https://evalprotocol.com) to enable standardized evaluation of browser-based VLM agents.

## Overview

The integration provides:

- **`KernelBrowserRolloutProcessor`**: A custom rollout processor that runs multi-step browser episodes using Kernel browser pools
- **`test_agent_auth.py`**: Example evaluation test for the Agent Auth benchmark using WebJudge

## How It Works

```
┌─────────────────────────────────────────────────────────────────┐
│                     Eval Protocol                               │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  @evaluation_test(...)                                    │  │
│  │  async def test_agent_auth(row):                         │  │
│  │      # Rollout already executed by processor              │  │
│  │      trajectory = get_trajectory(row)                     │  │
│  │      score = webjudge.evaluate(trajectory)               │  │
│  └──────────────────────────────────────────────────────────┘  │
│                            │                                    │
│                            ▼                                    │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  KernelBrowserRolloutProcessor                            │  │
│  │    1. Acquire browser from Kernel pool                    │  │
│  │    2. Navigate to initial URL                             │  │
│  │    3. Run agent loop (kernel-tinker-rl)                   │  │
│  │       - Screenshot → VLM predict → Execute → Repeat       │  │
│  │    4. Capture trajectory (screenshots, actions)           │  │
│  │    5. Release browser                                     │  │
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

## Setup

### Prerequisites

1. **Kernel API Key**: Get from https://onkernel.com
2. **OpenAI API Key**: Get from https://platform.openai.com (for WebJudge)
3. **Fireworks API Key**: Get from https://fireworks.ai (for VLM inference)
4. **Browser Pool**: Create a Kernel browser pool named `eval-browser-pool`

### Required: kernel-tinker-rl

This package depends on [kernel-tinker-rl](https://github.com/onkernel/kernel-tinker-rl) which must be cloned at the **same directory level** as this repo:

```
your-projects/
├── kernel-tinker-rl/   ← Clone this first
└── kernel-eval-protocol/
```

```bash
# Clone kernel-tinker-rl first (required dependency)
git clone https://github.com/onkernel/kernel-tinker-rl
```

### Installation

```bash
# Clone this repo at the same level as kernel-tinker-rl
git clone <this-repo> kernel-eval-protocol
cd kernel-eval-protocol

# Install with uv (will link to ../kernel-tinker-rl as editable dependency)
uv sync

# Set environment variables
export KERNEL_API_KEY="your-kernel-key"
export OPENAI_API_KEY="your-openai-key"
export FIREWORKS_API_KEY="your-fireworks-key"
```

### Create Browser Pool

```bash
# Using Kernel CLI
kernel pools create eval-browser-pool --size 8
```

## Usage

### Run Agent Auth Evaluation

```bash
# Activate venv
source .venv/bin/activate

# Run evaluation
pytest test_agent_auth.py -vs

# With limited dataset size
EP_MAX_ROWS=5 pytest test_agent_auth.py -vs
```

### Custom Evaluations

Create your own evaluation test:

```python
from eval_protocol.pytest import evaluation_test
from kernel_browser_rollout_processor import (
    KernelBrowserRolloutProcessor,
    decode_screenshots,
)

@evaluation_test(
    input_dataset=["your_tasks.jsonl"],
    rollout_processor=KernelBrowserRolloutProcessor(
        pool_name="your-pool",
        max_steps=15,
    ),
    completion_params=[{"model": "qwen/qwen3-vl-8b-instruct"}],
)
async def test_your_evaluation(row):
    # Trajectory data is in row.execution_metadata.extra
    extra = row.execution_metadata.extra
    screenshots = decode_screenshots(extra["screenshots_b64"])
    actions = extra["action_history"]
    
    # Your evaluation logic here
    score = your_scorer(screenshots, actions)
    
    row.evaluation_result = EvaluateResult(score=score, reason="...")
    return row
```

## Data Format

### Input Dataset (tasks.jsonl)

```json
{
  "id": "gandi-net-register",
  "initial_url": "https://gandi.net",
  "task": "Navigate to gandi.net and find the register page..."
}
```

### Trajectory Data (stored in row.execution_metadata.extra)

```python
{
    # For evaluation (WebJudge)
    "screenshots_b64": ["base64...", ...],  # PNG images as base64
    "action_history": ["click(100, 200)", ...],
    
    # Task info
    "task": "Navigate to...",
    "task_id": "gandi-net-register",
    "initial_url": "https://gandi.net",
    "final_url": "https://gandi.net/register",
    
    # Episode metadata
    "termination_reason": "terminal_action",
    "terminal_action": "found_inputs(...)",
    "steps_completed": 5,
    "error": null,
    
    # Step details (for debugging)
    "step_results": [...]
}
```

## Files

- `kernel_browser_rollout_processor.py` - The rollout processor
- `test_agent_auth.py` - Agent Auth evaluation test
- `pyproject.toml` - Package configuration
- `README.md` - This file

## Related Projects

- [kernel-tinker-rl](https://github.com/onkernel/kernel-tinker-rl) - VLM RL training for computer use
- [Eval Protocol](https://evalprotocol.com) - Pytest-based LLM evaluation framework
- [Kernel](https://onkernel.com) - Browser-as-a-service

