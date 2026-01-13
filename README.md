# Kernel Eval Protocol

Eval Protocol integration for evaluating VLM browser agents using [Kernel](https://onkernel.com) browser pools.

This package provides standardized evaluation of browser-based VLM agents using the [Eval Protocol](https://evalprotocol.com) framework.

## Overview

The integration provides:

- **`KernelBrowserRolloutProcessor`**: A custom rollout processor that runs multi-step browser episodes using Kernel browser pools
- **`core/`**: Vendored agent code (QwenAgent, WebJudge, browser adapter) from [kernel-tinker-rl](https://github.com/onkernel/kernel-tinker-rl)
- **`agent_auth/`**: Agent Auth benchmark configuration and custom actions
- **`test_agent_auth.py`**: Example evaluation test for the Agent Auth benchmark

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
│  │    3. Run agent loop (QwenAgent)                          │  │
│  │       - Screenshot → VLM predict → Execute → Repeat       │  │
│  │    4. Capture trajectory (screenshots, actions, messages) │  │
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
2. **OpenAI API Key**: Get from https://platform.openai.com (for WebJudge scoring)
3. **Fireworks API Key**: Get from https://fireworks.ai (for VLM inference)
4. **Browser Pool**: Create a Kernel browser pool named `eval-browser-pool`

### Installation

```bash
# Clone this repo
git clone <this-repo> kernel-eval-protocol
cd kernel-eval-protocol

# Install dependencies with uv
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
    # Trajectory data is in row.execution_metadata.extra
    extra = row.execution_metadata.extra
    screenshots = decode_screenshots(extra["screenshots_b64"])
    actions = extra["action_history"]
    
    # Message history (including tool_calls) is in row.messages
    messages = row.messages
    
    # Your evaluation logic here
    score = your_scorer(screenshots, actions)
    
    row.evaluation_result = EvaluateResult(score=score, reason="...")
    return row
```

## Project Structure

```
kernel-eval-protocol/
├── core/                          # Vendored from kernel-tinker-rl
│   ├── agent.py                   # QwenAgent - VLM agent with message history
│   ├── agent_loop.py              # Multi-step agent loop
│   ├── browser.py                 # Kernel browser adapter
│   ├── actions.py                 # Action definitions (click, type, etc.)
│   ├── prompts.py                 # System prompt builder
│   └── reward_models/
│       ├── base.py                # Base reward model
│       └── webjudge.py            # WebJudge LLM-as-Judge scorer
├── agent_auth/                    # Agent Auth benchmark
│   ├── actions.py                 # FoundInputsAction for form discovery
│   └── config.py                  # System prompt configuration
├── kernel_browser_rollout_processor.py  # Eval Protocol rollout processor
├── test_agent_auth.py             # Example evaluation test
├── tasks.jsonl                    # Agent Auth task dataset
├── pyproject.toml                 # Package configuration
└── README.md
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

### Message History (row.messages)

The agent preserves full conversation history including native tool calls:

```python
[
    {"role": "system", "content": "...system prompt..."},
    {"role": "user", "content": [{"type": "image_url", ...}, {"type": "text", ...}]},
    {"role": "assistant", "content": "...", "tool_calls": [{"id": "...", "function": {...}}]},
    ...
]
```

### Trajectory Data (row.execution_metadata.extra)

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

## Modifications to Vendored Code

The `core/` directory is vendored from [kernel-tinker-rl](https://github.com/onkernel/kernel-tinker-rl) with the following modifications:

### 1. Message History Preservation (`core/agent.py`)

The original `QwenAgent` only stored text responses, discarding native tool calls. We modified it to preserve the full conversation history:

- Added `messages: list[dict]` to `AgentState`
- In `predict()`, now stores:
  - System message (on first call)
  - User messages (with screenshots + instruction)
  - Assistant responses **including `tool_calls`** when present
- Added `get_messages()` method to retrieve the full history

This enables accurate conversation replay in the Eval Protocol UI.

### 2. WebJudge OpenAI API Compatibility (`core/reward_models/webjudge.py`)

The original code used `max_tokens` and `temperature=0` which work with OpenRouter but not with direct OpenAI API calls (newer models like `gpt-5-mini`):

- Changed `max_tokens` → `max_completion_tokens`
- Removed `temperature=0` (newer OpenAI models only support `temperature=1`)

These changes allow WebJudge to work with direct OpenAI API calls instead of requiring OpenRouter.

## Related Projects

- [kernel-tinker-rl](https://github.com/onkernel/kernel-tinker-rl) - VLM RL training for computer use (source of vendored core/)
- [Eval Protocol](https://evalprotocol.com) - Pytest-based LLM evaluation framework
- [Kernel](https://onkernel.com) - Browser-as-a-service
