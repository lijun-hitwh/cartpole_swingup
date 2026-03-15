# mjlab example: cartpole_swingup

Learn how to define and train a simple robotics task using mjlab. This tutorial walks 
through the end-to-end process of training a cartpole to perform a swing-up maneuver.

## Prerequisites
This project uses `uv` for lightning-fast dependency management. Ensure you have it installed:
```
curl -LsSf https://astral.sh/uv/install.sh | sh
```

## Project Setup

### Initialize the Environment
Create your project directory and install `mjlab`:
```
mkdir cartpole_swingup && cd cartpole_swingup
uv init
uv add "mjlab>=1.2.0"
uv sync
```

### Configure Entry Points
To allow mjlab to auto-discover your tasks, you must declare an entry point in your `pyproject.toml`.
Note: A [build-system] table is required to ensure the entry point is correctly registered in your environment.
```
[build-system]
requires = ["uv_build>=0.8.18,<0.9.0"]
build-backend = "uv_build"

[dependency-groups]
dev = [
    "ruff>=0.14.14",
]

[tool.ruff]
indent-width = 2
src = ["cartpole_swingup"]

[tool.ruff.lint]
select = ["E4", "E7", "E9", "F", "I", "B"]

[project.entry-points."mjlab.tasks"]
cartpole_swingup = "cartpole_swingup"
```

## Usage Guide

### Phase 1: Sanity Check
Before training, verify the environment physics by watching the robot under zero or random actions.
```
# Watch the cartpole fall under gravity
uv run play Cartpole-Swingup --agent zero

# Observe behavior with random control inputs
uv run play Cartpole-Swingup --agent random
```

# Phase 2: Training
Train the reinforcement learning policy. We recommend using a high number of parallel environments for faster convergence.
```
uv run train Cartpole-Swingup --env.scene.num-envs 4096 --agent.max-iterations 500
```

# Phase 3: Evaluation
Once training is complete, visualize the results by loading your trained checkpoint.
```
# Load from a local path
uv run play Cartpole-Swingup --checkpoint-file <path-to-logs>/model.pt

# Or pull directly from Weights & Biases
uv run play Cartpole-Swingup --wandb-run-path <entity>/<project>/<run_id>
```