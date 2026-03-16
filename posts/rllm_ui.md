---
title: "rLLM UI: Real-Time Observability Tool for Agent Training & Evaluation"
author: "Chanbin Park and the rLLM Team"
author_line: "Chanbin Park and the rLLM Team"
date: "2026-03-12"
citation_key: "rllm2026ui"
---

<aside>

## TL;DR

We introduce [**rLLM UI**](https://github.com/rllm-org/rllm-ui), a real-time observability platform for training and evaluating agents using the rLLM framework. Existing tools like [wandb](https://wandb.ai/site/) show *what* is happening during training, while rLLM UI shows you *why*. It not only shows important metrics as graphs, but lets you inspect exactly what the model generates at every step, and interact with our observability agent to pinpoint the issue. It seamlessly connects with the new CLI tool.
    
- 💻: https://github.com/rllm-org/rllm-ui
- 🕸️: https://ui.rllm-project.com
- 📖: https://docs.rllm-project.com/experimental/ui

</aside>

## Problem

When training goes wrong, metrics alone don't show the full picture. The natural next step is to look at what the model generated. But even then, how you see those traces is an entirely different problem.

## Introducing rLLM UI

We introduce rLLM UI, a web interface that gives you full visibility into your rLLM training and evaluation runs. Here's a snapshot of what the UI looks like for a specific training run:

![rLLM UI dashboard showing metrics and training traces for a training run.](https://hackmd.io/_uploads/ByvYCCgqWg.png)


### 1. View your Training Traces

rLLM is built around a rich structure for agent data. Here is an example of optimizing a solver-judge workflow with GRPO: 
- two solvers compete on a math problem
- a judge picks the better answer
- n rollouts are generated per prompt 

In each rollout, every solver and judge produces its own trajectory. An `Episode` is the collection of all those trajectories for a single rollout — the full picture from the agent's perspective. But during GRPO, advantages aren't computed per-episode; they're computed across trajectories of the same type: all solvers together, all judges together. We call each of these a `TrajectoryGroup`. For more details on the rLLM data format, check our [documentation](https://rllm-project.readthedocs.io/en/latest/core-concepts/agent-run-format/). Below is a diagram that shows the differences of `Episode` and `TrajectoryGroup`.

![Diagram comparing Episode and TrajectoryGroup](https://hackmd.io/_uploads/B1ucu0e9Zx.png)

rLLM UI makes it very easy to view the training traces in both `Episode` and `TrajectoryGroup` format. Many observability tools don't even support tracing, and the ones that do lack this rich, underlying structure. Here's a comparison of the same trajectories for the same prompt visualized in both views — you can toggle between them seamlessly on the UI.

![Comparison of Episode View and TrajectoryGroup View](https://hackmd.io/_uploads/SkwC2Ax5-x.png)


### 2. Ask the Observability Agent

When you have hundreds of episodes, manually filtering isn't practical. That's where the observability agent comes in. 

Seeing training not improving that much, I asked, *"Which episodes got 0 reward and why did this happen in step 19?"* Here's what the agent concluded:

![The observability agent identifying that episodes with 0 reward in step 19 are caused by the model running out of tokens.](https://hackmd.io/_uploads/BkDPqebcWg.png)


It correctly identified that the model is running out of tokens, and suggested that the max generation length should be increased. Pretty detailed and helpful, right? What sets it apart from a generic chatbot is that it understands your run, as it has tool access to your training code, training traces, and your configurations.

To enable it, add your ANTHROPIC_API_KEY in Settings.


### 3. Dig in yourself

You can also view the terminal logs, relevant code, and all configuration at one place.

Logs appear as you would see it in your terminal:

![Terminal logs captured and displayed in the rLLM UI.](https://hackmd.io/_uploads/r1JihAB9Wg.png)

Relevant code (workflow function and reward function) is extracted when the run starts and is displayed:

![Workflow and reward function code extracted and displayed in the rLLM UI.](https://hackmd.io/_uploads/Bk7cnCHcZl.png)

Both rLLM and backend-specific configs is shown:

![rLLM and backend-specific configuration displayed in the rLLM UI.](https://hackmd.io/_uploads/rk893RS9Wl.png)



### 4. Access UI through the rLLM CLI

To start viewing your tracing on the UI, you just need to use the CLI tool.

#### Login

From your terminal, run
```bash
rllm login
``` 
It will ask you to enter your API key, which you can obtain by signing up at [https://ui.rllm-project.com](https://ui.rllm-project.com).

If you want to host the UI yourself, please refer to the [self-hosted setup section](https://github.com/rllm-org/rllm-ui?tab=readme-ov-file#self-hosted-setup) in our GitHub repository.

#### Training run from bash script

Add ui to your trainer's logger list in your rLLM training script and start training.

```bash
trainer.logger="['console','ui']"
```

#### Training/Evaluation run from CLI

If you login and run train or eval through the CLI, you don't need to add anything.
```bash
rllm eval [dataset name]
rllm train [dataset name]
```

You can view your training traces in the Training tab, and your eval traces in the Evaluation tab. 


## Architecture Overview

Here's a high-level look at how the rLLM UI streams rich training information in real time — all while introducing minimal overhead to the main training job.

![Diagram of the high-level overview of how rllm-ui connects to rllm, the user and Anthropic API](https://hackmd.io/_uploads/rJ5-wy3F-g.png)


### rLLM <-> UI
rLLM connects to the UI via the UILogger backend, registered as "ui" in the Tracking class (`rllm/utils/tracking.py`).

On init, the logger:

1. Creates a training session via POST /api/sessions
2. Starts a background heartbeat thread (for crash detection)
3. Wraps stdout/stderr with TeeStream to capture training logs

During training, for every step, the logger sends metrics and episodes over HTTP asynchronously, using background workers.

### UI <-> User
On the UI side, a FastAPI backend receives this data and pushes updates to the frontend via Server-Sent Events (SSE). The result is a live view of training that updates without page refreshes.


## What's Next

We envision rLLM UI to be the central tracing layer for future rLLM features, and will keep releasing new features. Stay tuned!

---
