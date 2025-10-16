---
title: "rLLM v0.2: RL Training over General Agentic Programs"
author: "Sijun Tan, Kyle Montgomery and the rLLM Team"
author_line: "Sijun Tan, Kyle Montgomery, and the rLLM Team"
date: "2025-10-16"
citation_key: "rllm2025v0.2"
---

<aside>

**TL;DR**

We are excited to release **rLLM v0.2**, a major upgrade of our RL training framework. In v0.1, rLLM provided agent and OpenAI Gym-like environment abstractions to support training ReACT-style agents. In v0.2, we introduce `AgentWorkflowEngine` and `AgentWorkflowTrainer`—more general abstractions that **enable arbitrary agentic programs to be trained**. Agent builders and researchers can now define multi-agent systems, complex workflows (e.g., solver-judge, planner executor, MCTS), and agentic programs with custom reward functions, and train them with reinforcement learning **without rewriting their production code**.

👨‍💻 [Github](https://github.com/rllm-org/rllm)  |  📖 [Docs](https://rllm-project.readthedocs.io/en/stable/)  |  💬 [Discord](https://discord.gg/BDH46HT9en)

</aside>


## Key Features in v0.2

1. Support `verl==0.5.0` as training backend, no custom verl fork anymore! `verl==0.5.0` comes with support of the following features which are now supported in rLLM:
    * Megatron training support
    * SGLang as the rollout engine, in addition to vLLM.
2. Introduce `AgentWorkflowEngine`, which enables passing in arbitrary agentic programs for training.
3. Support more agents and environments
    * Terminus and TerminalBench
    * Tongyi DeepResearch agent
    * AppWorld and AppWorldReactAgent
4. Integration with other agentic framework/SDK
    * Strands SDK from AWS
    * SmolAgents



## From Agent-Environment Abstractions to General Agentic Programs

rLLM is our framework for post-training language agents. It enables users to build custom agents and environments, train them with reinforcement learning, and deploy them for real-world workloads. rLLM powers the training of Agentica's models such as **DeepScaleR**, **DeepCoder**, **DeepSWE**, as well as Tongyi's **DeepResearcher**.

### Agent-Environment Abstraction in v0.1

In rLLM v0.1, our abstraction was centered around the interaction between an agent and an environment, following the classic reinforcement learning paradigm:

```python
# v0.1 Pattern: Agent-Environment Loop
observation, info = env.reset(task)
agent.update_from_env(observation, 0, False, info)

for step in range(max_steps):
    # Agent generates action based on observation
    response = await get_model_response(agent.chat_completions)
    action = agent.update_from_model(response)
    
    # Environment processes action and returns feedback
    next_obs, reward, done, info = env.step(action)
    agent.update_from_env(next_obs, reward, done, info)
    
    if done:
        break
```

This abstraction, inspired by OpenAI Gym, is expressive enough for many single-agent workloads. It enabled us to train ReACT-style agents on diverse tasks like code generation, web navigation, and mathematical reasoning.

However, as we began scaling to more complex agentic systems, several limitations became clear:
1. **Rigid Structure**: The fixed agent → environment → agent interaction loop doesn’t naturally accommodate all forms of agentic systems.
2. **Challenges for Complex Control Flow**: Real-world agentic frameworks (e.g., LangGraph, AutoGPT, CrewAI) rarely follow the agent-environment paradigm. They often involve multiple LLM-based agents playing different roles (e.g., solver + judge, planner + executor), or workflows with conditional branching, parallel execution, or tree search (like MCTS). Such complex control flows are difficult to express within the current abstraction.
3. **Production Gap**. The mismatch between the training paradigm and production-level agentic architectures forces developers to reimplement their systems solely for training purposes.

### The v0.2 Solution: Workflow Abstraction

At its core, any agentic system—no matter how complex—is simply a composition of LLM calls orchestrated by a program that manages their control flow.

In v0.2, we introduce `AgentWorkflowEngine` and `AgentWorkflowTrainer`, which generalize the v0.1 approach: **treat any Python program with LLM calls as a trainable workflow**.

Instead of forcing your system into the agent–environment mold, you can now define your agentic program in a natural, declarative way:
1. Inherit from the `Workflow` base class
2. Implement the `run()` method with your program logic
3. Return an `Episode` object containing the trajectories and rewards to be optimized.

The engine takes care of the rest—handling parallel execution, retry logic, trajectory collection, advantage computation, and PPO/GRPO training—so you can focus on your workflow logic.

While we switch to a new programming paradigm in v0.2, we maintain backward compatibility with the original agent-environment abstraction. The original `AgentExecutionEngine` in v0.1 will continue to work without modification. 


## Example: Building a Solver-Judge Flow

Let’s walk through a concrete example that demonstrates the power of the new workflow abstraction: a **solver–judge workflow**, where the LLM acts as both the problem solver and the verifier.

In this setup, multiple solver agents generate candidate solutions in parallel, and a judge agent evaluates those solutions to select the best one. This classic setup for test-time scaling can now be trained end-to-end, allowing both the solver and judge to jointly optimize for high-quality outputs.


### 💡 Workflow Overview
More concretely, here's how the workflow operates:

1. Given a problem, `N` solver agents generate solutions in parallel.
2. Once all solvers finish, the judge compares their outputs and selects the best one.
3. Each solver receives a reward based on its solution’s correctness, while the judge’s reward depends on whether it selected a correct answer.
4. Finally, the workflow returns an `Episode` that bundles all trajectories—one for each solver and one for the judge.

This makes it easy to train complex, multi-agent reasoning systems under a single, unified framework.

### 🧩 Code Example

```python
class Solver:
    def __init__(self, rollout_engine: RolloutEngine, **kwargs):
        self.rollout_engine = rollout_engine

    async def generate_solution(self, problem: str) -> Trajectory:
        messages = [{"role": "user", "content": f"{problem}. Output the final answer within <answer>...</answer>"}]
        output: ModelOutput = await self.rollout_engine.get_model_response(messages)
        return Trajectory(
            name="solver",
            steps=[
                Step(
                    chat_completions=messages + [{"role": "assistant", "content": output.content, "reasoning": output.reasoning}],
                    thought=output.reasoning,
                    action=self._parse_solver_response(output.content),
                    model_output=output,
                )
            ],
        )

    async def generate_solutions(self, problem: str, n_solutions: int = 2) -> list[Trajectory]:
        tasks = [asyncio.create_task(self.generate_solution(problem)) for _ in range(n_solutions)]
        return await asyncio.gather(*tasks)
    
    
class Judge:
    def __init__(self, rollout_engine: RolloutEngine, **kwargs):
        self.rollout_engine = rollout_engine

    async def judge_solutions(self, problem: str, solutions: list[str]) -> Trajectory:
        messages = [{"role": "user", "content": self._create_judge_prompt(problem, solutions)}]
        output: ModelOutput = await self.rollout_engine.get_model_response(messages)
        return Trajectory(
            name="judge",
            steps=[
                Step(
                    chat_completions=messages + [{"role": "assistant", "content": output.content, "reasoning": output.reasoning}],
                    thought=output.reasoning,
                    action=self._parse_judge_response(output.content, solutions),
                    model_output=output,
                )
            ],
        )
        
        
class SolverJudgeWorkflow(Workflow):
    def __init__(self, rollout_engine: RolloutEngine, n_solutions: int = 2, reward_function: RewardFunction = None, **kwargs):
        super().__init__(rollout_engine, **kwargs)
        self.n_solutions = n_solutions
        self.reward_function = reward_function
        self.solver = Solver(rollout_engine)
        self.judge = Judge(rollout_engine)

    async def run(self, task: dict, uid: str, **kwargs) -> Episode:
        self.reset(task, uid)
        problem = task["question"]

        # Step 1: Solver generates multiple solutions in parallel
        solver_trajectories = await self.solver.generate_solutions(problem, self.n_solutions)

        # Assign rewards to solver trajectories
        solutions = []
        for traj in solver_trajectories:
            solution = traj.steps[0].action
            solutions.append(solution)
            reward = self.reward_function(task, solution).reward
            traj.steps[0].reward = reward

        # Step 2: Judge selects the best solution
        judge_trajectory = await self.judge.judge_solutions(problem, solutions)
        selected_solution = judge_trajectory.steps[0].action

        # Evaluate the selected solution
        reward_result = self.reward_function(task, selected_solution)
        judge_trajectory.steps[0].reward = reward_result.reward
        is_correct = reward_result.is_correct

        # Compute metrics
        solver_acc = sum(traj.steps[0].reward for traj in solver_trajectories) / len(solver_trajectories)
        judge_acc = int(is_correct)

        # Step 3: Return episode with multiple trajectories
        return Episode(
            id=uid,
            task=task,
            trajectories=[*solver_trajectories, judge_trajectory],
            is_correct=is_correct,
            metrics={"solver_acc": solver_acc, "judge_acc": judge_acc},
        )
```
### ⚙️ What Happens During Training

During GRPO training, the process looks like this:

1. For each task, the engine generates K rollouts, producing `K × N` solver trajectories and `K` judge trajectories.
2. Trajectories are grouped by their name (e.g., `"solver"` or `"judge"`) for advantage computation.
3. The system then updates the shared policy to improve both solution quality and judgment accuracy.

This design gives developers full control over how rewards and trajectory groupings are defined—making it easy to experiment with different trajectory grouping strategies (e.g., intra- or inter-workflow grouping).


## The AgentWorkflowEngine and AgentTrainer

### ⚙️ Running Workflows with `AgentWorkflowEngine`

The `AgentWorkflowEngine` is responsible for executing your workflow in parallel and collecting all the resulting trajectories. It’s conceptually similar to `AgentExecutionEngine` from v0.1, but designed to work seamlessly with the new workflow abstraction.

Here’s how you might run the `SolverJudgeWorkflow` we defined earlier:


```python
import asyncio
from transformers import AutoTokenizer
from rllm.engine import AgentWorkflowEngine, OpenAIEngine
from rllm.rewards.countdown_reward import countdown_reward_fn

# Setup rollout engine
model_name = "Qwen/Qwen3-0.6B"
tokenizer = AutoTokenizer.from_pretrained(model_name)

rollout_engine = OpenAIEngine(
    model=model_name,
    tokenizer=tokenizer,
    max_prompt_length=2048,
    max_response_length=1024,
    base_url="http://localhost:30000/v1",
    api_key="None",
    sampling_params={"temperature": 0.6, "top_p": 0.95},
)

# Create workflow engine
engine = AgentWorkflowEngine(
    workflow_cls=SolverJudgeWorkflow,
    workflow_args={
        "n_solutions": 2,
        "reward_function": countdown_reward_fn,
    },
    rollout_engine=rollout_engine,
)

# Execute tasks
tasks = [{"question": "What is 2+2?"}, {"question": "Solve x^2 = 16"}]
episodes = asyncio.run(engine.execute_tasks(tasks))

# Process results
for episode in episodes:
    print(f"Episode {episode.id}: {len(episode.trajectories)} trajectories")
    print(f"Correctness: {episode.is_correct}, Metrics: {episode.metrics}")
```

### 🧠 Training with AgentTrainer

Once you’ve verified that your workflow runs as expected, the next step is training it end-to-end.
The `AgentTrainer` takes care of sampling trajectories, computing advantages, and performing PPO optimization — all under the hood using the same `AgentWorkflowEngine` for trajectory rollouts.

```python
import hydra
from rllm.trainer.agent_trainer import AgentTrainer
from rllm.data.dataset import DatasetRegistry
from rllm.rewards.countdown_reward import countdown_reward_fn

@hydra.main(config_path="pkg://rllm.trainer.config", config_name="agent_ppo_trainer", version_base=None)
def main(config):
    # Load datasets
    train_dataset = DatasetRegistry.load_dataset("countdown", "train")
    test_dataset = DatasetRegistry.load_dataset("countdown", "test")

    # Create trainer with your workflow
    trainer = AgentTrainer(
        workflow_class=SolverJudgeWorkflow,
        workflow_args={
            "n_solutions": 2,
            "reward_function": countdown_reward_fn,
        },
        config=config,
        train_dataset=train_dataset,
        val_dataset=test_dataset,
    )
    
    # Train!
    trainer.train()

if __name__ == "__main__":
    main()
```


### 📊 Training Results

We ran a toy training experiment on the Countdown task using `Qwen3-0.6B` as the base model, with responses capped at 1,024 tokens. After just 250 training steps, the solver’s standalone accuracy improved from 50% to 65%, while the solver-judge workflow achieved around 75% accuracy — demonstrating the benefit of joint optimization.

![Accuracy of the solver-judge flow on the Countdown task as training progresses](assets/rllm/solver-judge-acc.png)

This example demonstrates the end-to-end process of building and training an agentic workflow with rLLM. You can easily modify this setup to explore different workflow variants, tasks, or reward functions — with minimal changes to the code. 

**Note**: You can find the codes for this example [here](https://github.com/rllm-org/rllm/tree/main/examples/solver_judge).

## Towards Building the RL Application Stack

rLLM v0.2 represents a fundamental evolution in how we think about training language agents. By generalizing from agent-environment abstractions to flexible workflow abstractions, we enable training of **any agentic program**—from traditional ReACT agents to complex multi-agent systems and sophisticated agentic workflows.

As agentic applications mature—from single-agent prototypes to production systems with multiple specialized agents, intricate control flow, and real-world feedback loops—the need for end-to-end reinforcement learning becomes essential. The next generation of agentic systems will be dynamic, compositional, and ever-evolving—demanding training infrastructure that can keep pace.

Our core insight is simple: **RL training should adapt to your system, not the other way around**. Reinforcement learning shouldn’t be confined to optimizing a single agent in isolation—it should drive the optimization of entire agentic applications, from individual agent policies to multi-agent coordination.

This is the vision behind rLLM: to build the **RL application stack** that empowers developers to train the next generation of intelligent agents and agentic systems—at any scale. Whether you’re fine-tuning a single reasoning model or optimizing a complex multi-agent research assistant with dozens of interconnected components, rLLM provides the foundation to do so seamlessly.

We can’t wait to see what you build with rLLM v0.2. Share your workflows, experiments, and insights with the community!

## Contributors & Advisors
The rLLM v0.2 release is led by Sijun Tan and Kyle Montgomery, with contributions from rLLM team members Tianhao Wu, Manan Roongta, Jason Wei, Julie Shi, Jeewoo Lee, Yuqi Chen, and Peihang Li, along with external contributors: Tianyi Zhang, Sunan Xiang, and Sida Li. 

rLLM was originally developed by Sijun Tan, Michael Luo, and Colin Cai together with the Agentica team.

The rLLM project is advised by Chenguang Wang, Li Erran Li, Raluca Ada Popa, and Ion Stoica.

## Acknowledgement
The rLLM project is generously supported by grants from Laude Institute, Hyperbolic, Fireworks, and AWS. We also pay special thanks to Together AI for the research partnership and compute support. 