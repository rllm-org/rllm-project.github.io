---
title: "Continual Learning via Real-Time RL for Agents"
author: "Linbo Liu, Yuzhe Lu, Youzhi Luo, Sijun Tan, Panpan Xu, Luke Huan"
author_line: "Linbo Liu, Yuzhe Lu, Youzhi Luo, Sijun Tan, Panpan Xu, Luke Huan, and the rLLM Team"
date: "2026-07-27"
citation_key: "rllm2026realtimerl"
---

<aside>

## TL;DR

We study **continual learning for agents** in a **real-time RL** setup, where each production task yields exactly one rollout instead of the repeated samples that methods like GRPO rely on. Using a **batch-normalized advantage** that centers each reward against the batch mean, we trained Qwen3-Coder-30B on [**MigrationBench**](https://arxiv.org/pdf/2505.09569) with [rLLM](https://github.com/rllm-org/rllm) and lifted migration success rate from **43%** to **59.2%**, matching GRPO under the same rollout budget. With PPO clipping and truncated importance sampling, training stays stable even when rollouts arrive asynchronously from a stale policy.

👨‍💻 [Code](https://github.com/rllm-org/rllm/tree/main/cookbooks/migrationbench) | 📖 [Dataset](https://huggingface.co/collections/AmazonScience/migrationbench)
</aside>

Imagine an AI agent that keeps learning after deployment, from the real user interactions it handles every day. This is different from the usual **build-time** training pipeline, where agentic RL happens before the model is shipped: tasks are curated, environments are controlled, and the trainer can often generate multiple rollouts for the same prompt. In a real-time continual learning setting, each production interaction may happen only once, so the model has to learn from a stream of one-off trajectories.

This matters for domains where experience accumulates over time. In automated research, agents could learn from failed hypotheses, experiments, and critiques; in software engineering, they could adapt to a team’s codebase, tests, review style, and recurring bugs. Continual learning lets deployed agents improve from real production experience, instead of staying fixed after launch.

In this blog post, we study the continual learning problem in a mocked **real-time RL** setup. We trained a Qwen3-Coder-30B on SWE-task: [MigrationBench](https://arxiv.org/pdf/2505.09569), a repository-level Java 8 to Java 17 migration benchmark, using [rLLM](https://github.com/rllm-org/rllm). We show that a simple **batch-normalized advantage**, which centers each reward against the batch mean, solves the single-rollout problem in real-time RL, improving migration success rate from **43%** to **59.2%**, a **16.2% absolute gain**.

Of course, a benchmark is only a proxy: real production traffic is much harder than MigrationBench. Feedback signals in the wild are noisy and unreliable, and a system that updates weights live needs safeguards such as regression testing before each new checkpoint serves users. We do not solve those problems here. Instead, MigrationBench's clean, verifiable reward lets us isolate the core algorithmic question: can a model learn effectively when every task yields exactly one rollout? The early results are promising, and the batch-level advantage is a building block that a full production system can adopt.

## The Challenge: From Build-Time to Real-Time

Most agentic RL methods are trained during model **build-time**, before the model is deployed in production, under a repeated-rollout setup. Take [GRPO](https://arxiv.org/pdf/2402.03300) (Group Relative Policy Optimization) for example: for each task, the trainer samples multiple rollouts, compares their outcomes, and computes group-relative advantages. This works well when the same task can be replayed several times.

But **real-time** production-style agents don't work that way. After deployment, user interactions arrive as one-off tasks. When a user asks your agent to perform a coding task or answer a question, you usually get exactly one rollout: one attempt, one outcome, and no second sample from the same task to compare against. That means the learning signal has to come from a single trajectory rather than from the group-relative structure that methods like GRPO depend on.

<img src="../assets/realtime_rl/buildtime_vs_realtime.png" alt="Figure 1: Build-time RL vs real-time RL. Build-time RL (top) replicates tasks to generate multiple rollouts in a sandboxed environment, while real-time RL (bottom) learns from uncontrolled production traffic." style="border: none;" />

Figure 1 highlights the core difference between GRPO-style build-time RL and real-time RL. This fundamental difference creates two immediate training challenges:

1. **High variance reward** from single-rollout-per-task, since we cannot average over multiple attempts for the same query
2. **Insufficient learning signal** from failed rollout, since a trajectory with zero reward can contribute no gradient in algorithms like [REINFORCE](https://link.springer.com/article/10.1007/BF00992696)

## Our Solution: Advantage Estimation from One Rollout

Our approach is to normalize rewards across the batch. The key idea is that even with only one rollout per prompt, the model can still obtain an advantage signal by comparing that rollout against a population-level baseline. Instead of requiring multiple attempts on the same query, we estimate the baseline from rewards across different tasks in the batch. This turns each isolated raw reward into a batch-relative advantage, allowing one-rollout training to retain the variance-reduction benefits normally associated with multi-rollout methods. Similar batch-level normalization has been explored in prior work such as [REINFORCE++](https://arxiv.org/pdf/2501.03262); our contribution is showing that it makes single-rollout, real-time RL practical for agents.

This also fixes a limitation of raw-reward REINFORCE. In REINFORCE, the policy-gradient update is weighted directly by the rollout reward, so a reward of 0 produces no update at all. With a batch-normalized advantage, however, a zero-reward rollout can still be informative: if the batch average reward is 0.3, then a reward of 0 becomes a negative advantage signal, telling the model that this response performed worse than the population baseline.

We keep this section high-level; the detailed derivation and implementation are in [How Batch Normalization Works](#how-batch-normalization-works).

## Results: Real-Time RL with Batch Normalized Advantages

We study this setting by mocking real-time RL in rLLM. Each prompt is rolled out once, each trajectory is consumed for training once, and the trainer operates on the stream of completed agent tasks rather than repeatedly sampling multiple attempts for the same query. This setup isolates the core single-rollout learning problem while keeping the infrastructure close to existing RL training stacks.

We train a Java migration agent on [MigrationBench](https://arxiv.org/pdf/2505.09569). MigrationBench evaluates repository-level code migration success rate from Java 8 to Java 17. It contains 5,102 repositories and provides a curated 300-repository evaluation subset. A task is counted as solved only if the edited project builds under Java 17, its tests pass, and the number of tests is non-decreasing. This last condition matters: the agent must preserve the test suite rather than making the project pass by deleting failing tests.

We use the minimal-migration setting, where the agent has access to shell and editor tools but does not need to upgrade any Java library to the newest available version. The maximal-migration setting adds that stronger library-upgrade requirement, and we leave it as a natural next target. Our agent is implemented with [Strands](https://strandsagents.com/). Table 1 summarizes the performance of the base model and models trained with real-time RL using REINFORCE and the batch-normalized advantage.

| Exp. | Method | Success Rate |
| --- | --- | --- |
| 0 | Base Qwen-Coder-30B | 43% |
| 1 | REINFORCE (batch size 128) | 48.2% |
| **2** | **Batch-normalized (batch size 32)** | **50.8%** |
| **3** | **Batch-normalized (batch size 128)** | **59.2%** |

Table 1: Real-time RL results on MigrationBench under the minimal-migration setting.

With batch normalization, real-time RL improves Qwen3-Coder-30B from **43%** to **59.2%** success rate on MigrationBench, a **16.2% absolute gain** over the baseline model. In contrast, REINFORCE reaches only **48.2%** in the same single-rollout setting with the same batch size, showing that the batch-centered advantage is key to making one-rollout learning effective. Batch size matters too: because the baseline is the mean reward across the batch, a larger batch gives a more reliable baseline, and raising it from 32 to 128 lifts the success rate from **50.8%** to **59.2%**.

The natural worry with single-rollout training is that spending your rollout budget on one attempt per prompt, instead of a group of attempts per prompt, leaves accuracy on the table. Figure 2 shows it doesn't. Here batch-normalized single-rollout training and GRPO consume the **same number of rollouts per training step (32 rollouts)** — the single-rollout run spreads its budget across many distinct prompts with one rollout each, while GRPO concentrates it on fewer prompts with a group of rollouts each. Under that equal budget, single-rollout training tracks GRPO the whole way and ends slightly ahead. In other words, single-rollout RL is not paying an accuracy tax for its efficiency.

<img src="../assets/realtime_rl/val_pass1_yoro_grpo.png" alt="Figure 2: Migration success pass@1 on evaluation set over training, batch-normalized single-rollout training vs GRPO, under the same per-step rollout budget." style="max-width: 80%; display: block; margin: 0 auto;" />

For more details on GRPO training on MigrationBench, read our earlier blog post: [Training a Frontier Java Code Migration Agent with AWS AgentCore Runtime
](post.html?post=agentcore_migrationbench.md).

## What We Learn

These results suggest that real-time RL is a feasible and promising direction for training agents from production-style interactions. The key is not just collecting one trajectory per task, but making that single trajectory informative enough to learn from. Batch normalization does this by turning raw rewards into batch-centered advantages.

This matters especially when rewards are coarse. Some MigrationBench tasks are hard, and the reward is not fine-grained enough to distinguish partial progress, so many rollouts receive zero reward at the end. REINFORCE gets no gradient signal from those trajectories. As shown in Figure 3, roughly 30-50% of the rollouts from REINFORCE (orange) have zero advantage; while nearly every rollout contributes a learning signal with batch normalization (blue).

<img src="../assets/realtime_rl/zero_advantage_fraction.png" alt="Figure 3: Fraction of zero-advantage samples on MigrationBench." style="max-width: 80%; display: block; margin: 0 auto;" />

## The Next Challenge: Stale Rollouts

So far, we have assumed that each completed rollout can be used for training right away. A real continual-learning system is rarely that tidy. While the trainer is updating model weights, rollout workers may continue generating new trajectories from a slightly older policy. By the time those trajectories are used for training, the policy that generated them is no longer the policy we are trying to improve.

![Figure 4: Synchronous real-time training waits for fresh rollout data and discards traces produced during model updates, while asynchronous real-time training keeps collecting trajectories and uses importance sampling when the data becomes stale.](../assets/realtime_rl/sync_async_training.png)

Simply discarding these stale rollouts would waste production interactions and reduce data efficiency. Instead, we borrow the core idea from PPO: use importance sampling to correct the mismatch between the policy that generated the rollout and the policy currently being optimized.

We mock this asynchronous setting in rLLM by allowing rollout workers to keep generating trajectories while the trainer updates the model. We use PPO clip loss to handle stale-policy data. There is also a second, subtler source of off-policyness: rollout engines such as vLLM and training engines such as FSDP can produce slightly different log-probabilities even when running the same model weights. To correct this rollout-training mismatch, we apply **truncated importance sampling (TIS)**, which re-weights the PPO loss by the probability ratio between the training engine and the rollout engine, truncated at a cap. As Figure 5 shows, without TIS training collapses: the success rate falls from the **43%** base model to **4%**. With TIS, training remains stable and reaches a **51.8%** success rate. We refer readers to [Your Efficient RL Framework Secretly Brings You Off-Policy RL Training](https://fengyao.notion.site/off-policy-rl) for a detailed discussion of TIS and rollout-training mismatch.

<img src="../assets/realtime_rl/val_pass1_tis_ablation.png" alt="Figure 5: Migration success pass@1 over training in the asynchronous setting with stale rollouts, with and without TIS. Without TIS, training collapses; with TIS, it remains stable and improves over the base model." style="max-width: 80%; display: block; margin: 0 auto;" />

---

## How Batch Normalization Works

This section covers the mathematical foundations and implementation details behind the batch-normalized advantage.

### Start from REINFORCE

The REINFORCE algorithm forms the foundation of policy gradient methods in reinforcement learning. At its core, REINFORCE optimizes a policy $\pi_\theta$ by maximizing the expected return:

$$
\max_\theta J_{\text{REINFORCE}}(\theta) = \max_\theta E_{y\sim\pi_{\theta}}[R(y)].
$$

The expected return $J_{\text{REINFORCE}}(\theta)$ is maximized by policy gradient method and its gradient is given by:

$$
\nabla_\theta J_{\text{REINFORCE}}(\theta) = E_{y\sim\pi_{\theta}}[R(y)\nabla_\theta\log\pi_\theta(y)].\tag{1}
$$

The key insight of REINFORCE is that by sampling trajectories from the current policy and weighting the log-probability gradients by the trajectory return, we reinforce trajectories in proportion to their reward. However, vanilla REINFORCE suffers from several challenges that make it impractical for continual learning scenarios:

### High variance

The return $R(y)$ can vary significantly across trajectories, leading to unstable gradient estimates, especially when only a single trajectory is available per prompt. A common way to reduce this variance is to subtract a baseline, so the model learns from relative performance rather than raw reward alone. GRPO adopts this idea by subtracting the within-group mean reward. In continual learning, where only one rollout is available per group, we instead use the cross-prompt mean within a training batch to standardize rewards.

### Insufficient learning signal

REINFORCE uses the raw reward as the advantage, so trajectories with zero reward produce zero gradient and contribute nothing to parameter updates. When rewards are frequently zero, this leads to poor sampling efficiency. Batch normalization addresses this by giving each single rollout a batch-centered learning signal.

### Batch-Normalized Advantage

#### Baselines and Advantages

The high variance of the vanilla REINFORCE gradient estimator stems from the fact that trajectory returns $R(y)$ can vary dramatically across different tasks, leading to noisy and unstable gradient updates. A fundamental technique for reducing this variance is to subtract a baseline $b$ from the return when computing policy gradients. The modified gradient estimator becomes:

$$
E_{y\sim\pi_{\theta}}[(R(y)-b)\nabla_\theta\log\pi_\theta(y)]\tag{2}.
$$

The term $b$ is called a baseline and $R(y)-b$ is called the advantage. A standard result in policy gradient methods is that subtracting a baseline leaves the gradient unbiased because $E_{y\sim \pi_\theta}\big[b\nabla_\theta\log\pi_\theta(y)\big] = 0$ for any baseline that does not depend on the sampled rollout $y$ (it may still depend on the task prompt $q$).

#### From a Group Baseline to a Batch Baseline

In GRPO, the baseline is defined per task. Concretely, for a given query $q$, the baseline is the expected reward under the current policy: $b(q) = E_{y\sim\pi_\theta(q)}[R(y)]$. In practice, this expectation is approximated by sampling multiple rollouts $y_1,\dots, y_g$ for the same query $q$, and computing their in-group sample mean. The resulting advantage is therefore group-relative.

In contrast, we define the baseline globally over the task distribution:

$$
b(D) = E_{q\sim D, y\sim\pi_\theta(q)}[R(y)].
$$

Rather than conditioned on a single query, this baseline captures the overall expected reward across all tasks. We estimate it using the cross-task mean of the one rollout sampled for each task in the batch:

$$
\nabla_\theta J(\theta) = E_{y\sim\pi_{\theta}}[(R(y)-b)\nabla_\theta\log\pi_\theta(y)] \approx \frac1{|\mathcal{B}|}\sum_{q\in\mathcal{B}}[R(y_i)-\text{avg}(R)]\nabla_\theta\log\pi_\theta(y_i|q)\tag{3},
$$

where $\mathcal{B}$ is the training batch and $y_i\sim\pi_\theta(\cdot|q_i)$ for each $q_i\in\mathcal{B}$.

![Figure 6: Batch normalization computes one rollout per task, estimates a batch-level reward baseline, and turns each reward into a centered advantage.](../assets/realtime_rl/yoro_population_advantage.png)

From the above equation, the term $A_i = R(y_i) - \text{avg}(R)$ is effectively the advantage estimates, where $\text{avg}(R)$ is computed across different prompts within a training batch. Note that this empirical average $\text{avg}(R)$ does depend on the sampled rollouts $y$ in the batch. This means it is not the exact rollout-independent baseline from the unbiasedness argument above. But in practice, this estimator works well, so we do not need to be overly strict about this distinction. Batch normalization reduces variance by removing common offsets in rewards and rescales the signal to reflect performance within the batch, while also mitigating the zero-reward issue by assigning non-zero advantages to trajectories that are above or below the batch average. As a result, it is better suited for the continual learning setting where only one rollout is available per task.

#### Scaling Normalization

Note that GRPO additionally divides the advantage by the standard deviation of $R(y_i)$. Although some recent work has started to remove this scaling normalization, GRPO still shows that it can be useful in practice. Therefore, we also divide the centered reward by the standard deviation. An implementation is available in rLLM in this [PR](https://github.com/rllm-org/rllm/pull/669).

#### Relation to REINFORCE++

Our implementation is close to [REINFORCE++](https://arxiv.org/pdf/2501.03262), which also uses global normalization, with two differences:

- When group size > 1, REINFORCE++ still uses a within-group mean baseline. In contrast, we use the batch mean regardless of the group size.
- REINFORCE++ incorporates token-level KL loss into the advantage. We do not include KL in the advantage, though it can still be included as a separate term in the policy-gradient loss.

#### Relation to SAO

We're not the only ones betting on single-rollout RL. [SAO](https://arxiv.org/abs/2607.07508), used to train the open GLM-5.2 model, lands on the same design — one rollout per prompt — but for a different reason. SAO is chasing async efficiency: GRPO's group sampling doesn't play well with asynchronous training, so it drops the group. We got here from the real-time RL side: in production, you simply don't get a second rollout to compare against. Different problems, same target. The other difference is where the learning signal comes from — SAO trains a value model to score each rollout, while we skip that entirely and just center rewards against the batch mean, which is what makes this approach easy to drop into an existing training setup.

## When Rollouts Become Stale

Production data can become stale. A user submits a query, the model generates a response, and the resulting trajectory is eventually added to the training queue—but by that time, the trainer may have already updated on earlier completed rollouts. As a result, the policy being optimized no longer matches the one that generated the data.

Rather than discard those trajectories, we treat them as off-policy data from a behavior policy $p$ (typically $\pi_{\theta-K}$, where $K$ is the allowed data staleness) and reuse the standard PPO clip objective, with the batch-normalized advantage $A$ from Equation (3) plugged in:

$$
J_{\text{PPO}}^{\text{CLIP}}(\theta) = E_{y\sim p}\left[\min\left(\frac{\pi_\theta(y)}{p(y)}A, \text{clip}\big(\frac{\pi_\theta(y)}{p(y)}, 1-\epsilon, 1+\epsilon\big)A\right)\right]
$$

The importance ratio $\frac{\pi_\theta(y)}{p(y)}$ corrects for the mismatch between $p$ and $\pi_\theta$, and clipping keeps the update conservative when that ratio grows large.

## Discussion

### Finding 1: Batch Normalization Makes Single-Rollout Data More Informative

MigrationBench rewards take only three values, $\{0, 0.5, 1\}$, so many failed rollouts receive exactly zero reward. Figure 3 shows the practical effect: without centering, roughly 30%-50% of rollouts have zero advantage and contribute no gradient; with batch normalization, nearly every rollout becomes informative. This helps explain why the batch-normalized runs in Table 1 substantially outperform REINFORCE in the real-time RL setting.

### Finding 2: Single-Rollout Training Matches GRPO Under an Equal Rollout Budget

Spending the rollout budget on one attempt per prompt instead of a group of attempts per prompt does not cost accuracy. In Figure 2, both runs consume **32 rollouts per training step**: the batch-normalized run spreads them across 32 distinct prompts with one rollout each, while GRPO concentrates them on fewer prompts with a group of rollouts each. Under that equal budget, the two curves track each other throughout training and the single-rollout run ends slightly ahead.


### Finding 3: TIS Can Stabilize Async Training

Asynchronous real-time RL improves data efficiency by continuing to collect rollouts while the trainer updates the model, but those rollouts can become stale. PPO clip loss handles the stale-policy mismatch, but in our setup it is not enough by itself because rollout-training mismatch remains. As shown in Figure 5, training without TIS collapses to **4%**. Adding TIS corrects this mismatch and recovers stable learning, reaching a **51.8%** success rate.

## References

[1] Liu, Linbo, Xinle Liu, Qiang Zhou, Lin Chen, Yihan Liu, Hoan Nguyen, Behrooz Omidvar-Tehrani et al. "[MigrationBench: Repository-Level Code Migration Benchmark from Java 8](https://arxiv.org/pdf/2505.09569)." Proceedings of the 32nd ACM SIGKDD Conference on Knowledge Discovery and Data Mining (2026).

[2] Shao, Zhihong, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei Zhang et al. "[DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models](https://arxiv.org/abs/2402.03300)." arXiv preprint arXiv:2402.03300 (2024).

[3] Williams, Ronald J. "[Simple Statistical Gradient-Following Algorithms for Connectionist Reinforcement Learning](https://link.springer.com/article/10.1007/BF00992696)." Machine Learning 8, no. 3 (1992): 229-256.

[4] Hu, Jian, Jason Klein Liu, Haotian Xu, and Wei Shen. "[REINFORCE++: Stabilizing Critic-Free Policy Optimization with Global Advantage Normalization](https://arxiv.org/abs/2501.03262)." arXiv preprint arXiv:2501.03262 (2025).

[5] Schulman, John, Filip Wolski, Prafulla Dhariwal, Alec Radford, and Oleg Klimov. "[Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347)." arXiv preprint arXiv:1707.06347 (2017).

[6] Yao, Feng, Liyuan Liu, Dinghuai Zhang, Chengyu Dong, Jingbo Shang, and Jianfeng Gao. "[Your Efficient RL Framework Secretly Brings You Off-Policy RL Training](https://fengyao.notion.site/off-policy-rl)." Feng Yao's Notion, August 2025.

[7] Bryan Lu, Youzhi Luo, Linbo Liu, Panpan Xu, Anoop Deoras, Sijun Tan, Kyle Montgomery, Tianhao Wu, Ion Stoica. "Training a Frontier Java Code Migration Agent with AWS AgentCore Runtime". rLLM Blog (June 2026). [https://rllm-project.com/post.html?post=agentcore_migrationbench.md](https://rllm-project.com/post.html?post=agentcore_migrationbench.md)

[8] Hou, Zhenyu, Yujiang Li, Jie Tang, and Yuxiao Dong. "[Single-Rollout Asynchronous Optimization for Agentic Reinforcement Learning](https://arxiv.org/abs/2607.07508)." arXiv preprint arXiv:2607.07508 (2026).
