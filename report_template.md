# Q-Learning on FrozenLake - Research Report

Alexander van Heteren  
January 2026

I used Claude Sonnet 4.5 to help with the code structure and analyzing some of the results.

## Introduction

This project is about testing Q-learning on the FrozenLake environment from Gymnasium. The main goal was to figure out how different hyperparameters affect how well the agent learns. I also wanted to understand the exploration-exploitation tradeoff better because that's pretty important in reinforcement learning.

FrozenLake is a grid world where you need to get from start to goal without falling in holes. The environment can be slippery (stochastic) or not slippery (deterministic), which makes it good for testing different scenarios.

## Methodology

I implemented a tabular Q-learning agent that learns by trying different actions and updating its Q-values. The update rule I used is:

Q(s,a) ← Q(s,a) + α[r + γ max Q(s',a') - Q(s,a)]

Where:
- α = learning rate (how much new info matters)
- γ = discount factor (how much future rewards matter)
- r = reward

For all experiments I used:
- Environment: FrozenLake 4x4
- Training: 10,000 episodes
- Evaluation: every 100 episodes on 100 test episodes
- Max steps per episode: 100

I ran 5 different experiment groups to test different things:
1. Learning rates (α = 0.1, 0.3, 0.7)
2. Slippery vs non-slippery environment
3. Reward shaping (adding penalties)
4. Exploration strategies (epsilon-greedy vs Boltzmann)
5. Discount factors (γ = 0.9, 0.95, 0.99)

## Results

### Experiment 1: Learning Rate Comparison

**Figure 1**: Learning curves for different learning rates

![Figure 1](plots/Fig1_Learning_Rate_Comparison_[TIMESTAMP].png)

*Learning rate comparison on FrozenLake 4x4 (slippery=True). Settings: γ=0.99, ε-greedy (ε: 1.0→0.01, decay=0.995), 10,000 episodes.*

Results:
- α=0.1: Final success rate = [XX]%, converged around episode [XXXX]
- α=0.3: Final success rate = [XX]%, converged around episode [XXXX]  
- α=0.7: Final success rate = [XX]%, [did it converge?]

The low learning rate (0.1) worked best and was most stable. Medium (0.3) also worked okay but was less stable. High learning rate (0.7) completely failed to learn anything - the success rate stayed near 0%. This makes sense because when the learning rate is too high, the Q-values change too much and the agent can't settle on good values.

**Figure 2**: Q-table heatmap for best learning rate

![Figure 2](plots/Fig1_Learning_Rate_Comparison_qtable_[TIMESTAMP].png)

*Q-table visualization showing state values and optimal policy. Best configuration from Experiment 1.*

### Experiment 2: Slippery vs Non-Slippery

**Figure 3**: Comparison of stochastic vs deterministic environment

![Figure 3](plots/Fig2_Slippery_Comparison_[TIMESTAMP].png)

*Slippery comparison. Settings: α=0.1, γ=0.99, ε-greedy, 10,000 episodes.*

Results:
- Slippery ON: Final success rate = [XX]%
- Slippery OFF: Final success rate = [XX]%

The non-slippery version (deterministic) learned way faster and got better results. This is obvious because when the environment is deterministic, the agent can actually learn what each action does. With slippery on, sometimes you do an action and end up somewhere random, which makes learning harder and slower.

**Figure 4**: Q-table for non-slippery environment

![Figure 4](plots/Fig2_Slippery_Comparison_qtable_[TIMESTAMP].png)

*Q-table for deterministic environment showing clearer optimal paths.*

### Experiment 3: Reward Shaping

**Figure 5**: Effect of different reward penalties

![Figure 5](plots/Fig3_Reward_Shaping_[TIMESTAMP].png)

*Reward shaping experiments. Settings: α=0.1, γ=0.99, ε-greedy, slippery=True.*

Results:
- No shaping: Final success rate = [XX]%
- Small step penalty (-0.01): Final success rate = [XX]%
- Large hole penalty (-1.0): Final success rate = [XX]%

Adding a small penalty for each step (-0.01) made the agent learn faster because it encourages finding shorter paths. The hole penalty didn't help as much as I thought it would. Maybe the penalty needs to be tuned differently or maybe falling in a hole already teaches the agent enough.

### Experiment 4: Exploration Strategies

**Figure 6**: Epsilon-greedy vs Boltzmann exploration

![Figure 6](plots/Fig4_Exploration_Strategies_[TIMESTAMP].png)

*Comparison of exploration methods. Settings: α=0.1, γ=0.99, slippery=True.*

Results:
- Epsilon-greedy: Final success rate = [XX]%
- Boltzmann (high temp): Final success rate = [XX]%
- Boltzmann (low temp): Final success rate = [XX]%

Epsilon-greedy was more reliable and stable. The learning curve was smoother. Boltzmann with high temperature explored a lot at first which was good, but sometimes it got stuck. Both ended up with similar final performance though. For FrozenLake I'd say epsilon-greedy is better.

**Figure 7**: Q-table comparison between exploration methods

![Figure 7](plots/Fig4_Exploration_Strategies_qtable_[TIMESTAMP].png)

*Q-table from best exploration strategy.*

### Experiment 5: Discount Factor

**Figure 8**: Impact of discount factor on learning

![Figure 8](plots/Fig5_Discount_Factor_Comparison_[TIMESTAMP].png)

*Discount factor comparison. Settings: α=0.1, ε-greedy, slippery=True.*

Results:
- γ=0.9: Final success rate = [XX]%
- γ=0.95: Final success rate = [XX]%
- γ=0.99: Final success rate = [XX]%

Higher discount factors (0.99) worked better. This makes sense for FrozenLake because the goal is multiple steps away, so you need to value future rewards. With γ=0.9 the agent is more short-sighted and doesn't plan ahead as well.

**Figure 9**: Q-table for high discount factor

![Figure 9](plots/Fig5_Discount_Factor_Comparison_qtable_[TIMESTAMP].png)

*Q-table showing value propagation with γ=0.99.*

## Analysis of Q-Table and Craving

Looking at the Q-table heatmaps, you can see clear "craving" behavior. States that are close to the goal have much higher Q-values (brighter colors in the heatmap). The optimal policy arrows show the agent learned to navigate around the holes toward the goal.

States next to holes have lower Q-values which shows the agent learned those are dangerous. The value function shows how rewards propagate backwards from the goal to earlier states.

## Discussion

Here's what I learned from all these experiments:

**Learning rate matters a lot.** Too high and it doesn't learn at all. Too low might work but could be slower. α=0.1 was the sweet spot for FrozenLake.

**Stochastic environments are harder.** The slippery version makes everything more variable and slower to learn. But it's more realistic for many problems.

**Reward shaping can help.** Small penalties for steps made learning faster. But you have to be careful not to make penalties too big or it might mess things up.

**Epsilon-greedy is solid.** It's simple and works well. Boltzmann is more complex and didn't really perform better for this task.

**Discount factor affects planning.** For tasks where the goal is far away, you need a high discount factor (0.99) so the agent values future rewards enough to plan ahead.

The moving average plots (100 episode window) made it way easier to see the actual learning trends instead of all the noise from individual episodes.

## Conclusions

Q-learning works pretty well on FrozenLake when you tune the hyperparameters right. The key things are:
- Use a reasonable learning rate (not too high)
- Higher discount factors for multi-step problems  
- Add small penalties to encourage efficient paths
- Epsilon-greedy exploration is reliable

The agent successfully learned to navigate to the goal and avoid holes. The Q-table visualizations clearly show it learned a good value function with high values near the goal.

Future work could test:
- Larger grids (8x8)
- Different epsilon decay schedules
- Deep Q-learning with neural networks instead of tables
- Prioritized experience replay (the optional assignment part)

## Code

All code is available on GitHub: https://github.com/einstein43/RL

The main files are:
- `src/q_learning.py` - Q-learning agent implementation
- `src/trainer.py` - Training loop
- `src/visualization.py` - Plotting functions
- `experiments/config.py` - Experiment configurations
- `run_all_experiments.py` - Master script to run everything

