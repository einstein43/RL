# Deep Q-Network (DQN) for CartPole - Part 2

This directory contains the implementation of Deep Q-Network (DQN) for solving the CartPole environment (Part 2 of the assignment).

## 📁 Project Structure

```
RL/
├── src/
│   ├── dqn_agent.py       # DQN agent with neural network
│   ├── replay_buffer.py   # Experience replay buffer
│   ├── dqn_trainer.py     # Training loop and evaluation
│   ├── visualization.py   # Plotting functions
│   └── ...
├── experiments/
│   ├── dqn_config.py           # Experiment configurations
│   ├── run_dqn_experiment.py   # Run single experiment
│   └── ...
├── results/
│   └── dqn_experiments/   # Saved models and results
├── run_all_dqn_experiments.py  # Run multiple experiments
└── requirements.txt       # Dependencies
```

## 🚀 Quick Start

### 1. Install Dependencies

First, install PyTorch and other requirements:

```bash
# Navigate to RL directory
cd "RL"

# Install dependencies
pip install -r requirements.txt
```

### 2. Run Baseline Experiment

```bash
# Run baseline DQN experiment on CartPole
python experiments/run_dqn_experiment.py baseline
```

### 3. Run Multiple Experiments

```bash
# Run all hyperparameter experiments
python run_all_dqn_experiments.py
```

## 🧪 Available Experiments

List all available experiments:
```bash
python experiments/run_dqn_experiment.py list
```

Available configurations:
- `baseline` - Standard DQN configuration
- `learning_rate_low` - Lower learning rate (0.0001)
- `learning_rate_high` - Higher learning rate (0.01)
- `gamma_low` - Lower discount factor (0.95)
- `gamma_high` - Higher discount factor (0.999)
- `network_small` - Smaller network (64x64)
- `network_large` - Larger network (256x256)
- `buffer_small` - Smaller replay buffer (1000)
- `buffer_large` - Larger replay buffer (50000)
- `batch_size_small` - Smaller batch size (32)
- `batch_size_large` - Larger batch size (128)
- `fast_exploration_decay` - Faster epsilon decay
- `slow_exploration_decay` - Slower epsilon decay

## 🎯 CartPole Environment

**State Space**: 4 continuous values
- Cart position
- Cart velocity
- Pole angle
- Pole angular velocity

**Action Space**: 2 discrete actions
- 0: Push cart left
- 1: Push cart right

**Success Criterion**: Reach 500 steps (CartPole-v1)

**Termination**:
- Pole angle > 12 degrees
- Cart position > 2.4 units
- Episode length reaches 500

## 🧠 DQN Algorithm

Key features implemented:
1. **Neural Network**: Multi-layer perceptron for Q-value approximation
2. **Experience Replay**: Break correlation between consecutive samples
3. **Target Network**: Stabilize training with periodic updates
4. **Epsilon-Greedy**: Balance exploration vs exploitation

## 📊 Visualizing Results

After training, results are saved to `results/dqn_experiments/<experiment_name>_<timestamp>/`

Each experiment saves:
- Model checkpoints (`.pt` files)
- Training history (`.npz` file)
- Configuration (`.json` file)

To visualize results, use the visualization functions:

```python
import numpy as np
from src.visualization import plot_dqn_training_curves

# Load training history
data = np.load('results/dqn_experiments/<experiment>/training_history_<timestamp>.npz')

# Plot
plot_dqn_training_curves(
    episode_rewards=data['episode_rewards'],
    episode_lengths=data['episode_lengths'],
    losses=data['losses'],
    epsilons=data['epsilons'],
    save_path='plots/dqn_training.png'
)
```

## 🎓 Key Hyperparameters

| Parameter | Symbol | Default | Description |
|-----------|--------|---------|-------------|
| Learning Rate | α | 0.001 | Step size for optimizer |
| Discount Factor | γ | 0.99 | Future reward importance |
| Epsilon Start | ε₀ | 1.0 | Initial exploration rate |
| Epsilon End | ε_min | 0.01 | Minimum exploration rate |
| Epsilon Decay | - | 0.995 | Exploration decay rate |
| Batch Size | - | 64 | Training batch size |
| Buffer Capacity | - | 10000 | Replay buffer size |
| Target Update | - | 10 | Episodes between target updates |

## 📈 Expected Results

A well-trained DQN agent should:
- Converge within 200-500 episodes
- Achieve 100% success rate (500 steps)
- Show smooth learning curves after initial exploration
- Maintain performance in evaluation

## 🔧 Troubleshooting

**Issue**: Training doesn't converge
- Try increasing buffer capacity
- Reduce learning rate
- Increase network size

**Issue**: Training is unstable
- Reduce learning rate
- Decrease batch size
- Check epsilon decay rate

**Issue**: Agent diverges after convergence
- This is expected - stop training when target is reached
- Save checkpoints frequently

## 📝 Assignment Requirements

For Part 2, you need to:
1. ✅ Implement DQN with neural network
2. ✅ Train on CartPole environment
3. ✅ Achieve 500-step success criterion
4. 📊 Create plots showing training progress
5. 📄 Write research report with:
   - Methods used
   - Experimental results
   - Analysis and discussion
   - Link to GitLab code

## 🎯 Next Steps

1. Run baseline experiment
2. Test different hyperparameters
3. Create comparison plots
4. Analyze results
5. Write report

## 💡 Tips

- Use tensorboard for real-time monitoring (optional)
- Save checkpoints frequently
- Evaluate regularly during training
- Compare multiple configurations
- Document your findings

Good luck with Part 2! 🚀
