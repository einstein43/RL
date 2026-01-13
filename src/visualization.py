import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional
import os


def plot_learning_curves(
    metrics_dict: Dict[str, Dict[str, List[float]]],
    save_path: Optional[str] = None,
    title: str = "Learning Curves",
    window_size: int = 100
):
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(title, fontsize=16)
    
    # Training rewards
    ax = axes[0, 0]
    for exp_name, metrics in metrics_dict.items():
        rewards = metrics['training_rewards']
        moving_avg = _moving_average(rewards, window_size)
        episodes = range(len(moving_avg))
        ax.plot(episodes, moving_avg, label=exp_name, alpha=0.8)
    
    ax.set_xlabel('Episode')
    ax.set_ylabel(f'Average Reward (window={window_size})')
    ax.set_title('Training Rewards')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Evaluation rewards
    ax = axes[0, 1]
    for exp_name, metrics in metrics_dict.items():
        eval_rewards = metrics['eval_rewards']
        eval_episodes = np.arange(len(eval_rewards)) * metrics.get('eval_frequency', 100)
        ax.plot(eval_episodes, eval_rewards, label=exp_name, marker='o', alpha=0.8)
    
    ax.set_xlabel('Episode')
    ax.set_ylabel('Evaluation Reward')
    ax.set_title('Evaluation Performance')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Success rate
    ax = axes[1, 0]
    for exp_name, metrics in metrics_dict.items():
        success_rates = metrics['eval_success_rates']
        eval_episodes = np.arange(len(success_rates)) * metrics.get('eval_frequency', 100)
        ax.plot(eval_episodes, success_rates, label=exp_name, marker='s', alpha=0.8)
    
    ax.set_xlabel('Episode')
    ax.set_ylabel('Success Rate')
    ax.set_title('Success Rate During Evaluation')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Exploration value
    ax = axes[1, 1]
    for exp_name, metrics in metrics_dict.items():
        exploration = metrics['exploration_values']
        episodes = range(len(exploration))
        ax.plot(episodes, exploration, label=exp_name, alpha=0.8)
    
    ax.set_xlabel('Episode')
    ax.set_ylabel('Exploration Value (ε or T)')
    ax.set_title('Exploration Parameter Decay')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved learning curves to {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_q_table_heatmap(
    q_table: np.ndarray,
    env_shape: tuple = (4, 4),
    save_path: Optional[str] = None,
    title: str = "Q-Table Heatmap with Optimal Policy",
    action_symbols: List[str] = ['←', '↓', '→', '↑']
):
    n_rows, n_cols = env_shape
    n_states = n_rows * n_cols
    
    fig, axes = plt.subplots(1, q_table.shape[1] + 1, figsize=(20, 4))
    fig.suptitle(title, fontsize=16)
    
    # Plot Q-values for each action
    for action in range(q_table.shape[1]):
        ax = axes[action]
        q_values = q_table[:, action].reshape(env_shape)
        
        sns.heatmap(
            q_values,
            annot=True,
            fmt='.2f',
            cmap='YlOrRd',
            ax=ax,
            cbar=True,
            square=True
        )
        ax.set_title(f'Action: {action_symbols[action]}')
        ax.set_xlabel('Column')
        ax.set_ylabel('Row')
    
    # Optimal policy
    ax = axes[-1]
    policy = np.argmax(q_table, axis=1).reshape(env_shape)
    max_q_values = np.max(q_table, axis=1).reshape(env_shape)
    
    sns.heatmap(
        max_q_values,
        annot=False,
        cmap='YlGnBu',
        ax=ax,
        cbar=True,
        square=True
    )
    
    # Add arrows
    for i in range(n_rows):
        for j in range(n_cols):
            action = policy[i, j]
            ax.text(
                j + 0.5,
                i + 0.5,
                action_symbols[action],
                ha='center',
                va='center',
                fontsize=20,
                color='black',
                weight='bold'
            )
    
    ax.set_title('Optimal Policy')
    ax.set_xlabel('Column')
    ax.set_ylabel('Row')
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved Q-table heatmap to {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_single_experiment(
    metrics: Dict[str, List[float]],
    save_path: Optional[str] = None,
    title: str = "Experiment Results",
    window_size: int = 100
):
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(title, fontsize=16)
    
    # Training rewards
    ax = axes[0, 0]
    rewards = metrics['training_rewards']
    moving_avg = _moving_average(rewards, window_size)
    ax.plot(rewards, alpha=0.3, label='Raw')
    ax.plot(moving_avg, linewidth=2, label=f'Moving Avg (window={window_size})')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Reward')
    ax.set_title('Training Rewards')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Steps per episode
    ax = axes[0, 1]
    steps = metrics['training_steps']
    moving_avg_steps = _moving_average(steps, window_size)
    ax.plot(steps, alpha=0.3, label='Raw')
    ax.plot(moving_avg_steps, linewidth=2, label=f'Moving Avg (window={window_size})')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Steps')
    ax.set_title('Steps per Episode')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Evaluation performance
    ax = axes[1, 0]
    eval_rewards = metrics['eval_rewards']
    eval_episodes = np.arange(len(eval_rewards)) * metrics.get('eval_frequency', 100)
    ax.plot(eval_episodes, eval_rewards, marker='o', linewidth=2)
    ax.set_xlabel('Episode')
    ax.set_ylabel('Evaluation Reward')
    ax.set_title('Evaluation Performance')
    ax.grid(True, alpha=0.3)
    
    # Exploration decay
    ax = axes[1, 1]
    exploration = metrics['exploration_values']
    ax.plot(exploration, linewidth=2)
    ax.set_xlabel('Episode')
    ax.set_ylabel('Exploration Value')
    ax.set_title('Exploration Parameter')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved experiment results to {save_path}")
    else:
        plt.show()
    
    plt.close()


def _moving_average(data: List[float], window_size: int) -> np.ndarray:
    data = np.array(data)
    if len(data) < window_size:
        return data
    
    cumsum = np.cumsum(np.insert(data, 0, 0))
    moving_avg = (cumsum[window_size:] - cumsum[:-window_size]) / window_size
    
    prefix = [np.mean(data[:i+1]) for i in range(min(window_size-1, len(data)))]
    
    return np.concatenate([prefix, moving_avg])
