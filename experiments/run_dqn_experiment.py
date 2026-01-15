"""
Run DQN experiments on CartPole environment.
"""

import os
import sys
import argparse
from datetime import datetime

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import gymnasium as gym
from src.dqn_agent import DQNAgent
from src.dqn_trainer import DQNTrainer
from experiments.dqn_config import get_config, list_experiments


def run_experiment(experiment_name: str, verbose: bool = True):
    """
    Run a DQN experiment.
    
    Args:
        experiment_name: Name of experiment from dqn_config
        verbose: Whether to print progress
    """
    # Get configuration
    config = get_config(experiment_name)
    agent_config = config['agent_config']
    
    if verbose:
        print("="*60)
        print(f"Running DQN Experiment: {experiment_name}")
        print("="*60)
        print(f"Environment: {config['environment']}")
        print(f"Learning Rate: {agent_config['learning_rate']}")
        print(f"Discount Factor (γ): {agent_config['discount_factor']}")
        print(f"Epsilon Decay: {agent_config['epsilon_decay']}")
        print(f"Network Architecture: {agent_config['hidden_dims']}")
        print(f"Batch Size: {agent_config['batch_size']}")
        print(f"Buffer Capacity: {agent_config['buffer_capacity']}")
        print("="*60 + "\n")
    
    # Create environment to get dimensions
    env = gym.make(config['environment'])
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    env.close()
    
    # Create agent
    agent = DQNAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        **agent_config
    )
    
    # Create save directory for this experiment
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = os.path.join("results", "dqn_experiments", f"{experiment_name}_{timestamp}")
    os.makedirs(save_dir, exist_ok=True)
    
    # Create trainer
    trainer = DQNTrainer(
        env_name=config['environment'],
        agent=agent,
        save_dir=save_dir
    )
    
    # Train
    training_history = trainer.train(
        n_episodes=config['n_episodes'],
        eval_frequency=config['eval_frequency'],
        save_frequency=config['save_frequency'],
        target_reward=config['target_reward'],
        early_stop_episodes=config['early_stop_episodes'],
        verbose=verbose
    )
    
    # Close trainer
    trainer.close()
    
    return training_history, save_dir


def main():
    """Main function to run experiments from command line."""
    parser = argparse.ArgumentParser(description="Run DQN experiments on CartPole")
    parser.add_argument(
        'experiment',
        type=str,
        nargs='?',
        default='baseline',
        help='Name of experiment to run (or "list" to see all)'
    )
    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Suppress verbose output'
    )
    
    args = parser.parse_args()
    
    # List experiments
    if args.experiment == 'list':
        print("Available experiments:")
        for exp in list_experiments():
            print(f"  - {exp}")
        return
    
    # Run experiment
    try:
        run_experiment(args.experiment, verbose=not args.quiet)
    except KeyError as e:
        print(f"Error: {e}")
        print("\nAvailable experiments:")
        for exp in list_experiments():
            print(f"  - {exp}")
        sys.exit(1)


if __name__ == "__main__":
    main()
