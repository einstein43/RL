"""
Run Q-Learning experiments.

Usage:
    python experiments/run_experiment.py --experiment 1
    python experiments/run_experiment.py --experiment all
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import numpy as np
import pickle
from datetime import datetime

from src.environment import create_environment
from src.q_learning import QLearningAgent
from src.trainer import QLearningTrainer
from src.visualization import plot_learning_curves, plot_q_table_heatmap, plot_single_experiment

from experiments.config import (
    EXPERIMENT_1_CONFIGS,
    EXPERIMENT_2_CONFIGS,
    EXPERIMENT_3_CONFIGS,
    EXPERIMENT_4_CONFIGS,
    EXPERIMENT_5_CONFIGS,
    ENV_CONFIGS
)


def run_single_experiment(exp_name: str, config: dict, env_config: dict, save_results: bool = True) -> dict:
    print(f"\n{'='*80}")
    print(f"Running experiment: {exp_name}")
    print(f"{'='*80}")
    print("Configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    print()
    
    np.random.seed(config.get('random_seed', 42))
    
    env = create_environment(**env_config)
    
    agent = QLearningAgent(
        n_states=env.n_states,
        n_actions=env.n_actions,
        learning_rate=config.get('learning_rate', 0.1),
        discount_factor=config.get('discount_factor', 0.99),
        exploration_strategy=config.get('exploration_strategy', 'epsilon_greedy'),
        epsilon_start=config.get('epsilon_start', 1.0),
        epsilon_end=config.get('epsilon_end', 0.01),
        epsilon_decay=config.get('epsilon_decay', 0.995),
        temperature=config.get('temperature', 1.0),
        temperature_decay=config.get('temperature_decay', 0.995),
        temperature_min=config.get('temperature_min', 0.01)
    )
    
    trainer = QLearningTrainer(
        env=env,
        agent=agent,
        n_episodes=config.get('n_episodes', 10000),
        max_steps=config.get('max_steps', 100),
        eval_frequency=config.get('eval_frequency', 100),
        eval_episodes=config.get('eval_episodes', 100),
        verbose=True
    )
    
    metrics = trainer.train()
    metrics['eval_frequency'] = config.get('eval_frequency', 100)
    
    if save_results:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_dir = f"results/{exp_name}_{timestamp}"
        os.makedirs(results_dir, exist_ok=True)
        
        with open(f"{results_dir}/metrics.pkl", 'wb') as f:
            pickle.dump(metrics, f)
        
        np.save(f"{results_dir}/q_table.npy", agent.q_table)
        
        with open(f"{results_dir}/config.txt", 'w') as f:
            f.write(f"Experiment: {exp_name}\n")
            f.write(f"Timestamp: {timestamp}\n\n")
            f.write("Configuration:\n")
            for key, value in config.items():
                f.write(f"  {key}: {value}\n")
            f.write("\nEnvironment Configuration:\n")
            for key, value in env_config.items():
                f.write(f"  {key}: {value}\n")
        
        print(f"\nResults saved to: {results_dir}")
    
    env.close()
    
    return {
        'metrics': metrics,
        'q_table': agent.q_table,
        'config': config,
        'env_config': env_config
    }


def run_experiment_group(experiment_num: int):
    # Select configs
    if experiment_num == 1:
        configs = EXPERIMENT_1_CONFIGS
        exp_group_name = "Hyperparameter_Tuning"
    elif experiment_num == 2:
        configs = EXPERIMENT_2_CONFIGS
        exp_group_name = "Slippery_Comparison"
    elif experiment_num == 3:
        configs = EXPERIMENT_3_CONFIGS
        exp_group_name = "Reward_Shaping"
    elif experiment_num == 4:
        configs = EXPERIMENT_4_CONFIGS
        exp_group_name = "Exploration_Strategies"
    elif experiment_num == 5:
        configs = EXPERIMENT_5_CONFIGS
        exp_group_name = "Discount_Factor"
    else:
        raise ValueError(f"Unknown experiment number: {experiment_num}")
    
    env_config = ENV_CONFIGS['frozen_lake_4x4'].copy()
    
    results = {}
    for exp_name, config in configs.items():
        exp_env_config = env_config.copy()
        if 'is_slippery' in config:
            exp_env_config['is_slippery'] = config['is_slippery']
        if 'step_penalty' in config:
            exp_env_config['step_penalty'] = config['step_penalty']
        if 'hole_penalty' in config:
            exp_env_config['hole_penalty'] = config['hole_penalty']
        
        result = run_single_experiment(exp_name, config, exp_env_config)
        results[exp_name] = result
    
    # Create plots
    print(f"\n{'='*80}")
    print(f"Creating comparison plots for {exp_group_name}")
    print(f"{'='*80}\n")
    
    metrics_dict = {name: result['metrics'] for name, result in results.items()}
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_path = f"plots/{exp_group_name}_{timestamp}_comparison.png"
    
    plot_learning_curves(
        metrics_dict,
        save_path=plot_path,
        title=f"{exp_group_name.replace('_', ' ')} - Comparison"
    )
    
    # Q-table for best experiment
    best_exp_name = max(results.keys(), 
                        key=lambda k: np.mean(results[k]['metrics']['eval_rewards'][-5:]))
    
    print(f"\nBest performing experiment: {best_exp_name}")
    
    q_table_path = f"plots/{exp_group_name}_{timestamp}_qtable_{best_exp_name}.png"
    plot_q_table_heatmap(
        results[best_exp_name]['q_table'],
        save_path=q_table_path,
        title=f"Q-Table Heatmap - {best_exp_name}"
    )


def main():
    parser = argparse.ArgumentParser(description='Run Q-Learning experiments')
    parser.add_argument(
        '--experiment',
        type=str,
        required=True,
        help='Experiment number (1-5) or "all"'
    )
    
    args = parser.parse_args()
    
    if args.experiment.lower() == 'all':
        for i in range(1, 6):
            run_experiment_group(i)
    else:
        try:
            exp_num = int(args.experiment)
            if exp_num < 1 or exp_num > 5:
                print("Error: Experiment number must be between 1 and 5")
                sys.exit(1)
            run_experiment_group(exp_num)
        except ValueError:
            print("Error: Experiment must be a number (1-5) or 'all'")
            sys.exit(1)


if __name__ == '__main__':
    main()
