"""
Master script to run all Q-Learning experiments and generate comprehensive report.

This script will:
1. Run all 5 experiment groups
2. Generate all comparison plots
3. Generate Q-table heatmaps
4. Save all results to results/ and plots/ folders

Usage:
    python run_all_experiments.py
"""
import sys
import os

import numpy as np
import pickle
from datetime import datetime

from src.environment import create_environment
from src.q_learning import QLearningAgent
from src.trainer import QLearningTrainer
from src.visualization import (
    plot_learning_curves,
    plot_q_table_heatmap,
    plot_single_experiment
)

from experiments.config import (
    EXPERIMENT_1_CONFIGS,
    EXPERIMENT_2_CONFIGS,
    EXPERIMENT_3_CONFIGS,
    EXPERIMENT_4_CONFIGS,
    EXPERIMENT_5_CONFIGS,
    ENV_CONFIGS
)


def run_single_experiment(exp_name: str, config: dict, env_config: dict, save_results: bool = True) -> dict:
    """Run a single experiment configuration."""
    print(f"\n{'='*80}")
    print(f"Running: {exp_name}")
    print(f"{'='*80}")
    print("Config:")
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
        
        print(f"Results saved to: {results_dir}")
    
    env.close()
    
    return {
        'metrics': metrics,
        'q_table': agent.q_table,
        'config': config,
        'env_config': env_config,
        'exp_name': exp_name
    }


def run_experiment_group(experiment_num: int, configs: dict, group_name: str):
    """Run a group of related experiments and create comparison plots."""
    print(f"\n{'#'*80}")
    print(f"# EXPERIMENT GROUP {experiment_num}: {group_name}")
    print(f"{'#'*80}\n")
    
    env_config = ENV_CONFIGS['frozen_lake_4x4'].copy()
    
    results = {}
    for exp_name, config in configs.items():
        # Override env config based on experiment settings
        exp_env_config = env_config.copy()
        if 'is_slippery' in config:
            exp_env_config['is_slippery'] = config['is_slippery']
        if 'step_penalty' in config:
            exp_env_config['step_penalty'] = config['step_penalty']
        if 'hole_penalty' in config:
            exp_env_config['hole_penalty'] = config['hole_penalty']
        
        result = run_single_experiment(exp_name, config, exp_env_config)
        results[exp_name] = result
    
    # Create comparison plots
    print(f"\nCreating comparison plots for {group_name}...")
    metrics_dict = {name: result['metrics'] for name, result in results.items()}
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs("plots", exist_ok=True)
    
    # Main comparison plot
    plot_path = f"plots/Fig{experiment_num}_{group_name}_{timestamp}.png"
    plot_learning_curves(
        metrics_dict,
        save_path=plot_path,
        title=f"Experiment {experiment_num}: {group_name.replace('_', ' ')}",
        window_size=100
    )
    print(f"Saved comparison plot: {plot_path}")
    
    # Generate Q-table heatmap for best performing experiment
    best_exp_name = max(results.keys(), 
                       key=lambda k: results[k]['metrics']['eval_success_rates'][-1])
    best_result = results[best_exp_name]
    
    qtable_path = f"plots/Fig{experiment_num}_{group_name}_qtable_{timestamp}.png"
    
    # Get env config for title
    env_cfg = best_result['env_config']
    slippery_text = "Slippery" if env_cfg.get('is_slippery', True) else "Not Slippery"
    config = best_result['config']
    
    qtable_title = (f"Q-Table Heatmap: {best_exp_name}\n"
                   f"({slippery_text}, α={config['learning_rate']}, "
                   f"γ={config['discount_factor']})")
    
    plot_q_table_heatmap(
        best_result['q_table'],
        env_shape=(4, 4),
        save_path=qtable_path,
        title=qtable_title
    )
    print(f"Saved Q-table heatmap: {qtable_path}")
    
    return results


def main():
    """Run all experiments."""
    print("\n" + "="*80)
    print("STARTING ALL Q-LEARNING EXPERIMENTS")
    print("="*80)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("This will take approximately 30-60 minutes...")
    print()
    
    all_results = {}
    
    # Experiment 1: Learning Rate
    results_1 = run_experiment_group(
        1, 
        EXPERIMENT_1_CONFIGS, 
        "Learning_Rate_Comparison"
    )
    all_results['exp1'] = results_1
    
    # Experiment 2: Slippery vs Non-Slippery
    results_2 = run_experiment_group(
        2,
        EXPERIMENT_2_CONFIGS,
        "Slippery_Comparison"
    )
    all_results['exp2'] = results_2
    
    # Experiment 3: Reward Shaping
    results_3 = run_experiment_group(
        3,
        EXPERIMENT_3_CONFIGS,
        "Reward_Shaping"
    )
    all_results['exp3'] = results_3
    
    # Experiment 4: Exploration Strategies
    results_4 = run_experiment_group(
        4,
        EXPERIMENT_4_CONFIGS,
        "Exploration_Strategies"
    )
    all_results['exp4'] = results_4
    
    # Experiment 5: Discount Factor
    results_5 = run_experiment_group(
        5,
        EXPERIMENT_5_CONFIGS,
        "Discount_Factor_Comparison"
    )
    all_results['exp5'] = results_5
    
    # Save summary
    print("\n" + "="*80)
    print("ALL EXPERIMENTS COMPLETED!")
    print("="*80)
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\nResults saved to: results/")
    print("Plots saved to: plots/")
    print("\nNext steps:")
    print("1. Check the plots/ folder for all generated figures")
    print("2. Update your research_report.md with the plots and results")
    print("3. Include figure numbers and detailed captions")
    print()
    
    # Save experiment summary
    summary_path = "results/experiment_summary.txt"
    with open(summary_path, 'w') as f:
        f.write("Q-Learning Experiments Summary\n")
        f.write("="*80 + "\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        for exp_key, exp_results in all_results.items():
            f.write(f"\n{exp_key.upper()}:\n")
            f.write("-"*40 + "\n")
            for exp_name, result in exp_results.items():
                final_success = result['metrics']['eval_success_rates'][-1]
                f.write(f"  {exp_name}: {final_success:.2%} success rate\n")
    
    print(f"Summary saved to: {summary_path}")


if __name__ == "__main__":
    main()
