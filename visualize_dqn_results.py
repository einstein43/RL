"""
Utility script to visualize results from completed DQN experiments.
"""

import os
import sys
import glob
import argparse
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.visualization import plot_dqn_training_curves, plot_dqn_comparison


def find_latest_experiment(experiment_name: str, base_dir: str = "results/dqn_experiments"):
    """Find the most recent result directory for an experiment."""
    pattern = os.path.join(base_dir, f"{experiment_name}_*")
    dirs = glob.glob(pattern)
    if not dirs:
        return None
    # Sort by timestamp (last part of directory name)
    return max(dirs, key=lambda x: os.path.basename(x).split('_')[-1])


def load_experiment_data(experiment_name: str):
    """Load training history for an experiment."""
    exp_dir = find_latest_experiment(experiment_name)
    if not exp_dir:
        print(f"Warning: No results found for experiment '{experiment_name}'")
        return None
    
    # Find training history file
    history_files = glob.glob(os.path.join(exp_dir, "training_history_*.npz"))
    if not history_files:
        print(f"Warning: No training history found in {exp_dir}")
        return None
    
    history_file = history_files[0]
    data = np.load(history_file)
    
    return {
        'episode_rewards': data['episode_rewards'],
        'episode_lengths': data['episode_lengths'],
        'losses': data['losses'],
        'epsilons': data['epsilons'],
        'experiment_dir': exp_dir
    }


def plot_single_experiment(experiment_name: str, output_dir: str = "plots"):
    """Create plots for a single experiment."""
    print(f"Loading data for experiment: {experiment_name}")
    data = load_experiment_data(experiment_name)
    
    if data is None:
        return False
    
    print(f"Creating visualization...")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{experiment_name}_results.png")
    
    plot_dqn_training_curves(
        episode_rewards=data['episode_rewards'],
        episode_lengths=data['episode_lengths'],
        losses=data['losses'],
        epsilons=data['epsilons'],
        save_path=output_path,
        title=f"DQN Training Results: {experiment_name}"
    )
    
    print(f"✓ Plot saved to: {output_path}")
    
    # Print summary statistics
    print("\nSummary Statistics:")
    print(f"  Total episodes: {len(data['episode_rewards'])}")
    print(f"  Final avg reward (last 100): {np.mean(data['episode_rewards'][-100:]):.2f}")
    print(f"  Max reward: {np.max(data['episode_rewards']):.2f}")
    print(f"  Final epsilon: {data['epsilons'][-1]:.4f}")
    
    return True


def plot_comparison(experiment_names: list, output_dir: str = "plots"):
    """Create comparison plots for multiple experiments."""
    print(f"Loading data for {len(experiment_names)} experiments...")
    
    experiments = {}
    for name in experiment_names:
        data = load_experiment_data(name)
        if data is not None:
            experiments[name] = data
        else:
            print(f"  Skipping {name} (no data found)")
    
    if len(experiments) == 0:
        print("Error: No valid experiments found")
        return False
    
    print(f"\nCreating comparison plot for {len(experiments)} experiments...")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "dqn_comparison.png")
    
    plot_dqn_comparison(
        experiments=experiments,
        save_path=output_path,
        title="DQN Experiments Comparison"
    )
    
    print(f"✓ Comparison plot saved to: {output_path}")
    
    # Print comparison table
    print("\nComparison Table:")
    print(f"{'Experiment':<30} {'Episodes':<12} {'Final Reward':<15} {'Max Reward':<12}")
    print("-" * 70)
    for name, data in experiments.items():
        episodes = len(data['episode_rewards'])
        final_avg = np.mean(data['episode_rewards'][-100:])
        max_reward = np.max(data['episode_rewards'])
        print(f"{name:<30} {episodes:<12} {final_avg:<15.2f} {max_reward:<12.2f}")
    
    return True


def list_available_experiments(base_dir: str = "results/dqn_experiments"):
    """List all available experiment results."""
    if not os.path.exists(base_dir):
        print(f"No experiments directory found: {base_dir}")
        return
    
    # Find all experiment directories
    exp_dirs = glob.glob(os.path.join(base_dir, "*_*"))
    
    if not exp_dirs:
        print("No experiments found.")
        return
    
    # Group by experiment name
    experiments = {}
    for exp_dir in exp_dirs:
        basename = os.path.basename(exp_dir)
        # Extract experiment name (everything before last underscore + timestamp)
        parts = basename.rsplit('_', 2)
        if len(parts) >= 2:
            exp_name = parts[0]
            if exp_name not in experiments:
                experiments[exp_name] = []
            experiments[exp_name].append(exp_dir)
    
    print("\nAvailable Experiments:")
    print("=" * 60)
    for exp_name in sorted(experiments.keys()):
        count = len(experiments[exp_name])
        latest = max(experiments[exp_name], key=lambda x: os.path.basename(x).split('_')[-1])
        print(f"  {exp_name:<30} ({count} run{'s' if count > 1 else ''})")
        print(f"    Latest: {os.path.basename(latest)}")
    print("=" * 60)
    print(f"Total: {len(experiments)} unique experiments")


def main():
    parser = argparse.ArgumentParser(
        description="Visualize DQN experiment results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # List available experiments
  python visualize_dqn_results.py --list
  
  # Plot single experiment
  python visualize_dqn_results.py --experiment baseline
  
  # Compare multiple experiments
  python visualize_dqn_results.py --compare baseline learning_rate_low learning_rate_high
        """
    )
    
    parser.add_argument(
        '--list',
        action='store_true',
        help='List all available experiments'
    )
    parser.add_argument(
        '--experiment',
        type=str,
        help='Plot results for a single experiment'
    )
    parser.add_argument(
        '--compare',
        nargs='+',
        help='Compare multiple experiments'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='plots',
        help='Output directory for plots (default: plots)'
    )
    
    args = parser.parse_args()
    
    # List experiments
    if args.list:
        list_available_experiments()
        return
    
    # Single experiment
    if args.experiment:
        success = plot_single_experiment(args.experiment, args.output_dir)
        sys.exit(0 if success else 1)
    
    # Compare experiments
    if args.compare:
        success = plot_comparison(args.compare, args.output_dir)
        sys.exit(0 if success else 1)
    
    # No action specified
    parser.print_help()


if __name__ == "__main__":
    main()
