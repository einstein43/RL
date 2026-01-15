"""
Run all DQN experiments sequentially.
Useful for comprehensive hyperparameter comparison.
"""

import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from experiments.run_dqn_experiment import run_experiment
from experiments.dqn_config import list_experiments


def run_all_dqn_experiments(experiments: list = None):
    """
    Run multiple DQN experiments.
    
    Args:
        experiments: List of experiment names to run (None = all)
    """
    if experiments is None:
        experiments = ['baseline', 'learning_rate_low', 'learning_rate_high', 
                      'gamma_low', 'gamma_high']
    
    print("="*70)
    print("RUNNING DQN EXPERIMENT SUITE")
    print("="*70)
    print(f"Total experiments: {len(experiments)}")
    print(f"Experiments: {', '.join(experiments)}")
    print("="*70 + "\n")
    
    results = {}
    
    for i, exp_name in enumerate(experiments, 1):
        print(f"\n{'='*70}")
        print(f"EXPERIMENT {i}/{len(experiments)}: {exp_name}")
        print(f"{'='*70}\n")
        
        try:
            history, save_dir = run_experiment(exp_name, verbose=True)
            results[exp_name] = {
                'history': history,
                'save_dir': save_dir
            }
            print(f"\n✓ Completed: {exp_name}")
        except Exception as e:
            print(f"\n✗ Failed: {exp_name}")
            print(f"Error: {e}")
            results[exp_name] = {'error': str(e)}
    
    # Summary
    print("\n" + "="*70)
    print("EXPERIMENT SUITE COMPLETE")
    print("="*70)
    
    successful = sum(1 for r in results.values() if 'error' not in r)
    failed = len(results) - successful
    
    print(f"Successful: {successful}/{len(results)}")
    print(f"Failed: {failed}/{len(results)}")
    
    if successful > 0:
        print("\nSuccessful experiments:")
        for exp_name, result in results.items():
            if 'error' not in result:
                final_eval = result['history']['final_eval']
                print(f"  - {exp_name}:")
                print(f"      Mean reward: {final_eval['mean_reward']:.2f}")
                print(f"      Success rate: {final_eval['success_rate']:.2%}")
    
    if failed > 0:
        print("\nFailed experiments:")
        for exp_name, result in results.items():
            if 'error' in result:
                print(f"  - {exp_name}: {result['error']}")
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run DQN experiment suite")
    parser.add_argument(
        '--experiments',
        nargs='+',
        default=None,
        help='List of experiments to run (default: baseline + hyperparameter variants)'
    )
    
    args = parser.parse_args()
    run_all_dqn_experiments(args.experiments)
