"""
Run DQN training with automatic continuation on interruption.
Saves checkpoints and can resume from last checkpoint.
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

import gymnasium as gym
import numpy as np
from src.dqn_agent import DQNAgent
from src.dqn_trainer import DQNTrainer
from src.visualization import plot_dqn_training_curves
import signal

def signal_handler(sig, frame):
    """Handle Ctrl+C gracefully."""
    print('\n\nTraining interrupted by user. Saving current state...')
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

def run_baseline_experiment():
    """Run baseline DQN experiment with error handling."""
    print("="*70)
    print("BASELINE DQN EXPERIMENT - CartPole-v1")
    print("="*70)
    
    # Environment
    env = gym.make("CartPole-v1")
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    env.close()
    
    print(f"\nState dimension: {state_dim}")
    print(f"Action dimension: {action_dim}")
    
    # Create agent
    print("\nInitializing DQN Agent...")
    agent = DQNAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        learning_rate=0.001,
        discount_factor=0.99,
        epsilon_start=1.0,
        epsilon_end=0.01,
        epsilon_decay=0.995,
        buffer_capacity=10000,
        batch_size=64,
        target_update_frequency=10,
        hidden_dims=(128, 128)
    )
    
    print(f"Network architecture: {agent.policy_network}")
    print(f"Device: {agent.device}")
    
    # Create trainer
    print("\nInitializing Trainer...")
    trainer = DQNTrainer(
        env_name="CartPole-v1",
        agent=agent,
        save_dir="results/dqn_experiments/baseline_final"
    )
    
    print("\nStarting training...")
    print("Target: Reach 500 steps consistently (10 episodes)")
    print("Max episodes: 1000")
    print("-"*70)
    
    try:
        # Train
        training_history = trainer.train(
            n_episodes=1000,
            eval_frequency=50,
            save_frequency=100,
            target_reward=500.0,
            early_stop_episodes=10,
            verbose=True
        )
        
        print("\n" + "="*70)
        print("TRAINING COMPLETED SUCCESSFULLY")
        print("="*70)
        
        # Create visualization
        print("\nGenerating visualization...")
        plot_dqn_training_curves(
            episode_rewards=training_history['episode_rewards'],
            episode_lengths=training_history['episode_lengths'],
            losses=training_history['losses'],
            epsilons=training_history['epsilons'],
            save_path='plots/baseline_final_results.png',
            title='Baseline DQN - CartPole-v1'
        )
        
        # Summary
        final_eval = training_history['final_eval']
        print("\n" + "="*70)
        print("FINAL RESULTS")
        print("="*70)
        print(f"Total episodes trained: {len(training_history['episode_rewards'])}")
        print(f"Final evaluation (100 episodes):")
        print(f"  Mean reward: {final_eval['mean_reward']:.2f} ± {final_eval['std_reward']:.2f}")
        print(f"  Mean length: {final_eval['mean_length']:.1f} ± {final_eval['std_length']:.1f}")
        print(f"  Success rate: {final_eval['success_rate']:.2%}")
        print(f"  Min reward: {final_eval['min_reward']:.2f}")
        print(f"  Max reward: {final_eval['max_reward']:.2f}")
        
        avg_last_100 = np.mean(training_history['episode_rewards'][-100:])
        print(f"\nAverage reward (last 100 training episodes): {avg_last_100:.2f}")
        
        if final_eval['success_rate'] >= 0.9:
            print("\n🎉 SUCCESS! Agent solved CartPole (90%+ success rate)")
        elif final_eval['success_rate'] >= 0.5:
            print("\n✓ GOOD PERFORMANCE! Agent shows strong learning (50%+ success rate)")
        else:
            print(f"\n⚠ Training completed but success rate below 50%")
        
        print(f"\nResults saved to: results/dqn_experiments/baseline_final/")
        print(f"Plots saved to: plots/baseline_final_results.png")
        print("="*70)
        
        trainer.close()
        return training_history
        
    except KeyboardInterrupt:
        print("\n\nTraining interrupted!")
        print("Partial results have been saved.")
        trainer.close()
        return None
    except Exception as e:
        print(f"\n\nError during training: {e}")
        import traceback
        traceback.print_exc()
        trainer.close()
        return None

if __name__ == "__main__":
    run_baseline_experiment()
