"""
Quick test script to verify DQN implementation works correctly.
Trains for fewer episodes to quickly test the system.
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import gymnasium as gym
import numpy as np
from src.dqn_agent import DQNAgent
from src.dqn_trainer import DQNTrainer
from src.visualization import plot_dqn_training_curves


def quick_test():
    """Run a quick test with reduced episodes."""
    print("="*60)
    print("QUICK DQN TEST")
    print("="*60)
    print("Running baseline DQN with reduced episodes for testing...\n")
    
    # Environment setup
    env = gym.make("CartPole-v1")
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    env.close()
    
    # Create agent
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
    
    # Create trainer
    trainer = DQNTrainer(
        env_name="CartPole-v1",
        agent=agent,
        save_dir="results/dqn_test"
    )
    
    # Train for fewer episodes
    print("Training for 200 episodes...")
    training_history = trainer.train(
        n_episodes=200,
        eval_frequency=50,
        save_frequency=100,
        target_reward=500.0,
        early_stop_episodes=10,
        verbose=True
    )
    
    trainer.close()
    
    # Create visualization
    print("\nCreating visualization...")
    plot_dqn_training_curves(
        episode_rewards=training_history['episode_rewards'],
        episode_lengths=training_history['episode_lengths'],
        losses=training_history['losses'],
        epsilons=training_history['epsilons'],
        save_path='plots/dqn_test_results.png',
        title='DQN Quick Test Results'
    )
    
    # Summary
    print("\n" + "="*60)
    print("TEST COMPLETE")
    print("="*60)
    final_eval = training_history['final_eval']
    print(f"Final mean reward: {final_eval['mean_reward']:.2f}")
    print(f"Final success rate: {final_eval['success_rate']:.2%}")
    print(f"Total episodes trained: {len(training_history['episode_rewards'])}")
    
    avg_last_100 = np.mean(training_history['episode_rewards'][-100:])
    print(f"Average reward (last 100 episodes): {avg_last_100:.2f}")
    
    if final_eval['success_rate'] >= 0.5:
        print("\n✓ Test PASSED: Agent shows good learning (50%+ success rate)")
    elif avg_last_100 > 100:
        print("\n✓ Test PASSED: Agent is learning (avg reward > 100)")
    else:
        print("\n⚠ Test INCOMPLETE: Agent needs more training")
    
    print(f"\nResults saved to: results/dqn_test/")
    print(f"Plot saved to: plots/dqn_test_results.png")
    
    return training_history


if __name__ == "__main__":
    quick_test()
