"""
DQN Trainer for CartPole environment.
Handles training loop, logging, and checkpointing.
"""

import gymnasium as gym
import numpy as np
from typing import Dict, List, Tuple, Optional
import os
from datetime import datetime
from tqdm import tqdm
import json

from src.dqn_agent import DQNAgent


class DQNTrainer:
    """
    Trainer for DQN agent on Gymnasium environments.
    
    Handles:
    - Training loop
    - Logging metrics
    - Saving checkpoints
    - Evaluation
    """
    
    def __init__(
        self,
        env_name: str = "CartPole-v1",
        agent: Optional[DQNAgent] = None,
        save_dir: str = "results/dqn_experiments"
    ):
        """
        Initialize DQN trainer.
        
        Args:
            env_name: Name of Gymnasium environment
            agent: DQN agent (if None, must call set_agent)
            save_dir: Directory to save results
        """
        self.env_name = env_name
        self.env = gym.make(env_name)
        self.agent = agent
        self.save_dir = save_dir
        
        # Training statistics
        self.episode_rewards: List[float] = []
        self.episode_lengths: List[int] = []
        self.losses: List[float] = []
        self.epsilons: List[float] = []
        self.success_count = 0
        self.total_episodes = 0
        
        # Create save directory
        os.makedirs(save_dir, exist_ok=True)
    
    def set_agent(self, agent: DQNAgent):
        """Set the agent to train."""
        self.agent = agent
    
    def train_episode(self) -> Tuple[float, int, float]:
        """
        Train for one episode.
        
        Returns:
            Tuple of (episode_reward, episode_length, average_loss)
        """
        state, _ = self.env.reset()
        episode_reward = 0
        episode_length = 0
        episode_losses = []
        done = False
        
        while not done:
            # Select and perform action
            action = self.agent.select_action(state, training=True)
            next_state, reward, terminated, truncated, _ = self.env.step(action)
            done = terminated or truncated
            
            # Store transition
            self.agent.store_transition(state, action, reward, next_state, done)
            
            # Train on batch
            loss = self.agent.train_step()
            if loss is not None:
                episode_losses.append(loss)
            
            # Update state and statistics
            state = next_state
            episode_reward += reward
            episode_length += 1
        
        # Decay exploration
        self.agent.decay_epsilon()
        
        avg_loss = np.mean(episode_losses) if episode_losses else 0.0
        return episode_reward, episode_length, avg_loss
    
    def evaluate(self, n_episodes: int = 100, render: bool = False) -> Dict[str, float]:
        """
        Evaluate agent performance.
        
        Args:
            n_episodes: Number of episodes to evaluate
            render: Whether to render environment
            
        Returns:
            Dictionary with evaluation metrics
        """
        if render:
            eval_env = gym.make(self.env_name, render_mode="human")
        else:
            eval_env = gym.make(self.env_name)
        
        rewards = []
        lengths = []
        successes = 0
        
        for _ in range(n_episodes):
            state, _ = eval_env.reset()
            episode_reward = 0
            episode_length = 0
            done = False
            
            while not done:
                action = self.agent.select_action(state, training=False)
                state, reward, terminated, truncated, _ = eval_env.step(action)
                done = terminated or truncated
                episode_reward += reward
                episode_length += 1
            
            rewards.append(episode_reward)
            lengths.append(episode_length)
            
            # CartPole success criterion (500 or 200 depending on version)
            if episode_length >= 500:
                successes += 1
        
        eval_env.close()
        
        return {
            'mean_reward': np.mean(rewards),
            'std_reward': np.std(rewards),
            'mean_length': np.mean(lengths),
            'std_length': np.std(lengths),
            'success_rate': successes / n_episodes,
            'min_reward': np.min(rewards),
            'max_reward': np.max(rewards)
        }
    
    def train(
        self,
        n_episodes: int = 1000,
        eval_frequency: int = 50,
        save_frequency: int = 100,
        target_reward: float = 500.0,
        early_stop_episodes: int = 10,
        verbose: bool = True
    ) -> Dict[str, List]:
        """
        Train the agent.
        
        Args:
            n_episodes: Number of training episodes
            eval_frequency: Evaluate every N episodes
            save_frequency: Save checkpoint every N episodes
            target_reward: Target average reward to consider solved
            early_stop_episodes: Stop if target reached for this many episodes
            verbose: Whether to print progress
            
        Returns:
            Dictionary with training history
        """
        if self.agent is None:
            raise ValueError("Agent not set. Call set_agent() first.")
        
        consecutive_success = 0
        best_mean_reward = -float('inf')
        
        pbar = tqdm(range(n_episodes), desc="Training DQN") if verbose else range(n_episodes)
        
        for episode in pbar:
            # Train one episode
            episode_reward, episode_length, avg_loss = self.train_episode()
            
            # Record statistics
            self.episode_rewards.append(episode_reward)
            self.episode_lengths.append(episode_length)
            if avg_loss > 0:
                self.losses.append(avg_loss)
            self.epsilons.append(self.agent.get_exploration_value())
            self.total_episodes += 1
            
            # Check for success
            if episode_reward >= target_reward:
                consecutive_success += 1
            else:
                consecutive_success = 0
            
            # Update progress bar
            if verbose and episode % 10 == 0:
                recent_rewards = self.episode_rewards[-100:]
                avg_reward = np.mean(recent_rewards) if recent_rewards else 0
                pbar.set_postfix({
                    'reward': f'{episode_reward:.1f}',
                    'avg_100': f'{avg_reward:.1f}',
                    'epsilon': f'{self.agent.epsilon:.3f}',
                    'loss': f'{avg_loss:.4f}'
                })
            
            # Evaluation
            if episode % eval_frequency == 0 and episode > 0:
                eval_results = self.evaluate(n_episodes=10)
                if verbose:
                    print(f"\n[Eval] Episode {episode}: "
                          f"Mean reward: {eval_results['mean_reward']:.2f}, "
                          f"Success rate: {eval_results['success_rate']:.2%}")
                
                # Save best model
                if eval_results['mean_reward'] > best_mean_reward:
                    best_mean_reward = eval_results['mean_reward']
                    self.save_checkpoint(episode, "best_model.pt")
            
            # Save checkpoint
            if episode % save_frequency == 0 and episode > 0:
                self.save_checkpoint(episode)
            
            # Early stopping
            if consecutive_success >= early_stop_episodes:
                if verbose:
                    print(f"\n✓ Solved! Target reward reached for {early_stop_episodes} consecutive episodes.")
                break
        
        # Final evaluation
        if verbose:
            print("\n" + "="*50)
            print("Training Complete - Final Evaluation")
            print("="*50)
        
        final_eval = self.evaluate(n_episodes=100)
        if verbose:
            print(f"Mean reward: {final_eval['mean_reward']:.2f} ± {final_eval['std_reward']:.2f}")
            print(f"Mean length: {final_eval['mean_length']:.1f} ± {final_eval['std_length']:.1f}")
            print(f"Success rate: {final_eval['success_rate']:.2%}")
        
        # Save final model and results
        self.save_checkpoint(self.total_episodes, "final_model.pt")
        self.save_results(final_eval)
        
        return {
            'episode_rewards': self.episode_rewards,
            'episode_lengths': self.episode_lengths,
            'losses': self.losses,
            'epsilons': self.epsilons,
            'final_eval': final_eval
        }
    
    def save_checkpoint(self, episode: int, filename: Optional[str] = None):
        """
        Save agent checkpoint.
        
        Args:
            episode: Current episode number
            filename: Custom filename (if None, use episode number)
        """
        if filename is None:
            filename = f"checkpoint_ep{episode}.pt"
        
        filepath = os.path.join(self.save_dir, filename)
        self.agent.save(filepath)
    
    def save_results(self, final_eval: Dict[str, float]):
        """
        Save training results and configuration.
        
        Args:
            final_eval: Final evaluation metrics
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save training history as numpy arrays
        np.savez(
            os.path.join(self.save_dir, f"training_history_{timestamp}.npz"),
            episode_rewards=np.array(self.episode_rewards),
            episode_lengths=np.array(self.episode_lengths),
            losses=np.array(self.losses),
            epsilons=np.array(self.epsilons)
        )
        
        # Save configuration and results
        # Convert numpy types to native Python types for JSON serialization
        def convert_to_native(obj):
            """Convert numpy types to native Python types."""
            if isinstance(obj, (np.integer, np.int64, np.int32)):
                return int(obj)
            elif isinstance(obj, (np.floating, np.float64, np.float32)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            else:
                return obj
        
        config = {
            'environment': self.env_name,
            'total_episodes': int(self.total_episodes),
            'state_dim': int(self.agent.state_dim),
            'action_dim': int(self.agent.action_dim),
            'learning_rate': float(self.agent.learning_rate),
            'discount_factor': float(self.agent.discount_factor),
            'epsilon_start': float(self.agent.epsilon_start),
            'epsilon_end': float(self.agent.epsilon_end),
            'epsilon_decay': float(self.agent.epsilon_decay),
            'batch_size': int(self.agent.batch_size),
            'buffer_capacity': int(self.agent.replay_buffer.capacity),
            'target_update_frequency': int(self.agent.target_update_frequency),
            'final_evaluation': {k: convert_to_native(v) for k, v in final_eval.items()},
            'timestamp': timestamp
        }
        
        with open(os.path.join(self.save_dir, f"config_{timestamp}.json"), 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"\n✓ Results saved to {self.save_dir}")
    
    def close(self):
        """Close the environment."""
        self.env.close()
