import numpy as np
from typing import Dict, List, Tuple
from src.q_learning import QLearningAgent
from src.environment import RewardShapedEnv


class QLearningTrainer:
    
    def __init__(
        self,
        env: RewardShapedEnv,
        agent: QLearningAgent,
        n_episodes: int = 10000,
        max_steps: int = 100,
        eval_frequency: int = 100,
        eval_episodes: int = 100,
        verbose: bool = True
    ):
        self.env = env
        self.agent = agent
        self.n_episodes = n_episodes
        self.max_steps = max_steps
        self.eval_frequency = eval_frequency
        self.eval_episodes = eval_episodes
        self.verbose = verbose
        
        self.training_rewards = []
        self.training_steps = []
        self.eval_rewards = []
        self.eval_success_rates = []
        self.exploration_values = []
        
    def train(self) -> Dict[str, List[float]]:
        for episode in range(self.n_episodes):
            episode_reward, episode_steps = self._run_episode(training=True)
            
            self.training_rewards.append(episode_reward)
            self.training_steps.append(episode_steps)
            self.exploration_values.append(self.agent.get_exploration_value())
            
            self.agent.decay_exploration()
            
            if (episode + 1) % self.eval_frequency == 0:
                eval_reward, success_rate = self._evaluate()
                self.eval_rewards.append(eval_reward)
                self.eval_success_rates.append(success_rate)
                
                if self.verbose:
                    print(f"Episode {episode + 1}/{self.n_episodes}")
                    print(f"  Avg Training Reward (last 100): {np.mean(self.training_rewards[-100:]):.3f}")
                    print(f"  Eval Reward: {eval_reward:.3f}")
                    print(f"  Success Rate: {success_rate:.2%}")
                    print(f"  Exploration Value: {self.exploration_values[-1]:.4f}")
                    print()
        
        return self._get_metrics()
    
    def _run_episode(self, training: bool = True) -> Tuple[float, int]:
        state, _ = self.env.reset()
        total_reward = 0
        steps = 0
        
        for step in range(self.max_steps):
            action = self.agent.select_action(state, training=training)
            next_state, reward, terminated, truncated, _ = self.env.step(action)
            done = terminated or truncated
            
            if training:
                self.agent.update(state, action, reward, next_state, terminated)
            
            total_reward += reward
            steps += 1
            state = next_state
            
            if done:
                break
        
        return total_reward, steps
    
    def _evaluate(self) -> Tuple[float, float]:
        rewards = []
        successes = 0
        
        for _ in range(self.eval_episodes):
            episode_reward, _ = self._run_episode(training=False)
            rewards.append(episode_reward)
            
            if episode_reward > 0:
                successes += 1
        
        avg_reward = np.mean(rewards)
        success_rate = successes / self.eval_episodes
        
        return avg_reward, success_rate
    
    def _get_metrics(self) -> Dict[str, List[float]]:
        return {
            'training_rewards': self.training_rewards,
            'training_steps': self.training_steps,
            'eval_rewards': self.eval_rewards,
            'eval_success_rates': self.eval_success_rates,
            'exploration_values': self.exploration_values,
        }
