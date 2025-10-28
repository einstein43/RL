import gymnasium as gym
import numpy as np
from typing import Tuple, Optional


class RewardShapedEnv:
    
    def __init__(
        self,
        env_name: str = 'FrozenLake-v1',
        is_slippery: bool = True,
        map_name: str = '4x4',
        step_penalty: float = 0.0,
        hole_penalty: float = 0.0,
        render_mode: Optional[str] = None
    ):
        self.env_name = env_name
        self.step_penalty = step_penalty
        self.hole_penalty = hole_penalty
        
        if 'FrozenLake' in env_name:
            self.env = gym.make(
                env_name,
                is_slippery=is_slippery,
                map_name=map_name,
                render_mode=render_mode
            )
        else:
            self.env = gym.make(env_name, render_mode=render_mode)
        
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space
        
    def reset(self, seed: Optional[int] = None) -> Tuple[int, dict]:
        return self.env.reset(seed=seed)
    
    def step(self, action: int) -> Tuple[int, float, bool, bool, dict]:
        next_state, reward, terminated, truncated, info = self.env.step(action)
        
        # Apply reward shaping
        shaped_reward = reward + self.step_penalty
        
        # Extra penalty for falling in holes
        if terminated and reward == 0:
            shaped_reward += self.hole_penalty
        
        return next_state, shaped_reward, terminated, truncated, info
    
    def render(self):
        return self.env.render()
    
    def close(self):
        self.env.close()
    
    @property
    def n_states(self) -> int:
        return self.observation_space.n
    
    @property
    def n_actions(self) -> int:
        return self.action_space.n


def create_environment(env_name: str = 'FrozenLake-v1', **kwargs) -> RewardShapedEnv:
    return RewardShapedEnv(env_name=env_name, **kwargs)
