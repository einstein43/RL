"""
Experience Replay Buffer for Deep Q-Network (DQN).
Stores transitions and provides random sampling for training.
"""

import numpy as np
from collections import deque
import random
from typing import Tuple, List


class ReplayBuffer:
    """
    Experience replay buffer to store and sample transitions.
    
    This buffer implements the experience replay mechanism that is crucial for
    DQN stability. It breaks the correlation between consecutive samples by
    randomly sampling from past experiences.
    """
    
    def __init__(self, capacity: int = 10000):
        """
        Initialize the replay buffer.
        
        Args:
            capacity: Maximum number of transitions to store
        """
        self.buffer = deque(maxlen=capacity)
        self.capacity = capacity
    
    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool
    ):
        """
        Add a transition to the buffer.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode ended
        """
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Sample a random batch of transitions.
        
        Args:
            batch_size: Number of transitions to sample
            
        Returns:
            Tuple of (states, actions, rewards, next_states, dones) as numpy arrays
        """
        batch = random.sample(self.buffer, batch_size)
        
        states, actions, rewards, next_states, dones = zip(*batch)
        
        return (
            np.array(states, dtype=np.float32),
            np.array(actions, dtype=np.int64),
            np.array(rewards, dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            np.array(dones, dtype=np.float32)
        )
    
    def __len__(self) -> int:
        """Return current size of the buffer."""
        return len(self.buffer)
    
    def is_ready(self, batch_size: int) -> bool:
        """
        Check if buffer has enough samples for training.
        
        Args:
            batch_size: Required batch size
            
        Returns:
            True if buffer has at least batch_size samples
        """
        return len(self.buffer) >= batch_size
