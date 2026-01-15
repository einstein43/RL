"""
Deep Q-Network (DQN) Agent for continuous state spaces.
Implements DQN with target network and experience replay.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Tuple, Optional
from src.replay_buffer import ReplayBuffer


class DQNNetwork(nn.Module):
    """
    Neural network for approximating Q-values.
    
    Architecture:
    - Input: state features
    - Hidden layers: configurable with ReLU activation
    - Output: Q-values for each action
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: Tuple[int, ...] = (128, 128)
    ):
        """
        Initialize the DQN network.
        
        Args:
            state_dim: Dimension of state space
            action_dim: Number of possible actions
            hidden_dims: Tuple of hidden layer sizes
        """
        super(DQNNetwork, self).__init__()
        
        layers = []
        prev_dim = state_dim
        
        # Build hidden layers
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, action_dim))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            state: State tensor
            
        Returns:
            Q-values for each action
        """
        return self.network(state)


class DQNAgent:
    """
    Deep Q-Network agent with experience replay and target network.
    
    Key features:
    - Neural network for Q-value approximation
    - Target network for stable training
    - Experience replay buffer
    - Epsilon-greedy exploration
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        learning_rate: float = 0.001,
        discount_factor: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.01,
        epsilon_decay: float = 0.995,
        buffer_capacity: int = 10000,
        batch_size: int = 64,
        target_update_frequency: int = 10,
        hidden_dims: Tuple[int, ...] = (128, 128),
        device: Optional[str] = None
    ):
        """
        Initialize DQN agent.
        
        Args:
            state_dim: Dimension of state space
            action_dim: Number of possible actions
            learning_rate: Learning rate (α) for optimizer
            discount_factor: Discount factor (γ) for future rewards
            epsilon_start: Initial exploration rate
            epsilon_end: Minimum exploration rate
            epsilon_decay: Decay rate for epsilon
            buffer_capacity: Capacity of replay buffer
            batch_size: Batch size for training
            target_update_frequency: Episodes between target network updates
            hidden_dims: Hidden layer dimensions
            device: Device to use (cuda/cpu), auto-detect if None
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.batch_size = batch_size
        self.target_update_frequency = target_update_frequency
        
        # Device setup
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        # Exploration parameters
        self.epsilon = epsilon_start
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        
        # Networks
        self.policy_network = DQNNetwork(state_dim, action_dim, hidden_dims).to(self.device)
        self.target_network = DQNNetwork(state_dim, action_dim, hidden_dims).to(self.device)
        self.target_network.load_state_dict(self.policy_network.state_dict())
        self.target_network.eval()  # Target network is always in eval mode
        
        # Optimizer and loss
        self.optimizer = optim.Adam(self.policy_network.parameters(), lr=learning_rate)
        self.loss_fn = nn.MSELoss()
        
        # Replay buffer
        self.replay_buffer = ReplayBuffer(capacity=buffer_capacity)
        
        # Training statistics
        self.training_step = 0
        self.episode_count = 0
    
    def select_action(self, state: np.ndarray, training: bool = True) -> int:
        """
        Select action using epsilon-greedy policy.
        
        Args:
            state: Current state
            training: Whether in training mode (affects exploration)
            
        Returns:
            Selected action
        """
        # Exploration: random action
        if training and np.random.random() < self.epsilon:
            return np.random.randint(self.action_dim)
        
        # Exploitation: best action according to policy network
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.policy_network(state_tensor)
            return q_values.argmax().item()
    
    def store_transition(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool
    ):
        """
        Store transition in replay buffer.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode ended
        """
        self.replay_buffer.push(state, action, reward, next_state, done)
    
    def train_step(self) -> Optional[float]:
        """
        Perform one training step using a batch from replay buffer.
        
        Returns:
            Loss value if training performed, None otherwise
        """
        if not self.replay_buffer.is_ready(self.batch_size):
            return None
        
        # Sample batch from replay buffer
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
        
        # Convert to tensors
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        
        # Compute current Q-values
        current_q_values = self.policy_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # Compute target Q-values using target network
        with torch.no_grad():
            next_q_values = self.target_network(next_states).max(1)[0]
            target_q_values = rewards + (1 - dones) * self.discount_factor * next_q_values
        
        # Compute loss
        loss = self.loss_fn(current_q_values, target_q_values)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(self.policy_network.parameters(), 1.0)
        self.optimizer.step()
        
        self.training_step += 1
        
        return loss.item()
    
    def update_target_network(self):
        """Update target network with policy network weights."""
        self.target_network.load_state_dict(self.policy_network.state_dict())
    
    def decay_epsilon(self):
        """Decay epsilon (exploration rate) after each episode."""
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        self.episode_count += 1
        
        # Update target network periodically
        if self.episode_count % self.target_update_frequency == 0:
            self.update_target_network()
    
    def save(self, filepath: str):
        """
        Save agent state.
        
        Args:
            filepath: Path to save checkpoint
        """
        torch.save({
            'policy_network': self.policy_network.state_dict(),
            'target_network': self.target_network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'episode_count': self.episode_count,
            'training_step': self.training_step
        }, filepath)
    
    def load(self, filepath: str):
        """
        Load agent state.
        
        Args:
            filepath: Path to load checkpoint from
        """
        checkpoint = torch.load(filepath, map_location=self.device)
        self.policy_network.load_state_dict(checkpoint['policy_network'])
        self.target_network.load_state_dict(checkpoint['target_network'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.epsilon = checkpoint['epsilon']
        self.episode_count = checkpoint['episode_count']
        self.training_step = checkpoint['training_step']
    
    def get_exploration_value(self) -> float:
        """Get current exploration value (epsilon)."""
        return self.epsilon
