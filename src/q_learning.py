import numpy as np
from typing import Tuple, Optional


class QLearningAgent:
    
    def __init__(
        self,
        n_states: int,
        n_actions: int,
        learning_rate: float = 0.1,
        discount_factor: float = 0.99,
        exploration_strategy: str = 'epsilon_greedy',
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.01,
        epsilon_decay: float = 0.995,
        temperature: float = 1.0,
        temperature_decay: float = 0.995,
        temperature_min: float = 0.01
    ):
        self.n_states = n_states
        self.n_actions = n_actions
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_strategy = exploration_strategy
        
        self.q_table = np.zeros((n_states, n_actions))
        
        # Epsilon-greedy
        self.epsilon = epsilon_start
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        
        # Boltzmann
        self.temperature = temperature
        self.temperature_decay = temperature_decay
        self.temperature_min = temperature_min
    
    def select_action(self, state: int, training: bool = True) -> int:
        if not training:
            return np.argmax(self.q_table[state])
        
        if self.exploration_strategy == 'epsilon_greedy':
            return self._epsilon_greedy_action(state)
        elif self.exploration_strategy == 'boltzmann':
            return self._boltzmann_action(state)
        else:
            raise ValueError(f"Unknown exploration strategy: {self.exploration_strategy}")
    
    def _epsilon_greedy_action(self, state: int) -> int:
        if np.random.random() < self.epsilon:
            return np.random.randint(self.n_actions)
        else:
            return np.argmax(self.q_table[state])
    
    def _boltzmann_action(self, state: int) -> int:
        q_values = self.q_table[state]
        q_values_scaled = (q_values - np.max(q_values)) / self.temperature
        exp_q = np.exp(q_values_scaled)
        probabilities = exp_q / np.sum(exp_q)
        return np.random.choice(self.n_actions, p=probabilities)
    
    def update(self, state: int, action: int, reward: float, next_state: int, done: bool) -> float:
        # Q-learning update rule
        current_q = self.q_table[state, action]
        
        if done:
            target_q = reward
        else:
            target_q = reward + self.discount_factor * np.max(self.q_table[next_state])
        
        td_error = target_q - current_q
        self.q_table[state, action] += self.learning_rate * td_error
        
        return abs(td_error)
    
    def decay_exploration(self):
        if self.exploration_strategy == 'epsilon_greedy':
            self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        elif self.exploration_strategy == 'boltzmann':
            self.temperature = max(self.temperature_min, self.temperature * self.temperature_decay)
    
    def get_exploration_value(self) -> float:
        if self.exploration_strategy == 'epsilon_greedy':
            return self.epsilon
        else:
            return self.temperature
    
    def reset_exploration(self):
        self.epsilon = self.epsilon_start
        self.temperature = self.temperature_decay
