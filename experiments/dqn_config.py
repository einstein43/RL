"""
Configuration for DQN experiments on CartPole.
"""

from typing import Dict, Any

# Base configuration for all DQN experiments
BASE_CONFIG = {
    'environment': 'CartPole-v1',
    'n_episodes': 1000,
    'eval_frequency': 50,
    'save_frequency': 100,
    'target_reward': 500.0,
    'early_stop_episodes': 10
}

# Agent configuration presets
AGENT_CONFIGS = {
    'baseline': {
        'learning_rate': 0.001,
        'discount_factor': 0.99,
        'epsilon_start': 1.0,
        'epsilon_end': 0.01,
        'epsilon_decay': 0.995,
        'buffer_capacity': 10000,
        'batch_size': 64,
        'target_update_frequency': 10,
        'hidden_dims': (128, 128)
    },
    
    'learning_rate_low': {
        'learning_rate': 0.0001,
        'discount_factor': 0.99,
        'epsilon_start': 1.0,
        'epsilon_end': 0.01,
        'epsilon_decay': 0.995,
        'buffer_capacity': 10000,
        'batch_size': 64,
        'target_update_frequency': 10,
        'hidden_dims': (128, 128)
    },
    
    'learning_rate_high': {
        'learning_rate': 0.01,
        'discount_factor': 0.99,
        'epsilon_start': 1.0,
        'epsilon_end': 0.01,
        'epsilon_decay': 0.995,
        'buffer_capacity': 10000,
        'batch_size': 64,
        'target_update_frequency': 10,
        'hidden_dims': (128, 128)
    },
    
    'gamma_low': {
        'learning_rate': 0.001,
        'discount_factor': 0.95,
        'epsilon_start': 1.0,
        'epsilon_end': 0.01,
        'epsilon_decay': 0.995,
        'buffer_capacity': 10000,
        'batch_size': 64,
        'target_update_frequency': 10,
        'hidden_dims': (128, 128)
    },
    
    'gamma_high': {
        'learning_rate': 0.001,
        'discount_factor': 0.999,
        'epsilon_start': 1.0,
        'epsilon_end': 0.01,
        'epsilon_decay': 0.995,
        'buffer_capacity': 10000,
        'batch_size': 64,
        'target_update_frequency': 10,
        'hidden_dims': (128, 128)
    },
    
    'network_small': {
        'learning_rate': 0.001,
        'discount_factor': 0.99,
        'epsilon_start': 1.0,
        'epsilon_end': 0.01,
        'epsilon_decay': 0.995,
        'buffer_capacity': 10000,
        'batch_size': 64,
        'target_update_frequency': 10,
        'hidden_dims': (64, 64)
    },
    
    'network_large': {
        'learning_rate': 0.001,
        'discount_factor': 0.99,
        'epsilon_start': 1.0,
        'epsilon_end': 0.01,
        'epsilon_decay': 0.995,
        'buffer_capacity': 10000,
        'batch_size': 64,
        'target_update_frequency': 10,
        'hidden_dims': (256, 256)
    },
    
    'buffer_small': {
        'learning_rate': 0.001,
        'discount_factor': 0.99,
        'epsilon_start': 1.0,
        'epsilon_end': 0.01,
        'epsilon_decay': 0.995,
        'buffer_capacity': 1000,
        'batch_size': 64,
        'target_update_frequency': 10,
        'hidden_dims': (128, 128)
    },
    
    'buffer_large': {
        'learning_rate': 0.001,
        'discount_factor': 0.99,
        'epsilon_start': 1.0,
        'epsilon_end': 0.01,
        'epsilon_decay': 0.995,
        'buffer_capacity': 50000,
        'batch_size': 64,
        'target_update_frequency': 10,
        'hidden_dims': (128, 128)
    },
    
    'batch_size_small': {
        'learning_rate': 0.001,
        'discount_factor': 0.99,
        'epsilon_start': 1.0,
        'epsilon_end': 0.01,
        'epsilon_decay': 0.995,
        'buffer_capacity': 10000,
        'batch_size': 32,
        'target_update_frequency': 10,
        'hidden_dims': (128, 128)
    },
    
    'batch_size_large': {
        'learning_rate': 0.001,
        'discount_factor': 0.99,
        'epsilon_start': 1.0,
        'epsilon_end': 0.01,
        'epsilon_decay': 0.995,
        'buffer_capacity': 10000,
        'batch_size': 128,
        'target_update_frequency': 10,
        'hidden_dims': (128, 128)
    },
    
    'fast_exploration_decay': {
        'learning_rate': 0.001,
        'discount_factor': 0.99,
        'epsilon_start': 1.0,
        'epsilon_end': 0.01,
        'epsilon_decay': 0.98,  # Faster decay
        'buffer_capacity': 10000,
        'batch_size': 64,
        'target_update_frequency': 10,
        'hidden_dims': (128, 128)
    },
    
    'slow_exploration_decay': {
        'learning_rate': 0.001,
        'discount_factor': 0.99,
        'epsilon_start': 1.0,
        'epsilon_end': 0.01,
        'epsilon_decay': 0.999,  # Slower decay
        'buffer_capacity': 10000,
        'batch_size': 64,
        'target_update_frequency': 10,
        'hidden_dims': (128, 128)
    }
}


def get_config(experiment_name: str) -> Dict[str, Any]:
    """
    Get full configuration for an experiment.
    
    Args:
        experiment_name: Name of the experiment (must be in AGENT_CONFIGS)
        
    Returns:
        Complete configuration dictionary
        
    Raises:
        KeyError: If experiment_name not found
    """
    if experiment_name not in AGENT_CONFIGS:
        raise KeyError(f"Unknown experiment: {experiment_name}. "
                      f"Available: {list(AGENT_CONFIGS.keys())}")
    
    config = BASE_CONFIG.copy()
    config['agent_config'] = AGENT_CONFIGS[experiment_name]
    config['experiment_name'] = experiment_name
    
    return config


def list_experiments() -> list:
    """Get list of all available experiment names."""
    return list(AGENT_CONFIGS.keys())
