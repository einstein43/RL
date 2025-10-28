BASE_CONFIG = {
    'n_episodes': 10000,
    'max_steps': 100,
    'eval_frequency': 100,
    'eval_episodes': 100,
    'random_seed': 42,
}

# Experiment 1: Learning Rate Comparison
EXPERIMENT_1_CONFIGS = {
    'learning_rate_low': {
        **BASE_CONFIG,
        'learning_rate': 0.1,
        'discount_factor': 0.99,
        'exploration_strategy': 'epsilon_greedy',
        'epsilon_decay': 0.995,
    },
    'learning_rate_medium': {
        **BASE_CONFIG,
        'learning_rate': 0.3,
        'discount_factor': 0.99,
        'exploration_strategy': 'epsilon_greedy',
        'epsilon_decay': 0.995,
    },
    'learning_rate_high': {
        **BASE_CONFIG,
        'learning_rate': 0.7,
        'discount_factor': 0.99,
        'exploration_strategy': 'epsilon_greedy',
        'epsilon_decay': 0.995,
    },
}

# Experiment 2: Slippery vs Non-Slippery
EXPERIMENT_2_CONFIGS = {
    'slippery_on': {
        **BASE_CONFIG,
        'learning_rate': 0.1,
        'discount_factor': 0.99,
        'exploration_strategy': 'epsilon_greedy',
        'is_slippery': True,
    },
    'slippery_off': {
        **BASE_CONFIG,
        'learning_rate': 0.1,
        'discount_factor': 0.99,
        'exploration_strategy': 'epsilon_greedy',
        'is_slippery': False,
    },
}

# Experiment 3: Reward Shaping
EXPERIMENT_3_CONFIGS = {
    'no_shaping': {
        **BASE_CONFIG,
        'learning_rate': 0.1,
        'discount_factor': 0.99,
        'exploration_strategy': 'epsilon_greedy',
        'step_penalty': 0.0,
        'hole_penalty': 0.0,
    },
    'small_step_penalty': {
        **BASE_CONFIG,
        'learning_rate': 0.1,
        'discount_factor': 0.99,
        'exploration_strategy': 'epsilon_greedy',
        'step_penalty': -0.01,
        'hole_penalty': 0.0,
    },
    'large_hole_penalty': {
        **BASE_CONFIG,
        'learning_rate': 0.1,
        'discount_factor': 0.99,
        'exploration_strategy': 'epsilon_greedy',
        'step_penalty': -0.01,
        'hole_penalty': -1.0,
    },
}

# Experiment 4: Exploration Strategies
EXPERIMENT_4_CONFIGS = {
    'epsilon_greedy': {
        **BASE_CONFIG,
        'learning_rate': 0.1,
        'discount_factor': 0.99,
        'exploration_strategy': 'epsilon_greedy',
        'epsilon_start': 1.0,
        'epsilon_end': 0.01,
        'epsilon_decay': 0.995,
    },
    'boltzmann_high_temp': {
        **BASE_CONFIG,
        'learning_rate': 0.1,
        'discount_factor': 0.99,
        'exploration_strategy': 'boltzmann',
        'temperature': 5.0,
        'temperature_decay': 0.995,
        'temperature_min': 0.01,
    },
    'boltzmann_low_temp': {
        **BASE_CONFIG,
        'learning_rate': 0.1,
        'discount_factor': 0.99,
        'exploration_strategy': 'boltzmann',
        'temperature': 1.0,
        'temperature_decay': 0.995,
        'temperature_min': 0.01,
    },
}

# Experiment 5: Discount Factor
EXPERIMENT_5_CONFIGS = {
    'gamma_low': {
        **BASE_CONFIG,
        'learning_rate': 0.1,
        'discount_factor': 0.9,
        'exploration_strategy': 'epsilon_greedy',
    },
    'gamma_medium': {
        **BASE_CONFIG,
        'learning_rate': 0.1,
        'discount_factor': 0.95,
        'exploration_strategy': 'epsilon_greedy',
    },
    'gamma_high': {
        **BASE_CONFIG,
        'learning_rate': 0.1,
        'discount_factor': 0.99,
        'exploration_strategy': 'epsilon_greedy',
    },
}

# Environment configurations
ENV_CONFIGS = {
    'frozen_lake_4x4': {
        'env_name': 'FrozenLake-v1',
        'map_name': '4x4',
        'is_slippery': True,
        'step_penalty': 0.0,
        'hole_penalty': 0.0,
    },
    'frozen_lake_8x8': {
        'env_name': 'FrozenLake-v1',
        'map_name': '8x8',
        'is_slippery': True,
        'step_penalty': 0.0,
        'hole_penalty': 0.0,
    },
}
