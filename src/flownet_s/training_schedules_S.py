LONG_SCHEDULE = {
    'step_values': [300000, 400000, 500000, 600000],
    'learning_rates': [0.0001, 0.00005, 0.000025, 0.0000125, 0.00000625],
    'momentum': 0.9,
    'momentum2': 0.999,
    'weight_decay': 0.0004,
    'max_iter': 1200000,
}

TEST_SCHEDULE = {
    'step_values': [300000, 600000, 800000, 1000000, 1200400, 1200800],
    'learning_rates': [1e-4, 5e-5, 2.5e-5, 1.25e-5, 6.25e-6, 5e-6, 4e-6],
    # 'learning_rates': [1e-4, 5e-5, 2.5e-5, 1.25e-5, 1e-8],

    'momentum': 0.9,
    'momentum2': 0.999,
    'weight_decay': 0.0004,
    'max_iter': 1500000,
}