LONG_SCHEDULE = {
    'step_values': [300000, 600000, 800000, 1000000],
    # 'learning_rates': [0.00010, 0.00005, 0.000025, 0.0000125, 0.00000625],
    'learning_rates': [1e-4, 5e-5, 2.5e-5, 1.25e-5, 6.25e-6],
    'momentum': 0.9,
    'momentum2': 0.999,
    'weight_decay': 0.0004,
    'max_iter': 1200000,
}

FINETUNE_SCHEDULE = {
    # TODO: Finetune schedule
}


# LONG_SCHEDULE = {
#     'step_values': [400000, 600000, 800000, 1000000],
#     'learning_rates': [0.0001, 0.00005, 0.000025, 0.0000125, 0.00000625],
#     'momentum': 0.9,
#     'momentum2': 0.999,
#     'weight_decay': 0.0004,
#     'max_iter': 1200000,
# }
