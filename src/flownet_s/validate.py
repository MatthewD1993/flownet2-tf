from ..dataloader import *
from ..dataset_configs_S import FLYING_CHAIRS_DATASET_CONFIG
from ..training_schedules_S import LONG_SCHEDULE
from .flownet_s import FlowNetS


import os
os.environ["CUDA_VISIBLE_DEVICES"]="1"

checkpoint_path = "./checkpoints/FlowNetS/flownet-S.ckpt-0"


# Create a new network
net = FlowNetS()

# Load a batch of data
input_a, input_b, flow = load_val_batch(FLYING_CHAIRS_DATASET_CONFIG, 'validate', net.global_step)

# Train on the data
net.validate(
    log_dir='./logs/flownet_s_train',
    training_schedule=LONG_SCHEDULE,
    input_a=input_a,
    input_b=input_b,
    flow=flow,
    checkpoints=checkpoint_path
    # checkpoints={checkpoint_path : ('FlowNetS', 'FlowNetS'),}
)
