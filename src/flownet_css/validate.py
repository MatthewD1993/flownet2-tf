from ..dataloader import *
from ..dataset_configs import FLYING_CHAIRS_DATASET_CONFIG
from ..training_schedules import LONG_SCHEDULE
from .flownet_css import FlowNetCSS


import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Create a new network
net = FlowNetCSS()

# Load a batch of data
input_a, input_b, flow = load_val_batch(FLYING_CHAIRS_DATASET_CONFIG, 'validate', net.global_step)

# Train on the data
net.validate(
    log_dir='./logs/flownet_2',
    training_schedule=LONG_SCHEDULE,
    input_a=input_a,
    input_b=input_b,
    flow=flow,
    # Load trained weights for CSS and SD parts of network
    checkpoints='./checkpoints/FlowNetCSS-ft-sd/flownet-CSS-ft-sd.ckpt-0'
    # checkpoints={
    #     './checkpoints/FlowNet2/flownet-2.ckpt-0': ('FlowNet2', 'FlowNet2'),
    #     # './checkpoints/FlowNetCSS-ft-sd/flownet-CSS-ft-sd.ckpt-0': ('FlowNet2/FlowNetCSS', 'FlowNet2'),
    #     # './checkpoints/FlowNetSD/flownet-SD.ckpt-0': ('FlowNet2/FlowNetSD', 'FlowNet2'),
    # }
)
