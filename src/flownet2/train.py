from ..dataloader import load_batch
from ..dataset_configs import FLYING_CHAIRS_DATASET_CONFIG
from ..training_schedules import LONG_SCHEDULE
from .flownet2 import FlowNet2


import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Create a new network
net = FlowNet2()

# Load a batch of data
input_a, input_b, flow = load_batch(FLYING_CHAIRS_DATASET_CONFIG, 'train', net.global_step)

# Train on the data
net.train(
    log_dir='./logs/flownet_2_lr1e-4',
    training_schedule=LONG_SCHEDULE,
    input_a=input_a,
    input_b=input_b,
    flow=flow,
    # Load trained weights for CSS and SD parts of network
    checkpoints={
        './logs/flownet_2_lr1e-4/91986/model.ckpt-91986': ('FlowNet2', 'FlowNet2'),
        # './checkpoints/FlowNetCSS-ft-sd/flownet-CSS-ft-sd.ckpt-0': ('FlowNet2/FlowNetCSS', 'FlowNet2'),
        # './checkpoints/FlowNetSD/flownet-SD.ckpt-0': ('FlowNet2/FlowNetSD', 'FlowNet2'),
    }
)
