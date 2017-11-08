import os

from .training_schedules_S import LONG_SCHEDULE
from .flownet_s import FlowNetS
from ..dataloader import *
from ..dataset_configs import FLYING_CHAIRS_DATASET_CONFIG
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

# checkpoint_path = "./checkpoints/FlowNetS/flownet-S.ckpt-0"


# Create a new network
net = FlowNetS()

# Load a batch of data
input_a, input_b, flow = load_batch(FLYING_CHAIRS_DATASET_CONFIG, 'train', net.global_step)

# Train on the data
net.train(
    log_dir='./logs/flownet_s',
    training_schedule=LONG_SCHEDULE,
    input_a=input_a,
    input_b=input_b,
    flow=flow,
    # checkpoints=checkpoint_path
    # checkpoints={checkpoint_path : ('FlowNetS', 'FlowNetS'),}
)
