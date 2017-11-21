from ..dataloader import load_batch
from ..dataset_configs_S import FLYING_CHAIRS_DATASET_CONFIG
from .training_schedules_c import LONG_SCHEDULE
from .flownet_c import FlowNetC
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

# Create a new network
net = FlowNetC()

# Load a batch of data
input_a, input_b, flow = load_batch(FLYING_CHAIRS_DATASET_CONFIG, 'train', net.global_step)

# Train on the data
net.train(
    log_dir='./logs/flownet_c',
    training_schedule=LONG_SCHEDULE,
    input_a=input_a,
    input_b=input_b,
    flow=flow
)
