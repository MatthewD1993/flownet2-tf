import os
from .training_schedules_S import TEST_SCHEDULE
from .flownet_s import FlowNetS
from ..dataloader import *
from .dataset_configs_S import FLYING_CHAIRS_DATASET_CONFIG
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

# checkpoint_path = "./logs/flownet_s/model.ckpt-1200000"
checkpoint_path = "./checkpoints/FlowNetS/flownet-S.ckpt-0"

set_global_step = True
# with tf.variable_scope("Global") as scope:
if set_global_step:
    global_step = tf.get_variable(name='global_step', shape=(), trainable=False, dtype=tf.int64,
                                  initializer=tf.constant_initializer(1200000))
else:
    global_step = None
        # net.global_step.assign(1200000)


# Create a new network
net = FlowNetS(global_step=global_step)

# Load a batch of data
input_a, input_b, flow = load_batch(FLYING_CHAIRS_DATASET_CONFIG, 'train', net.global_step)

# Train on the data
net.restore_train(
    log_dir='./logs/flownet_s_test',
    training_schedule=TEST_SCHEDULE,
    input_a=input_a,
    input_b=input_b,
    flow=flow,
    checkpoints=checkpoint_path,
    set_global_step=set_global_step
)
