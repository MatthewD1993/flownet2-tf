from .flowlib import flow_error, read_flow, visualize_flow
import os
import numpy as np
flow_file_path = os.path.abspath('./data/samples/0flow.flo')
flow = read_flow(flow_file_path)
print("max", np.max(flow))
print("min", np.min(flow))
visualize_flow(flow)
