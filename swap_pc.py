import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import os
import numpy as np

model_state_dict = torch.load(
    "/NAS/spa176/papr-retarget/experiments/hummingbird-start-1/model.pth"
)
point_cloud = torch.load("/NAS/spa176/papr-retarget/rotated_bird_pc.pth")


for step, state_dict in model_state_dict.items():
    print(state_dict.keys())
    assert point_cloud.shape == state_dict["points"].shape
    state_dict["points"] = point_cloud * 10.0

torch.save(model_state_dict, "/NAS/spa176/papr-retarget/experiments/hummingbird-start-1/model.pth")
