import torch
import numpy as np

# x
def get_roll_mtx(roll: float, device: torch.device, dtype: torch.dtype):
    roll = torch.tensor(roll)
    roll = torch.deg2rad(roll)
    return torch.tensor([[1., 0., 0.],
                           [0., torch.cos(roll), -torch.sin(roll)],
                           [0., torch.sin(roll), torch.cos(roll)]],
                         dtype=dtype, 
                         device=device) 

# y
def get_pitch_mtx(pitch: float, device: torch.device, dtype: torch.dtype):
    pitch = torch.tensor(pitch)
    pitch = torch.deg2rad(pitch)
    return torch.tensor([[torch.cos(pitch), 0., torch.sin(pitch)],
                           [0., 1., 0.],
                           [-torch.sin(pitch), 0., torch.cos(pitch)]],
                         dtype=dtype)

# z
def get_yaw_mtx(yaw: float, device: torch.device, dtype: torch.dtype):
    yaw = torch.tensor(yaw)
    yaw = torch.deg2rad(yaw)
    return torch.tensor([[torch.cos(yaw), -torch.sin(yaw), 0.],
                           [torch.sin(yaw), torch.cos(yaw), 0.],
                           [0., 0., 1.]],
                         dtype=dtype)

def get_rt_mtx(roll: float, pitch: float, yaw: float,
               device: torch.device, dtype: torch.dtype):
    RX = get_roll_mtx(roll, device, dtype)
    RZ = get_yaw_mtx(yaw, device, dtype)
    RY = get_pitch_mtx(pitch, device, dtype)

    R = torch.mm(RZ, RY)
    R = torch.mm(R, RX)

    return R

def get_upward_with_dev(rot: float, x_dev: float, z_dev: float,
                        device: torch.device = torch.device('cpu'),
                        dtype: torch.dtype = torch.float64):
    return get_rt_mtx(x_dev, rot, z_dev, device, dtype)

def get_random_upward(x_dev: float = 0., z_dev: float = 0.):
    random_angle = torch.randint(high=90, size=(1,)).float()
    return get_upward_with_dev(float(random_angle), x_dev, z_dev), random_angle
    # return  get_upward_with_dev(0., x_dev, z_dev), 0.

def rand_rot_mtx():
    roll = torch.randint(high=90, size=(1,)).to(torch.float64)
    yaw = torch.randint(high=90, size=(1,)).to(torch.float64)
    pitch = torch.randint(high=90, size=(1,)).to(torch.float64)
    return get_rt_mtx(roll, pitch, yaw)

def get_rotation_error(gt_R, R):
    assert abs((np.trace(torch.matmul(gt_R.T, R)) - 1.0 - 1e-7) / 2.0) <= 1
    return np.arccos((np.trace(torch.matmul(gt_R.T, R)) - 1.0 - 1e-7) / 2.0) * 180.0 / np.pi

def validate_sol(R, t, Rgt, tgt):
    return get_rotation_error(Rgt, R), (t - tgt).norm()
