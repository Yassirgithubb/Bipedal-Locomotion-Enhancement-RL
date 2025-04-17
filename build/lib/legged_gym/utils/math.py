# isaac-gym
from isaacgym.torch_utils import quat_apply, normalize, quat_mul, quat_conjugate

# python
import torch
import numpy as np
from typing import Tuple


# @ torch.jit.script
def quat_apply_yaw(quat: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
    """Rotate a vector only around the yaw-direction.

    Args:
        quat (torch.Tensor): Input orientation to extract yaw from.
        vec (torch.Tensor): Input vector.

    Returns:
        torch.Tensor: Rotated vector.
    """
    quat_yaw = quat.clone().view(-1, 4)
    qx = quat_yaw[:, 0]
    qy = quat_yaw[:, 1]
    qz = quat_yaw[:, 2]
    qw = quat_yaw[:, 3]
    yaw = torch.atan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))
    quat_yaw[:, :2] = 0.0
    quat_yaw[:, 2] = torch.sin(yaw / 2)
    quat_yaw[:, 3] = torch.cos(yaw / 2)
    quat_yaw = normalize(quat_yaw)
    return quat_apply(quat_yaw, vec)


# @ torch.jit.script
def box_minus(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    """Implements box-minur operator (quaternion difference)
    https://docs.leggedrobotics.com/kindr/cheatsheet_latest.pdf

    Args:
        q1 (torch.Tensor): quaternion
        q2 (torch.Tensor): quaternion

    Returns:
        torch.Tensor: q2 box-minus q1
    """
    return 2.0 * (
        q1[:, -1].unsqueeze(-1) * q2[:, 0:3]
        - q2[:, -1].unsqueeze(-1) * q1[:, 0:3]
        + torch.cross(q1[:, 0:3], q2[:, 0:3], dim=1)
    )


# @ torch.jit.script
def wrap_to_pi(angles: torch.Tensor) -> torch.Tensor:
    """Wraps input angles (in radians) to the range [-pi, pi].

    Args:
        angles (torch.Tensor): Input angles.

    Returns:
        torch.Tensor: Angles in the range [-pi, pi].
    """
    angles %= 2 * np.pi
    angles -= 2 * np.pi * (angles > np.pi)
    return angles


# @ torch.jit.script
def torch_rand_sqrt_float(lower: float, upper: float, size: Tuple[int, int], device: str) -> torch.Tensor:
    """Randomly samples tensor from a triangular distribution.

    Args:
        lower (float): The lower range of the sampled tensor.
        upper (float): The upper range of the sampled tensor.
        size (Tuple[int, int]): The shape of the tensor.
        device (str): Device to create tensor on.

    Returns:
        torch.Tensor: Sampled tensor of shape :obj:`size`.
    """
    # create random tensor in the range [-1, 1]
    r = 2 * torch.rand(*size, device=device) - 1
    # convert to triangular distribution
    r = torch.where(r < 0.0, -torch.sqrt(-r), torch.sqrt(r))
    # rescale back to [0, 1]
    r = (r + 1.0) / 2.0
    # rescale to range [lower, upper]
    return (upper - lower) * r + lower


def angle_axis_from_quat(quat: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    w = quat[:, -1]
    w_sqr = w ** 2
    angle = 2 * torch.acos(w)
    axis = torch.zeros(quat.shape[0], 3, device=quat.device)
    axis[:, 0] = quat[:, 0] / torch.sqrt(1 - w_sqr)
    axis[:, 1] = quat[:, 1] / torch.sqrt(1 - w_sqr)
    axis[:, 2] = quat[:, 2] / torch.sqrt(1 - w_sqr)
    return angle, axis
