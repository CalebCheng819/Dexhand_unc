from __future__ import annotations

import numpy as np
import torch
from scipy.spatial.transform import Rotation as R

from utils.rotation import rot6d_to_matrix

ROT_REPR_DIMS = {
    "joint_value": 3,
    "rot_euler": 3,
    "rot_vec": 3,
    "rot_quat": 4,  # xyzw
    "rot_6d": 6,
    "rot_mat": 9,
}


def get_rot_repr_dim(action_type: str) -> int:
    if action_type not in ROT_REPR_DIMS:
        raise ValueError(f"Unsupported action_type: {action_type}")
    return ROT_REPR_DIMS[action_type]


def encode_euler_xyz_np(euler_xyz: np.ndarray, action_type: str) -> np.ndarray:
    """euler_xyz: (T,3) -> repr: (T,K)"""
    if action_type in ("joint_value", "rot_euler"):
        return euler_xyz.astype(np.float32)

    rot = R.from_euler("xyz", euler_xyz.astype(np.float64))
    if action_type == "rot_vec":
        return rot.as_rotvec().astype(np.float32)
    if action_type == "rot_quat":
        return rot.as_quat().astype(np.float32)  # xyzw
    if action_type == "rot_mat":
        return rot.as_matrix().reshape(-1, 9).astype(np.float32)
    if action_type == "rot_6d":
        m = rot.as_matrix().astype(np.float32)  # (T,3,3)
        # col0=(T,3), col1=(T,3) → concat → (T,6); matches decode: a1=x[:3] (col0), a2=x[3:6] (col1)
        return np.concatenate([m[:, :, 0], m[:, :, 1]], axis=-1).astype(np.float32)

    raise ValueError(f"Unsupported action_type: {action_type}")


def repr_to_matrix_torch(x: torch.Tensor, action_type: str) -> torch.Tensor:
    """x: (...,K) -> R: (...,3,3)"""
    if action_type == "joint_value":
        action_type = "rot_euler"

    if action_type == "rot_mat":
        r = x.view(*x.shape[:-1], 3, 3)
        # project to SO(3)
        U, _, Vh = torch.linalg.svd(r)
        det = torch.det(U @ Vh)
        D = torch.diag_embed(torch.stack([torch.ones_like(det), torch.ones_like(det), det], dim=-1))
        return U @ D @ Vh

    if action_type == "rot_6d":
        a1 = x[..., :3]
        a2 = x[..., 3:6]
        b1 = torch.nn.functional.normalize(a1, dim=-1, eps=1e-8)
        a2 = a2 - (b1 * a2).sum(dim=-1, keepdim=True) * b1
        b2 = torch.nn.functional.normalize(a2, dim=-1, eps=1e-8)
        b3 = torch.cross(b1, b2, dim=-1)
        return torch.stack([b1, b2, b3], dim=-1)

    if action_type == "rot_quat":
        q = torch.nn.functional.normalize(x, dim=-1, eps=1e-8)
        xx, yy, zz = q[..., 0] * q[..., 0], q[..., 1] * q[..., 1], q[..., 2] * q[..., 2]
        xy, xz, yz = q[..., 0] * q[..., 1], q[..., 0] * q[..., 2], q[..., 1] * q[..., 2]
        xw, yw, zw = q[..., 0] * q[..., 3], q[..., 1] * q[..., 3], q[..., 2] * q[..., 3]

        m00 = 1 - 2 * (yy + zz)
        m01 = 2 * (xy - zw)
        m02 = 2 * (xz + yw)
        m10 = 2 * (xy + zw)
        m11 = 1 - 2 * (xx + zz)
        m12 = 2 * (yz - xw)
        m20 = 2 * (xz - yw)
        m21 = 2 * (yz + xw)
        m22 = 1 - 2 * (xx + yy)
        return torch.stack([m00, m01, m02, m10, m11, m12, m20, m21, m22], dim=-1).view(*x.shape[:-1], 3, 3)

    if action_type == "rot_euler":
        rx, ry, rz = x[..., 0], x[..., 1], x[..., 2]
        cx, cy, cz = torch.cos(rx), torch.cos(ry), torch.cos(rz)
        sx, sy, sz = torch.sin(rx), torch.sin(ry), torch.sin(rz)

        m00 = cy * cz
        m01 = -cy * sz
        m02 = sy
        m10 = sx * sy * cz + cx * sz
        m11 = -sx * sy * sz + cx * cz
        m12 = -sx * cy
        m20 = -cx * sy * cz + sx * sz
        m21 = cx * sy * sz + sx * cz
        m22 = cx * cy
        return torch.stack([m00, m01, m02, m10, m11, m12, m20, m21, m22], dim=-1).view(*x.shape[:-1], 3, 3)

    if action_type == "rot_vec":
        theta = torch.linalg.norm(x, dim=-1, keepdim=True).clamp_min(1e-8)
        k = x / theta
        kx, ky, kz = k[..., 0], k[..., 1], k[..., 2]
        K = torch.stack(
            [
                torch.zeros_like(kx), -kz, ky,
                kz, torch.zeros_like(kx), -kx,
                -ky, kx, torch.zeros_like(kx),
            ],
            dim=-1,
        ).view(*x.shape[:-1], 3, 3)
        I = torch.eye(3, device=x.device, dtype=x.dtype).expand(*x.shape[:-1], 3, 3)
        th = theta[..., None]
        return I + torch.sin(th) * K + (1 - torch.cos(th)) * (K @ K)

    raise ValueError(f"Unsupported action_type: {action_type}")


def rot_geodesic_torch(pred: torch.Tensor, gt: torch.Tensor, action_type: str) -> torch.Tensor:
    """pred/gt: (...,K) -> geodesic radians: (...)"""
    Rp = repr_to_matrix_torch(pred, action_type)
    Rg = repr_to_matrix_torch(gt, action_type)
    rel = Rg.transpose(-1, -2) @ Rp
    tr = rel[..., 0, 0] + rel[..., 1, 1] + rel[..., 2, 2]
    cos = torch.clamp((tr - 1.0) * 0.5, -1 + 1e-7, 1 - 1e-7)
    return torch.acos(cos)


def rot_geodesic_deg_torch(pred: torch.Tensor, gt: torch.Tensor, action_type: str) -> torch.Tensor:
    """pred/gt: (...,K) -> geodesic deg: (...)"""
    return rot_geodesic_torch(pred, gt, action_type) * (180.0 / np.pi)


def decode_to_euler_xyz_np(x: np.ndarray, action_type: str) -> np.ndarray:
    if action_type in ("joint_value", "rot_euler"):
        return x.astype(np.float32)
    if action_type == "rot_vec":
        return R.from_rotvec(x).as_euler("xyz").astype(np.float32)
    if action_type == "rot_quat":
        return R.from_quat(x).as_euler("xyz").astype(np.float32)
    if action_type == "rot_mat":
        return R.from_matrix(x.reshape(-1, 3, 3)).as_euler("xyz").astype(np.float32)
    if action_type == "rot_6d":
        m = rot6d_to_matrix(x)
        return R.from_matrix(m).as_euler("xyz").astype(np.float32)
    raise ValueError(f"Unsupported action_type: {action_type}")
