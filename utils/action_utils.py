import os
import sys
import time
import viser
import numpy as np
import torch
from scipy.spatial.transform import Rotation as R
import torch.nn.functional as F
import math

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)

from utils.rotation import rot6d_to_matrix

ROT_DIMS = {'joint_value': 1, 'rot_quat': 4, 'rot_6d': 6, 'rot_vec': 3, 'rot_euler': 3}


# def R_to_rotvec_torch(R: torch.Tensor, eps: float = 1e-7):
#     # R: (..., 3, 3), 正交矩阵
#     trace = R[..., 0,0] + R[..., 1,1] + R[..., 2,2]
#     cos = ((trace - 1.0) * 0.5).clamp(-1 + eps, 1 - eps)
#     theta = torch.acos(cos)  # (...,)

#     # 反对称部分的“vee”
#     vee = torch.stack([
#         R[..., 2,1] - R[..., 1,2],
#         R[..., 0,2] - R[..., 2,0],
#         R[..., 1,0] - R[..., 0,1]
#     ], dim=-1) * 0.5

#     # 小角度稳定化：theta / (2*sin theta)
#     small = theta < 1e-3
#     k = torch.empty_like(theta)
#     # 泰勒: 1 + θ^2/6 + 7θ^4/360 ~ θ/(2sinθ)
#     k[small] = 1.0 + (theta[small]**2)/6.0 + 7.0*(theta[small]**4)/360.0
#     k[~small] = theta[~small] / (2.0 * torch.sin(theta[~small]))

#     rotvec = k.unsqueeze(-1) * vee

#     # 半空间对齐（单帧版，时序请见下节）
#     idx = torch.argmax(rotvec.abs() > 1e-6, dim=-1)  # 找到第一个显著分量
#     gather = torch.gather(rotvec, -1, idx.unsqueeze(-1)).squeeze(-1)
#     sign = torch.where(gather < 0, -1.0, 1.0)
#     rotvec = rotvec * sign.unsqueeze(-1)

#     # 裁掉接近 π 的角
#     rot_angle = theta
#     rotvec = rotvec * ((rot_angle.clamp(max=math.pi - 1e-4)) / (rot_angle + 1e-12)).unsqueeze(-1)
#     return rotvec

def R_to_rotvec_torch(R: torch.Tensor, eps: float = 1e-7):
    # R: (..., 3, 3), 正交矩阵
    trace = R[..., 0, 0] + R[..., 1, 1] + R[..., 2, 2]
    cos = ((trace - 1.0) * 0.5).clamp(-1 + eps, 1 - eps)
    theta = torch.acos(cos)  # (...,)

    # 反对称部分的“vee”
    vee = torch.stack([
        R[..., 2, 1] - R[..., 1, 2],
        R[..., 0, 2] - R[..., 2, 0],
        R[..., 1, 0] - R[..., 0, 1]
    ], dim=-1) * 0.5

    # 小角度稳定化：theta / (2*sin theta)
    small = theta < 1e-3
    k = torch.empty_like(theta)
    # 泰勒: 1 + θ^2/6 + 7θ^4/360 ~ θ/(2sinθ)
    k[small] = 1.0 + (theta[small] ** 2) / 6.0 + 7.0 * (theta[small] ** 4) / 360.0
    k[~small] = theta[~small] / (2.0 * torch.sin(theta[~small]))

    rotvec = k.unsqueeze(-1) * vee

    # 半空间对齐（单帧版，时序请见下节）

    # idx = torch.argmax(rotvec.abs() > 1e-6, dim=-1)  # 找到第一个显著分量
    # gather = torch.gather(rotvec, -1, idx.unsqueeze(-1)).squeeze(-1)
    # sign = torch.where(gather < 0, -1.0, 1.0)
    # rotvec = rotvec * sign.unsqueeze(-1)
    # 5) 半空间对齐（避免 (n,θ) 与 (-n,θ) 随机跳变）
    _, idx = rotvec.abs().max(dim=-1, keepdim=True)  # (...,1)
    main_comp = torch.gather(rotvec, dim=-1, index=idx).squeeze(-1)  # (...)
    sign = torch.where(main_comp < 0, -1.0, 1.0).unsqueeze(-1)  # (...,1)
    rotvec = rotvec * sign

    # 裁掉接近 π 的角
    rot_angle = theta
    rotvec = rotvec * ((rot_angle.clamp(max=math.pi - 1e-4)) / (rot_angle + 1e-12)).unsqueeze(-1)
    return rotvec


def rotvec_to_R_torch(rv: torch.Tensor, eps: float = 1e-7):
    # rv: (..., 3)
    theta = rv.norm(dim=-1)  # (...)
    small = theta < 1e-5

    # 归一化轴
    u = torch.zeros_like(rv)
    u[~small] = rv[~small] / theta[~small].unsqueeze(-1)

    # 罗德里格公式
    ct = torch.cos(theta)[..., None, None]
    st = torch.sin(theta)[..., None, None]
    # 轴的叉乘矩阵
    ux, uy, uz = u[..., 0], u[..., 1], u[..., 2]
    K = torch.stack([
        torch.stack([torch.zeros_like(ux), -uz, uy], dim=-1),
        torch.stack([uz, torch.zeros_like(ux), -ux], dim=-1),
        torch.stack([-uy, ux, torch.zeros_like(ux)], dim=-1),
    ], dim=-2)

    I = torch.eye(3, device=rv.device, dtype=rv.dtype).expand_as(K)
    # 小角稳定化：sinθ/θ ~ 1 - θ^2/6，(1 - cosθ)/θ^2 ~ 1/2 - θ^2/24
    s_by_t = torch.ones_like(theta)
    one_m_c_by_t2 = 0.5 * torch.ones_like(theta)
    t2 = theta * theta
    s_by_t[~small] = torch.sin(theta[~small]) / theta[~small]
    one_m_c_by_t2[~small] = (1 - torch.cos(theta[~small])) / (t2[~small] + eps)

    s_by_t = s_by_t[..., None, None]
    one_m_c_by_t2 = one_m_c_by_t2[..., None, None]

    R = I + s_by_t * K + one_m_c_by_t2 * (K @ K)
    return R


# 单帧半空间对齐（数据增强/编码时）
# def halfspace_fix(rv: torch.Tensor, thr: float = 1e-6):
#     # rv: (...,3)
#     idx = torch.argmax(rv.abs() > thr, dim=-1)
#     key = torch.gather(rv, -1, idx.unsqueeze(-1)).squeeze(-1)
#     sign = torch.where(key < 0, -1.0, 1.0)
#     return rv * sign.unsqueeze(-1)


# 时序半空间对齐
def temporal_sign_align(rv_seq: torch.Tensor, thr: float = 1e-6):
    # rv_seq: (..., T, 3)
    out = [halfspace_fix(rv_seq[..., 0, :], thr)]
    for t in range(1, rv_seq.shape[-2]):
        cur = halfspace_fix(rv_seq[..., t, :], thr)
        prev = out[-1]
        # 若与上一帧夹角>90°，整体取反
        flip = (torch.sum(prev * cur, dim=-1) < 0).float().unsqueeze(-1)
        cur = cur * (1 - 2 * flip)
        out.append(cur)
    return torch.stack(out, dim=-2)


def halfspace_fix(rv: torch.Tensor, thr: float = 1e-6) -> torch.Tensor:
    """
    单帧半空间对齐：让 |rv| 最大的分量非负。
    rv: (..., 3)
    返回: (..., 3)
    """
    # 取绝对值最大的分量索引（float 上取 max，不是 bool）
    vals, idx = rv.abs().max(dim=-1, keepdim=True)  # (...,1)
    main = torch.gather(rv, dim=-1, index=idx).squeeze(-1)  # (...,)

    # 该分量为负则整体翻转
    sign = torch.where(main < 0, -1.0, 1.0).unsqueeze(-1)  # (...,1)

    # 如果向量几乎为 0（max(|rv|) < thr），不要翻转（固定为 +1），避免噪声抖动
    sign = torch.where(vals < thr, torch.ones_like(sign), sign)

    return rv * sign


def convert_q(hand_model, q: torch.Tensor, output_q_type):
    """
    Convert joint values to specified rotation representation for each joint.

    Args:
        hand_model: Hand model with get_joint_transform method and joint_orders attribute
        q: Input tensor of shape (B, DoF) containing joint values
        output_q_type: Desired rotation representation ('rot_quat', 'rot_6d', 'rot_vec', 'rot_euler')

    Returns:
        Tensor of shape (B, D*K) where K is the dimension of the rotation representation
    """
    assert output_q_type in ROT_DIMS.keys(), f"Unsupported action type: {output_q_type}"
    if len(q.shape) == 1:
        q = q[None, :]

    N = q.shape[0]
    D = len(hand_model.joint_orders)
    K = ROT_DIMS[output_q_type]
    assert q.shape[1] == D, f"Expected shape (N, {D}), got {q.shape}"
    output_q = torch.zeros((N, D * K), dtype=torch.float32, device=q.device)

    joint_transforms = hand_model.get_joint_transform(q)
    for idx, joint_name in enumerate(hand_model.joint_orders):
        assert joint_name in joint_transforms, f"Joint {joint_name} not found in joint_transforms"
        joint_matrix = joint_transforms[joint_name][:, :3, :3]
        joint_matrix_np = joint_matrix.cpu().numpy()
        joint_rotation = R.from_matrix(joint_matrix_np)  # 将 3×3 旋转矩阵（rotation matrix） 转换为一个 Rotation 对象。

        if output_q_type == 'rot_quat':  # 二义性
            quat_np = joint_rotation.as_quat(scalar_first=True)
            for i in range(1, len(quat_np)):  # Ensure quaternion continuity
                if np.dot(quat_np[i - 1], quat_np[i]) < 0:
                    quat_np[i] *= -1
            joint_rot = torch.from_numpy(quat_np).to(q.device)
        elif output_q_type == 'rot_6d':
            joint_rot = joint_matrix.mT.reshape(N, 9)[:, :6]
        elif output_q_type == 'rot_vec':
            # joint_rot = torch.from_numpy(joint_rotation.as_rotvec()).to(q.device)  # 用的Π
            joint_rot = R_to_rotvec_torch(joint_matrix)  # (N,3)
        elif output_q_type == 'rot_euler':
            joint_rot = torch.from_numpy(joint_rotation.as_euler('xyz')).to(q.device)
        else:
            raise NotImplementedError(f"Unknown action_type: {output_q_type}")

        output_q[:, idx * K: (idx + 1) * K] = joint_rot

    return output_q


def projection_q(hand_model, q: torch.Tensor, input_q_type):
    """
    Project rotation representations back to joint angles with hierarchical projection.

    Args:
        hand_model: Hand model with necessary methods and attributes
        q: Input tensor of shape (B, DoF * K) in specified rotation type
        input_q_type: Input rotation representation ('rot_quat', 'rot_6d', 'rot_vec', 'rot_euler')

    Returns:
        Tensor of shape (B, DoF) with projected joint angles
    """
    assert input_q_type in ROT_DIMS.keys(), f"Unsupported input type: {input_q_type}"
    if len(q.shape) == 1:
        q = q.unsqueeze(0)

    N = q.shape[0]
    D = len(hand_model.joint_orders)
    K = ROT_DIMS[input_q_type]
    assert q.shape[1] == D * K, f"Expected shape (N, {D * K}), got {q.shape}"

    joint_matrices = {}
    for idx, joint_name in enumerate(hand_model.joint_orders):
        q_slice = q[:, idx * K: (idx + 1) * K].cpu().numpy()
        if input_q_type == 'rot_quat':
            rot = R.from_quat(q_slice, scalar_first=True)
        elif input_q_type == 'rot_6d':
            rot = R.from_matrix(rot6d_to_matrix(q_slice))
        elif input_q_type == 'rot_vec':
            rot = R.from_rotvec(q_slice)
        elif input_q_type == 'rot_euler':
            rot = R.from_euler('xyz', q_slice)
        else:
            raise NotImplementedError(f"Unknown action_type: {input_q_type}")
        joint_matrices[joint_name] = rot.as_matrix()

    output_q = hand_model.get_canonical_q().unsqueeze(0).repeat(N, 1).to(q.device)
    lower, upper = hand_model.pk_chain.get_joint_limits()

    for joint_layer in hand_model.joint_layers:
        curr_transform = hand_model.get_joint_transform(output_q)
        for joint_name in joint_layer:
            curr_matrix = curr_transform[joint_name][:, :3, :3].cpu().numpy()
            pred_matrix = curr_matrix.swapaxes(-2, -1) @ joint_matrices[joint_name]  # root frame -> joint local frame
            pred_rotvec = torch.from_numpy(R.from_matrix(pred_matrix).as_rotvec()).to(q.device)
            joint_axis = hand_model.joint_axes[joint_name]
            joint_angle = (pred_rotvec * joint_axis).sum(-1)
            joint_index = hand_model.joint_orders.index(joint_name)
            output_q[:, joint_index] = torch.clamp(joint_angle, lower[joint_index], upper[joint_index])

    return output_q


# ---- rot_6d → R （Gram–Schmidt）----
def rot6d_to_R(x):  # x: (...,6)
    a1, a2 = x[..., :3], x[..., 3:6]
    b1 = F.normalize(a1, dim=-1, eps=1e-8)
    a2 = a2 - (b1 * a2).sum(dim=-1, keepdim=True) * b1
    b2 = F.normalize(a2, dim=-1, eps=1e-8)
    b3 = torch.cross(b1, b2, dim=-1)
    R = torch.stack([b1, b2, b3], dim=-1)  # (...,3,3)
    # 可选：保证右手系
    det = torch.linalg.det(R)
    if (det < 0).any():
        # 翻转 b3 修正
        b3 = torch.where(det.unsqueeze(-1).unsqueeze(-1) < 0, -b3, b3)
        R = torch.stack([b1, b2, b3], dim=-1)
    return R


# ---- quat(scalar-first) → R ----
def quat_to_R(q):  # q: (...,4) [w,x,y,z]
    q = F.normalize(q, dim=-1, eps=1e-8)
    w, x, y, z = q.unbind(-1)
    R = torch.stack([
        1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
        2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
        2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)
    ], dim=-1).reshape(q.shape[:-1] + (3, 3))
    return R


# ---- rotvec(轴角) → R （so(3) exp）----
def so3_exp(w, eps=1e-8):  # w: (...,3)
    th = torch.linalg.norm(w, dim=-1, keepdim=True)
    A = torch.where(th > eps, torch.sin(th) / th, 1 - th ** 2 / 6)
    B = torch.where(th > eps, (1 - torch.cos(th)) / (th ** 2), 0.5 - th ** 2 / 24)
    wx, wy, wz = w.unbind(-1);
    O = torch.zeros_like(wx)
    W = torch.stack([O, -wz, wy, wz, O, -wx, -wy, wx, O], dim=-1).reshape(w.shape[:-1] + (3, 3))
    I = torch.eye(3, device=w.device, dtype=w.dtype).expand(W.shape)
    return I + A[..., None] * W + B[..., None] * (W @ W)


# ---- euler(xyz) → R（请确保与数据一致的轴序）----
def euler_xyz_to_R(e):
    cx, cy, cz = torch.cos(e[..., 0]), torch.cos(e[..., 1]), torch.cos(e[..., 2])
    sx, sy, sz = torch.sin(e[..., 0]), torch.sin(e[..., 1]), torch.sin(e[..., 2])
    R = torch.stack([
        cy * cz, -cy * sz, sy,
        sx * sy * cz + cx * sz, -sx * sy * sz + cx * cz, -sx * cy,
        -cx * sy * cz + sx * sz, cx * sy * sz + sx * cz, cx * cy
    ], dim=-1).reshape(e.shape[:-1] + (3, 3))
    return R


# ---- 把 (B,T,24*rot_dims) 解码成 (B,T,24,3,3) ----
def decode_rotations_to_R(rot_tensor, action_type):
    B, T, D = rot_tensor.shape
    if action_type == 'rot_6d':
        x = rot_tensor.reshape(B, T, 24, 6);
        R = rot6d_to_R(x)
    elif action_type == 'rot_quat':
        x = rot_tensor.reshape(B, T, 24, 4);
        R = quat_to_R(x)
    elif action_type == 'rot_vec':
        x = rot_tensor.reshape(B, T, 24, 3);
        R = rotvec_to_R_torch(x)  # 使用最新的rotvec_to_R_torch
    elif action_type == 'rot_euler':
        x = rot_tensor.reshape(B, T, 24, 3);
        R = euler_xyz_to_R(x)
    else:
        raise NotImplementedError(action_type)
    return R  # (B,T,24,3,3)