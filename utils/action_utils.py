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


# 绝对→相对：给 (T, J*dim) 的旋转段，转相对
# def absolute_rot_to_relative(rot_seq, action_type, J=24, order='xyz'):
#     # rot_seq: (T, J*dim)
#     if not torch.is_tensor(rot_seq):
#         rot_seq = torch.tensor(rot_seq, dtype=torch.float32)
#     dim = ROT_DIMS[action_type]
#     T = rot_seq.shape[0]
#     x = rot_seq.view(T, J, dim)
#     R_abs = decode_rotations_to_R(x,action_type)
#     R_prev = R_abs[:-1]
#     R_curr = R_abs[1:]
#     R_rel = R_prev.transpose(-1, -2) @ R_curr               # (T-1,J,3,3)
#     rel = encode_from_R(R_rel, action_type, order=order)    # (T-1,J,dim)
#     # 首帧增量置零
#     zero = torch.zeros_like(rel[0:1])
#     rel = torch.cat([zero, rel], dim=0)                     # (T,J,dim)
#     return rel.view(T, J*dim)
# def absolute_rot_to_relative(rot_seq: torch.Tensor,
#                              action_type: str,
#                              J: int = 24,
#                              order: str = 'xyz') -> torch.Tensor:
#     """
#     输入:
#       rot_seq: (T, J*dim) 绝对旋转参数序列
#     输出:
#       (T, J*dim) 相对旋转参数序列（定义为 R_rel[t] = R_abs[t-1]^T R_abs[t]；首帧为单位）
#     """
#     if not torch.is_tensor(rot_seq):
#         rot_seq = torch.tensor(rot_seq, dtype=torch.float32)
#
#     rot_seq = rot_seq.contiguous()  # 更安全
#     T, JD = rot_seq.shape
#     dim = ROT_DIMS[action_type]
#     assert JD == J * dim, f"Shape mismatch: got {JD}, expected {J}*{dim}"
#
#     # (T,J,dim)
#     x = rot_seq.reshape(T, J, dim)
#
#     # 解到矩阵 (T,J,3,3)
#     R_abs = decode_rotations_to_R(x, action_type)
#     # 可选：数值投影到 SO(3)
#     # R_abs = project_to_so3(R_abs)
#
#     if T == 1:
#         # 只有一帧：全是单位相对旋转
#         I_rep = identity_rep(action_type, J, rot_seq.device, rot_seq.dtype)
#         return I_rep  # (1, J*dim)
#
#     # 相对增量 (T-1,J,3,3)
#     R_prev = R_abs[:-1]
#     R_curr = R_abs[1:]
#     R_rel = R_prev.transpose(-1, -2) @ R_curr
#     # 可选：再投影一次提升稳定性
#     # R_rel = project_to_so3(R_rel)
#
#     # 回编码 (T-1,J,dim)
#     rel = encode_from_R(R_rel, action_type, order=order)
#
#     # 首帧设为单位（按表示类型正确编码）
#     I_rep_1 = identity_rep(action_type, J, rel.device, rel.dtype).reshape(1, J, dim)
#     rel = torch.cat([I_rep_1, rel], dim=0)  # (T,J,dim)
#
#     return rel.reshape(T, J * dim)
def absolute_rot_to_relative(rot_seq, action_type, J=24, order='xyz'):
    """
    rot_seq: (T, J*dim) 或 (T, J, dim)
    return:  (T, J*dim)（与输入维度风格一致，首帧为0增量）
    """
    if not torch.is_tensor(rot_seq):
        rot_seq = torch.tensor(rot_seq, dtype=torch.float32)

    rot_seq = rot_seq.to(torch.float32)
    dim = ROT_DIMS[action_type]
    if rot_seq.ndim == 2:  # (T, J*dim)
        T = rot_seq.shape[0]
        assert rot_seq.shape[1] % dim == 0, "last dim not multiple of representation dim"
        J_infer = rot_seq.shape[1] // dim
        assert J_infer == J, f"J mismatch: got {J_infer}, expect {J}"
        x = rot_seq.reshape(T, J, dim)
    elif rot_seq.ndim == 3:  # (T, J, dim)
        T, J_infer, dim_infer = rot_seq.shape
        assert J_infer == J and dim_infer == dim
        x = rot_seq
    else:
        raise ValueError(f"rot_seq shape not supported: {rot_seq.shape}")

    R_abs = decode_rotations_to_R(x, action_type)   # (T, J, 3, 3)
    R_prev = R_abs[:-1]
    R_curr = R_abs[1:]
    R_rel = R_prev.transpose(-1, -2) @ R_curr       # (T-1, J, 3, 3)

    rel = encode_from_R(R_rel, action_type, order=order)  # (T-1, J, dim)
    zero = torch.zeros_like(rel[0:1])
    rel = torch.cat([zero, rel], dim=0)             # (T, J, dim)

    return rel.reshape(rel.shape[0], -1)            # (T, J*dim)


# 相对→绝对：推理/回放时累计
# def relative_rot_to_absolute(rel_seq, action_type, R0, order='xyz'):
#     # rel_seq: (T,J*dim), R0: (J,3,3) 初始绝对
#     dim = ROT_DIMS[action_type]
#     T = rel_seq.shape[0]
#     x = rel_seq.view(T, -1, dim)
#     R_rel = decode_rotations_to_R(x,action_type)
#     R_abs = []
#     R = R0
#     for t in range(T):
#         R = R @ R_rel[t]
#         R_abs.append(R)
#     return torch.stack(R_abs, dim=0)                         # (T,J,3,3)
def relative_rot_to_absolute(rel_seq: torch.Tensor,
                             action_type: str,
                             R0: torch.Tensor,
                             order: str = 'xyz') -> torch.Tensor:
    """
    输入:
      rel_seq: (T, J*dim) 相对旋转（R_rel[t] = R_abs[t-1]^T R_abs[t]）
      R0:     (J,3,3) 初始绝对旋转
    输出:
      (T, J, 3, 3) 绝对旋转
    """
    assert R0.shape == (R0.shape[0], 3, 3) and R0.dim() == 3, "R0 must be (J,3,3)"
    if not torch.is_tensor(rel_seq):
        rel_seq = torch.tensor(rel_seq, dtype=torch.float32)

    rel_seq = rel_seq.contiguous()
    T, JD = rel_seq.shape
    dim = ROT_DIMS[action_type]
    J = R0.shape[0]
    assert JD == J * dim, f"Shape mismatch: got {JD}, expected {J}*{dim}"

    x = rel_seq.reshape(T, J, dim)
    R_rel = decode_rotations_to_R(x, action_type)  # (T,J,3,3)
    # R_rel = project_to_so3(R_rel)

    # 累计
    R_abs_list = []
    R = R0.to(rel_seq.device)
    for t in range(T):
        R = R @ R_rel[t]
        # R = project_to_so3(R)  # 可选
        R_abs_list.append(R)

    return torch.stack(R_abs_list, dim=0)  # (T,J,3,3)

def encode_from_R(R: torch.Tensor, action_type: str, order: str = 'xyz') -> torch.Tensor:
    """
    R: (..., J, 3, 3) 旋转矩阵（正交，右手）
    return: (..., J, K) 其中 K = ROT_DIMS[action_type]
    """
    assert R.shape[-2:] == (3, 3), f"R should end with (3,3), got {R.shape}"
    J = R.shape[-3]

    if action_type == 'rot_6d':
        # 与 convert_q 中的实现严格对齐：
        # joint_rot = joint_matrix.mT.reshape(N, 9)[:, :6]
        # → 这里也用 mT 再 flatten 取前6个
        Rt = R.transpose(-1, -2)                             # (..., J, 3, 3)
        six = Rt.reshape(*Rt.shape[:-2], 9)[..., :6]         # (..., J, 6)
        return six

    elif action_type == 'rot_vec':
        # 稳定 log 映射（你项目里已有 R_to_rotvec_torch）
        rv = R_to_rotvec_torch(R)                            # (..., J, 3)
        return rv

    elif action_type == 'rot_quat':
        # 返回 scalar_first=True 的 (w,x,y,z)，与 convert_q 对齐
        q = matrix_to_quaternion_torch(R, scalar_first=True) # (..., J, 4)
        return q

    elif action_type == 'rot_euler':
        assert order.lower() == 'xyz', "目前只实现了 'xyz'"
        eul = matrix_to_euler_xyz_torch(R)                   # (..., J, 3) 弧度
        return eul

    else:
        raise ValueError(f"Unsupported action_type: {action_type}")

def matrix_to_quaternion_torch(R: torch.Tensor, scalar_first: bool = True, eps: float = 1e-8) -> torch.Tensor:
    """
    R: (..., 3, 3)
    返回 (w,x,y,z) 若 scalar_first=True，否则 (x,y,z,w)
    算法：基于 trace 的分支，避免数值不稳定；并投影保正交可选。
    """
    assert R.shape[-2:] == (3, 3)
    t = R[..., 0, 0] + R[..., 1, 1] + R[..., 2, 2]  # trace
    q = torch.zeros(*R.shape[:-2], 4, device=R.device, dtype=R.dtype)

    # 四种分支
    case1 = t > 0
    case2 = (R[..., 0, 0] >= R[..., 1, 1]) & (R[..., 0, 0] >= R[..., 2, 2]) & (~case1)
    case3 = (R[..., 1, 1] > R[..., 2, 2]) & (~case1) & (~case2)
    case4 = ~(case1 | case2 | case3)

    # 1) t > 0
    if case1.any():
        t1 = t[case1]
        s = torch.sqrt(t1 + 1.0) * 2.0
        qw = 0.25 * s
        qx = (R[case1, 2, 1] - R[case1, 1, 2]) / s
        qy = (R[case1, 0, 2] - R[case1, 2, 0]) / s
        qz = (R[case1, 1, 0] - R[case1, 0, 1]) / s
        q[case1] = torch.stack([qw, qx, qy, qz], dim=-1)

    # 2) R00 是最大对角
    if case2.any():
        s = torch.sqrt(1.0 + R[case2, 0, 0] - R[case2, 1, 1] - R[case2, 2, 2] + eps) * 2.0
        qw = (R[case2, 2, 1] - R[case2, 1, 2]) / s
        qx = 0.25 * s
        qy = (R[case2, 0, 1] + R[case2, 1, 0]) / s
        qz = (R[case2, 0, 2] + R[case2, 2, 0]) / s
        q[case2] = torch.stack([qw, qx, qy, qz], dim=-1)

    # 3) R11 是最大对角
    if case3.any():
        s = torch.sqrt(1.0 + R[case3, 1, 1] - R[case3, 0, 0] - R[case3, 2, 2] + eps) * 2.0
        qw = (R[case3, 0, 2] - R[case3, 2, 0]) / s
        qx = (R[case3, 0, 1] + R[case3, 1, 0]) / s
        qy = 0.25 * s
        qz = (R[case3, 1, 2] + R[case3, 2, 1]) / s
        q[case3] = torch.stack([qw, qx, qy, qz], dim=-1)

    # 4) R22 最大
    if case4.any():
        s = torch.sqrt(1.0 + R[case4, 2, 2] - R[case4, 0, 0] - R[case4, 1, 1] + eps) * 2.0
        qw = (R[case4, 1, 0] - R[case4, 0, 1]) / s
        qx = (R[case4, 0, 2] + R[case4, 2, 0]) / s
        qy = (R[case4, 1, 2] + R[case4, 2, 1]) / s
        qz = 0.25 * s
        q[case4] = torch.stack([qw, qx, qy, qz], dim=-1)

    # 归一化
    q = F.normalize(q, dim=-1, eps=eps)

    if scalar_first:
        return q  # (w,x,y,z)
    else:
        # 变成 (x,y,z,w)
        return torch.stack([q[..., 1], q[..., 2], q[..., 3], q[..., 0]], dim=-1)

def matrix_to_euler_xyz_torch(R: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    R: (..., 3, 3)
    返回 intrinsic-'xyz' 欧拉角 (a,b,c) 对应 Rx(a) Ry(b) Rz(c)，单位：弧度
    """
    assert R.shape[-2:] == (3, 3)
    r00, r01, r02 = R[..., 0, 0], R[..., 0, 1], R[..., 0, 2]
    r12, r22      = R[..., 1, 2], R[..., 2, 2]

    # b = asin(r02)
    b = torch.asin(torch.clamp(r02, -1.0, 1.0))
    cb = torch.cos(b)

    # 正常情况
    a = torch.atan2(-r12, r22)
    c = torch.atan2(-r01, r00)

    # gimbal lock: |cos(b)| ~ 0
    lock = cb.abs() < 1e-6
    if lock.any():
        # 设 c = 0，a 从另一对分量恢复：a = atan2(R[2,0], R[1,0])，或其他等价形式
        r20 = R[..., 2, 0]
        r10 = R[..., 1, 0]
        a_lock = torch.atan2(r20, r10)
        a = torch.where(lock, a_lock, a)
        c = torch.where(lock, torch.zeros_like(c), c)

    eul = torch.stack([a, b, c], dim=-1)  # (..., 3)
    return eul


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
# def decode_rotations_to_R(rot_tensor, action_type):
#     B, T, D = rot_tensor.shape
#     if action_type == 'rot_6d':
#         x = rot_tensor.reshape(B, T, 24, 6);
#         R = rot6d_to_R(x)
#     elif action_type == 'rot_quat':
#         x = rot_tensor.reshape(B, T, 24, 4);
#         R = quat_to_R(x)
#     elif action_type == 'rot_vec':
#         x = rot_tensor.reshape(B, T, 24, 3);
#         R = rotvec_to_R_torch(x)  # 使用最新的rotvec_to_R_torch
#     elif action_type == 'rot_euler':
#         x = rot_tensor.reshape(B, T, 24, 3);
#         R = euler_xyz_to_R(x)
#     else:
#         raise NotImplementedError(action_type)
#     return R  # (B,T,24,3,3)
def _to_BTJdim(rot_tensor: torch.Tensor, action_type: str, J: int):
    """将任意 (T,J*dim)/(T,J,dim)/(B,T,J*dim)/(B,T,J,dim) 规范化为 (B,T,J,dim)，并返回形状标记。"""
    x = rot_tensor
    dim = ROT_DIMS[action_type]

    if x.ndim == 2:  # (T, J*dim)
        T, D = x.shape
        assert D % dim == 0, "last dim not multiple of representation dim"
        J_infer = D // dim
        assert J_infer == J, f"J mismatch: got {J_infer}, expect {J}"
        x = x.reshape(1, T, J, dim)
        flags = dict(has_batch=False, has_time=True)

    elif x.ndim == 3:
        # 优先判断 (T, J, dim)
        if x.shape[-1] == dim:
            T, J_infer, dim_infer = x.shape
            assert J_infer == J and dim_infer == dim
            x = x.reshape(1, T, J, dim)
            flags = dict(has_batch=False, has_time=True)
        else:
            # 尝试 (B, T, J*dim)
            B, T, D = x.shape
            assert D % dim == 0, "last dim not multiple of representation dim"
            J_infer = D // dim
            assert J_infer == J, f"J mismatch: got {J_infer}, expect {J}"
            x = x.reshape(B, T, J, dim)
            flags = dict(has_batch=True, has_time=True)

    elif x.ndim == 4:  # (B, T, J, dim)
        B, T, J_infer, dim_infer = x.shape
        assert J_infer == J and dim_infer == dim
        flags = dict(has_batch=True, has_time=True)

    else:
        raise ValueError(f"Unsupported input shape: {tuple(x.shape)}")

    return x, flags  # x: (B,T,J,dim)

def _from_BTJdim(R: torch.Tensor, flags: dict):
    """将 (B,T,J,3,3) 还原回调用前的前缀维度。"""
    has_batch = flags.get('has_batch', True)
    has_time  = flags.get('has_time',  True)

    if has_batch and has_time:
        return R                    # (B,T,J,3,3)
    if (not has_batch) and has_time:
        return R.squeeze(0)         # (T,J,3,3)
    if has_batch and (not has_time):
        return R.squeeze(1)         # (B,J,3,3) —— 基本不会用到
    # 两者都没有的场景极少见
    return R.squeeze(0).squeeze(0)  # (J,3,3)

def decode_rotations_to_R(rot_tensor, action_type, J=24, order='xyz'):
    """
    支持输入：
      - (T, J*dim) / (T, J, dim)
      - (B, T, J*dim) / (B, T, J, dim)
    输出与输入前缀维度匹配的 (..., J, 3, 3)
    """
    x, flags = _to_BTJdim(rot_tensor, action_type, J)   # (B,T,J,dim)
    B, T, J, dim = x.shape

    if action_type == 'rot_6d':
        # Gram-Schmidt 还原
        a1 = x[..., 0:3]                      # (B,T,J,3)
        a2 = x[..., 3:6]
        b1 = F.normalize(a1, dim=-1, eps=1e-8)
        a2_proj = (b1 * (a2 * b1).sum(dim=-1, keepdim=True))
        b2 = F.normalize(a2 - a2_proj, dim=-1, eps=1e-8)
        b3 = torch.cross(b1, b2, dim=-1)
        R = torch.stack([b1, b2, b3], dim=-2) # (B,T,J,3,3)

    elif action_type == 'rot_quat':
        # 假设四元数为 (w,x,y,z)
        q = F.normalize(x, dim=-1, eps=1e-8)
        w, xq, yq, zq = q.unbind(dim=-1)
        ww, xx, yy, zz = w*w, xq*xq, yq*yq, zq*zq
        wx, wy, wz = w*xq, w*yq, w*zq
        xy, xz, yz = xq*yq, xq*zq, yq*zq
        R = torch.empty(B, T, J, 3, 3, device=q.device, dtype=q.dtype)
        R[..., 0, 0] = ww + xx - yy - zz
        R[..., 0, 1] = 2*(xy - wz)
        R[..., 0, 2] = 2*(xz + wy)
        R[..., 1, 0] = 2*(xy + wz)
        R[..., 1, 1] = ww - xx + yy - zz
        R[..., 1, 2] = 2*(yz - wx)
        R[..., 2, 0] = 2*(xz - wy)
        R[..., 2, 1] = 2*(yz + wx)
        R[..., 2, 2] = ww - xx - yy + zz

    elif action_type == 'rot_vec':
        R = rotvec_to_R_torch(x)              # 你的实现应支持任意前缀形状

    elif action_type == 'rot_euler':
        # 仅示例 intrinsic 'xyz'
        assert order.lower() == 'xyz'
        R = euler_xyz_to_R(x)                  # 你的实现应支持任意前缀形状

    else:
        raise ValueError(f"Unsupported action_type: {action_type}")

    return _from_BTJdim(R, flags)