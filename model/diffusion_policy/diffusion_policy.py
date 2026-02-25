import time
import numpy as np
import torch
import torch.nn as nn
from diffusers import DDPMScheduler, DDIMScheduler

from model.diffusion_policy.conditional_unet1d import ConditionalUnet1D
from utils.action_utils import ROT_DIMS,project_to_rotmat

from torch_geometric.nn import GCNConv, global_mean_pool
import torch.nn.functional as F


# ====== 手指关节分组（从 URDF 抽出来的链结构） ======

JOINT_NAMES = [
    "WRJ1", "WRJ2",                     # 2 wrist DoF

    "FFJ4", "FFJ3", "FFJ2", "FFJ1",     # index 指 (First finger)
    "MFJ4", "MFJ3", "MFJ2", "MFJ1",     # middle 指
    "RFJ4", "RFJ3", "RFJ2", "RFJ1",     # ring 指
    "LFJ5", "LFJ4", "LFJ3", "LFJ2", "LFJ1",  # little 指 + metacarpal
    "THJ5", "THJ4", "THJ3", "THJ2", "THJ1",  # thumb
]
NAME_TO_IDX = {n: i for i, n in enumerate(JOINT_NAMES)}

# 在每个手指内部，从掌根到指尖的 joint 链
JOINT_CHAINS = {
    "wrist":  ["WRJ1", "WRJ2"],  # 也可只把 WRJ1 当 depth 0, WRJ2 当 depth 1

    "index":  ["FFJ4", "FFJ3", "FFJ2", "FFJ1"],
    "middle": ["MFJ4", "MFJ3", "MFJ2", "MFJ1"],
    "ring":   ["RFJ4", "RFJ3", "RFJ2", "RFJ1"],
    "little": ["LFJ5", "LFJ4", "LFJ3", "LFJ2", "LFJ1"],
    "thumb":  ["THJ5", "THJ4", "THJ3", "THJ2", "THJ1"],
}
# 1) 每个关节 → 它在同一条链上的上一个关节（根部 parent = None）
JOINT_PARENT = {}
for chain in JOINT_CHAINS.values():
    for i, jname in enumerate(chain):
        if i == 0:
            JOINT_PARENT[jname] = None      # 链的第一个：没有 parent
        else:
            JOINT_PARENT[jname] = chain[i - 1]  # 其余：上一个就是 parent

# 2) 把 parent 映射成 index，方便快速查
PARENT_IDX = [-1] * len(JOINT_NAMES)          # 默认 -1 表示没有 parent
for jname in JOINT_NAMES:
    j_idx = NAME_TO_IDX[jname]
    parent = JOINT_PARENT.get(jname, None)
    if parent is None:
        PARENT_IDX[j_idx] = -1
    else:
        PARENT_IDX[j_idx] = NAME_TO_IDX[parent]

def build_edge_index():
    """
    构建一个 24 关节的无向图：
    - 每个 JOINT_CHAINS 里面相邻两关节连边
    - 再补一条反向边，变成无向
    返回 edge_index: (2, E)
    """
    name_to_idx = {n: i for i, n in enumerate(JOINT_NAMES)}
    edges = []

    for chain in JOINT_CHAINS.values():
        for i in range(len(chain) - 1):
            u = name_to_idx[chain[i]]
            v = name_to_idx[chain[i + 1]]
            edges.append((u, v))
            edges.append((v, u))  # 无向边

    edge_index = torch.tensor(edges, dtype=torch.long).t()  # (2, E)
    return edge_index
def build_joint_depths():
    """给每个关节一个“在该手指中的深度”：0 = 根部，1 = 第二节，…"""
    joint_to_depth = {}
    for _, joints in JOINT_CHAINS.items():
        for d, jname in enumerate(joints):
            joint_to_depth[jname] = d
    return joint_to_depth


def build_joint_depths():
    joint_to_depth = {}
    for chain_name, joints in JOINT_CHAINS.items():
        for d, jname in enumerate(joints):
            joint_to_depth[jname] = d
    return joint_to_depth

class JointGNNEncoder(nn.Module):
    def __init__(self, node_in_dim, hidden_dim, out_dim):
        super().__init__()
        # 预先构建单个手的边
        base_edge_index = build_edge_index()      # (2, E)
        # 注册成 buffer，方便自动搬到 GPU 上
        self.register_buffer("base_edge_index", base_edge_index)

        self.conv1 = GCNConv(node_in_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, out_dim)

    def forward(self, node_feats):
        """
        node_feats: (B, N, C_in), N=24
        返回:
            global_feat: (B, out_dim)
        """
        B, N, C_in = node_feats.shape
        device = node_feats.device

        # (1) 展开 batch 维度： (B*N, C_in)
        x = node_feats.reshape(B * N, C_in)

        # (2) 把 24 关节的 edge_index 复制 B 份，并加 offset
        base_edge_index = self.base_edge_index.to(device)  # (2, E)
        E = base_edge_index.shape[1]

        edge_index_list = []
        batch_index_list = []

        for b in range(B):
            offset = b * N
            ei_b = base_edge_index + offset           # (2, E)
            edge_index_list.append(ei_b)
            # 这个 batch 张量用来告诉 global_mean_pool 每 N 个节点属于同一张图
            batch_index_list.append(
                torch.full((N,), b, dtype=torch.long, device=device)
            )

        edge_index = torch.cat(edge_index_list, dim=1)       # (2, B*E)
        batch_index = torch.cat(batch_index_list, dim=0)     # (B*N,)

        # (3) 两层 GCNConv
        h = self.conv1(x, edge_index)    # (B*N, hidden_dim)
        h = F.relu(h)
        h = self.conv2(h, edge_index)    # (B*N, out_dim)
        h = F.relu(h)

        # (4) 对每个图做 mean pooling -> (B, out_dim)
        global_feat = global_mean_pool(h, batch_index)

        return global_feat





class DiffusionPolicy(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.action_type = cfg.dataset.action_type
        self.obs_dim = cfg.dataset.obs_dim
        self.act_dim = cfg.dataset.act_dim
        self.obs_horizon = cfg.model.obs_horizon
        self.act_horizon = cfg.model.act_horizon
        self.pred_horizon = cfg.model.pred_horizon
        self.num_diffusion_iters = cfg.model.num_diffusion_iters
        self.diffusion_prediction_type = cfg.model.diffusion_prediction_type
        #  新增：层数（1 = 平铺；2 = 根 vs 其余；>=3 = 多层）
        self.hier_num_levels = getattr(cfg.model, "hier_num_levels", 1)
        self.env_act_dim = cfg.dataset.env_act_dim
        #统一global维度
        self.obs_cond_dim = self.obs_dim * self.obs_horizon
        self.low_feat_dim = getattr(cfg.model, "low_feat_dim", self.obs_cond_dim)
        self.hier_context_mode = getattr(cfg.model, "hier_context_mode", "all_lower")  # 'all_lower' or 'parent_only'
        self.global_cond_dim = self.obs_cond_dim + self.low_feat_dim
        # self.hier_levels: List[List[(start, end, jname)]]
        self.hier_levels = []  # List[List[(start, end, jname)]]
        self.level_joint_indices: List[List[int]]  # 每层的关节 index（0..23）
        if self.action_type == 'joint_value1':
            self.noise_pred_net = ConditionalUnet1D(
                input_dim=self.act_dim,  # act_horizon is not used (U-Net doesn't care)
                global_cond_dim=self.obs_dim * self.obs_horizon,
                # global_cond_dim=self.global_cond_dim,  # ★ 改这里
                diffusion_step_embed_dim=cfg.model.diffusion_step_embed_dim,
                down_dims=cfg.model.unet_dims,
            )
        else:#分层处理不同维度的动作
    
            env_act_dim = cfg.dataset.env_act_dim
            #self.sub_ranges = [(0, env_act_dim)] if env_act_dim > 0 else []
            rot_dims = ROT_DIMS[cfg.dataset.action_type]
            self.rot_dims = rot_dims  # 记录下来后面要用
            assert self.act_dim - env_act_dim == 24 * rot_dims, \
                f"Action dimension mismatch! Got {self.act_dim}, expected {cfg.dataset.env_act_dim + 24 * rot_dims}."
            if self.hier_num_levels==1:
                self.sub_ranges = [(0, env_act_dim)] if env_act_dim > 0 else []
                self.sub_ranges.extend([(env_act_dim + idx * rot_dims, env_act_dim + (idx + 1) * rot_dims) for idx in range(24)])
                self.noise_pred_net = nn.ModuleList([
                    ConditionalUnet1D(
                        input_dim=(end - start),
                        global_cond_dim=self.obs_dim * self.obs_horizon,
                        diffusion_step_embed_dim=cfg.model.diffusion_step_embed_dim,
                        down_dims=cfg.model.unet_dims,
                    ) for start, end in self.sub_ranges
                ])

            else:
                self.env_act_dim = cfg.dataset.env_act_dim
                rot_dims = ROT_DIMS[cfg.dataset.action_type]

                assert self.act_dim - self.env_act_dim == 24 * rot_dims

                # 1) 深度分层
                joint_to_depth = build_joint_depths()
                self.hier_levels = self._build_hier_levels(
                    joint_to_depth=joint_to_depth,
                    num_levels=self.hier_num_levels,
                    env_act_dim=self.env_act_dim,
                    rot_dims=rot_dims,
                )

                # 2) 每层有哪些 joint index
                self.level_joint_indices = []  # List[List[int]]
                for level_ranges in self.hier_levels:
                    j_idx_list = []
                    for (_, _, jname) in level_ranges:
                        j_idx_list.append(NAME_TO_IDX[jname])
                    self.level_joint_indices.append(j_idx_list)

                # 3) 为每一层、每个关节建 Unet，输入 = 本关节 rot + 所有低层关节 rot
                if self.hier_context_mode=="all_lower":
                    self.rot_dims = rot_dims
                    self.noise_pred_levels = nn.ModuleList()

                    for level_id, level_ranges in enumerate(self.hier_levels):
                        # 统计所有低层关节数量
                        lower_joint_count = sum(len(self.level_joint_indices[l]) for l in range(level_id))
                        extra_in_dim = lower_joint_count * rot_dims

                        level_nets = nn.ModuleList()
                        for (start, end, jname) in level_ranges:
                            cur_dim = end - start  # = rot_dims
                            input_dim = cur_dim + extra_in_dim

                            net = ConditionalUnet1D(
                                input_dim=input_dim,
                                global_cond_dim=self.obs_dim * self.obs_horizon,
                                diffusion_step_embed_dim=cfg.model.diffusion_step_embed_dim,
                                down_dims=cfg.model.unet_dims,
                            )
                            level_nets.append(net)

                        self.noise_pred_levels.append(level_nets)
                elif self.hier_context_mode=="parent_only":
                    self.rot_dims = rot_dims
                    self.noise_pred_levels = nn.ModuleList()

                    for level_id, level_ranges in enumerate(self.hier_levels):
                        level_nets = nn.ModuleList()

                        for (start, end, jname) in level_ranges:
                            cur_dim = end - start  # 一般就是 rot_dims
                            j_idx = NAME_TO_IDX[jname]
                            parent_idx = PARENT_IDX[j_idx]

                            if parent_idx == -1:
                                # 根关节：没有上一节，只看自己
                                extra_in_dim = 0
                            else:
                                # 非根关节：额外多 1 个 parent 的 rot
                                extra_in_dim = self.rot_dims  # 只上一节，不是所有低层

                            input_dim = cur_dim + extra_in_dim  # root: 6, 其它: 12 (以 rot6d 为例)

                            net = ConditionalUnet1D(
                                input_dim=input_dim,
                                global_cond_dim=self.obs_dim * self.obs_horizon,
                                diffusion_step_embed_dim=cfg.model.diffusion_step_embed_dim,
                                down_dims=cfg.model.unet_dims,
                            )
                            level_nets.append(net)

                        self.noise_pred_levels.append(level_nets)

                # # 1) env 动作：一个单独 Unet（如果有的话）
                # if self.env_act_dim > 0:
                #     self.env_unet = ConditionalUnet1D(
                #         input_dim=self.env_act_dim,
                #         #global_cond_dim=self.obs_dim * self.obs_horizon,
                #         global_cond_dim=self.global_cond_dim,  # ★ 也用统一 cond 维度
                #         diffusion_step_embed_dim=cfg.model.diffusion_step_embed_dim,
                #         down_dims=cfg.model.unet_dims,
                #     )
                # else:
                #     self.env_unet = None
                #     # 2) 为 24 个关节按深度分层
                #     joint_to_depth = build_joint_depths()
                #     self.hier_levels = self._build_hier_levels(
                #         joint_to_depth=joint_to_depth,
                #         num_levels=self.hier_num_levels,
                #         env_act_dim=self.env_act_dim,
                #         rot_dims=rot_dims,
                #     )
                #     # hier_levels: List[List[(start, end, joint_name)]]
                #     # 记录每层有哪些 joint idx
                #     self.level_joint_indices = []
                #     for level_ranges in self.hier_levels:
                #         idxs = [NAME_TO_IDX[jname] for (_, _, jname) in level_ranges]
                #         self.level_joint_indices.append(idxs)
                #     # 3) 为每一层、每个关节建一个 Unet
                #     self.noise_pred_levels = nn.ModuleList()
                #     for level_ranges in self.hier_levels:
                #         level_nets = nn.ModuleList()
                #         for (start, end, jname) in level_ranges:
                #             level_nets.append(
                #                 ConditionalUnet1D(
                #                     input_dim=(end - start),  # = rot_dims
                #                     # global_cond_dim=self.obs_dim * self.obs_horizon,
                #                     global_cond_dim=self.global_cond_dim,
                #                     diffusion_step_embed_dim=cfg.model.diffusion_step_embed_dim,
                #                     down_dims=cfg.model.unet_dims,
                #                 )
                #             )
                #         self.noise_pred_levels.append(level_nets)
                #
                #     # 4) 为每个 level>=1 建 MLP，把低层关节编码成 low_feat
                #     self.low_cond_mlps = nn.ModuleList()
                #     for level_id in range(1, self.hier_num_levels):
                #         # 统计所有 < level_id 层的关节数量
                #         lower_joint_count = sum(len(self.level_joint_indices[l]) for l in range(level_id))
                #         in_dim = lower_joint_count * self.rot_dims  # 我们后面会对时间平均
                #         mlp = nn.Sequential(
                #             nn.Linear(in_dim, self.low_feat_dim),
                #             nn.ReLU(),
                #             nn.Linear(self.low_feat_dim, self.low_feat_dim),
                #         )
                #         self.low_cond_mlps.append(mlp)

        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=cfg.model.num_diffusion_iters,
            beta_schedule='squaredcos_cap_v2',
            clip_sample=cfg.model.scheduler_clip_sample,
            prediction_type=self.diffusion_prediction_type
        )
        self.noise_ddim_scheduler = DDIMScheduler(
            num_train_timesteps=cfg.model.num_diffusion_iters,
            beta_schedule='squaredcos_cap_v2',
            clip_sample=cfg.model.scheduler_clip_sample,
            prediction_type=self.diffusion_prediction_type,
        )
    #  新增：根据关节深度 + hier_num_levels 生成层级 ranges

    def _build_hier_levels(self, joint_to_depth, num_levels, env_act_dim, rot_dims):
        """
        返回：List[level_id] = List[(start, end, joint_name)]
        规则：
          - num_levels = 1: 所有关节都放在 level 0
          - num_levels = 2: depth=0 -> level0, depth>=1 -> level1
          - num_levels >=3: depth=0..L-2 -> level0..L-2, depth>=L-1 -> level(L-1)
        """
        levels = [[] for _ in range(max(1, num_levels))]

        for jname in JOINT_NAMES:
            j_idx = NAME_TO_IDX[jname]  # 0..23
            depth = joint_to_depth[jname]  # 每根手指内部的 0,1,2,...

            level_id = min(depth, num_levels - 1)

            start = env_act_dim + j_idx * rot_dims
            end = start + rot_dims
            levels[level_id].append((start, end, jname))

        return levels

    def forward(self, obs_seq, action_seq):
        device = obs_seq.device
        B = obs_seq.shape[0]

        # observation as FiLM conditioning
        obs_cond = obs_seq.flatten(start_dim=1)  # (B, obs_horizon * obs_dim)

        # sample noise to add to actions & diffusion iteration for each data point
        noise = torch.randn((B, self.pred_horizon, self.act_dim), device=device)
        timesteps = torch.randint(0, self.noise_scheduler.config.num_train_timesteps, (B,), device=device).long()

        # add noise to the clean actions according to the noise magnitude at each diffusion iteration
        noisy_action_seq = self.noise_scheduler.add_noise(action_seq, noise, timesteps)
        if self.action_type == 'joint_value1':
            pred = self.noise_pred_net(noisy_action_seq, timesteps, global_cond=obs_cond)
        else:
            if self.hier_num_levels==1:
                sub_noisy_action_seqs = [noisy_action_seq[:, :, start: end] for start, end in self.sub_ranges]
                pred = []
                for idx, sub_noise_pred_net in enumerate(self.noise_pred_net):
                    sub_pred = sub_noise_pred_net(sub_noisy_action_seqs[idx], timesteps, global_cond=obs_cond)
                    pred.append(sub_pred)
                pred = torch.cat(pred, dim=-1)
            else:
                # ========= 分层版本：每个关节输入 = 当前关节 rot + 所有低层关节 rot =========
                pred = torch.zeros_like(noisy_action_seq, device=device)

                # 1) env 动作（如果有）
                if  self.env_act_dim > 0:
                    sub_noisy_env = noisy_action_seq[:, :, :self.env_act_dim]
                    sub_pred_env = self.env_unet(
                        sub_noisy_env,
                        timesteps,
                        global_cond=obs_cond,  # 只用 obs_cond
                    )  # (B, T, env_act_dim)
                    pred[:, :, :self.env_act_dim] = sub_pred_env

                # 2) 预切所有关节的 noisy rot
                joint_rot_slices = {}
                for jname in JOINT_NAMES:
                    j_idx = NAME_TO_IDX[jname]
                    s = self.env_act_dim + j_idx * self.rot_dims
                    e = s + self.rot_dims
                    joint_rot_slices[j_idx] = noisy_action_seq[:, :, s:e]  # (B, T, rot_dims)
                if self.hier_context_mode == "all_lower":
                    # 3) 按层处理：低层 joint 的 rot 直接拼接到输入里
                    for level_id, (level_ranges, level_nets) in enumerate(
                            zip(self.hier_levels, self.noise_pred_levels)
                    ):
                        # 收集所有“更低层”的 joint index
                        lower_joint_indices = []
                        for l in range(level_id):
                            lower_joint_indices.extend(self.level_joint_indices[l])

                        if len(lower_joint_indices) > 0:
                            lower_parts = [joint_rot_slices[j_idx] for j_idx in lower_joint_indices]
                            # (B, T, lower_count * rot_dims)
                            lower_feat = torch.cat(lower_parts, dim=-1)
                        else:
                            lower_feat = None

                        # 当前层每个关节：输入 = 当前 rot + 所有低层 rot
                        for (net, (start, end, jname)) in zip(level_nets, level_ranges):
                            cur = noisy_action_seq[:, :, start:end]  # (B, T, rot_dims)

                            if lower_feat is not None:
                                # (B, T, rot_dims * (1 + lower_count))，和 __init__ 里的 input_dim 对齐
                                net_input = torch.cat([cur, lower_feat], dim=-1)
                            else:
                                net_input = cur  # 第 0 层，没有低层

                            sub_pred = net(
                                net_input,
                                timesteps,
                                global_cond=obs_cond,  # global_cond 只用 obs_cond
                            )  # (B, T, input_dim)

                            # 兼容某些实现里 squeeze 掉 batch 维的情况
                            if sub_pred.dim() == 2:
                                sub_pred = sub_pred.unsqueeze(0)  # (1, T, C)

                            # 只取“当前关节”对应的那 rot_dims 维写回
                            sub_pred_joint = sub_pred[:, :, : (end - start)]  # (B, T, rot_dims)
                            pred[:, :, start:end] = sub_pred_joint
                elif self.hier_context_mode == "parent_only":
                    # 分层：主要是决定调用顺序，但每个关节只看自己的 parent
                    for level_ranges, level_nets in zip(self.hier_levels, self.noise_pred_levels):

                        for (net, (start, end, jname)) in zip(level_nets, level_ranges):
                            j_idx = NAME_TO_IDX[jname]

                            # 当前关节 noisy rot
                            cur = noisy_action_seq[:, :, start:end]  # (B, T, rot_dims)

                            parent_idx = PARENT_IDX[j_idx]
                            if parent_idx == -1:
                                # 根关节：没有上一节
                                net_input = cur  # (B, T, 6)
                            else:
                                # parent 的 rot
                                parent_rot = joint_rot_slices[parent_idx]  # (B, T, 6)
                                # 输入 = 当前 + parent，维度 = 12
                                net_input = torch.cat([cur, parent_rot], dim=-1)  # (B, T, 2*rot_dims)

                            # 走对应的 U-Net
                            sub_pred = net(
                                net_input,
                                timesteps,
                                global_cond=obs_cond,
                            )  # 输出维度 = input_dim

                            # 有些实现可能 squeeze 掉了 batch 维，兼容一下
                            if sub_pred.dim() == 2:
                                sub_pred = sub_pred.unsqueeze(0)  # -> (B, T, C)

                            # 只把“当前关节”的那 rot_dims 写回：前 (end-start) 维
                            sub_pred_joint = sub_pred[:, :, : (end - start)]  # (B, T, rot_dims)
                            pred[:, :, start:end] = sub_pred_joint

            # sub_noisy_action_seqs = [noisy_action_seq[:, :, start: end] for start, end in self.sub_ranges]
            # pred = []
            # for idx, sub_noise_pred_net in enumerate(self.noise_pred_net):
            #     sub_pred = sub_noise_pred_net(sub_noisy_action_seqs[idx], timesteps, global_cond=obs_cond)
            #     pred.append(sub_pred)
            # pred = torch.cat(pred, dim=-1)
            # ====== 分层版本 ======
            # 初始化 pred（和 noisy_action_seq 形状相同，后面逐段写进去）
            #pred = torch.zeros_like(noisy_action_seq, device=device)

            # # 1) env 动作（如果有）
            # if self.env_unet is not None and self.env_act_dim > 0:
            #     sub_noisy_env = noisy_action_seq[:, :, :self.env_act_dim]
            #     sub_pred_env = self.env_unet(
            #         sub_noisy_env, timesteps, global_cond=obs_cond
            #     )
            #     pred[:, :, :self.env_act_dim] = sub_pred_env
            #
            # # 2) 各层的关节
            # for level_ranges, level_nets in zip(self.hier_levels, self.noise_pred_levels):
            #     for (net, (start, end, jname)) in zip(level_nets, level_ranges):
            #         sub_noisy = noisy_action_seq[:, :, start:end]
            #         sub_pred = net(sub_noisy, timesteps, global_cond=obs_cond)
            #         pred[:, :, start:end] = sub_pred
            # 1) env 动作（如果有），统一当作 "level 0" 的 cond
            # if self.env_unet is not None and self.env_act_dim > 0:
            #     low_feat_env = torch.zeros((B, self.low_feat_dim), device=device)
            #     env_cond = torch.cat([obs_cond, low_feat_env], dim=-1)
            #     sub_noisy_env = noisy_action_seq[:, :, :self.env_act_dim]
            #     sub_pred_env = self.env_unet(
            #         sub_noisy_env, timesteps, global_cond=env_cond
            #     )
            #     pred[:, :, :self.env_act_dim] = sub_pred_env
            #
            # # 2) finger 各层
            # for level_id, (level_ranges, level_nets) in enumerate(
            #         zip(self.hier_levels, self.noise_pred_levels)
            # ):
            #     if level_id == 0:
            #         # 第一层没有更低的关节，用 0 向量
            #         low_feat = torch.zeros((B, self.low_feat_dim), device=device)
            #     else:
            #         # 收集所有 <level_id 的关节 GT
            #         lower_slices = []
            #         for l in range(level_id):
            #             for j_idx in self.level_joint_indices[l]:
            #                 s = self.env_act_dim + j_idx * self.rot_dims
            #                 e = s + self.rot_dims
            #                 lower_slices.append(action_seq[:, :, s:e])  # (B,T,rot_dims)
            #         if len(lower_slices) > 0:
            #             lower_gt = torch.cat(lower_slices, dim=-1)  # (B,T, N_low*rot_dims)
            #             lower_feat_input = lower_gt.mean(dim=1)  # (B, N_low*rot_dims)
            #             low_feat = self.low_cond_mlps[level_id - 1](lower_feat_input)
            #         else:
            #             low_feat = torch.zeros((B, self.low_feat_dim), device=device)
            #
            #     level_cond = torch.cat([obs_cond, low_feat], dim=-1)  # (B, global_cond_dim)
            #
            #     # 这一层内所有关节共享 level_cond，并行预测
            #     for (net, (start, end, jname)) in zip(level_nets, level_ranges):
            #         sub_noisy = noisy_action_seq[:, :, start:end]
            #         sub_pred = net(sub_noisy, timesteps, global_cond=level_cond)
            #         pred[:, :, start:end] = sub_pred
            # 预先把每个关节的 rot 切出来
            # joint_rot_slices = {}
            # for jname in JOINT_NAMES:
            #     j_idx = NAME_TO_IDX[jname]
            #     s = self.env_act_dim + j_idx * self.rot_dims
            #     e = s + self.rot_dims
            #     joint_rot_slices[j_idx] = noisy_action_seq[:, :, s:e]  # (B, T, rot_dims)
            #
            # # env 部分如果有，照旧处理
            # if  self.env_act_dim > 0:
            #     sub_noisy_env = noisy_action_seq[:, :, :self.env_act_dim]
            #     sub_pred_env = self.env_unet(sub_noisy_env, timesteps, global_cond=obs_cond)
            #     pred[:, :, :self.env_act_dim] = sub_pred_env
            #
            # # 分层：每一层的关节
            # for level_id, (level_ranges, level_nets) in enumerate(
            #         zip(self.hier_levels, self.noise_pred_levels)
            # ):
            #     # 收集所有低层 joint 的 rot
            #     lower_joint_indices = []
            #     for l in range(level_id):
            #         lower_joint_indices.extend(self.level_joint_indices[l])
            #
            #     if len(lower_joint_indices) > 0:
            #         lower_parts = [joint_rot_slices[j_idx] for j_idx in lower_joint_indices]
            #         lower_feat = torch.cat(lower_parts, dim=-1)  # (B, T, lower_count*rot_dims)
            #     else:
            #         lower_feat = None
            #
            #     # 当前层的每一个关节
            #     for (net, (start, end, jname)) in zip(level_nets, level_ranges):
            #         cur = noisy_action_seq[:, :, start:end]  # (B, T, rot_dims)
            #
            #         if lower_feat is not None:
            #             net_input = torch.cat([cur, lower_feat], dim=-1)  # (B, T, rot_dims + ...)
            #         else:
            #             net_input = cur
            #
            #         sub_pred = net(net_input, timesteps, global_cond=obs_cond)
            #         pred[:, :, start:end] = sub_pred
                # ========= 这里开始：根据 prediction_type 决定监督目标 gt =========
        if self.diffusion_prediction_type == 'epsilon':
                    # 传统 DDPM：网络预测噪声 ε
            gt = noise  # (B, T, act_dim)

        elif self.diffusion_prediction_type == 'sample':
                    # x0-prediction：网络直接预测干净动作 x0
            gt = action_seq  # (B, T, act_dim)

        elif self.diffusion_prediction_type == 'v_prediction':
                    # v = α_t * ε - σ_t * x0
                    # 从 scheduler 里取 ᾱ_t（alphas_cumprod），注意要按当前 batch 的 timestep 索引
            alphas_cumprod = self.noise_scheduler.alphas_cumprod  # shape: (num_train_timesteps,)

                    # 按 batch 的随机 timestep 取出 ᾱ_t，timesteps: (B,)
            a_bar = alphas_cumprod[timesteps].to(action_seq.dtype)  # (B,)
            a_bar = a_bar.view(B, 1, 1)  # (B,1,1) 方便 broadcast

            alpha = a_bar.sqrt()  # √ᾱ_t，shape: (B,1,1)
            sigma = (1.0 - a_bar).sqrt()  # √(1-ᾱ_t)，shape: (B,1,1)

                    # noise, action_seq: (B, T, act_dim)
                    # broadcast 后得到 v: (B, T, act_dim)
            gt = alpha * noise - sigma * action_seq

        else:
            raise ValueError(f"Unknown diffusion_prediction_type: {self.diffusion_prediction_type}")

        return {
            'gt': gt,
            'pred': pred,
            'timesteps': timesteps,
        }
        # return {
        #     'gt': noise if self.diffusion_prediction_type == 'epsilon' else action_seq,
        #     'pred': pred,
        #     'timesteps': timesteps,   # ★ 新增：供几何项反解 x0 使用
        # }

    # def get_action(self, obs_seq):
    #     self.noise_ddim_scheduler.set_timesteps(num_inference_steps=10)
    #
    #     device = obs_seq.device
    #     B = obs_seq.shape[0]
    #     with torch.no_grad():
    #         obs_cond = obs_seq.flatten(start_dim=1)  # (B, obs_horizon * obs_dim)
    #
    #         # initialize action from Gaussian noise
    #         noisy_action_seq = torch.randn((B, self.pred_horizon, self.act_dim), device=device)
    #
    #         for t in self.noise_ddim_scheduler.timesteps:  # inverse order
    #             if self.action_type == 'joint_value':
    #                 model_output = self.noise_pred_net(
    #                     sample=noisy_action_seq,
    #                     timestep=t,
    #                     global_cond=obs_cond,
    #                 )
    #             else:
    #                 # sub_noisy_action_seqs = [noisy_action_seq[:, :, start: end] for start, end in self.sub_ranges]
    #                 # model_output = []
    #                 # for idx, sub_noise_pred_net in enumerate(self.noise_pred_net):
    #                 #     pred = sub_noise_pred_net(sample=sub_noisy_action_seqs[idx], timestep=t, global_cond=obs_cond)
    #                 #     model_output.append(pred)
    #                 # model_output = torch.cat(model_output, dim=-1)
    #                 # ====== 分层版本 ======
    #                 model_output = torch.zeros_like(noisy_action_seq, device=device)
    #
    #                 # env
    #                 if self.env_unet is not None and self.env_act_dim > 0:
    #                     sub_noisy_env = noisy_action_seq[:, :, :self.env_act_dim]
    #                     sub_pred_env = self.env_unet(
    #                         sample=sub_noisy_env,
    #                         timestep=t,
    #                         global_cond=obs_cond,
    #                     )
    #                     model_output[:, :, :self.env_act_dim] = sub_pred_env
    #
    #                 # 每一层的关节
    #                 for level_ranges, level_nets in zip(self.hier_levels, self.noise_pred_levels):
    #                     for (net, (start, end, jname)) in zip(level_nets, level_ranges):
    #                         sub_noisy = noisy_action_seq[:, :, start:end]
    #                         sub_pred = net(
    #                             sample=sub_noisy,
    #                             timestep=t,
    #                             global_cond=obs_cond,
    #                         )
    #                         model_output[:, :, start:end] = sub_pred
    #
    #             # inverse diffusion step (remove noise)
    #             noisy_action_seq = self.noise_ddim_scheduler.step(
    #                 model_output=model_output,
    #                 timestep=t,
    #                 sample=noisy_action_seq,
    #             ).prev_sample
    #
    #     # only take act_horizon number of actions
    #     start = self.obs_horizon - 1
    #     end = start + self.act_horizon
    #     return noisy_action_seq[:, start:end]  # (B, act_horizon, act_dim)
    def get_action(self, obs_seq):
        if self.hier_num_levels==1:
            self.noise_ddim_scheduler.set_timesteps(num_inference_steps=10)

            device = obs_seq.device
            B = obs_seq.shape[0]
            with torch.no_grad():
                obs_cond = obs_seq.flatten(start_dim=1)  # (B, obs_horizon * obs_dim)

                # initialize action from Gaussian noise
                noisy_action_seq = torch.randn((B, self.pred_horizon, self.act_dim), device=device)

                for t in self.noise_ddim_scheduler.timesteps:  # inverse order
                    if self.action_type == 'joint_value1':
                        model_output = self.noise_pred_net(
                            sample=noisy_action_seq,
                            timestep=t,
                            global_cond=obs_cond,
                        )
                    else:
                        sub_noisy_action_seqs = [noisy_action_seq[:, :, start: end] for start, end in self.sub_ranges]
                        model_output = []
                        for idx, sub_noise_pred_net in enumerate(self.noise_pred_net):
                            pred = sub_noise_pred_net(sample=sub_noisy_action_seqs[idx], timestep=t,
                                                      global_cond=obs_cond)
                            model_output.append(pred)
                        model_output = torch.cat(model_output, dim=-1)

                    # inverse diffusion step (remove noise)
                    noisy_action_seq = self.noise_ddim_scheduler.step(
                        model_output=model_output,
                        timestep=t,
                        sample=noisy_action_seq,
                    ).prev_sample
        else:
            self.noise_ddim_scheduler.set_timesteps(num_inference_steps=10)

            device = obs_seq.device
            B = obs_seq.shape[0]
            obs_cond = obs_seq.flatten(start_dim=1)  # (B, obs_horizon * obs_dim)

            with torch.no_grad():
                noisy_action_seq = torch.randn(
                    (B, self.pred_horizon, self.act_dim), device=device
                )

                for t in self.noise_ddim_scheduler.timesteps:
                    if self.action_type == 'joint_value1':
                        low_feat = torch.zeros((B, self.low_feat_dim), device=device)
                        if self.hier_num_levels == 1:
                            level_cond=obs_cond
                        else:
                            level_cond = torch.cat([obs_cond, low_feat], dim=-1)
                        model_output = self.noise_pred_net(
                            sample=noisy_action_seq,
                            timestep=t,
                            global_cond=level_cond,
                        )
                    else:
                        if self.hier_context_mode == "all_lower":
                            # ========= 分层版本：本关节 rot + 所有低层关节 rot 拼接 =========
                            model_output = torch.zeros_like(noisy_action_seq, device=device)

                            # 1) env 动作（如果有），输入维度 = env_act_dim
                            if  self.env_act_dim > 0:
                                sub_noisy_env = noisy_action_seq[:, :, :self.env_act_dim]
                                sub_pred_env = self.env_unet(
                                    sample=sub_noisy_env,
                                    timestep=t,
                                    global_cond=obs_cond,  # 统一只用 obs_cond
                                )
                                model_output[:, :, :self.env_act_dim] = sub_pred_env

                            # 2) 预先切出所有关节的 noisy rot（当前 DDIM 步的动作）
                            joint_rot_slices = {}
                            for jname in JOINT_NAMES:
                                j_idx = NAME_TO_IDX[jname]  # 0..23
                                s = self.env_act_dim + j_idx * self.rot_dims
                                e = s + self.rot_dims
                                joint_rot_slices[j_idx] = noisy_action_seq[:, :, s:e]  # (B, T, rot_dims)

                            # 3) 分层：每一层的关节输入 = 当前关节 rot + 所有低层关节 rot
                            for level_id, (level_ranges, level_nets) in enumerate(
                                    zip(self.hier_levels, self.noise_pred_levels)
                            ):
                                # 收集所有低层 joint index（比当前 level 更“靠近根”的层）
                                lower_joint_indices = []
                                for l in range(level_id):
                                    lower_joint_indices.extend(self.level_joint_indices[l])

                                if len(lower_joint_indices) > 0:
                                    lower_parts = [
                                        joint_rot_slices[j_idx] for j_idx in lower_joint_indices
                                    ]
                                    # lower_feat: (B, T, lower_joint_count * rot_dims)
                                    lower_feat = torch.cat(lower_parts, dim=-1)
                                else:
                                    lower_feat = None

                                # 当前层每一个关节
                                for (net, (start, end, jname)) in zip(level_nets, level_ranges):
                                    # 当前关节的 noisy rot
                                    cur = noisy_action_seq[:, :, start:end]  # (B, T, rot_dims)

                                    if lower_feat is not None:
                                        # 拼接后的输入： (B, T, rot_dims * (1 + lower_joint_count))
                                        net_input = torch.cat([cur, lower_feat], dim=-1)
                                    else:
                                        # 第一层，没有低层关节
                                        net_input = cur

                                    # 调对应 Unet：注意 global_cond 仍然传 obs_cond
                                    sub_pred = net(
                                        sample=net_input,
                                        timestep=t,
                                        global_cond=obs_cond,
                                    )
                                    # 只取当前关节对应的那部分通道（前 rot_dims = end-start）
                                    sub_pred_joint = sub_pred[:, :, : (end - start)]  # (B, T, 6)

                                    model_output[:, :, start:end] = sub_pred_joint
                        elif self.hier_context_mode == "parent_only":
                            
                            model_output = torch.zeros_like(noisy_action_seq, device=device)

                            # 1) env 先算
                            if  self.env_act_dim > 0:
                                sub_noisy_env = noisy_action_seq[:, :, :self.env_act_dim]
                                sub_pred_env = self.env_unet(
                                    sample=sub_noisy_env,
                                    timestep=t,
                                    global_cond=obs_cond,
                                )
                                model_output[:, :, :self.env_act_dim] = sub_pred_env

                            # 2) 预切所有关节的 noisy rot
                            joint_rot_slices = {}
                            for jname in JOINT_NAMES:
                                j_idx = NAME_TO_IDX[jname]
                                s = self.env_act_dim + j_idx * self.rot_dims
                                e = s + self.rot_dims
                                joint_rot_slices[j_idx] = noisy_action_seq[:, :, s:e]

                            # 3) 分层 + 只看 parent
                            for level_ranges, level_nets in zip(self.hier_levels, self.noise_pred_levels):

                                for (net, (start, end, jname)) in zip(level_nets, level_ranges):
                                    j_idx = NAME_TO_IDX[jname]
                                    cur = noisy_action_seq[:, :, start:end]  # (B, T, rot_dims)

                                    parent_idx = PARENT_IDX[j_idx]
                                    if parent_idx == -1:
                                        net_input = cur
                                    else:
                                        parent_rot = joint_rot_slices[parent_idx]
                                        net_input = torch.cat([cur, parent_rot], dim=-1)

                                    sub_pred = net(
                                        sample=net_input,
                                        timestep=t,
                                        global_cond=obs_cond,
                                    )
                                    if sub_pred.dim() == 2:
                                        sub_pred = sub_pred.unsqueeze(0)

                                    sub_pred_joint = sub_pred[:, :, : (end - start)]
                                    model_output[:, :, start:end] = sub_pred_joint

                    # model_output = torch.zeros_like(noisy_action_seq, device=device)
                    #
                    # # env 先算（不依赖 finger）
                    # if  self.env_act_dim > 0:
                    #     low_feat_env = torch.zeros((B, self.low_feat_dim), device=device)
                    #     env_cond = torch.cat([obs_cond, low_feat_env], dim=-1)
                    #     sub_noisy_env = noisy_action_seq[:, :, :self.env_act_dim]
                    #     sub_pred_env = self.env_unet(
                    #         sample=sub_noisy_env,
                    #         timestep=t,
                    #         global_cond=obs_cond,
                    #     )
                    #     model_output[:, :, :self.env_act_dim] = sub_pred_env
                    #
                    # # finger 各层：用当前 noisy_action_seq 中的“低层预测”构造 cond
                    # for level_id, (level_ranges, level_nets) in enumerate(
                    #         zip(self.hier_levels, self.noise_pred_levels)
                    # ):
                    #     if level_id == 0:
                    #         low_feat = torch.zeros((B, self.low_feat_dim), device=device)
                    #     else:
                    #         lower_slices = []
                    #         for l in range(level_id):
                    #             for j_idx in self.level_joint_indices[l]:
                    #                 s = self.env_act_dim + j_idx * self.rot_dims
                    #                 e = s + self.rot_dims
                    #                 # 注意：这里用的是 noisy_action_seq 当前状态，
                    #                 # 它包含了前几轮 step 写进来的内容
                    #                 lower_slices.append(noisy_action_seq[:, :, s:e])
                    #         if len(lower_slices) > 0:
                    #             lower_pred = torch.cat(lower_slices, dim=-1)
                    #             lower_feat_input = lower_pred.mean(dim=1)
                    #             #low_feat = self.low_cond_mlps[level_id - 1](lower_feat_input)
                    #         else:
                    #             low_feat = torch.zeros((B, self.low_feat_dim), device=device)
                    #
                    #     level_cond = torch.cat([obs_cond, low_feat], dim=-1)
                    #
                    #     for (net, (start, end, jname)) in zip(level_nets, level_ranges):
                    #         sub_noisy = noisy_action_seq[:, :, start:end]
                    #         sub_pred = net(
                    #             sample=sub_noisy,
                    #             timestep=t,
                    #             global_cond=obs_cond,
                    #         )
                    #         model_output[:, :, start:end] = sub_pred

                noisy_action_seq = self.noise_ddim_scheduler.step(
                    model_output=model_output,
                    timestep=t,
                    sample=noisy_action_seq,
                ).prev_sample
                if self.action_type == "rot_mat":
                    for jname in JOINT_NAMES:
                        j_idx = NAME_TO_IDX[jname]
                        s = self.env_act_dim + j_idx * 9
                        e = s + 9

                        R_flat = noisy_action_seq[:, :, s:e]
                        R = R_flat.view(B, self.pred_horizon, 3, 3)
                        R_proj = project_to_rotmat(R)
                        noisy_action_seq[:, :, s:e] = R_proj.view(B, self.pred_horizon, 9)

        start = self.obs_horizon - 1
        end = start + self.act_horizon
        return noisy_action_seq[:, start:end]


def create_model(cfg):
    model = DiffusionPolicy(cfg)
    return model
