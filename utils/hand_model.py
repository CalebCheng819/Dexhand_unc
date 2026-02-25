import os
import sys
import json
import numpy as np
import torch
import trimesh
import xml.etree.ElementTree as ET
import pytorch_kinematics as pk
from termcolor import colored

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)

from utils.mesh_utils import load_link_meshes


class HandModel:
    def __init__(self, robot_name, meta_data, device):
        self.robot_name = robot_name
        self.device = device

        self.urdf_path = os.path.join(ROOT_DIR, meta_data['urdf_path'][robot_name])
        self.pk_chain = pk.build_chain_from_urdf(open(self.urdf_path).read()).to(dtype=torch.float32, device=device)
        self.dof = len(self.pk_chain.get_joint_parameter_names())
        self.meshes = load_link_meshes(self.urdf_path, self.pk_chain.get_link_names(), use_collision=False)
        self.removed_links = meta_data['removed_links'][robot_name]  # tmp useless TODO
        self.joint_orders = [joint.name for joint in self.pk_chain.get_joints()]
        self.joint_axes = self.get_joint_axes()
        self.link2joint_map = self.get_link2joint_map()  # child link name -> joint name
        self.joint_layers = self.get_joint_layers()  # list of list of joint names in each layer
        self.frame_status = None
        #  新增：指尖 link 名单
        self.tip_links = meta_data.get("tip_links", {}).get(robot_name, [])

    def update_status(self, q):
        self.frame_status = self.pk_chain.forward_kinematics(q.to(self.device))

    def get_joint_limits(self):
        lower, upper = self.pk_chain.get_joint_limits()
        return (torch.tensor(lower, dtype=torch.float32, device=self.device),
                torch.tensor(upper, dtype=torch.float32, device=self.device))

    def get_joint_layers(self):
        self.joints_with_layer = []
        self.max_layer = 0
        self.print_kinematic_tree(use_print=False)

        joint_layers = []
        for layer in range(self.max_layer + 1):
            if layer_list := [j for j, l in self.joints_with_layer if l == layer]:
                joint_layers.append(layer_list)
        return joint_layers

    def get_joint_axes(self):
        joint_axes = {}
        for joint in ET.parse(self.urdf_path).getroot().findall('joint'):
            if joint.get('type') == 'revolute':
                axis = joint.find('axis')
                assert axis is not None, f"Joint {joint.get('name')} has no axis defined."
                axis_xyz = [float(x) for x in axis.get('xyz').split()]
                joint_axes[joint.get('name')] = torch.tensor(axis_xyz, dtype=torch.float32, device=self.device)
        return joint_axes

    def get_link2joint_map(self):
        link2joint_map = {}  # child link name -> joint name
        for joint in ET.parse(self.urdf_path).getroot().findall('joint'):
            if joint.get('type') == 'revolute':
                link2joint_map[joint.find('child').get('link')] = joint.get('name')
        return link2joint_map

    def print_kinematic_tree(self, use_print=True, frame=None, layer=0):
        if frame is None:
            frame = self.pk_chain._root
        indent = '----' * layer
        joint_type = frame.joint.joint_type
        joint_name = frame.joint.name
        if use_print:
            if joint_type == 'revolute':
                joint_name = colored(joint_name, "red")
            print(f"{indent}{joint_name} ({joint_type})")
        else:  # only for get_joint_layers()
            if joint_type == 'revolute':
                self.joints_with_layer.append((joint_name, layer))
                self.max_layer = max(self.max_layer, layer)

        for child in frame.children:
            self.print_kinematic_tree(use_print, child, layer + 1)

# 用于计算手部模型中每个链接的变换矩阵。它根据给定的关节角度 
# q 更新手部模型的状态，并返回每个链接的变换矩阵
    def get_joint_transform(self, q):
    
        self.update_status(q)
        transform = {}
        for link_name in self.link2joint_map:
            transform[self.link2joint_map[link_name]] = self.frame_status[link_name].get_matrix()
        return transform

    def get_canonical_q(self):
        """ For visualization purposes only. """
        lower, upper = self.pk_chain.get_joint_limits()
        # canonical_q = torch.tensor(lower) * 0.5 + torch.tensor(upper) * 0.5
        # canonical_q[:6] = 0
        return torch.zeros_like(torch.tensor(lower), dtype=torch.float32, device=self.device)

    def get_trimesh_q(self, q):
        """ Return the hand trimesh object corresponding to the input joint value q. """
        self.update_status(q)

        scene = trimesh.Scene()
        for link_name in self.meshes:
            mesh_transform_matrix = self.frame_status[link_name].get_matrix()[0].cpu().numpy()
            scene.add_geometry(self.meshes[link_name].copy().apply_transform(mesh_transform_matrix))

        vertices = []
        faces = []
        vertex_offset = 0
        for geom in scene.geometry.values():
            if isinstance(geom, trimesh.Trimesh):
                vertices.append(geom.vertices)
                faces.append(geom.faces + vertex_offset)
                vertex_offset += len(geom.vertices)
        all_vertices = np.vstack(vertices)
        all_faces = np.vstack(faces)

        parts = {}
        for link_name in self.meshes:
            mesh_transform_matrix = self.frame_status[link_name].get_matrix()[0].cpu().numpy()
            part_mesh = self.meshes[link_name].copy().apply_transform(mesh_transform_matrix)
            parts[link_name] = part_mesh

        return_dict = {
            'visual': trimesh.Trimesh(vertices=all_vertices, faces=all_faces),
            'parts': parts
        }
        return return_dict

    def get_trimesh_se3(self, transform, index):
        """ Return the hand trimesh object corresponding to the input transform. """
        scene = trimesh.Scene()
        for link_name in transform:
            mesh_transform_matrix = transform[link_name][index].cpu().numpy()
            scene.add_geometry(self.meshes[link_name].copy().apply_transform(mesh_transform_matrix))

        vertices = []
        faces = []
        vertex_offset = 0
        for geom in scene.geometry.values():
            if isinstance(geom, trimesh.Trimesh):
                vertices.append(geom.vertices)
                faces.append(geom.faces + vertex_offset)
                vertex_offset += len(geom.vertices)
        all_vertices = np.vstack(vertices)
        all_faces = np.vstack(faces)

        return trimesh.Trimesh(vertices=all_vertices, faces=all_faces)

    def q_control2real(self, q_control):
        q_lower, q_upper = self.get_joint_limits()
        q_real = (q_control + 1) / 2 * (q_upper - q_lower) + q_lower
        return q_real

    def q_real2control(self, q_real):
        q_lower, q_upper = self.get_joint_limits()
        q_control = 2 * (q_real - q_lower) / (q_upper - q_lower) - 1
        return q_control
#新增用于计算末端位姿的函数
    def compute_tip_positions(self, q: torch.Tensor, tip_links=None) -> torch.Tensor:
        """
        q: (N, J) 真实关节角（弧度）
        返回: (N, K, 3)，K = len(tip_links)
        """
        if tip_links is None:
            tip_links = self.tip_links

        # 更新一次 FK，frame_status[link] 会是一个批量 frame
        self.update_status(q)  # pk_chain.forward_kinematics(q)

        positions = []
        for link_name in tip_links:
            T_link = self.frame_status[link_name].get_matrix()  # (N,4,4)
            pos = T_link[:, :3, 3]  # (N,3)
            positions.append(pos)

        tips = torch.stack(positions, dim=1)  # (N,K,3)
        return tips


def create_hand_model(robot_name, device):
    meta_data = json.load(open(os.path.join(ROOT_DIR, 'robot_urdf/meta_data.json')))
    hand_model = HandModel(robot_name, meta_data, device)
    return hand_model

