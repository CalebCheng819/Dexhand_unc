import os
import sys
import time
import viser
import torch
from scipy.spatial.transform import Rotation as R

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)

from utils.hand_model import create_hand_model
from utils.action_utils import convert_q, projection_q


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    hand = create_hand_model('shadowhand', device)
    lower, upper = hand.pk_chain.get_joint_limits()

    canonical_q = torch.tensor(lower) * 0.5 + torch.tensor(upper) * 0.5
    canonical_q = hand.get_canonical_q()
    hand.update_status(canonical_q)

    server = viser.ViserServer(host='127.0.0.1', port=8080)

    def visualize(q, prefix='joint'):
        hand.update_status(q)
        if prefix == 'joint':
            robot_trimesh = hand.get_trimesh_q(q)['visual']
            server.scene.add_mesh_simple(
                name='ShadowHand',
                vertices=robot_trimesh.vertices,
                faces=robot_trimesh.faces,
                color=(102, 192, 255),
                opacity=0.8
            )

        for link, transform in hand.frame_status.items():
            if link in hand.link2joint_map:
                joint_name = hand.link2joint_map[link]
                # print(f"[Link: {link}, Joint: {joint_name}] {transform.get_matrix()[0]}")
                matrix = transform.get_matrix()[0].cpu().numpy()
                if prefix == 'joint':
                    server.scene.add_frame(
                        name=f"{prefix}/{joint_name}",
                        axes_length=0.03,
                        axes_radius=0.002,
                        wxyz=R.from_matrix(matrix=matrix[:3, :3]).as_quat(scalar_first=True),
                        position=matrix[:3, 3]
                    )

        print("q:", q)
        new_convert_q = convert_q(hand, q, 'rot_quat')
        print("new_convert_q:", new_convert_q[0, :4])
        new_q = projection_q(hand, new_convert_q, 'rot_quat')
        print("new_q", new_q)
        print("Diff:", (new_q - q).abs().max())
        server.scene.add_frame(
            name=f"debug/WRJ2",
            axes_length=0.03,
            axes_radius=0.002,
            wxyz=new_q[0, :4],
            position=hand.frame_status['wrist'].get_matrix()[0][:3, 3].cpu().numpy()
        )

    visualize(canonical_q)
    gui_joints = []
    for i, joint_name in enumerate(hand.joint_orders):
        slider = server.gui.add_slider(
            label=joint_name,
            min=round(lower[i], 2),
            max=round(upper[i], 2),
            step=(upper[i] - lower[i]) / 100,
            initial_value=0,
        )
        slider.on_update(lambda _: visualize(torch.tensor([gui.value for gui in gui_joints])))
        gui_joints.append(slider)

    while True:
        time.sleep(1)


if __name__ == "__main__":
    main()
