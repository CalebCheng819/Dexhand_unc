import time
import viser
import numpy as np
import torch

from utils.hand_model import create_hand_model
from leap_node import LeapNode


def main():
    leaphand_sim = create_hand_model("leaphand", device="cpu")
    leaphand_real = LeapNode()
    leaphand_real.set_allegro(np.zeros(16))

    lower, upper = leaphand_sim.pk_chain.get_joint_limits()

    server = viser.ViserServer(host='127.0.0.1', port=8080)

    canonical_trimesh = leaphand_sim.get_trimesh_q(leaphand_sim.get_canonical_q())["visual"]
    server.scene.add_mesh_simple(
        "LEAP_Hand",
        canonical_trimesh.vertices,
        canonical_trimesh.faces,
        color=(102, 192, 255),
        opacity=0.8
    )

    def update(q):
        trimesh = leaphand_sim.get_trimesh_q(q)["visual"]
        server.scene.add_mesh_simple(
            "LEAP_Hand",
            trimesh.vertices,
            trimesh.faces,
            color=(102, 192, 255),
            opacity=0.8
        )
        leaphand_real.set_allegro(q.numpy())
        time.sleep(0.05)

    gui_joints = []
    for i, joint_name in enumerate(leaphand_sim.get_joint_orders()):
        slider = server.gui.add_slider(
            label=joint_name,
            min=round(lower[i], 2),
            max=round(upper[i], 2),
            step=(upper[i] - lower[i]) / 100,
            initial_value=0,
        )
        slider.on_update(lambda _: update(torch.tensor([gui.value for gui in gui_joints])))
        gui_joints.append(slider)

    while True:
        time.sleep(1)


if __name__ == "__main__":
    # main()
    leaphand_real = LeapNode()
    while True:
        leaphand_real.set_allegro(np.zeros(16))
        time.sleep(0.5)
        # leaphand_real.set_allegro(np.array([1, 0, 0.5, 0.5, 1, 0, 0.5, 0.5, 1, 0, 0.5, 0.5, 0.5, 1.5, 0.5, 0.5]))
        time.sleep(0.5)
