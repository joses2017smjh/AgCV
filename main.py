#!/usr/bin/env python3
from pybullet_tree_sim.pruning_environment import PruningEnv
from pybullet_tree_sim.robot import Robot
from pybullet_tree_sim.tree import Tree
from pybullet_tree_sim.utils.pyb_utils import PyBUtils
from pybullet_tree_sim.utils.camera_helpers import get_fov_from_dfov

import numpy as np
import secrets
import time
from zenlog import log

from scipy.spatial.transform import Rotation


def main():
    pbutils = PyBUtils(renders=True)
    robot_start_orientation = Rotation.from_euler("xyz", [0, 0, 180], degrees=True).as_quat()
    robot = Robot(pbclient=pbutils.pbclient, position=[0, 1, 0], orientation=robot_start_orientation)

    penv = PruningEnv(
        pbutils=pbutils,
        verbose=True,
    )

    # _1_inch = 0.0254
    # penv.activate_shape(
    #     shape="cylinder",
    #     radius=_1_inch * 2,
    #     height=2.85,
    #     orientation=[0, np.pi / 2, 0],
    # )
    # penv.activate_shape(shape="cylinder", radius=0.01, height=2.85, orientation=[0, np.pi / 2, 0])

    tree_name = penv.load_tree(
        pbutils=pbutils,
        scale=1.0,
        tree_id=2,
        tree_type="envy",
        tree_namespace="LPy",
        # tree_urdf_path=os.path.join(URDF_PATH, "trees", "envy", "generated", "LPy_envy_tree0.urdf"),
        save_tree_urdf=False,
        # randomize_pose=True
    )
    penv.activate_tree_by_id_str(tree_id_str=tree_name)

    # # Run the sim a little just to get the environment properly loaded.
    for i in range(100):
        pbutils.pbclient.stepSimulation()
        time.sleep(0.1)

    # log.debug(robot.sensors['tof0'].tf_frame)

    while True:
        try:
            # log.debug(f"{robot.sensors['tof0']}")
            tof0_view_matrix = robot.get_view_mat_at_curr_pose(camera=robot.sensors["tof0"])
            tof0_rgbd = robot.get_rgbd_at_cur_pose(
                camera=robot.sensors["tof0"],
                type="sensor",
                view_matrix=tof0_view_matrix,
            )
            tof1_view_matrix = robot.get_view_mat_at_curr_pose(camera=robot.sensors["tof1"])
            tof1_rgbd = robot.get_rgbd_at_cur_pose(
                camera=robot.sensors["tof1"],
                type="sensor",
                view_matrix=tof1_view_matrix,
            )
            # tof0_view_matrix = np.asarray(tof0_view_matrix).reshape((4, 4), order="F")
            # log.debug(f"{tof0_view_matrix[:3, 3]}")

            # Get user keyboard input, map to robot movement, camera capture, controller action
            keys_pressed = penv.get_key_pressed()
            move_action = robot.get_key_move_action(keys_pressed=keys_pressed)
            sensor_data = robot.get_key_sensor_action(keys_pressed=keys_pressed)

            joint_vels, jacobian = robot.calculate_joint_velocities_from_ee_velocity_dls(
                end_effector_velocity=move_action
            )
            singularity = robot.set_joint_velocities(joint_velocities=joint_vels)

            # Step simulation
            pbutils.pbclient.stepSimulation()
            time.sleep(0.001)
        except KeyboardInterrupt:
            break

    # # penv.deactivate_tree(tree_id_str="LPy_tree1")
    # penv.pbutils.pbclient.disconnect()
    return


if __name__ == "__main__":
    main()
