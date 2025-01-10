import trimesh
import numpy as np
import pickle
from scipy.spatial.transform import Rotation as R
import argparse
from utils.visualization import create_gripper_marker

color_code_1 = np.array([0, 0, 255])    # color code for affordance region
color_code_2 = np.array([0, 255, 0])    # color code for gripper pose
num_pose = 100 # number of poses to visualize per each object-affordance pair


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize")
    parser.add_argument("--result", help="result file")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    result_file = args.result_file
    with open(result_file, 'rb') as f:
        result = pickle.load(f)

    for i in range(len(result)):
        if result[i]['semantic class'] == 'Bottle':
            shape_index = i
            shape = result[shape_index]

            for affordance in shape['affordance']:
                colors = np.transpose(shape['result'][affordance][0]) * color_code_1
                point_cloud = trimesh.points.PointCloud(shape['full_shape']['coordinate'], colors=colors)
                print(f"Affordance: {affordance}")
                T = shape['result'][affordance][1][:num_pose]
                rotation = np.concatenate((R.from_quat(T[:, :4]).as_matrix(), np.zeros((num_pose, 1, 3), dtype=np.float32)), axis=1)
                translation = np.expand_dims(np.concatenate((T[:, 4:], np.ones((num_pose, 1), dtype=np.float32)), axis=1), axis=2)
                T = np.concatenate((rotation, translation), axis=2)
                poses = [create_gripper_marker(color=color_code_2).apply_transform(t) for t in T
                         if np.min(np.linalg.norm(point_cloud - (t @ np.array([0., 0., 6.59999996e-02, 1.]))[:3], axis=1)) <= 0.03] # this line is used to get reliable poses only
                scene = trimesh.Scene([point_cloud, poses])
                scene.show(line_settings={'point size': 10})