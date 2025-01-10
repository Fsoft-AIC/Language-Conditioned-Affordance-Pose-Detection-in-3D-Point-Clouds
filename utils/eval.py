import numpy as np
from scipy.spatial.distance import cdist
from scipy.spatial.transform import Rotation as R


def affordance_eval(affordance_list, result):
    """_summary_
    This fuction evaluates the affordance detection capability.
    `result` is loaded from  result.pkl file produced by detect.py.
    """
    num_correct = 0
    num_all = 0
    num_points = {aff: 0 for aff in affordance_list}
    num_label_points = {aff: 0 for aff in affordance_list}
    num_correct_fg_points = {aff: 0 for aff in affordance_list}
    num_correct_bg_points = {aff: 0 for aff in affordance_list}
    num_union_points = {aff: 0 for aff in affordance_list}
    num_appearances = {aff: 0 for aff in affordance_list}

    for shape in result:
        for affordance in shape['affordance']:
            label = np.transpose(shape['full_shape']['label'][affordance])
            prediction = shape['result'][affordance][0]
            
            num_correct += np.sum(label == prediction)
            num_all += 2048
            num_points[affordance] += 2048
            num_label_points[affordance] += np.sum(label == 1.)
            num_correct_fg_points[affordance] += np.sum((label == 1.) & (prediction == 1.))
            num_correct_bg_points[affordance] += np.sum((label == 0.) & (prediction == 0.))
            num_union_points[affordance] += np.sum((label == 1.) | (prediction == 1.))
    mIoU = np.average(np.array(list(num_correct_fg_points.values())) / np.array(list(num_union_points.values())),
                      weights=np.array(list(num_appearances.values())))
    Acc = num_correct / num_all
    mAcc = np.mean((np.array(list(num_correct_fg_points.values())) + np.array(list(num_correct_bg_points.values()))) / \
        np.array(list(num_points.values())))
    
    return mIoU, Acc, mAcc


def pose_eval(result):
    """_summary_
    This function evaluates the pose detection capability.
    `result` is loaded from  result.pkl file produced by detect.py.
    """
    all_min_dist = []
    all_rate = []
    for object in result:
        for affordance in object['affordance']:
            gt_poses = np.array([np.concatenate((R.from_matrix(p[:3, :3]).as_quat(), p[:3, 3]), axis=0) for p in object['pose'][affordance]])
            distances = cdist(gt_poses, object['result'][affordance][1])
            rate = np.sum(np.any(distances <= 0.2, axis=1)) / len(object['pose'][affordance])
            all_rate.append(rate)
       
            g = gt_poses[:, np.newaxis, :]
            g_pred = object['result'][affordance][1]
            l2_distances = np.sqrt(np.sum((g-g_pred)**2, axis=2))
            min_distance = np.min(l2_distances)

            # discard cases when set of gt poses and set of detected poses too far from each other, to get a stable result
            if min_distance <= 1.0:
                all_min_dist.append(min_distance)
    return (np.mean(np.array(all_min_dist)), np.mean(np.array(all_rate)))