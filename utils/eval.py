import numpy as np
from scipy.spatial.distance import cdist


def affordance_eval(affordance_list, result):
    """
    This fuction evaluates the affordance detection capability.
    `result` is loaded from  result.pkl file produced by detect.py
    """
    num_correct = 0
    num_all = 0
    num_points = {aff: 0 for aff in affordance_list}
    num_label_points = {aff: 0 for aff in affordance_list}
    num_correct_fg_points = {aff: 0 for aff in affordance_list}
    num_correct_bg_points = {aff: 0 for aff in affordance_list}
    num_union_points = {aff: 0 for aff in affordance_list}

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
    mIoU = np.mean(np.array(list(num_correct_fg_points.values())) / np.array(list(num_union_points.values())))
    Acc = num_correct / num_all
    mAcc = np.mean((np.array(list(num_correct_fg_points.values())) + np.array(list(num_correct_bg_points.values()))) / \
        np.array(list(num_points.values())))
    
    return mIoU, Acc, mAcc


def pose_eval(gt_poses, pred_poses):
    """
    This function evaluates the pose detection capability,
    returning two metrics mentioned in the paper.
    """
    all_min_dist = []
    all_rate = []
    for id in range(len(gt_poses)):
        for affordance in gt_poses[id].keys():
            distances = cdist(gt_poses[id][affordance], pred_poses[id][affordance])
            rate = np.sum(np.any(distances <= 0.2, axis=1)) / len(gt_poses[id][affordance])
            all_rate.append(rate)
            
            g = gt_poses[id][affordance][:, np.newaxis, :]
            g_pred = pred_poses[id][affordance]
            l1_distances = np.sum(np.abs(g - g_pred), axis=2)
            min_distance = np.min(l1_distances)
            all_min_dist.append(min_distance)
    return np.mean(np.array(all_min_dist)), np.mean(np.array(all_rate))