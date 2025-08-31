import torch
import torch.nn as nn
import numpy as np
from scipy import interpolate
from scipy.interpolate import interp1d
import os
import sys
import random
import config


def upgrade_resolution(arr, scale):
    x = np.arange(0, arr.shape[0])
    f = interp1d(x, arr, kind='linear', axis=0, fill_value='extrapolate')
    scale_x = np.arange(0, arr.shape[0], 1 / scale)
    up_scale = f(scale_x)
    return up_scale


def calculate_saliency_score(proposal_center, point_annotations, proposal_length):
    """
    计算提议的显著性得分
    
    Args:
        proposal_center: 提议的中心点
        point_annotations: 标注点的位置列表
        proposal_length: 提议的长度
    
    Returns:
        ssal: 显著性得分，范围 [0,1]
    """
    if point_annotations is None or len(point_annotations) == 0:
        return 1.0  # 如果没有标注点，则默认为1.0
        
    # 计算提议中心到最近标注点的距离
    distances = []
    for point in point_annotations:
        if point >= 0:  # 有效的标注点
            distance = abs(proposal_center - point)
            distances.append(distance)
    
    if not distances:  # 没有有效的标注点
        return 1.0
        
    min_distance = min(distances)
    ssal = 1.0 - 2.0 * min_distance / max(proposal_length, 1e-5)
    return max(0.0, ssal)  # 确保得分非负


def get_oic_score(cas_sigmoid_fuse, start, end, delta=0.25, point_annotations=None):
    """
    计算内外对比得分（OIC），现在结合了显著性得分
    
    Args:
        cas_sigmoid_fuse: 分数序列
        start: 起始位置
        end: 结束位置
        delta: 外部区域比例
        point_annotations: 标注点位置列表
    
    Returns:
        rep_score: 代表性得分 (OIC * Ssal)
    """
    length = end - start + 1

    inner_score = torch.mean(cas_sigmoid_fuse[start:end+1])
    
    outer_s = max(0, int(start - delta * length))
    outer_e = min(int(cas_sigmoid_fuse.shape[0] - 1), int(end + delta * length))

    outer_seg = list(range(outer_s, start)) + list(range(end + 1, outer_e + 1))

    if len(outer_seg) == 0:
        outer_score = 0
    else:
        outer_score = torch.mean(cas_sigmoid_fuse[outer_seg])
    
    oic_score = inner_score - outer_score
    
    # 如果提供了标注点，计算显著性得分
    if point_annotations is not None and len(point_annotations) > 0:
        proposal_center = (start + end) / 2
        ssal = calculate_saliency_score(proposal_center, point_annotations, length)
        # 结合两个分数
        rep_score = oic_score * ssal
        return rep_score
    else:
        return oic_score


def get_proposal_oic(tempseg_list, int_temp_scores, c_pred, c_pred_scores, t_factor, lamb=0.25, gamma=0.20,
                     dynamic_segment_weights_cumsum=None, vid_duration=None, point_annotations=None):
    """
    生成基于内外对比和显著性校准的动作提议
    
    Args:
        tempseg_list: 临时段列表
        int_temp_scores: 临时分数
        c_pred: 类别预测
        c_pred_scores: 类别预测分数
        t_factor: 时间因子
        lamb: lambda参数
        gamma: gamma参数
        dynamic_segment_weights_cumsum: 动态段权重累积和
        vid_duration: 视频时长
        point_annotations: 标注点信息，格式为{类别索引: [点位置1, 点位置2, ...]}
    """
    temp = []
    if not dynamic_segment_weights_cumsum is None:
        f_upsample = interpolate.interp1d(dynamic_segment_weights_cumsum / dynamic_segment_weights_cumsum[-1],
                                          np.linspace(0, 1, dynamic_segment_weights_cumsum.shape[0]),
                                          kind='linear', axis=0, fill_value='extrapolate')

    for i in range(len(tempseg_list)):
        c_temp = []
        temp_list = np.array(tempseg_list[i])[0]
        if temp_list.any():
            grouped_temp_list = grouping(temp_list)
            for j in range(len(grouped_temp_list)):
                if len(grouped_temp_list[j]) < 2:
                    continue

                inner_score = np.mean(int_temp_scores[grouped_temp_list[j], i])

                len_proposal = len(grouped_temp_list[j])
                outer_s = max(0, int(grouped_temp_list[j][0] - lamb * len_proposal))
                outer_e = min(int(int_temp_scores.shape[0] - 1), int(grouped_temp_list[j][-1] + lamb * len_proposal))

                outer_temp_list = list(range(outer_s, int(grouped_temp_list[j][0]))) + list(
                    range(int(grouped_temp_list[j][-1] + 1), outer_e + 1))

                if len(outer_temp_list) == 0:
                    outer_score = 0
                else:
                    outer_score = np.mean(int_temp_scores[outer_temp_list, i])

                # 计算OIC分数
                oic_score = inner_score - outer_score + gamma * c_pred_scores[c_pred[i]]
                
                # 计算显著性分数（如果有标注点）
                ssal_score = 1.0  # 默认值
                if point_annotations is not None and c_pred[i] in point_annotations:
                    proposal_center = (grouped_temp_list[j][0] + grouped_temp_list[j][-1]) / 2
                    proposal_length = grouped_temp_list[j][-1] - grouped_temp_list[j][0] + 1
                    class_points = point_annotations[c_pred[i]]
                    if len(class_points) > 0:
                        ssal_score = calculate_saliency_score(proposal_center, class_points, proposal_length)
                
                # 结合OIC和显著性得分
                c_score = oic_score * ssal_score
                
                t_start = (grouped_temp_list[j][0] + 0) * t_factor
                t_end = (grouped_temp_list[j][-1] + 1) * t_factor
                if not dynamic_segment_weights_cumsum is None:
                    t_start = f_upsample(t_start / vid_duration) * vid_duration
                    t_end = f_upsample(t_end / vid_duration) * vid_duration
                c_temp.append([c_pred[i], c_score, t_start, t_end])
            temp.append(c_temp)
    return temp


def result2json(result):
    result_file = []    
    for i in range(len(result)):
        line = {'label': config.class_dict[result[i][0]], 'score': result[i][1],
                'segment': [result[i][2], result[i][3]]}
        result_file.append(line)
    return result_file


def grouping(arr):
    return np.split(arr, np.where(np.diff(arr) != 1)[0] + 1)


def save_best_record_thumos(test_info, file_path):
    fo = open(file_path, "w")
    fo.write("Iter: {}\n".format(test_info["iter"][-1]))
    fo.write("Step: {}\n".format(test_info["step"][-1]+1))
    fo.write("Test_acc: {:.4f}\n".format(test_info["test_acc"][-1]))
    fo.write("average_mAP[0.1:0.7]: {:.4f}\n".format(test_info["average_mAP[0.1:0.7]"][-1]))
    fo.write("average_mAP[0.1:0.5]: {:.4f}\n".format(test_info["average_mAP[0.1:0.5]"][-1]))
    fo.write("average_mAP[0.3:0.7]: {:.4f}\n".format(test_info["average_mAP[0.3:0.7]"][-1]))
    
    tIoU_thresh = np.linspace(0.1, 0.7, 7)
    for i in range(len(tIoU_thresh)):
        fo.write("mAP@{:.1f}: {:.4f}\n".format(tIoU_thresh[i], test_info["mAP@{:.1f}".format(tIoU_thresh[i])][-1]))

    fo.close()


def minmax_norm(act_map, min_val=None, max_val=None):
    if min_val is None or max_val is None:
        relu = nn.ReLU()
        max_val = relu(torch.max(act_map, dim=1)[0])
        min_val = relu(torch.min(act_map, dim=1)[0])

    delta = max_val - min_val
    delta[delta <= 0] = 1
    ret = (act_map - min_val) / delta.detach()

    ret[ret > 1] = 1
    ret[ret < 0] = 0

    return ret


def nms(proposals, thresh):
    proposals = np.array(proposals)
    x1 = proposals[:, 2]
    x2 = proposals[:, 3]
    scores = proposals[:, 1]

    areas = x2 - x1 + 1
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]

        keep.append(proposals[i].tolist())
        xx1 = np.maximum(x1[i], x1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])

        inter = np.maximum(0.0, xx2 - xx1 + 1)

        iou = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(iou < thresh)[0]
        order = order[inds + 1]
        
    return keep


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.benchmark=False


def save_config(config, file_path):
    fo = open(file_path, "w")
    fo.write("Configurtaions:\n")
    fo.write(str(config))
    fo.close()


def feature_sampling(features, start, end, num_divide):
    divide = (end - start + 1) // 3
    num_divide = max(num_divide, divide)
    step = (end - start) / num_divide

    feature_lst = torch.zeros((num_divide, features.shape[1])).to(torch.device('cuda:1'))
    for i in range(num_divide):
        start_point = int(start + step * i)
        end_point = int(start + step * (i+1))
        
        if start_point >= end_point:
            end_point += 1

        sample_id = np.random.randint(start_point, end_point)

        feature_lst[i] = features[sample_id]

    return feature_lst.mean(dim=0)


def select_seed(cas_sigmoid_fuse, point_anno):
    point_anno_agnostic = point_anno.max(dim=2)[0]
    bkg_seed = torch.zeros_like(point_anno_agnostic)
    act_seed = point_anno.clone().detach()

    # act_thresh = 0.1
    # bkg_thresh = 0.95
    act_thresh = 0.25
    bkg_thresh = 0.85

    bkg_score = cas_sigmoid_fuse[:,:,-1]

    for b in range(point_anno.shape[0]):
        act_idx = torch.nonzero(point_anno_agnostic[b]).squeeze(1)

        """ most left """
        if act_idx[0] > 0:
            bkg_score_tmp = bkg_score[b,:act_idx[0]]
            idx_tmp = bkg_seed[b,:act_idx[0]]
            idx_tmp[bkg_score_tmp >= bkg_thresh] = 1

            if idx_tmp.sum() >= 1:
                start_index = idx_tmp.nonzero().squeeze(1)[-1]
                idx_tmp[:start_index] = 1
            else:
                max_index = bkg_score_tmp.argmax(dim=0)
                idx_tmp[:max_index+1] = 1

            """ pseudo action point selection """
            for j in range(act_idx[0] - 1, -1, -1):
                if bkg_score[b][j] <= act_thresh and bkg_seed[b][j] < 1:
                    act_seed[b, j] = act_seed[b, act_idx[0]]
                else:
                    break

        """ most right """
        if act_idx[-1] < (point_anno.shape[1] - 1):
            bkg_score_tmp = bkg_score[b,act_idx[-1]+1:]
            idx_tmp = bkg_seed[b,act_idx[-1]+1:]
            idx_tmp[bkg_score_tmp >= bkg_thresh] = 1

            if idx_tmp.sum() >= 1:
                start_index = idx_tmp.nonzero().squeeze(1)[0]
                idx_tmp[start_index:] = 1
            else:
                max_index = bkg_score_tmp.argmax(dim=0)
                idx_tmp[max_index:] = 1

            """ pseudo action point selection """
            for j in range(act_idx[-1] + 1, point_anno.shape[1]):
                if bkg_score[b][j] <= act_thresh and bkg_seed[b][j] < 1:
                    act_seed[b, j] = act_seed[b, act_idx[-1]]
                else:
                    break
            
        """ between two instances """
        for i in range(len(act_idx) - 1):
            if act_idx[i+1] - act_idx[i] <= 1:
                continue

            bkg_score_tmp = bkg_score[b,act_idx[i]+1:act_idx[i+1]]
            idx_tmp = bkg_seed[b,act_idx[i]+1:act_idx[i+1]]
            idx_tmp[bkg_score_tmp >= bkg_thresh] = 1

            if idx_tmp.sum() >= 2:
                start_index = idx_tmp.nonzero().squeeze(1)[0]
                end_index = idx_tmp.nonzero().squeeze(1)[-1]
                idx_tmp[start_index+1:end_index] = 1                                   
            else:
                max_index = bkg_score_tmp.argmax(dim=0)
                idx_tmp[max_index] = 1

            """ pseudo action point selection """
            for j in range(act_idx[i] + 1, act_idx[i+1]):
                if bkg_score[b][j] <= act_thresh and bkg_seed[b][j] < 1:
                    act_seed[b, j] = act_seed[b, act_idx[i]]
                else:
                    break
            for j in range(act_idx[i+1] - 1, act_idx[i], -1):
                if bkg_score[b][j] <= act_thresh and bkg_seed[b][j] < 1:
                    act_seed[b, j] = act_seed[b, act_idx[i+1]]
                else:
                    break

    return act_seed, bkg_seed