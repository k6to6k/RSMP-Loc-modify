import torch.utils.data as data
import os
import csv
import json
import re
import numpy as np
import pandas as pd
import torch
import pdb
import time
import random
import utils
import config
from collections import defaultdict
# from torch._six import string_classes
import collections.abc as container_abcs
from scipy import interpolate

int_classes = int
string_classes = str
np_str_obj_array_pattern = re.compile(r'[SaUO]')


class ThumosFeature(data.Dataset):
    def __init__(self, cfg, data_path, mode, modal, feature_fps, num_segments, sampling, step, seed=-1, supervision='point'):
        if seed >= 0:
            utils.set_seed(seed)

        self.mode = mode
        self.modal = modal
        self.feature_fps = feature_fps
        self.num_segments = num_segments
        self.sampling = sampling
        self.supervision = supervision
        self.pred_segment_path = cfg.pred_segment_path
        self.dynamic_segment_weight_path = os.path.join(cfg.output_path, 'dynamic_segment_weights_pred_step{}'.format(step))
        self.max_segments_num = 200
        self.delta = cfg.delta
        self.pseudo_segment_dict = {}
        self.pseudo_segment_dict['results'] = defaultdict(list)
        if not self.pred_segment_path is None:
            with open(self.pred_segment_path, 'r') as pred_f:
                self.pseudo_segment_dict = json.load(pred_f)

        if 'train' in self.mode:
            self.feature_path = os.path.join(data_path, 'features', self.mode)
            split_path = os.path.join(data_path, 'split_{}.txt'.format(self.mode))
            split_file = open(split_path, 'r')
            self.vid_list = []
            for line in split_file:
                self.vid_list.append(line.strip())
            split_file.close()
        elif 'test' in self.mode:
            self.feature_path = os.path.join(data_path, 'features', self.mode)
            split_path = os.path.join(data_path, 'split_{}.txt'.format(self.mode))
            split_file = open(split_path, 'r')
            self.vid_list = []
            for line in split_file:
                self.vid_list.append(line.strip())
            split_file.close()
        elif 'full' in self.mode:
            self.feature_path = []
            for _mode in ['train', 'test']:
                self.feature_path.append(os.path.join(data_path, 'features', _mode))
            split_file = list(open(os.path.join(data_path, "split_test.txt"))) + list(open(os.path.join(data_path, "split_train.txt")))
            self.vid_list = [item.strip() for item in split_file]

        self.fps_dict = json.load(open(os.path.join(data_path, 'fps_dict.json')))

        # anno_path = os.path.join(data_path, 'gt.json')
        anno_path = cfg.gt_path
        anno_file = open(anno_path, 'r')
        self.anno = json.load(anno_file)
        cfg.gt_dict = self.anno["database"]
        anno_file.close()
        
        self.class_name_to_idx = dict((v, k) for k, v in config.class_dict.items())        
        self.num_classes = len(self.class_name_to_idx.keys())
        
        if self.supervision == 'point':
            self.point_anno = pd.read_csv(os.path.join(data_path, 'point_gaussian', 'point_labels.csv'))
            
        self.stored_info_all = {'new_dense_anno': [-1] * len(self.vid_list), 'sequence_score': [-1] * len(self.vid_list)}

        self.get_proposals(self.pseudo_segment_dict)

    def get_proposals(self, pseudo_segment_dict):
        self.label_dict = {}
        self.gt_action_dict = defaultdict(list)
        self.pseudo_segment_dict = pseudo_segment_dict
        for vid_name in self.vid_list:
            item_label = np.zeros(self.num_classes)
            for ann in self.anno["database"][vid_name]["annotations"]:
                ann_label = ann["label"]
                item_label[self.class_name_to_idx[ann_label]] = 1.0
                self.gt_action_dict[vid_name].append([ann['segment'][0], ann['segment'][1], 1.0, ann_label])
            self.label_dict[vid_name] = item_label

        self.pseudo_segment_dict_att = defaultdict(list)
        self.pseudo_segment_dict_pseudo = defaultdict(list)
        self.pseudo_segment_dict_all = defaultdict(list)
        for vid_name in self.vid_list:
            label_set = set()
            if 'validation' in vid_name:
                for ann in self.anno["database"][vid_name]['annotations']:
                    label_set.add(ann['label'])
            elif 'test' in vid_name:
                for pred in self.pseudo_segment_dict['results'][vid_name]:
                    label_set.add(pred['label'])

            prediction_list_all = []
            for label in label_set:
                prediction_list = []
                for pred in self.pseudo_segment_dict['results'][vid_name]:
                    if pred['label'] == label:
                        t_start = pred["segment"][0]
                        t_end = pred["segment"][1]
                        prediction_list.append([t_start, t_end, pred["score"], pred["label"]])
                prediction_list = sorted(prediction_list, key=lambda k: k[2], reverse=True)
                prediction_list_all += prediction_list
            self.pseudo_segment_dict_all[vid_name] = prediction_list_all

            # remove duplicate proposals
            prediction_list_nodup = []
            for pred in prediction_list_all:
                t_start = pred[0]
                t_end = pred[1]
                if [t_start, t_end] not in prediction_list_nodup:
                    prediction_list_nodup.append([t_start, t_end])
            prediction_list_nodup = sorted(prediction_list_nodup, key=lambda k: k[0], reverse=True)

            # remove the proposals inside another proposal 
            prediction_list_att = []
            if len(prediction_list_nodup) > 0:
                prediction_list_att.append(prediction_list_nodup.pop(-1))
                while len(prediction_list_nodup) > 0:
                    prev_segment = prediction_list_att.pop(-1)
                    cur_segment = prediction_list_nodup.pop(-1)
                    if prev_segment[1] >= cur_segment[1]:
                        prediction_list_att.append(prev_segment)
                    else:
                        prediction_list_att.append(prev_segment)
                        prediction_list_att.append(cur_segment)
            self.pseudo_segment_dict_att[vid_name] = prediction_list_att
    

    def __len__(self):
        return len(self.vid_list)

    def __getitem__(self, index):
        vid_name = self.vid_list[index]
        data, vid_num_seg, sample_idx, sample_time, dynamic_flag, dynamic_segment_weights_cumsum, vid_len_yuan = self.get_data(index)
        label, point_anno = self.get_label(index, vid_num_seg, sample_idx, sample_time, dynamic_flag)

        # print(self.stored_info_all['new_dense_anno'][index])
        stored_info = {'new_dense_anno': self.stored_info_all['new_dense_anno'][index],
                       'sequence_score': self.stored_info_all['sequence_score'][index]}

        proposal_bbox = torch.zeros((self.max_segments_num, 2), dtype=torch.int32)
        pseudo_instance_label = torch.zeros((vid_num_seg, self.num_classes + 1), dtype=torch.float32)
        # init all the timestep with bg class = 1
        pseudo_instance_label[:, -1] = 1

        time_to_index_factor = 25 / 16
        upsample_scale = time_to_index_factor * vid_num_seg / vid_len_yuan
        if dynamic_segment_weights_cumsum is not None and (vid_len_yuan + 1) == dynamic_segment_weights_cumsum.shape[0]:
            f_upsample = interpolate.interp1d(np.arange(vid_len_yuan + 1), dynamic_segment_weights_cumsum, kind='linear',
                                              axis=0, fill_value='extrapolate')
            upsample_scale = time_to_index_factor * vid_num_seg / round(dynamic_segment_weights_cumsum[-1])
        else:
            dynamic_segment_weights_cumsum = None

        ########## generate proposal_bbox from pseudo_segment_att for Intra & Inter-Segment Attention modules ##########
        proposal_list_att = []
        for k, segment in enumerate(self.pseudo_segment_dict_att[vid_name]):
            t_start = segment[0]
            t_end = segment[1]
            t_mid = (t_start + t_end) / 2
            segment_duration = t_end - t_start
            if dynamic_segment_weights_cumsum is not None:
                t_start = (f_upsample(t_start * time_to_index_factor + 1) - 1) / time_to_index_factor
                t_end = (f_upsample(t_end * time_to_index_factor + 1) - 1) / time_to_index_factor
                t_mid = (t_start + t_end) / 2
            segment_duration = t_end - t_start
            index_start = max(round((t_mid - (self.delta + 0.5) * segment_duration) * upsample_scale), 0)
            index_end = min(round((t_mid + (self.delta + 0.5) * segment_duration) * upsample_scale), vid_num_seg - 1)
            proposal_list_att.append([index_start, index_end])
        proposal_list_att = sorted(proposal_list_att, key=lambda k: k[0], reverse=True)

        proposal_count_by_video = len(proposal_list_att)
        for k, segment in enumerate(proposal_list_att):
            proposal_bbox[k, 0] = segment[0]
            proposal_bbox[k, 1] = segment[1]

        ########## generate pseudo_instance_label from pseudo_segment_all for Pseudo Instance-level Loss ##########
        fg_label_set_gt = np.where(self.label_dict[vid_name] == 1)[0]
        for segment in self.pseudo_segment_dict_all[vid_name]:
            t_start = segment[0]
            t_end = segment[1]
            t_label = self.class_name_to_idx[segment[3]]
            if not t_label in fg_label_set_gt:
                continue
            if dynamic_segment_weights_cumsum is not None:
                t_start = (f_upsample(t_start * time_to_index_factor + 1) - 1) / time_to_index_factor
                t_end = (f_upsample(t_end * time_to_index_factor + 1) - 1) / time_to_index_factor
            index_start = max(int(round(t_start * upsample_scale)), 0)
            index_end = min(int(round(t_end * upsample_scale)), vid_num_seg - 1)
            pseudo_instance_label[index_start:index_end + 1, t_label] = 1
            pseudo_instance_label[index_start:index_end + 1, -1] = 0
        pseudo_instance_label = pseudo_instance_label / torch.sum(pseudo_instance_label, dim=-1, keepdim=True).clamp(min=1e-6)
        return index, data, label, point_anno, stored_info, self.vid_list[index], vid_num_seg, proposal_bbox, proposal_count_by_video, pseudo_instance_label, dynamic_segment_weights_cumsum

    def get_data(self, index):
        vid_name = self.vid_list[index]

        vid_num_seg = 0

        if self.mode == 'full':
            if 'validation' in vid_name:
                feature = np.load(os.path.join(self.feature_path[0],
                                               vid_name + '.npy')).astype(np.float32)
            else:
                feature = np.load(os.path.join(self.feature_path[1],
                                               vid_name + '.npy')).astype(np.float32)
        else:
            feature = np.load(os.path.join(self.feature_path, vid_name + '.npy')).astype(np.float32)

        vid_num_seg = feature.shape[0]

        dynamic_segment_weights_cumsum = None
        sample_time = None
        dynamic_flag = False
        if self.sampling == 'random':
            vid_len_yuan = vid_num_seg
            sample_idx = self.random_perturb(vid_num_seg)
        elif self.sampling == 'uniform':
            vid_len_yuan = vid_num_seg
            sample_idx = self.uniform_sampling(vid_num_seg)
        elif self.sampling == 'dynamic_random':
            vid_len_yuan = vid_num_seg
            dynamic_segment_weights = np.load(os.path.join(self.dynamic_segment_weight_path, vid_name + ".npy"))
            feature, dynamic_segment_weights_cumsum, sample_time = dynamic_segment_sample(feature, dynamic_segment_weights)
            # 动态上采样后，更新特征长度
            vid_num_seg = feature.shape[0]
            sample_idx = self.random_perturb(vid_num_seg)
            dynamic_flag = True
        elif self.sampling == 'dynamic_uniform':
            vid_len_yuan = vid_num_seg
            dynamic_segment_weights = np.load(os.path.join(self.dynamic_segment_weight_path, vid_name + ".npy"))
            feature, dynamic_segment_weights_cumsum, sample_time = dynamic_segment_sample(feature, dynamic_segment_weights)
            # 动态上采样后，更新特征长度
            vid_num_seg = feature.shape[0]
            sample_idx = self.uniform_sampling(vid_num_seg)
            dynamic_flag = True
        else:
            raise AssertionError('Not supported sampling !')

        feature = feature[sample_idx]

        return torch.from_numpy(feature), vid_num_seg, sample_idx, sample_time, dynamic_flag, dynamic_segment_weights_cumsum, vid_len_yuan

    def get_label(self, index, vid_num_seg, sample_idx, sample_time, dynamic_flag):
        vid_name = self.vid_list[index]
        anno_list = self.anno['database'][vid_name]['annotations']
        label = np.zeros([self.num_classes], dtype=np.float32)

        classwise_anno = [[]] * self.num_classes

        for _anno in anno_list:
            label[self.class_name_to_idx[_anno['label']]] = 1
            classwise_anno[self.class_name_to_idx[_anno['label']]].append(_anno)

        if self.supervision == 'video':
            return label, torch.Tensor(0)

        elif self.supervision == 'point':
            temp_anno = np.zeros([vid_num_seg, self.num_classes], dtype=np.float32)
            t_factor = self.feature_fps / (self.fps_dict[vid_name] * 16)

            temp_df = self.point_anno[self.point_anno["video_id"] == vid_name][['point', 'class_index']]

            if dynamic_flag:
                for key in temp_df['point'].keys():
                    point = temp_df['point'][key]
                    class_idx = temp_df['class_index'][key]

                    temp_idx = int(point * t_factor)+1
                    mark_idx = np.where((sample_time >= (temp_idx-0.5)) & (sample_time < temp_idx+0.5))[0]
                    # print(sample_time)
                    # print(mark_idx, type(mark_idx))
                    # print(temp_idx)
                    start_idx = mark_idx[0]
                    end_idx = mark_idx[-1]+1
                    for t_dx in range(start_idx, end_idx):
                        temp_anno[t_dx][class_idx] = 1
            else:
                for key in temp_df['point'].keys():
                    point = temp_df['point'][key]
                    class_idx = temp_df['class_index'][key]

                    temp_anno[int(point * t_factor)][class_idx] = 1

            point_label = temp_anno[sample_idx, :]

            return label, torch.from_numpy(point_label)

    def random_perturb(self, length):
        if self.num_segments == length or self.num_segments == -1:
            return np.arange(length).astype(int)
        samples = np.arange(self.num_segments) * length / self.num_segments
        for i in range(self.num_segments):
            if i < self.num_segments - 1:
                if int(samples[i]) != int(samples[i + 1]):
                    samples[i] = np.random.choice(range(int(samples[i]), int(samples[i + 1]) + 1))
                else:
                    samples[i] = int(samples[i])
            else:
                if int(samples[i]) < length - 1:
                    samples[i] = np.random.choice(range(int(samples[i]), length))
                else:
                    samples[i] = int(samples[i])
        return samples.astype(int)

    def uniform_sampling(self, length):
        if length <= self.num_segments or self.num_segments == -1:
            return np.arange(length).astype(int)
        samples = np.arange(self.num_segments) * length / self.num_segments
        samples = np.floor(samples)
        return samples.astype(int)

def my_collate_fn(batch):
    r"""Puts each data field into a tensor with outer dimension batch size"""
    elem = batch[0]
    elem_type = type(elem)
    if isinstance(elem, torch.Tensor):
        out = None
        if torch.utils.data.get_worker_info() is not None:
            # If we're in a background process, concatenate directly into a
            # shared memory tensor to avoid an extra copy
            numel = sum([x.numel() for x in batch])
            storage = elem.storage()._new_shared(numel)
            out = elem.new(storage)
        return torch.stack(batch, 0, out=out)
    elif elem_type.__module__ == 'numpy' and elem_type.__name__ != 'str_' \
            and elem_type.__name__ != 'string_':
        if elem_type.__name__ == 'ndarray' or elem_type.__name__ == 'memmap':
            # array of string classes and object
            if np_str_obj_array_pattern.search(elem.dtype.str) is not None:
                raise TypeError(default_collate_err_msg_format.format(elem.dtype))

            return my_collate_fn([torch.as_tensor(b) for b in batch])
        elif elem.shape == ():  # scalars
            return torch.as_tensor(batch)
    elif isinstance(elem, float):
        return torch.tensor(batch, dtype=torch.float64)
    elif isinstance(elem, int_classes):
        return torch.tensor(batch)
    elif isinstance(elem, string_classes):
        return batch
    elif isinstance(elem, container_abcs.Mapping):
        return {key: my_collate_fn([d[key] for d in batch]) for key in elem}
    elif isinstance(elem, tuple) and hasattr(elem, '_fields'):  # namedtuple
        return elem_type(*(my_collate_fn(samples) for samples in zip(*batch)))
    elif isinstance(elem, container_abcs.Sequence):
        # check to make sure that the elements in batch have consistent size
        it = iter(batch)
        elem_size = len(next(it))
        if not all(len(elem) == elem_size for elem in it):
            raise RuntimeError('each element in list of batch should be of equal size')
        transposed = zip(*batch)
        return [my_collate_fn(samples) for samples in transposed]
    else:
        return batch

    # raise TypeError(default_collate_err_msg_format.format(elem_type))

default_collate_err_msg_format = (
    "default_collate: batch must contain tensors, numpy arrays, numbers, "
    "dicts or lists; found {}")

# def my_collate_fn(batch):
#     batched_output_list = []
#     for i in range(len(batch[0])):
#         if torch.is_tensor(batch[0][i]):
#             batched_output = torch.stack([item[i] for item in batch], dim=0)
#         else:
#             batched_output = [item[i] for item in batch]
#         batched_output_list.append(batched_output)
#     return batched_output_list


def dynamic_segment_sample(input_feature, dynamic_segment_weights):
    input_len = input_feature.shape[0]
    if input_len == 1:
        sample_len = 2
        sample_idxs = np.rint(np.linspace(0, input_len-1, sample_len))
        dynamic_segment_weights_cumsum = np.concatenate((np.zeros((1,), dtype=float), np.array([0.5, 1.0], dtype=float)), axis=0)
        return input_feature[sample_idxs.astype(np.int), :], dynamic_segment_weights_cumsum
    else:
        dynamic_segment_weights_cumsum = np.concatenate((np.zeros((1,), dtype=float), np.cumsum(dynamic_segment_weights)), axis=0)
        max_dynamic_segment_weights_cumsum = np.round(dynamic_segment_weights_cumsum[-1]).astype(int)
        f_upsample = interpolate.interp1d(dynamic_segment_weights_cumsum, np.arange(input_len+1), kind='linear', axis=0, fill_value='extrapolate')
        scale_x = np.linspace(1, max_dynamic_segment_weights_cumsum, max_dynamic_segment_weights_cumsum)
        sampled_time = f_upsample(scale_x)
        f_feature = interpolate.interp1d(np.arange(1, input_len+1), input_feature, kind='linear', axis=0, fill_value='extrapolate')
        sampled_feature = f_feature(sampled_time)
        return sampled_feature, dynamic_segment_weights_cumsum, sampled_time