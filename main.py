import pdb
import numpy as np
import torch
import torch.utils.data as data
import utils
from options import *
from config import *
from train import *
from test import *
from model import *
from search import *
from tensorboard_logger import Logger
from thumos_features import *
from tqdm import tqdm


def generate_pseudo_segment(config, net, test_loader, step):
    with torch.no_grad():
        net.eval()


        final_res = {}
        final_res['version'] = 'VERSION 1.3'
        final_res['results'] = {}
        final_res['external_data'] = {'used': True, 'details': 'Features from I3D Network'}

        num_correct = 0.
        num_total = 0.

        load_iter = iter(test_loader)

        for i in range(len(test_loader.dataset)):

            _, _data, _label, _point_anno, _, vid_name, vid_num_seg, proposal_bbox, proposal_count_by_video, pseudo_instance_label, dynamic_segment_weights_cumsum = next(load_iter)

            _data = _data.to(torch.device('cuda:1'))
            _label = _label.to(torch.device('cuda:1'))

            vid_num_seg = vid_num_seg[0].cpu().item()

            num_segments = _data.shape[1]

            vid_score, cas_sigmoid_fuse, _, _, _ = net(_data, vid_labels=_label, proposal_bbox=proposal_bbox,
                                                    proposal_count_by_video=proposal_count_by_video)
            vid_score = vid_score[0]
            cas_sigmoid_fuse = cas_sigmoid_fuse[0]

            agnostic_score = 1 - cas_sigmoid_fuse[:, :, -1].unsqueeze(2)
            cas_sigmoid_fuse = cas_sigmoid_fuse[:, :, :-1]

            label_np = _label.cpu().data.numpy()
            score_np = vid_score[0].cpu().data.numpy()

            pred_np = np.zeros_like(score_np)
            pred_np[np.where(score_np < config.class_thresh)] = 0
            pred_np[np.where(score_np >= config.class_thresh)] = 1

            if pred_np.sum() == 0:
                pred_np[np.argmax(score_np)] = 1

            correct_pred = np.sum(label_np == pred_np, axis=1)

            num_correct += np.sum((correct_pred == config.num_classes).astype(np.float32))
            num_total += correct_pred.shape[0]

            cas = cas_sigmoid_fuse

            pred = np.where(score_np >= config.class_thresh)[0]

            if len(pred) == 0:
                pred = np.array([np.argmax(score_np)])

            cas_pred = cas[0].cpu().numpy()[:, pred]
            cas_pred = np.reshape(cas_pred, (num_segments, -1, 1))

            cas_pred = utils.upgrade_resolution(cas_pred, config.scale)

            proposal_dict = {}

            agnostic_score = agnostic_score.expand((-1, -1, config.num_classes))
            agnostic_score_np = agnostic_score[0].cpu().data.numpy()[:, pred]
            agnostic_score_np = np.reshape(agnostic_score_np, (num_segments, -1, 1))
            agnostic_score_np = utils.upgrade_resolution(agnostic_score_np, config.scale)

            t_factor = float(16 * vid_num_seg) / (config.scale * num_segments * config.feature_fps)
            vid_duration = float(16 * vid_num_seg) / config.feature_fps
            for i in range(len(config.act_thresh_cas)):
                cas_temp = cas_pred.copy()

                zero_location = np.where(cas_temp[:, :, 0] < config.act_thresh_cas[i])
                cas_temp[zero_location] = 0

                seg_list = []
                for c in range(len(pred)):
                    pos = np.where(cas_temp[:, c, 0] > 0)
                    seg_list.append(pos)

                # 提取当前视频的标注点信息
                point_annotations = {}
                if _point_anno.shape[0] > 0:
                    _point_anno_np = _point_anno[0].cpu().numpy()
                    for c_idx in range(_point_anno_np.shape[1]):
                        point_positions = np.where(_point_anno_np[:,c_idx] > 0)[0].tolist()
                        if len(point_positions) > 0:
                            point_annotations[c_idx] = point_positions
                
                # 将标注点信息传递给get_proposal_oic函数
                proposals = utils.get_proposal_oic(seg_list, cas_temp, pred, score_np, t_factor, 
                                                   dynamic_segment_weights_cumsum=dynamic_segment_weights_cumsum[0] if dynamic_segment_weights_cumsum is not None else None, 
                                                   vid_duration=vid_duration,
                                                   point_annotations=point_annotations)

                for i in range(len(proposals)):
                    if len(proposals[i]) == 0:
                        continue
                    class_id = proposals[i][0][0]

                    if class_id not in proposal_dict.keys():
                        proposal_dict[class_id] = []

                    proposal_dict[class_id] += proposals[i]

            for i in range(len(config.act_thresh_agnostic)):
                cas_temp = cas_pred.copy()

                agnostic_score_np_temp = agnostic_score_np.copy()

                zero_location = np.where(agnostic_score_np_temp[:, :, 0] < config.act_thresh_agnostic[i])
                agnostic_score_np_temp[zero_location] = 0

                seg_list = []
                for c in range(len(pred)):
                    pos = np.where(agnostic_score_np_temp[:, c, 0] > 0)
                    seg_list.append(pos)
                
                # 使用相同的点标注信息
                proposals = utils.get_proposal_oic(seg_list, cas_temp, pred, score_np, t_factor,
                                                   dynamic_segment_weights_cumsum=dynamic_segment_weights_cumsum[0] if dynamic_segment_weights_cumsum is not None else None,
                                                   vid_duration=vid_duration,
                                                   point_annotations=point_annotations)

                for i in range(len(proposals)):
                    if len(proposals[i]) == 0:
                        continue
                    class_id = proposals[i][0][0]

                    if class_id not in proposal_dict.keys():
                        proposal_dict[class_id] = []

                    proposal_dict[class_id] += proposals[i]

            final_proposals = []
            for class_id in proposal_dict.keys():
                final_proposals.append(utils.nms(proposal_dict[class_id], thresh=0.7))

            final_proposals = [final_proposals[i][j] for i in range(len(final_proposals)) for j in
                               range(len(final_proposals[i]))]

            final_res['results'][vid_name[0]] = utils.result2json(final_proposals)

        test_acc = num_correct / num_total

        json_path = os.path.join(config.output_path, "pseudo_proposals_step{}.json".format(step))
        with open(json_path, 'w') as f:
            json.dump(final_res, f)
            f.close()

        tIoU_thresh = np.linspace(0.1, 0.7, 7)
        anet_detection = ANETdetection(config.gt_path, json_path,
                                       subset='train', tiou_thresholds=tIoU_thresh,
                                       verbose=False, check_status=False)
        mAP, _ = anet_detection.evaluate()


        return final_res


def generate_dynamic_segment_weights(args, pseudo_segment_dict, step=0):
    dynamic_segment_weight_path = os.path.join(args.output_path, 'dynamic_segment_weights_pred_step{}'.format(step))
    os.makedirs(dynamic_segment_weight_path, exist_ok=True)
    for vid_name in pseudo_segment_dict['results']:
        if 'test' in vid_name:
            feature = np.load(os.path.join('dataset/THUMOS14/features/{}'.format('test'), vid_name+".npy"))
        elif 'validation' in vid_name:
            feature = np.load(os.path.join('dataset/THUMOS14/features/{}'.format('train'), vid_name+".npy"))
        vid_len = feature.shape[0]

        label_set = set()
        if 'validation' in vid_name:
            for ann in args.gt_dict[vid_name]["annotations"]:
                label_set.add(ann['label'])
        else:
            for pred in pseudo_segment_dict['results'][vid_name]:
                label_set.add(pred['label'])

        prediction_list_all = []
        for label in label_set:
            prediction_list = []
            for pred in pseudo_segment_dict['results'][vid_name]:
                if pred['label'] == label:
                    t_start = pred["segment"][0]
                    t_end = pred["segment"][1]
                    prediction_list.append([t_start, t_end, pred["score"], pred["label"]])
            prediction_list = sorted(prediction_list, key=lambda k: k[2], reverse=True)

            # select top Q% segments to filter out low-confidence segments
            segment_score_list = []
            for pred in prediction_list:
                segment_score_list.append(pred[2])
            segment_score = np.array(segment_score_list)
            segment_score_cumsum = np.cumsum(segment_score)
            if segment_score_cumsum.shape[0] > 0:
                score_thres = np.max(segment_score_cumsum) * args.alpha
            else:
                score_thres = 0
                assert(len(prediction_list) == 0), 'num_segments not equal to 0'
            selected_proposal_count_by_video = np.where(segment_score_cumsum <= score_thres)[0].shape[0]
            prediction_list_all += prediction_list[:selected_proposal_count_by_video]

        time_to_index_factor = 25 / 16
        proposal_list = []
        for segment in prediction_list_all:
            t_start = segment[0]
            t_end = segment[1]
            t_mid = (t_start + t_end) / 2
            segment_duration = t_end - t_start
            index_start = max(round((t_mid - (args.delta+0.5) * segment_duration) * time_to_index_factor), 0)
            index_end = min(round((t_mid + (args.delta+0.5) * segment_duration) * time_to_index_factor), vid_len-1)
            if index_start < index_end:
                proposal_list.append([index_start, index_end])
        proposal_list = sorted(proposal_list, key=lambda k: k[0], reverse=True)

        upscale_duration = args.gamma * (2 * args.delta + 1)
        dynamic_segment_weights = np.ones((vid_len,), dtype=float)
        for proposal in proposal_list:
            index_start = proposal[0]
            index_end = proposal[1]
            if (index_end - index_start + 1) <= float(upscale_duration):
                for index in range(index_start, index_end+1):
                    dynamic_segment_weights[index] = max(dynamic_segment_weights[index], min(float(upscale_duration) / (index_end - index_start + 1), float(upscale_duration)))

        ### normalize the weights of fg segments ###
        fg_pos = np.where(dynamic_segment_weights > 1.0)
        fg_temp_list = np.array(fg_pos)[0]
        if fg_temp_list.any():
            grouped_fg_temp_list = utils.grouping(fg_temp_list)
            for k in range(len(grouped_fg_temp_list)):
                segment_score_sum = np.sum(dynamic_segment_weights[grouped_fg_temp_list[k]])
                segment_score_sum_round = np.round(segment_score_sum)
                dynamic_segment_weights[grouped_fg_temp_list[k]] = segment_score_sum_round * dynamic_segment_weights[grouped_fg_temp_list[k]] / segment_score_sum
        np.save(os.path.join(dynamic_segment_weight_path, "{}.npy".format(vid_name)), dynamic_segment_weights)
    return


if __name__ == "__main__":
    args = parse_args()
    if args.debug:
        pdb.set_trace()

    config = Config(args)
    worker_init_fn = None
   
    if config.seed >= 0:
        utils.set_seed(config.seed)
        worker_init_fn = np.random.seed(config.seed)

    utils.save_config(config, os.path.join(config.output_path, "config.txt"))

    model_list = []
    optimizer_list = []
    criterion_list = []
    train_dataset_list = []
    test_dataset_list = []
    full_dataset_list = []

    for step in range(config.num_steps+1):
        if step == 0:
            # For the first step
            net = Model(config.len_feature, config.num_classes, config.r_act)
            criterion = Total_loss(config.lambdas)
            train_dataset = ThumosFeature(config, data_path=config.data_path, mode='train',
                                          modal=config.modal, feature_fps=config.feature_fps,
                                          num_segments=-1, sampling='random', step=step,
                                          supervision='point', seed=config.seed)
            test_dataset = ThumosFeature(config, data_path=config.data_path, mode='test',
                                         modal=config.modal, feature_fps=config.feature_fps,
                                         num_segments=-1, sampling='random', step=step,
                                         supervision='point', seed=config.seed)
            full_dataset = ThumosFeature(config, data_path=config.data_path, mode='full',
                                         modal=config.modal, feature_fps=config.feature_fps,
                                         num_segments=-1, sampling='random', step=step,
                                         supervision='point', seed=config.seed)
        else:
            # For the following steps
            net = Model_GAI(config.len_feature, config.num_classes, config.r_act)
            
            # 加载上一个阶段的模型权重（如果存在）
            if args.resume and step > 0:
                prev_model_path = os.path.join(args.model_path, f"model_seed_{config.seed}_Iter_{step-1}.pkl")
                if os.path.exists(prev_model_path):
                    print(f"Loading weights from {prev_model_path} for step {step}")
                    state_dict = torch.load(prev_model_path, map_location=torch.device('cpu'))
                    # 尝试加载尽可能多的参数，忽略不匹配的部分
                    try:
                        net.load_state_dict(state_dict, strict=False)
                        print("Successfully loaded model weights")
                    except Exception as e:
                        print(f"Warning: Error loading weights: {e}")
                else:
                    print(f"Warning: Could not find model file {prev_model_path} for step {step}. Starting from scratch.")
            
            criterion = Total_loss_Gai(config.lambdas)
            train_dataset = ThumosFeature(config, data_path=config.data_path, mode='train',
                                          modal=config.modal, feature_fps=config.feature_fps,
                                          num_segments=-1, sampling='dynamic_random', step=step,
                                          supervision='point', seed=config.seed)
            test_dataset = ThumosFeature(config, data_path=config.data_path, mode='test',
                                         modal=config.modal, feature_fps=config.feature_fps,
                                         num_segments=-1, sampling='random', step=step,
                                         supervision='point', seed=config.seed)
            full_dataset = ThumosFeature(config, data_path=config.data_path, mode='full',
                                         modal=config.modal, feature_fps=config.feature_fps,
                                         num_segments=-1, sampling='random', step=step,
                                         supervision='point', seed=config.seed)
        device = torch.device("cuda:1")

        net = net.to(device)
        optimizer = torch.optim.Adam(net.parameters(), lr=config.lr[0],
                                     betas=(0.9, 0.999), weight_decay=0.0005)
        train_dataset_list.append(train_dataset)
        test_dataset_list.append(test_dataset)
        full_dataset_list.append(full_dataset)
        model_list.append(net)
        criterion_list.append(criterion)
        optimizer_list.append(optimizer)

    test_info = {"iter": [], "step": [], "test_acc": [],
                 "average_mAP[0.1:0.7]": [], "average_mAP[0.1:0.5]": [], "average_mAP[0.3:0.7]": [],
                 "mAP@0.1": [], "mAP@0.2": [], "mAP@0.3": [], "mAP@0.4": [],
                 "mAP@0.5": [], "mAP@0.6": [], "mAP@0.7": []}

    best_mAP = -1

    logger = Logger(config.log_path)

    step_iter = 0
    net = model_list[step_iter]
    optimizer = optimizer_list[step_iter]
    criterion = criterion_list[step_iter]
    train_loader = data.DataLoader(
        train_dataset_list[step_iter],
        batch_size=config.batch_size, collate_fn=my_collate_fn,
        shuffle=True, num_workers=config.num_workers,
        worker_init_fn=worker_init_fn if config.num_workers > 0 else None,
        pin_memory=config.num_workers > 0)  # 当num_workers=0时不需要pin_memory
    test_loader = data.DataLoader(
        test_dataset_list[step_iter],
        batch_size=1, collate_fn=my_collate_fn,
        shuffle=True, num_workers=config.num_workers,
        worker_init_fn=worker_init_fn,
        pin_memory=True)
    full_loader = data.DataLoader(
        full_dataset_list[step_iter],
        batch_size=1, collate_fn=my_collate_fn,
        shuffle=True, num_workers=config.num_workers,
        worker_init_fn=worker_init_fn,
        pin_memory=True)
    print(net, flush=True)

    # final_res = {}
    # final_res['version'] = 'VERSION 1.3'
    # final_res['results'] = {}
    # final_res['external_data'] = {'used': True, 'details': 'Features from I3D Network'}

    loader_iter = iter(train_loader)
    
    # 支持从指定迭代次数开始训练
    start_iter = args.start_iter if args.resume else 0
    if start_iter > 0:
        print(f"Resuming training from iteration {start_iter}")
        # 计算当前应该处于哪个step_iter
        if start_iter <= config.epochs_per_step * config.num_steps:
            current_step_iter = start_iter // config.epochs_per_step
            if current_step_iter > 0:
                step_iter = current_step_iter
                # 重要：加载最近一个已保存的模型
                prev_model_path = os.path.join(args.model_path, f"model_seed_{config.seed}_Iter_{step_iter-1}.pkl")
                if os.path.exists(prev_model_path):
                    print(f"Loading weights from {prev_model_path}")
                    # 确保所有模型都加载这个权重，防止评估时使用未训练的模型
                    state_dict = torch.load(prev_model_path, map_location=torch.device('cpu'))
                    for i in range(step_iter+1):  # 包括当前step_iter
                        try:
                            model_list[i].load_state_dict(state_dict, strict=False)
                            print(f"Successfully loaded weights for model_list[{i}]")
                        except Exception as e:
                            print(f"Warning: Error loading weights for model_list[{i}]: {e}")
                
                net = model_list[step_iter]
                optimizer = optimizer_list[step_iter]
                criterion = criterion_list[step_iter]
                train_loader = data.DataLoader(
                    train_dataset_list[step_iter],
                    batch_size=config.batch_size, collate_fn=my_collate_fn,
                    shuffle=True, num_workers=config.num_workers,
                    worker_init_fn=worker_init_fn if config.num_workers > 0 else None,
                    pin_memory=config.num_workers > 0)
                loader_iter = iter(train_loader)
                print(f"Switched to step_iter {step_iter} based on start_iter")
    
    for step in tqdm(
            range(start_iter, config.num_iters),
            total=config.num_iters - start_iter,
            dynamic_ncols=True
    ):
        if 0 < step <= config.epochs_per_step * config.num_steps and step % config.epochs_per_step == 0:
            step_iter = step // config.epochs_per_step
            best_mAP = -1
            with torch.no_grad():
                train_loader_pseudo = data.DataLoader(
                    ThumosFeature(config, data_path=config.data_path, mode='train',
                                  modal=config.modal, feature_fps=config.feature_fps,
                                  num_segments=-1, sampling='random', step=step,
                                  supervision='point', seed=config.seed),
                    batch_size=1, collate_fn=my_collate_fn,
                    shuffle=True, num_workers=config.num_workers,
                    worker_init_fn=worker_init_fn)
                pseudo_segment_dict = generate_pseudo_segment(config, net, train_loader_pseudo, step=step_iter)
                # generate dynamic segment weights according to the predicted pseudo_segment_dict
                generate_dynamic_segment_weights(config, pseudo_segment_dict, step=step_iter)
                # pass the generated pseudo_segment_dict into the dataset class to generate the proposal bounding box and pseudo label
                train_dataset_list[step_iter].get_proposals(pseudo_segment_dict)

                train_loader = data.DataLoader(
                    train_dataset_list[step_iter],
                    batch_size=config.batch_size, collate_fn=my_collate_fn, # <-- 使用真实的批处理
                    shuffle=True, num_workers=config.num_workers,
                    worker_init_fn=worker_init_fn if config.num_workers > 0 else None,
                    pin_memory=config.num_workers > 0)  # 当num_workers=0时不需要pin_memory
                test_loader = data.DataLoader(
                    test_dataset_list[step_iter],
                    batch_size=1, collate_fn=my_collate_fn,
                    shuffle=True, num_workers=config.num_workers,
                    worker_init_fn=worker_init_fn if config.num_workers > 0 else None,
                    pin_memory=config.num_workers > 0)  # 当num_workers=0时不需要pin_memory
                full_loader = data.DataLoader(
                    full_dataset_list[step_iter],
                    batch_size=1, collate_fn=my_collate_fn,
                    shuffle=True, num_workers=config.num_workers,
                    worker_init_fn=worker_init_fn if config.num_workers > 0 else None,
                    pin_memory=config.num_workers > 0)  # 当num_workers=0时不需要pin_memory
            net = model_list[step_iter]
            optimizer = optimizer_list[step_iter]
            criterion = criterion_list[step_iter]
            loader_iter = iter(train_loader)
            print(net, flush=True)

        if step > 0 and config.lr[step] != config.lr[step - 1]:
            for param_group in optimizer.param_groups:
                param_group["lr"] = config.lr[step]
        
        try:
            batch = next(loader_iter)
        except StopIteration:
            loader_iter = iter(train_loader)
            batch = next(loader_iter)

        train(net, config, batch, optimizer, criterion, logger, step)

        if (step+1) % config.search_freq == 0:
            optimal_sequence_search(net, config, logger, train_loader)

        # 减少评估频率，加快训练
        if (step+1) % 50 == 0:
            test_res = test(net, config, logger, test_loader, test_info, step, step_iter)
            if test_info["average_mAP[0.1:0.7]"][-1] > best_mAP:
                best_mAP = test_info["average_mAP[0.1:0.7]"][-1]

                utils.save_best_record_thumos(test_info,
                                              os.path.join(config.output_path,
                                                           "best_record_seed_{}_Iter_{}.txt".format(config.seed, step_iter)))

                # 确保保存正确的模型文件名，特别是在断点续训时
                current_iter = step_iter
                if args.resume and args.start_iter > 0:
                    # 如果是从中间恢复训练，确保使用正确的step_iter
                    # 这样可以避免覆盖之前的模型文件
                    print(f"Saving model with correct step_iter: {current_iter}")
                
                torch.save(net.state_dict(), os.path.join(args.model_path, \
                                                          "model_seed_{}_Iter_{}.pkl".format(config.seed, current_iter)))




