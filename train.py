import torch
import torch.nn as nn
import utils
from losses import AttLoss, NormalizedCrossEntropy, CategoryCrossEntropy


class Total_loss(nn.Module):
    def __init__(self, lambdas):
        super(Total_loss, self).__init__()
        self.tau = 0.1
        self.sampling_size = 3
        self.lambdas = lambdas
        self.ce_criterion = nn.BCELoss(reduction='none')
        self.loss_att = AttLoss(8.0)
        self.loss_nce = NormalizedCrossEntropy()
        self.loss_spl = CategoryCrossEntropy(0.2)

    def forward(self, vid_score, cas_sigmoid_fuse, features, cls_att, stored_info, label, point_anno, step, pseudo_instance_label, uncertainty):
        loss = {}

        loss_vid = self.ce_criterion(vid_score[0], label) * 0.5 + self.ce_criterion(vid_score[1], label)
        loss_vid = loss_vid.mean()

        # 修改 2023-0212 ***1**
        o_vid_ca_pred, m_vid_ca_pred, o_vid_cw_pred, m_vid_cw_pred, o_att, m_att, o_frm_pred, m_frm_pred = cls_att
        f_labels = torch.cat([label, torch.zeros(label.size(0), 1).to(torch.device('cuda:1'))], -1)
        b_labels = torch.cat([label, torch.ones(label.size(0), 1).to(torch.device('cuda:1'))], -1)

        vid_fore_loss = (self.ce_criterion(o_vid_ca_pred, f_labels)).mean() + (self.ce_criterion(m_vid_ca_pred, f_labels)).mean()
        vid_back_loss = (self.ce_criterion(o_vid_cw_pred, b_labels)).mean() + (self.ce_criterion(m_vid_cw_pred, b_labels)).mean()

        vid_att_loss = self.loss_att(o_att)
        # print(vid_att_loss)
        # vid_spl_loss = self.loss_spl(o_frm_pred, m_frm_pred)
        loss_vid = vid_fore_loss + vid_back_loss * 0.2 + vid_att_loss * 0.1

        
        point_anno = torch.cat((point_anno, torch.zeros((point_anno.shape[0], point_anno.shape[1], 1)).to(torch.device('cuda:1'))), dim=2)
        
        weighting_seq_act = point_anno.max(dim=2, keepdim=True)[0]
        num_actions = point_anno.max(dim=2)[0].sum(dim=1)

        focal_weight_act = (1 - cas_sigmoid_fuse[0]) * point_anno + cas_sigmoid_fuse[0] * (1 - point_anno)
        focal_weight_act = focal_weight_act ** 2

        focal_weight_act_r = (1 - cas_sigmoid_fuse[1]) * point_anno + cas_sigmoid_fuse[1] * (1 - point_anno)
        focal_weight_act_r = focal_weight_act_r ** 2

        loss_frame_1 = (((focal_weight_act * self.ce_criterion(cas_sigmoid_fuse[0], point_anno) * weighting_seq_act).sum(dim=2)).sum(dim=1) / num_actions).mean()
        loss_frame_2 = (((focal_weight_act_r * self.ce_criterion(cas_sigmoid_fuse[1], point_anno) * weighting_seq_act).sum(dim=2)).sum(dim=1) / num_actions).mean()
        loss_frame = loss_frame_1 * 0.5 + loss_frame_2 * 0.5

        _, bkg_seed_1 = utils.select_seed(cas_sigmoid_fuse[0].detach().cpu(), point_anno.detach().cpu())
        _, bkg_seed_2 = utils.select_seed(cas_sigmoid_fuse[1].detach().cpu(), point_anno.detach().cpu())

            
        bkg_seed_1 = bkg_seed_1.unsqueeze(-1).to(torch.device('cuda:1'))
        bkg_seed_2 = bkg_seed_2.unsqueeze(-1).to(torch.device('cuda:1'))

        point_anno_bkg = torch.zeros_like(point_anno).to(torch.device('cuda:1'))
        point_anno_bkg[:,:,-1] = 1

        weighting_seq_bkg_1 = bkg_seed_1
        num_bkg_1 = bkg_seed_1.sum(dim=1)

        focal_weight_bkg = (1 - cas_sigmoid_fuse[0]) * point_anno_bkg + cas_sigmoid_fuse[0] * (1 - point_anno_bkg)
        focal_weight_bkg = focal_weight_bkg ** 2

        loss_frame_bkg_1 = (((focal_weight_bkg * self.ce_criterion(cas_sigmoid_fuse[0], point_anno_bkg) * weighting_seq_bkg_1).sum(dim=2)).sum(dim=1) / num_bkg_1).mean()

        weighting_seq_bkg_2 = bkg_seed_2
        num_bkg_2 = bkg_seed_2.sum(dim=1)

        focal_weight_bkg_r = (1 - cas_sigmoid_fuse[1]) * point_anno_bkg + cas_sigmoid_fuse[1] * (1 - point_anno_bkg)
        focal_weight_bkg_r = focal_weight_bkg_r ** 2
        loss_frame_bkg_2 = (((focal_weight_bkg_r * self.ce_criterion(cas_sigmoid_fuse[1], point_anno_bkg) * weighting_seq_bkg_2).sum(dim=2)).sum(dim=1) / num_bkg_2).mean()
        loss_frame_bkg = loss_frame_bkg_1 * 0.5 + loss_frame_bkg_2 * 0.5
        
        loss_score_act = 0
        loss_score_bkg = 0
        loss_feat = 0

        # 批处理后, stored_info 是一个字典, 其值是列表
        new_dense_anno_list = stored_info['new_dense_anno']

        # 检查列表中的第一个元素，以确定如何处理
        if isinstance(new_dense_anno_list[0], torch.Tensor) and new_dense_anno_list[0].ndim > 1:
            # 列表推导和张量操作需要逐个处理
            for i in range(len(new_dense_anno_list)):
                new_dense_anno = new_dense_anno_list[i].to(torch.device('cuda:1'))
                
                # 确保 new_dense_anno 是三维的
                if new_dense_anno.ndim == 2:
                    new_dense_anno = new_dense_anno.unsqueeze(0)

                if new_dense_anno.ndim <= 1: continue

                new_dense_anno = torch.cat((new_dense_anno, torch.zeros((new_dense_anno.shape[0], new_dense_anno.shape[1], 1)).to(torch.device('cuda:1'))), dim=2)
                        
                act_idx_diff = new_dense_anno[:,1:] - new_dense_anno[:,:-1]
                loss_score_act = 0
                loss_feat = 0
                for b in range(new_dense_anno.shape[0]):
                    gt_classes = torch.nonzero(label[b]).squeeze(1)
                    act_count = 0
                    loss_score_act_batch = 0
                    loss_feat_batch = 0

                    for c in gt_classes:
                        range_idx = torch.nonzero(act_idx_diff[b,:,c]).squeeze(1)
                        range_idx = range_idx.cpu().data.numpy().tolist()
                        if type(range_idx) is not list:
                            range_idx = [range_idx]
                        if len(range_idx) == 0:
                            continue
                        # 排除两个边界存在动作实例c
                        if act_idx_diff[b, range_idx[0], c] != 1:
                            range_idx = [-1] + range_idx 
                        if act_idx_diff[b, range_idx[-1], c] != -1:
                            range_idx = range_idx + [act_idx_diff.shape[1] - 1]
                            
                        label_lst = []
                        feature_lst = []

                        if range_idx[0] > -1:
                            start_bkg = 0
                            end_bkg = range_idx[0]
                            bkg_len = end_bkg - start_bkg + 1

                            label_lst.append(0)
                            feature_lst.append(utils.feature_sampling(features[b], start_bkg, end_bkg + 1, self.sampling_size))

                        for i in range(len(range_idx) // 2):
                            if range_idx[2*i + 1] - range_idx[2*i] < 1:
                                continue

                            label_lst.append(1)
                            feature_lst.append(utils.feature_sampling(features[b], range_idx[2*i] + 1, range_idx[2*i + 1] + 1, self.sampling_size))

                            if range_idx[2*i + 1] != act_idx_diff.shape[1] - 1:
                                start_bkg = range_idx[2*i + 1] + 1

                                if i == (len(range_idx) // 2 - 1):
                                    end_bkg = act_idx_diff.shape[1] - 1
                                else:
                                    end_bkg = range_idx[2*i + 2]

                                bkg_len = end_bkg - start_bkg + 1

                                label_lst.append(0)
                                feature_lst.append(utils.feature_sampling(features[b], start_bkg, end_bkg + 1, self.sampling_size))

                        start_act = range_idx[2*i] + 1
                        end_act = range_idx[2*i + 1]

                        # 获取当前类别的标注点位置
                        point_positions = []
                        for t in range(point_anno.shape[1]):
                            if point_anno[b, t, c] > 0:  # 如果该位置有标注点
                                point_positions.append(t)
                                
                        # 结合显著性计算代表性得分
                        complete_score_act_1 = utils.get_oic_score(cas_sigmoid_fuse[0][b,:,c], start=start_act, end=end_act, point_annotations=point_positions)
                        complete_score_act_2 = utils.get_oic_score(cas_sigmoid_fuse[1][b,:,c], start=start_act, end=end_act, point_annotations=point_positions)
                        
                        loss_score_act_batch += 1 - complete_score_act_1 * 0.5 - complete_score_act_2 * 0.5

                        act_count += 1

                    if sum(label_lst) > 1:
                        feature_lst_tensor = torch.stack(feature_lst, 0).clone()
                        feature_lst_tensor = feature_lst_tensor / torch.norm(feature_lst_tensor, dim=1, p=2).unsqueeze(1)
                        label_lst_tensor = torch.tensor(label_lst).to(torch.device('cuda:1')).float()

                        sim_matrix = torch.matmul(feature_lst_tensor, torch.transpose(feature_lst_tensor, 0, 1)) / self.tau
                        sim_matrix = torch.exp(sim_matrix)
                        sim_matrix = sim_matrix.clone().fill_diagonal_(0).to(torch.device('cuda:1'))

                        scores = (sim_matrix * label_lst_tensor.unsqueeze(1)).sum(dim=0) / sim_matrix.sum(dim=0)
                        
                        loss_feat_batch = (-label_lst_tensor * torch.log(scores)).sum() / label_lst_tensor.sum()

                if act_count > 0:
                    loss_score_act += loss_score_act_batch / act_count
                    loss_feat += loss_feat_batch

                
            bkg_idx_diff = (1 - new_dense_anno[:,1:]) - (1 - new_dense_anno[:,:-1])
            loss_score_bkg = 0
            for b in range(new_dense_anno.shape[0]):
                gt_classes = torch.nonzero(label[b]).squeeze(1)
                loss_score_bkg_batch = 0
                bkg_count = 0

                for c in gt_classes:
                    range_idx = torch.nonzero(bkg_idx_diff[b,:,c]).squeeze(1)
                    range_idx = range_idx.cpu().data.numpy().tolist()
                    if type(range_idx) is not list:
                        range_idx = [range_idx]
                    if len(range_idx) == 0:
                        continue
                    if bkg_idx_diff[b, range_idx[0], c] != 1:
                        range_idx = [-1] + range_idx 
                    if bkg_idx_diff[b, range_idx[-1], c] != -1:
                        range_idx = range_idx + [bkg_idx_diff.shape[1] - 1]

                    for i in range(len(range_idx) // 2):
                        if range_idx[2*i + 1] - range_idx[2*i] < 1:
                            continue
                        
                        start_bkg = range_idx[2*i] + 1
                        end_bkg = range_idx[2*i + 1]

                        # 获取当前类别的标注点位置（用于背景区域的显著性计算）
                        point_positions = []
                        for t in range(point_anno.shape[1]):
                            if point_anno[b, t, c] > 0:  # 如果该位置有标注点
                                point_positions.append(t)
                                
                        # 结合显著性计算背景区域的代表性得分
                        complete_score_bkg_1 = utils.get_oic_score(1 - cas_sigmoid_fuse[0][b,:,c], start=start_bkg, end=end_bkg, point_annotations=point_positions)
                        complete_score_bkg_2 = utils.get_oic_score(1 - cas_sigmoid_fuse[1][b, :, c], start=start_bkg, end=end_bkg, point_annotations=point_positions)
                        
                        loss_score_bkg_batch += 1 - complete_score_bkg_1 * 0.5 - complete_score_bkg_2 * 0.5

                        bkg_count += 1

                if bkg_count > 0:
                    loss_score_bkg += loss_score_bkg_batch / bkg_count
                    
            loss_score_act = loss_score_act / new_dense_anno.shape[0]
            loss_score_bkg = loss_score_bkg / new_dense_anno.shape[0]
            
            loss_feat = loss_feat / new_dense_anno.shape[0]

        loss_score = (loss_score_act + loss_score_bkg) ** 2

        # loss_total = self.lambdas[0] * loss_vid + self.lambdas[1] * loss_frame + self.lambdas[2] * loss_frame_bkg
        loss_total = self.lambdas[0] * loss_vid + self.lambdas[1] * loss_frame + self.lambdas[2] * loss_frame_bkg + self.lambdas[3] * loss_score + self.lambdas[4] * loss_feat


        loss["loss_vid"] = loss_vid
        loss["loss_frame"] = loss_frame
        loss["loss_frame_bkg"] = loss_frame_bkg
        loss["loss_score_act"] = loss_score_act
        loss["loss_score_bkg"] = loss_score_bkg
        loss["loss_score"] = loss_score
        loss["loss_feat"] = loss_feat
        loss["loss_total"] = loss_total

        return loss_total, loss


class Total_loss_Gai(nn.Module):
    def __init__(self, lambdas):
        super(Total_loss_Gai, self).__init__()
        self.tau = 0.1
        self.sampling_size = 3
        self.lambdas = lambdas
        self.ce_criterion = nn.BCELoss(reduction='none')
        self.beta = 0.2
        self.loss_att = AttLoss(8.0)
        self.loss_nce = NormalizedCrossEntropy()
        self.loss_spl = CategoryCrossEntropy(0.2)

    def cls_criterion(self, inputs, label, uncertainty=None):
        if not uncertainty is None:
            loss1 = -torch.mean(torch.sum(torch.exp(-uncertainty) * torch.log(inputs.clamp(min=1e-7)) * label, dim=-1)) #(B, T, C) -> (B, T) -> 1
            loss2 = self.beta * torch.mean(uncertainty) #(B, T, 1) -> 1
            return loss1 + loss2
        else:
            return -torch.mean(torch.sum(torch.log(inputs.clamp(min=1e-7)) * label, dim=-1))

    def forward(self, vid_score, cas_sigmoid_fuse, features, cls_att, stored_info, label, point_anno, step, pseudo_instance_label, uncertainty):
        loss = {}

        loss_vid = self.ce_criterion(vid_score[0], label) * 0.5 + self.ce_criterion(vid_score[1], label)
        loss_vid = loss_vid.mean()

        # 修改 2023-0212 ***1**
        o_vid_ca_pred, m_vid_ca_pred, o_vid_cw_pred, m_vid_cw_pred, o_att, m_att, o_frm_pred, m_frm_pred = cls_att
        f_labels = torch.cat([label, torch.zeros(label.size(0), 1).to(torch.device('cuda:1'))], -1)
        b_labels = torch.cat([label, torch.ones(label.size(0), 1).to(torch.device('cuda:1'))], -1)

        vid_fore_loss = (self.ce_criterion(o_vid_ca_pred, f_labels)).mean() + (
            self.ce_criterion(m_vid_ca_pred, f_labels)).mean()
        vid_back_loss = (self.ce_criterion(o_vid_cw_pred, b_labels)).mean() + (
            self.ce_criterion(m_vid_cw_pred, b_labels)).mean()
        vid_att_loss = self.loss_att(o_att)
        # vid_att_loss = self.loss_att(o_att)
        # vid_spl_loss = self.loss_spl(o_frm_pred, m_frm_pred)
        loss_vid = vid_fore_loss + vid_back_loss * 0.2 + vid_att_loss * 0.1


        point_anno = torch.cat((point_anno, torch.zeros((point_anno.shape[0], point_anno.shape[1], 1)).to(torch.device('cuda:1'))), dim=2)

        weighting_seq_act = point_anno.max(dim=2, keepdim=True)[0]
        num_actions = point_anno.max(dim=2)[0].sum(dim=1)

        focal_weight_act = (1 - cas_sigmoid_fuse[0]) * point_anno + cas_sigmoid_fuse[0] * (1 - point_anno)
        focal_weight_act = focal_weight_act ** 2

        focal_weight_act_r = (1 - cas_sigmoid_fuse[1]) * point_anno + cas_sigmoid_fuse[1] * (1 - point_anno)
        focal_weight_act_r = focal_weight_act_r ** 2

        loss_frame_1 = (((focal_weight_act * self.ce_criterion(cas_sigmoid_fuse[0],
                                                               point_anno) * weighting_seq_act).sum(dim=2)).sum(
            dim=1) / num_actions).mean()
        loss_frame_2 = (((focal_weight_act_r * self.ce_criterion(cas_sigmoid_fuse[1],
                                                                 point_anno) * weighting_seq_act).sum(dim=2)).sum(
            dim=1) / num_actions).mean()
        loss_frame = loss_frame_1 * 0.5 + loss_frame_2 * 0.5

        _, bkg_seed_1 = utils.select_seed(cas_sigmoid_fuse[0].detach().cpu(), point_anno.detach().cpu())
        _, bkg_seed_2 = utils.select_seed(cas_sigmoid_fuse[1].detach().cpu(), point_anno.detach().cpu())

        bkg_seed_1 = bkg_seed_1.unsqueeze(-1).to(torch.device('cuda:1'))
        bkg_seed_2 = bkg_seed_2.unsqueeze(-1).to(torch.device('cuda:1'))

        point_anno_bkg = torch.zeros_like(point_anno).to(torch.device('cuda:1'))
        point_anno_bkg[:, :, -1] = 1

        weighting_seq_bkg_1 = bkg_seed_1
        num_bkg_1 = bkg_seed_1.sum(dim=1)

        focal_weight_bkg = (1 - cas_sigmoid_fuse[0]) * point_anno_bkg + cas_sigmoid_fuse[0] * (1 - point_anno_bkg)
        focal_weight_bkg = focal_weight_bkg ** 2

        loss_frame_bkg_1 = (((focal_weight_bkg * self.ce_criterion(cas_sigmoid_fuse[0],
                                                                   point_anno_bkg) * weighting_seq_bkg_1).sum(
            dim=2)).sum(dim=1) / num_bkg_1).mean()

        weighting_seq_bkg_2 = bkg_seed_2
        num_bkg_2 = bkg_seed_2.sum(dim=1)

        focal_weight_bkg_r = (1 - cas_sigmoid_fuse[1]) * point_anno_bkg + cas_sigmoid_fuse[1] * (1 - point_anno_bkg)
        focal_weight_bkg_r = focal_weight_bkg_r ** 2
        loss_frame_bkg_2 = (((focal_weight_bkg_r * self.ce_criterion(cas_sigmoid_fuse[1],
                                                                     point_anno_bkg) * weighting_seq_bkg_2).sum(
            dim=2)).sum(dim=1) / num_bkg_2).mean()
        loss_frame_bkg = loss_frame_bkg_1 * 0.5 + loss_frame_bkg_2 * 0.5

        loss_score_act = 0
        loss_score_bkg = 0
        loss_feat = 0

        # 批处理后, stored_info 是一个字典, 其值是列表
        new_dense_anno_list = stored_info['new_dense_anno']

        # 检查列表中的第一个元素，以确定如何处理
        if isinstance(new_dense_anno_list[0], torch.Tensor) and new_dense_anno_list[0].ndim > 1:
            # 列表推导和张量操作需要逐个处理
            for i in range(len(new_dense_anno_list)):
                new_dense_anno = new_dense_anno_list[i].to(torch.device('cuda:1'))
                
                # 确保 new_dense_anno 是三维的
                if new_dense_anno.ndim == 2:
                    new_dense_anno = new_dense_anno.unsqueeze(0)

                if new_dense_anno.ndim <= 1: continue
 
                new_dense_anno = torch.cat(
                    (new_dense_anno, torch.zeros((new_dense_anno.shape[0], new_dense_anno.shape[1], 1)).to(torch.device('cuda:1'))), dim=2)

                act_idx_diff = new_dense_anno[:, 1:] - new_dense_anno[:, :-1]
                loss_score_act = 0
                loss_feat = 0
                for b in range(new_dense_anno.shape[0]):
                    gt_classes = torch.nonzero(label[b]).squeeze(1)
                    act_count = 0
                    loss_score_act_batch = 0
                    loss_feat_batch = 0

                    for c in gt_classes:
                        range_idx = torch.nonzero(act_idx_diff[b, :, c]).squeeze(1)
                        range_idx = range_idx.cpu().data.numpy().tolist()
                        if type(range_idx) is not list:
                            range_idx = [range_idx]
                        if len(range_idx) == 0:
                            continue
                        if act_idx_diff[b, range_idx[0], c] != 1:
                            range_idx = [-1] + range_idx
                        if act_idx_diff[b, range_idx[-1], c] != -1:
                            range_idx = range_idx + [act_idx_diff.shape[1] - 1]

                        label_lst = []
                        feature_lst = []

                        if range_idx[0] > -1:
                            start_bkg = 0
                            end_bkg = range_idx[0]
                            bkg_len = end_bkg - start_bkg + 1

                            label_lst.append(0)
                            feature_lst.append(
                                utils.feature_sampling(features[b], start_bkg, end_bkg + 1, self.sampling_size))

                        for i in range(len(range_idx) // 2):
                            if range_idx[2 * i + 1] - range_idx[2 * i] < 1:
                                continue

                            label_lst.append(1)
                            feature_lst.append(
                                utils.feature_sampling(features[b], range_idx[2 * i] + 1, range_idx[2 * i + 1] + 1,
                                                   self.sampling_size))

                        if range_idx[2 * i + 1] != act_idx_diff.shape[1] - 1:
                            start_bkg = range_idx[2 * i + 1] + 1

                            if i == (len(range_idx) // 2 - 1):
                                end_bkg = act_idx_diff.shape[1] - 1
                            else:
                                end_bkg = range_idx[2 * i + 2]

                            bkg_len = end_bkg - start_bkg + 1

                            label_lst.append(0)
                            feature_lst.append(
                                utils.feature_sampling(features[b], start_bkg, end_bkg + 1, self.sampling_size))

                        start_act = range_idx[2 * i] + 1
                        end_act = range_idx[2 * i + 1]

                        # 获取当前类别的标注点位置
                        point_positions = []
                        for t in range(point_anno.shape[1]):
                            if point_anno[b, t, c] > 0:  # 如果该位置有标注点
                                point_positions.append(t)
                                
                        # 结合显著性计算代表性得分
                        complete_score_act_1 = utils.get_oic_score(cas_sigmoid_fuse[0][b, :, c], start=start_act,
                                                                   end=end_act, point_annotations=point_positions)
                        complete_score_act_2 = utils.get_oic_score(cas_sigmoid_fuse[1][b, :, c], start=start_act,
                                                                   end=end_act, point_annotations=point_positions)

                        loss_score_act_batch += 1 - complete_score_act_1 * 0.5 - complete_score_act_2 * 0.5

                        act_count += 1

                    if sum(label_lst) > 1:
                        feature_lst_tensor = torch.stack(feature_lst, 0).clone()
                        feature_lst_tensor = feature_lst_tensor / torch.norm(feature_lst_tensor, dim=1, p=2).unsqueeze(1)
                        label_lst_tensor = torch.tensor(label_lst).to(torch.device('cuda:1')).float()

                        sim_matrix = torch.matmul(feature_lst_tensor, torch.transpose(feature_lst_tensor, 0, 1)) / self.tau
                        sim_matrix = torch.exp(sim_matrix).to(torch.device('cuda:1'))
                        sim_matrix = sim_matrix.clone().fill_diagonal_(0)
                        scores = (sim_matrix * label_lst_tensor.unsqueeze(1)).sum(dim=0) / sim_matrix.sum(dim=0)
                        loss_feat_batch = (-label_lst_tensor * torch.log(scores)).sum() / label_lst_tensor.sum()

                if act_count > 0:
                    loss_score_act += loss_score_act_batch / act_count
                    loss_feat += loss_feat_batch

            bkg_idx_diff = (1 - new_dense_anno[:, 1:]) - (1 - new_dense_anno[:, :-1])
            loss_score_bkg = 0
            for b in range(new_dense_anno.shape[0]):
                gt_classes = torch.nonzero(label[b]).squeeze(1)
                loss_score_bkg_batch = 0
                bkg_count = 0

                for c in gt_classes:
                    range_idx = torch.nonzero(bkg_idx_diff[b, :, c]).squeeze(1)
                    range_idx = range_idx.cpu().data.numpy().tolist()
                    if type(range_idx) is not list:
                        range_idx = [range_idx]
                    if len(range_idx) == 0:
                        continue
                    if bkg_idx_diff[b, range_idx[0], c] != 1:
                        range_idx = [-1] + range_idx
                    if bkg_idx_diff[b, range_idx[-1], c] != -1:
                        range_idx = range_idx + [bkg_idx_diff.shape[1] - 1]

                    for i in range(len(range_idx) // 2):
                        if range_idx[2 * i + 1] - range_idx[2 * i] < 1:
                            continue

                        start_bkg = range_idx[2 * i] + 1
                        end_bkg = range_idx[2 * i + 1]

                        # 获取当前类别的标注点位置（用于背景区域的显著性计算）
                        point_positions = []
                        for t in range(point_anno.shape[1]):
                            if point_anno[b, t, c] > 0:  # 如果该位置有标注点
                                point_positions.append(t)
                                
                        # 结合显著性计算背景区域的代表性得分
                        complete_score_bkg_1 = utils.get_oic_score(1 - cas_sigmoid_fuse[0][b, :, c], start=start_bkg,
                                                                   end=end_bkg, point_annotations=point_positions)
                        complete_score_bkg_2 = utils.get_oic_score(1 - cas_sigmoid_fuse[1][b, :, c], start=start_bkg,
                                                                   end=end_bkg, point_annotations=point_positions)

                        loss_score_bkg_batch += 1 - complete_score_bkg_1 * 0.5 - complete_score_bkg_2 * 0.5

                        bkg_count += 1

                if bkg_count > 0:
                    loss_score_bkg += loss_score_bkg_batch / bkg_count

            loss_score_act = loss_score_act / new_dense_anno.shape[0]
            loss_score_bkg = loss_score_bkg / new_dense_anno.shape[0]

            loss_feat = loss_feat / new_dense_anno.shape[0]

        loss_score = (loss_score_act + loss_score_bkg) ** 2

        # pseudo_instance_loss = self.cls_criterion(cas_sigmoid_fuse[0], pseudo_instance_label, uncertainty)
        pseudo_instance_loss = self.ce_criterion(cas_sigmoid_fuse[0], pseudo_instance_label) * 0.5 + self.ce_criterion(cas_sigmoid_fuse[1], pseudo_instance_label) * 0.5
        pseudo_instance_loss = pseudo_instance_loss.mean()

        # loss_total = self.lambdas[0] * loss_vid + self.lambdas[1] * loss_frame + self.lambdas[2] * loss_frame_bkg
        loss_total = self.lambdas[0] * loss_vid + self.lambdas[1] * loss_frame + self.lambdas[2] * loss_frame_bkg + \
                     self.lambdas[3] * loss_score + self.lambdas[4] * loss_feat + pseudo_instance_loss

        loss["loss_vid"] = loss_vid
        loss["loss_frame"] = loss_frame
        loss["loss_frame_bkg"] = loss_frame_bkg
        loss["loss_score_act"] = loss_score_act
        loss["loss_score_bkg"] = loss_score_bkg
        loss["loss_score"] = loss_score
        loss["loss_feat"] = loss_feat
        loss["loss_total"] = loss_total

        return loss_total, loss


def train(net, config, batch, optimizer, criterion, logger, step):
    net.train()
    optimizer.zero_grad()

    _, _data, _label, _point_anno, stored_info, _, _, proposal_bbox, proposal_count_by_video, pseudo_instance_label, dynamic_segment_weights_cumsum = batch

    _data = _data.to(torch.device('cuda:1'), non_blocking=True)
    _label = _label.to(torch.device('cuda:1'), non_blocking=True)
    _point_anno = _point_anno.to(torch.device('cuda:1'), non_blocking=True)
    proposal_bbox = proposal_bbox.to(torch.device('cuda:1'), non_blocking=True)
    pseudo_instance_label = pseudo_instance_label.to(torch.device('cuda:1'), non_blocking=True)

    # Forward pass
    vid_score, cas_sigmoid_fuse, features, cls_att, uncertainty = net(_data, vid_labels=_label, proposal_bbox=proposal_bbox, proposal_count_by_video=proposal_count_by_video)
        
    # Loss calculation
    cost, loss = criterion(vid_score, cas_sigmoid_fuse, features, cls_att, stored_info, _label, _point_anno, step, pseudo_instance_label=pseudo_instance_label, uncertainty=uncertainty)

    # Backward pass and optimization
    cost.backward()
    optimizer.step()

    # Logging
    for key in loss.keys():
        if loss[key] > 0:
            logger.log_value("loss/" + key, loss[key].detach().cpu().item(), step)
        else:
            logger.log_value("loss/" + key, loss[key], step)
