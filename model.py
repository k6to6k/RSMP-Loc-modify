import torch
import torch.nn as nn
import torch.nn.init as torch_init
import torch.nn.functional as F
import numpy as np
from typing import Optional
from torch import Tensor
from torch.types import _float


# def Nan_To_Num(input: Tensor, nan: Optional[_float]=None, posinf: Optional[_float]=None, neginf: Optional[_float]=None, *, out: Optional[Tensor]=None) -> Tensor: ...

class MHSA_Intra(nn.Module):
    """
    compute intra-segment attention
    """

    def __init__(self, dim_in, heads, pos_enc_type='relative', use_pos=True):
        super(MHSA_Intra, self).__init__()

        self.dim_in = dim_in
        self.dim_inner = self.dim_in
        self.heads = heads
        self.dim_head = self.dim_inner // self.heads

        self.scale = self.dim_head ** -0.5

        self.conv_query = nn.Conv1d(
            self.dim_in, self.dim_inner, kernel_size=1, stride=1, padding=0
        )
        self.conv_key = nn.Conv1d(
            self.dim_in, self.dim_inner, kernel_size=1, stride=1, padding=0
        )
        self.conv_value = nn.Conv1d(
            self.dim_in, self.dim_inner, kernel_size=1, stride=1, padding=0
        )
        self.conv_out = nn.Conv1d(
            self.dim_inner, self.dim_in, kernel_size=1, stride=1, padding=0
        )
        self.bn = nn.BatchNorm1d(
            num_features=self.dim_in, eps=1e-5, momentum=0.1
        )
        self.bn.weight.data.zero_()
        self.bn.bias.data.zero_()

    def forward(self, input, intra_attn_mask):
        B, C, T = input.shape
        query = self.conv_query(input).view(B, self.heads, self.dim_head, T).permute(0, 1, 3,
                                                                                     2).contiguous()  # (B, h, T, dim_head)
        key = self.conv_key(input).view(B, self.heads, self.dim_head, T)  # (B, h, dim_head, T)
        value = self.conv_value(input).view(B, self.heads, self.dim_head, T).permute(0, 1, 3,
                                                                                     2).contiguous()  # (B, h, T, dim_head)

        query *= self.scale
        sim = torch.matmul(query, key)  # (B, h, T, T)
        intra_attn_mask = intra_attn_mask.view(B, 1, T, T)
        sim.masked_fill_(intra_attn_mask == 0, -np.inf)
        attn = F.softmax(sim, dim=-1)  # (B, h, T, T)
        attn = torch.nan_to_num(attn, nan=0.0)
        output = torch.matmul(attn, value)  # (B, h, T, dim_head)

        output = output.permute(0, 1, 3, 2).contiguous().view(B, C, T)  # (B, C, T)
        output = input + self.bn(self.conv_out(output))
        return output


class MHSA_Inter(nn.Module):
    """
    compute inter-segment attention
    """

    def __init__(self, dim_in, heads, pos_enc_type='relative', use_pos=True):
        super(MHSA_Inter, self).__init__()

        self.dim_in = dim_in
        self.dim_inner = self.dim_in
        self.heads = heads
        self.dim_head = self.dim_inner // self.heads

        self.scale = self.dim_head ** -0.5

        self.conv_query = nn.Conv1d(
            self.dim_in, self.dim_inner, kernel_size=1, stride=1, padding=0
        )
        self.conv_key = nn.Conv1d(
            self.dim_in, self.dim_inner, kernel_size=1, stride=1, padding=0
        )
        self.conv_value = nn.Conv1d(
            self.dim_in, self.dim_inner, kernel_size=1, stride=1, padding=0
        )
        self.conv_out = nn.Conv1d(
            self.dim_inner, self.dim_in, kernel_size=1, stride=1, padding=0
        )
        self.bn = nn.BatchNorm1d(
            num_features=self.dim_in, eps=1e-5, momentum=0.1
        )
        self.bn.weight.data.zero_()
        self.bn.bias.data.zero_()

    def forward(self, input, inter_attn_mask, proposal_project_mask):
        B, C, T = input.shape
        B, T, K = proposal_project_mask.shape
        assert K == inter_attn_mask.shape[2], "segment num not equal"

        proposal_project_mask_norm = proposal_project_mask / torch.sum(proposal_project_mask, dim=1,
                                                                       keepdim=True).clamp(min=1e-5)  # (B, T, K)
        segment_feature = torch.matmul(input, proposal_project_mask_norm)  # (B, C, K)
        query_global = self.conv_query(segment_feature).view(B, self.heads, self.dim_head, K).permute(0, 1, 3,
                                                                                                      2).contiguous()  # (B, h, K, dim_head)
        key_global = self.conv_key(segment_feature).view(B, self.heads, self.dim_head, K)  # (B, h, dim_head, K)
        value_global = self.conv_value(segment_feature).view(B, self.heads, self.dim_head, K).permute(0, 1, 3,
                                                                                                      2).contiguous()  # (B, h, K, dim_head)

        query_global *= self.scale
        sim_global = torch.matmul(query_global, key_global)  # (B, h, K, K)
        inter_attn_mask = inter_attn_mask.view(B, 1, K, K)
        sim_global.masked_fill_(inter_attn_mask == 0, -np.inf)
        attn_global = F.softmax(sim_global, dim=-1)  # (B, h, K, K)
        attn_global = torch.nan_to_num(attn_global, nan=0.0)
        output_global = torch.matmul(attn_global, value_global)  # (B, h, K, dim_head)

        output_global = output_global.permute(0, 1, 3, 2).contiguous().view(B, C, K)  # (B, C, K)
        proposal_project_mask_reverse_norm = proposal_project_mask.permute(0, 2, 1) / torch.sum(
            proposal_project_mask.permute(0, 2, 1), dim=1, keepdim=True).clamp(min=1e-5)  # (B, K, T)
        output_global = torch.matmul(output_global, proposal_project_mask_reverse_norm)  # (B, C, T)
        output_global = input + self.bn(self.conv_out(output_global))  # (B, C, T)
        return output_global


class Cls_Module(nn.Module):
    def __init__(self, len_feature, num_classes):
        super(Cls_Module, self).__init__()
        self.len_feature = len_feature
        self.conv_1 = nn.Sequential(
            nn.Conv1d(in_channels=self.len_feature, out_channels=2048, kernel_size=3,
                      stride=1, padding=1),
            nn.ReLU()
        )
        self.classifier = nn.Sequential(
            nn.Conv1d(in_channels=2048, out_channels=num_classes + 1, kernel_size=1,
                      stride=1, padding=0, bias=False)
        )
        self.drop_out = nn.Dropout(p=0.7)

    def forward(self, x):
        # x: (B, T, F)
        out = x.permute(0, 2, 1)
        # out: (B, F, T)

        # out = self.conv_1(out)
        # feat = out.permute(0, 2, 1)

        out = self.drop_out(out)
        cas = self.classifier(out)

        cas = cas.permute(0, 2, 1)

        # out: (B, T, C + 1)
        return None, cas


class Model(nn.Module):
    def __init__(self, len_feature, num_classes, r_act):
        super(Model, self).__init__()
        self.len_feature = len_feature
        self.num_classes = num_classes
        self.r_act = r_act

        self.cls_module = Cls_Module(len_feature, num_classes)

        self.mu = nn.Parameter(torch.randn(8, 2048))
        torch_init.xavier_uniform_(self.mu)

        self.sigmoid = nn.Sigmoid()

        self.conv_1 = nn.Sequential(
            nn.Conv1d(in_channels=self.len_feature, out_channels=2048, kernel_size=3,
                      stride=1, padding=1),
            nn.ReLU()
        )

        self.att_branch = nn.Conv1d(in_channels=self.len_feature, out_channels=2, kernel_size=1, padding=0)

        self.dropout = nn.Dropout(p=0.7)
        self.ac_center = nn.Parameter(torch.randn(self.num_classes + 1, 2048))
        torch_init.xavier_uniform_(self.ac_center)
        self.fg_center = nn.Parameter(-1.0 * self.ac_center[-1, ...][None, ...])

    def EM(self, mu, x):
        # propagation -> make mu as video-specific mu
        norm_x = calculate_l1_norm(x.float())
        for _ in range(2):
            norm_mu = calculate_l1_norm(mu.float())
            latent_z = F.softmax(torch.einsum('nkd,ntd->nkt', [norm_mu, norm_x]) * 5.0, 1)
            norm_latent_z = latent_z / (latent_z.sum(dim=-1, keepdim=True) + 1e-9)
            mu = torch.einsum('nkt,ntd->nkd', [norm_latent_z, x.float()])
        return mu

    def PredictionModule(self, x):
        # normalization
        norms_x = calculate_l1_norm(x)
        norms_ac = calculate_l1_norm(self.ac_center)
        norms_fg = calculate_l1_norm(self.fg_center)

        # generate class scores
        frm_scrs = torch.einsum('ntd,cd->ntc', [norms_x, norms_ac]) * 20.
        frm_fb_scrs = torch.einsum('ntd,kd->ntk', [norms_x, norms_fg]).squeeze(-1) * 20.

        # generate attention
        class_agno_att = self.sigmoid(frm_fb_scrs)
        class_wise_att = self.sigmoid(frm_scrs)
        class_agno_norm_att = class_agno_att / (torch.sum(class_agno_att, dim=1, keepdim=True) + 1e-5)
        class_wise_norm_att = class_wise_att / (torch.sum(class_wise_att, dim=1, keepdim=True) + 1e-5)

        ca_vid_feat = torch.einsum('ntd,nt->nd', [x, class_agno_norm_att])
        cw_vid_feat = torch.einsum('ntd,ntc->ncd', [x, class_wise_norm_att])

        # normalization
        norms_ca_vid_feat = calculate_l1_norm(ca_vid_feat)
        norms_cw_vid_feat = calculate_l1_norm(cw_vid_feat)

        # classification
        frm_scr = torch.einsum('ntd,cd->ntc', [norms_x, norms_ac]) * 20.
        ca_vid_scr = torch.einsum('nd,cd->nc', [norms_ca_vid_feat, norms_ac]) * 20.
        cw_vid_scr = torch.einsum('ncd,cd->nc', [norms_cw_vid_feat, norms_ac]) * 20.

        # prediction
        ca_vid_pred = F.softmax(ca_vid_scr, -1)
        cw_vid_pred = F.softmax(cw_vid_scr, -1)

        return ca_vid_pred, cw_vid_pred, class_agno_att, frm_scr

    # 原来的前往传播
    def forward(self, x, vid_labels=None, proposal_bbox=None, proposal_count_by_video=None):
        device = x.device
        batch_size, num_segments = x.shape[0], x.shape[1]
        n_size = x.shape[0]
        k_act = num_segments // self.r_act
        b_act = num_segments // 3

        out = x.permute(0, 2, 1)
        # out: (B, F, T)
        out = out.float()
        features = self.conv_1(out)

        features = features.permute(0, 2, 1)

        mu = self.mu[None, ...].repeat(n_size, 1, 1)
        mu = self.EM(mu, x)
        reallocated_feat = random_walk(features, mu, 0.5)

        _, cas = self.cls_module(features)
        _, r_cas = self.cls_module(reallocated_feat)

        cas_sigmoid = self.sigmoid(cas)
        r_cas_sigmoid = self.sigmoid(r_cas)

        cas_sigmoid_fuse = cas_sigmoid[:, :, :-1] * (1 - cas_sigmoid[:, :, -1].unsqueeze(2))
        cas_sigmoid_fuse = torch.cat((cas_sigmoid_fuse, cas_sigmoid[:, :, -1].unsqueeze(2)), dim=2)

        r_cas_sigmoid_fuse = r_cas_sigmoid[:, :, :-1] * (1 - r_cas_sigmoid[:, :, -1].unsqueeze(2))
        r_cas_sigmoid_fuse = torch.cat((r_cas_sigmoid_fuse, r_cas_sigmoid[:, :, -1].unsqueeze(2)), dim=2)

        value, _ = cas_sigmoid.sort(descending=True, dim=1)
        r_value, _ = r_cas_sigmoid.sort(descending=True, dim=1)

        topk_scores = value[:, :k_act, :-1]

        r_topk_scores = r_value[:, :k_act, :-1]

        if vid_labels is None:
            vid_score = torch.mean(topk_scores, dim=1)
            r_vid_score = torch.mean(r_topk_scores, dim=1)
        else:
            vid_score = (torch.mean(topk_scores, dim=1) * vid_labels) + (
                    torch.mean(cas_sigmoid[:, :, :-1], dim=1) * (1 - vid_labels))
            r_vid_score = (torch.mean(r_topk_scores, dim=1) * vid_labels) + (
                    torch.mean(r_cas_sigmoid[:, :, :-1], dim=1) * (1 - vid_labels))

        # original feature branch
        o_vid_ca_pred, o_vid_cw_pred, o_att, o_frm_pred = self.PredictionModule(features)
        # reallocated feature branch
        m_vid_ca_pred, m_vid_cw_pred, m_att, m_frm_pred = self.PredictionModule(reallocated_feat)

        return [vid_score, r_vid_score], [cas_sigmoid_fuse, r_cas_sigmoid_fuse], features, [o_vid_ca_pred, m_vid_ca_pred, o_vid_cw_pred, m_vid_cw_pred, o_att, m_att, o_frm_pred, m_frm_pred], None


class Model_GAI(nn.Module):
    def __init__(self, len_feature, num_classes, r_act):
        super(Model_GAI, self).__init__()
        self.len_feature = len_feature
        self.num_classes = num_classes
        self.max_segments_num = 200
        self.r_act = r_act

        self.cls_module = Cls_Module(len_feature, num_classes)

        self.mu = nn.Parameter(torch.randn(8, 2048))
        torch_init.xavier_uniform_(self.mu)

        self.sigmoid = nn.Sigmoid()

        self.conv_1 = nn.Sequential(
            nn.Conv1d(in_channels=self.len_feature, out_channels=2048, kernel_size=3,
                      stride=1, padding=1),
            nn.ReLU()
        )

        self.att_branch = nn.Conv1d(in_channels=self.len_feature, out_channels=2, kernel_size=1, padding=0)

        self.dropout = nn.Dropout(p=0.7)

        self.MHSA_Intra = MHSA_Intra(dim_in=self.len_feature, heads=8)
        self.MHSA_Inter = MHSA_Inter(dim_in=self.len_feature, heads=8)
        self.uncertainty_branch = nn.Conv1d(in_channels=self.len_feature, out_channels=1, kernel_size=1, padding=0,
                                            bias=False)
        self.uncertainty_branch.weight.data.normal_(0.0, 0.0001)

        self.ac_center = nn.Parameter(torch.randn(self.num_classes + 1, 2048))
        torch_init.xavier_uniform_(self.ac_center)
        self.fg_center = nn.Parameter(-1.0 * self.ac_center[-1, ...][None, ...])

    def EM(self, mu, x):
        # propagation -> make mu as video-specific mu
        x = x.float()
        norm_x = calculate_l1_norm(x)
        for _ in range(2):
            norm_mu = calculate_l1_norm(mu)
            latent_z = F.softmax(torch.einsum('nkd,ntd->nkt', [norm_mu, norm_x]) * 5.0, 1)
            norm_latent_z = latent_z / (latent_z.sum(dim=-1, keepdim=True) + 1e-9)
            mu = torch.einsum('nkt,ntd->nkd', [norm_latent_z, x])
        return mu

    def PredictionModule(self, x):
        # normalization
        norms_x = calculate_l1_norm(x)
        norms_ac = calculate_l1_norm(self.ac_center)
        norms_fg = calculate_l1_norm(self.fg_center)

        # generate class scores
        frm_scrs = torch.einsum('ntd,cd->ntc', [norms_x, norms_ac]) * 20.
        frm_fb_scrs = torch.einsum('ntd,kd->ntk', [norms_x, norms_fg]).squeeze(-1) * 20.

        # generate attention
        class_agno_att = self.sigmoid(frm_fb_scrs)
        class_wise_att = self.sigmoid(frm_scrs)
        class_agno_norm_att = class_agno_att / (torch.sum(class_agno_att, dim=1, keepdim=True) + 1e-5)
        class_wise_norm_att = class_wise_att / (torch.sum(class_wise_att, dim=1, keepdim=True) + 1e-5)

        ca_vid_feat = torch.einsum('ntd,nt->nd', [x, class_agno_norm_att])
        cw_vid_feat = torch.einsum('ntd,ntc->ncd', [x, class_wise_norm_att])

        # normalization
        norms_ca_vid_feat = calculate_l1_norm(ca_vid_feat)
        norms_cw_vid_feat = calculate_l1_norm(cw_vid_feat)

        # classification
        frm_scr = torch.einsum('ntd,cd->ntc', [norms_x, norms_ac]) * 20.
        ca_vid_scr = torch.einsum('nd,cd->nc', [norms_ca_vid_feat, norms_ac]) * 20.
        cw_vid_scr = torch.einsum('ncd,cd->nc', [norms_cw_vid_feat, norms_ac]) * 20.

        # prediction
        ca_vid_pred = F.softmax(ca_vid_scr, -1)
        cw_vid_pred = F.softmax(cw_vid_scr, -1)

        return ca_vid_pred, cw_vid_pred, class_agno_att, frm_scr

    # 原来的前往传播
    def forward(self, x, vid_labels=None, proposal_bbox=None, proposal_count_by_video=None):
        device = x.device
        batch_size, num_segments = x.shape[0], x.shape[1]
        n_size = x.shape[0]
        k_act = max(num_segments // self.r_act, 1)
        b_act = max(num_segments // 3, 1)

        out = x.permute(0, 2, 1)
        # out: (B, F, T)
        x = x.float()
        out = out.float()
        features = self.conv_1(out)

        features = features.permute(0, 2, 1)

        mu = self.mu[None, ...].repeat(n_size, 1, 1)
        mu = self.EM(mu, x)
        reallocated_feat = random_walk(features, mu, 0.5)

        _, cas = self.cls_module(self.dropout(features))
        _, r_cas = self.cls_module(self.dropout(reallocated_feat))

        cas_sigmoid = self.sigmoid(cas)
        r_cas_sigmoid = self.sigmoid(r_cas)

        cas_sigmoid_fuse = cas_sigmoid[:, :, :-1] * (1 - cas_sigmoid[:, :, -1].unsqueeze(2))
        cas_sigmoid_fuse = torch.cat((cas_sigmoid_fuse, cas_sigmoid[:, :, -1].unsqueeze(2)), dim=2)

        r_cas_sigmoid_fuse = r_cas_sigmoid[:, :, :-1] * (1 - r_cas_sigmoid[:, :, -1].unsqueeze(2))
        r_cas_sigmoid_fuse = torch.cat((r_cas_sigmoid_fuse, r_cas_sigmoid[:, :, -1].unsqueeze(2)), dim=2)

        value, _ = cas_sigmoid.sort(descending=True, dim=1)
        r_value, _ = r_cas_sigmoid.sort(descending=True, dim=1)

        topk_scores = value[:, :k_act, :-1]

        r_topk_scores = r_value[:, :k_act, :-1]


        if vid_labels is None:
            vid_score = torch.mean(topk_scores, dim=1)
            r_vid_score = torch.mean(r_topk_scores, dim=1)
        else:
            vid_score = (torch.mean(topk_scores, dim=1) * vid_labels) + (
                    torch.mean(cas_sigmoid[:, :, :-1], dim=1) * (1 - vid_labels))
            r_vid_score = (torch.mean(r_topk_scores, dim=1) * vid_labels) + (
                    torch.mean(r_cas_sigmoid[:, :, :-1], dim=1) * (1 - vid_labels))

        # original feature branch
        o_vid_ca_pred, o_vid_cw_pred, o_att, o_frm_pred = self.PredictionModule(features)
        # reallocated feature branch
        m_vid_ca_pred, m_vid_cw_pred, m_att, m_frm_pred = self.PredictionModule(reallocated_feat)

        # return [vid_score, r_vid_score], [cas_sigmoid_fuse, r_cas_sigmoid_fuse], features, uncertainty.permute(0, 2, 1)
        return [vid_score, r_vid_score], [cas_sigmoid_fuse, r_cas_sigmoid_fuse], features, [o_vid_ca_pred, m_vid_ca_pred, o_vid_cw_pred, m_vid_cw_pred, o_att, m_att, o_frm_pred, m_frm_pred], None

def calculate_l1_norm(f):
    f_norm = torch.norm(f, p=2, dim=-1, keepdim=True)
    f = f / (f_norm + 1e-9)
    return f.float()


def random_walk(x, y, w):
    x_norm = calculate_l1_norm(x)
    y_norm = calculate_l1_norm(y)
    x_norm = x_norm.float()
    y_norm = y_norm.float()

    '''
    修改
    '''
    # eye_x = torch.eye(x.size(1)).float().to(x.device)
    N, T = x.size(0), x.size(1)
    eye_x = torch.eye(T).float().to(x.device)

    latent_z = F.softmax(torch.einsum('nkd,ntd->nkt', [y_norm, x_norm]) * 5.0, 1)
    norm_latent_z = latent_z / (latent_z.sum(dim=-1, keepdim=True) + 1e-9)
    affinity_mat = torch.einsum('nkt,nkd->ntd', [latent_z, norm_latent_z])
    try:
        '''
        修改
        '''
        # mat_inv_x, _ = torch.solve(eye_x, eye_x - (w ** 2) * affinity_mat)
        # 创建一个 (N, T, T) 的单位矩阵批次，以保持维度一致
        mat_inv_x = eye_x.unsqueeze(0).repeat(N, 1, 1)
    except:
        mat_inv_x = eye_x
    y2x_sum_x = w * torch.einsum('nkt,nkd->ntd', [latent_z, y]) + x
    refined_x = (1 - w) * torch.einsum('ntk,nkd->ntd', [mat_inv_x, y2x_sum_x])

    return refined_x
