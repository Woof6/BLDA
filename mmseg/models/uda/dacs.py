# ---------------------------------------------------------------
# Copyright (c) 2021-2022 ETH Zurich, Lukas Hoyer. All rights reserved.
# Licensed under the Apache License, Version 2.0
# ---------------------------------------------------------------

# The ema model update and the domain-mixing are based on:
# https://github.com/vikolss/DACS
# Copyright (c) 2020 vikolss. Licensed under the MIT License.
# A copy of the license is available at resources/license_dacs

import math
import os
import random
from copy import deepcopy

import mmcv
import numpy as np
import torch
from matplotlib import pyplot as plt
from timm.models.layers import DropPath
from torch.nn.modules.dropout import _DropoutNd

from mmseg.core import add_prefix
from mmseg.models import UDA, build_segmentor
from mmseg.models.uda.uda_decorator import UDADecorator, get_module
from mmseg.models.utils.dacs_transforms import (denorm, get_class_masks,
                                                get_mean_std, strong_transform)
from mmseg.models.utils.visualization import subplotimg
from mmseg.utils.utils import downscale_label_ratio
import logging

def _params_equal(ema_model, model):
    for ema_param, param in zip(ema_model.named_parameters(),
                                model.named_parameters()):
        if not torch.equal(ema_param[1].data, param[1].data):
            # print("Difference in", ema_param[0])
            return False
    return True


def calc_grad_magnitude(grads, norm_type=2.0):
    norm_type = float(norm_type)
    if norm_type == math.inf:
        norm = max(p.abs().max() for p in grads)
    else:
        norm = torch.norm(
            torch.stack([torch.norm(p, norm_type) for p in grads]), norm_type)

    return norm



def gaussian(x, mean, covariance):
    exponent = -0.5 * ((x - mean) ** 2) / covariance
    coefficient = 1 / np.sqrt(2 * np.pi * covariance)
    return coefficient * np.exp(exponent)

def gmm_curve(x, weights, means, covariances):
    curve = np.zeros_like(x)
    for weight, mean, covariance in zip(weights, means, covariances):
        curve += weight * gaussian(x, mean, covariance)
    return curve


@UDA.register_module()
class DACS(UDADecorator):

    def __init__(self, **cfg):
        super(DACS, self).__init__(**cfg)
        self.local_iter = 0
        self.max_iters = cfg['max_iters']
        self.alpha = cfg['alpha']
        self.pseudo_threshold = cfg['pseudo_threshold']
        self.psweight_ignore_top = cfg['pseudo_weight_ignore_top']
        self.psweight_ignore_bottom = cfg['pseudo_weight_ignore_bottom']
        self.fdist_lambda = cfg['imnet_feature_dist_lambda']
        self.fdist_classes = cfg['imnet_feature_dist_classes']
        self.fdist_scale_min_ratio = cfg['imnet_feature_dist_scale_min_ratio']
        self.enable_fdist = self.fdist_lambda > 0
        self.mix = cfg['mix']
        self.blur = cfg['blur']
        self.color_jitter_s = cfg['color_jitter_strength']
        self.color_jitter_p = cfg['color_jitter_probability']
        self.debug_img_interval = cfg['debug_img_interval']
        self.print_grad_magnitude = cfg['print_grad_magnitude']
        assert self.mix == 'class'

        self.debug_fdist_mask = None
        self.debug_gt_rescale = None

        self.class_probs = {}
        ema_cfg = deepcopy(cfg['model'])
        self.ema_model = build_segmentor(ema_cfg)

        if self.enable_fdist:
            self.imnet_model = build_segmentor(deepcopy(cfg['model']))
        else:
            self.imnet_model = None

        # 打印weight
        log_directory = os.path.join(self.train_cfg['work_dir'],
                                   'class_weight.txt')
        logging.basicConfig(filename=log_directory, level=logging.INFO)
        self.logger =  logging.getLogger()
        self.counts = torch.zeros(19)
        
    def get_ema_model(self):
        return get_module(self.ema_model)

    def get_imnet_model(self):
        return get_module(self.imnet_model)

    def _init_ema_weights(self):
        for param in self.get_ema_model().parameters():
            param.detach_()
        mp = list(self.get_model().parameters())
        mcp = list(self.get_ema_model().parameters())
        for i in range(0, len(mp)):
            if not mcp[i].data.shape:  # scalar tensor
                mcp[i].data = mp[i].data.clone()
            else:
                mcp[i].data[:] = mp[i].data[:].clone()

    def _update_ema(self, iter):
        alpha_teacher = min(1 - 1 / (iter + 1), self.alpha)
        for ema_param, param in zip(self.get_ema_model().parameters(),
                                    self.get_model().parameters()):
            if not param.data.shape:  # scalar tensor
                ema_param.data = \
                    alpha_teacher * ema_param.data + \
                    (1 - alpha_teacher) * param.data
            else:
                ema_param.data[:] = \
                    alpha_teacher * ema_param[:].data[:] + \
                    (1 - alpha_teacher) * param[:].data[:]

    def train_step(self, data_batch, optimizer, **kwargs):
        """The iteration step during training.

        This method defines an iteration step during training, except for the
        back propagation and optimizer updating, which are done in an optimizer
        hook. Note that in some complicated cases or models, the whole process
        including back propagation and optimizer updating is also defined in
        this method, such as GAN.

        Args:
            data (dict): The output of dataloader.
            optimizer (:obj:`torch.optim.Optimizer` | dict): The optimizer of
                runner is passed to ``train_step()``. This argument is unused
                and reserved.

        Returns:
            dict: It should contain at least 3 keys: ``loss``, ``log_vars``,
                ``num_samples``.
                ``loss`` is a tensor for back propagation, which can be a
                weighted sum of multiple losses.
                ``log_vars`` contains all the variables to be sent to the
                logger.
                ``num_samples`` indicates the batch size (when the model is
                DDP, it means the batch size on each GPU), which is used for
                averaging the logs.
        """

        optimizer.zero_grad()
        log_vars = self(**data_batch)
        optimizer.step()

        log_vars.pop('loss', None)  # remove the unnecessary 'loss'
        outputs = dict(
            log_vars=log_vars, num_samples=len(data_batch['img_metas']))
        return outputs

    def masked_feat_dist(self, f1, f2, mask=None):
        feat_diff = f1 - f2
        # mmcv.print_log(f'fdiff: {feat_diff.shape}', 'mmseg')
        pw_feat_dist = torch.norm(feat_diff, dim=1, p=2)
        # mmcv.print_log(f'pw_fdist: {pw_feat_dist.shape}', 'mmseg')
        if mask is not None:
            # mmcv.print_log(f'fd mask: {mask.shape}', 'mmseg')
            pw_feat_dist = pw_feat_dist[mask.squeeze(1)]
            # mmcv.print_log(f'fd masked: {pw_feat_dist.shape}', 'mmseg')
        return torch.mean(pw_feat_dist)

    def calc_feat_dist(self, img, gt, feat=None):
        assert self.enable_fdist
        with torch.no_grad():
            self.get_imnet_model().eval()
            feat_imnet = self.get_imnet_model().extract_feat(img)
            feat_imnet = [f.detach() for f in feat_imnet]
        lay = -1
        if self.fdist_classes is not None:
            fdclasses = torch.tensor(self.fdist_classes, device=gt.device)
            scale_factor = gt.shape[-1] // feat[lay].shape[-1]
            gt_rescaled = downscale_label_ratio(gt, scale_factor,
                                                self.fdist_scale_min_ratio,
                                                self.num_classes,
                                                255).long().detach()
            fdist_mask = torch.any(gt_rescaled[..., None] == fdclasses, -1)
            feat_dist = self.masked_feat_dist(feat[lay], feat_imnet[lay],
                                              fdist_mask)
            self.debug_fdist_mask = fdist_mask
            self.debug_gt_rescale = gt_rescaled
        else:
            feat_dist = self.masked_feat_dist(feat[lay], feat_imnet[lay])
        feat_dist = self.fdist_lambda * feat_dist
        feat_loss, feat_log = self._parse_losses(
            {'loss_imnet_feat_dist': feat_dist})
        feat_log.pop('loss', None)
        return feat_loss, feat_log

    def forward_train(self, img, img_metas, gt_semantic_seg, target_img,
                      target_img_metas):
        """Forward function for training.

        Args:
            img (Tensor): Input images.
            img_metas (list[dict]): List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:Collect`.
            gt_semantic_seg (Tensor): Semantic segmentation masks
                used if the architecture supports semantic segmentation task.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        log_vars = {}
        batch_size = img.shape[0]
        dev = img.device

        # Init/update ema model
        if self.local_iter == 0:
            self._init_ema_weights()
            # assert _params_equal(self.get_ema_model(), self.get_model())

        if self.local_iter > 0:
            self._update_ema(self.local_iter)
            # assert not _params_equal(self.get_ema_model(), self.get_model())
            # assert self.get_ema_model().training

        means, stds = get_mean_std(img_metas, dev)
        strong_parameters = {
            'mix': None,
            'color_jitter': random.uniform(0, 1),
            'color_jitter_s': self.color_jitter_s,
            'color_jitter_p': self.color_jitter_p,
            'blur': random.uniform(0, 1) if self.blur else 0,
            'mean': means[0].unsqueeze(0),  # assume same normalization
            'std': stds[0].unsqueeze(0)
        }

        # Train on source images
        clean_losses = self.get_model().forward_train(
            img, img_metas, gt_semantic_seg, return_feat=True)
        
        
        if self.local_iter%1000==0:
            for key in clean_losses.keys():
                if 'class_weight' in key :
                    class_weight = clean_losses[key ]
                    self.logger.info(f'{key} {self.local_iter }: {class_weight}')
                
        src_feat = clean_losses.pop('features')
        clean_loss, clean_log_vars = self._parse_losses(clean_losses)
        log_vars.update(clean_log_vars)
        clean_loss.backward(retain_graph=self.enable_fdist)
        if self.print_grad_magnitude:
            params = self.get_model().backbone.parameters()
            seg_grads = [
                p.grad.detach().clone() for p in params if p.grad is not None
            ]
            grad_mag = calc_grad_magnitude(seg_grads)
            mmcv.print_log(f'Seg. Grad.: {grad_mag}', 'mmseg')

        # ImageNet feature distance
        if self.enable_fdist:
            feat_loss, feat_log = self.calc_feat_dist(img, gt_semantic_seg,
                                                      src_feat)
            feat_loss.backward()
            log_vars.update(add_prefix(feat_log, 'src'))
            if self.print_grad_magnitude:
                params = self.get_model().backbone.parameters()
                fd_grads = [
                    p.grad.detach() for p in params if p.grad is not None
                ]
                fd_grads = [g2 - g1 for g1, g2 in zip(seg_grads, fd_grads)]
                grad_mag = calc_grad_magnitude(fd_grads)
                mmcv.print_log(f'Fdist Grad.: {grad_mag}', 'mmseg')

        # Generate pseudo-label
        for m in self.get_ema_model().modules():
            if isinstance(m, _DropoutNd):
                m.training = False
            if isinstance(m, DropPath):
                m.training = False
        ema_logits = self.get_ema_model().encode_decode(
            target_img, target_img_metas)

        if isinstance(ema_logits,list):
            ema_logits = ema_logits[0]
          
        ema_softmax = torch.softmax(ema_logits.detach(), dim=1)
        pseudo_prob, pseudo_label = torch.max(ema_softmax, dim=1)
        
        self.counts = self.counts.to(pseudo_label.device)
        # 统计伪标签
        for i in range(19):
            self.counts[i]+= torch.sum(pseudo_label==i)
        if self.local_iter%1000==0:
            self.logger.info(f'label_count {self.local_iter }: {self.counts}')    
            
        ps_large_p = pseudo_prob.ge(self.pseudo_threshold).long() == 1
        ps_size = np.size(np.array(pseudo_label.cpu()))
        pseudo_weight = torch.sum(ps_large_p).item() / ps_size
        pseudo_weight = pseudo_weight * torch.ones(
            pseudo_prob.shape, device=dev)

        if self.psweight_ignore_top > 0:
            # Don't trust pseudo-labels in regions with potential
            # rectification artifacts. This can lead to a pseudo-label
            # drift from sky towards building or traffic light.
            pseudo_weight[:, :self.psweight_ignore_top, :] = 0
        if self.psweight_ignore_bottom > 0:
            pseudo_weight[:, -self.psweight_ignore_bottom:, :] = 0
        gt_pixel_weight = torch.ones((pseudo_weight.shape), device=dev)

        # Apply mixing
        mixed_img, mixed_lbl = [None] * batch_size, [None] * batch_size
        mix_masks = get_class_masks(gt_semantic_seg)

        for i in range(batch_size):
            strong_parameters['mix'] = mix_masks[i]
            mixed_img[i], mixed_lbl[i] = strong_transform(
                strong_parameters,
                data=torch.stack((img[i], target_img[i])),
                target=torch.stack((gt_semantic_seg[i][0], pseudo_label[i])))
            _, pseudo_weight[i] = strong_transform(
                strong_parameters,
                target=torch.stack((gt_pixel_weight[i], pseudo_weight[i])))
        mixed_img = torch.cat(mixed_img)
        mixed_lbl = torch.cat(mixed_lbl)

        # Train on mixed images
        mix_losses = self.get_model().forward_train(
            mixed_img, img_metas, mixed_lbl, pseudo_weight, return_feat=True)
        
        if self.local_iter%1000==0:
            for key in mix_losses.keys():
                if 'class_weight' in key :
                    class_weight = mix_losses[key ]
                    self.logger.info(f'{key} {self.local_iter }: {class_weight}')
                    
        mix_losses.pop('features')
        mix_losses = add_prefix(mix_losses, 'mix')
        mix_loss, mix_log_vars = self._parse_losses(mix_losses)
        log_vars.update(mix_log_vars)
        mix_loss.backward()

        if self.local_iter % self.debug_img_interval == 0:
            out_dir = os.path.join(self.train_cfg['work_dir'],
                                   'class_mix_debug')
            os.makedirs(out_dir, exist_ok=True)
            vis_img = torch.clamp(denorm(img, means, stds), 0, 1)
            vis_trg_img = torch.clamp(denorm(target_img, means, stds), 0, 1)
            vis_mixed_img = torch.clamp(denorm(mixed_img, means, stds), 0, 1)
            
            if 'decode.cdf_gt' in clean_losses.keys():
                vis_cdfgt = torch.clamp(clean_losses['decode.cdf_gt'], 0, 1)
                vis_reg = torch.clamp(clean_losses['decode.reg_score'], 0, 1)
                vis_mixed_cdfgt = torch.clamp(mix_losses['mix.decode.cdf_gt'], 0, 1)
                vis_mixed_reg = torch.clamp(mix_losses['mix.decode.reg_score'], 0, 1)
            else:
                vis_cdfgt = torch.zeros_like(vis_img)
                vis_reg = torch.zeros_like(vis_img)
                vis_mixed_cdfgt = torch.zeros_like(vis_img)
                vis_mixed_reg = torch.zeros_like(vis_img)
                
            for j in range(batch_size):
                rows, cols = 2, 7
                fig, axs = plt.subplots(
                    rows,
                    cols,
                    figsize=(3 * cols, 3 * rows),
                    gridspec_kw={
                        'hspace': 0.1,
                        'wspace': 0,
                        'top': 0.95,
                        'bottom': 0,
                        'right': 1,
                        'left': 0
                    },
                )
                
                subplotimg(axs[0][0], vis_img[j], 'Source Image')
                subplotimg(axs[1][0], vis_trg_img[j], 'Target Image')
                subplotimg(axs[0][5], vis_cdfgt[j][0], 'cdfgt', cmap='gray')
                subplotimg(axs[1][5], vis_reg[j][0], 'reg', cmap='gray')
                subplotimg(axs[0][6], vis_mixed_cdfgt[j][0], 'mixed_cdfgt', cmap='gray')
                subplotimg(axs[1][6], vis_mixed_reg[j][0], 'mixed_reg', cmap='gray')
                
                subplotimg(
                    axs[0][1],
                    gt_semantic_seg[j],
                    'Source Seg GT',
                    cmap='cityscapes')
                subplotimg(
                    axs[1][1],
                    pseudo_label[j],
                    'Target Seg (Pseudo) GT',
                    cmap='cityscapes')
                subplotimg(axs[0][2], vis_mixed_img[j], 'Mixed Image')
                subplotimg(
                    axs[1][2], mix_masks[j][0], 'Domain Mask', cmap='gray')
                # subplotimg(axs[0][3], pred_u_s[j], "Seg Pred",
                #            cmap="cityscapes")
                subplotimg(
                    axs[1][3], mixed_lbl[j], 'Seg Targ', cmap='cityscapes')
                subplotimg(
                    axs[0][3], pseudo_weight[j], 'Pseudo W.', vmin=0, vmax=1)
                if self.debug_fdist_mask is not None:
                    subplotimg(
                        axs[0][4],
                        self.debug_fdist_mask[j][0],
                        'FDist Mask',
                        cmap='gray')
                if self.debug_gt_rescale is not None:
                    subplotimg(
                        axs[1][4],
                        self.debug_gt_rescale[j],
                        'Scaled GT',
                        cmap='cityscapes')
                for ax in axs.flat:
                    ax.axis('off')
                plt.savefig(
                    os.path.join(out_dir,
                                 f'{(self.local_iter + 1):06d}_{j}.png'))
                plt.close()
                
        if  (self.local_iter) % 1000 == 0:
            out_dir = os.path.join(self.train_cfg['work_dir'],
                                'gmm_debug')
            os.makedirs(out_dir, exist_ok=True)
            for key in clean_losses.keys():
                if 'gmm_list' in key :
                    gmm_list = clean_losses[key ]
                    gmm_list_weights = gmm_list[0].cpu().numpy()
                    gmm_list_means = gmm_list[1].cpu().numpy()
                    gmm_list_covariances = gmm_list[2].cpu().numpy()
                    np.save(os.path.join(out_dir,
                                        key+'_weights.npy'),gmm_list_weights)
                    np.save(os.path.join(out_dir,
                                        key+'_means.npy'),gmm_list_means)
                    np.save(os.path.join(out_dir,
                                        key+'_covariances.npy'),gmm_list_covariances)
                    rows, cols = 19, 19
                    fig, axs = plt.subplots(
                        rows,
                        cols,
                        figsize=(10 * cols,10 * rows),
                        gridspec_kw={
                            'hspace': 0.1,
                            'wspace': 0,
                            'top': 0.95,
                            'bottom': 0,
                            'right': 1,
                            'left': 0
                        },
                    )
                                    # 遍历每个子图
                    for i in range(19):
                        for j in range(19):
                            ax = axs[i, j]
                            weights, means, covariances = gmm_list_weights[i][j], gmm_list_means[i][j], gmm_list_covariances[i][j]
                            # 创建x轴的数据
                            x = np.linspace(-10, 10, 1000)
                            # 计算GMM的曲线
                            curve = gmm_curve(x, weights, means, covariances)
                            # 绘制曲线
                            ax.plot(x, curve)
                            # 设置子图标题
                            title = f"({i}, {j}) Gaussian Components:\n"
                            for k in range(5):
                                weight = weights[k]
                                mean = means[k]
                                covariance = covariances[k]
                                title += f"Component {k+1}: weight={weight:.2f}, mean={mean:.2f}, covariance={covariance:.2f}\n"
                            ax.set_title(title)
                            # 隐藏坐标轴
                            ax.axis('off')
            
                    plt.savefig(
                        os.path.join(out_dir,
                                        f'{(self.local_iter + 1):06d}_{key}.png'))
                    plt.close()
    
                if 'pos_neg_gmm' in key:
                    pos_neg_gmm = clean_losses[key ]
                    pos_neg_gmm_weights = pos_neg_gmm[0].cpu().numpy()
                    pos_neg_gmm_means = pos_neg_gmm[1].cpu().numpy()
                    pos_neg_gmm_covariances = pos_neg_gmm[2].cpu().numpy()
                    np.save(os.path.join(out_dir,
                                        key+'_weights.npy'),pos_neg_gmm_weights)
                    np.save(os.path.join(out_dir,
                                        key+'_means.npy'),pos_neg_gmm_means)
                    np.save(os.path.join(out_dir,
                                        key+'_covariances.npy'),pos_neg_gmm_covariances)
                    rows, cols = 2, 1  
                    fig, axs = plt.subplots(
                        rows,
                        cols,
                        figsize=(10 * cols, 10 * rows),
                        gridspec_kw={
                            'hspace': 0.1,
                            'wspace': 0,
                            'top': 0.95,
                            'bottom': 0,
                            'right': 1,
                            'left': 0
                        },
                    )               
                    for i in range(2):
                        ax = axs[i]
                        weights, means, covariances = pos_neg_gmm_weights[i], pos_neg_gmm_means[i], pos_neg_gmm_covariances[i]
                        # 创建x轴的数据
                        x = np.linspace(-10, 10, 1000)
                        # 计算GMM的曲线
                        curve = gmm_curve(x, weights, means, covariances)
                        # 绘制曲线
                        ax.plot(x, curve)
                        # 设置子图标题
                        title = f"({i}) Gaussian Components:\n"
                        for k in range(5):
                            weight = weights[k]
                            mean = means[k]
                            covariance = covariances[k]
                            title += f"Component {k+1}: weight={weight:.2f}, mean={mean:.2f}, covariance={covariance:.2f}\n"
                        ax.set_title(title)
                        # 隐藏坐标轴
                        ax.axis('off')
            
                    plt.savefig(
                        os.path.join(out_dir,
                                        f'{(self.local_iter + 1):06d}_{key}.png'))
                    plt.close()
    
      
        self.local_iter += 1

        return log_vars
















## 注释掉target

@UDA.register_module()
class DACS_target(UDADecorator):

    def __init__(self, **cfg):
        super(DACS_target, self).__init__(**cfg)
        self.local_iter = 0
        self.max_iters = cfg['max_iters']
        self.alpha = cfg['alpha']
        self.pseudo_threshold = cfg['pseudo_threshold']
        self.psweight_ignore_top = cfg['pseudo_weight_ignore_top']
        self.psweight_ignore_bottom = cfg['pseudo_weight_ignore_bottom']
        self.fdist_lambda = cfg['imnet_feature_dist_lambda']
        self.fdist_classes = cfg['imnet_feature_dist_classes']
        self.fdist_scale_min_ratio = cfg['imnet_feature_dist_scale_min_ratio']
        self.enable_fdist = self.fdist_lambda > 0
        self.mix = cfg['mix']
        self.blur = cfg['blur']
        self.color_jitter_s = cfg['color_jitter_strength']
        self.color_jitter_p = cfg['color_jitter_probability']
        self.debug_img_interval = cfg['debug_img_interval']
        self.print_grad_magnitude = cfg['print_grad_magnitude']
        assert self.mix == 'class'

        self.debug_fdist_mask = None
        self.debug_gt_rescale = None

        self.class_probs = {}
        ema_cfg = deepcopy(cfg['model'])
        self.ema_model = build_segmentor(ema_cfg)

        if self.enable_fdist:
            self.imnet_model = build_segmentor(deepcopy(cfg['model']))
        else:
            self.imnet_model = None

        
        # 打印weight
        log_directory = os.path.join(self.train_cfg['work_dir'],
                                   'class_weight.txt')
        logging.basicConfig(filename=log_directory, level=logging.INFO)
        self.logger =  logging.getLogger()
        
    def get_ema_model(self):
        return get_module(self.ema_model)

    def get_imnet_model(self):
        return get_module(self.imnet_model)

    def _init_ema_weights(self):
        for param in self.get_ema_model().parameters():
            param.detach_()
        mp = list(self.get_model().parameters())
        mcp = list(self.get_ema_model().parameters())
        for i in range(0, len(mp)):
            if not mcp[i].data.shape:  # scalar tensor
                mcp[i].data = mp[i].data.clone()
            else:
                mcp[i].data[:] = mp[i].data[:].clone()

    def _update_ema(self, iter):
        alpha_teacher = min(1 - 1 / (iter + 1), self.alpha)
        for ema_param, param in zip(self.get_ema_model().parameters(),
                                    self.get_model().parameters()):
            if not param.data.shape:  # scalar tensor
                ema_param.data = \
                    alpha_teacher * ema_param.data + \
                    (1 - alpha_teacher) * param.data
            else:
                ema_param.data[:] = \
                    alpha_teacher * ema_param[:].data[:] + \
                    (1 - alpha_teacher) * param[:].data[:]

    def train_step(self, data_batch, optimizer, **kwargs):
        """The iteration step during training.

        This method defines an iteration step during training, except for the
        back propagation and optimizer updating, which are done in an optimizer
        hook. Note that in some complicated cases or models, the whole process
        including back propagation and optimizer updating is also defined in
        this method, such as GAN.

        Args:
            data (dict): The output of dataloader.
            optimizer (:obj:`torch.optim.Optimizer` | dict): The optimizer of
                runner is passed to ``train_step()``. This argument is unused
                and reserved.

        Returns:
            dict: It should contain at least 3 keys: ``loss``, ``log_vars``,
                ``num_samples``.
                ``loss`` is a tensor for back propagation, which can be a
                weighted sum of multiple losses.
                ``log_vars`` contains all the variables to be sent to the
                logger.
                ``num_samples`` indicates the batch size (when the model is
                DDP, it means the batch size on each GPU), which is used for
                averaging the logs.
        """

        optimizer.zero_grad()
        log_vars = self(**data_batch)
        optimizer.step()

        log_vars.pop('loss', None)  # remove the unnecessary 'loss'
        outputs = dict(
            log_vars=log_vars, num_samples=len(data_batch['img_metas']))
        return outputs

    def masked_feat_dist(self, f1, f2, mask=None):
        feat_diff = f1 - f2
        # mmcv.print_log(f'fdiff: {feat_diff.shape}', 'mmseg')
        pw_feat_dist = torch.norm(feat_diff, dim=1, p=2)
        # mmcv.print_log(f'pw_fdist: {pw_feat_dist.shape}', 'mmseg')
        if mask is not None:
            # mmcv.print_log(f'fd mask: {mask.shape}', 'mmseg')
            pw_feat_dist = pw_feat_dist[mask.squeeze(1)]
            # mmcv.print_log(f'fd masked: {pw_feat_dist.shape}', 'mmseg')
        return torch.mean(pw_feat_dist)

    def calc_feat_dist(self, img, gt, feat=None):
        assert self.enable_fdist
        with torch.no_grad():
            self.get_imnet_model().eval()
            feat_imnet = self.get_imnet_model().extract_feat(img)
            feat_imnet = [f.detach() for f in feat_imnet]
        lay = -1
        if self.fdist_classes is not None:
            fdclasses = torch.tensor(self.fdist_classes, device=gt.device)
            scale_factor = gt.shape[-1] // feat[lay].shape[-1]
            gt_rescaled = downscale_label_ratio(gt, scale_factor,
                                                self.fdist_scale_min_ratio,
                                                self.num_classes,
                                                255).long().detach()
            fdist_mask = torch.any(gt_rescaled[..., None] == fdclasses, -1)
            feat_dist = self.masked_feat_dist(feat[lay], feat_imnet[lay],
                                              fdist_mask)
            self.debug_fdist_mask = fdist_mask
            self.debug_gt_rescale = gt_rescaled
        else:
            feat_dist = self.masked_feat_dist(feat[lay], feat_imnet[lay])
        feat_dist = self.fdist_lambda * feat_dist
        feat_loss, feat_log = self._parse_losses(
            {'loss_imnet_feat_dist': feat_dist})
        feat_log.pop('loss', None)
        return feat_loss, feat_log

    def forward_train(self, img, img_metas, gt_semantic_seg, target_img,
                      target_img_metas):
        """Forward function for training.

        Args:
            img (Tensor): Input images.
            img_metas (list[dict]): List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:Collect`.
            gt_semantic_seg (Tensor): Semantic segmentation masks
                used if the architecture supports semantic segmentation task.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        log_vars = {}
        batch_size = img.shape[0]
        dev = img.device

        # Init/update ema model
        if self.local_iter == 0:
            self._init_ema_weights()
            # assert _params_equal(self.get_ema_model(), self.get_model())

        if self.local_iter > 0:
            self._update_ema(self.local_iter)
            # assert not _params_equal(self.get_ema_model(), self.get_model())
            # assert self.get_ema_model().training

        means, stds = get_mean_std(img_metas, dev)
   
        # Train on source images
        clean_losses = self.get_model().forward_train(
            img, img_metas, gt_semantic_seg, return_feat=True)
        
        for key in clean_losses.keys():
            if 'class_weight' in key :
                class_weight = clean_losses[key ]
                self.logger.info(f'{key} {self.local_iter }: {class_weight}')
        
        
        src_feat = clean_losses.pop('features')
        clean_loss, clean_log_vars = self._parse_losses(clean_losses)
        log_vars.update(clean_log_vars)
        clean_loss.backward(retain_graph=self.enable_fdist)
        
        
        if self.local_iter % self.debug_img_interval == 0:
            out_dir = os.path.join(self.train_cfg['work_dir'],
                                   'class_mix_debug')
            os.makedirs(out_dir, exist_ok=True)
            vis_img = torch.clamp(denorm(img, means, stds), 0, 1)
            vis_trg_img = torch.clamp(denorm(target_img, means, stds), 0, 1)
          
            for j in range(batch_size):
                rows, cols = 2, 5
                fig, axs = plt.subplots(
                    rows,
                    cols,
                    figsize=(3 * cols, 3 * rows),
                    gridspec_kw={
                        'hspace': 0.1,
                        'wspace': 0,
                        'top': 0.95,
                        'bottom': 0,
                        'right': 1,
                        'left': 0
                    },
                )
                subplotimg(axs[0][0], vis_img[j], 'Source Image')
                subplotimg(axs[1][0], vis_trg_img[j], 'Target Image')
                subplotimg(
                    axs[0][1],
                    gt_semantic_seg[j],
                    'Source Seg GT',
                    cmap='cityscapes')
             
                if self.debug_fdist_mask is not None:
                    subplotimg(
                        axs[0][4],
                        self.debug_fdist_mask[j][0],
                        'FDist Mask',
                        cmap='gray')
                if self.debug_gt_rescale is not None:
                    subplotimg(
                        axs[1][4],
                        self.debug_gt_rescale[j],
                        'Scaled GT',
                        cmap='cityscapes')
                for ax in axs.flat:
                    ax.axis('off')
                plt.savefig(
                    os.path.join(out_dir,
                                 f'{(self.local_iter + 1):06d}_{j}.png'))
                plt.close()
                
        if  (self.local_iter) % 1000 == 0:
            out_dir = os.path.join(self.train_cfg['work_dir'],
                                   'gmm_debug')
            os.makedirs(out_dir, exist_ok=True)
            for key in clean_losses.keys():
                if 'gmm_list' in key :
                    gmm_list = clean_losses[key ]
                if 'pos_neg_gmm' in key:
                    pos_neg_gmm = clean_losses[key ]
            gmm_list_weights = gmm_list[0].cpu().numpy()
            gmm_list_means = gmm_list[1].cpu().numpy()
            gmm_list_covariances = gmm_list[2].cpu().numpy()
            pos_neg_gmm_weights = pos_neg_gmm[0].cpu().numpy()
            pos_neg_gmm_means = pos_neg_gmm[1].cpu().numpy()
            pos_neg_gmm_covariances = pos_neg_gmm[2].cpu().numpy()
            
            np.save(os.path.join(out_dir,
                                'pos_neg_gmm_weights.npy'),pos_neg_gmm_weights)
            np.save(os.path.join(out_dir,
                                'pos_neg_gmm_means.npy'),pos_neg_gmm_means)
            np.save(os.path.join(out_dir,
                                'pos_neg_gmm_covariances.npy'),pos_neg_gmm_covariances)
            np.save(os.path.join(out_dir,
                                'gmm_list_weights.npy'),gmm_list_weights)
            np.save(os.path.join(out_dir,
                                'gmm_list_means.npy'),gmm_list_means)
            np.save(os.path.join(out_dir,
                                'gmm_list_covariances.npy'),gmm_list_covariances)
            
            rows, cols = 19, 19
            fig, axs = plt.subplots(
                rows,
                cols,
                figsize=(10 * cols,10 * rows),
                gridspec_kw={
                    'hspace': 0.1,
                    'wspace': 0,
                    'top': 0.95,
                    'bottom': 0,
                    'right': 1,
                    'left': 0
                },
            )
                            # 遍历每个子图
            for i in range(19):
                for j in range(19):
                    ax = axs[i, j]
                    weights, means, covariances = gmm_list_weights[i][j], gmm_list_means[i][j], gmm_list_covariances[i][j]
                    # 创建x轴的数据
                    x = np.linspace(-10, 10, 1000)
                    # 计算GMM的曲线
                    curve = gmm_curve(x, weights, means, covariances)
                    # 绘制曲线
                    ax.plot(x, curve)
                    # 设置子图标题
                    title = f"({i}, {j}) Gaussian Components:\n"
                    for k in range(5):
                        weight = weights[k]
                        mean = means[k]
                        covariance = covariances[k]
                        title += f"Component {k+1}: weight={weight:.2f}, mean={mean:.2f}, covariance={covariance:.2f}\n"
                    ax.set_title(title)
                    # 隐藏坐标轴
                    ax.axis('off')
    
            plt.savefig(
                os.path.join(out_dir,
                                f'{(self.local_iter + 1):06d}_19x19.png'))
            plt.close()
            
            
            
            rows, cols = 2, 1  
            fig, axs = plt.subplots(
                rows,
                cols,
                figsize=(10 * cols, 10 * rows),
                gridspec_kw={
                    'hspace': 0.1,
                    'wspace': 0,
                    'top': 0.95,
                    'bottom': 0,
                    'right': 1,
                    'left': 0
                },
            )
            
                        
            for i in range(2):
                ax = axs[i]
                weights, means, covariances = pos_neg_gmm_weights[i], pos_neg_gmm_means[i], pos_neg_gmm_covariances[i]
                # 创建x轴的数据
                x = np.linspace(-10, 10, 1000)
                # 计算GMM的曲线
                curve = gmm_curve(x, weights, means, covariances)
                # 绘制曲线
                ax.plot(x, curve)
                # 设置子图标题
                title = f"({i}) Gaussian Components:\n"
                for k in range(5):
                    weight = weights[k]
                    mean = means[k]
                    covariance = covariances[k]
                    title += f"Component {k+1}: weight={weight:.2f}, mean={mean:.2f}, covariance={covariance:.2f}\n"
                ax.set_title(title)
                # 隐藏坐标轴
                ax.axis('off')
    
            plt.savefig(
                os.path.join(out_dir,
                                f'{(self.local_iter + 1):06d}_pos_neg.png'))
            plt.close()
            
                
        self.local_iter += 1

        return log_vars
