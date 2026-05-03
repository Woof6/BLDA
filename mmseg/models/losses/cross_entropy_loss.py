# Obtained from: https://github.com/open-mmlab/mmsegmentation/tree/v0.16.0

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..builder import LOSSES
from .utils import get_class_weight, weight_reduce_loss
import numpy as np
from sklearn.mixture import GaussianMixture
from collections import namedtuple
import numpy as np
from scipy.stats import norm
import time
import random

class FocalLoss(nn.Module):
    def __init__(self, gama=1.5, alpha=0.25, weight=None, reduction="mean") -> None:
        super().__init__() 
        self.loss_fcn = torch.nn.CrossEntropyLoss(weight=weight, reduction=reduction)
        self.gama = gama 
        self.alpha = alpha 

    def forward(self, pre, target):
        logp = self.loss_fcn(pre, target)
        p = torch.exp(-logp) 
        loss = (1-p)**self.gama * self.alpha * logp
        return loss.mean()
    
def cross_entropy(pred,
                  label,
                  weight=None,
                  class_weight=None,
                  reduction='mean',
                  avg_factor=None,
                  ignore_index=-100):
    """The wrapper function for :func:`F.cross_entropy`"""
    # class_weight is a manual rescaling weight given to each class.
    # If given, has to be a Tensor of size C element-wise losses
    loss = F.cross_entropy(
        pred,
        label,
        weight=class_weight,
        reduction='none',
        ignore_index=ignore_index)
    
    # ## focal
    # gama = 2

    # loss = (1-torch.exp(-loss))**gama  *loss

    # apply weights and do the reduction
    if weight is not None:
        weight = weight.float()
    loss = weight_reduce_loss(
        loss, weight=weight, reduction=reduction, avg_factor=avg_factor)

    return loss


def _expand_onehot_labels(labels, label_weights, target_shape, ignore_index):
    """Expand onehot labels to match the size of prediction."""
    bin_labels = labels.new_zeros(target_shape)
    valid_mask = (labels >= 0) & (labels != ignore_index)
    inds = torch.nonzero(valid_mask, as_tuple=True)

    if inds[0].numel() > 0:
        if labels.dim() == 3:
            bin_labels[inds[0], labels[valid_mask], inds[1], inds[2]] = 1
        else:
            bin_labels[inds[0], labels[valid_mask]] = 1

    valid_mask = valid_mask.unsqueeze(1).expand(target_shape).float()
    if label_weights is None:
        bin_label_weights = valid_mask
    else:
        bin_label_weights = label_weights.unsqueeze(1).expand(target_shape)
        bin_label_weights *= valid_mask

    return bin_labels, bin_label_weights


def binary_cross_entropy(pred,
                         label,
                         weight=None,
                         reduction='mean',
                         avg_factor=None,
                         class_weight=None,
                         ignore_index=255):
    """Calculate the binary CrossEntropy loss.

    Args:
        pred (torch.Tensor): The prediction with shape (N, 1).
        label (torch.Tensor): The learning label of the prediction.
        weight (torch.Tensor, optional): Sample-wise loss weight.
        reduction (str, optional): The method used to reduce the loss.
            Options are "none", "mean" and "sum".
        avg_factor (int, optional): Average factor that is used to average
            the loss. Defaults to None.
        class_weight (list[float], optional): The weight for each class.
        ignore_index (int | None): The label index to be ignored. Default: 255

    Returns:
        torch.Tensor: The calculated loss
    """
    if pred.dim() != label.dim():
        assert (pred.dim() == 2 and label.dim() == 1) or (
                pred.dim() == 4 and label.dim() == 3), \
            'Only pred shape [N, C], label shape [N] or pred shape [N, C, ' \
            'H, W], label shape [N, H, W] are supported'
        label, weight = _expand_onehot_labels(label, weight, pred.shape,
                                              ignore_index)

    # weighted element-wise losses
    if weight is not None:
        weight = weight.float()
    loss = F.binary_cross_entropy_with_logits(
        pred, label.float(), pos_weight=class_weight, reduction='none')
    # do the reduction for the weighted loss
    loss = weight_reduce_loss(
        loss, weight, reduction=reduction, avg_factor=avg_factor)

    return loss


def mask_cross_entropy(pred,
                       target,
                       label,
                       reduction='mean',
                       avg_factor=None,
                       class_weight=None,
                       ignore_index=None):
    """Calculate the CrossEntropy loss for masks.

    Args:
        pred (torch.Tensor): The prediction with shape (N, C), C is the number
            of classes.
        target (torch.Tensor): The learning label of the prediction.
        label (torch.Tensor): ``label`` indicates the class label of the mask'
            corresponding object. This will be used to select the mask in the
            of the class which the object belongs to when the mask prediction
            if not class-agnostic.
        reduction (str, optional): The method used to reduce the loss.
            Options are "none", "mean" and "sum".
        avg_factor (int, optional): Average factor that is used to average
            the loss. Defaults to None.
        class_weight (list[float], optional): The weight for each class.
        ignore_index (None): Placeholder, to be consistent with other loss.
            Default: None.

    Returns:
        torch.Tensor: The calculated loss
    """
    assert ignore_index is None, 'BCE loss does not support ignore_index'
    # TODO: handle these two reserved arguments
    assert reduction == 'mean' and avg_factor is None
    num_rois = pred.size()[0]
    inds = torch.arange(0, num_rois, dtype=torch.long, device=pred.device)
    pred_slice = pred[inds, label].squeeze(1)
    return F.binary_cross_entropy_with_logits(
        pred_slice, target, weight=class_weight, reduction='mean')[None]


@LOSSES.register_module()
class CrossEntropyLoss(nn.Module):
    """CrossEntropyLoss.

    Args:
        use_sigmoid (bool, optional): Whether the prediction uses sigmoid
            of softmax. Defaults to False.
        use_mask (bool, optional): Whether to use mask cross entropy loss.
            Defaults to False.
        reduction (str, optional): . Defaults to 'mean'.
            Options are "none", "mean" and "sum".
        class_weight (list[float] | str, optional): Weight of each class. If in
            str format, read them from a file. Defaults to None.
        loss_weight (float, optional): Weight of the loss. Defaults to 1.0.
    """

    def __init__(self,
                 use_sigmoid=False,
                 use_mask=False,
                 reduction='mean',
                 class_weight=None,
                 loss_weight=1.0):
        super(CrossEntropyLoss, self).__init__()
        assert (use_sigmoid is False) or (use_mask is False)
        self.use_sigmoid = use_sigmoid
        self.use_mask = use_mask
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.class_weight = get_class_weight(class_weight)

        if self.use_sigmoid:
            self.cls_criterion = binary_cross_entropy
        elif self.use_mask:
            self.cls_criterion = mask_cross_entropy
        else:
            self.cls_criterion = cross_entropy

    def forward(self,
                cls_score,
                label,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                **kwargs):
        """Forward function."""
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        if self.class_weight is not None:
            class_weight = cls_score.new_tensor(self.class_weight)
        else:
            class_weight = None
        loss_cls = self.loss_weight * self.cls_criterion(
            cls_score,
            label,
            weight,
            class_weight=class_weight,
            reduction=reduction,
            avg_factor=avg_factor,
            **kwargs)
        return loss_cls






@LOSSES.register_module()
class CrossEntropyLoss_RW(nn.Module):
    """CrossEntropyLoss.

    Args:
        use_sigmoid (bool, optional): Whether the prediction uses sigmoid
            of softmax. Defaults to False.
        use_mask (bool, optional): Whether to use mask cross entropy loss.
            Defaults to False.
        reduction (str, optional): . Defaults to 'mean'.
            Options are "none", "mean" and "sum".
        class_weight (list[float] | str, optional): Weight of each class. If in
            str format, read them from a file. Defaults to None.
        loss_weight (float, optional): Weight of the loss. Defaults to 1.0.
    """

    def __init__(self,
                 use_sigmoid=False,
                 use_mask=False,
                 reduction='mean',
                 class_weight=None,
                 loss_weight=1.0):
        super(CrossEntropyLoss_RW, self).__init__()
        assert (use_sigmoid is False) or (use_mask is False)
        self.use_sigmoid = use_sigmoid
        self.use_mask = use_mask
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.class_weight = get_class_weight(class_weight)
        self.num_classes = 19
        self.pixel_counts_source =  torch.ones(self.num_classes, dtype=torch.float32)
        self.pixel_counts_target =  torch.ones(self.num_classes, dtype=torch.float32)

        if self.use_sigmoid:
            self.cls_criterion = binary_cross_entropy
        elif self.use_mask:
            self.cls_criterion = mask_cross_entropy
        else:
            self.cls_criterion = cross_entropy

    def forward(self,
                cls_score,
                label,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                **kwargs):
        """Forward function."""
        assert reduction_override in (None, 'none', 'mean', 'sum')
        
        if weight is None:
            for class_id in range(self.num_classes):
                self.pixel_counts_source[class_id] += torch.sum(label == class_id).cpu()
        
            reduction = (
                reduction_override if reduction_override else self.reduction)
            if self.class_weight is not None:
                class_weight = cls_score.new_tensor(self.class_weight)
            else:
                class_weight = None
            # class_weight =  self.pixel_counts_source/torch.sum(self.pixel_counts_source ).cpu() 
            # class_weight = 1/torch.log(1.02+class_weight.to(cls_score.device))        
            class_weight = 1/torch.log(self.pixel_counts_source.to(cls_score.device)+1) 
            class_weight = class_weight/torch.sum(class_weight)*19
        
            loss_cls = self.loss_weight * self.cls_criterion(
                cls_score,
                label,
                weight,
                class_weight=  class_weight ,
                reduction=reduction,
                avg_factor=avg_factor,
                **kwargs)
    
      
            return [loss_cls,class_weight]
        
        else:
            source_mask = weight ==1
            target_mask = weight <1
            for class_id in range(self.num_classes):
                self.pixel_counts_source[class_id] += torch.sum((label == class_id)*source_mask ).cpu()
                self.pixel_counts_target[class_id] += torch.sum((label == class_id)*target_mask*weight).cpu()
                
            reduction = (
                reduction_override if reduction_override else self.reduction)
            if self.class_weight is not None:
                class_weight = cls_score.new_tensor(self.class_weight)
            else:
                class_weight = None
            # class_weight_source = self.pixel_counts_source/torch.sum(self.pixel_counts_source ).cpu() 
            # class_weight_target=  self.pixel_counts_target/torch.sum(self.pixel_counts_target).cpu()
            # class_weight_source = 1/torch.log(1.02+class_weight_source.to(cls_score.device))     
            # class_weight_target = 1/torch.log(1.02+class_weight_target.to(cls_score.device))     
            class_weight_source  = 1/torch.log(self.pixel_counts_source.to(cls_score.device)+1)
            class_weight_target = 1/torch.log(self.pixel_counts_target.to(cls_score.device)+1)
            class_weight_source = class_weight_source/torch.sum(class_weight_source)*19
            class_weight_target = class_weight_target/torch.sum(class_weight_target)*19
            # print(class_weight )
            # print(torch.max(self.pixel_counts).cpu())
            # print(self.pixel_counts)
            
            loss_cls = self.loss_weight * self.cls_criterion(
                cls_score,
                label,
                weight*source_mask,
                class_weight= class_weight_source,
                reduction=reduction,
                avg_factor=avg_factor,
                **kwargs)+ self.loss_weight * self.cls_criterion(
                cls_score,
                label,
                weight*target_mask,
                class_weight= class_weight_target,
                reduction=reduction,
                avg_factor=avg_factor,
                **kwargs)
    
    
     
    
            return [loss_cls,class_weight_target]
        
        
# from scipy.interpolate import interp1d
   
# @LOSSES.register_module()
# class CrossEntropyLoss_GMM(nn.Module):
#     """CrossEntropyLoss.

#     Args:
#         use_sigmoid (bool, optional): Whether the prediction uses sigmoid
#             of softmax. Defaults to False.
#         use_mask (bool, optional): Whether to use mask cross entropy loss.
#             Defaults to False.
#         reduction (str, optional): . Defaults to 'mean'.
#             Options are "none", "mean" and "sum".
#         class_weight (list[float] | str, optional): Weight of each class. If in
#             str format, read them from a file. Defaults to None.
#         loss_weight (float, optional): Weight of the loss. Defaults to 1.0.
#     """

#     def __init__(self,
#                  use_sigmoid=False,
#                  use_mask=False,
#                  reduction='mean',
#                  class_weight=None,
#                  loss_weight=1.0,
#                  k =5,
#                  iteration = 1,
#                  tau = 0.99,
#                  min_pixels = 500):
#         super(CrossEntropyLoss_GMM, self).__init__()
#         assert (use_sigmoid is False) or (use_mask is False)
#         self.use_sigmoid = use_sigmoid
#         self.use_mask = use_mask
#         self.reduction = reduction
#         self.loss_weight = loss_weight
#         self.class_weight = get_class_weight(class_weight)
#         self.num_classes = 19
#         self.pixel_counts_source =  torch.ones(self.num_classes, dtype=torch.float32)
#         self.pixel_counts_target =  torch.ones(self.num_classes, dtype=torch.float32)

#         if self.use_sigmoid:
#             self.cls_criterion = binary_cross_entropy
#         elif self.use_mask:
#             self.cls_criterion = mask_cross_entropy
#         else:
#             self.cls_criterion = cross_entropy

#         self.k = k
#         self.iteration = iteration
#         self.tau = tau
#         self.num_classes = 19
#         self.min_pixels = min_pixels
#         self.gmm_list = [[None] * self.num_classes for _ in range(self.num_classes)]
#         self.pos_gmm = None
#         self.neg_gmm = None
   

#     def forward(self,
#                 cls_score,
#                 label,
#                 weight=None,
#                 avg_factor=None,
#                 reduction_override=None,
#                 **kwargs):
#         """Forward function."""
#         assert reduction_override in (None, 'none', 'mean', 'sum')
        
#         reduction = (
#             reduction_override if reduction_override else self.reduction)
        
        
#         logits = cls_score.detach().cpu().numpy()
#         gt = label.detach().cpu().numpy()
#         cdf = np.zeros_like(logits)   ##存储各个logits对应的累计分布
 
        
#         label_onehot, weight_ = _expand_onehot_labels(label, weight,  cls_score.shape,ignore_index = 255)
#         # 统计 pos
#         flattened_logits = logits.flatten()
#         pos_mask = (label_onehot==1)
#         logits_filtered = flattened_logits[pos_mask.detach().cpu().numpy().flatten()]
#         if self.pos_gmm is None:
#             gmm = GaussianMixture(n_components=self.k, max_iter= self.iteration)
#             gmm.fit(logits_filtered.reshape(-1, 1))
#             self.pos_gmm = {
#                             'means': gmm.means_,
#                             'covariances': gmm.covariances_,
#                             'weights': gmm.weights_
#                         }
#         else:
#             gmm_params = self.pos_gmm
#             means = gmm_params['means']
#             covariances = gmm_params['covariances']
#             weights = gmm_params['weights']
#             gmm = GaussianMixture(n_components=self.k, means_init= means, precisions_init= np.linalg.inv(covariances),weights_init=weights, max_iter= self.iteration)
#             gmm.fit(logits_filtered.reshape(-1, 1))
#             # EMA 更新
#             self.pos_gmm = {
#                             'means': self.tau*means +(1-self.tau)*gmm.means_,
#                             'covariances':  self.tau*covariances +(1-self.tau)*gmm.covariances_,
#                             'weights': self.tau*weights +(1-self.tau)*gmm.weights_,
#                         }
        
        
#         # 统计 neg
#         flattened_logits = flattened_logits.flatten()
#         neg_mask = (label_onehot==0)* (label!=255).unsqueeze(1)
#         logits_filtered = flattened_logits[neg_mask.detach().cpu().numpy().flatten()]
#         if self.neg_gmm is None:
#             gmm = GaussianMixture(n_components=self.k, max_iter= self.iteration)
#             gmm.fit(logits_filtered.reshape(-1, 1))
#             self.neg_gmm = {
#                             'means': gmm.means_,
#                             'covariances': gmm.covariances_,
#                             'weights': gmm.weights_
#                         }
#         else:
#             gmm_params = self.neg_gmm
#             means = gmm_params['means']
#             covariances = gmm_params['covariances']
#             weights = gmm_params['weights']
#             gmm = GaussianMixture(n_components=self.k, means_init= means, precisions_init= np.linalg.inv(covariances),weights_init=weights, max_iter= self.iteration)
#             gmm.fit(logits_filtered.reshape(-1, 1))
#             # EMA 更新
#             self.neg_gmm = {
#                             'means': self.tau*means +(1-self.tau)*gmm.means_,
#                             'covariances':  self.tau*covariances +(1-self.tau)*gmm.covariances_,
#                             'weights': self.tau*weights +(1-self.tau)*gmm.weights_,
#                         }
            
#         for class_id in range(self.num_classes):
#                 class_gt_mask = (gt == class_id)
#                 for class_id2 in range(self.num_classes):
#                     flattened_logits = logits[:,class_id2].flatten()
#                     class_logits_filtered = flattened_logits[class_gt_mask.flatten()]
#                     data = class_logits_filtered 
#                     if len(class_logits_filtered >=self.min_pixels): # 满足一定像素 更新GMM
#                         if self.gmm_list[class_id][class_id2] is None:
#                             gmm = GaussianMixture(n_components=self.k, max_iter= self.iteration)
#                             gmm.fit(data.reshape(-1, 1))
#                             self.gmm_list[class_id][class_id2] = {
#                                             'means': gmm.means_,
#                                             'covariances': gmm.covariances_,
#                                             'weights': gmm.weights_
#                                         }
#                         else:
#                             gmm_params = self.gmm_list[class_id][class_id2]
#                             means = gmm_params['means']
#                             covariances = gmm_params['covariances']
#                             weights = gmm_params['weights']
#                             gmm = GaussianMixture(n_components=self.k, means_init= means, precisions_init= np.linalg.inv(covariances),weights_init=weights, max_iter= self.iteration)
#                             gmm.fit(data.reshape(-1, 1))
#                             # EMA 更新
#                             self.gmm_list[class_id][class_id2] = {
#                                             'means': self.tau*means +(1-self.tau)*gmm.means_,
#                                             'covariances':  self.tau*covariances +(1-self.tau)*gmm.covariances_,
#                                             'weights': self.tau*weights +(1-self.tau)*gmm.weights_,
#                                         }

#                     if self.gmm_list[class_id][class_id2]  is  not None:
#                         # 获取 GMM 的参数
#                         gmm_params = self.gmm_list[class_id][class_id2]
#                         means = gmm_params['means'].flatten()
#                         covariances = gmm_params['covariances'].flatten()
#                         weights = gmm_params['weights']
                
#                         for w, mu, cov in zip(weights, means, covariances):
#                             cdf[:,class_id] += w * norm.cdf(logits[:,class_id], mu, np.sqrt(cov))*class_gt_mask


#         # 获取 pos GMM 的参数
#         gmm_params = self.pos_gmm
#         means = gmm_params['means'].flatten()
#         stds = np.sqrt(gmm_params['covariances'].flatten()).flatten()
#         weights =gmm_params['weights']
#         # 创建GMM分布对象
#         gmm_dist = []
#         for i in range(len(means)):
#             gmm_dist.append(norm(loc=means[i], scale=stds[i]))

#         # 创建连续的CDF函数表示
#         x = np.linspace(-50, 50, 1000)  # 定义查询范围和精度
#         y = np.array([np.sum([weights[i] * gmm_dist[i].cdf(val) for i in range(len(means))]) for val in x])

#         # 创建插值函数
#         cdf_interp_func = interp1d(y, x, fill_value='extrapolate')
#         # 直接查询反函数
#         logits_pos_gt = cdf_interp_func(cdf)

#         # 获取 neg GMM 的参数
#         gmm_params = self.neg_gmm
#         means = gmm_params['means'].flatten()
#         stds = np.sqrt(gmm_params['covariances'].flatten()).flatten()
#         weights =gmm_params['weights']
#         # 创建GMM分布对象
#         gmm_dist = []
#         for i in range(len(means)):
#             gmm_dist.append(norm(loc=means[i], scale=stds[i]))

#         # 创建连续的CDF函数表示
#         x = np.linspace(-50, 50, 1000)  # 定义查询范围和精度
#         y = np.array([np.sum([weights[i] * gmm_dist[i].cdf(val) for i in range(len(means))]) for val in x])

#         # 创建插值函数
#         cdf_interp_func = interp1d(y, x, fill_value='extrapolate')
#         # 直接查询反函数
#         logits_neg_gt = cdf_interp_func(cdf)          
        
        
#         logits_pos_gt = torch.from_numpy(logits_pos_gt)
#         logits_pos_gt = logits_pos_gt.to(cls_score.device)
#         logits_neg_gt = torch.from_numpy(logits_neg_gt)
#         logits_neg_gt = logits_neg_gt.to(cls_score.device)
#         logits_gt =  logits_pos_gt*pos_mask + logits_neg_gt*neg_mask       
#         logit_mask = torch.from_numpy(logits_pos_gt)
#         loss_adjust_logit =  torch.abs((logits_gt-cls_score)*(cdf>0)*logit_mask )                    
#         loss_adjust_logit = loss_adjust_logit.mean()
                    
#         if self.class_weight is not None:
#             class_weight = cls_score.new_tensor(self.class_weight)
#         else:
#             class_weight = None
#         loss_cls = self.loss_weight * self.cls_criterion(
#             cls_score,
#             label,
#             weight,
#             class_weight=class_weight,
#             reduction=reduction,
#             avg_factor=avg_factor,
#             **kwargs)
#         print(loss_adjust_logit)
#         return [loss_cls,loss_adjust_logit]


# 定义正态分布的CDF近似计算函数
def approx_normal_cdf(x):
    t = 1 / (1 + 0.2316419 * torch.abs(x))
    d = 0.3989423 * torch.exp(-x**2 / 2)

    p = d * t * (0.3193815 + t * (-0.3565638 + t * (1.781478 + t * (-1.821256 + t * 1.330274))))

    mask = x > 0
    
    p[mask] = 1 - p[mask]

    return p
    
def interp1d(x, y, x_new,ratio):
    
  
    x_new =  torch.clamp(x_new ,min=ratio,max=1-ratio)
    indices = torch.searchsorted(x, x_new)
    # indices_right = torch.searchsorted(x, x_new,right= True)
    
     # 处理超出范围的索引
    indices = torch.clamp(indices, min = 1, max = len(x)-1)
    # indices_right = torch.clamp(indices_right, min = 1, max = len(x)-1)

    x_left = x[indices - 1]
    x_right = x[indices]
    y_left = y[indices - 1]
    y_right = y[indices]
    

    
    interpolation = (x_new - x_left) / (x_right - x_left)
    y_interp = y_left + interpolation * (y_right - y_left)
    
    
    # 最小最大值时 按左右最靠近得索引来
    # y_interp = torch.where(x_new == x[0], y[indices_right], y_interp)
    # y_interp = torch.where(x_new == x[-1], y[indices], y_interp)
     
    return y_interp


class GaussianMixtureModel1d:
    def __init__(self, shape,num_components, tau):
        self.new_shape = list(shape) + [num_components]
        self.num_components = num_components
        # 初始化模型参数
        self.weights = torch.ones(self.new_shape) / self.num_components
        self.means = torch.randn(*self.new_shape)
        self.covariances = torch.ones(self.new_shape)

        #warmup_ema
        self.tau = tau
        self.i = 0
        self.warmup = 1500
        
        self.weights_pre = None
        self.covariances_pre = None
        self.means_pre = None
        self.updated_mask = torch.zeros(shape)
        # 记录有多少代没更新
        self.noupdated_mask = torch.ones(shape).unsqueeze(-1)
        
    def update_device(self,device):
        self.means= self.means.to(device)
        self.weights = self.weights.to(device)
        self.covariances= self.covariances.to(device)
        self.updated_mask = self.updated_mask.to(device)
        self.noupdated_mask = self.noupdated_mask.to(device)
        
    def fit(self, data,  num_iterations=100):

        for _ in range(num_iterations):
            # E 步骤：计算每个样本属于每个分量的后验概率
            posteriors = self._expectation(data)
    
            # M 步骤：更新模型参数
            self._maximization(data, posteriors)
           
    def _expectation(self, data):

        
        posteriors  = self.weights.unsqueeze(-2)*self._multivariate_normal(data, self.means, self.covariances) #shape，num_samples，num_components
        posteriors += 1e-7 # 防止NAN
        # # 归一化后验概率
        # print( torch.min(posteriors ),torch.max(posteriors))
        posteriors /= torch.sum(posteriors, -1, keepdims=True)
        # print("1", torch.min(posteriors ),torch.max(posteriors))
        return posteriors

    def _maximization(self, data, posteriors):
        num_samples = data.shape[-1]
        total_posteriors = torch.sum(posteriors, -2)

        # 更新权重
        self.weights = total_posteriors / num_samples #shape，num_components
        # print( torch.min(self.weights ),torch.max(self.weights))
        
        # # 防止除以0
        # total_posteriors = torch.clamp(total_posteriors,min=1e-4)
        
        # 更新均值
        self.means = torch.sum(posteriors*data.unsqueeze(-1),-2) / total_posteriors
    
        # 更新协方差矩阵
      
        diff = data.unsqueeze(-1)-self.means.unsqueeze(-2)
        self.covariances = torch.sum( diff*diff*posteriors,-2) / total_posteriors
        self.covariances = torch.clip(self.covariances,min = 1e-4, max=None)
        
        # print( torch.min(self.weights ),torch.max(self.weights))
        # nan_count = torch.sum(torch.isnan(total_posteriors)).item()
        # print("Number of NaN values:", nan_count) 
        # print( torch.min(total_posteriors),torch.max(total_posteriors))
        # nan_count = torch.sum(torch.isnan(self.means)).item()
        # print("Number of NaN values:", nan_count) 
        
        # 更新协方差矩阵
      
    def _multivariate_normal(self, data, mean, covariance):
        
        data = data.unsqueeze(-1)#19,19,100,1
        mean = mean.unsqueeze(-2)  #19,19,1,5
        covariance =  covariance.unsqueeze(-2)  
        likelihood = 1 / torch.sqrt(2 * np.pi * covariance ) * torch.exp(-(data - mean)**2 / (2 * covariance)) 
 
        return likelihood
    
    def copy_weight(self):
        self.means_pre = self.means.clone()
        self.weights_pre = self.weights.clone()
        self.covariances_pre = self.covariances.clone()
        # self.means_pre = self.means
        # self.weights_pre = self.weights
        # self.covariances_pre = self.covariances
        
    def EMA_updata_weight(self,mask,w=1):
        self.updated_mask += mask
        mask = mask.unsqueeze(-1)
        # print(w)
        tau = self.tau *min(self.i/self.warmup,1)
        tau = torch.pow(tau,self.noupdated_mask)
        # print(torch.max(tau),torch.min(tau))
        self.weights = w*mask*(tau*self.weights_pre+(1-tau)*self.weights) + (1-w*mask)*self.weights_pre
        self.means = w*mask*(tau*self.means_pre+(1-tau)*self.means) + (1-w*mask)*self.means_pre 
        self.covariances = w*mask*(tau*self.covariances_pre+(1-tau)*self.covariances)+ (1-w*mask)*self.covariances_pre
        self.i += w
        self.noupdated_mask = mask + (1-mask)*(self.noupdated_mask+1)
         
def calculate_min_category_pixel_count(labels, min_required_count, ignored_category):
    # 将标签列表转换为PyTorch张量，并展平为一维张量
    labels_tensor = labels.flatten()

    # 计算每个类别的像素数量
    pixel_counts = torch.bincount(labels_tensor)
    
    # 排除被忽略的类别ID的像素数量
    if len(pixel_counts) > ignored_category:
        pixel_counts[ignored_category] = 0

    # 找到满足要求的类别ID
    valid_categories = torch.nonzero(pixel_counts >= min_required_count).flatten()
    
    # 如果不存在满足要求的类别，返回 -1
    if valid_categories.numel() == 0:
        return -1

    # 找到满足要求的类别ID中的最小像素数量
    min_pixel_count = torch.min(pixel_counts[valid_categories])
    
    # return min_required_count
    return min_pixel_count.item()  

@LOSSES.register_module()
class CrossEntropyLoss_GMM(nn.Module):
    """CrossEntropyLoss.

    Args:
        use_sigmoid (bool, optional): Whether the prediction uses sigmoid
            of softmax. Defaults to False.
        use_mask (bool, optional): Whether to use mask cross entropy loss.
            Defaults to False.
        reduction (str, optional): . Defaults to 'mean'.
            Options are "none", "mean" and "sum".
        class_weight (list[float] | str, optional): Weight of each class. If in
            str format, read them from a file. Defaults to None.
        loss_weight (float, optional): Weight of the loss. Defaults to 1.0.
    """

    def __init__(self,
                 use_sigmoid=False,
                 use_mask=False,
                 reduction='mean',
                 class_weight=None,
                 loss_weight=1.0,
                 k =5,
                 iteration = 3,
                 tau = 0.99,
                 min_pixels = 100,
                 ratio = 0.05,
                 t= 0.1,
                 t2= 0.2):
        super(CrossEntropyLoss_GMM, self).__init__()
        assert (use_sigmoid is False) or (use_mask is False)
        self.use_sigmoid = use_sigmoid
        self.use_mask = use_mask
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.class_weight = get_class_weight(class_weight)
        self.num_classes = 19
        self.pixel_counts_source =  torch.ones(self.num_classes, dtype=torch.float32)
        self.pixel_counts_target =  torch.ones(self.num_classes, dtype=torch.float32)

        if self.use_sigmoid:
            self.cls_criterion = binary_cross_entropy
        elif self.use_mask:
            self.cls_criterion = mask_cross_entropy
        else:
            self.cls_criterion = cross_entropy

        self.k = k
        self.iteration = iteration
        self.tau = tau
        self.num_classes = 19
        self.t = t
        self.t2 = t2
        self.min_pixels = min_pixels
        self.gmm_list = GaussianMixtureModel1d(shape = (self.num_classes,self.num_classes),num_components=self.k, tau=tau) 
        self.pos_neg_gmm = GaussianMixtureModel1d(shape = (2,),num_components=self.k,tau=tau)
        self.ratio = ratio
        self.gmm_list_target = GaussianMixtureModel1d(shape = (self.num_classes,self.num_classes),num_components=self.k, tau=tau) 
        self.pos_neg_gmm_target = GaussianMixtureModel1d(shape = (2,),num_components=self.k,tau=tau)

    def forward(self,
                cls_score,
                label,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                **kwargs):
        """Forward function."""
        assert reduction_override in (None, 'none', 'mean', 'sum')
        
        reduction = (
            reduction_override if reduction_override else self.reduction)
        
        reg_score = None
        if isinstance(cls_score,list):
            reg_score = cls_score[1]
            cls_score = cls_score[0]
        # 权重低于1的是target pixel
        if weight is None:
            logits = cls_score.detach().clone()
            gt = label.detach().clone()
            cdf = torch.zeros_like(logits,device=cls_score.device)   ##存储各个logits对应的累计分布
    
            
            self.gmm_list.update_device(cls_score.device)
            self.pos_neg_gmm.update_device(cls_score.device)
            
            
            label_onehot, weight_ = _expand_onehot_labels(label, weight,  cls_score.shape,ignore_index = 255)
            
            # 统计 pos 和 neg
            flattened_logits = logits
            pos_mask = (label_onehot==1)
            logits_filtered1 = flattened_logits[pos_mask]
            neg_mask = (label_onehot==0)* (label!=255).unsqueeze(1)
            logits_filtered2 = flattened_logits[neg_mask]
            mask = torch.zeros((2),device=cls_score.device)
            data = torch.zeros((2, self.min_pixels ),device=cls_score.device)
            
        
            if len(logits_filtered1) >=self.min_pixels:
                total_len = len(logits_filtered1)
                # total_len = self.min_pixels
                data = torch.zeros((2, total_len ),device=cls_score.device)
                data[0] = logits_filtered1[torch.randperm(len(logits_filtered1))[:total_len]]
                data[1] = logits_filtered2[torch.randperm(len(logits_filtered2))[:total_len]]
                mask = torch.tensor([1,1],device=cls_score.device)
            elif len(logits_filtered2) >=self.min_pixels:
                total_len = len(logits_filtered2)
                # total_len = self.min_pixels
                data = torch.zeros((2, total_len ),device=cls_score.device)
                data[1] = logits_filtered2[torch.randperm(len(logits_filtered2))[:total_len]]
                mask = torch.tensor([0,1],device=cls_score.device)
                
            self.pos_neg_gmm.copy_weight()
            self.pos_neg_gmm.fit(data, self.iteration)
            self.pos_neg_gmm.EMA_updata_weight(mask) 

            total_len = calculate_min_category_pixel_count(labels = label, min_required_count= self.min_pixels, ignored_category=255)    
    
            if total_len >0:
                mask = torch.zeros((self.num_classes,self.num_classes),device=cls_score.device)
                data = torch.zeros((self.num_classes,self.num_classes,total_len),device=cls_score.device)
                
                for class_id in range(self.num_classes):
                    class_gt_mask = (gt == class_id)
                    if torch.sum(class_gt_mask) >= self.min_pixels:
                        true_indices = torch.nonzero(class_gt_mask)
                        sample_indices = true_indices[random.sample(range(true_indices.size(0)), total_len)].T.tolist()
                        for class_id2 in range(self.num_classes):
                            flattened_logits = logits[:,class_id2]
                            data[class_id][class_id2] =   flattened_logits[sample_indices]
                            mask[class_id][class_id2] =1
                
                                    
                self.gmm_list.copy_weight()
                self.gmm_list.fit(data, self.iteration)
                self.gmm_list.EMA_updata_weight(mask) 
            
                

            
            index = gt*(gt!=255)
            weights_matrix = self.gmm_list.weights[index ,:,:]  #2 512 512 19 5
            weights_matrix = weights_matrix.permute(0,3,1,2,4)#2 19 512 512  5
            means_matrix = self.gmm_list.means[index ,:,:]
            means_matrix = means_matrix.permute(0,3,1,2,4)
            covariances_matrix = self.gmm_list.covariances[index ,:,:]
            covariances_matrix =covariances_matrix.permute(0,3,1,2,4)
            cdf_mask = self.gmm_list.updated_mask>0
            cdf_mask = cdf_mask[index ,:] #2 512 512 19 
            cdf_mask  = cdf_mask.permute(0,3,1,2)
            
        

            data = (logits.unsqueeze(-1)-means_matrix)/ torch.sqrt(covariances_matrix)
            cdf = torch.sum(weights_matrix*approx_normal_cdf(data)*cdf_mask.unsqueeze(-1),-1) #2 19  512 512
            cdf_gt = torch.sum(cdf*pos_mask,1,keepdim=True)
            cdf_gt_mask = torch.sum(cdf_mask,1,keepdim=True)>0
            
            # 获取 pos GMM 的参数
            means = self.pos_neg_gmm.means[0]
            covariances = self.pos_neg_gmm.covariances[0]
            weights =self.pos_neg_gmm.weights[0]
            
            # 创建连续的CDF函数表示
            left = means-3*torch.sqrt(covariances)
            right = means+3*torch.sqrt(covariances)
            x = torch.linspace(torch.min(left), torch.max(right), 1000,device=cls_score.device)   
            y  = x.unsqueeze(1)
            y  = (y -means)/torch.sqrt(covariances)
            y = torch.sum(weights*approx_normal_cdf(y ),1)
            # 创建插值函数
            logits_pos_gt = interp1d(y, x,cdf,self.ratio)   
        
            # 获取 neg GMM 的参数
            means = self.pos_neg_gmm.means[1]
            covariances = self.pos_neg_gmm.covariances[1]
            weights =self.pos_neg_gmm.weights[1]
            
            # 创建连续的CDF函数表示
            left = means-3*torch.sqrt(covariances)
            right = means+3*torch.sqrt(covariances)
            x = torch.linspace(torch.min(left), torch.max(right), 1000,device=cls_score.device)  
            y  = x.unsqueeze(1)
            y  = (y -means)/torch.sqrt(covariances)
            y = torch.sum(weights*approx_normal_cdf(y ),1)
            # 创建插值函数
            logits_neg_gt = interp1d(y, x, cdf,self.ratio)      
            # print("neg",torch.max(cdf[neg_mask*cdf_mask]),torch.min(cdf[neg_mask*cdf_mask]),torch.mean(cdf[neg_mask*cdf_mask]))
            # print(torch.max(logits_neg_gt[neg_mask*cdf_mask]),torch.min(logits_neg_gt[neg_mask*cdf_mask]))
        
    
        
            logits_gt =  logits_pos_gt*pos_mask + logits_neg_gt*neg_mask    
   
            # loss_adjust_logit =  torch.abs((logits_gt-cls_score)* cdf_mask*(label.unsqueeze(1)!=255) )                    
            # loss_adjust_logit = loss_adjust_logit.mean()
            
            # 修正 logit
            loss_adjust_logit =  (logits_gt-logits)* cdf_mask*(label.unsqueeze(1)!=255)                  
    
            
        else:
            
            source_mask = weight==1
            target_mask = (weight<1)&(weight>0)
            logits = cls_score.detach().clone()
            gt = label.detach().clone()
            cdf = torch.zeros_like(logits,device=cls_score.device)   ##存储各个logits对应的累计分布
    
            
            # self.gmm_list.update_device(cls_score.device)
            # self.pos_neg_gmm.update_device(cls_score.device)
            self.gmm_list_target.update_device(cls_score.device)
            self.pos_neg_gmm_target.update_device(cls_score.device)
            
            
            label_onehot, weight_ = _expand_onehot_labels(label, None,  cls_score.shape,ignore_index = 255)
            
            # # 统计 source pos 和 neg
            flattened_logits = logits
            pos_mask = (label_onehot==1)
            neg_mask = (label_onehot==0)* (label!=255).unsqueeze(1)
          
            
            # 统计 target list
            total_len = calculate_min_category_pixel_count(labels = label*target_mask+(~target_mask)*255, min_required_count= self.min_pixels, ignored_category=255)    
    
            if total_len >0:
                mask = torch.zeros((self.num_classes,self.num_classes),device=cls_score.device)
                data = torch.zeros((self.num_classes,self.num_classes,total_len),device=cls_score.device)
                
                for class_id in range(self.num_classes):
                    class_gt_mask = (gt == class_id)*target_mask
                    if torch.sum(class_gt_mask) >= self.min_pixels:
                        true_indices = torch.nonzero(class_gt_mask)
                        sample_indices = true_indices[random.sample(range(true_indices.size(0)), total_len)].T.tolist()
                        for class_id2 in range(self.num_classes):
                            flattened_logits = logits[:,class_id2]
                            data[class_id][class_id2] =   flattened_logits[sample_indices]
                            mask[class_id][class_id2] =1
         
                self.gmm_list_target.copy_weight()
                self.gmm_list_target.fit(data, self.iteration)
                self.gmm_list_target.EMA_updata_weight(mask,1) 
   
            
                

            
            index = gt*(gt!=255)
            
            
            ##soucre
            weights_matrix = self.gmm_list.weights[index ,:,:]  #2 512 512 19 5
            weights_matrix = weights_matrix.permute(0,3,1,2,4)#2 19 512 512  5
            means_matrix = self.gmm_list.means[index ,:,:]
            means_matrix = means_matrix.permute(0,3,1,2,4)
            covariances_matrix = self.gmm_list.covariances[index ,:,:]
            covariances_matrix =covariances_matrix.permute(0,3,1,2,4)
            cdf_mask = self.gmm_list.updated_mask>0
            cdf_mask = cdf_mask[index ,:] #2 512 512 19 
            cdf_mask  = cdf_mask.permute(0,3,1,2)
            data = (logits.unsqueeze(-1)-means_matrix)/ torch.sqrt(covariances_matrix)
            cdf = torch.sum(weights_matrix*approx_normal_cdf(data)*cdf_mask.unsqueeze(-1),-1) #2 19  512 512
                          

            ## target
            weights_matrix = self.gmm_list_target.weights[index ,:,:]  #2 512 512 19 5
            weights_matrix = weights_matrix.permute(0,3,1,2,4)#2 19 512 512  5
            means_matrix = self.gmm_list_target.means[index ,:,:]
            means_matrix = means_matrix.permute(0,3,1,2,4)
            covariances_matrix = self.gmm_list_target.covariances[index ,:,:]
            covariances_matrix=covariances_matrix.permute(0,3,1,2,4)
            cdf_mask_target = self.gmm_list_target.updated_mask>0
            cdf_mask_target = cdf_mask_target[index ,:] #2 512 512 19 
            cdf_mask_target = cdf_mask_target.permute(0,3,1,2)
            data = (logits.unsqueeze(-1)-means_matrix)/ torch.sqrt(covariances_matrix)
            cdf_target = torch.sum(weights_matrix*approx_normal_cdf(data)*cdf_mask_target.unsqueeze(-1),-1) #2 19  512 512
            
            cdf_mask = cdf_mask*source_mask.unsqueeze(1)+cdf_mask_target*target_mask.unsqueeze(1)
            
            # 总的cdf
            cdf = cdf*source_mask.unsqueeze(1)+cdf_target*target_mask.unsqueeze(1)
            
            cdf_gt = torch.sum(cdf*pos_mask,1,keepdim=True)
            cdf_gt_mask = torch.sum(cdf_mask,1,keepdim=True)>0  
            
            # 获取 pos GMM 的参数
            means = self.pos_neg_gmm.means[0]
            covariances = self.pos_neg_gmm.covariances[0]
            weights =self.pos_neg_gmm.weights[0]
            # 创建连续的CDF函数表示
            left = means-3*torch.sqrt(covariances)
            right = means+3*torch.sqrt(covariances)
            x = torch.linspace(torch.min(left), torch.max(right), 1000,device=cls_score.device)  
            y  = x.unsqueeze(1)
            y  = (y -means)/torch.sqrt(covariances)
            y = torch.sum(weights*approx_normal_cdf(y ),1)
            # 创建插值函数
            logits_pos_gt = interp1d(y, x,cdf,self.ratio)  
            # 获取 neg GMM 的参数
            means = self.pos_neg_gmm.means[1]
            covariances = self.pos_neg_gmm.covariances[1]
            weights =self.pos_neg_gmm.weights[1]
            # 创建连续的CDF函数表示
            left = means-3*torch.sqrt(covariances)
            right = means+3*torch.sqrt(covariances)
            x = torch.linspace(torch.min(left), torch.max(right), 1000,device=cls_score.device)  
            y  = x.unsqueeze(1)
            y  = (y -means)/torch.sqrt(covariances)
            y = torch.sum(weights*approx_normal_cdf(y ),1)
            # 创建插值函数
            logits_neg_gt = interp1d(y, x, cdf,self.ratio)         
            logits_gt =  logits_pos_gt*pos_mask + logits_neg_gt*neg_mask    
            # 修正 logit
            loss_adjust_logit =  (logits_gt-logits)* cdf_mask*(label.unsqueeze(1)!=255)           

   
            
            
        if self.class_weight is not None:
            class_weight = cls_score.new_tensor(self.class_weight)
        else:
            class_weight = None
        loss_cls = self.loss_weight * self.cls_criterion(
            cls_score -self.t*loss_adjust_logit,
  
            label,
            weight,
            class_weight=class_weight,
            reduction=reduction,
            avg_factor=avg_factor,
            **kwargs)
        # print(loss_adjust_logit.item())
        if reg_score is not None:
            if weight is None:
                loss_reg = torch.abs((cdf_gt-reg_score)*cdf_gt_mask*(label.unsqueeze(1)!=255) )
            else:
                loss_reg = torch.abs((cdf_gt-reg_score)*cdf_gt_mask*(label.unsqueeze(1)!=255)*weight )
            loss_reg = loss_reg.mean()
        else:
            loss_reg = 0*loss_cls
        return [loss_cls,self.t2*loss_reg,
                [self.gmm_list.weights,self.gmm_list.means,self.gmm_list.covariances],[self.pos_neg_gmm.weights,self.pos_neg_gmm.means,self.pos_neg_gmm.covariances],
                [self.gmm_list_target.weights,self.gmm_list_target.means,self.gmm_list_target.covariances],[self.pos_neg_gmm_target.weights,self.pos_neg_gmm_target.means,self.pos_neg_gmm_target.covariances],
                cdf_gt.detach(),reg_score.detach()]

