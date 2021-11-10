# https://github.com/WXinlong/SOLO/blob/master/mmdet/core/post_processing/matrix_nms.py


import torch


def matrix_nms(masks, labels, scores, kernel='gaussian', sigma=0.5):
    """Matrix NMS for multi-class masks.
    Args:
        masks (Tensor): shape (n, h, w)
        labels (Tensor): shape (n), mask labels in descending order
        scores (Tensor): shape (n), mask scores in descending order
        kernel (str):  'linear' or 'gauss'
        sigma (float): std in gaussian method
    Returns:
        Tensor: cate_scores_update, tensors of shape (n)
    """

    N = len(labels)
    if N == 0:
        return []

    # reshape for computation: Nx(HxW)
    masks = masks.reshape(N, -1).float()
    # pre-compute the IoU matrix: NxN
    inter = torch.mm(masks, masks.T)
    areas = masks.sum(dim=1).expand(N, N)
    union = areas + areas.T - inter
    ious = (inter / union).triu(diagonal=1)

    # max IoU for each: NxN
    ious_cmax, _ = ious.max(0)
    ious_cmax = ious_cmax.expand(N, N).T

    # Matrix NMS, Eqn.(4): NxN
    if kernel == 'gaussian':
        decay = torch.exp(-(ious*ious - ious_cmax*ious_cmax) / sigma)
    else: # linear
        decay = (1 - ious) / ( 1 - ious_cmax)

    decay, _ = decay.min(dim=0)
    return scores * decay