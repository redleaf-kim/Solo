import torch
import torch.nn.functional as F


from src.utils import matrix_nms


class SoloPost:
    def __init__(self,
                 post_process_cfg,
                 num_classes=81,
                 strides=(4, 8, 16, 32, 64),
                 grid_num=[40,36,24,16,12],
                 ):

        self.post_process_cfg = post_process_cfg
        self.strides = strides
        self.grid_num = grid_num
        self.cate_out_channels = num_classes - 1


    def __call__(self, seg_preds, cate_preds, img_metas, rescale=None):
        assert len(seg_preds) == len(cate_preds)
        num_levels = len(cate_preds)
        featmap_size = seg_preds[0].size()[-2:]

        result_list = []
        for img_id in range(len(img_metas)):
            cate_pred_list = [
                cate_preds[i][img_id].view(-1, self.cate_out_channels).detach().cpu() for i in range(num_levels)
            ]
            seg_pred_list = [
                seg_preds[i][img_id].detach().cpu() for i in range(num_levels)
            ]
            img_shape = img_metas[img_id]['img_shape']
            ori_shape = img_metas[img_id]['ori_shape']

            cate_pred_list = torch.cat(cate_pred_list, dim=0)
            seg_pred_list = torch.cat(seg_pred_list, dim=0)

            result = self.get_seg_single(cate_pred_list, seg_pred_list,
                                         featmap_size, img_shape, ori_shape,
                                         self.post_process_cfg)
            result_list.append(result)
        return result_list


    def get_seg_single(self,
                       cate_preds,
                       seg_preds,
                       featmap_size,
                       img_shape,
                       ori_shape,
                       post_process_cfg):
        assert len(cate_preds) == len(seg_preds)

        # overall info.
        h, w = img_shape
        upsampled_size_out = (featmap_size[0] * 4, featmap_size[1] * 4)

        # process.
        inds = (cate_preds > post_process_cfg.score_thr)
        # category scores.
        cate_scores = cate_preds[inds]
        if len(cate_scores) == 0:
            return None
        # category labels.
        inds = inds.nonzero()
        cate_labels = inds[:, 1]

        # strides.
        size_trans = cate_labels.new_tensor(self.grid_num).pow(2).cumsum(0)
        strides = cate_scores.new_ones(size_trans[-1])
        n_stage = len(self.grid_num)
        strides[:size_trans[0]] *= self.strides[0]
        for ind_ in range(1, n_stage):
            strides[size_trans[ind_ - 1]:size_trans[ind_]] *= self.strides[ind_]
        strides = strides[inds[:, 0]]

        # masks.
        seg_preds = seg_preds[inds[:, 0]]
        seg_masks = seg_preds > post_process_cfg.mask_thr
        sum_masks = seg_masks.sum((1, 2)).float()

        # filter.
        keep = sum_masks > strides
        if keep.sum() == 0:
            return None

        seg_masks = seg_masks[keep, ...]
        seg_preds = seg_preds[keep, ...]
        sum_masks = sum_masks[keep]
        cate_scores = cate_scores[keep]
        cate_labels = cate_labels[keep]

        # maskness.
        seg_scores = (seg_preds * seg_masks.float()).sum((1, 2)) / sum_masks
        cate_scores *= seg_scores

        # sort and keep top nms_pre
        sort_inds = torch.argsort(cate_scores, descending=True)
        if len(sort_inds) > post_process_cfg.nms_pre:
            sort_inds = sort_inds[:post_process_cfg.nms_pre]
        seg_masks = seg_masks[sort_inds, :, :]
        seg_preds = seg_preds[sort_inds, :, :]
        sum_masks = sum_masks[sort_inds]
        cate_scores = cate_scores[sort_inds]
        cate_labels = cate_labels[sort_inds]

        # Matrix NMS
        # cate_scores_v1 = matrix_nms_v1(seg_masks, cate_labels, cate_scores,
        #                          kernel=post_process_cfg.kernel, sigma=post_process_cfg.sigma, sum_masks=sum_masks)
        cate_scores = matrix_nms(seg_masks, cate_labels, cate_scores,
                                 kernel=post_process_cfg.kernel, sigma=post_process_cfg.sigma)

        # filter.
        keep = cate_scores >= post_process_cfg.update_thr
        if keep.sum() == 0:
            return None
        seg_preds = seg_preds[keep, :, :]
        cate_scores = cate_scores[keep]
        cate_labels = cate_labels[keep]

        # sort and keep top_k
        sort_inds = torch.argsort(cate_scores, descending=True)
        if len(sort_inds) > post_process_cfg.max_per_img:
            sort_inds = sort_inds[:post_process_cfg.max_per_img]
        seg_preds = seg_preds[sort_inds, :, :]
        cate_scores = cate_scores[sort_inds]
        cate_labels = cate_labels[sort_inds]

        seg_preds = F.interpolate(seg_preds.unsqueeze(0),
                                  size=upsampled_size_out,
                                  mode='bilinear')[:, :, :h, :w]
        seg_masks = F.interpolate(seg_preds,
                                  size=ori_shape[:2],
                                  mode='bilinear').squeeze(0)
        seg_masks = seg_masks > post_process_cfg.mask_thr
        return seg_masks, cate_labels, cate_scores



class DSoloPost:
    def __init__(self,
                 post_process_cfg,
                 num_classes=81,
                 strides=(4, 8, 16, 32, 64),
                 grid_num=[40,36,24,16,12],
                 ):

        self.post_process_cfg = post_process_cfg
        self.strides = strides
        self.grid_num = grid_num
        self.cate_out_channels = num_classes


    def __call__(self, seg_x_preds, seg_y_preds, cate_preds, img_metas, rescale=None):
        assert len(seg_x_preds) == len(cate_preds)
        num_levels = len(cate_preds)
        featmap_size = seg_x_preds[0].size()[-2:]

        result_list = []
        for img_id in range(len(img_metas)):
            cate_pred_list = [
                cate_preds[i][img_id].view(-1, self.cate_out_channels).detach().cpu() for i in range(num_levels)
            ]
            seg_x_pred_list = [
                seg_x_preds[i][img_id].detach().cpu() for i in range(num_levels)
            ]

            seg_y_pred_list = [
                seg_y_preds[i][img_id].detach().cpu() for i in range(num_levels)
            ]

            img_shape = img_metas[img_id]['img_shape']
            ori_shape = img_metas[img_id]['ori_shape']

            cate_pred_list = torch.cat(cate_pred_list, dim=0)
            seg_x_pred_list = torch.cat(seg_x_pred_list, dim=0)
            seg_y_pred_list = torch.cat(seg_y_pred_list, dim=0)

            result = self.get_seg_single(cate_pred_list, seg_x_pred_list, seg_y_pred_list,
                                         featmap_size, img_shape, ori_shape,
                                         self.post_process_cfg)
            result_list.append(result)
        return result_list


    def get_seg_single(self,
                       cate_preds,
                       seg_x_preds,
                       seg_y_preds,
                       featmap_size,
                       img_shape,
                       ori_shape,
                       post_process_cfg):
        assert len(seg_x_preds) == len(seg_y_preds)

        # overall info.
        h, w = img_shape
        upsampled_size_out = (featmap_size[0] * 4, featmap_size[1] * 4)

        # trans trans_diff.
        trans_size = torch.Tensor(self.grid_num).pow(2).cumsum(0).long()
        trans_diff = torch.ones(trans_size[-1].item(), device=cate_preds.device).long()
        num_grids = torch.ones(trans_size[-1].item(), device=cate_preds.device).long()
        seg_size = torch.Tensor(self.grid_num).cumsum(0).long()
        seg_diff = torch.ones(trans_size[-1].item(), device=cate_preds.device).long()
        strides = torch.ones(trans_size[-1].item(), device=cate_preds.device)

        n_stage = len(self.grid_num)
        trans_diff[:trans_size[0]] *= 0
        seg_diff[:trans_size[0]] *= 0
        num_grids[:trans_size[0]] *= self.grid_num[0]
        strides[:trans_size[0]] *= self.strides[0]

        for ind_ in range(1, n_stage):
            trans_diff[trans_size[ind_ - 1]:trans_size[ind_]] *= trans_size[ind_ - 1]
            seg_diff[trans_size[ind_ - 1]:trans_size[ind_]] *= seg_size[ind_ - 1]
            num_grids[trans_size[ind_ - 1]:trans_size[ind_]] *= self.grid_num[ind_]
            strides[trans_size[ind_ - 1]:trans_size[ind_]] *= self.strides[ind_]

        # process.
        inds = (cate_preds > post_process_cfg.score_thr)
        cate_scores = cate_preds[inds]

        inds = inds.nonzero()
        trans_diff = torch.index_select(trans_diff, dim=0, index=inds[:, 0])
        seg_diff = torch.index_select(seg_diff, dim=0, index=inds[:, 0])
        num_grids = torch.index_select(num_grids, dim=0, index=inds[:, 0])
        strides = torch.index_select(strides, dim=0, index=inds[:, 0])

        y_inds = (inds[:, 0] - trans_diff) // num_grids
        x_inds = (inds[:, 0] - trans_diff) % num_grids
        y_inds += seg_diff
        x_inds += seg_diff

        cate_labels = inds[:, 1]
        seg_masks_soft = seg_x_preds[x_inds, ...] * seg_y_preds[y_inds, ...]
        seg_masks = seg_masks_soft > post_process_cfg.mask_thr
        sum_masks = seg_masks.sum((1, 2)).float()
        keep = sum_masks > strides

        seg_masks_soft = seg_masks_soft[keep, ...]
        seg_masks = seg_masks[keep, ...]
        cate_scores = cate_scores[keep]
        sum_masks = sum_masks[keep]
        cate_labels = cate_labels[keep]
        # maskness
        seg_score = (seg_masks_soft * seg_masks.float()).sum((1, 2)) / sum_masks
        cate_scores *= seg_score

        if len(cate_scores) == 0:
            return None

        # sort and keep top nms_pre
        sort_inds = torch.argsort(cate_scores, descending=True)
        if len(sort_inds) > post_process_cfg.nms_pre:
            sort_inds = sort_inds[:post_process_cfg.nms_pre]
        seg_masks_soft = seg_masks_soft[sort_inds, :, :]
        seg_masks = seg_masks[sort_inds, :, :]
        cate_scores = cate_scores[sort_inds]
        sum_masks = sum_masks[sort_inds]
        cate_labels = cate_labels[sort_inds]

        # Matrix NMS
        cate_scores = matrix_nms(seg_masks, cate_labels, cate_scores,
                                 kernel=post_process_cfg.kernel, sigma=post_process_cfg.sigma)

        keep = cate_scores >= post_process_cfg.update_thr
        seg_masks_soft = seg_masks_soft[keep, :, :]
        cate_scores = cate_scores[keep]
        cate_labels = cate_labels[keep]
        # sort and keep top_k
        sort_inds = torch.argsort(cate_scores, descending=True)
        if len(sort_inds) > post_process_cfg.max_per_img:
            sort_inds = sort_inds[:post_process_cfg.max_per_img]
        seg_masks_soft = seg_masks_soft[sort_inds, :, :]
        cate_scores = cate_scores[sort_inds]
        cate_labels = cate_labels[sort_inds]

        seg_masks_soft = F.interpolate(seg_masks_soft.unsqueeze(0),
                                    size=upsampled_size_out,
                                    mode='bilinear')[:, :, :h, :w]
        seg_masks = F.interpolate(seg_masks_soft,
                               size=ori_shape[:2],
                               mode='bilinear').squeeze(0)
        seg_masks = seg_masks > post_process_cfg.mask_thr
        return seg_masks, cate_labels, cate_scores