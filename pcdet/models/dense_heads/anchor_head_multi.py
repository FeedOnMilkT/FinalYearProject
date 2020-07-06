import numpy as np
import torch.nn as nn
from .anchor_head_template import AnchorHeadTemplate
from ..backbones_2d import BaseBEVBackbone
import torch

class SingleHead(BaseBEVBackbone):
    def __init__(self, model_cfg, input_channels, num_class, num_anchors_per_location, code_size, encode_conv_cfg=None):
        super().__init__(encode_conv_cfg, input_channels)

        self.num_anchors_per_location = num_anchors_per_location
        self.num_class = num_class
        self.code_size = code_size
        self.model_cfg = model_cfg

        self.conv_cls = nn.Conv2d(
            input_channels, self.num_anchors_per_location * self.num_class,
            kernel_size=1
        )
        self.conv_box = nn.Conv2d(
            input_channels, self.num_anchors_per_location * self.code_size,
            kernel_size=1
        )

        if self.model_cfg.get('USE_DIRECTION_CLASSIFIER', None) is not None:
            self.conv_dir_cls = nn.Conv2d(
                input_channels,
                self.num_anchors_per_location * self.model_cfg.NUM_DIR_BINS,
                kernel_size=1
            )
        else:
            self.conv_dir_cls = None
        self.use_multihead = self.model_cfg.get('USE_MULTIHEAD', False)
        self.init_weights()

    def init_weights(self):
        pi = 0.01
        nn.init.constant_(self.conv_cls.bias, -np.log((1 - pi) / pi))

    def forward(self, spatial_features_2d):
        ret_dict = {}
        spatial_features_2d = super().forward({'spatial_features': spatial_features_2d})['spatial_features_2d']

        cls_preds = self.conv_cls(spatial_features_2d)
        box_preds = self.conv_box(spatial_features_2d)

        if not self.use_multihead:
            box_preds = box_preds.permute(0, 2, 3, 1).contiguous()
            cls_preds = cls_preds.permute(0, 2, 3, 1).contiguous()
        else:
            H, W = box_preds.shape[2:]
            batch_size = box_preds.shape[0]
            box_preds = box_preds.view(-1, self.num_anchors_per_location,
                                       self.code_size, H, W).permute(0, 1, 3, 4, 2).contiguous()
            cls_preds = cls_preds.view(-1, self.num_anchors_per_location,
                                       self.num_class, H, W).permute(0, 1, 3, 4, 2).contiguous()
            box_preds = box_preds.view(batch_size, -1, self.code_size)
            cls_preds = cls_preds.view(batch_size, -1, self.num_class)
        
        if self.conv_dir_cls is not None:
            dir_cls_preds = self.conv_dir_cls(spatial_features_2d)
            if self.use_multihead:
                dir_cls_preds = dir_cls_preds.view(
                    -1, self.num_anchors_per_location, self.model_cfg.NUM_DIR_BINS, H, W).permute(0, 1, 3, 4, 2).contiguous()
                dir_cls_preds = dir_cls_preds.view(batch_size, -1, self.model_cfg.NUM_DIR_BINS)
            else:
                dir_cls_preds = dir_cls_preds.permute(0, 2, 3, 1).contiguous()

        else:
            dir_cls_preds = None

        ret_dict['cls_preds'] = cls_preds
        ret_dict['box_preds'] = box_preds
        ret_dict['dir_cls_preds'] = dir_cls_preds

        return ret_dict

class AnchorHeadMulti(AnchorHeadTemplate):
    def __init__(self, model_cfg, input_channels, num_class, class_names, grid_size, point_cloud_range, predict_boxes_when_training=True):
        super().__init__(
            model_cfg=model_cfg, num_class=num_class, class_names=class_names, grid_size=grid_size, point_cloud_range=point_cloud_range, predict_boxes_when_training=predict_boxes_when_training
        )
        self.model_cfg = model_cfg
        self.seperate_multihead = self.model_cfg.get('SEPERATE_MULTIHEAD', False)
        
        shared_conv_num_filter = self.model_cfg.SHARED_CONV_NUM_FILTER
        self.shared_conv = nn.Sequential(
                nn.Conv2d(input_channels, shared_conv_num_filter, 3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(shared_conv_num_filter, eps=1e-3, momentum=0.01),
                nn.ReLU(),
            )
        self.make_multihead(shared_conv_num_filter)

    def make_multihead(self, input_channels):
        rpn_head_cfgs = self.model_cfg.RPN_HEAD_CFGS
        rpn_heads = []
        class_names = []
        for rpn_head_cfg in rpn_head_cfgs:
            class_names.extend(rpn_head_cfg['HEAD_CLS_NAME'])
        for rpn_head_cfg in rpn_head_cfgs:
            num_anchors_per_location = sum([self.num_anchors_per_location[class_names.index(head_cls)] for head_cls in rpn_head_cfg['HEAD_CLS_NAME']])
            rpn_head = SingleHead(self.model_cfg, input_channels, len(rpn_head_cfg['HEAD_CLS_NAME']) if self.seperate_multihead else self.num_class, num_anchors_per_location, self.box_coder.code_size, rpn_head_cfg)
            rpn_heads.append(rpn_head)
        self.rpn_heads = nn.ModuleList(rpn_heads)

    def forward(self, data_dict):
        spatial_features_2d = data_dict['spatial_features_2d']
        spatial_features_2d = self.shared_conv(spatial_features_2d)

        ret_dicts = []
        for rpn_head in self.rpn_heads:
            ret_dicts.append(rpn_head(spatial_features_2d))
        
        cls_preds = [ret_dict['cls_preds'] for ret_dict in ret_dicts]
        box_preds = [ret_dict['box_preds'] for ret_dict in ret_dicts]
        
        ret = {
            'cls_preds': cls_preds if self.seperate_multihead else torch.cat(cls_preds, dim=1),
            'box_preds': box_preds if self.seperate_multihead else torch.cat(box_preds, dim=1),
        }

        if self.model_cfg.get('USE_DIRECTION_CLASSIFIER', False):
            dir_cls_preds = [ret_dict['dir_cls_preds'] for ret_dict in ret_dicts]
            ret['dir_cls_preds'] = dir_cls_preds if self.seperate_multihead else torch.cat(dir_cls_preds, dim=1)
        else:
            dir_cls_preds = None
 
        self.forward_ret_dict.update(ret)
       
        if self.training:
            targets_dict = self.assign_targets(
                gt_boxes=data_dict['gt_boxes']
            )
            self.forward_ret_dict.update(targets_dict)
        else:
            batch_cls_preds, batch_box_preds = self.generate_predicted_boxes(
                batch_size=data_dict['batch_size'],
                cls_preds=cls_preds, box_preds=box_preds, dir_cls_preds=dir_cls_preds
            )
            data_dict['batch_cls_preds'] = batch_cls_preds
            data_dict['batch_box_preds'] = batch_box_preds
            data_dict['cls_preds_normalized'] = False

        return data_dict

    def get_cls_layer_loss(self):
        cls_preds = self.forward_ret_dict['cls_preds']
        box_cls_labels = self.forward_ret_dict['box_cls_labels']
        if not isinstance(cls_preds, list):
            cls_preds = [cls_preds]
        batch_size = int(cls_preds[0].shape[0])
        cared = box_cls_labels >= 0  # [N, num_anchors]
        positives = box_cls_labels > 0
        negatives = box_cls_labels == 0
        negative_cls_weights = negatives * 1.0
        cls_weights = (negative_cls_weights + 1.0 * positives).float()
        reg_weights = positives.float()
        if self.num_class == 1:
            # class agnostic
            box_cls_labels[positive] = 1
        pos_normalizer = positives.sum(1, keepdim=True).float()
        reg_weights /= torch.clamp(pos_normalizer, min=1.0)
        cls_weights /= torch.clamp(pos_normalizer, min=1.0)
        cls_targets = box_cls_labels * cared.type_as(box_cls_labels)
        one_hot_target = torch.zeros(
            *list(cls_targets.shape), cls_preds[0].shape[-1] + 1 if self.seperate_multihead else self.num_class + 1, dtype=cls_preds[0].dtype, device=cls_targets.device
            )
        one_hot_target.scatter_(-1, cls_targets.unsqueeze(dim=-1).long(), 1.0)
        one_hot_targets = one_hot_target[..., 1:]
        start_idx = 0
        cls_losses = 0
        for cls_pred in cls_preds:
            cls_pred = cls_pred.view(batch_size, -1, cls_pred.shape[-1])
            one_hot_target = one_hot_targets[:, start_idx:start_idx+cls_pred.shape[1]]
            cls_weight = cls_weights[:, start_idx:start_idx+cls_pred.shape[1]]
            cls_loss_src = self.cls_loss_func(cls_pred, one_hot_target, weights=cls_weight)  # [N, M]
            cls_loss = cls_loss_src.sum() / batch_size
            cls_loss = cls_loss * self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS['cls_weight']
            cls_losses += cls_loss
            start_idx += cls_pred.shape[1]
        tb_dict = {
            'rpn_loss_cls': cls_losses.item()
        }
        return cls_losses, tb_dict

    def get_box_reg_layer_loss(self):
        box_preds = self.forward_ret_dict['box_preds']
        box_dir_cls_preds = self.forward_ret_dict.get('dir_cls_preds', None)
        box_reg_targets = self.forward_ret_dict['box_reg_targets']
        box_cls_labels = self.forward_ret_dict['box_cls_labels']

        positives = box_cls_labels > 0
        reg_weights = positives.float()
        pos_normalizer = positives.sum(1, keepdim=True).float()
        reg_weights /= torch.clamp(pos_normalizer, min=1.0)

        if not isinstance(box_preds, list):
            box_preds = [box_preds]
        batch_size = int(box_preds[0].shape[0])

        if isinstance(self.anchors, list):
            if self.use_multihead:
                anchors = torch.cat(
                    [anchor.permute(3, 4, 0, 1, 2, 5).contiguous().view(-1, anchor.shape[-1]) for anchor in
                     self.anchors], dim=0)
            else:
                anchors = torch.cat(self.anchors, dim=-3)
        else:
            anchors = self.anchors
        anchors = anchors.view(1, -1, anchors.shape[-1]).repeat(batch_size, 1, 1)

        start_idx = 0
        box_losses = 0
        tb_dict = {}
        for idx, box_pred in enumerate(box_preds):
            box_pred = box_pred.view(batch_size, -1,
                                   box_pred.shape[-1] // self.num_anchors_per_location if not self.use_multihead else
                                   box_pred.shape[-1])
            box_reg_target = box_reg_targets[:, start_idx:start_idx+box_pred.shape[1]]
            reg_weight = reg_weights[:, start_idx:start_idx+box_pred.shape[1]]
            # sin(a - b) = sinacosb-cosasinb
            box_pred_sin, reg_target_sin = self.add_sin_difference(box_pred, box_reg_target)
            loc_loss_src = self.reg_loss_func(box_pred_sin, reg_target_sin, weights=reg_weight)  # [N, M]
            loc_loss = loc_loss_src.sum() / batch_size

            loc_loss = loc_loss * self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS['loc_weight']
            box_losses += loc_loss
            tb_dict['rpn_loss_loc'] = tb_dict.get('rpn_loss_loc', 0) + loc_loss

            if box_dir_cls_preds is not None:
                if not isinstance(box_dir_cls_preds, list):
                    box_dir_cls_preds = [box_dir_cls_preds]
                dir_targets = self.get_direction_target(
                    anchors, box_reg_targets,
                    dir_offset=self.model_cfg.DIR_OFFSET,
                    num_bins=self.model_cfg.NUM_DIR_BINS
                )
                box_dir_cls_pred = box_dir_cls_preds[idx]
                dir_logit = box_dir_cls_pred.view(batch_size, -1, self.model_cfg.NUM_DIR_BINS)
                weights = positives.type_as(dir_logit)
                weights /= torch.clamp(weights.sum(-1, keepdim=True), min=1.0)
                
                weight = weights[:, start_idx:start_idx+box_pred.shape[1]]
                dir_target = dir_targets[:, start_idx:start_idx+box_pred.shape[1]]
                dir_loss = self.dir_loss_func(dir_logit, dir_target, weights=weight)
                dir_loss = dir_loss.sum() / batch_size
                dir_loss = dir_loss * self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS['dir_weight']
                box_losses += dir_loss
                tb_dict['rpn_loss_dir'] = tb_dict.get('rpn_loss_dir', 0) + dir_loss.item()
            start_idx += box_pred.shape[1]
        return box_losses, tb_dict
