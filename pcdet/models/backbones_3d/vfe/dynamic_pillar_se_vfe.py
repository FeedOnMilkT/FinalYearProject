import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import torch_scatter
except Exception as e:
    # If torch_scatter is not installed, we can still use the normal scatter
    pass

from .vfe_template import VFETemplate
from ...model_utils.attention_utils import SEAttention, SESparse3D

class PFNLayerV2(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 use_norm=True,
                 last_layer=False):
        super().__init__()
        
        self.last_vfe = last_layer
        self.use_norm = use_norm
        if not self.last_vfe:
            out_channels = out_channels // 2

        if self.use_norm:
            self.linear = nn.Linear(in_channels, out_channels, bias=False)
            self.norm = nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01)
        else:
            self.linear = nn.Linear(in_channels, out_channels, bias=True)
        
        self.relu = nn.ReLU()

    def forward(self, inputs, unq_inv):

        x = self.linear(inputs)
        x = self.norm(x) if self.use_norm else x
        x = self.relu(x)
        x_max = torch_scatter.scatter_max(x, unq_inv, dim=0)[0]

        if self.last_vfe:
            return x_max
        else:
            x_concatenated = torch.cat([x, x_max[unq_inv, :]], dim=1)
            return x_concatenated


class DeepSEDynamicPillarVFE(VFETemplate):
    def __init__(self,
                 model_cfg,
                 num_point_features,
                 voxel_size,
                 grid_size, 
                 point_cloud_range,
                 **kwargs):
        super().__init__(model_cfg=model_cfg)

        self.use_norm = model_cfg.USE_NORM
        self.with_distance = model_cfg.WITH_DISTANCE
        self.use_absolute_xyz = self.model_cfg.USE_ABSLOTE_XYZ
        num_point_features += 6 if self.use_absolute_xyz else 3
        if self.with_distance:
            num_point_features += 1

        self.num_filters = self.model_cfg.NUM_FILTERS
        assert len(self.num_filters) > 0
        num_filters = [num_point_features] + list(self.num_filters)

        self.use_attention = self.model_cfg.get('USE_ATTENTION', True)
        self.se_reduction = self.model_cfg.get('SE_REDUCTION', 16)

        # Multi-scale attention
        self.multi_scale_attention = self.model_cfg.get('MULTI_SCALE_ATTENTION', False)

        # Region adaptive attention
        # self.region_adaptive = self.model_cfg.get('REGION_ADAPTIVE', False)
        # self.near_threshold = self.model_cfg.get('NEAR_THRESHOLD', 20.0)
        # self.far_threshold = self.model_cfg.get('FAR_THRESHOLD', 50.0)

        # Density aware attention
        self.density_aware = self.model_cfg.get('DENSITY_AWARE', False)

        # Dynamic voxel size adjustment
        self.dynamic_voxel_size_adjustment = self.model_cfg.get('DYNAMIC_VOXEL_SIZE_ADJUSTMENT', False)
        self.voxel_size_range = self.model_cfg.get('VOXEL_SIZE_RANGE', [0.8, 1.2])

        pfn_layers = []
        for i in range(len(num_filters) - 1):
            in_filters = num_filters[i]
            out_filters = num_filters[i + 1]
            pfn_layers.append(
                PFNLayerV2(in_filters, out_filters, self.use_norm, last_layer=(i >= len(num_filters) - 2))
            )
        self.pfn_layers = nn.ModuleList(pfn_layers)


        if self.multi_scale_attention and self.use_attention:
            se_modules = []
            for i in range(len(num_filters) - 1):
                out_filters = num_filters[i + 1]
                if i < len(num_filters) - 2: 
                    out_filters = out_filters
                se_modules.append(SEAttention(out_filters, self.se_reduction))
            self.se_modules = nn.ModuleList(se_modules)
        elif self.use_attention:
            self.se_module = SEAttention(self.num_filters[-1], self.se_reduction)           

        if self.region_adaptive and self.use_attention:
            self.near_se = SEAttention(self.num_filters[-1], self.se_reduction // 2)  # Near distance use smaller reduction
            self.mid_se = SEAttention(self.num_filters[-1], self.se_reduction)
            self.far_se = SEAttention(self.num_filters[-1], self.se_reduction * 2)  # Far distance use larger reduction

        # Density aware attention
        if self.density_aware:
            self.density_encoder = nn.Sequential(
                nn.Linear(1, 32),
                nn.ReLU(),
                nn.Linear(32, self.num_filters[-1]),
                nn.Sigmoid()
            )
            
        # Dynamic voxel size adjustment
        if self.dynamic_voxel_size_adjustment:
            self.voxel_size_predictor = nn.Sequential(
                nn.Linear(self.num_filters[-1], 64),
                nn.ReLU(),
                nn.Linear(64, 3),  # x, y, z scaling factor
                nn.Sigmoid() 
            )

        self.voxel_x = voxel_size[0]
        self.voxel_y = voxel_size[1]
        self.voxel_z = voxel_size[2]
        self.x_offset = self.voxel_x / 2 + point_cloud_range[0]
        self.y_offset = self.voxel_y / 2 + point_cloud_range[1]
        self.z_offset = self.voxel_z / 2 + point_cloud_range[2]
        
        self.scale_xy = grid_size[0] * grid_size[1]
        self.scale_y = grid_size[1]
        
        self.grid_size = torch.tensor(grid_size).cuda()
        self.voxel_size = torch.tensor(voxel_size).cuda()
        self.point_cloud_range = torch.tensor(point_cloud_range).cuda()

    def get_output_feature_dim(self):
        return self.num_filters[-1]
        
    # def get_region_masks(self, points_xyz, unq_inv):

    #def apply_region_adaptive_se(self, features, points_xyz, unq_inv):

    def apply_density_aware_se(self, features, unq_cnt):
        pillar_density = unq_cnt.float().unsqueeze(1)
        normalized_density = pillar_density / pillar_density.max()

        se_weights = self.density_encoder(normalized_density)
        features = features * se_weights

        return features
    
    def predict_voxel_size(self, features):

        global_feat = features.mean(dim=0, keepdim=True)
        scale_factors = self.voxel_size_predictor(global_feat)

        min_scale, max_scale = self.voxel_size_range
        scale_factors = min_scale + scale_factors * (max_scale - min_scale)

        return scale_factors.squeeze(0)

    def forward(self, batch_dict):
        points = batch_dict['points'] # (batch_idx, x, y, z, i, e)

        points_coords = torch.floor(
            (points[:, [1, 2]] - self.point_cloud_range[[0, 1]]) / self.voxel_size[[0, 1]]).int()
        mask = ((points_coords >= 0) & (points_coords < self.grid_size[[0, 1]])).all(dim=1)
        points = points[mask]
        points_coords = points_coords[mask]
        points_xyz = points[:, [1, 2, 3]].contiguous()

        merge_coords = points[:, 0].int() * self.scale_xy + \
                      points_coords[:, 0] * self.scale_y + \
                      points_coords[:, 1]

        unq_coords, unq_inv, unq_cnt = torch.unique(merge_coords, return_inverse=True, return_counts=True, dim=0)

        if self.dynamic_voxel_size_adjustment and 'pillar_features' in batch_dict:
            # Adjust the voxel size based on the previous features
            prev_features = batch_dict['pillar_features']
            voxel_scale_factors = self.predict_voxel_size(prev_features)
            
            adjusted_voxel_size = self.voxel_size * voxel_scale_factors
            points_coords = torch.floor(
                (points[:, [1, 2]] - self.point_cloud_range[[0, 1]]) / adjusted_voxel_size[[0, 1]]).int()                                                    

            # Re-calculate the unique coordinates
            mask = ((points_coords >= 0) & (points_coords < self.grid_size[[0, 1]])).all(dim=1)
            points = points[mask]
            points_coords = points_coords[mask]
            points_xyz = points[:, [1, 2, 3]].contiguous()

            # Re-calculate the merge coordinates
            merge_coords = points[:, 0].int() * self.scale_xy + \
                          points_coords[:, 0] * self.scale_y + \
                          points_coords[:, 1]                            
            
            unq_coords, unq_inv, unq_cnt = torch.unique(merge_coords, return_inverse=True, return_counts=True, dim=0)
            
            batch_dict['adjusted_voxel_size'] = adjusted_voxel_size            

        points_mean = torch_scatter.scatter_mean(points_xyz, unq_inv, dim=0)
        f_cluster = points_xyz - points_mean[unq_inv, :]

        f_center = torch.zeros_like(points_xyz)
        f_center[:, 0] = points_xyz[:, 0] - (points_coords[:, 0].to(points_xyz.dtype) * self.voxel_x + self.x_offset)
        f_center[:, 1] = points_xyz[:, 1] - (points_coords[:, 1].to(points_xyz.dtype) * self.voxel_y + self.y_offset)
        f_center[:, 2] = points_xyz[:, 2] - self.z_offset

        if self.use_absolute_xyz:
            features = [points[:, 1:], f_cluster, f_center]
        else:
            features = [points[:, 4:], f_cluster, f_center]
        
        if self.with_distance:
            points_dist = torch.norm(points[:, 1:4], 2, dim=1, keepdim=True)
            features.append(points_dist)
        features = torch.cat(features, dim=-1)

        if self.multi_scale_attention and self.use_attention:
            for i, pfn in enumerate(self.pfn_layers):
                features = pfn(features, unq_inv)
                # Add SE module in each PFN layer
                features = self.se_modules[i](features)
        else:
            # Normal PFN layers
            for pfn in self.pfn_layers:
                features = pfn(features, unq_inv)

        unq_coords = unq_coords.int()
        voxel_coords = torch.stack((unq_coords // self.scale_xy,
                                  (unq_coords % self.scale_xy) // self.scale_y,
                                  unq_coords % self.scale_y,
                                  torch.zeros(unq_coords.shape[0]).to(unq_coords.device).int()
                                  ), dim=1)
        voxel_coords = voxel_coords[:, [0, 3, 2, 1]]

        batch_dict['voxel_features'] = batch_dict['pillar_features'] = features
        batch_dict['voxel_coords'] = voxel_coords
        
        return batch_dict