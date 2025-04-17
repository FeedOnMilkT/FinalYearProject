import torch
import torch.nn as nn

from ...utils.spconv_utils import replace_feature

class SEAttention(nn.Module):
    # SEAttention block for dense feature
    def __init__(self, channels, reduction = 16):
        super(SEAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

        for m in self.fc.modules():
            if isinstance (m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    
    def forward(self, x):
        # Operate the 3-dimensional tensor
        # x: [B, C, N]
        if x.dim() == 3:
            b, c, _ = x.size()
            y = self.avg_pool(x).view(b, c) + 1e-5
            y = self.fc(y).view(b, c, 1)
            y = torch.clamp(y, min=0.1, max=2.0) 
            return x * y
        else:
            # Default PFN Output
            c = x.size(1)
            y = x.mean(dim = 0, keepdim = True) + 1e-5
            y = self.fc(y)
            y = torch.clamp(y, min=0.1, max=2.0)
            return x * y

# This module fit the pointpillars only
class ECAPFNLayer(nn.Module):
    def __init__(self, channels, k_size = 3):
        super(ECAPFNLayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.conv = nn.Conv1d(channels, channels, kernel_size = k_size, padding=(k_size - 1)//2, bias = False, groups = channels)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x here only has [num_voxels, channels]
        x = x.unsqueeze(0).transpose(1, 2) # 1 * channels * pillar_nums
        y = self.avg_pool(x) # Output 1 * channels_num * 1 tensor
        y = self.conv(y)
        y = self.sigmoid(y)
        # y = y.view(1, -1) # [1, channels]
        y = torch.clamp(y, min = 0.1, max = 2.0)
        x = x * y # 1 * channels * pillar_nums
        x = x.transpose(1, 2).squeeze(0) # channels * pillar_nums
        return x

# This module fit the pointpillars only as well
class CBMAPFNLayer(nn.Module):
    def __init__(self, channels, reduction = 16, k_size = 7):
        super(CBMAPFNLayer, self).__init__()
        # For Channel Attention
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)

        self.channelMLP = nn.Sequential(
            nn.Linear (channels, channels // reduction, bias = False),
            nn.ReLU(),
            nn.Linear (channels // reduction, channels, bias = False)

        )

        self.sigmoid_channel = nn.Sigmoid()

        # For Spatial Attention
        self.spatialMLP = nn.Sequential(
            nn.Conv1d(2, 1, kernel_size = k_size, padding = k_size // 2, bias = False),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x is pillar_nums * channels
        temp_x = x # For residual connection operation

        # Channel:
        x_trans = x.transpose(0, 1).unsqueeze(0) # 1 * channel * pillar_nums
        avg_out = self.avg_pool(x_trans).squeeze(-1) # 1 * channel
        max_out = self.max_pool(x_trans).squeeze(-1) # 1 * channel

        avg_out = self.channelMLP(avg_out)
        max_out = self.channelMLP(max_out)

        channel_attention_weight = self.sigmoid_channel(avg_out + max_out)
        # channel_attention_weight = torch.clamp(channel_attention_weight, min = 0.1, max = 2.0) # 1 * channel
        x_channel = x * channel_attention_weight

        # Spatial:
        x_spatial = x_channel.transpose(0, 1).unsqueeze(0) # 1 * channel * pillar_nums
        avg_map = torch.mean(x_spatial, dim = 1, keepdim = True) # 1 * 1 * pillar_nums
        max_map, _ = torch.max(x_spatial, dim = 1, keepdim = True)

        spa_input = torch.cat([avg_map, max_map], dim=1) # 1 * 2 * pillar_nums
        spatial_attention_weight = self.spatialMLP(spa_input).squeeze(0).transpose(0, 1) # pillar_nums * 1

        x_spatial = spatial_attention_weight

        out = x_channel * x_spatial
        out += temp_x

        return out







class SESparse3D(nn.Module):
    # SEAttention block for sparse feature
    def __init__(self, channels, reduction = 16):
        super(SESparse3D, self).__init__()
        # self.reduction = reduction
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )
        
        for m in self.fc.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')        

    def forward(self, x):
        # Operate the sparse tensor
        features = x.features # [N, C]
        y = features.mean(dim = 0, keepdim = True) + 1e-5 # [1, C]
        y = self.fc(y) # [1, C]
        y = torch.clamp(y, min=0.1, max=2.0)
        new_features = features * y # [N, C]

        out = replace_feature(x, new_features)

        return out

class SESparse2D(nn.Module): 
    def __init__(self, channels, reduction = 16):
        super(SESparse2D, self).__init__()
        # self.reduction = reduction
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )
        for m in self.fc.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    

    def forward(self, x):
        features = x.features

        y = features.mean(dim = 0, keepdim = True) + 1e-5
        y = self.fc(y)
        y = torch.clamp(y, min=0.1, max=2.0)
        new_features = features * y
        out = replace_feature(x, new_features)

        return out
    
class SE2D(nn.Module):
    def __init__(self, channels, reduction = 16):
        super(SE2D, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )
        for m in self.fc.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c) + 1e-5
        y = self.fc(y)
        y = torch.clamp(y, min=0.1, max=2.0)        
        y = y.view(b, c, 1, 1)
        return x * y.expand_as(x)
    
           
    