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
            c = x.size(1)
            y = x.mean(dim = 0, keepdim = True) + 1e-5
            y = self.fc(y)
            y = torch.clamp(y, min=0.1, max=2.0)
            return x * y

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
    
           
    