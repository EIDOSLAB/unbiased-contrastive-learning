import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision import models as resnet_big
from models import resnet_small

model_dict = {
    # Imagenet variants
    'resnet18': [resnet_big.resnet18, 512],
    'resnet34': [resnet_big.resnet34, 512],
    'resnet50': [resnet_big.resnet50, 2048],
    'resnet101': [resnet_big.resnet101, 2048],

    # Cifar variants
    'resnet32': [resnet_small.resnet32, 64],
    'resnet32x4': [lambda: resnet_small.resnet32(pool_k=4), 256]
}


class SupConResNet(nn.Module):
    """backbone + projection head"""
    def __init__(self, name='resnet18', head='mlp', feat_dim=128, num_classes=10,
                 train_on_head=True):
        super().__init__()

        model_fun, dim_in = model_dict[name]
        feat_dim = min(dim_in, feat_dim)
        self.feat_dim = feat_dim
        
        self.encoder = model_fun()

        if hasattr(self.encoder, 'fc'):
            self.encoder.fc = nn.Identity()

        if head == 'linear':
            self.head = nn.Linear(dim_in, feat_dim)
        elif head == 'mlp':
            self.head = nn.Sequential(
                nn.Linear(dim_in, dim_in),
                nn.ReLU(inplace=True),
                nn.Linear(dim_in, feat_dim)
            )
        else:
            raise NotImplementedError(
                'head not supported: {}'.format(head))

        self.fc = nn.Linear(dim_in, num_classes)
        if train_on_head:
            self.head = nn.Identity()
            # self.fc = nn.Linear(feat_dim, num_classes)
        
        self.train_on_head = train_on_head

    def forward(self, x):
        feat = self.encoder(x)
        proj = F.normalize(self.head(feat), dim=1)
        
        if self.train_on_head:
            logits = self.fc(proj)
        else:
            logits = self.fc(feat)
        
        return proj, feat, logits


class CEResNet(nn.Module):
    """encoder + classifier"""
    def __init__(self, name='resnet50', num_classes=10, normalize=False):
        super().__init__()
        model_fun, dim_in = model_dict[name]
        self.encoder = model_fun()
        if hasattr(self.encoder, 'fc'):
            self.encoder.fc = nn.Identity()
        self.fc = nn.Linear(dim_in, num_classes)
        self.normalize = normalize

    def forward(self, x):
        feats = self.encoder(x)
        if self.normalize:
            feats = F.normalize(feats, dim=1)
        return self.fc(feats), feats