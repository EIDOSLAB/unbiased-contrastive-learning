import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleConvNetEncoder(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        padding = kernel_size // 2
        layers = [
            nn.Conv2d(3, 16, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm2d(128) 
        ]
        self.relu = nn.ReLU()

        self.extracter = nn.Sequential(*layers)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        print(f'SimpleConvNet')

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.extracter(x)
        x = self.relu(x)
        x = self.avgpool(x)
        feats = torch.flatten(x, 1)
        return feats

class SupConSimpleConvNet(nn.Module):
    def __init__(self, head='mlp', feat_dim=128, num_classes=10,
                train_on_head=True):
        super().__init__()

        self.encoder = SimpleConvNetEncoder()

        dim_in = 128
        self.feat_dim = feat_dim

        if head == 'linear':
            self.head = nn.Linear(dim_in, feat_dim)
        elif head == 'mlp':
            self.head = nn.Sequential(
                nn.Linear(dim_in, dim_in),
                nn.ReLU(inplace=True),
                nn.Linear(dim_in, feat_dim)
            )
        else:
            raise NotImplementedError('head not supported: {}'.format(head))

        self.fc = nn.Linear(dim_in, num_classes)
        if train_on_head:
            self.head = nn.Identity()
            self.encoder.relu = nn.Tanh()
            # self.fc = nn.Linear(feat_dim, num_classes)

        self.train_on_head = train_on_head
        print(f'SupConSimplConvNet')

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        feat = self.encoder(x)
        proj = F.normalize(self.head(feat), dim=1)
        
        if self.train_on_head:
            logits = self.fc(proj)
        else:
            logits = self.fc(feat)
        
        return proj, feat, logits

class CESimpleConvNet(nn.Module):
    def __init__(self, num_classes=10, normalize=False):
        super().__init__()

        self.encoder = SimpleConvNetEncoder()
        self.fc = nn.Linear(128, num_classes)
        self.normalize = normalize
    
    def forward(self, x):
        feats = self.encoder(x)
        if self.normalize:
            feats = F.normalize(feats)
        return self.fc(feats), feats