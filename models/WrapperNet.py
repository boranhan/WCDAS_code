import torch.nn as nn

from . import Loss

class WrapperNet(nn.Module):
    def __init__(self, model, num_classes, sample_per_class, loss='CosLoss', **kwargs):
        super(WrapperNet, self).__init__()
        self.feat = model
        self.sample_per_class = sample_per_class
        self.fc_loss = Loss.__dict__[loss](model.get_dim(), num_classes, **kwargs)
    
    def forward(self, x, target):
        x = self.feat(x)
        x, loss, gamma, s  = self.fc_loss(x, target, self.sample_per_class)
        return x, loss, gamma, s