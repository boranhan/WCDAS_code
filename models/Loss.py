import torch
import torch.nn as nn
import torch.nn.functional as F


class CosLoss(nn.Linear):
    r"""
    Cosine Loss
    """
    def __init__(self, in_features, out_features, bias=False):
        super(CosLoss, self).__init__(in_features, out_features, bias)
        self.s_ = torch.nn.Parameter(torch.Tensor([0.]), requires_grad=True)
        self.g = nn.parameter.Parameter(data=1.0*torch.ones(out_features), requires_grad=True)
        self.out_features=out_features

    def loss(self, Z, target, sample_per_class):
        l = F.cross_entropy(Z, target, weight=None, ignore_index=-100, reduction='mean')
        return l
        
    def forward(self, input, target, sample_per_class):
        cosine = F.linear(F.normalize(input, p=2, dim=1), F.normalize(self.weight, p=2, dim=1), None) # [N x out_features]
        s = F.softplus(self.s_).add(1.)#sample_per_class.log().to(Z.device))
        logit = cosine.mul(s)
        l = self.loss(logit, target, sample_per_class)
        return logit, l, torch.ones(1), s

class WCDAS(CosLoss):
    def __init__(self, in_features, out_features, bias=False, gamma = -1, s_trainable=True):
        super(WCDAS, self).__init__(in_features, out_features, bias)
        self.g = nn.parameter.Parameter(data=gamma*torch.ones(out_features), requires_grad=True) 
        self.out_features=out_features
        self.s_trainable=s_trainable

    def forward(self, input, target, sample_per_class):
        assert target is not None
        self.gamma =  1 / (1 + torch.exp (-self.g))
        cosine = F.linear(F.normalize(input, p=2, dim=1), F.normalize(self.weight, p=2, dim=1), None) 
        logit =  1/2/3.14*(1-self.gamma**2)/(1+self.gamma**2-2*self.gamma*cosine)
        s = F.softplus(self.s_).add(1.0)
        if self.s_trainable:
            logit = s * logit   
        else:
            logit = 250 * logit     
        l = self.loss(logit, target, sample_per_class)
        return logit, l, self.gamma, s
    
    def extra_repr(self):
        return super(cvMFLoss, self).extra_repr() + ', gamma={}'.format(self.gamma)

