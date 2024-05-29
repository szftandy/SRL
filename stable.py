import torch
import torch.nn as nn
from torch import optim
from torch.autograd import Variable
from torch.nn.utils import clip_grad_norm_
from torch.distributions import Categorical
import math
import numpy as np

class DWR(nn.Module):
    def __init__(self, args, device=None):
        super().__init__()
        self.order = args.order
        self.num_steps = args.decorr_steps
        self.lr = args.lr_dwr
        self.tol = args.tolerance
        self.device = device

    def decorr_loss(self, X, weight):
        n, p = X.size()
        balance_loss = 0
        for a in range(1, self.order+1):
            for b in range(1, self.order+1):
                cov_mat = self.weighted_cov(X**a, X**b, W = weight**2/n) if a!=b else self.weighted_cov(X**a, W=weight**2/n)
                cov_mat = cov_mat**2
                balance_loss += torch.sum(torch.sqrt(torch.sum(cov_mat, dim=1)-torch.diag(cov_mat)+ 1e-10))
        loss_weight_sum = (torch.sum(weight*weight) - n)**2
        loss_weight_l2 = torch.sum((weight*weight)**2)
        loss = 2000.0 / p*balance_loss + 0.5*loss_weight_sum + 0.00005*loss_weight_l2
        return loss

    def weighted_cov(self, X, Y=None, W=None):
        X_bar = torch.matmul(X.T, W)
        if Y is None:
            return torch.matmul(X.T, W*X) - torch.matmul(X_bar, X_bar.T)
        else:
            Y_bar = torch.matmul(Y.T, W)
            return torch.matmul(X.T, W*Y) - torch.matmul(X_bar, Y_bar.T)
        
    def forward(self, X, epoch):
        X = X.clone().detach()
        weight = torch.ones(X.size(0), 1, device=self.device, requires_grad=True)
        optimizer = optim.Adam([weight,], lr=self.lr)
        for _ in range(self.num_steps):
            optimizer.zero_grad()
            loss = self.decorr_loss(X, weight)
            loss.backward()
            optimizer.step()
        weight = (weight**2); weight /= weight.sum()
        return weight.detach()

class StableNet(nn.Module):
    def __init__(self, args, device):
        super().__init__()
        self.device = device
        self.epoch_stable = args.epoch_stable
        self.lr_stable = args.lr_stable
        self.num_f = args.n_feature
        self.lambda_decay_rate = args.lambda_decay_rate
        self.lambda_decay_epoch = args.lambda_decay_epoch
        self.lambda_p = args.lambda_p
        self.decay_pow = args.decay_pow
        self.min_lambda_times = args.min_lambda_times
        self.softmax = nn.Softmax(0)
        self.register_buffer('pre_features', torch.zeros(args.n_feature, args.emb_size))
        self.register_buffer('pre_weight', torch.ones(args.n_feature, 1))

    def RFF(self, x, sigma=1):
        n, r = x.size(); x = x.view(n, r, 1)
        w = 1/sigma*(torch.randn(size=(self.num_f, 1), device=self.device))
        b = 2*np.pi*torch.rand(size=(r, self.num_f), device=self.device).repeat((n, 1, 1))
        mid = torch.matmul(x, w.t()) + b
        mid -= mid.min(dim=1, keepdim=True)[0]
        mid /= mid.max(dim=1, keepdim=True)[0]
        mid *= np.pi/2.0
        return math.sqrt(2.0/self.num_f)*(torch.cos(mid)+torch.sin(mid))

    def loss(self, x, w):
        w = w.view(-1, 1)
        loss = Variable(torch.FloatTensor([0]).cuda())
        for i in range(x.size()[-1]):
            x_ = x[:, :, i]
            cov = torch.matmul((w*x_).t(), x_)
            e = torch.sum(w*x_, dim=0).view(-1, 1)
            cov_matrix = torch.pow(cov-torch.matmul(e, e.t()), 2)
            loss += torch.sum(cov_matrix) - torch.trace(cov_matrix)
        return loss

    def forward(self, cfeatures, epoch):
        weight = Variable(torch.ones(cfeatures.size()[0], 1).to(self.device), requires_grad=True)
        cfeaturec = Variable(cfeatures.detach().clone()).to(self.device)
        all_feature = torch.cat([cfeaturec, self.pre_features.detach()], dim=0)
        optimizerbl = torch.optim.SGD([weight], lr=self.lr_stable, momentum=0.9)

        for _ in range(self.epoch_stable):
            all_weight = torch.cat((weight, self.pre_weight.detach()), dim=0)
            optimizerbl.zero_grad()
            features = self.RFF(all_feature)
            lossb = self.loss(features, self.softmax(all_weight))
            lossp = self.softmax(weight).pow(self.decay_pow).sum()
            lambdap = self.lambda_p*max((self.lambda_decay_rate**(epoch//self.lambda_decay_epoch)),
                                        self.min_lambda_times)
            lossg = lossb/lambdap + lossp
            lossg.backward(retain_graph=True)
            optimizerbl.step()

        if epoch <= 10:
            pre_features = (self.pre_features*epoch+cfeatures) / (epoch+1)
            pre_weight = (self.pre_weight*epoch+weight) / (epoch+1)
        elif cfeatures.size()[0] < self.pre_features.size()[0]:
            pre_features[:cfeatures.size()[0]] = self.pre_features[:cfeatures.size()[0]]*self.presave_ratio + cfeatures*(1-self.presave_ratio)
            pre_weight[:cfeatures.size()[0]] = self.pre_weight[:cfeatures.size()[0]]*self.presave_ratio + weight*(1-self.presave_ratio)
        else:
            pre_features = self.pre_features*self.presave_ratio + cfeatures*(1-self.presave_ratio)
            pre_weight = self.pre_weight*self.presave_ratio + weight*(1-self.presave_ratio)
        self.pre_features.data.copy_(pre_features)
        self.pre_weight.data.copy_(pre_weight)
        return self.softmax(weight)