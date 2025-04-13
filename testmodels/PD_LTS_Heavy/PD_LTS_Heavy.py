import torch
import torch.nn as nn
import numpy as np
from torch import Tensor
from typing import List
from enum import Enum
import torch.nn.functional as F
from pytorch3d.ops import knn_points
from pytorch3d.ops import knn_gather
import torch.distributed as dist
import math
import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
models_dir = os.path.join(current_dir, 'models')
sys.path.append(models_dir)

from layers import ActNorm
import layers.base as base_layers
import layers as layers

class Distribution:
    def log_prob(self, x: Tensor):
        raise NotImplementedError()
    def sample(self, shape, device):
        raise NotImplementedError()

class GaussianDistribution(Distribution):

    def log_prob(self, x: Tensor, means=None, logs=None):
        if means is None:
            means = torch.zeros_like(x)
        if logs is None:
            logs = torch.zeros_like(x)
        sldj = -0.5 * ((x - means) ** 2 / (2 * logs).exp() + np.log(2 * np.pi) + 2 * logs)
        sldj = sldj.flatten(1).sum(-1)
        return sldj

    def sample(self, shape, device):
        return torch.randn(shape, device=device)

def knn_group(x: Tensor, i: Tensor):
    """
    x: [B, N, C]
    i: [B, M, k]
    return: [B, M, k, C]
    """
    (B, N, C), (_, M, k) = x.shape, i.shape

    # approach 1
    # x = x.unsqueeze(1).expand(B, M, N, C)
    # i = i.unsqueeze(3).expand(B, M, k, C)
    # return torch.gather(x, dim=2, index=i)

    # approach 2 (save some gpu memory)
    idxb = torch.arange(B).view(-1, 1)
    i = i.reshape(B, M * k)
    y = x[idxb, i].view(B, M, k, C)  # [B, M * k, C]
    return y

# -----------------------------------------------------------------------------------------
def get_knn_idx(k: int, f: Tensor, q: Tensor=None, offset=None, return_features=False):
    """
    f: [B, N, C]
    q: [B, M, C]
    index of points in f: [B, M, k]
    """
    if offset is None:
        offset = 0
    if q is None:
        q = f

    (B, N, C), (_, M, _) = f.shape, q.shape

    _f = f.unsqueeze(1).expand(B, M, N, C)
    _q = q.unsqueeze(2).expand(B, M, N, C)

    dist = torch.sum((_f - _q) ** 2, dim=3, keepdim=False)  # [B, M, N]
    knn_idx = torch.argsort(dist, dim=2)[..., offset:k+offset]  # [B, M, k]

    if return_features is True:
        knn_f = knn_group(f, knn_idx)
        return knn_f, knn_idx
    else:
        return knn_idx


# -----------------------------------------------------------------------------------------
class FullyConnectedLayer(torch.nn.Module):

    def __init__(self, in_features, out_features, bias=True, activation=None):
        super(FullyConnectedLayer, self).__init__()

        self.linear = torch.nn.Linear(in_features, out_features, bias=bias)

        if activation is None:
            self.activation = torch.nn.Identity()
        elif activation == 'relu':
            self.activation = torch.nn.ReLU()
        elif activation == 'elu':
            self.activation = torch.nn.ELU(alpha=1.0)
        elif activation == 'lrelu':
            self.activation = torch.nn.LeakyReLU(0.1)
        else:
            raise ValueError()

    def forward(self, x):
        return self.activation(self.linear(x))




class noiseEdgeConv(nn.Module):
    def __init__(self, in_channel, hidden_channel, out_channel, bias=True):
        super(noiseEdgeConv, self).__init__()

        self.linear1 = nn.Linear(in_channel * 2, hidden_channel, bias=bias)
        self.linear2 = nn.Linear(hidden_channel, hidden_channel, bias=bias)
        self.linear3 = nn.Linear(in_channel, hidden_channel, bias=bias)
        self.linear4 = nn.Linear(hidden_channel, hidden_channel, bias=bias)
        self.linear5 = nn.Linear(hidden_channel, out_channel, bias=bias)

        nn.init.normal_(self.linear5.weight, mean=0.0, std=0.05)
        nn.init.zeros_(self.linear5.bias)

    def forward(self, f: Tensor, _c=None, knn_idx: Tensor = None):
        """
        f: [B, N, C]
        knn_idx: [B, N, k]
        return: [B, N, out]
        """
        if knn_idx is None:
            knn_idx = get_knn_idx(32, f)
        knn_feat = knn_gather(f, knn_idx)  # [B, M, k, C]

        f_tiled = f.unsqueeze(2).expand_as(knn_feat)  # [B, N, k, C]
        x = torch.cat([knn_feat, knn_feat - f_tiled], dim=-1)  # [B, N, k, C * 3]

        x = F.relu(self.linear1(x))  # [B, N, k, h]
        x = F.relu(self.linear2(x))  # [B, N, k, h]
        x, _ = torch.max(x, dim=2, keepdim=False)  # [B, N, h]
        f = F.relu(self.linear3(f))
        f = F.relu(self.linear4(f))
        x = x + f
        x = self.linear5(x)  # [B, N, out]
        return x

class PreConv(nn.Module):
    def __init__(self, in_channel, out_channel, bias=True):
        super(PreConv, self).__init__()
        in_channel = in_channel * 2
        self.conv = nn.Sequential(*[
            nn.Conv2d(in_channel, out_channel, kernel_size=[1, 1]),
            nn.BatchNorm2d(out_channel),
            nn.LeakyReLU(0.05, inplace=True),
        ])
    def forward(self, f: Tensor, knn_idx: Tensor = None):
        """
        f: [B, N, C]
        knn_idx: [B, N, k]
        return: [B, N, out]
        """
        if knn_idx is None:
            knn_idx = get_knn_idx(32, f)
        knn_feat = knn_gather(f, knn_idx)  # [B, M, k, 3]
        f_tiled = f.unsqueeze(2).expand_as(knn_feat)  # [B, N, k, 3]
        x = torch.cat([f_tiled, knn_feat - f_tiled], dim=-1)  # [B, N, k, 2 * 3]
        x = x.permute(0, 3, 1, 2)  # [B, 2 * 3, N, k]
        x = self.conv(x)
        x, _ = torch.max(x, dim=-1, keepdim=False)
        x = torch.transpose(x, 1, 2)
        return x




class EdgeConv(nn.Module):
    def __init__(self, in_channel, hidden_channel, out_channel, concat=True):
        super(EdgeConv, self).__init__()
        self.concat = concat
        if concat == False:
            hidden_channel = hidden_channel + 32
        self.convs = nn.ModuleList()
        conv_first = nn.Sequential(*[
            nn.Conv2d(in_channel * 2, hidden_channel, kernel_size=[1, 1]),
            nn.BatchNorm2d(hidden_channel),
            nn.LeakyReLU(0.05, inplace=True),
        ])
        self.convs.append(conv_first)
        conv = nn.Sequential(*[
            nn.Conv2d(hidden_channel, out_channel, kernel_size=[1, 1], bias=True),
            nn.BatchNorm2d(out_channel),
            nn.LeakyReLU(0.05, inplace=True),
        ])
        self.convs.append(conv)




    def forward(self, f: Tensor, knn_idx: Tensor = None):
        """
        f: [B, N, C]
        knn_idx: [B, N, k]
        return: [B, N, out]
        """
        # if self.convs[0][0].weight.grad != None:
        #     print(torch.max(self.convs[0][0].weight.grad))
        if knn_idx is None:
            knn_idx = get_knn_idx(32, f)
        knn_feat = knn_gather(f, knn_idx)  # [B, M, k, C]
        f_tiled = f.unsqueeze(2).expand_as(knn_feat)  # [B, N, k, C]
        x = torch.cat([f_tiled, knn_feat - f_tiled], dim=-1)  # [B, N, k, C * 2]

        x = x.permute(0, 3, 1, 2)  # [B, C * 2, N, k]
        for i in range(len(self.convs)):
            x = self.convs[i](x)
        x, _ = torch.max(x, dim=-1, keepdim=False)
        x = torch.transpose(x, 1, 2)

        if self.concat:
            x = torch.cat([x, f], dim=-1)
        return x


class FeatMergeUnit(nn.Module):
    def __init__(self, in_channel, hidden_channel, out_channel):
        super(FeatMergeUnit, self).__init__()
        self.convs = nn.ModuleList()
        conv1 = nn.Sequential(*[
            nn.Conv1d(in_channel, hidden_channel, kernel_size=1),
            nn.BatchNorm1d(hidden_channel),
            nn.ReLU(inplace=True)
        ])
        self.convs.append(conv1)
        conv2 = nn.Sequential(*[
            nn.Conv1d(hidden_channel, out_channel, kernel_size=1),
            nn.BatchNorm1d(out_channel),
            nn.ReLU(inplace=True)
        ])
        self.convs.append(conv2)
    def forward(self, x: Tensor):
        x = torch.transpose(x, 1, 2)
        for i in range(len(self.convs)):
            x = self.convs[i](x)
        x = torch.transpose(x, 1, 2)
        return x


ACT_FNS = {
    'softplus': lambda b: nn.Softplus(),
    'elu': lambda b: nn.ELU(inplace=b),
    'swish': lambda b: base_layers.Swish(),
    'LeakyLSwish': lambda b: base_layers.LeakyLSwish(),
    'CLipSwish': lambda b: base_layers.CLipSwish(),
    'ALCLipSiLU': lambda b: base_layers.ALCLipSiLU(),
    'pila': lambda b: base_layers.Pila(),
    'CPila': lambda b: base_layers.CPila(),
    'lcube': lambda b: base_layers.LipschitzCube(),
    'identity': lambda b: base_layers.Identity(),
    'relu': lambda b: base_layers.MyReLU(inplace=b),
    'CReLU': lambda b: base_layers.CReLU(),
}


class PD_LTS_Heavy(nn.Module):

    def __init__(

        self,
        pc_channel=3,
        aug_channel=32,
        n_injector=10,
        num_neighbors=32,
        cut_channel=16,
        nflow_module=10,
        coeff=0.98,
        n_lipschitz_iters=None,
        sn_atol=1e-3,
        sn_rtol=1e-3,
        n_power_series=None,
        n_dist='geometric',
        n_samples=1,
        activation_fn='swish',
        n_exact_terms=2,
        neumann_grad=True,
        grad_in_forward=True,
        nhidden=2,
        idim=64,
        densenet=False,
        densenet_depth=3,
        densenet_growth=32,
        learnable_concat=True,
        lip_coeff=0.98,

    ):
        super(PD_LTS_Heavy, self).__init__()  # super 代表父类对象，在子类中访问父类成员
        self.pc_channel = pc_channel
        self.aug_channel = aug_channel
        self.n_injector = n_injector
        self.num_neighbors = num_neighbors
        self.cut_channel = cut_channel
        self.nflow_module = nflow_module
        self.coeff = coeff
        self.n_lipschitz_iters = n_lipschitz_iters
        self.sn_atol = sn_atol
        self.sn_rtol = sn_rtol
        self.n_power_series = n_power_series
        self.n_dist = n_dist
        self.n_samples = n_samples
        self.activation_fn = activation_fn
        self.n_exact_terms = n_exact_terms
        self.neumann_grad = neumann_grad
        self.grad_in_forward = grad_in_forward
        self.nhidden = nhidden
        self.idim = idim
        self.densenet = densenet
        self.densenet_depth = densenet_depth
        self.densenet_growth = densenet_growth
        self.learnable_concat = learnable_concat
        self.lip_coeff = lip_coeff

        self.dist = GaussianDistribution()
        self.noise_params = noiseEdgeConv(pc_channel, hidden_channel=32, out_channel=aug_channel)
        self.feat_Conv = nn.ModuleList()
        self.AdaptConv = nn.ModuleList()
        self.PreConv = PreConv(in_channel=self.pc_channel, out_channel=16)
        in_channelE = [16, 48, 80, 112, 144, 176, 96, 120, 144, 168]
        in_channelA = [48, 80, 112, 144, 176, 96, 120, 144, 168, 96]
        hidden_channel = 64
        out_channel = [32, 32, 32, 32, 32, 96, 24, 24, 24, 96]
        for i in range(self.n_injector):
            concat = False if i == 5 or i == 9 else True
            self.feat_Conv.append(
                EdgeConv(in_channelE[i], hidden_channel=hidden_channel, out_channel=out_channel[i], concat=concat))
            self.AdaptConv.append(FeatMergeUnit(in_channel=in_channelA[i], hidden_channel=hidden_channel,
                                                out_channel=self.pc_channel + self.aug_channel))


        flow_assemblies = []
        for i in range(self.nflow_module):
            flow = FlowAssembly(
                id=i,
                nflow_module=self.nflow_module,
                channel=self.pc_channel + self.aug_channel,
                coeff=self.coeff,
                n_lipschitz_iters=self.n_lipschitz_iters,
                sn_atol=self.sn_atol,
                sn_rtol=self.sn_rtol,
                n_power_series=self.n_power_series,
                n_dist=self.n_dist,
                n_samples=self.n_samples,
                activation_fn=self.activation_fn,
                n_exact_terms=self.n_exact_terms,
                neumann_grad=self.neumann_grad,
                grad_in_forward=self.grad_in_forward,
                nhidden=self.nhidden,
                idim=self.idim,
                densenet=self.densenet,
                densenet_depth=self.densenet_depth,
                densenet_growth=self.densenet_growth,
                learnable_concat=self.learnable_concat,
                lip_coeff=self.lip_coeff,
            )
            flow_assemblies.append(flow)
        self.flow_assemblies = nn.ModuleList(flow_assemblies)

        
        self.channel_mask = nn.Parameter(torch.ones((1, 1, self.pc_channel + self.aug_channel)),
                                             requires_grad=False)
        self.channel_mask[:, :, -self.cut_channel:] = 0.0


    def f(self, x: Tensor, inj_f: List[Tensor]):
        B, N, _ = x.shape
        log_det_J = torch.zeros((B,), device=x.device)  # 同样要注意和原数据一个设备
        for i in range(self.nflow_module):
            if i < self.n_injector:
                x = x + inj_f[i]


            x = self.flow_assemblies[i].forward(x)

        return x, log_det_J

    def g(self, z: Tensor, inj_f: List[Tensor]):
        for i in reversed(range(self.nflow_module)):
            z = self.flow_assemblies[i].inverse(z)
            if i < self.n_injector:
                z = z - inj_f[i]

        return z

    def log_prob(self, xyz: Tensor, inj_f: List[Tensor]):

        aug_feat = self.unit_coupling(xyz)
        x = torch.cat([xyz, aug_feat], dim=-1)  # [B, N, 3 + C]
        z, flow_ldj = self.f(x, inj_f)
        logp = 0
        return z, logp



    def sample(self, z: Tensor, inj_f: List[Tensor]):
        full_x = self.g(z, inj_f)
        clean_x = full_x[..., :self.pc_channel]  # [B, N, 3]
        return clean_x

    def forward(self, x: Tensor, y: Tensor = None):
        p = x
        inj_f = self.feat_extract(p)
        z, ldj = self.log_prob(x, inj_f)

        loss_denoise = torch.tensor(0.0, dtype=torch.float32, device=x.device)  # 同样要注意和原数据一个设备
        z[:, :, -self.cut_channel:] = 0
        predict_z = z
        predict_x = self.sample(predict_z, inj_f)

        return predict_x

    def unit_coupling(self, xyz):
        B, N, _ = xyz.shape
        _, knn_idx, _ = knn_points(xyz, xyz, K=self.num_neighbors, return_nn=False, return_sorted=False)
        f = self.noise_params(xyz, knn_idx)
        return f

    def feat_extract(self, xyz: Tensor):
        cs = []
        _, knn_idx, _ = knn_points(xyz, xyz, K=self.num_neighbors, return_nn=False, return_sorted=False)
        f = self.PreConv(xyz, knn_idx)
        for i in range(self.n_injector):
            f = self.feat_Conv[i](f, knn_idx)
            inj_f = self.AdaptConv[i](f)
            cs.append(inj_f)
        return cs

    def nll_loss(self, pts_shape, sldj):
        # ll = sldj - np.log(self.k) * torch.prod(pts_shape[1:])
        # ll = torch.nan_to_num(sldj, nan=1e3)
        ll = sldj

        nll = -torch.mean(ll)

        return nll

    def denoise(self, noisy_pc: Tensor):
        clean_pc, _, _ = self(noisy_pc)
        return clean_pc

    def init_as_trained_state(self):
        """Set the network to initialized state, needed for evaluation(significant performance impact)"""
        for i in range(self.nflow_module):
            self.flow_assemblies[i].chain[1].is_inited = True
            self.flow_assemblies[i].chain[3].is_inited = True


class FlowAssembly(layers.SequentialFlow):

    def __init__(
            self,
            id,
            nflow_module,
            channel,
            coeff=0.9,
            n_lipschitz_iters=None,
            sn_atol=None,
            sn_rtol=None,
            n_power_series=5,
            n_dist='geometric',
            n_samples=1,
            activation_fn='elu',
            n_exact_terms=0,
            neumann_grad=True,
            grad_in_forward=False,
            nhidden=2,
            idim=64,
            densenet=False,
            densenet_depth=3,
            densenet_growth=32,
            learnable_concat=False,
            lip_coeff=0.98,

    ):
        chain = []

        def _quadratic_layer(channel):
            return layers.InvertibleLinear(channel)

        def _actnorm(channel):
            return ActNorm(channel)

        def _lipschitz_layer():
            return base_layers.get_linear

        def _iMonotoneBlock(preact=False):
            return layers.iMonotoneBlock(
                FCNet(
                    preact=preact,
                    channel=channel,
                    lipschitz_layer=_lipschitz_layer(),
                    coeff=coeff,
                    n_iterations=n_lipschitz_iters,
                    activation_fn=activation_fn,
                    sn_atol=sn_atol,
                    sn_rtol=sn_rtol,
                    nhidden=nhidden,
                    idim=idim,
                    densenet=densenet,
                    densenet_depth=densenet_depth,
                    densenet_growth=densenet_growth,
                    learnable_concat=learnable_concat,
                    lip_coeff=lip_coeff,
                ),
                n_power_series=n_power_series,
                n_dist=n_dist,
                n_samples=n_samples,
                n_exact_terms=n_exact_terms,
                neumann_grad=neumann_grad,
                grad_in_forward=grad_in_forward,
            )
        chain.append(_iMonotoneBlock())
        chain.append(_actnorm(channel))
        chain.append(_iMonotoneBlock(preact=True))
        chain.append(_actnorm(channel))


        super(FlowAssembly, self).__init__(chain)

class FCNet(nn.Module):

    def __init__(
        self, preact, channel, lipschitz_layer, coeff, n_iterations, activation_fn, sn_atol, sn_rtol,
        nhidden=2, idim=64, densenet=False, densenet_depth=3, densenet_growth=32, learnable_concat=False, lip_coeff=0.98
    ):
        super(FCNet, self).__init__()
        nnet = []
        last_dim = channel
        if not densenet:
            if activation_fn in ['CLipSwish', 'CPila', 'ALCLipSiLU', 'CReLU']:
                idim_out = idim // 2
                last_dim_in = last_dim * 2
            else:
                idim_out = idim
                last_dim_in = last_dim
            if(preact):
                nnet.append(ACT_FNS[activation_fn](False))
            for i in range(nhidden):
                nnet.append(
                    lipschitz_layer(
                        last_dim_in, idim_out, coeff=coeff, n_iterations=n_iterations, domain=2, codomain=2,
                        atol=sn_atol, rtol=sn_rtol
                    )
                )
                nnet.append(ACT_FNS[activation_fn](True))
                last_dim_in = idim_out * 2 if activation_fn in ['CLipSwish', 'CPila', 'ALCLipSiLU', 'CReLU'] else idim_out
            nnet.append(
                lipschitz_layer(
                    last_dim_in, last_dim, coeff=coeff, n_iterations=n_iterations, domain=2, codomain=2,
                    atol=sn_atol, rtol=sn_rtol
                )
            )
        else:
            first_channels = 64

            nnet.append(
                lipschitz_layer(
                    channel, first_channels, coeff=coeff, n_iterations=n_iterations, domain=2, codomain=2,
                    atol=sn_atol, rtol=sn_rtol
                )
            )

            total_in_channels = first_channels

            for i in range(densenet_depth):
                part_net = []

                # Change growth size for CLipSwish:
                if activation_fn in ['CLipSwish', 'CPila', 'ALCLipSiLU', 'CReLU']:
                    output_channels = densenet_growth // 2
                else:
                    output_channels = densenet_growth

                part_net.append(
                    lipschitz_layer(
                        total_in_channels, output_channels, coeff=coeff, n_iterations=n_iterations, domain=2,
                        codomain=2, atol=sn_atol, rtol=sn_rtol
                    )
                )

                part_net.append(ACT_FNS[activation_fn](True))

                nnet.append(
                    layers.LipschitzDenseLayer(layers.ExtendedSequential(*part_net),
                                               learnable_concat,
                                               lip_coeff
                                               )
                )

                total_in_channels += densenet_growth

            nnet.append(
                lipschitz_layer(
                    total_in_channels, last_dim, coeff=coeff, n_iterations=n_iterations, domain=2, codomain=2,
                    atol=sn_atol, rtol=sn_rtol
                )
            )

        self.nnet = layers.ExtendedSequential(*nnet)

    def forward(self, x):
        y = self.nnet(x)
        return y

    def build_clone(self):
        class FCNetClone(nn.Module):
            def __init__(self, nnet):
                super(FCNetClone, self).__init__()
                self.nnet = nnet

            def forward(self, x):
                y = self.nnet(x)
                return y

        return FCNetClone(self.nnet.build_clone())

    def build_jvp_net(self, x):
        class FCNetJVP(nn.Module):
            def __init__(self, nnet):
                super(FCNetJVP, self).__init__()
                self.nnet = nnet

            def forward(self, v):
                jv = self.nnet(v)
                return jv

        nnet, y = self.nnet.build_jvp_net(x)
        return FCNetJVP(nnet), y
