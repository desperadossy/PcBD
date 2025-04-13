import torch
from torch.nn import Linear, Module, ModuleList, Identity, ReLU, Parameter, Sequential
from .build import MODELS
from .funcs import score_extract
from extensions.chamfer_dist import ChamferDistanceL1, ChamferDistanceL1_PM
from extensions.emd.emd_module import emdModule

class DenseEdgeConv(Module):

    def __init__(self, in_channels, num_layers, layer_out_dim, knn=16, aggr='max', activation='relu'):
        super().__init__()
        self.in_channels = in_channels
        self.knn = knn
        assert num_layers > 2
        self.num_layers = num_layers
        self.layer_out_dim = layer_out_dim
        
        # Densely Connected Layers
        self.layer_first = FullyConnected(3*in_channels, layer_out_dim, bias=True, activation=activation)
        self.layer_last = FullyConnected(in_channels + (num_layers - 1) * layer_out_dim, layer_out_dim, bias=True, activation=None)
        self.layers = ModuleList()
        for i in range(1, num_layers-1):
            self.layers.append(FullyConnected(in_channels + i * layer_out_dim, layer_out_dim, bias=True, activation=activation))

        self.aggr = Aggregator(aggr)

    @property
    def out_channels(self):
        return self.in_channels + self.num_layers * self.layer_out_dim

    def get_edge_feature(self, x, knn_idx):
        """
        :param  x:          (B, N, d)
        :param  knn_idx:    (B, N, K)
        :return (B, N, K, 2*d)
        """
        knn_feat = group(x, knn_idx)   # B * N * K * d
        x_tiled = x.unsqueeze(-2).expand_as(knn_feat)
        edge_feat = torch.cat([x_tiled, knn_feat, knn_feat - x_tiled], dim=3)
        return edge_feat

    def forward(self, x, pos):
        """
        :param  x:  (B, N, d)
        :return (B, N, d+L*c)
        """
        knn_idx = get_knn_idx(pos, pos, k=self.knn, offset=1)

        # First Layer
        edge_feat = self.get_edge_feature(x, knn_idx)
        y = torch.cat([
            self.layer_first(edge_feat),              # (B, N, K, c)
            x.unsqueeze(-2).repeat(1, 1, self.knn, 1) # (B, N, K, d)
        ], dim=-1)  # (B, N, K, d+c)

        # Intermediate Layers
        for layer in self.layers:
            y = torch.cat([
                layer(y),           # (B, N, K, c)
                y,                  # (B, N, K, c+d)
            ], dim=-1)  # (B, N, K, d+c+...)
        
        # Last Layer
        y = torch.cat([
            self.layer_last(y), # (B, N, K, c)
            y                   # (B, N, K, d+(L-1)*c)
        ], dim=-1)  # (B, N, K, d+L*c)

        # Pooling
        y = self.aggr(y, dim=-2)
        
        return y


class GPool(Module):

    def __init__(self, n, dim, use_mlp=False, mlp_activation='relu'):
        super().__init__()
        self.use_mlp = use_mlp
        if use_mlp:
            self.pre = Sequential(
                FullyConnected(dim, dim // 2, bias=True, activation=mlp_activation),
                FullyConnected(dim // 2, dim // 4, bias=True, activation=mlp_activation),
            )
            self.p = Linear(dim // 4, 1, bias=True)
        else:
            self.p = Linear(dim, 1, bias=True)
        self.n = n

    def forward(self, pos, x):
        # pos       : B * N * 3
        # x         : B * N * Fin
        batchsize = x.size(0)
        if self.n < 1:
            k = int(x.size(1) * self.n)
        else:
            k = self.n

        if self.use_mlp:
            y = self.pre(x)
        else:
            y = x

        y = (self.p(y) / torch.norm(self.p.weight, p='fro')).squeeze(-1)  # B * N

        top_idx = torch.argsort(y, dim=1, descending=True)[:, 0:k]  # B * k 
        y = torch.gather(y, dim=1, index=top_idx)  # B * k
        y = torch.sigmoid(y)

        pos = torch.gather(pos, dim=1, index=top_idx.unsqueeze(-1).expand(batchsize, k, 3))
        x = torch.gather(x, dim=1, index=top_idx.unsqueeze(-1).expand(batchsize, k, x.size(-1)))
        x = x * y.unsqueeze(-1).expand_as(x)

        return top_idx, pos, x


class RandomPool(Module):

    def __init__(self, n):
        super().__init__()
        self.n = n

    def get_choice(self, batch, num_points):
        if self.n < 1:
            n = int(num_points * self.n)
        else:
            n = self.n
        choice = np.arange(0, num_points)
        np.random.shuffle(choice)
        choice = torch.from_numpy(choice[:n]).long()

        return choice.unsqueeze(0).repeat(batch, 1)

    def forward(self, pos, x):
        B, N, _ = pos.size()
        idx = self.get_choice(B, N).to(device=pos.device)     # (B, K)

        pos = torch.gather(pos, dim=1, index=idx.unsqueeze(-1).repeat(1, 1, 3))
        x = torch.gather(x, dim=1, index=idx.unsqueeze(-1).repeat(1, 1, x.size(-1)))

        return idx, pos, x


class Aggregator(torch.nn.Module):

    def __init__(self, oper):
        super().__init__()
        assert oper in ('mean', 'sum', 'max')
        self.oper = oper

    def forward(self, x, dim=2):
        if self.oper == 'mean':
            return x.mean(dim=dim, keepdim=False)
        elif self.oper == 'sum':
            return x.sum(dim=dim, keepdim=False)
        elif self.oper == 'max':
            ret, _ = x.max(dim=dim, keepdim=False)
            return ret


class FullyConnected(torch.nn.Module):

    def __init__(self, in_features, out_features, bias=True, activation=None):
        super().__init__()

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

        
def normalize_point_cloud(pc, center=None, scale=None):
    """
    :param  pc: (B, N, 3)
    :return (B, N, 3)
    """
    if center is None:
        center = torch.mean(pc, dim=-2, keepdim=True).expand_as(pc)
    if scale is None:
        scale, _ = torch.max(pc.reshape(pc.size(0), -1).abs(), dim=1, keepdim=True)
        scale = scale.unsqueeze(-1).expand_as(pc)
    norm = (pc - center) / scale
    return norm, center, scale


def denormalize_point_cloud(pc, center, scale):
    """
    :param  pc: (B, N, 3)
    :return (B, N, 3)
    """
    return pc * scale + center


def get_knn_idx_dist(pos:torch.FloatTensor, query:torch.FloatTensor, k, offset=0):
    """
    :param  pos:     (B, N, F)
    :param  query:   (B, M, F)
    :return knn_idx: (B, M, k)
    """
    B, N, F = tuple(pos.size())
    M = query.size(1)

    pos = pos.unsqueeze(1).expand(B, M, N, F)
    query  = query.unsqueeze(2).expand(B, M, N, F)   # B * M * N * F
    dist = torch.sum((pos - query) ** 2, dim=3, keepdim=False)   # B * M * N
    knn_idx = torch.argsort(dist, dim=2)[:, :, offset:k+offset]   # B * M * k
    knn_dist = torch.gather(dist, dim=2, index=knn_idx)           # B * M * k

    return knn_idx, knn_dist
    

def get_knn_idx(pos, query, k, offset=0):
    """
    :param  pos:     (B, N, F)
    :param  query:   (B, M, F)
    :return knn_idx: (B, M, k)
    """
    knn_idx, _ = get_knn_idx_dist(pos=pos, query=query, k=k, offset=offset)

    return knn_idx


def group(x:torch.FloatTensor, idx:torch.LongTensor):
    """
    :param  x:      (B, N, F)
    :param  idx:    (B, M, k)
    :return (B, M, k, F)
    """
    B, N, F = tuple(x.size())
    _, M, k = tuple(idx.size())

    x = x.unsqueeze(1).expand(B, M, N, F)
    idx = idx.unsqueeze(3).expand(B, M, k, F)

    return torch.gather(x, dim=2, index=idx)


def gather(x:torch.FloatTensor, idx:torch.LongTensor):
    """
    :param  x:      (B, N, F)
    :param  idx:    (B, M)
    :return (B, M, F)
    """
    # x       : B * N * F
    # idx     : B * M
    # returns : B * M * F
    B, N, F = tuple(x.size())
    _, M    = tuple(idx.size())

    idx = idx.unsqueeze(2).expand(B, M, F)

    return torch.gather(x, dim=1, index=idx)


def feature_interp(k, feat, pos, pos_new, avg_dist, feat_new=None, avg_feat_diff=None):
    """
    :param  feat:     (B, N, F)
    :param  pos:      (B, N, 3)
    :param  pos_new:  (B, M, 3)
    :param  feat_new: (B, M, F)
    :return (B, M, F)
    """
    knn_idx = get_knn_idx(pos, pos_new, k=k, offset=0)
    pos_grouped = group(pos, idx=knn_idx)    # (B, M, k, 3)
    feat_grouped = group(feat, idx=knn_idx)  # (B, M, k, F)

    d_pos = ((pos_grouped - pos_new.unsqueeze(-2).expand_as(pos_grouped)) ** 2).sum(dim=-1)     # (B, M, k)
    weight = - d_pos / (avg_dist ** 2)

    if feat_new is not None:
        d_feat = ((feat_grouped - feat_new.unsqueeze(-2).expand_as(feat_grouped)) ** 2).sum(dim=-1) # (B, M, k)
        weight = weight - d_feat / (avg_feat_diff ** 2)

    weight = weight.softmax(dim=-1)   # (B, M, k)
    return (feat_grouped * weight.unsqueeze(-1).expand_as(feat_grouped)).sum(dim=-2)


def get_1d_mesh(steps, start=-0.2, end=0.2):
    return torch.linspace(start=start, end=end, steps=steps).unsqueeze(-1)

def get_2d_mesh(steps, start=-0.2, end=0.2):
    mesh_1d = get_1d_mesh(steps=steps, start=start, end=end).flatten()
    return torch.cartesian_prod(mesh_1d, mesh_1d)

def get_mesh(dim, steps, start=-0.2, end=0.2):
    assert dim in (1, 2)
    if dim == 1:
        return get_1d_mesh(steps, start=start, end=end)
    elif dim == 2:
        return get_2d_mesh(steps, start=start, end=end)

def get_sample_points(dim, samples, num_points=1, num_batch=None, start=-0.3, end=0.3):
    length = end - start
    if num_batch is None:
        size = [samples * num_points, dim]
    else:
        size = [num_batch, samples * num_points, dim]
    return (torch.rand(size) * length) - (length / 2)


class FeatureExtraction(Module):

    def __init__(self, in_channels=3, dynamic_graph=True, conv_channels=24, num_convs=4, conv_num_layers=3, conv_layer_out_dim=12, conv_knn=16, conv_aggr='max', activation='relu'):
        super().__init__()
        self.in_channels = in_channels
        self.dynamic_graph = dynamic_graph
        self.num_convs = num_convs

        # Edge Convolution Units
        self.transforms = ModuleList()
        self.convs = ModuleList()
        for i in range(num_convs):
            if i == 0:
                trans = FullyConnected(in_channels, conv_channels, bias=True, activation=None)
            else:
                trans = FullyConnected(in_channels, conv_channels, bias=True, activation=activation)
            conv = DenseEdgeConv(conv_channels, num_layers=conv_num_layers, layer_out_dim=conv_layer_out_dim, knn=conv_knn, aggr=conv_aggr, activation=activation)
            self.transforms.append(trans)
            self.convs.append(conv)
            in_channels = conv.out_channels

    @property
    def out_channels(self):
        return self.convs[-1].out_channels

    def dynamic_graph_forward(self, x):
        for i in range(self.num_convs):
            x = self.transforms[i](x)
            x = self.convs[i](x, x)
        return x

    def static_graph_forward(self, pos):
        x = pos
        for i in range(self.num_convs):
            x = self.transforms[i](x)
            x = self.convs[i](x, pos)
        return x 

    def forward(self, x):
        if self.dynamic_graph:
            return self.dynamic_graph_forward(x)
        else:
            return self.static_graph_forward(x)
        

class Downsampling(Module):

    def __init__(self, feature_dim, ratio=0.5):
        super().__init__()
        self.pool = GPool(ratio, dim=feature_dim)

    def forward(self, pos, x):
        """
        :param  pos:    (B, N, 3)
        :param  x:      (B, N, d)
        :return (B, rN, d)
        """
        idx, pos, x = self.pool(pos, x)
        return idx, pos, x


class DownsampleAdjust(Module):
    def __init__(self, feature_dim, ratio=0.5, use_mlp=False, activation='relu', random_pool=False, pre_filter=True):
        super().__init__()
        self.pre_filter = pre_filter
        if random_pool:
            self.pool = RandomPool(ratio)
        else:
            self.pool = GPool(ratio, dim=feature_dim, use_mlp=use_mlp, mlp_activation=activation)
        self.mlp = Sequential(
            FullyConnected(feature_dim, feature_dim // 2, activation=activation),
            FullyConnected(feature_dim // 2, feature_dim // 4, activation=activation),
            FullyConnected(feature_dim // 4, 3, activation=None)
        )

    def forward(self, pos, x):
        """
        :param  pos:    (B, N, 3)
        :param  x:      (B, N, d)
        :return (B, rN, d)
        """
        idx, pos, x = self.pool(pos, x)
        if self.pre_filter:
            pos = pos + self.mlp(x)
        return idx, pos, x


class Upsampling(Module):

    def __init__(self, feature_dim, mesh_dim=1, mesh_steps=2, use_random_mesh=False, activation='relu'):
        super().__init__()
        self.mesh_dim = mesh_dim
        self.mesh_steps = mesh_steps
        self.mesh = Parameter(get_mesh(dim=mesh_dim, steps=mesh_steps), requires_grad=False)    # Regular mesh
        self.use_random_mesh = use_random_mesh
        if use_random_mesh:
            self.ratio = mesh_steps
            print('[INFO] Using random mesh.')
        else:
            self.ratio = mesh_steps ** mesh_dim


        self.folding = Sequential(
            FullyConnected(feature_dim+mesh_dim, 128, bias=True, activation=activation),
            FullyConnected(128, 128, bias=True, activation=activation),
            FullyConnected(128, 64, bias=True, activation=activation),
            FullyConnected(64, 3, bias=True, activation=None),
        )

    def forward(self, pos, x):
        """
        :param  pos:    (B, N, 3)
        :param  x:      (B, N, d)
        :return (B, rN, d)
        """
        batchsize, n_pts, _ = x.size()
        x_tiled = x.repeat(1, self.ratio, 1)

        if self.use_random_mesh:
            mesh = get_sample_points(dim=self.mesh_dim, samples=self.mesh_steps, num_points=n_pts, num_batch=batchsize).to(device=x.device)
            x_expanded = torch.cat([x_tiled, mesh], dim=-1)   # (B, rN, d+d_mesh)
        else:        
            mesh_tiled = self.mesh.unsqueeze(-1).repeat(1, 1, n_pts).transpose(1, 2).reshape(1, -1, self.mesh_dim).repeat(batchsize, 1, 1)
            x_expanded = torch.cat([x_tiled, mesh_tiled], dim=-1)   # (B, rN, d+d_mesh)

        residual = self.folding(x_expanded) # (B, rN, 3)

        upsampled = pos.repeat(1, self.ratio, 1) + residual
        return upsampled

@MODELS.register_module() 
class DMRDeNoise(Module):
    def __init__(self, config):
        super().__init__()
        self.feats = ModuleList()
        self.feat_dim = 0
        for knn in [16]:
            feat_unit = FeatureExtraction(dynamic_graph=True, conv_knn=knn, conv_channels=32, conv_layer_out_dim=24, activation='relu')
            self.feats.append(feat_unit)
            self.feat_dim += feat_unit.out_channels
        
        self.downsample = DownsampleAdjust(feature_dim=self.feat_dim, ratio=0.5, use_mlp=False, activation='relu', random_pool=False, pre_filter=True)
        self.upsample = Upsampling(feature_dim=self.feat_dim, mesh_dim=1, mesh_steps=2, use_random_mesh=False, activation='relu')
        self.build_loss_func()
    
    def build_loss_func(self):
        self.loss_func_CD = ChamferDistanceL1()
        self.loss_func_emd = emdModule()

    def get_loss(self, output, xyz, label, bound): 
        nlabel = label[:,:,1].unsqueeze(-1)
        blabel = label[:,:,0].unsqueeze(-1)
        cxyz = score_extract(xyz.permute(0,2,1).contiguous(), nlabel.permute(0,2,1).contiguous(), reserve_high=False).permute(0,2,1).contiguous()
        pred_bound = score_extract(output.permute(0,2,1).contiguous(), blabel.permute(0,2,1).contiguous(), reserve_high=True).permute(0,2,1).contiguous()
        denoise_loss = 1e3 * self.loss_func_CD(output, cxyz)
        bound_loss = 1e3 * self.loss_func_CD(pred_bound, bound)
        return denoise_loss, bound_loss

    def forward(self, pos):
        feats = []
        for feat_unit in self.feats:
            feats.append(feat_unit(pos))
        feat = torch.cat(feats, dim=-1)

        idx, pos, feat = self.downsample(pos, feat)
        pred = self.upsample(pos, feat)
        return pred

