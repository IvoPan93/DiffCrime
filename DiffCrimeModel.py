import torch
import torch.nn as nn
from inspect import isfunction

from einops import rearrange
from torch import einsum


def exists(val):
    return val is not None


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


class GEGLU(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out * 2)

    def forward(self, x):
        x, gate = self.proj(x).chunk(2, dim=-1)
        return x * nn.functional.gelu(gate)


class FeedForward(nn.Module):
    def __init__(self, dim, dim_out=None, mult=4, glu=False, dropout=0.):
        super().__init__()
        inner_dim = int(dim * mult)
        dim_out = default(dim_out, dim)
        project_in = nn.Sequential(
            nn.Linear(dim, inner_dim),
            nn.GELU()
        ) if not glu else GEGLU(dim, inner_dim)

        self.net = nn.Sequential(
            project_in,
            nn.Dropout(dropout),
            nn.Linear(inner_dim, dim_out)
        )

    def forward(self, x):
        return self.net(x)


class TimeStepAwareModalityFusion(nn.Module):
    def __init__(self, query_dim, x_h_dim, heads=1, dim_head=256, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads

        self.scale = dim_head ** -0.5
        self.heads = heads

        self.t_transform = nn.Linear(query_dim, inner_dim)

        self.key_transform_x = nn.Linear(x_h_dim, inner_dim, bias=False)
        self.value_transform_x = nn.Linear(x_h_dim, inner_dim, bias=False)
        self.layerNorm_x = nn.LayerNorm(x_h_dim)

        self.key_transform_H = nn.Linear(x_h_dim, inner_dim, bias=False)
        self.value_transform_H = nn.Linear(x_h_dim, inner_dim, bias=False)
        self.layerNorm_H = nn.LayerNorm(x_h_dim)

        self.key_transform_S = nn.Linear(x_h_dim, inner_dim, bias=False)
        self.value_transform_S = nn.Linear(x_h_dim, inner_dim, bias=False)
        self.layerNorm_S = nn.LayerNorm(x_h_dim)

        self.key_transform_M = nn.Linear(x_h_dim, inner_dim, bias=False)
        self.value_transform_M = nn.Linear(x_h_dim, inner_dim, bias=False)
        self.layerNorm_M = nn.LayerNorm(x_h_dim)

        self.ff = FeedForward(inner_dim, mult=1, glu=True)
        self.layerNorm_result = nn.LayerNorm(inner_dim)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, x_h_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, h, s, m, t, imageWidth):
        query = self.t_transform(t).unsqueeze(-1).unsqueeze(-1).expand(-1, -1, imageWidth, imageWidth)
        query = rearrange(query, 'b c h w -> b (h w) c')

        x = rearrange(x, 'b c h w -> b (h w) c')
        h = rearrange(h, 'b c h w -> b (h w) c')
        s = rearrange(s, 'b c h w -> b (h w) c')
        m = rearrange(m, 'b c h w -> b (h w) c')

        x = self.layerNorm_x(x)
        h = self.layerNorm_H(h)
        s = self.layerNorm_S(s)
        m = self.layerNorm_M(m)

        k_x = self.key_transform_x(x)
        v_x = self.value_transform_x(x)

        k_h = self.key_transform_H(h)
        v_h = self.value_transform_H(h)

        k_s = self.key_transform_S(s)
        v_s = self.value_transform_S(s)

        k_m = self.key_transform_M(m)
        v_m = self.value_transform_M(m)

        sim_x = einsum('b i d, b j d -> b i j', query, k_x) * self.scale
        sim_h = einsum('b i d, b j d -> b i j', query, k_h) * self.scale
        sim_s = einsum('b i d, b j d -> b i j', query, k_s) * self.scale
        sim_m = einsum('b i d, b j d -> b i j', query, k_m) * self.scale

        combined_keys = torch.cat([sim_x, sim_h, sim_s, sim_m], dim=-1)
        combined_scores = combined_keys.softmax(dim=-1)

        split_size = combined_scores.size(-1) // 4
        attn_scores_x, attn_scores_H, attn_scores_S, attn_scores_M = torch.split(combined_scores, split_size, dim=-1)

        out_x = einsum('b i j, b j d -> b i d', attn_scores_x, v_x)
        out_h = einsum('b i j, b j d -> b i d', attn_scores_H, v_h)
        out_s = einsum('b i j, b j d -> b i d', attn_scores_S, v_s)
        out_m = einsum('b i j, b j d -> b i d', attn_scores_M, v_m)

        x = self.ff(self.layerNorm_result(out_x + out_h + out_s + out_m))
        x = self.to_out(x)
        return rearrange(x, 'b (h w) c -> b c h w', h=imageWidth)


class ResidualConvBlock(nn.Module):
    def __init__(
            self, in_channels: int, out_channels: int, is_res: bool = False
    ) -> None:
        super().__init__()
        '''
        standard ResNet style convolutional block
        '''
        self.same_channels = in_channels == out_channels
        self.is_res = is_res
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, 1, 1),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.is_res:
            x1 = self.conv1(x)
            x2 = self.conv2(x1)
            if self.same_channels:
                out = x + x2
            else:
                out = x1 + x2
            return out / 1.414
        else:
            x1 = self.conv1(x)
            x2 = self.conv2(x1)
            return x2


class BlendDownsampling1(nn.Module):
    def __init__(self, in_channels, out_channels, context_channels, his_channel, his_size):
        super(BlendDownsampling1, self).__init__()

        self.out_channels = out_channels
        self.timeEmb = EmbedFC(1, out_channels)
        self.conv1 = nn.Sequential(nn.BatchNorm2d(num_features=in_channels),
                                   nn.ReLU(),
                                   nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3,
                                             stride=2, padding=1))

        self.conEmbMap = nn.Sequential(
            nn.BatchNorm2d(num_features=context_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels=context_channels, out_channels=out_channels, kernel_size=3,
                      stride=2, padding=1))

        self.conEmbSatellite = nn.Sequential(
            nn.BatchNorm2d(num_features=context_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels=context_channels, out_channels=out_channels, kernel_size=3,
                      stride=2, padding=1))

        self.TAMF = TimeStepAwareModalityFusion(query_dim=1, x_h_dim=out_channels)

        self.conv2 = nn.Sequential(nn.BatchNorm2d(num_features=out_channels),
                                   nn.ReLU(),
                                   nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3,
                                             stride=1, padding=1))
        self.convInit = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=2,
                                  stride=2)

        self.hisDown = nn.Sequential(nn.BatchNorm2d(num_features=in_channels),
                                     nn.ReLU(),
                                     nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3,
                                               stride=2, padding=1),
                                     nn.BatchNorm2d(num_features=out_channels),
                                     nn.ReLU(),
                                     nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3,
                                               stride=1, padding=1)
                                     )
        self.w1 = nn.Parameter(torch.randn(his_channel, his_size, his_size))
        self.w2 = nn.Parameter(torch.randn(his_channel, his_size, his_size))
        self.b = nn.Parameter(torch.randn(his_channel, his_size, his_size))

    def forward(self, x, s, m, his, t):
        xInit = self.convInit(x)
        x = self.conv1(x)
        condMapEmb = self.conEmbMap(m)
        condSatelliteEmb = self.conEmbSatellite(s)
        his = self.hisDown(his)
        x = self.TAMF(x, his, condSatelliteEmb, condMapEmb, t, imageWidth=8) + x
        x = self.conv2(x)
        x = x + xInit

        feg = nn.Sigmoid()(torch.matmul(self.w1, x) + torch.matmul(self.w2, his) + self.b)
        return torch.multiply(x, feg) + torch.multiply(his, 1 - feg), his


class BlendDownsampling2(nn.Module):
    def __init__(self, in_channels, out_channels, context_channels, his_channel, his_size):
        super(BlendDownsampling2, self).__init__()

        self.out_channels = out_channels
        self.timeEmb = EmbedFC(1, out_channels)
        self.conv1 = nn.Sequential(nn.BatchNorm2d(num_features=in_channels),
                                   nn.ReLU(),
                                   nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3,
                                             stride=2, padding=1))

        self.conEmbMap = nn.Sequential(
            nn.BatchNorm2d(num_features=context_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels=context_channels, out_channels=context_channels, kernel_size=3,
                      stride=2, padding=1),
            nn.BatchNorm2d(num_features=context_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels=context_channels, out_channels=out_channels, kernel_size=3,
                      stride=2, padding=1)
        )

        self.conEmbSatellite = nn.Sequential(
            nn.BatchNorm2d(num_features=context_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels=context_channels, out_channels=context_channels, kernel_size=3,
                      stride=2, padding=1),
            nn.BatchNorm2d(num_features=context_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels=context_channels, out_channels=out_channels, kernel_size=3,
                      stride=2, padding=1)
        )

        self.TAMF = TimeStepAwareModalityFusion(query_dim=1, x_h_dim=out_channels)

        self.conv2 = nn.Sequential(nn.BatchNorm2d(num_features=out_channels),
                                   nn.ReLU(),
                                   nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3,
                                             stride=1, padding=1))
        self.convInit = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=2,
                                  stride=2)

        self.hisDown = nn.Sequential(nn.BatchNorm2d(num_features=in_channels),
                                     nn.ReLU(),
                                     nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3,
                                               stride=2, padding=1),
                                     nn.BatchNorm2d(num_features=out_channels),
                                     nn.ReLU(),
                                     nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3,
                                               stride=1, padding=1)
                                     )
        self.w1 = nn.Parameter(torch.randn(his_channel, his_size, his_size))
        self.w2 = nn.Parameter(torch.randn(his_channel, his_size, his_size))
        self.b = nn.Parameter(torch.randn(his_channel, his_size, his_size))

    def forward(self, x, s, m, his, t):
        xInit = self.convInit(x)
        x = self.conv1(x)
        condMapEmb = self.conEmbMap(m)
        condSatelliteEmb = self.conEmbSatellite(s)
        his = self.hisDown(his)
        x = self.TAMF(x, his, condSatelliteEmb, condMapEmb, t, imageWidth=4) + x
        x = self.conv2(x)
        x = x + xInit

        feg = nn.Sigmoid()(torch.matmul(self.w1, x) + torch.matmul(self.w2, his) + self.b)
        return torch.multiply(x, feg) + torch.multiply(his, 1 - feg), his


class BlendDownsampling3(nn.Module):
    def __init__(self, in_channels, out_channels, context_channels, his_channel, his_size):
        super(BlendDownsampling3, self).__init__()

        self.out_channels = out_channels
        self.timeEmb = EmbedFC(1, out_channels)
        self.conv1 = nn.Sequential(nn.BatchNorm2d(num_features=in_channels),
                                   nn.ReLU(),
                                   nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3,
                                             stride=2, padding=1))

        self.conEmbMap = nn.Sequential(
            nn.BatchNorm2d(num_features=context_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels=context_channels, out_channels=context_channels, kernel_size=3,
                      stride=2, padding=1),
            nn.BatchNorm2d(num_features=context_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels=context_channels, out_channels=context_channels, kernel_size=3,
                      stride=2, padding=1),
            nn.BatchNorm2d(num_features=context_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels=context_channels, out_channels=out_channels, kernel_size=3,
                      stride=2, padding=1)
        )

        self.conEmbSatellite = nn.Sequential(
            nn.BatchNorm2d(num_features=context_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels=context_channels, out_channels=context_channels, kernel_size=3,
                      stride=2, padding=1),
            nn.BatchNorm2d(num_features=context_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels=context_channels, out_channels=context_channels, kernel_size=3,
                      stride=2, padding=1),
            nn.BatchNorm2d(num_features=context_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels=context_channels, out_channels=out_channels, kernel_size=3,
                      stride=2, padding=1)
        )

        self.TAMF = TimeStepAwareModalityFusion(query_dim=1, x_h_dim=out_channels)

        self.conv2 = nn.Sequential(nn.BatchNorm2d(num_features=out_channels),
                                   nn.ReLU(),
                                   nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3,
                                             stride=1, padding=1))
        self.convInit = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=2,
                                  stride=2)

        self.hisDown = nn.Sequential(nn.BatchNorm2d(num_features=in_channels),
                                     nn.ReLU(),
                                     nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3,
                                               stride=2, padding=1),
                                     nn.BatchNorm2d(num_features=out_channels),
                                     nn.ReLU(),
                                     nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3,
                                               stride=1, padding=1)
                                     )
        self.w1 = nn.Parameter(torch.randn(his_channel, his_size, his_size))
        self.w2 = nn.Parameter(torch.randn(his_channel, his_size, his_size))
        self.b = nn.Parameter(torch.randn(his_channel, his_size, his_size))

    def forward(self, x, s, m, his, t):
        xInit = self.convInit(x)
        x = self.conv1(x)
        condMapEmb = self.conEmbMap(m)
        condSatelliteEmb = self.conEmbSatellite(s)
        his = self.hisDown(his)
        x = self.TAMF(x, his, condSatelliteEmb, condMapEmb, t, imageWidth=2) + x
        x = self.conv2(x)
        x = x + xInit

        feg = nn.Sigmoid()(torch.matmul(self.w1, x) + torch.matmul(self.w2, his) + self.b)
        return torch.multiply(x, feg) + torch.multiply(his, 1 - feg), his


class BlendDownsampling4(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(BlendDownsampling4, self).__init__()

        self.out_channels = out_channels
        self.conv1 = nn.Sequential(nn.BatchNorm2d(num_features=in_channels),
                                   nn.ReLU(),
                                   nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3,
                                             stride=2, padding=1))

        self.hisDown = nn.Sequential(nn.BatchNorm2d(num_features=in_channels),
                                     nn.ReLU(),
                                     nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3,
                                               stride=2, padding=1))

    def forward(self, x, his):
        x = self.conv1(x)
        his = self.hisDown(his)
        return x, his


class BlendUpsampling4(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(BlendUpsampling4, self).__init__()
        self.model = nn.Sequential(nn.ConvTranspose2d(in_channels, out_channels, 2, 2),
                                   )
        self.hisUp = nn.Sequential(nn.ConvTranspose2d(in_channels, out_channels, 2, 2),
                                   )

    def forward(self, x, his):
        x = self.model(x)
        his = self.hisUp(his)
        return x, his


class BlendUpsampling3(nn.Module):
    def __init__(self, in_channels, out_channels, context_channels, his_channel, his_size):
        super(BlendUpsampling3, self).__init__()
        self.out_channels = out_channels
        self.timeEmb = EmbedFC(1, out_channels)
        self.conv1 = nn.Sequential(nn.ConvTranspose2d(in_channels, out_channels, 2, 2),
                                   ResidualConvBlock(out_channels, out_channels),
                                   ResidualConvBlock(out_channels, out_channels))

        self.conEmbMap = nn.Sequential(
            nn.BatchNorm2d(num_features=context_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels=context_channels, out_channels=context_channels, kernel_size=3,
                      stride=2, padding=1),
            nn.BatchNorm2d(num_features=context_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels=context_channels, out_channels=out_channels, kernel_size=3,
                      stride=2, padding=1)
        )

        self.conEmbSatellite = nn.Sequential(
            nn.BatchNorm2d(num_features=context_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels=context_channels, out_channels=context_channels, kernel_size=3,
                      stride=2, padding=1),
            nn.BatchNorm2d(num_features=context_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels=context_channels, out_channels=out_channels, kernel_size=3,
                      stride=2, padding=1)
        )

        self.TAMF = TimeStepAwareModalityFusion(query_dim=1, x_h_dim=out_channels)

        self.conv2 = nn.Sequential(nn.BatchNorm2d(num_features=out_channels),
                                   nn.ReLU(),
                                   nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3,
                                             stride=1, padding=1))
        self.convInit = nn.ConvTranspose2d(in_channels, out_channels, 2, 2)

        self.hisUp = nn.Sequential(nn.ConvTranspose2d(in_channels, out_channels, 2, 2),
                                   ResidualConvBlock(out_channels, out_channels),
                                   ResidualConvBlock(out_channels, out_channels),
                                   nn.BatchNorm2d(num_features=out_channels),
                                   nn.ReLU(),
                                   nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3,
                                             stride=1, padding=1)
                                   )
        self.w1 = nn.Parameter(torch.randn(his_channel, his_size, his_size))
        self.w2 = nn.Parameter(torch.randn(his_channel, his_size, his_size))
        self.b = nn.Parameter(torch.randn(his_channel, his_size, his_size))

    def forward(self, x, skip, s, m, his, t):
        x = torch.cat((x, skip), 1)
        xInit = self.convInit(x)
        x = self.conv1(x)
        condMapEmb = self.conEmbMap(m)
        condSatelliteEmb = self.conEmbSatellite(s)
        his = self.hisUp(his)
        x = self.TAMF(x, his, condSatelliteEmb, condMapEmb, t, imageWidth=4) + x

        x = self.conv2(x)
        x = x + xInit

        feg = nn.Sigmoid()(torch.matmul(self.w1, x) + torch.matmul(self.w2, his) + self.b)
        return torch.multiply(x, feg) + torch.multiply(his, 1 - feg), his


class BlendUpsampling2(nn.Module):
    def __init__(self, in_channels, out_channels, context_channels, his_channel, his_size):
        super(BlendUpsampling2, self).__init__()
        self.out_channels = out_channels
        self.timeEmb = EmbedFC(1, out_channels)
        self.conv1 = nn.Sequential(nn.ConvTranspose2d(in_channels, out_channels, 2, 2),
                                   ResidualConvBlock(out_channels, out_channels),
                                   ResidualConvBlock(out_channels, out_channels))

        self.conEmbMap = nn.Sequential(
            nn.BatchNorm2d(num_features=context_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels=context_channels, out_channels=out_channels, kernel_size=3,
                      stride=2, padding=1)
        )
        self.conEmbSatellite = nn.Sequential(
            nn.BatchNorm2d(num_features=context_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels=context_channels, out_channels=out_channels, kernel_size=3,
                      stride=2, padding=1)
        )

        self.TAMF = TimeStepAwareModalityFusion(query_dim=1, x_h_dim=out_channels)

        self.conv2 = nn.Sequential(nn.BatchNorm2d(num_features=out_channels),
                                   nn.ReLU(),
                                   nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3,
                                             stride=1, padding=1))
        self.convInit = nn.ConvTranspose2d(in_channels, out_channels, 2, 2)

        self.hisUp = nn.Sequential(nn.ConvTranspose2d(in_channels, out_channels, 2, 2),
                                   ResidualConvBlock(out_channels, out_channels),
                                   ResidualConvBlock(out_channels, out_channels),
                                   nn.BatchNorm2d(num_features=out_channels),
                                   nn.ReLU(),
                                   nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3,
                                             stride=1, padding=1)
                                   )
        self.w1 = nn.Parameter(torch.randn(his_channel, his_size, his_size))
        self.w2 = nn.Parameter(torch.randn(his_channel, his_size, his_size))
        self.b = nn.Parameter(torch.randn(his_channel, his_size, his_size))

    def forward(self, x, skip, s, m, his, t):
        x = torch.cat((x, skip), 1)
        xInit = self.convInit(x)
        x = self.conv1(x)
        condMapEmb = self.conEmbMap(m)
        condSatelliteEmb = self.conEmbSatellite(s)
        his = self.hisUp(his)
        x = self.TAMF(x, his, condSatelliteEmb, condMapEmb, t, imageWidth=8) + x

        x = self.conv2(x)
        x = x + xInit

        feg = nn.Sigmoid()(torch.matmul(self.w1, x) + torch.matmul(self.w2, his) + self.b)
        return torch.multiply(x, feg) + torch.multiply(his, 1 - feg), his


class BlendUpsampling1(nn.Module):
    def __init__(self, in_channels, out_channels, context_channels, his_channel, his_size):
        super(BlendUpsampling1, self).__init__()
        self.out_channels = out_channels
        self.timeEmb = EmbedFC(1, out_channels)
        self.conv1 = nn.Sequential(nn.ConvTranspose2d(in_channels, out_channels, 2, 2),
                                   ResidualConvBlock(out_channels, out_channels),
                                   ResidualConvBlock(out_channels, out_channels))

        self.conEmbMap = ResidualConvBlock(context_channels, out_channels)
        self.conEmbSatellite = ResidualConvBlock(context_channels, out_channels)

        self.TAMF = TimeStepAwareModalityFusion(query_dim=1, x_h_dim=out_channels)

        self.conv2 = nn.Sequential(nn.BatchNorm2d(num_features=out_channels),
                                   nn.ReLU(),
                                   nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3,
                                             stride=1, padding=1))
        self.convInit = nn.ConvTranspose2d(in_channels, out_channels, 2, 2)

        self.hisUp = nn.Sequential(nn.ConvTranspose2d(in_channels, out_channels, 2, 2),
                                   ResidualConvBlock(out_channels, out_channels),
                                   ResidualConvBlock(out_channels, out_channels),
                                   nn.BatchNorm2d(num_features=out_channels),
                                   nn.ReLU(),
                                   nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3,
                                             stride=1, padding=1)
                                   )
        self.w1 = nn.Parameter(torch.randn(his_channel, his_size, his_size))
        self.w2 = nn.Parameter(torch.randn(his_channel, his_size, his_size))
        self.b = nn.Parameter(torch.randn(his_channel, his_size, his_size))

    def forward(self, x, skip, s, m, his, t):
        x = torch.cat((x, skip), 1)
        xInit = self.convInit(x)
        x = self.conv1(x)
        condMapEmb = self.conEmbMap(m)
        condSatelliteEmb = self.conEmbSatellite(s)
        his = self.hisUp(his)
        x = self.TAMF(x, his, condSatelliteEmb, condMapEmb, t, imageWidth=16) + x

        x = self.conv2(x)
        x = x + xInit

        feg = nn.Sigmoid()(torch.matmul(self.w1, x) + torch.matmul(self.w2, his) + self.b)
        return torch.multiply(x, feg) + torch.multiply(his, 1 - feg), his


class EmbedFC(nn.Module):
    def __init__(self, input_dim, emb_dim):
        super(EmbedFC, self).__init__()
        '''
        generic one layer FC NN for embedding things  
        '''
        self.input_dim = input_dim
        layers = [
            nn.Linear(input_dim, emb_dim),
            nn.GELU(),
            nn.Linear(emb_dim, emb_dim),
        ]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        x = x.view(-1, self.input_dim)
        return self.model(x)


class HamNet(nn.Module):
    def __init__(self, in_channels, n_feat, context_out_channels=32):
        super(HamNet, self).__init__()

        self.in_channels = in_channels
        self.n_feat = n_feat
        self.contextDim = 32

        self.encoderWideMap = nn.Sequential(
            nn.Conv2d(3, context_out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=context_out_channels),
            nn.ReLU(),

            nn.Conv2d(context_out_channels, context_out_channels, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(num_features=context_out_channels),
            nn.ReLU(),

            nn.Conv2d(context_out_channels, context_out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=context_out_channels),
            nn.ReLU(),

            nn.Conv2d(context_out_channels, context_out_channels, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(num_features=context_out_channels),
            nn.ReLU(),

            nn.Conv2d(context_out_channels, context_out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=context_out_channels),
            nn.ReLU(),

            nn.Conv2d(context_out_channels, context_out_channels, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(num_features=context_out_channels),
            nn.ReLU(),

            nn.Conv2d(context_out_channels, context_out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=context_out_channels),
            nn.ReLU(),

            nn.Conv2d(context_out_channels, context_out_channels, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(num_features=context_out_channels),
            nn.ReLU(),
        )

        self.encoderWideSatellite = nn.Sequential(
            nn.Conv2d(3, context_out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=context_out_channels),
            nn.ReLU(),

            nn.Conv2d(context_out_channels, context_out_channels, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(num_features=context_out_channels),
            nn.ReLU(),

            nn.Conv2d(context_out_channels, context_out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=context_out_channels),
            nn.ReLU(),

            nn.Conv2d(context_out_channels, context_out_channels, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(num_features=context_out_channels),
            nn.ReLU(),

            nn.Conv2d(context_out_channels, context_out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=context_out_channels),
            nn.ReLU(),

            nn.Conv2d(context_out_channels, context_out_channels, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(num_features=context_out_channels),
            nn.ReLU(),

            nn.Conv2d(context_out_channels, context_out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=context_out_channels),
            nn.ReLU(),

            nn.Conv2d(context_out_channels, context_out_channels, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(num_features=context_out_channels),
            nn.ReLU(),
        )

        self.init_conv = ResidualConvBlock(in_channels, n_feat, is_res=True)
        self.init_his = ResidualConvBlock(in_channels, n_feat, is_res=True)

        self.down1 = BlendDownsampling1(n_feat, n_feat, context_out_channels, n_feat, 8)
        self.down2 = BlendDownsampling2(n_feat, 2 * n_feat, context_out_channels, 2 * n_feat, 4)
        self.down3 = BlendDownsampling3(2 * n_feat, 4 * n_feat, context_out_channels, 4 * n_feat, 2)
        self.down4 = BlendDownsampling4(4 * n_feat, 8 * n_feat)

        self.up4 = BlendUpsampling4(8 * n_feat, 4 * n_feat)
        self.up3 = BlendUpsampling3(8 * n_feat, 2 * n_feat, context_out_channels, 2 * n_feat, 4)
        self.up2 = BlendUpsampling2(4 * n_feat, n_feat, context_out_channels, n_feat, 8)
        self.up1 = BlendUpsampling1(2 * n_feat, n_feat, context_out_channels, n_feat, 16)

        self.out = nn.Sequential(
            nn.Conv2d(2 * n_feat, n_feat, 3, 1, 1),
            nn.GroupNorm(8, n_feat),
            nn.ReLU(),
            nn.Conv2d(n_feat, self.in_channels * 2, 3, 1, 1),
        )

        self.hisOut = nn.Sequential(
            nn.Conv2d(2 * n_feat, n_feat, 3, 1, 1),
            nn.GroupNorm(8, n_feat),
            nn.ReLU(),
            nn.Conv2d(n_feat, self.in_channels, 3, 1, 1),
        )

    def forward(self, x, c, history, t, context_mask):
        t = t.unsqueeze(-1)
        context_mask = context_mask[:, None]
        context_mask_c = context_mask.repeat(1, c[0].numel())
        context_mask_c = (-1 * (1 - context_mask_c))
        context_mask_c = context_mask_c.view(c.shape)
        c = c * context_mask_c

        c = c.permute(0, 3, 1, 2)

        cmap = c[:, 0:3, :, :]
        csatellite = c[:, 3:, :, :]

        cmap = self.encoderWideMap(cmap)
        csatellite = self.encoderWideSatellite(csatellite)

        context_mask_his = context_mask.repeat(1, history[0].numel())
        context_mask_his = (-1 * (1 - context_mask_his))
        context_mask_his = context_mask_his.view(history.shape)
        history = history * context_mask_his

        x = self.init_conv(x)
        history = self.init_his(history)

        down1, history1 = self.down1(x, csatellite, cmap, history, t)
        down2, history2 = self.down2(down1, csatellite, cmap, history1, t)
        down3, history3 = self.down3(down2, csatellite, cmap, history2, t)
        down4, history4 = self.down4(down3, history3)

        up4, history5 = self.up4(down4, history4)
        up3, history6 = self.up3(up4, down3, csatellite, cmap, torch.cat((history3, history5), 1), t)
        up2, history7 = self.up2(up3, down2, csatellite, cmap, torch.cat((history2, history6), 1), t)
        up1, history8 = self.up1(up2, down1, csatellite, cmap, torch.cat((history1, history7), 1), t)

        out = self.out(torch.cat((up1, x), 1))
        history = self.hisOut(torch.cat((history, history8), 1))

        pred = out[:, 1:2, :, :]
        ieg = out[:, 0:1, :, :]
        ieg = nn.Sigmoid()(ieg)

        out = torch.multiply(pred, ieg) + torch.multiply(history, 1 - ieg)

        return out


class DDPM(nn.Module):
    def __init__(self, nn_model, betas, n_T, device, drop_prob=0.1):
        super(DDPM, self).__init__()
        self.nn_model = nn_model.to(device)

        for k, v in ddpm_schedules(betas[0], betas[1], n_T).items():
            self.register_buffer(k, v)

        self.n_T = n_T
        self.device = device
        self.drop_prob = drop_prob
        self.loss_mse = nn.MSELoss()

    def forward(self, x, c, history):

        _ts = torch.randint(1, self.n_T + 1, (x.shape[0],)).to(self.device)
        noise = torch.randn_like(x)

        x_t = (
                self.sqrtab[_ts, None, None, None] * x
                + self.sqrtmab[_ts, None, None, None] * noise
        )

        context_mask = torch.bernoulli(torch.zeros(len(x)) + self.drop_prob).to(self.device)

        return self.loss_mse(noise, self.nn_model(x_t, c, history, _ts / self.n_T, context_mask))

    def crime_sample(self, n_sample, size, cond, history, device, guide_w=0.0):
        x_i = torch.randn(n_sample, *size).to(device)

        cond = cond.repeat(2, 1, 1, 1)
        history = history.repeat(2, 1, 1, 1)

        context_mask = torch.zeros(n_sample).to(device)

        context_mask = context_mask.repeat(2)
        context_mask[n_sample:] = 1.

        for i in range(self.n_T, 0, -1):
            print('\r' + f'sampling timestep {i}', end='')
            t_is = torch.tensor([i / self.n_T]).to(device)
            t_is = t_is.repeat(n_sample)

            x_i = x_i.repeat(2, 1, 1, 1)
            t_is = t_is.repeat(2)

            z = torch.randn(n_sample, *size).to(device) if i > 1 else 0

            eps = self.nn_model(x_i, cond, history, t_is, context_mask)
            eps1 = eps[:n_sample]
            eps2 = eps[n_sample:]
            eps = (1 + guide_w) * eps1 - guide_w * eps2
            x_i = x_i[:n_sample]
            x_i = (
                    self.oneover_sqrta[i] * (x_i - eps * self.mab_over_sqrtmab[i])
                    + self.sqrt_beta_t[i] * z
            )

        return x_i


def ddpm_schedules(beta1, beta2, T):
    assert beta1 < beta2 < 1.0, "beta1 and beta2 must be in (0, 1)"

    beta_t = (beta2 - beta1) * torch.arange(0, T + 1, dtype=torch.float32) / T + beta1
    sqrt_beta_t = torch.sqrt(beta_t)
    alpha_t = 1 - beta_t
    log_alpha_t = torch.log(alpha_t)
    alphabar_t = torch.cumsum(log_alpha_t, dim=0).exp()

    sqrtab = torch.sqrt(alphabar_t)
    oneover_sqrta = 1 / torch.sqrt(alpha_t)

    sqrtmab = torch.sqrt(1 - alphabar_t)
    mab_over_sqrtmab_inv = (1 - alpha_t) / sqrtmab

    return {
        "alpha_t": alpha_t,
        "oneover_sqrta": oneover_sqrta,
        "sqrt_beta_t": sqrt_beta_t,
        "alphabar_t": alphabar_t,
        "sqrtab": sqrtab,
        "sqrtmab": sqrtmab,
        "mab_over_sqrtmab": mab_over_sqrtmab_inv,
    }
