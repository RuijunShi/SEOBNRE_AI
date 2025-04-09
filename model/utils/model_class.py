import torch
import torch.nn as nn
from .CNN_block import ResidualBlock

class ResidualMLP(nn.Module):
    def __init__(
        self, input_size, hidden_size, output_size, num_layers, bn=True, res_connet=True
    ):
        super(ResidualMLP, self).__init__()
        self.res_connet = res_connet
        self.bn_bool = bn
        # 输入层
        self.input_layer = nn.Linear(input_size, hidden_size)

        # 隐藏层
        self.hidden_layers = nn.ModuleList(
            [nn.Linear(hidden_size, hidden_size) for _ in range(num_layers)]
        )

        # 输出层
        self.output_layer = nn.Linear(hidden_size, output_size)
        if bn:
            self.bn = nn.BatchNorm1d(hidden_size)
        else:
            self.bn = nn.InstanceNorm1d(1, affine=True)
        self.activation = nn.ReLU()

    def forward(self, x):
        residual = self.input_layer(x)
        out = self.activation(residual)

        for hidden_layer in self.hidden_layers:
            residual = out
            out = hidden_layer(out)
            if self.res_connet:
                out = self.activation(out + residual)
            else:
                out = self.activation(out)

            if not self.bn_bool:
                out = self.bn(out.unsqueeze(1))
                out = out.squeeze(1)
            else:
                out = self.bn(out)

        out = self.output_layer(out)
        return out


class Par_MergeTime(nn.Module):
    def __init__(self, par_dim, hidden_dim, hidden_layer=4, v2=False):
        super(Par_MergeTime, self).__init__()
        self.v2 = v2
        if v2:
            par_dim = int(par_dim - 1)
        self.input_layer = nn.Linear(par_dim, hidden_dim)

        self.hidden_layers = nn.ModuleList(
            [nn.Linear(hidden_dim, hidden_dim) for _ in range(hidden_layer)]
        )

        self.output_layer = nn.Linear(hidden_dim, 1)
        self.activation = nn.ReLU()

    def forward(self, x):
        if self.v2:
            tmp = torch.zeros((x.shape[0], x.shape[1] - 1)).to(x.device)
            tmp[:, 0] = x[:, 0] / x[:, 1]
            tmp[:, 1] = x[:, 2]
            x = tmp

        residual = self.input_layer(x)
        out = self.activation(residual)

        for hidden_layer in self.hidden_layers:
            out = hidden_layer(out)
            out = self.activation(out)

        out = self.output_layer(out)
        return out.squeeze(1)


class Interpolation(nn.Module):
    def __init__(self, par_dim, hidden_dim, hidden_layer, waveform_len, bn=False):
        super(Interpolation, self).__init__()

        self.waveform_len = waveform_len
        self.MLP_layer = nn.Sequential(
            ResidualMLP(
                par_dim, hidden_dim, waveform_len, hidden_layer, bn=bn, res_connet=True
            )
        )
        self.CNN_block = nn.Sequential(
            nn.Conv1d(1, 32, 3, stride=1, padding=1),
            ResidualBlock(32, bn=bn),
            ResidualBlock(32, bn=bn),
            ResidualBlock(32, bn=bn),
            nn.Conv1d(32, 1, 3, stride=1, padding=1),
        )

    def forward(self, x):
        x = self.MLP_layer(x).unsqueeze(1)
        x = self.CNN_block(x)
        return x.squeeze(1)



