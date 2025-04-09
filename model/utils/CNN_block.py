import torch.nn as nn


class MLP_block(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(MLP_block, self).__init__()
        self.MLP = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.ReLU(True),
        )

    def forward(self, x):
        x = self.MLP(x)
        return x


class MLP_block_resnet(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(MLP_block_resnet, self).__init__()
        self.MLP = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.ReLU(True),
        )

    def forward(self, x):
        x_res = self.MLP(x)
        x += x_res
        return x


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, bn=True):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, in_channels, 3, stride=1, padding=1)
        self.relu = nn.ReLU(True)
        self.conv2 = nn.Conv1d(in_channels, in_channels, 3, stride=1, padding=1)
        if bn:
            self.bn = nn.BatchNorm1d(in_channels)
        else:
            self.bn = nn.InstanceNorm1d(in_channels, affine=True)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn(x)
        out = self.relu(out)
        out = self.conv2(x)
        out += residual
        return out


class CNN_block(nn.Module):
    def __init__(self, in_channels, out_channels, kernal_size=5, stride=2, padding=1):
        super(CNN_block, self).__init__()
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernal_size, stride=stride, padding=padding
        )
        self.relu = nn.ReLU(True)
        self.residual = ResidualBlock(out_channels)
        self.bn_in = nn.BatchNorm1d(in_channels)
        self.bn_out = nn.BatchNorm1d(out_channels)

    def forward(self, x):
        x = self.bn_in(x)
        x = self.relu(x)
        x = self.conv(x)
        x = self.bn_out(x)
        x = self.relu(x)
        x = self.residual(x)
        return x


class TranCNN_block(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernal_size=3,
        stride=2,
        padding=1,
        output_padding=1,
    ):
        super(TranCNN_block, self).__init__()
        self.TranCNN = nn.ConvTranspose1d(
            in_channels,
            out_channels,
            kernal_size,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
        )
        self.relu = nn.ReLU(True)
        self.residual = ResidualBlock(out_channels)
        self.bn_in = nn.BatchNorm1d(in_channels)
        self.bn_out = nn.BatchNorm1d(out_channels)

    def forward(self, x):
        x = self.bn_in(x)
        x = self.relu(x)
        x = self.TranCNN(x)
        x = self.bn_out(x)
        x = self.relu(x)
        x = self.residual(x)
        return x
