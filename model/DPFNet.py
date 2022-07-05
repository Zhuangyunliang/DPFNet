import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision


# Multi-level Dilated Convolution Blocks
class MDBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation):
        super(MDBlock, self).__init__()

        self.main = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=dilation,
                      dilation=dilation, groups=1, bias=False),
            nn.InstanceNorm2d(out_channels),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=dilation,
                      dilation=dilation, groups=1, bias=False),
            nn.InstanceNorm2d(out_channels),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=dilation+1,
                      dilation=dilation+1)
        )

    def forward(self, x):
        y = self.main(x)
        return y + x


# Multi-level Dilated Convolution Module
class MDConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(MDConv, self).__init__()

        self.down = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, 1, 1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(64, 64, 3, 1, 1),
            # nn.LeakyReLU(inplace=True),
            # nn.Conv2d(64, 64, 2, 2)
        )

        self.dc2 = MDBlock(64, 64, 3, 2)
        self.dc4 = MDBlock(64, 64, 3, 4)

        self.fu = nn.Conv2d(64*3, 64, 1)

        self.up = nn.Sequential(
            # nn.ConvTranspose2d(64, 64, 4, 2, 1),
            # nn.LeakyReLU(inplace=True),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(64, out_channels, 1)
        )

    def forward(self, x):

        y = self.down(x)
        y2 = self.dc2(y)
        y4 = self.dc4(y2)

        output = self.up(self.fu(torch.cat([y, y2, y4], dim=1)))
        return output + x


class ComplexConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(ComplexConv, self).__init__()

        self.real = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
            # nn.InstanceNorm2d(out_channels)
        )
        self.imag = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
            # nn.InstanceNorm2d(out_channels)
        )

    def forward(self, inputs):

        r = inputs[0]
        i = inputs[1]

        r_r = self.real(r)
        i_i = self.imag(i)

        i_r = self.imag(r)
        r_i = self.real(i)

        return [r_r-i_i, i_r+r_i]


class ComplexLeakyReLU(nn.Module):
    def __init__(self):
        super(ComplexLeakyReLU, self).__init__()

        self.real = nn.LeakyReLU()
        self.imag = nn.LeakyReLU()

    def forward(self, inputs):
        r = inputs[0]
        i = inputs[1]

        r = self.real(r)
        i = self.imag(i)

        return [r, i]


class PFModule(nn.Module):
    def __init__(self):
        super(PFModule, self).__init__()
        self.conv_1 = nn.Sequential(
            # ComplexConv(3, 64, 3, 1, 1),
            # ComplexLeakyReLU()
            nn.Conv2d(3, 64, 3, 1, 1),
            nn.LeakyReLU()
        )

        self.conv_2 = nn.Sequential(
            ComplexConv(3, 64, 3, 1, 1),
            ComplexLeakyReLU(),
            ComplexConv(64, 64, 3, 1, 1),
            ComplexLeakyReLU(),
            ComplexConv(64, 64, 3, 1, 1),
            ComplexLeakyReLU()
        )

        self.conv_3 = nn.Sequential(
            ComplexConv(64, 64, 3, 1, 1),
            ComplexLeakyReLU(),
            ComplexConv(64, 64, 3, 1, 1),
            ComplexLeakyReLU(),
            ComplexConv(64, 64, 3, 1, 1),
            ComplexLeakyReLU()
        )

        self.conv_4 = nn.Sequential(
            ComplexConv(64, 64, 3, 1, 1),
            ComplexLeakyReLU(),
            ComplexConv(64, 64, 3, 1, 1),
            ComplexLeakyReLU(),
            ComplexConv(64, 64, 3, 1, 1),
            ComplexLeakyReLU()
        )


        self.out = nn.Sequential(
            # ComplexConv(64, 3, 1),
            # ComplexLeakyReLU(),
            # ComplexConv(64, 3, 1)
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.LeakyReLU(),
            nn.Conv2d(64, 3, 1)
        )

    def forward(self, x):

        # x_fft = self.conv_1(x)

        x_fft = torch.fft.rfftn(x, dim=[-2, -1])

        x_r = x_fft.real
        x_i = x_fft.imag

        f1 = self.conv_2((x_r, x_i))
        f2 = self.conv_3(tuple(f1)) + f1
        out = self.conv_4(tuple(f2)) + f2

        out = torch.fft.irfftn(torch.complex(out[0], out[1]), dim=[-2, -1])

        out = self.out(out)

        return out

# Residual Blocks
class Residual(nn.Module):
    def __init__(self, input_channels, num_channels,
                 use_1x1conv=False, strides=1):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, num_channels,
                               kernel_size=3, padding=1, stride=strides)
        self.conv2 = nn.Conv2d(num_channels, num_channels,
                               kernel_size=3, padding=1)
        if use_1x1conv:
            self.conv3 = nn.Conv2d(input_channels, num_channels,
                                   kernel_size=1, stride=strides)
        else:
            self.conv3 = None
        # self.bn1 = nn.BatchNorm2d(num_channels)
        # self.bn2 = nn.BatchNorm2d(num_channels)

    def forward(self, X):
        Y = F.relu(self.conv1(X))
        Y = self.conv2(Y)
        if self.conv3:
            X = self.conv3(X)
        Y += X
        return Y


class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()

        self.md = MDConv(in_channels=3, out_channels=3)
        self.pfm = PFModule()

        self.afm = nn.Conv2d(6, 6, 3, 1, 1, bias=False)

        self.u = nn.Sequential(
            Residual(3, 3)
            # Residual(3, 3)
        )

    def forward(self, img):

        f_s = self.md(img)
        f_f = self.pfm(img) 

        torchvision.utils.save_image(f_s, '/home2/zyl/Enhancement/preview/spatial.png')
        torchvision.utils.save_image(f_f, '/home2/zyl/Enhancement/preview/frequency.png')

        w = F.sigmoid(self.afm(torch.cat([f_s, f_f], dim=1)))

        out = f_s * w[:, 0:3,:,:] + f_f * w[:, 3:6,:,:]
        # out = self.u(out)
        # out = self.fu(torch.cat([f_s, f_f], dim=1))

        return out


if __name__ == '__main__':
    input = torch.randn(4, 3, 256, 256)
    model = Network()
    pre = model(input)
    print(pre.shape)

    # model = ComplexConv(3, 3, 3, 1, 1)
    # pre = model(input, input)
