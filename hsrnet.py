import torch.nn as nn
import torch
import torch.nn.functional as F

def make_model(args):
    return Net(args)

class MeanShift(nn.Conv2d):
    def __init__(self, rgb_range, rgb_mean, rgb_std, sign=-1):
        super(MeanShift, self).__init__(3, 3, kernel_size=1)
        std = torch.Tensor(rgb_std)
        self.weight.data = torch.eye(3).view(3, 3, 1, 1)
        self.weight.data.div_(std.view(3, 1, 1, 1))
        self.bias.data = sign * rgb_range * torch.Tensor(rgb_mean)
        self.bias.data.div_(std)
        self.requires_grad = False

class ConvReLU(nn.Module):
    def __init__(self, n_in, n_cout):
        super(ConvReLU, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(n_in, n_cout, 3, padding=3//2),
            nn.LeakyReLU(inplace=True),
        )
    def forward(self, x):
        return self.conv(x)

class HierarchicalConv(nn.Module):
    def __init__(self):
        super(HierarchicalConv, self).__init__()
        self.conv_in = ConvReLU(64, 64)

        self.convA_0 = nn.Sequential(ConvReLU(16, 16))
        self.convA_1 = nn.Conv2d(32, 16, 1)

        self.convB_0 = nn.Sequential(ConvReLU(16, 16), ConvReLU(16, 16))
        self.convB_1 = nn.Conv2d(32, 16, 1)

        self.convC_0 = nn.Sequential(ConvReLU(16, 16), ConvReLU(16, 16), ConvReLU(16, 16))
        self.convC_1 = nn.Conv2d(32, 16, 1)

        self.conv_out = nn.Conv2d(64, 64, 3, padding=3//2)
    def forward(self, x):
        y = self.conv_in(x)
        y1, y2, y3, y4 = torch.chunk(y, 4, dim=1)
        y2 = self.convA_1(torch.cat((self.convA_0(y2), y1), 1))
        y3 = self.convB_1(torch.cat((self.convB_0(y3), y2), 1))
        y4 = self.convC_1(torch.cat((self.convC_0(y4), y3), 1))
        y = torch.cat((y1, y2, y3, y4), 1)
        y = self.conv_out(y) + x
        return y

class UNetAttention(nn.Module):
    def __init__(self):
        super(UNetAttention, self).__init__()
        self.conv_in = nn.Conv2d(64, 48, 1)
        self.conv_1x1 = ConvReLU(16, 16)
        self.conv_2x2 = ConvReLU(16, 16)
        self.conv_4x4 = ConvReLU(16, 16)
        self.conv_out = nn.Conv2d(48, 64, 1)
    def forward(self, x):
        y = self.conv_in(x)
        y1, y2, y4 = torch.chunk(y, 3, 1)
        y1 = self.conv_1x1(y1)
        y2 = F.max_pool2d(y2, kernel_size=2, stride=2)
        y2 = self.conv_2x2(y2)
        y2 = F.interpolate(y2, size=(y1.size(2), y1.size(3)), mode='bilinear', align_corners=False)
        y4 = self.conv_4x4(y4)
        y4 = F.interpolate(y4, size=(y1.size(2), y1.size(3)), mode='bilinear', align_corners=False)
        y = torch.cat((y1, y2, y4), 1)
        y = self.conv_out(y)
        y = torch.sigmoid(y) * x
        return y

class BasicBlock(nn.Module):
    def __init__(self):
        super(BasicBlock, self).__init__()
        self.conv = nn.Sequential(
            HierarchicalConv(),
            UNetAttention(),
        )
    def forward(self, x):
        return self.conv(x)

class Upscale(nn.Module):
    def __init__(self, scale):
        super(Upscale, self).__init__()
        self.conv = nn.Conv2d(64, 3*scale*scale, 3, padding=3//2)
        self.upsample = nn.PixelShuffle(scale)
    def forward(self, x):
        x = self.conv(x)
        x = self.upsample(x)
        return x

class Net(nn.Module):
    def __init__(self, args):
        super(Net, self).__init__()
        scale = args.scale[0]
        rgb_mean = (0.4488, 0.4371, 0.4040)
        rgb_std = (1.0, 1.0, 1.0)

        self.n_stage = 10
        self.n_iters = 3

        self.sub_mean = MeanShift(255, rgb_mean, rgb_std)
        self.add_mean = MeanShift(255, rgb_mean, rgb_std, 1)

        self.head = nn.Sequential(nn.Conv2d(3, 64, 3, padding=3//2))

        self.Trans = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=3//2),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=3//2),
        )

        self.SolverLS = nn.Sequential(
            nn.Conv2d(128, 64, 1),
            nn.Conv2d(64, 64, 3, padding=3//2),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=3//2),
        )
        
        module_body = [BasicBlock() for _ in range(self.n_stage)]
        self.body = nn.Sequential(
            *module_body,
            nn.Conv2d(64, 64, 3, padding=3//2),
        )

        self.Denoiser = self.body
        
        self.tail = nn.Sequential(
                        nn.Conv2d(64, 64, 3, padding=3//2),
                        Upscale(scale),
                    )

    def forward(self, x):
        x = self.sub_mean(x)

        y = self.head(x)
        y = self.Trans(y) + y
        u = y

        for i in range(self.n_iters):
            u = self.SolverLS(torch.cat((u, y), 1)) + u
            u = self.Denoiser(u) + u

        x = self.tail(u)

        x = self.add_mean(x)
        return x 

    def load_state_dict(self, state_dict, strict=True):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name in own_state:
                if isinstance(param, nn.Parameter):
                    param = param.data
                try:
                    own_state[name].copy_(param)
                except Exception:
                    if name.find('tail') == -1:
                        raise RuntimeError('While copying the parameter named {}, '
                                           'whose dimensions in the model are {} and '
                                           'whose dimensions in the checkpoint are {}.'
                                           .format(name, own_state[name].size(), param.size()))
            elif strict:
                if name.find('tail') == -1:
                    raise KeyError('unexpected key "{}" in state_dict'
                                   .format(name))

