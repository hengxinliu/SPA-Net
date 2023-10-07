import torch.nn as nn
import torch.nn.functional as F
import torch

from utils.ptflops import get_model_complexity_info


try:
    from .sync_batchnorm import SynchronizedBatchNorm3d
except:
    print("异常处理！出现问题")


def normalization(planes, norm='bn'):
    if norm == 'bn':
        m = nn.BatchNorm3d(planes)
    elif norm == 'gn':
        m = nn.GroupNorm(4, planes)
    elif norm == 'in':
        m = nn.InstanceNorm3d(planes)
    elif norm == 'sync_bn':
        m = SynchronizedBatchNorm3d(planes)
    else:
        raise ValueError('normalization type {} is not supported'.format(norm))
    return m


class Conv_1x1x1(nn.Module):
    def __init__(self, in_dim, out_dim, activation=nn.ReLU(inplace=True)):
        super(Conv_1x1x1, self).__init__()
        self.conv1 = nn.Conv3d(in_dim, out_dim, kernel_size=1, stride=1, padding=0, bias=True)
        self.norm = SynchronizedBatchNorm3d(out_dim)
        self.act = activation

    def forward(self, x):
        x = self.act(self.norm(self.conv1(x)))
        return x


class Conv_3x3x1(nn.Module):
    def __init__(self, in_dim, out_dim, activation=nn.ReLU(inplace=True),g=1):
        super(Conv_3x3x1, self).__init__()
        self.conv1 = nn.Conv3d(in_dim, out_dim, kernel_size=(3, 3, 1), stride=1,groups=g, padding=(1, 1, 0), bias=True)
        self.norm = SynchronizedBatchNorm3d(out_dim)
        self.act = activation

    def forward(self, x):
        x = self.act(self.norm(self.conv1(x)))
        return x


class Conv_1x3x3(nn.Module):
    def __init__(self, in_dim, out_dim, activation=nn.ReLU(inplace=True)):
        super(Conv_1x3x3, self).__init__()
        self.conv1 = nn.Conv3d(in_dim, out_dim, kernel_size=(1, 3, 3), stride=1, padding=(0, 1, 1), bias=True)
        self.norm = SynchronizedBatchNorm3d(out_dim)
        self.act = activation

    def forward(self, x):
        x = self.act(self.norm(self.conv1(x)))
        return x


class Conv_3x3x3(nn.Module):
    def __init__(self, in_dim, out_dim,activation=nn.ReLU(inplace=True),g=1,d=1):
        super(Conv_3x3x3, self).__init__()
        self.conv1 = nn.Conv3d(in_dim, out_dim, kernel_size=(3, 3, 3), stride=1,groups=g,dilation=d,padding=(1, 1, 1), bias=True)
        self.norm = SynchronizedBatchNorm3d(out_dim)
        self.act = activation

    def forward(self, x):
        x = self.act(self.norm(self.conv1(x)))
        return x


class Conv_down(nn.Module):
    def __init__(self, in_dim, out_dim, activation=nn.ReLU(inplace=True)):
        super(Conv_down, self).__init__()
        self.conv1 = nn.Conv3d(in_dim, out_dim, kernel_size=(3, 3, 3), stride=2, padding=(1, 1, 1), bias=True)
        self.norm = SynchronizedBatchNorm3d(out_dim)
        self.act = activation

    def forward(self, x):
        x = self.act(self.norm(self.conv1(x)))
        return x

class Conv3d_Block(nn.Module):
    def __init__(self, num_in, num_out, kernel_size=1, stride=1, g=1, padding=None, norm=None):
        super(Conv3d_Block, self).__init__()
        if padding == None:
            padding = (kernel_size - 1) // 2
        self.bn = normalization(num_in, norm=norm)
        self.act_fn = nn.ReLU(inplace=True)
        self.conv = nn.Conv3d(num_in, num_out, kernel_size=kernel_size, padding=padding, stride=stride, groups=g,
                              bias=False)

    def forward(self, x):  # BN + Relu + Conv
        h = self.act_fn(self.bn(x))
        h = self.conv(h)
        return h



#
class DilatedConv3DBlock(nn.Module):
    def __init__(self, num_in, num_out, kernel_size=(1, 1, 1), stride=1, g=1, d=(1, 1, 1), norm=None):
        super(DilatedConv3DBlock, self).__init__()
        assert isinstance(kernel_size, tuple) and isinstance(d, tuple)

        padding = tuple(
            [(ks - 1) // 2 * dd for ks, dd in zip(kernel_size, d)]
        )

        self.conv = nn.Conv3d(num_in, num_out, kernel_size=kernel_size, padding=padding, stride=stride, groups=g,
                              dilation=d, bias=False)
        self.bn = normalization(num_out, norm=norm)
        self.act_fn = nn.ReLU(inplace=True)


    def forward(self, x):
        h = self.conv(x)
        h = self.act_fn(self.bn(h))
        return h

class EMM_block(nn.Module):
    def __init__(self,in_dim, out_dim, g=1,activation=nn.ReLU(inplace=True),norm=None):
        super(EMM_block, self).__init__()
        self.weight1 = nn.Parameter(torch.ones(1))
        self.weight2 = nn.Parameter(torch.ones(1))
        self.weight3 = nn.Parameter(torch.ones(1))
        self.conv_3x3x1_1 = DilatedConv3DBlock(in_dim, out_dim, kernel_size=(3, 3, 1),d=(1, 1, 1), norm=norm, g=g)
        self.conv_3x3x1_2 = DilatedConv3DBlock(in_dim, out_dim, kernel_size=(3, 3, 1),d=(2, 2, 1), norm=norm, g=g)
        self.conv_3x3x1_3 = DilatedConv3DBlock(in_dim, out_dim, kernel_size=(3, 3, 1),d=(4, 4, 1), norm=norm, g=g)

        self.combine_dim_down = Conv_3x3x1(in_dim * 3, out_dim, activation)
        self.norm = SynchronizedBatchNorm3d(out_dim)
        self.act = activation
    def forward(self,x):
        # d0 = x
        d1 = self.conv_3x3x1_1(x)*self.weight1
        d2 = self.conv_3x3x1_2(x)*self.weight2
        d4 = self.conv_3x3x1_3(x)*self.weight3

        add1 = d1+d2
        add2 = add1+d4

        combine = torch.cat([d1,add1,add2],dim=1)
        combine = self.combine_dim_down(combine)
        combine = x + combine
        # combine = self.act(self.norm(combine))

        return combine



class EMM_unit(nn.Module):
    def __init__(self, in_dim, out_dim, g=1,activation=nn.ReLU(inplace=True),norm=None):
        super(EMM_unit, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.inter_dim = in_dim // 4
        self.out_inter_dim = out_dim // 4
        self.conv_3x3x1_1 = EMM_block(self.out_inter_dim, self.out_inter_dim,norm=norm,g=g)
        self.conv_3x3x1_2 = EMM_block(self.out_inter_dim, self.out_inter_dim,norm=norm,g=g)
        self.conv_3x3x1_3 = EMM_block(self.out_inter_dim, self.out_inter_dim,norm=norm,g=g)
        # self.conv_3x3x1_4 = DilatedConv3DBlock(self.out_inter_dim, self.out_inter_dim, kernel_size=(3,3,1),d=(8,8,1),norm=norm,g=g)

        self.conv_1x1x1_1 = Conv_1x1x1(in_dim, out_dim, activation)
        self.conv_1x1x1_2 = Conv_1x1x1(out_dim, out_dim, activation)
        if self.in_dim != self.out_dim:
            self.conv_1x1x1_3 = Conv_1x1x1(in_dim, out_dim, activation)
        self.conv_1x3x3 = Conv_1x3x3(out_dim, out_dim, activation)

    def forward(self, x):
        x_1 = self.conv_1x1x1_1(x)
        x_1 = torch.chunk(x_1,4,1)
        x1 = x_1[0]
        x2 = x_1[1]
        x3 = x_1[2]
        x4 = x_1[3]
        # x1 = x_1[:, 0:self.out_inter_dim, ...]
        # x2 = x_1[:, self.out_inter_dim:self.out_inter_dim * 2, ...]
        # x3 = x_1[:, self.out_inter_dim * 2:self.out_inter_dim * 3, ...]
        # x4 = x_1[:, self.out_inter_dim * 3:self.out_inter_dim * 4, ...]
        x2 = self.conv_3x3x1_1(x2)
        x3 = self.conv_3x3x1_2(x2 + x3)
        x4 = self.conv_3x3x1_3(x3 + x4)
        x_1 = torch.cat((x1, x2, x3, x4), dim=1)
        x_1 = self.conv_1x1x1_2(x_1)
        if self.in_dim != self.out_dim:
            x = self.conv_1x1x1_3(x)
        x_1 = self.conv_1x3x3(x + x_1)
        return x_1


class MFunit(nn.Module):
    def __init__(self, num_in, num_out, g=1, stride=1, d=(1, 1), norm=None):
        """  The second 3x3x1 group conv is replaced by 3x3x3.
        :param num_in: number of input channels
        :param num_out: number of output channels
        :param g: groups of group conv.
        :param stride: 1 or 2
        :param d: tuple, d[0] for the first 3x3x3 conv while d[1] for the 3x3x1 conv
        :param norm: Batch Normalization
        """
        super(MFunit, self).__init__()
        self.stride = stride
        num_mid = num_in if num_in <= num_out else num_out
        self.conv1x1x1_in1 = Conv3d_Block(num_in, num_in // 4, kernel_size=1, stride=1, norm=norm)
        self.conv1x1x1_in2 = Conv3d_Block(num_in // 4, num_mid, kernel_size=1, stride=1, norm=norm)
        # self.activation = nn.ReLU(inplace=True)
        ### shuffle
        # self.conv3x3x3_m1 = DilatedConv3DBlock(num_mid, num_out, kernel_size=(3, 3, 3), stride=stride, g=g,
        #                                        d=(d[0], d[0], d[0]), norm=norm)  # dilated
        # self.conv3x3x3_m2 = DilatedConv3DBlock(num_out, num_out, kernel_size=(3, 3, 1), stride=1, g=g,
        #                                        d=(d[1], d[1], 1), norm=norm)
        # # self.conv3x3x3_m2 = DilatedConv3DBlock(num_out,num_out,kernel_size=(1,3,3),stride=1,g=g,d=(1,d[1],d[1]),norm=norm)

        self.conv3x3x3_m1 = EMM_unit(num_mid,num_out,g=g,norm=norm)
        self.conv3x3x3_m2 = EMM_unit(num_out,num_out,g=g,norm=norm)

        # skip connection
        if num_in != num_out or stride != 1:
            if stride == 1:
                self.conv1x1x1_shortcut = Conv3d_Block(num_in, num_out, kernel_size=1, stride=1, padding=0, norm=norm)
            if stride == 2:
                # if MF block with stride=2, 2x2x2
                self.conv_down = Conv_down(num_mid,num_mid)
                self.conv2x2x2_shortcut = Conv3d_Block(num_in, num_out, kernel_size=2, stride=2, padding=0,
                                                       norm=norm)  # params

    def forward(self, x):
        x1 = self.conv1x1x1_in1(x)
        x2 = self.conv1x1x1_in2(x1)
        if self.stride == 2:
            x2 = self.conv_down(x2)
        # print(x2.shape)
        x3 = self.conv3x3x3_m1(x2)
        x4 = self.conv3x3x3_m2(x3)

        shortcut = x

        if hasattr(self, 'conv1x1x1_shortcut'):
            shortcut = self.conv1x1x1_shortcut(shortcut)
        if hasattr(self, 'conv2x2x2_shortcut'):
            shortcut = self.conv2x2x2_shortcut(shortcut)

        return x4 + shortcut


class MEU3DModule(nn.Module):
    def __init__(self, channels_high, channels_low, channel_out):
        super(MEU3DModule, self).__init__()
        self.gamma1 = nn.Parameter(torch.ones(1))
        self.gamma2 = nn.Parameter(torch.ones(1))
        self.conv1x1x1_low = nn.Conv3d(channels_low, channel_out, kernel_size=1, bias=False)
        self.bn_low = normalization(channel_out,norm = 'sync_bn')
        self.sa_conv = nn.Conv3d(1, 1, kernel_size=1, bias=False)

        self.conv1x1x1_high = nn.Conv3d(channels_high, channel_out, kernel_size=1, bias=False)
        self.bn_high = normalization(channel_out,norm='sync_bn')
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.ca_conv = nn.Conv3d(channel_out, channel_out, kernel_size=1, bias=False)

        self.sa_sigmoid = nn.Sigmoid()
        self.ca_sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, fms_high, fms_low):
        """
        :param fms_high:  High level Feature map. Tensor.
        :param fms_low: Low level Feature map. Tensor.
        """
        _, _, h, w , d = fms_low.shape

        #
        fms_low = self.conv1x1x1_low(fms_low)
        fms_low= self.bn_low(fms_low)
        sa_avg_out = self.sa_sigmoid(self.sa_conv(torch.mean(fms_low, dim=1, keepdim=True)))

        #
        fms_high = self.conv1x1x1_high(fms_high)
        fms_high = self.bn_high(fms_high)
        ca_avg_out = self.ca_sigmoid(self.relu(self.ca_conv(self.avg_pool(fms_high))))

        #
        fms_high_up = F.interpolate(fms_high, size=(h,w,d), mode='trilinear', align_corners=True)
        fms_sa_att = sa_avg_out * fms_high_up
        #
        fms_ca_att = ca_avg_out * fms_low

        out = self.gamma1*fms_ca_att + self.gamma2*fms_sa_att

        return out

class MFNet(nn.Module):  #
    # [96]   Flops:  13.361G  &  Params: 1.81M
    # [112]  Flops:  16.759G  &  Params: 2.46M
    # [128]  Flops:  20.611G  &  Params: 3.19M
    def __init__(self, c=4, n=32, channels=128, groups=16, norm='bn', num_classes=4):
        super(MFNet, self).__init__()

        # Entry flow
        self.encoder_block1 = nn.Conv3d(c, n, kernel_size=3, padding=1, stride=2, bias=False)  # H//2
        self.encoder_block2 = nn.Sequential(
            MFunit(n, channels, g=groups, stride=2, norm=norm),  # H//4 down
            MFunit(channels, channels, g=groups, stride=1, norm=norm),
            MFunit(channels, channels, g=groups, stride=1, norm=norm)
        )
        #
        self.encoder_block3 = nn.Sequential(
            MFunit(channels, channels * 2, g=groups, stride=2, norm=norm),  # H//8
            MFunit(channels * 2, channels * 2, g=groups, stride=1, norm=norm),
            MFunit(channels * 2, channels * 2, g=groups, stride=1, norm=norm)
        )

        self.encoder_block4 = nn.Sequential(  # H//8,channels*4
            MFunit(channels * 2, channels * 3, g=groups, stride=2, norm=norm),  # H//16
            MFunit(channels * 3, channels * 3, g=groups, stride=1, norm=norm),
            MFunit(channels * 3, channels * 2, g=groups, stride=1, norm=norm),
        )

        self.meu1 = MEU3DModule(channels*2,channels*2,channels)
        self.meu2 = MEU3DModule(channels,channels,channels)
        self.meu3 = MEU3DModule(channels,channels,n)

        # self.upsample1 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)  # H//8
        # self.decoder_block1 = MFunit(channels * 2 + channels * 2, channels * 2, g=groups, stride=1, norm=norm)
        #
        # self.upsample2 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)  # H//4
        # self.decoder_block2 = MFunit(channels * 2 + channels, channels, g=groups, stride=1, norm=norm)
        #
        # self.upsample3 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)  # H//2
        # self.decoder_block3 = MFunit(channels + n, n, g=groups, stride=1, norm=norm)
        self.upsample4 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)  # H
        self.seg = nn.Conv3d(n, num_classes, kernel_size=1, padding=0, stride=1, bias=False)

        self.softmax = nn.Softmax(dim=1)

        # Initialization
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                torch.nn.init.torch.nn.init.kaiming_normal_(m.weight)  #
            elif isinstance(m, nn.BatchNorm3d) or isinstance(m, nn.GroupNorm) or isinstance(m, SynchronizedBatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # Encoder0.
        x1 = self.encoder_block1(x)  # H//2 down
        x2 = self.encoder_block2(x1)  # H//4 down
        x3 = self.encoder_block3(x2)  # H//8 down
        x4 = self.encoder_block4(x3)  # H//16
        # print(x4.shape)
        # Decoder
        # y1 = self.upsample1(x4)  # H//8
        # # print(y1.shape)
        # y1 = torch.cat([x3, y1], dim=1)
        # y1 = self.decoder_block1(y1)
        #
        # y2 = self.upsample2(y1)  # H//4
        # y2 = torch.cat([x2, y2], dim=1)
        # y2 = self.decoder_block2(y2)
        #
        # y3 = self.upsample3(y2)  # H//2
        # y3 = torch.cat([x1, y3], dim=1)
        # y3 = self.decoder_block3(y3)
        y1 = self.meu1(x4,x3)
        y2 = self.meu2(y1,x2)
        y3 = self.meu3(y2,x1)

        y4 = self.seg(y3)
        y4 = self.upsample4(y4)
        if hasattr(self, 'softmax'):
            y4 = self.softmax(y4)
        return y4


if __name__ == '__main__':
    import os

    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    device = torch.device('cuda:0')
    x = torch.rand((1, 4, 128, 128, 128), device=device)  # [bsize,channels,Height,Width,Depth]
    # model = DMFNet(c=4, groups=16, norm='sync_bn', num_classes=4)
    # model = DMFNet(c=4, groups=4, norm='sync_bn', num_classes=4)
    model = MFNet(c=4,n=32, channels=32,groups=1, norm='sync_bn', num_classes=4)
    model.cuda(device)

    y = model(x)
    print(y.shape)


