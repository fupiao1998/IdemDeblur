import torch
import torch.nn as nn
import torch.nn.functional as F


def conv(in_channels, out_channels, kernel_size=3, stride=1,dilation=1, bias=True):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, 
                  dilation=dilation, padding=((kernel_size-1)//2)*dilation, bias=bias),
        nn.LeakyReLU(0.1, inplace=True)
    )


def conv_norelu(in_channels, out_channels, kernel_size=3, stride=1,dilation=1, bias=True):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, 
                  dilation=dilation, padding=((kernel_size-1)//2)*dilation, bias=bias))


def upconv(in_channels, out_channels):
    return nn.Sequential(
        nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=True),
        nn.LeakyReLU(0.1, inplace=True)
    )


def resnet_block(in_channels,  kernel_size=3, dilation=[1,1], bias=True):
    return ResnetBlock(in_channels, kernel_size, dilation, bias=bias)


class ResnetBlock(nn.Module):
    def __init__(self, in_channels, kernel_size, dilation, bias):
        super(ResnetBlock, self).__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, stride=1, dilation=dilation[0], 
                      padding=((kernel_size-1)//2)*dilation[0], bias=bias),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, stride=1, dilation=dilation[1], 
                      padding=((kernel_size-1)//2)*dilation[1], bias=bias),
        )
    def forward(self, x):
        out = self.stem(x) + x
        return out


class MTRNN(nn.Module):
    def __init__(self):
        super(MTRNN, self).__init__()
        act = nn.ReLU(True)
        # RGB mean for DIV2K
        # rgb_mean = (0.4488, 0.4371, 0.4040)
        # rgb_std = (1.0, 1.0, 1.0)
        # self.sub_mean = MeanShift(255, rgb_mean, rgb_std)
        # self.add_mean = MeanShift(255, rgb_mean, rgb_std, 1)

        ks = 3
        ch1 = 32
        ch2 = 64
        ch3 = 80

        self.conv1_1 = conv_norelu(3, ch1, kernel_size=ks, stride=1)
        self.conv1_c = conv_norelu(ch1*2, ch1, kernel_size=1, stride=1)
        self.conv1_2 = resnet_block(ch1, kernel_size=ks)
        self.conv1_3 = resnet_block(ch1, kernel_size=ks)
        self.conv1_4 = resnet_block(ch1, kernel_size=ks)
        self.conv1_5 = resnet_block(ch1, kernel_size=ks)

        self.conv2_1 = conv(ch1, ch2, kernel_size=ks, stride=2)
        self.conv2_c = conv_norelu(ch2*2, ch2, kernel_size=1, stride=1)
        self.conv2_2 = resnet_block(ch2, kernel_size=ks)
        self.conv2_3 = resnet_block(ch2, kernel_size=ks)
        self.conv2_4 = resnet_block(ch2, kernel_size=ks)
        self.conv2_5 = resnet_block(ch2, kernel_size=ks)

        self.conv3_1 = conv(ch2, ch3, kernel_size=ks, stride=2)
        self.conv3_c = conv_norelu(ch3*2, ch3, kernel_size=1, stride=1)
        self.conv3_2 = resnet_block(ch3, kernel_size=ks)
        self.conv3_3 = resnet_block(ch3, kernel_size=ks)
        self.conv3_4 = resnet_block(ch3, kernel_size=ks)
        self.conv3_5 = resnet_block(ch3, kernel_size=ks)

        self.conv4_1 = conv(ch3, ch3, kernel_size=ks, stride=2)
        self.conv4_c = conv_norelu(ch3*2, ch3, kernel_size=1, stride=1)
        self.conv4_2 = resnet_block(ch3, kernel_size=ks)
        self.conv4_3 = resnet_block(ch3, kernel_size=ks)
        self.conv4_4 = resnet_block(ch3, kernel_size=ks)
        self.conv4_5 = resnet_block(ch3, kernel_size=ks)

        # decoder
        self.conv5_1 = upconv(ch3, ch3)
        self.conv5_c = conv_norelu(ch3*2, ch3, kernel_size=1, stride=1)
        self.conv5_2 = resnet_block(ch3, kernel_size=ks)
        self.conv5_3 = resnet_block(ch3, kernel_size=ks)
        self.conv5_4 = resnet_block(ch3, kernel_size=ks)
        self.conv5_5 = resnet_block(ch3, kernel_size=ks)

        self.conv6_1 = upconv(ch3, ch2)
        self.conv6_c = conv_norelu(ch2*2, ch2, kernel_size=1, stride=1)
        self.conv6_2 = resnet_block(ch2, kernel_size=ks)
        self.conv6_3 = resnet_block(ch2, kernel_size=ks)
        self.conv6_4 = resnet_block(ch2, kernel_size=ks)
        self.conv6_5 = resnet_block(ch2, kernel_size=ks)

        self.conv7_1 = upconv(ch2, ch1)
        self.conv7_c = conv_norelu(ch1, ch1, kernel_size=1, stride=1)
        self.conv7_2 = resnet_block(ch1, kernel_size=ks)
        self.conv7_3 = resnet_block(ch1, kernel_size=ks)
        self.conv7_4 = resnet_block(ch1, kernel_size=ks)
        self.conv7_5 = resnet_block(ch1, kernel_size=ks)

        self.img_prd = conv_norelu(ch1, 3, kernel_size=ks)
        
    def forward(self, x):
        x_in, feature_0, feature_1, feature_2 = x
        
        x_d_x1_p, feature_0_out, feature_1_out, feature_2_out = self.multi(x_in, feature_0, feature_1, feature_2)

        if self.training:
            return [x_d_x1_p, feature_0_out, feature_1_out, feature_2_out]
        else:
            return [x_d_x1_p, feature_0_out, feature_1_out, feature_2_out]

    def multi(self, x_in, feature_0, feature_1, feature_2):
        x_inf = x_in

        conv1_d = self.conv1_1(x_inf)
        if feature_0.shape[1] == 3:
            feature_0 = conv1_d   # torch.zeros_like(conv1_d)
        conv1_d_c = torch.cat([conv1_d, feature_0], 1)
        conv1_d = self.conv1_c(conv1_d_c)
        conv1_d = self.conv1_5(self.conv1_4(self.conv1_3(self.conv1_2(conv1_d))))   # 1/1

        conv2_d = self.conv2_1(conv1_d)
        if feature_1.shape[1] == 3:
            feature_1 = conv2_d   # torch.zeros_like(conv2_d)
        conv2_d_c = torch.cat([conv2_d, feature_1], 1)
        conv2_d = self.conv2_c(conv2_d_c)
        conv2_d = self.conv2_5(self.conv2_4(self.conv2_3(self.conv2_2(conv2_d))))   # 1/2

        conv3_d = self.conv3_1(conv2_d)
        if feature_2.shape[1] == 3:
            feature_2 = conv3_d   # torch.zeros_like(conv3_d)
        conv3_d_c = torch.cat([conv3_d, feature_2], 1)
        conv3_d = self.conv3_c(conv3_d_c)
        conv3_d = self.conv3_5(self.conv3_4(self.conv3_3(self.conv3_2(conv3_d))))   # 1/4
        
        conv4_d = self.conv4_1(conv3_d)
        conv4_d = self.conv4_5(self.conv4_4(self.conv4_3(self.conv4_2(conv4_d))))   # 1/8

        # Decoder
        conv5_d = self.conv5_c(torch.cat([self.conv5_1(conv4_d), conv3_d], 1))
        conv5_d = self.conv5_5(self.conv5_4(self.conv5_3(self.conv5_2(conv5_d))))
        feature_2_out = conv5_d     # 1/4
        conv6_d = self.conv6_c(torch.cat([self.conv6_1(conv5_d), conv2_d], 1))
        conv6_d = self.conv6_5(self.conv6_4(self.conv6_3(self.conv6_2(conv6_d))))
        feature_1_out = conv6_d     # 1/2
        conv7_d = self.conv7_c(self.conv7_1(conv6_d))
        conv7_d = self.conv7_5(self.conv7_4(self.conv7_3(self.conv7_2(conv7_d))))
        feature_0_out = conv7_d     # 1/1
        output_img = self.img_prd(conv7_d) + x_in

        return output_img, feature_0_out, feature_1_out, feature_2_out


class IdemNet(nn.Module):
    def __init__(self):
        super(IdemNet, self).__init__()
        self.mtrnn = MTRNN()

    def forward(self, input_img):
        feature_0 = input_img.clone()
        feature_1 = input_img.clone()
        feature_2 = input_img.clone()
        deblur_result = input_img.clone()
        for i in range(4):
            output = self.mtrnn([input_img, feature_0, feature_1, feature_2])
            deblur_result, feature_0, feature_1, feature_2 = output
        
        return [deblur_result]
