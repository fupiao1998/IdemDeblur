import math
import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F


class convLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, activation='relu', norm=None):
        super(convLayer, self).__init__()

        bias = False if norm == 'BN' else True
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)
        if activation is not None:
            self.activation = getattr(torch, activation, 'relu')
        else:
            self.activation = None

        self.norm = norm
        if norm == 'BN':
            self.norm_layer = nn.BatchNorm2d(out_channels)
        elif norm == 'IN':
            self.norm_layer = nn.InstanceNorm2d(out_channels, track_running_stats=True)

    def forward(self, x):
        out = self.conv2d(x)

        if self.norm in ['BN', 'IN']:
            out = self.norm_layer(out)

        if self.activation is not None:
            out = self.activation(out)

        return out


class convGRU(nn.Module):
    """
    Generate a convolutional GRU cell
    Adapted from: https://github.com/jacobkimmel/pytorch_convgru/blob/master/convgru.py
    """
    def __init__(self, input_size, hidden_size, kernel_size):
        super().__init__()
        padding = kernel_size // 2
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.reset_gate = nn.Conv2d(input_size + hidden_size, hidden_size, kernel_size, padding=padding)
        self.update_gate = nn.Conv2d(input_size + hidden_size, hidden_size, kernel_size, padding=padding)
        self.out_gate = nn.Conv2d(input_size + hidden_size, hidden_size, kernel_size, padding=padding)

        init.orthogonal_(self.reset_gate.weight)
        init.orthogonal_(self.update_gate.weight)
        init.orthogonal_(self.out_gate.weight)
        init.constant_(self.reset_gate.bias, 0.)
        init.constant_(self.update_gate.bias, 0.)
        init.constant_(self.out_gate.bias, 0.)
        
    def forward(self, input_, prev_state):
        if prev_state is None:
            prev_state = torch.zeros_like(input_, device=input_.device)
        
        # data size is [batch, channel, height, width]
        stacked_inputs = torch.cat([input_, prev_state], dim=1)
        update = torch.sigmoid(self.update_gate(stacked_inputs))
        reset = torch.sigmoid(self.reset_gate(stacked_inputs))
        out_inputs = torch.tanh(self.out_gate(torch.cat([input_, prev_state * reset], dim=1)))
        new_state = prev_state * (1 - update) + out_inputs * update

        return new_state


class RecurrentconvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0, activation='relu', norm=None, before_conv=True):
        super(RecurrentconvLayer, self).__init__()

        self.before_conv = before_conv
        RecurrentBlock = convGRU
        if self.before_conv:
            self.conv = convLayer(in_channels, out_channels, kernel_size, stride, padding, activation, norm)
        self.recurrent_block = RecurrentBlock(input_size=out_channels, hidden_size=out_channels, kernel_size=3)

    def forward(self, x, prev_state):
        if self.before_conv:
            x = self.conv(x)
        state = self.recurrent_block(x, prev_state)
        x = state
        return x, state


class UEncoder_RNN_GRU(nn.Module):
    def __init__(self):
        super(UEncoder_RNN_GRU, self).__init__()
        # Conv1
        self.layer1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(),  # nn.LeakyReLU(0.1, inplace=True)
            nn.Conv2d(32, 32, kernel_size=3, padding=1)
            )
        self.layer3 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(),   # nn.LeakyReLU(0.1, inplace=True)
            nn.Conv2d(32, 32, kernel_size=3, padding=1)
            )
        # Conv2
        self.layer5 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1)   # Downsample second time 1/2
        self.layer6 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),   # nn.LeakyReLU(0.1, inplace=True)
            nn.Conv2d(64, 64, kernel_size=3, padding=1)
            )
        self.layer7 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),   # nn.LeakyReLU(0.1, inplace=True)
            nn.Conv2d(64, 64, kernel_size=3, padding=1)
            )
        # Conv3
        self.layer9 = nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1)   # Downsample third time 1/4
        self.layer10 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),   # nn.LeakyReLU(0.1, inplace=True)
            nn.Conv2d(128, 128, kernel_size=3, padding=1)
            )
        self.layer11 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),   # nn.LeakyReLU(0.1, inplace=True)
            nn.Conv2d(128, 128, kernel_size=3, padding=1)
            )
        self.gru_block = RecurrentconvLayer(128, 128, padding=1, before_conv=True)   # convgru
        # ConvGRU:    3117411-147456=2969955 -- 2.97M
        # ConvSRU:    2527459-147456=2380003 -- 2.38M
        # ConvSepGRU: 2822883-147456=2675427 -- 2.68M

    def forward(self, x):
        [state3, x_stage_1_de, x_stage_2_de, x_stage_3_de] = x
        # pdb.set_trace()
        # Conv1
        x = self.layer1(x_stage_3_de)
        x = self.layer2(x) + x
        x_stage_2 = self.layer3(x) + x  # (1, 32 channels)
        if x_stage_2_de is not None:
            x_stage_2_cat = torch.cat((x_stage_2, x_stage_2_de), 1)
        else:
            x_stage_2_cat = torch.cat((x_stage_2, x_stage_2), 1)
        # Conv2
        x = self.layer5(x_stage_2_cat)  # 1/2
        x = self.layer6(x) + x
        x_stage_1 = self.layer7(x) + x  # (1/2, 64 channels)
        if x_stage_1_de is not None:
            x_stage_1_cat = torch.cat((x_stage_1, x_stage_1_de), 1)
        else:
            x_stage_1_cat = torch.cat((x_stage_1, x_stage_1), 1)  
        # Conv3
        x = self.layer9(x_stage_1_cat)
        x = self.layer10(x) + x
        x_stage_3 = self.layer11(x) + x  # (1/4, 128 channels)
        
        x_stage_3, state3 = self.gru_block(x_stage_3, state3)    # IF use this, param 2084707 -> 3117411 (Toooooo many)
        return [state3, x_stage_1, x_stage_2, x_stage_3]


class UDecoder_cat_RNN_GRU(nn.Module):
    def __init__(self):
        super(UDecoder_cat_RNN_GRU, self).__init__()        
        # Deconv3
        self.layer13 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),  # nn.LeakyReLU(0.1, inplace=True)
            nn.Conv2d(128, 128, kernel_size=3, padding=1)
            )
        self.layer14 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),  # nn.LeakyReLU(0.1, inplace=True)
            nn.Conv2d(128, 128, kernel_size=3, padding=1)
            )
        #Deconv2
        self.layer17 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),  # nn.LeakyReLU(0.1, inplace=True)
            nn.Conv2d(128, 64, kernel_size=3, padding=1)
            )
        self.layer18 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),  # nn.LeakyReLU(0.1, inplace=True)
            nn.Conv2d(64, 64, kernel_size=3, padding=1)
            )
        #Deconv1
        self.layer21 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),  # nn.LeakyReLU(0.1, inplace=True)
            nn.Conv2d(64, 32, kernel_size=3, padding=1)
            )
        self.layer22 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(),  # nn.LeakyReLU(0.1, inplace=True)
            nn.Conv2d(32, 32, kernel_size=3, padding=1)
            )
        self.layer24 = nn.Conv2d(32, 3, kernel_size=3, padding=1)

        self.layer16 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
        self.layer20 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)

    def forward(self, x):
        [state3, x_stage_1, x_stage_2, x_stage_3] = x
        #Deconv3
        x = self.layer13(x_stage_3) + x_stage_3  # 1/4
        x = self.layer14(x) + x       # 1/4
        x_stage_1_de = self.layer16(x)   # x_stage_1_de(1/2, 64 channels)
        #Deconv2
        x = self.layer17(torch.cat((x_stage_1_de, x_stage_1), 1)) + x_stage_1_de
        x = self.layer18(x) + x
        x_stage_2_de = self.layer20(x)   # x_stage_2_de(1, 32 channels)
        #Deconv1
        x = self.layer21(torch.cat((x_stage_2_de, x_stage_2), 1)) + x_stage_2_de
        x = self.layer22(x) + x
        x_stage_3_de = self.layer24(x)   # x_stage_3_de(1, 3 channels)
        return [state3, x_stage_1_de, x_stage_2_de, x_stage_3_de]


def weight_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, 0.5*math.sqrt(2. / n))
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('BatchNorm') != -1:
        m.weight.data.fill_(1)
        m.bias.data.zero_()
    elif classname.find('Linear') != -1:
        n = m.weight.size(1)
        m.weight.data.normal_(0, 0.01)
        m.bias.data = torch.ones(m.bias.data.size())


class IdemNetStage3(nn.Module):
    def __init__(self):
        super(IdemNetStage3, self).__init__()
        self.encoder = UEncoder_RNN_GRU()
        self.decoder = UDecoder_cat_RNN_GRU()
 
    def forward(self, input_img):
        for i in range(6):
            if i == 0:
                deblur_feature = self.encoder([None, None, None, input_img])
            else:
                deblur_feature = self.encoder(input_img)
            deblur_result = self.decoder(deblur_feature)  # deblur_result = (state3, x_stage_1_de, x_stage_2_de, x_stage_3_de)

            if isinstance(deblur_result, list) and isinstance(input_img, list):
                deblur_result[-1] += input_img[-1]
            elif isinstance(deblur_result, list) and not isinstance(input_img, list):
                deblur_result[-1] += input_img
            else:
                deblur_result += input_img

            input_img = deblur_result

        if isinstance(deblur_result, list):
            return [deblur_result[-1]]
        else:
            return [deblur_result]


class IdemNetStage3NotShare(nn.Module):
    def __init__(self):
        super(IdemNetStage3NotShare, self).__init__()
        self.encoder_1 = UEncoder_RNN_GRU()
        self.decoder_1 = UDecoder_cat_RNN_GRU()

        self.encoder_2 = UEncoder_RNN_GRU()
        self.decoder_2 = UDecoder_cat_RNN_GRU()

        self.encoder_3 = UEncoder_RNN_GRU()
        self.decoder_3 = UDecoder_cat_RNN_GRU()

        self.encoder_4 = UEncoder_RNN_GRU()
        self.decoder_4 = UDecoder_cat_RNN_GRU()

        self.encoder_5 = UEncoder_RNN_GRU()
        self.decoder_5 = UDecoder_cat_RNN_GRU()

        self.encoder_6 = UEncoder_RNN_GRU()
        self.decoder_6 = UDecoder_cat_RNN_GRU()

    def forward(self, input_img):
        # import pdb; pdb.set_trace()
        deblur_feature_1 = self.encoder_1([None, None, None, input_img])
        deblur_result_1 = self.decoder_1(deblur_feature_1)
        deblur_result_1[-1] += input_img

        deblur_feature_2 = self.encoder_2(deblur_result_1)
        deblur_result_2 = self.decoder_2(deblur_feature_2)
        deblur_result_2[-1] += input_img

        deblur_feature_3 = self.encoder_3(deblur_result_2)
        deblur_result_3 = self.decoder_3(deblur_feature_3)
        deblur_result_3[-1] += input_img

        deblur_feature_4 = self.encoder_4(deblur_result_3)
        deblur_result_4 = self.decoder_4(deblur_feature_4)
        deblur_result_4[-1] += input_img

        deblur_feature_5 = self.encoder_5(deblur_result_4)
        deblur_result_5 = self.decoder_5(deblur_feature_5)
        deblur_result_5[-1] += input_img

        deblur_feature_6 = self.encoder_6(deblur_result_5)
        deblur_result_6 = self.decoder_6(deblur_feature_6)
        deblur_result_6[-1] += input_img

        return [deblur_result_6[-1]]
