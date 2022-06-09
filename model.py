import torch
import torch.nn as nn

class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channel, out_channel, stride=1, alpha=1.0):
        super(DepthwiseSeparableConv, self).__init__()
        # width multiplier for thinner models
        in_channel = int(in_channel * alpha)
        out_channel = int(out_channel * alpha)
        # depthwise conv
        self.conv1 = nn.Conv2d(in_channel, in_channel, kernel_size=3, stride=stride, padding=1, bias=False,
                               groups=in_channel)
        self.bn1 = nn.BatchNorm2d(in_channel)
        self.relu1 = nn.ReLU(inplace=True)
        # pointwise conv
        self.conv2 = nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channel)
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x):
        # depthwise conv
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        # pointwise conv
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)

        return out

class MobileNet(nn.Module):
    def __init__(self, alpha=1.0):
        super(MobileNet, self).__init__()
        # Conv / s2
        self.conv = nn.Conv2d(3, int(32*alpha), kernel_size=3, stride=2, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(int(32*alpha))
        self.relu = nn.ReLU(inplace=True)
        # depthwise conv and pointwise conv
        self.ds_conv_1 = DepthwiseSeparableConv(32, 64, 1, alpha)
        self.ds_conv_2 = DepthwiseSeparableConv(64, 128, 2, alpha)
        self.ds_conv_3 = DepthwiseSeparableConv(128, 128, 1, alpha)
        self.ds_conv_4 = DepthwiseSeparableConv(128, 256, 2, alpha)
        self.ds_conv_5 = DepthwiseSeparableConv(256, 256, 1, alpha)
        self.ds_conv_6 = DepthwiseSeparableConv(256, 512, 2, alpha)
        self.ds_conv_7_1 = DepthwiseSeparableConv(512, 512, 1, alpha)
        self.ds_conv_7_2 = DepthwiseSeparableConv(512, 512, 1, alpha)
        self.ds_conv_7_3 = DepthwiseSeparableConv(512, 512, 1, alpha)
        self.ds_conv_7_4 = DepthwiseSeparableConv(512, 512, 1, alpha)
        self.ds_conv_7_5 = DepthwiseSeparableConv(512, 512, 1, alpha)
        self.ds_conv_8 = DepthwiseSeparableConv(512, 1024, 2, alpha)
        self.ds_conv_9 = DepthwiseSeparableConv(1024, 1024, 2, alpha)
        # Avg Pool / s1
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        # FC / s1
        self.fc = nn.Linear(int(1024*alpha), 1)

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)

        out = self.ds_conv_1(out)
        out = self.ds_conv_2(out)
        out = self.ds_conv_3(out)
        out = self.ds_conv_4(out)
        out = self.ds_conv_5(out)
        out = self.ds_conv_6(out)
        out = self.ds_conv_7_1(out)
        out = self.ds_conv_7_2(out)
        out = self.ds_conv_7_3(out)
        out = self.ds_conv_7_4(out)
        out = self.ds_conv_7_5(out)
        out = self.ds_conv_8(out)
        out = self.ds_conv_9(out)

        out = self.global_avg_pool(out).squeeze(2).squeeze(2)
        out = self.fc(out)

        return out





if __name__ == '__main__':
    x = torch.ones(size=(8, 3, 224, 224))
    mobilenet = MobileNet()
    out = mobilenet(x)
    print(out.shape)