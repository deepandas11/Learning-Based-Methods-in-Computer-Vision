
"""
All the additional models that were used for experimentation purposes have been included in here. 

Use: 
    Write the following lines of code on student_code if you want to use: 
    1. For SkipNet : 
    from Additional_Models import SkipNet

    2. For MixNet:
    from Additional_Models import MixNet

    3. For DresNet:
    from Additional_Models import DresNet

    Then set default_model to imported model.
"""

import torch.nn as nn
import torch.nn.functional as f


class SkipNet(nn.Module):
  # a simple CNN for image classifcation
  def __init__(self, conv_op=nn.Conv2d, num_classes=100):
    super(SkipNet, self).__init__()

    # introconv: 1/4 spatial map, channels: 3->64
    self.introconv = nn.Sequential(
      # conv1 block: 3x conv 3x3
      conv_op(3, 64, kernel_size=7, stride=2, padding=3),
      nn.BatchNorm2d(64),
      nn.ReLU(inplace=True),
      # max pooling 1/2
      nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
      conv_op(64, 64, kernel_size=3, stride=1, padding=1),
    )


    # bottleneck 1 layer, retains spatial, channels: 64->128
    self.bottleneck1 = nn.Sequential(
      conv_op(64, 64, kernel_size=1, stride=1, padding=0),
      nn.BatchNorm2d(64),
      nn.ReLU(inplace=True),
      conv_op(64, 64, kernel_size=3, stride=1, padding=1),
      nn.BatchNorm2d(64),
      nn.ReLU(inplace=True),
      conv_op(64, 128, kernel_size=1, stride=1, padding=0),
      nn.BatchNorm2d(128),
      # max pooling 1/2
      # nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
    )

    #identity connection for bottleneck 1
    self.id_bn1 = nn.Sequential(
        conv_op(64, 128, kernel_size=1, stride=1, bias=False),
    )

    # bottleneck 2 layer, retains spatial, channels: 128->256
    self.bottleneck2 = nn.Sequential(
      # conv3 block: simple bottleneck
      conv_op(128, 64, kernel_size=1, stride=1, padding=0),
      nn.BatchNorm2d(64),
      nn.ReLU(inplace=True),
      conv_op(64, 64, kernel_size=3, stride=1, padding=1),
      nn.BatchNorm2d(64),
      nn.ReLU(inplace=True),
      conv_op(64, 256, kernel_size=1, stride=1, padding=0),
      nn.BatchNorm2d(256),
      nn.ReLU(inplace=True),
      # max pooling 1/2
      #nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
    )

    # identity connection for bottleneck 2
    self.id_bn2 = nn.Sequential(
        conv_op(128, 256, kernel_size=1, stride=1, bias=False),
    )

    # bottleneck 3 layer, 1/2 spatial, channels: 256->512
    self.bottleneck3 = nn.Sequential(
        conv_op(256, 64, kernel_size=1, stride=1, padding=0),
        nn.BatchNorm2d(64),
        nn.ReLU(inplace=True),
        conv_op(64, 64, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(64),
        nn.ReLU(inplace=True),
        conv_op(64, 512, kernel_size=1, stride=1, padding=0),
        nn.BatchNorm2d(512),
        # nn.ReLU(inplace=True),
        # max pooling 1/2
        nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
    )

    # identity connection for bottleneck 3
    self.id_bn3 = nn.Sequential(
        conv_op(256, 512, kernel_size=1, stride=2, bias=False),
    )
    # global avg pooling + FC
    self.avgpool =  nn.AdaptiveAvgPool2d((1, 1))
    self.fc1 = nn.Linear(512, num_classes)

  def features(self, x):
    # 3x128x128 -> 64x32x32
    x = self.introconv(x)
    # 64x32x32 -> 128x32x32
    preserve_1 = x
    x = self.bottleneck1(x)
    preserve_1 = self.id_bn1(preserve_1)
    x = x+preserve_1
    x = nn.functional.relu(x,inplace=True)
    # 128x32x32 -> 256x16x16
    preserve_2 = x
    x = self.bottleneck2(x)
    preserve_2 = self.id_bn2(preserve_2)
    x = x + preserve_2
    x = nn.functional.relu(x,inplace=True)
    # 256x16x16 -> 512x8x8
    preserve_3 = x
    x = self.bottleneck3(x)
    preserve_3 = self.id_bn3(preserve_3)
    x = x + preserve_3
    x = nn.functional.relu(x,inplace=True)
    # 512x8x8 -> 512x8x8

    return x

  def logits(self, x):
    x = self.avgpool(x)
    x = x.view(x.size(0), -1)
    x = self.fc1(x)

    return x

  def forward(self, x):
    x = self.features(x)
    x = self.logits(x)

    return x

class InceptA(nn.Module):

    def __init__(self, conv_op = nn.Conv2d, in_planes=128):
        super(InceptA, self).__init__()

        self.branch0 = conv_op(in_planes, 96, kernel_size=1, stride=1)
        self.branch0bn = nn.BatchNorm2d(96)

        self.branch1 = nn.Sequential(
            conv_op(in_planes, 48, kernel_size=1, stride=1),
            nn.BatchNorm2d(48),
            conv_op(48, 64, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(64),
        )

        self.branch2 = nn.Sequential(
            conv_op(in_planes, 64, kernel_size=1, stride=1),
            nn.BatchNorm2d(64),
            conv_op(64, 96, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(96),
            conv_op(96, 96, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(96),
        )

        self.branch3 = nn.Sequential(
            nn.AvgPool2d(3, stride=1, padding=1, count_include_pad=False),
            conv_op(in_planes, 64, kernel_size=1, stride=1),
            nn.BatchNorm2d(64),
        )

    def forward(self, x):
        x0 = self.branch0(x)
        x0 = self.branch0bn(x0)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        out = torch.cat((x0, x1, x2, x3), 1)
        return out
class InceptB(nn.Module):

    def __init__(self, conv_op = nn.Conv2d, in_planes = 320):
        super(InceptB, self).__init__()

        self.branch0 = conv_op(in_planes, 360, kernel_size=3, stride=2)
        self.branch0bn = nn.BatchNorm2d(360)

        self.branch1 = nn.Sequential(
            conv_op(in_planes, 128, kernel_size=1, stride=1),
            nn.BatchNorm2d(128),
            conv_op(128, 128, kernel_size=3, stride=2),
            nn.BatchNorm2d(128),
            conv_op(128, 360, kernel_size=1, stride=1),
            nn.BatchNorm2d(360),
        )

        self.branch2 = nn.MaxPool2d(3, stride=2)

    def forward(self, x):
        x0 = self.branch0(x)
        x0 = self.branch0bn(x0)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        out = torch.cat((x0, x1, x2), 1)
        return out
class IncRes(nn.Module):

    def __init__(self, scale=1.0, conv_op = nn.Conv2d, in_planes=320):
        super(IncRes, self).__init__()

        self.scale = scale

        self.branch0 = conv_op(in_planes, 32, kernel_size=1, stride=1)
        self.branch0bn = nn.BatchNorm2d(32)

        self.branch1 = nn.Sequential(
            conv_op(in_planes, 32, kernel_size=1, stride=1),
            nn.BatchNorm2d(32),
            conv_op(32, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            conv_op(32, 32, kernel_size=1, stride=1),
            nn.BatchNorm2d(32),
        )

        self.branch2 = nn.Sequential(
            conv_op(in_planes, 32, kernel_size=1, stride=1),
            nn.BatchNorm2d(32),
            conv_op(32, 48, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(48),
            conv_op(48, 64, kernel_size=1, stride=1),
            nn.BatchNorm2d(64),
        )

        self.conv2d = nn.Conv2d(128, in_planes, kernel_size=1, stride=1)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        out = torch.cat((x0, x1, x2), 1)
        out = self.conv2d(out)
        out = out * self.scale + x
        out = self.relu(out)
        return out
class MixNet(nn.Module):

    def __init__(self, conv_op = nn.Conv2d, num_classes=100):
        super(MixNet, self).__init__()

        # introconv: 1/4 spatial map, channels: 3->128
        self.introconv = nn.Sequential(
            conv_op(3, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            # max pooling 1/2
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            conv_op(64, 128, kernel_size=3, stride=1, padding=1, dilation=2),
        )

        self.inception1 = InceptA(in_planes=128)
        self.incres = IncRes(in_planes=320)
        self.inception2 = InceptB(in_planes=320)

        self.bottleneck = nn.Sequential(
            conv_op(1040, 64, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            conv_op(64, 64, kernel_size=3, stride=1, padding=2, dilation=4),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            conv_op(64, 1024, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(1024),
        )
        self.id_bn = nn.Sequential(
            conv_op(1040, 1024, kernel_size=3, stride=1, padding=2, dilation=4, bias=False),
        )

        # global avg pooling + FC
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(1024, num_classes)


    def features(self,x):
        x = self.introconv(x)
        x = self.inception1(x)
        x = self.incres(x)
        x = self.inception2(x)
        preserve = x
        x = self.bottleneck(x)
        preserve = self.id_bn(preserve)
        x += preserve
        x = f.relu(x, inplace=True)

        return x

    def logits(self, x):
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)

        return x

    def forward(self, x):
        x = self.features(x)
        x = self.logits(x)

        return  x

class DresNet(nn.Module):

    def __init__(self, conv_op=nn.Conv2d, num_classes=100):
        super(DresNet, self).__init__()

        # introconv: 1/4 spatial map, channels: 3->64
        self.introconv = nn.Sequential(
            conv_op(3, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            # max pooling 1/2
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            # conv_op(64, 64, kernel_size=3, stride=1, padding=1),
        )

        # bottleneck 1 layer, retains spatial, channels: 64->128
        self.bottleneck1 = nn.Sequential(
            conv_op(64, 64, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            conv_op(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            conv_op(64, 128, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(128),
            # max pooling 1/2
            # nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        # identity connection for bottleneck 1
        self.id_bn1 = nn.Sequential(
            conv_op(64, 128, kernel_size=1, stride=1, bias=False),
        )

        # bottleneck 2 layer, 1/2 spatial, channels: 128->256
        self.bottleneck2 = nn.Sequential(
            conv_op(128, 64, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            conv_op(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            conv_op(64, 256, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(256),
            # nn.ReLU(inplace=True),
            # max pooling 1/2
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        # identity connection for bottleneck 2
        self.id_bn2 = nn.Sequential(
            conv_op(128, 256, kernel_size=1, stride=2, bias=False),
        )

        # bottleneck 3 layer, retains spatial, uses dilation, channels: 256->512
        self.bottleneck3 = nn.Sequential(
            conv_op(256, 64, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            conv_op(64, 64, kernel_size=3, stride=1, padding=1, dilation=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            conv_op(64, 512, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(512),
            # nn.ReLU(inplace=True),
            # max pooling 1/2
            # nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        # identity connection for bottleneck 3
        self.id_bn3 = nn.Sequential(
            conv_op(256, 512, kernel_size=1, stride=1, bias=False),
        )

        # bottleneck 4 layer, retains spatial, uses dilation, channels: 512->1024
        self.bottleneck4 = nn.Sequential(
            conv_op(512, 64, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            conv_op(64, 64, kernel_size=3, stride=1, padding=2, dilation=4),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            conv_op(64, 1024, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(1024),
            # nn.ReLU(inplace=True),
            # max pooling 1/2
            # nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        # identity connection for bottleneck 3
        self.id_bn4 = nn.Sequential(
            conv_op(512, 1024, kernel_size=1, stride=1, bias=False),
        )

        # global avg pooling + FC
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(1024, num_classes)

    def features(self, x):
        # 3x128x128 -> 64x32x32
        x = self.introconv(x)

        # 64x32x32 -> 128x32x32
        preserve_1 = x
        x = self.bottleneck1(x)
        preserve_1 = self.id_bn1(preserve_1)
        x = x + preserve_1
        x = nn.functional.relu(x, inplace=True)

        # 128x32x32 -> 256x16x16
        preserve_2 = x
        x = self.bottleneck2(x)
        preserve_2 = self.id_bn2(preserve_2)
        x = x + preserve_2
        x = nn.functional.relu(x, inplace=True)

        # Use dilation 2
        # 256x16x16 -> 512x16x16
        preserve_3 = x
        x = self.bottleneck3(x)
        preserve_3 = self.id_bn3(preserve_3)
        x = x + preserve_3
        x = nn.functional.relu(x, inplace=True)

        # Use dilation 4
        # 512x16x16 -> 1024x16x16
        preserve_4 = x
        x = self.bottleneck4(x)
        preserve_4 = self.id_bn4(preserve_4)
        x = x + preserve_4
        x = nn.functional.relu(x, inplace=True)

        return x

    def logits(self, x):
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)

        return x

    def forward(self, x):
        x = self.features(x)
        x = self.logits(x)

        return x

