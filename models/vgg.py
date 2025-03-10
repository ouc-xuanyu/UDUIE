import torch
import torch.nn as nn

# ------------------------------------------------------------------------------
# 暴露接口
__all__ = [
    'VGG', 'vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn',
    'vgg19_bn', 'vgg19',
]

# ------------------------------------------------------------------------------
# 预训练权重下载地址
model_urls = {
    'vgg11': 'https://download.pytorch.org/models/vgg11-bbd30ac9.pth',
    'vgg13': 'https://download.pytorch.org/models/vgg13-c768596a.pth',
    'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth',
    'vgg19': 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth',
    'vgg11_bn': 'https://download.pytorch.org/models/vgg11_bn-6002323d.pth',
    'vgg13_bn': 'https://download.pytorch.org/models/vgg13_bn-abd245e5.pth',
    'vgg16_bn': 'https://download.pytorch.org/models/vgg16_bn-6c64b313.pth',
    'vgg19_bn': 'https://download.pytorch.org/models/vgg19_bn-c79401a0.pth',
}

# ------------------------------------------------------------------------------

class VGG(nn.Module):
    '''
    VGG通用网络模型
    输入features为网络的特征提取部分网络层列表
    分类数为 1000
    '''
    def __init__(self, features, num_classes=1000, init_weights=True):
        super(VGG, self).__init__()

        # 特征提取部分
        self.features = features

        # 自适应平均池化，特征图池化到 7×7 大小
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))

        # 分类部分
        # self.classifier = nn.Sequential(
        #     nn.Linear(512 * 7 * 7, 4096),   # 512*7*7 --> 4096
        #     nn.ReLU(True),
        #     nn.Dropout(),
        #     nn.Linear(4096, 4096),          # 4096 --> 4096
        #     nn.ReLU(True),
        #     nn.Dropout(),
        #     nn.Linear(4096, num_classes),   # 4096 --> 1000
        # )

        # 权重初始化
        if init_weights:
            self._initialize_weights()

    def forward(self, x):

        # 特征提取
        x = self.features(x)
        # 自适应平均池化
        x = self.avgpool(x)
        # 特征图展平成向量
        x = torch.flatten(x, 1)
        # 分类器分类输出
        #x = self.classifier(x)
        return x

    def _initialize_weights(self):
        '''
        权重初始化
        '''
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # 卷积层使用 kaimming 初始化
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
                # 偏置初始化为0
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            # 批归一化层权重初始化为1 
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            # 全连接层权重初始化
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


# ------------------------------------------------------------------------------
def make_layers(cfg, batch_norm=False):
    '''
    根据配置表，返回模型层列表
    '''
    layers = [] # 层列表初始化

    in_channels = 3 # 输入3通道图像

    # 遍历配置列表
    for v in cfg:
        if v == 'M': # 添加池化层
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else: # 添加卷积层

            # 3×3 卷积
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)

            # 卷积-->批归一化（可选）--> ReLU激活
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]

            # 通道数方面，下一层输入即为本层输出
            in_channels = v

    # 以sequencial类型返回模型层列表
    return nn.Sequential(*layers)


# 网络参数配置表
'''
数字代表通道数，如 64 表示输出 64 通道特征图，对应于论文中的 Conv3-64;
M 代表最大池化操作，对应于论文中的 maxpool 
A-LRN使用了局部归一化响应，C网络存在1×1卷积，这两个网络比较特殊，所以排除在配置表中
'''
cfgs = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

# ------------------------------------------------------------------------------

def _vgg(arch, cfg, batch_norm, pretrained, progress, **kwargs):
    '''
    通用网络构造器，主要实现网络模型生成，以及预训练权重的导入
    '''
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG(make_layers(cfgs[cfg], batch_norm=batch_norm), **kwargs)

    if pretrained:
        state_dict = torch.load('https://download.pytorch.org/models/vgg16-397923af.pth')
        model.load_state_dict(state_dict, strict=False)
    return model

# ------------------------------------------------------------------------------

def vgg11(pretrained=False, progress=True, **kwargs):
    r"""VGG 11-layer model (configuration "A") from
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vgg('vgg11', 'A', False, pretrained, progress, **kwargs)


def vgg11_bn(pretrained=False, progress=True, **kwargs):
    r"""VGG 11-layer model (configuration "A") with batch normalization
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vgg('vgg11_bn', 'A', True, pretrained, progress, **kwargs)


def vgg13(pretrained=False, progress=True, **kwargs):
    r"""VGG 13-layer model (configuration "B")
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vgg('vgg13', 'B', False, pretrained, progress, **kwargs)


def vgg13_bn(pretrained=False, progress=True, **kwargs):
    r"""VGG 13-layer model (configuration "B") with batch normalization
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vgg('vgg13_bn', 'B', True, pretrained, progress, **kwargs)


def vgg16(pretrained=False, progress=True, **kwargs):
    r"""VGG 16-layer model (configuration "D")
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vgg('vgg16', 'D', False, pretrained, progress, **kwargs)


def vgg16_bn(pretrained=False, progress=True, **kwargs):
    r"""VGG 16-layer model (configuration "D") with batch normalization
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vgg('vgg16_bn', 'D', True, pretrained, progress, **kwargs)


def vgg19(pretrained=False, progress=True, **kwargs):
    r"""VGG 19-layer model (configuration "E")
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vgg('vgg19', 'E', False, pretrained, progress, **kwargs)


def vgg19_bn(pretrained=False, progress=True, **kwargs):
    r"""VGG 19-layer model (configuration 'E') with batch normalization
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vgg('vgg19_bn', 'E', True, pretrained, progress, **kwargs)