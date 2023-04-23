import torch
import torch.nn as nn
import torch.nn.functional as F



def mish(x):
    """
    Mish: A Self Regularized Non-Monotonic Neural Activation Function (https://arxiv.org/abs/1908.08681)
    Description: Cette fonction calcule la fonction d'activation Mish, qui est une fonction régularisée auto-ajustée et non-monotone.
    Entrée: x (torch.Tensor) - Un tenseur de PyTorch.
    Sortie: Un tenseur de PyTorch.
    """
    return x * torch.tanh(F.softplus(x))


class PSBatchNorm2d(nn.BatchNorm2d):
    """
    How Does BN Increase Collapsed Neural Network Filters? (https://arxiv.org/abs/2001.11216)
    Description: Cette classe étend la classe nn.BatchNorm2d en ajoutant un paramètre alpha 
        qui est ajouté à la sortie de la couche de normalisation de lot. 
    Attributs:
        alpha (float) - Le paramètre alpha qui est ajouté à la sortie de la couche de normalisation de lot.
    Méthodes:
        forward(self, x) - Applique la normalisation de lot à l'entrée x et ajoute self.alpha à la sortie.
    """

    def __init__(self, num_features, alpha=0.1, eps=1e-05, momentum=0.001, affine=True, track_running_stats=True):
        super().__init__(num_features, eps, momentum, affine, track_running_stats)
        self.alpha = alpha

    def forward(self, x):
        return super().forward(x) + self.alpha


class BasicBlock(nn.Module):
    """
    Description: Cette classe implémente un bloc de base de la Wide ResNet. 
        Chaque bloc est composé de deux couches de convolution avec une couche de normalisation de lot 
        et une fonction d'activation LeakyReLU entre elles. Si la résolution spatiale de l'entrée change, 
        le bloc utilise une convolution de projection pour changer la taille de la représentation. 
        Il peut également appliquer un taux d'abandon à la sortie de la première couche de convolution.
    Attributs:
        bn1 (nn.BatchNorm2d) - La première couche de normalisation de lot.
        relu1 (nn.LeakyReLU) - La première fonction d'activation LeakyReLU.
        conv1 (nn.Conv2d) - La première couche de convolution.
        bn2 (nn.BatchNorm2d) - La deuxième couche de normalisation de lot.
        relu2 (nn.LeakyReLU) - La deuxième fonction d'activation LeakyReLU.
        conv2 (nn.Conv2d) - La deuxième couche de convolution.
        drop_rate (float) - Le taux d'abandon à appliquer à la sortie de la première couche de convolution.
        equalInOut (bool) - Vrai si l'entrée et la sortie ont le même nombre de canaux.
        convShortcut (nn.Conv2d ou None) - La convolution de projection utilisée pour changer la taille de la représentation si la résolution spatiale de l'entrée change. Si l'entrée et la sortie ont le même nombre de canaux, cela est None.
        activate_before_residual (bool) - Vrai si la première couche de normalisation de lot et la première fonction d'activation LeakyReLU doivent être appliquées avant la projection résiduelle.
    Méthodes:
        forward(self, x) - Calcule la sortie du bloc pour l'entrée x.
        Classe NetworkBlock
    Description: Cette classe implémente un bloc de réseau de la Wide ResNet. 
        Il est composé de plusieurs blocs de base (BasicBlock) empilés les uns sur les autres.
    """
    def __init__(self, in_planes, out_planes, stride, drop_rate=0.0, activate_before_residual=False):
        super(BasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes, momentum=0.001)
        self.relu1 = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes, momentum=0.001)
        self.relu2 = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.drop_rate = drop_rate
        self.equalInOut = (in_planes == out_planes)
        self.convShortcut = (not self.equalInOut) and nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                                                                padding=0, bias=False) or None
        self.activate_before_residual = activate_before_residual

    def forward(self, x):
        if not self.equalInOut and self.activate_before_residual == True:
            x = self.relu1(self.bn1(x))
        else:
            out = self.relu1(self.bn1(x))
        out = self.relu2(self.bn2(self.conv1(out if self.equalInOut else x)))
        if self.drop_rate > 0:
            out = F.dropout(out, p=self.drop_rate, training=self.training)
        out = self.conv2(out)
        return torch.add(x if self.equalInOut else self.convShortcut(x), out)


class NetworkBlock(nn.Module):
    def __init__(self, nb_layers, in_planes, out_planes, block, stride, drop_rate=0.0, activate_before_residual=False):
        super(NetworkBlock, self).__init__()
        self.layer = self._make_layer(
            block, in_planes, out_planes, nb_layers, stride, drop_rate, activate_before_residual)

    def _make_layer(self, block, in_planes, out_planes, nb_layers, stride, drop_rate, activate_before_residual):
        layers = []
        for i in range(int(nb_layers)):
            layers.append(block(i == 0 and in_planes or out_planes, out_planes,
                                i == 0 and stride or 1, drop_rate, activate_before_residual))
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.layer(x)


class WideResNet(nn.Module):
    def __init__(self, num_classes, depth=28, widen_factor=2, drop_rate=0.0):
        super(WideResNet, self).__init__()
        channels = [16, 16*widen_factor, 32*widen_factor, 64*widen_factor]
        assert((depth - 4) % 6 == 0)
        n = (depth - 4) / 6
        block = BasicBlock
        # 1st conv before any network block
        self.conv1 = nn.Conv2d(3, channels[0], kernel_size=3, stride=1,
                               padding=1, bias=False)
        # 1st block
        self.block1 = NetworkBlock(
            n, channels[0], channels[1], block, 1, drop_rate, activate_before_residual=True)
        # 2nd block
        self.block2 = NetworkBlock(
            n, channels[1], channels[2], block, 2, drop_rate)
        # 3rd block
        self.block3 = NetworkBlock(
            n, channels[2], channels[3], block, 2, drop_rate)
        # global average pooling and classifier
        self.bn1 = nn.BatchNorm2d(channels[3], momentum=0.001)
        self.relu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.fc = nn.Linear(channels[3], num_classes)
        self.channels = channels[3]

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight,
                                        mode='fan_out',
                                        nonlinearity='leaky_relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0.0)

    def forward(self, x):
        out = self.conv1(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.relu(self.bn1(out))
        out = F.adaptive_avg_pool2d(out, 1)
        out = out.view(-1, self.channels)
        return self.fc(out)


def build_wideresnet(depth, widen_factor, dropout, num_classes):
    return WideResNet(depth=depth,
                      widen_factor=widen_factor,
                      drop_rate=dropout,
                      num_classes=num_classes)
