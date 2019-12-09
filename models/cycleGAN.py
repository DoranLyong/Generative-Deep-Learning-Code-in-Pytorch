import torch as t
from .torch_commons import *

def norm(module):
    if isinstance(module, nn.Conv2d) or isinstance(module, nn.ConvTranspose2d):
        t.nn.init.normal_(module.weight, mean=0, std=.02)
        t.nn.init.constant_(module.bias, .0)

def conv_layer(in_f, out_f, ks, s=None, p=None, op=None, activ=True, rps=None, transpose=False):
    sp_kwargs = {k: v for k, v in {'stride': s, 'padding': p, 'output_padding': op}.items() if v}
    layers = [nn.ReflectionPad2d(rps)] if rps else []
    if transpose:
        conv = nn.ConvTranspose2d(in_f, out_f, ks, **sp_kwargs)
    else:
        conv = nn.Conv2d(in_f, out_f, ks, **sp_kwargs)
    layers.extend([conv, nn.InstanceNorm2d(out_f)])
    if activ:
        layers.append(nn.ReLU(inplace=True))
    return nn.Sequential(*layers)


class ResBlock(Module):
    def __init__(self, in_f):
        self.conv1 = conv_layer(in_f, in_f, 3, rps=1)
        self.conv2 = conv_layer(in_f, in_f, 3, activ=False, rps=1)

    def forward(self, x):
        return x + self.conv2(self.conv1(x))


class GenResNet(Module):
    def __init__(self, first_c, n_res_blocks=1):
        self.model = nn.Sequential(conv_layer(first_c, 64, 7, rps=first_c),
                                   conv_layer(64, 128, 3, s=2, p=1),
                                   conv_layer(128, 256, 3, s=2, p=1),
                                   nn.Sequential(*[ResBlock(256) for _ in range(n_res_blocks)]),
                                   conv_layer(256, 128, 3, s=2, p=1, op=1, transpose=True),
                                   conv_layer(128, 64, 3, s=2, p=1, op=1, transpose=True),
                                   conv_layer(64, first_c, 7, activ=False, rps=first_c),
                                   nn.Tanh())

    def forward(self, x):
        return self.model(x)

class Critic(Module):
    def __init__(self, in_shape):
        c, h, w = in_shape

        def discriminator_block(in_f, out_f, norm=True):
            layers = [nn.Conv2d(in_f, out_f, 4, stride=2, padding=1)]
            if norm:
                layers.append(nn.InstanceNorm2d(out_f))
            layers.append(nn.LeakyReLU(0.2))
            return layers

        self.model = nn.Sequential(*discriminator_block(c, 64, norm=False),
                                   *discriminator_block(64, 128),
                                   *discriminator_block(128, 256),
                                   *discriminator_block(256, 512),
                                   nn.ZeroPad2d((1, 0, 1, 0)),
                                   nn.Conv2d(512, 1, 4, padding=1))

        sample_input = t.zeros(1, *in_shape)
        self.out_shape = self.model(sample_input).shape[1:]

    def forward(self, x):
        return self.model(x)
