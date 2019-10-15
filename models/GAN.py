import numpy as np
import torch as t
from .torch_commons import *
from typing import List
from functools import partial

def norm(module):
    t.nn.init.normal_(module.weight, mean=0, std=.02)
    t.nn.init.normal_(module.bias, mean=0, std=.02)
    return module


def build_generator(gen_init_size, gen_upsample_flags, gen_c: List, gen_ks: List, gen_strides, gen_pads, z_dim, output_act=nn.Sigmoid, leaky_relu=False, first_gen_c=None, leaky_relu_a=.2):
    relu = partial(nn.LeakyReLU, leaky_relu_a) if leaky_relu else nn.ReLU
    gen_layers = [norm(nn.Linear(z_dim, np.prod(gen_init_size)))]
    gen_layers.extend([relu(), nn.BatchNorm1d(np.prod(gen_init_size), momentum=.9), View((1,) + gen_init_size, bs=True)])
    gen_c.insert(0, first_gen_c or gen_c[-2])

    for usf, in_c, out_c, ks, stride, pad, i in \
            zip(gen_upsample_flags, gen_c[0:], gen_c[1:], gen_ks, gen_strides, gen_pads, range(len(gen_c))):

        if usf:
            gen_layers.append(nn.UpsamplingBilinear2d(scale_factor=2))
            gen_layers.append(norm(nn.Conv2d(in_c, out_c, ks, stride, pad)))
        else:
            gen_layers.append(norm(nn.ConvTranspose2d(in_c, out_c, ks, stride, pad)))

        if i == len(gen_c) - 2:
            gen_layers.append(output_act())
        else:
            gen_layers.extend([relu(), nn.BatchNorm2d(out_c, momentum=.9)])

    return nn.Sequential(*gen_layers)


def build_critic(input_x_to_determine_size, critic_c: List, critic_ks: List, critic_strides, critic_pads,
                 wgan=False, wgan_gp=False, leaky_relu=False, leaky_relu_a=.2):
    relu = partial(nn.LeakyReLU, leaky_relu_a) if leaky_relu else nn.ReLU
    critic_layers = []

    for in_c, out_c, ks, stride, pad, i in \
            zip(critic_c[0:], critic_c[1:], critic_ks, critic_strides, critic_pads, range(len(critic_c))):

        critic_layers.append(norm(nn.Conv2d(in_c, out_c, ks, stride, pad)))
        critic_layers.append(relu())
        if i > 0 and not wgan_gp:
            critic_layers.append(nn.BatchNorm2d(out_c))
        critic_layers.append(nn.Dropout(.4))

    critic_layers.append(Flatten(bs=True))
    critic_lin_in_shape = np.prod(nn.Sequential(*critic_layers)(input_x_to_determine_size).shape[1])
    if wgan or wgan_gp:
        critic_layers.extend([norm(nn.Linear(critic_lin_in_shape, 1))])
    else:
        critic_layers.extend([norm(nn.Linear(critic_lin_in_shape, 1)), nn.Sigmoid()])

    return nn.Sequential(*critic_layers)

