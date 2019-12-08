import numpy as np
import torch as t
from .torch_commons import *
from typing import List

class VariationalAutoencoder(Module):
    def __init__(self, input_x_to_determine_size, in_c:int, enc_out_c:List, enc_ks:List, enc_strides, enc_pads,
                 dec_out_c:List, dec_ks:List, dec_strides, dec_pads, dec_op_pads, z_dim):

        assert len(input_x_to_determine_size.shape) == 4
        assert len(enc_out_c) == len(enc_ks) == len(enc_strides) == len(enc_pads) and len(enc_out_c) > 1
        assert len(dec_out_c) == len(dec_ks) == len(dec_strides) == len(dec_pads) and len(enc_out_c) > 1

        enc_conv_layers = []
        enc_out_c.insert(0, in_c)

        for in_c_, out_c, ks, stride, pad in zip(enc_out_c[0:], enc_out_c[1:], enc_ks, enc_strides, enc_pads):
            enc_conv_layer = []
            enc_conv_layer.append(nn.Conv2d(in_c_, out_c, ks, stride, pad))
            enc_conv_layer.extend([nn.LeakyReLU(), nn.BatchNorm2d(out_c), nn.Dropout(.25)])
            enc_conv_layers.append(nn.Sequential(*enc_conv_layer))

        dec_conv_layers = []
        dec_out_c.insert(0, out_c)
        n_layers_decoder = range(len(dec_out_c))

        for in_c_, out_c, ks, stride, i, pad, op_pad \
                in zip(dec_out_c[0:], dec_out_c[1:], dec_ks, dec_strides, n_layers_decoder, dec_pads, dec_op_pads):

            dec_conv_layer = []
            dec_conv_layer.append(nn.ConvTranspose2d(in_c_, out_c, ks, stride, pad, op_pad))
            if i == len(dec_out_c) - 2:
                dec_conv_layer.append(nn.Sigmoid())
            else:
                dec_conv_layer.extend([nn.LeakyReLU(), nn.BatchNorm2d(out_c), nn.Dropout(.25)])
            dec_conv_layers.append(nn.Sequential(*dec_conv_layer))

        # run through part of model once to determine sizes dynamically
        x = nn.Sequential(*enc_conv_layers)(input_x_to_determine_size)
        pre_flatten_shape = x.shape
        x = Flatten(bs=True)(x)

        device = t.device('cuda') if t.cuda.is_available() else 'cpu'

        def sampling(args):
            mu, log_var = args
            std = t.exp(log_var / 2)
            return std * t.randn(mu.shape).float().to(device) + mu

        enc_conv_layers.append(Flatten(bs=True))
        enc_conv_layers.append(TwoTupleLinear(x.shape[1], z_dim))
        self.enc_conv_layers = nn.Sequential(*enc_conv_layers)
        self.lamdba_sampling_layer = Lambda(sampling)

        dec_conv_layers.insert(0, nn.Linear(z_dim, np.prod(pre_flatten_shape[1:])))
        dec_conv_layers.insert(1, View(pre_flatten_shape, bs=True))
        self.dec_conv_layers = nn.Sequential(*dec_conv_layers)

    def forward(self, x):
        mu, log_var = self.enc_conv_layers(x)
        x = self.lamdba_sampling_layer((mu, log_var))
        return self.dec_conv_layers(x), mu, log_var


