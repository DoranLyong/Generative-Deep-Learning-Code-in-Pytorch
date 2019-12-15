# from fastai https://github.com/fastai/fastai_dev/blob/master/dev/local/core.py#L23
# from fastai https://github.com/fastai/fastai/blob/master/fastai/torch_core.py#L87
import torch.nn as nn
import functools
import torch as t
import torch.nn.functional as F
from torchvision import transforms, models

class PrePostInitMeta(type):
    "A metaclass that calls optional `__pre_init__` and `__post_init__` methods"
    def __new__(mcs, name, bases, dct):
        x = super().__new__(mcs, name, bases, dct)
        old_init = x.__init__

        @functools.wraps(old_init)
        def _init(self, *args, **kwargs):
            self.__pre_init__()
            old_init(self, *args, **kwargs)
            self.__post_init__()

        x.__init__ = _init
        def _pass(self): pass
        if not hasattr(x, '__pre_init__'):
            x.__pre_init__ = _pass
        if not hasattr(x, '__post_init__'):
            x.__post_init__ = _pass
        return x

class Module(nn.Module, metaclass=PrePostInitMeta):
    "Same as `nn.Module`, but no need for subclasses to call `super().__init__`"
    def __pre_init__(self):
        super().__init__()

    def __init__(self):
        pass

class Lambda(Module):
    def __init__(self, func):
        self.func = func

    def forward(self, x):
        return self.func(x)

class Flatten(Module):
    def __init__(self, bs=False):
        self.bs = bs

    def forward(self, x):
        return x.view(x.shape[0], -1) if self.bs else x.view(-1)

class View(Module):
    def __init__(self, shape, bs=False):
        self.bs = bs
        self.shape = shape

    def forward(self, x):
        return x.view(x.shape[0], *self.shape[1:]) if self.bs else x.view(*self.shape)

class TwoTupleLinear(Module):
    def __init__(self, in_shape, out_shape):
        self.in_shape = in_shape
        self.out_shape = out_shape
        self.lin1 = nn.Linear(in_shape, out_shape)
        self.lin2 = nn.Linear(in_shape, out_shape)

    def forward(self, x):
        return self.lin1(x), self.lin2(x)
