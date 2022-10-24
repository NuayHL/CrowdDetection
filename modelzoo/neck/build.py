import torch

from utility.register import Register

NeckRegister = Register('neck')

def build_neck(name):
    """return neckClass"""
    try:
        return NeckRegister[name]
    except:
        raise NotImplementedError('No neck named %s' % name)
