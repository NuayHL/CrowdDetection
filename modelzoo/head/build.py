"""
head init format:
        def __init__(self, classes, anchors_per_grid, p3c)
"""

from utility.register import Register
HeadRegister = Register('head')

def build_head(name):
    try:
        return HeadRegister[name]
    except:
        raise NotImplementedError('No head named %s' % name)
