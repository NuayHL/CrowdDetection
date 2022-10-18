from utility.register import Register

BackboneRegister = Register('backbone')


def build_backbone(name):
    """return backboneClass, output p3c"""
    try:
        return BackboneRegister[name]
    except:
        raise NotImplementedError('No backbone named %s' % name)

