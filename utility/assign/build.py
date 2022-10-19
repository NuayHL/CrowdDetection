# this .py is for the assignment methods
from utility.register import Register

AssignRegister = Register('assign')

def get_assign_method(config, device):
    type = config.model.assignment_type
    try:
        assignment = AssignRegister[type]
    except:
        raise NotImplementedError('%s assign type not found' % type)

    return assignment(config, device)
