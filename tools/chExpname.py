import os
import sys
path = os.getcwd()
sys.path.append(os.path.join(path, '../odcore'))
os.chdir(os.path.join(path, '..'))
from odcore.utils.exp import Exp

exp = Exp('running_log/YOLOX_s2_1', is_main_process=True)
exp.rename('YOLOX_s2_poly')