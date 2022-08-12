import os
import sys
path = os.getcwd()
sys.path.append(os.path.join(path,'odcore'))
from odcore.utils.visualization import draw_loss

draw_loss('running_log/YOLOv3_siou_1/YOLOv3_siou_1_loss.log')