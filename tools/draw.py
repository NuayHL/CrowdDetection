import os
path = os.getcwd()
os.chdir(os.path.join(path, '..'))

import matplotlib
matplotlib.use('TkAgg')

import matplotlib.pyplot as plt
from odcore.utils.visualization import ValLog

def val_compare(*val_files, metric_type='mr', short_exp_name=True):
    metric_name = 'IoU.5' if metric_type == 'coco' else 'AP'
    val_logs = list()
    for val_file in val_files:
        if short_exp_name:
            val_file = 'running_log/'+val_file+'/'+val_file+'_val.log'
        temp_log_obj = ValLog(val_file)
        if metric_type == 'mr':
            temp_log_obj.mr_val(zero_start=False)
        elif metric_type == 'coco':
            temp_log_obj.coco_val(zero_start=False)
        elif metric_type == 'mip':
            metric_name = 'mAP'
            temp_log_obj.mip_val(zero_start=False)
        else:
            raise NotImplementedError
        val_logs.append(temp_log_obj)

    fig, ax = plt.subplots()

    for val_log in val_logs:
        ax.plot(*val_log.data[metric_name])

    fig.legend(val_files)

    ax.set(xlabel="Epochs", ylabel=metric_name, title="Compare...")
    ax.grid()
    fig.legend()
    plt.show()

if __name__ == '__main__':
    val_compare('YOLOX_R_k9_0.2_lb',
                'YOLOX_R_k9_0.2_nb',
                metric_type='mip')



