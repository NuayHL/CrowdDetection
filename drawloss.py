import os
import sys
path = os.getcwd()
sys.path.append(os.path.join(path, 'odcore'))
import argparse
from odcore.utils.visualization import LossLog


def get_args_parser(add_help=True):
    parser = argparse.ArgumentParser(description='Args for loss drawing', add_help=add_help)
    parser.add_argument('--loss-log', default='', type=str, help='_loss.log file')
    parser.add_argument('-exp', default='', type=str, help='experiment name')
    return parser


def main(args):
    if args.loss_log != '':
        loss_log = args.loss_log
    else:
        loss_log = 'running_log/%s/%s_loss.log' % (args.exp, args.exp)
    log = LossLog(loss_log)
    log.draw_loss()
    log.draw_sum_loss()
    log.draw_epoch_loss()
    log.draw_sum_epoch_loss()


if __name__ == '__main__':
    args = get_args_parser().parse_args()
    main(args)
