import os
import sys
path = os.getcwd()
sys.path.append(os.path.join(path, 'odcore'))
import argparse
from odcore.utils.visualization import LossLog


def get_args_parser(add_help=True):
    parser = argparse.ArgumentParser(description='Args for loss drawing', add_help=add_help)
    parser.add_argument('--loss-log', default='', type=str, help='_loss.log file')
    return parser


def main(args):
    log = LossLog(args.loss_log)
    log.draw_loss()
    log.draw_sum_loss()
    log.draw_epoch_loss()
    log.draw_sum_epoch_loss()


if __name__ == '__main__':
    args = get_args_parser().parse_args()
    main(args)
