import os
import sys
path = os.getcwd()
sys.path.append(os.path.join(path, 'odcore'))
import argparse
from odcore.utils.visualization import draw_coco_eval


def get_args_parser(add_help=True):
    parser = argparse.ArgumentParser(description='Args for val drawing', add_help=add_help)
    parser.add_argument('--val-log', default='', type=str, help='full path of _loss.log file')
    parser.add_argument('-exp', default='', type=str, help='experiment name')
    return parser


def main(args):
    if args.val_log != '':
        draw_coco_eval(args.val_log)
    else:
        draw_coco_eval('running_log/%s/%s_val.log' % (args.exp, args.exp))

if __name__ == '__main__':
    args = get_args_parser().parse_args()
    main(args)