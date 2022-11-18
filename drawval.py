import os
import sys
path = os.getcwd()
sys.path.append(os.path.join(path, 'odcore'))
import argparse
from odcore.utils.visualization import ValLog


def get_args_parser(add_help=True):
    parser = argparse.ArgumentParser(description='Args for val drawing', add_help=add_help)
    parser.add_argument('--val-log', default='', type=str, help='full path of _loss.log file')
    parser.add_argument('-exp', default='', type=str, help='experiment name')
    parser.add_argument('-type', default='coco', type=str, help='Val metric type: coco or mr')
    return parser


def main(args):
    if args.val_log != '':
        vallog = ValLog(args.val_log)
    else:
        vallog = ValLog('running_log/%s/%s_val.log' % (args.exp, args.exp))
    if args.type == 'coco':
        vallog.coco_draw(zero_start=True)
    else:
        vallog.mr_draw(zero_start=True)

if __name__ == '__main__':
    args = get_args_parser().parse_args()
    main(args)
