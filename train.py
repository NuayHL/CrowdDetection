import sys
import os
path = os.getcwd()
sys.path.append(os.path.join(path,'odcore'))
from engine.train import Train
from args import get_args_parser

def main(args):
    print(args)

if __name__ == '__main__':
    args = get_args_parser().parse_args()
    main(args)