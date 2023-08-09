"""Compute perplexity for a set of models."""

import argparse

MODELS=[]

# Put this here so readers can see the command-line arguments first :)
def build_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--layers', type=int, dest='layers', default=1000000,
        help='Maximum number of layers to put on GPU.')
    return parser


def main(argv=None):
    args = build_args().parse_args(argv)
    pass


if __name__ = '__main__':
    main()


