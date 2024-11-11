import os
from .infer import parse_args as parse_args_infer


def parse_args(base_parser, args, namespace):
    parser = base_parser

    parser.add_argument("--infer-loss-dims-num", default=2, type=int)
    parser.add_argument("--infer-loss-magnitude", default=1e-5, type=float)
    parser.add_argument("--infer-loss-points", default=300, type=int)
    return parse_args_infer(base_parser, args, namespace)
