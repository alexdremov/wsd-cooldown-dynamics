import os


def parse_args(base_parser, args, namespace):
    parser = base_parser
    # General training params
    parser.add_argument("--data-seed", default=1337, type=int)
    parser.add_argument("--infer-batches", default=256, type=int)
    parser.add_argument("--batch-size", default=50, type=int)
    parser.add_argument("--device", default="cuda:0", type=str)
    parser.add_argument("--seed", default=0, type=int)

    # exp
    parser.add_argument("--exp-path", type=str)
    parser.add_argument("--infer-name", default=None, type=str)
    parser.add_argument("--infer-result-path", default=None, type=str)
    parser.add_argument("--chkp-path", default=None, type=str)

    # Dataset params
    parser.add_argument("--datasets-dir", type=str, default=os.path.join(os.environ.get('PERSISTENT_PATH', './'), 'datasets'))
    parser.add_argument(
        "--dataset",
        default="slimpajama",
        choices=[
            "wikitext",
            "shakespeare-char",
            "arxiv",
            "arxiv2000",
            "arxiv+wiki",
            "openwebtext2",
            "redpajama",
            "slimpajama",
            "slimpajama_chunk1",
            "redpajamav2",
            "fineweb",
        ],
    )
    parser.add_argument(
        "--tokenizer", default="gpt2", type=str, choices=["gpt2", "mistral"]
    )
    parser.add_argument("--vocab-size", default=50304, type=int)
    parser.add_argument(
        "--data-in-ram", action="store_true"
    )  # force the data to RAM, mostly useless except for openwebtext2
    # Model params
    parser.add_argument(
        "--model",
        default="llama",
        choices=[
            "base",
            "llama",
        ],
    )
    parser.add_argument("--parallel-block", action="store_true")
    parser.add_argument(
        "--use-pretrained", default="none", type=str
    )  # 'none', 'gpt-2' or a path to the pretraind model
    parser.add_argument("--from-dense", action="store_true")
    parser.add_argument("--init-std", default=0.02, type=float)
    parser.add_argument("--dropout", default=0.0, type=float)
    parser.add_argument("--n-head", default=12, type=int)
    parser.add_argument("--n-layer", default=24, type=int)  # depths in att + ff blocks
    parser.add_argument("--sequence-length", default=512, type=int)
    parser.add_argument(
        "--n-embd", default=768, type=int  # embedding size / hidden size ...
    )
    parser.add_argument(
        "--multiple-of",  # make SwiGLU hidden layer size multiple of large power of 2
        default=256,
        type=int,
    )
    parser.add_argument("--rmsnorm-eps", default=1e-5, type=float)
    parser.add_argument(
        "--dtype",
        default="bfloat16",
        type=str,
        choices=["float32", "float16", "bfloat16"],
    )
    parser.add_argument("--bias", default=False, type=bool)
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--mlp-dim-exp-factor", default=1.0, type=float)
    return parser.parse_args(args, namespace)
