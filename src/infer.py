import argparse
from pathlib import Path
import random

import numpy as np
import torch
import torch.nn.functional as F

from tqdm.auto import trange

import config
from data.utils import DataReader, get_dataset
from models.utils import get_model


@torch.inference_mode()
def main(args):
    args.world_size = 1

    # NOTE args.seed is offset per worker in get_adjusted_args_for_process
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    device = args.device
    if "cuda" in device:
        torch.cuda.set_device(torch.device(device))
    # torch.use_deterministic_algorithms(True)  # CUBLAS_WORKSPACE_CONFIG=:4096:8

    model = get_model(args).to(device).eval()

    result_path = Path(args.infer_result_path or args.exp_path) / args.infer_name
    result_path.mkdir(exist_ok=True)
    result_path = result_path / "result.npz"

    chkp_path = args.chkp_path or (Path(args.exp_path) / "ckpts" / "latest" / "main.pt")

    ckpt = torch.load(chkp_path, map_location=device)
    model.load_state_dict(
        {
            k.replace('_orig_mod.', '').replace('module.', ''): v for k, v in ckpt["model"].items()
        }
    )

    if args.compile:
        model = torch.compile(model)

    datareader = get_data_reader(args)

    results = None
    with torch.cuda.amp.autocast(
        enabled=args.dtype != 'float32',
        dtype=torch.bfloat16 if args.dtype == 'bfloat16' else torch.float16
    ):
        for i in trange(min(args.infer_batches, len(datareader))):
            print(f"Processing batch {i}")
            x, y = datareader.sample_batch()
            if "cuda" in torch.device(device).type:
                # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
                x = x.pin_memory().to(device, non_blocking=True)
                y = y.pin_memory().to(device, non_blocking=True)
            else:
                x = x.to(device)
                y = y.to(device)

            out = model(x, targets=y, get_logits=True)
            if results is None:
                results = to_cpu(F.softmax(out['logits'], dim=-1))
            else:
                results = torch.concat(
                    (
                        results,
                        to_cpu(F.softmax(out['logits'], dim=-1))
                    )
                )
    results = results.to(torch.float16).numpy()
    np.savez(result_path, probs=results)


def to_cpu(x):
    if isinstance(x, list):
        return list(map(to_cpu, x))
    if isinstance(x, dict):
        return {k : to_cpu(v) for k, v in x.items()}
    if isinstance(x, torch.Tensor):
        return x.cpu()
    return x


def get_args():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument(
        "--config_format", default="base", choices=config.registered_formats()
    )

    args, rem_args = parser.parse_known_args()

    return config.parse_args_with_format(
        format=args.config_format, base_parser=parser, args=rem_args, namespace=args
    )


def get_data_reader(args, verbose=True):
    data_srcs = get_dataset(args)
    val_reader = DataReader(
        data_src=data_srcs["val"],
        batch_size=args.batch_size,
        sequence_length=args.sequence_length,
        seed=args.data_seed,
        with_replacement=False,
        auto_shard=False,  # NOTE Identical Per Rank
        keep_in_ram=True,
    )

    return val_reader


if __name__ == "__main__":
    args = get_args()
    main(args)
