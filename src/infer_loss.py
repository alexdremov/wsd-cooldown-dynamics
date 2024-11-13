import argparse
from pathlib import Path
import random

import numpy as np
import torch
import json
import torch.nn.functional as F

from tqdm.auto import trange

import config
from data.utils import DataReader, get_dataset
from models.utils import get_model


@torch.inference_mode()
def direction_dot(a, b):
    result = 0
    for k in a:
        result += (a[k] * b[k].to(a[k])).sum().item()
    return result


@torch.inference_mode()
def direction_mul(c, b):
    return {
        k: c * v for k, v in b.items()
    }


@torch.inference_mode()
def direction_add(a, b):
    return {
        k: a[k] + v.to(a[k]) for k, v in b.items()
    }


@torch.inference_mode()
def direction_sub(a, b):
    return {
        k: a[k] - v.to(a[k]) for k, v in b.items()
    }


def is_weight_filtered(k):
    return ('.ln_1.' in k) or ('.ln_2.' in k)


@torch.inference_mode()
def get_delta(basis, coords):
    base = {
        k: torch.zeros_like(w) for k, w in basis[0].items()
    }
    for c, vector in zip(coords, basis):
        base = direction_add(base, direction_mul(c, vector))
    return base


def gram_shmidt(initial_basis):
    orthogonal_basis = []

    for i, ui in enumerate(initial_basis):
        new_vector = ui
        for j, vi in enumerate(orthogonal_basis):
            new_vector = direction_add(
                new_vector,
                direction_mul(-direction_dot(vi, ui) / direction_dot(vi, vi), vi)
            )
        orthogonal_basis.append(new_vector)

    for i, u in enumerate(orthogonal_basis):
        for j, v in enumerate(orthogonal_basis):
            if i == j:
                continue
            basis_dot = direction_dot(u, v)
            assert abs(basis_dot) < 1e-2, f"Got nonorthogonal basis {basis_dot = }"
    return orthogonal_basis


def normalize_directions(state_dict, directions, filtering=None):
    return [
        {
            k: v * (0.0 if filtering is not None and filtering(k) else (torch.norm(state_dict[k]) / torch.norm(v)))
            for k, v in vector.items()
        }
        for vector in directions
    ]


def get_basis(state_dict, n=2):
    initial_basis = [
        {
            k: torch.randn_like(w, dtype=torch.float32) for k, w in state_dict.items()
        }
        for _ in range(n)
    ]

    orthogonal_basis = gram_shmidt(initial_basis)
    orthonormalized_basis = normalize_directions(state_dict, orthogonal_basis, filtering=is_weight_filtered)

    return orthonormalized_basis


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

    result_path = Path(args.infer_result_path or args.exp_path) / (args.infer_name or "infer_loss")
    result_path.mkdir(exist_ok=True)
    result_path_json = result_path / "result.jsonl"
    result_path_pt = result_path / "result_directions.pt"

    chkp_path = args.chkp_path or (Path(args.exp_path) / "ckpts" / "latest" / "main.pt")

    ckpt = torch.load(chkp_path, map_location=device)
    model.load_state_dict(
        {
            k.replace('_orig_mod.', '').replace('module.', ''): v for k, v in ckpt["model"].items()
        }
    )

    if args.compile:
        model = torch.compile(model)

    if args.infer_loss_directions_file is None:
        directions = get_basis(model.state_dict(), n=args.infer_loss_dims_num)
    else:
        directions = torch.load(args.infer_loss_directions_file)
        assert len(directions) == args.infer_loss_dims_num
    torch.save(directions, result_path_pt)

    axes = np.linspace(-args.infer_loss_magnitude, args.infer_loss_magnitude, int(args.infer_loss_points ** (1 / args.infer_loss_dims_num)))
    axes = sorted(axes.tolist() + [0.0])
    mesh = np.meshgrid(*([axes] * args.infer_loss_dims_num))
    points = np.stack(mesh, axis=0).reshape(2, -1).T

    datareader = get_data_reader(args)
    batches = [
        datareader.sample_batch() for _ in range(args.infer_batches)
    ]

    start_state = {
        k: w.clone() for k, w in model.state_dict().items()
    }

    with torch.cuda.amp.autocast(
        enabled=args.dtype != 'float32',
        dtype=torch.bfloat16 if args.dtype == 'bfloat16' else torch.float16
    ), open(result_path_json, 'w') as file:
        for i, point in enumerate(points):
            point = point.tolist()
            new_state = direction_add(
                start_state,
                get_delta(directions, point)
            )
            model.load_state_dict(new_state)
            losses = []
            for x, y in batches:
                if "cuda" in torch.device(device).type:
                    # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
                    x = x.pin_memory().to(device, non_blocking=True)
                    y = y.pin_memory().to(device, non_blocking=True)
                else:
                    x = x.to(device)
                    y = y.to(device)

                out = model(x, targets=y)
                losses.append(out['loss'].item())
            loss = np.mean(losses)
            print(f"Processed point {i + 1}/{len(points)}: {point} -> {loss}")
            json.dump(
                dict(
                    point=point,
                    loss=loss,
                ),
                file
            )
            file.write('\n')


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
        "--config_format", default="infer_loss", choices=config.registered_formats()
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
