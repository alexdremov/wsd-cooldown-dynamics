import torch
import torch.nn as nn
import torch.nn.functional as F

from data.utils import DataReader
from models.llama import Llama
from optim.utils import get_batch
from tqdm.auto import trange
from collections import defaultdict
import numpy as np


class LinearProber(nn.Module):
    def __init__(self, layers_num, hid_dim, dict_size):
        super().__init__()
        self.layers_num = layers_num
        self.dict_size = dict_size

        self.projs = nn.ModuleList([
            nn.Linear(hid_dim, dict_size)
            for _ in range(layers_num)
        ])

    def forward(self, states):
        assert len(states) == len(self.projs)
        return [
            proj(state)
            for proj, state in zip(self.projs, states)
        ]


def train_score_linear_probe(model: Llama, dataset: DataReader, device):
    training_before = model.training
    dataset_step = dataset.step

    model.eval()

    prober = LinearProber(
        layers_num=model.config.n_layer + 1,
        hid_dim=model.config.n_embd,
        dict_size=model.config.vocab_size
    ).train().to(device)

    opt = torch.optim.AdamW(prober.parameters())

    train_batches = dataset.num_batches() // 2
    eval_batches = dataset.num_batches() - train_batches
    dataset.set_step(0)

    type_ctx = torch.amp.autocast(
        device_type="cuda",
        dtype=torch.bfloat16,
    )

    final_train_loss = 0
    for idx in trange(train_batches):
        x, targets = get_batch(dataset, device=device)
        with torch.no_grad(), type_ctx:
            states = model(x, get_all_states=True)['states']

        probed = prober(states)

        loss = 0
        for logits in probed:
            loss += F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1
            )
        opt.zero_grad()
        loss.backward()
        opt.step()

    final_train_loss = loss.item()
    prober.eval()

    final_eval_loss = defaultdict(list)
    with torch.inference_mode():
        for _ in trange(eval_batches):
            x, targets = get_batch(dataset, device=device)
            with type_ctx:
                states = model(x, get_all_states=True)['states']

            probed = prober(states)
            for i, logits in enumerate(probed):
                loss = F.cross_entropy(
                    logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1
                )
                final_eval_loss[i].append(loss.item())

    dataset.set_step(dataset_step)
    model.train(training_before)
    return final_train_loss, {
        k: np.mean(v) for k, v in final_eval_loss.items()
    }
