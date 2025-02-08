import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim.lr_scheduler

from torch.nn.parallel import DistributedDataParallel as DDP
from data.utils import DataReader
from models.llama import Llama, RMSNorm
from optim.utils import get_batch
from tqdm.auto import trange
from collections import defaultdict
import numpy as np


class LinearProber:
    def __init__(self, layers_num, hid_dim, dict_size, device, train_steps, init_with):
        super().__init__()
        self.layers_num = layers_num
        self.dict_size = dict_size

        def make_head():
            l = nn.Linear(hid_dim, dict_size, device=device, bias=False)
            l.weight.copy_(init_with)
            return l

        self.projs = [
            nn.Sequential(
                RMSNorm(hid_dim).to(device),
                make_head()
            )
            for _ in range(layers_num)
        ]
        self.projs = [
            DDP(i) for i in self.projs
        ]
        self.opts = [
            torch.optim.AdamW(i.parameters(), lr=1e-2, weight_decay=1e-3) for i in self.projs
        ]
        self.scheds = [
            torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=opt, T_max=train_steps)
            for opt in self.opts
        ]


def train_score_linear_probe(model: Llama, dataset: DataReader, device):
    training_before = model.training
    dataset_step = dataset.step

    model.eval()

    prober = None

    train_batches = min(dataset.num_batches() // 2, 2000)
    eval_batches = min(dataset.num_batches() - train_batches, 1000)
    dataset.set_step(0)

    def init_prober(states, dict_size):
        hid_dim = states[0].size(-1)
        prober = LinearProber(
            layers_num=len(states),
            hid_dim=hid_dim,
            dict_size=dict_size,
            device=device,
            train_steps=train_batches,
            init_with=model.module.lm_head.weight,
        )
        return prober

    type_ctx = torch.amp.autocast(
        device_type="cuda",
        dtype=torch.bfloat16,
    )

    final_train_loss = 0
    p_bar = trange(train_batches, desc="probe train")

    for idx in p_bar:
        with torch.no_grad(), type_ctx:
            x, targets = get_batch(dataset, device=device)
            output = model(x, get_all_states=True, get_logits=True)
            states = output['states']
            dict_size = output['logits'].size(-1)

        if prober is None:
            prober = init_prober(states, dict_size)

        losses = []
        for proj, opt, sched, state in zip(prober.projs, prober.opts, prober.scheds, states):
            with type_ctx:
                logits = proj(state)
                loss = F.cross_entropy(
                    logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1
                )
                losses.append(loss.item())
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
        p_bar.set_description(f"loss = {np.mean(losses):.4f}", refresh=False)

    final_train_loss = loss.item()

    final_eval_loss = defaultdict(list)
    with torch.inference_mode(), torch.no_grad():
        for _ in trange(eval_batches, desc="probe eval"):
            x, targets = get_batch(dataset, device=device)
            with type_ctx:
                states = model(x, get_all_states=True)['states']

            assert prober is not None
            for i, (proj, opt, state) in enumerate(zip(prober.projs, prober.opts, states)):
                logits = proj(state)
                loss = F.cross_entropy(
                    logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1
                )
                final_eval_loss[i].append(loss.item())

    dataset.set_step(dataset_step)
    model.train(training_before)
    return final_train_loss, {
        k: np.mean(v) for k, v in final_eval_loss.items()
    }
