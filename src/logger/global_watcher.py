import functools
import numpy as np
import os
import torch

from collections import defaultdict
from copy import deepcopy

_prev_states = list()
_state = dict()
_enabled = False


@torch.compiler.disable()
def log_stat(name, value):
    if not _enabled:
        return

    if int(os.environ.get("RANK", 0)) != 0:
        return

    if not isinstance(name, tuple):
        name = (name, )
    already_logged = sum(name == k[:-1] for k in _state)
    name = (*name, already_logged)
    if isinstance(value, torch.Tensor):
        value = value.item()
    _state[name] = value


def mark_step_end():
    _prev_states.append(deepcopy(_state))
    _state.clear()


def reset_logging():
    global _enabled
    _state.clear()
    _prev_states.clear()


def disable_logging():
    global _enabled
    _enabled = False


def enable_logging():
    global _enabled
    _enabled = True


def dump_and_reset():# -> dict[Any, Any]:
    if len(_state) > 0:
        mark_step_end()

    if len(_prev_states) == 0:
        return dict()

    keys = set(list(_prev_states[0].keys()))
    for state in _prev_states:
        if set(list(state.keys())) != keys:
            print(f"Logged different stats on different passes:\n{keys = }\n{set(list(state.keys())) = }")

    states_accum = defaultdict(list)

    for state in _prev_states:
        for key, value in state.items():
            states_accum[key].append(value)

    states_agg = dict()
    for key in states_accum:
        states_agg[key] = np.mean(states_accum[key])

    @functools.lru_cache()
    def get_logged_counts(key):
        return sum(key[:-1] == k[:-1] for k in states_agg)

    output = dict()
    for key, value in states_agg.items():
        already_logged = get_logged_counts(key)
        name = '/'.join(key[:-1])
        if already_logged > 1:
            name = f"{name}-{key[-1]}"
        output[name] = value

    reset_logging()
    return output
