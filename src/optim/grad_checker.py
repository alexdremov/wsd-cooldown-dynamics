import torch


def fix_state_dict(d):
    return {
        k.replace('_orig_mod.module.', '').replace('_orig_mod.', ''): v for k, v in d.items()
    }


@torch.no_grad()
def direction_dot(a, b):
    a, b = fix_state_dict(a), fix_state_dict(b)
    result = 0
    for k in a:
        result += (a[k] * b[k].to(a[k])).sum().item()
    return result


@torch.no_grad()
def direction_cos(a, b):
    a, b = fix_state_dict(a), fix_state_dict(b)
    return direction_dot(a, b) / ((direction_dot(a, a) ** 0.5) * (direction_dot(b, b) ** 0.5))


@torch.no_grad()
def direction_mul(c, b):
    b = fix_state_dict(b)
    return {
        k: c * v for k, v in b.items()
    }


@torch.no_grad()
def direction_add(a, b):
    a, b = fix_state_dict(a), fix_state_dict(b)
    return {
        k: a[k] + v.to(a[k]) for k, v in b.items()
    }


@torch.no_grad()
def direction_sub(a, b):
    a, b = fix_state_dict(a), fix_state_dict(b)
    return {
        k: a[k] - v.to(a[k]) for k, v in b.items()
    }
