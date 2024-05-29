import torch


def fix_state_dict(d):
    return {
        k.replace('_orig_mod.module.', '').replace('_orig_mod.', ''): v for k, v in d.items()
    }


@torch.no_grad()
def direction_dot(a, b):
    a, b = fix_state_dict(a), fix_state_dict(b)
    result = None
    for k in a:
        dot = (a[k] * b[k].to(a[k])).sum()
        if result is None:
            result = dot
        else:
            result += dot
    if result is None:
        return 0.0
    return result.item()


@torch.no_grad()
def direction_cos(a, b):
    a, b = fix_state_dict(a), fix_state_dict(b)
    denom = ((direction_dot(a, a) ** 0.5) * (direction_dot(b, b) ** 0.5))
    if abs(denom) < 1e-8:
        return None
    return direction_dot(a, b) / denom


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
