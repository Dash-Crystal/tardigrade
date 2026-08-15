#!/usr/bin/env python3
"""One implementation, two array libraries — the character-family shim.

The conformance registry calls our implementation with numpy arrays; the
renderer calls it with torch tensors. Writing the expression twice would
make the A/B compare one copy against a reference while the renderer runs
the other, which is the "measures nothing" failure the registry's
SELF_PAIRED refusal exists to prevent, one level up.

So every expression in `character.py`, `eyeball.py` and `customglove.py` is
written ONCE against these helpers. Only operators common to both libraries
are used inline (`+ - * /`, `**`, `abs()`, `.sum(-1)`, `[..., i]`, and
comparison operators, which both return an array that multiplies as 0/1);
anything that differs in name or in signature is wrapped here and nowhere
else.

This is a SEPARATE FILE from `post/_backend.py` on purpose. That one is
W5's, this one is W4's, and merging two branches that both edited one shim
is how a shared file becomes a conflict nobody reviews. They agree where
they overlap; neither imports the other.
"""
from __future__ import annotations

import numpy as np


def xp(x):
    """The array module that owns `x`."""
    if type(x).__module__.split(".")[0] == "torch":
        import torch
        return torch
    return np


def is_torch(x):
    return type(x).__module__.split(".")[0] == "torch"


def stack(arrs, axis=-1):
    """`np.stack(..., axis=)` / `torch.stack(..., dim=)` — the kwarg differs."""
    if is_torch(arrs[0]):
        import torch
        return torch.stack(list(arrs), dim=axis)
    return np.stack(list(arrs), axis=axis)


def cross(a, b):
    """Cross product over the last axis; `axis=` vs `dim=` again."""
    if is_torch(a):
        import torch
        return torch.cross(a, b, dim=-1)
    return np.cross(a, b, axis=-1)


def where(cond, a, b):
    m = xp(cond)
    if m is np:
        return np.where(cond, a, b)
    import torch
    a = a if torch.is_tensor(a) else torch.as_tensor(a, dtype=cond.dtype
                                                     if cond.dtype.is_floating_point
                                                     else torch.float32,
                                                     device=cond.device)
    b = b if torch.is_tensor(b) else torch.as_tensor(b, dtype=a.dtype,
                                                     device=a.device)
    return torch.where(cond, a, b)


def maximum(a, b):
    if is_torch(a):
        import torch
        if not torch.is_tensor(b):
            b = torch.as_tensor(b, dtype=a.dtype, device=a.device)
        return torch.maximum(a, b)
    return np.maximum(a, b)


def minimum(a, b):
    if is_torch(a):
        import torch
        if not torch.is_tensor(b):
            b = torch.as_tensor(b, dtype=a.dtype, device=a.device)
        return torch.minimum(a, b)
    return np.minimum(a, b)


def clip(x, lo, hi):
    """`saturate`-class clamp. Both libraries spell it `clip` as a free
    function; only torch also spells it `.clamp` as a method, which is the
    spelling that does NOT work on numpy and is therefore not used."""
    return maximum(minimum(x, hi), lo)


def sqrt(x):
    return x ** 0.5


def log(x):
    return xp(x).log(x)


def cos(x):
    return xp(x).cos(x)


def sin(x):
    return xp(x).sin(x)


def sign(x):
    return xp(x).sign(x)


def step(edge, x):
    """GLSL `step(edge, x)` = x >= edge ? 1 : 0, as a float."""
    return (x >= edge) * 1.0


def dot(a, b):
    """Dot over the last axis, keeping the leading shape."""
    return (a * b).sum(-1)


def length(a):
    return sqrt(dot(a, a))


def normalize(a, eps: float = 1e-9):
    """`normalize()` with the renderer's own floor.

    This is `_nrm` from gpu_render.py:5352 verbatim -- `v / clamp(|v|, min=1e-9)`
    -- and the epsilon is 1e-9 because THAT is the epsilon the renderer has
    used everywhere since long before this family. The reference shader has
    no floor at all (a degenerate input is undefined in float), but the
    conformance sample deliberately spans the degenerate corner, and a nan
    on either side is a refusal rather than a measurement. Both sides carry
    the same floor, so the comparison is of the expression and not of the
    guard."""
    return a / maximum(length(a), eps)[..., None]


def const_vec(ref, values):
    """A literal vector in `ref`'s library, dtype and device."""
    if is_torch(ref):
        import torch
        return torch.as_tensor(list(values), dtype=ref.dtype,
                               device=ref.device)
    return np.asarray(list(values), dtype=getattr(ref, "dtype", np.float64))


def ones_like(x):
    return xp(x).ones_like(x)


def zeros_like(x):
    return xp(x).zeros_like(x)
