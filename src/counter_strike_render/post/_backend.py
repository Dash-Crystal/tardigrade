#!/usr/bin/env python3
"""One implementation, two array libraries.

The conformance registry calls our implementation with numpy arrays; the
renderer calls it with torch tensors. Writing the expression twice would
make the A/B compare one copy against a reference while the renderer runs
the other — the exact "measures nothing" failure the registry's
SELF_PAIRED refusal exists to prevent, one level up.

So every expression in this package is written ONCE against a resolved
array module. Only the operator names common to both libraries are used;
anything that differs in name or semantics is wrapped here and nowhere
else.
"""
from __future__ import annotations

import numpy as np


def xp(x):
    """The array module that owns `x`."""
    mod = type(x).__module__.split(".")[0]
    if mod == "torch":
        import torch
        return torch
    return np


def clip(x, lo, hi):
    m = xp(x)
    if m is np:
        return np.clip(x, lo, hi)
    # torch refuses a positional (Tensor, int, Tensor) mix; the keyword form
    # takes tensor or scalar bounds interchangeably.
    return m.clamp(x, min=lo, max=hi)


def where(cond, a, b):
    return xp(cond).where(cond, a, b)


def maximum(a, b):
    """max of two operands, dispatching on EITHER of them.

    It used to dispatch on `a` alone, so maximum(<python float or ndarray>,
    <cuda tensor>) took the numpy branch and np.maximum tried to pull the
    tensor across the device boundary:
        TypeError: can't convert cuda:0 device type tensor to numpy
    That killed every FSR-tier render (preset0/preset1) inside rcas(), and it
    only surfaced now because RCAS previously did not run at all -- enabling a
    pass exposed a helper that had never been handed a device tensor in the
    first position's shadow. A dispatcher that inspects one operand is not a
    dispatcher; it is a coin flip on argument order.
    """
    m = xp(a)
    if m is np:
        m = xp(b)
    if m is np:
        return np.maximum(a, b)
    import torch
    t = a if torch.is_tensor(a) else b
    if not torch.is_tensor(a):
        a = torch.as_tensor(a, dtype=t.dtype, device=t.device)
    if not torch.is_tensor(b):
        b = torch.as_tensor(b, dtype=t.dtype, device=t.device)
    return torch.maximum(a, b)


def minimum(a, b):
    m = xp(a)
    if m is np:
        return np.minimum(a, b)
    import torch
    if not torch.is_tensor(b):
        b = torch.as_tensor(b, dtype=a.dtype, device=a.device)
    return torch.minimum(a, b)


def full_like(x, v):
    m = xp(x)
    return m.full_like(x, v)


def is_torch(x):
    return type(x).__module__.split(".")[0] == "torch"


def is_array(x):
    """True for a numpy ndarray or a torch tensor, false for a python
    scalar. Used where an expression has a branch: the scalar path is a
    python `if`, the array path is an elementwise `where`, and both must
    produce the same number on the same input."""
    return isinstance(x, np.ndarray) or is_torch(x)


def scalar_like(ref, v):
    """`v` broadcast into `ref`'s library, dtype and device."""
    if is_torch(ref):
        import torch
        return torch.as_tensor(v, dtype=ref.dtype, device=ref.device)
    return np.asarray(v, dtype=getattr(ref, "dtype", None))
