#!/usr/bin/env python3
"""A numpy-shaped facade that dispatches to torch when handed tensors.

WHY THIS EXISTS, AND WHY IT IS SHAPED LIKE NUMPY
------------------------------------------------
132 of 240 registered impls RAISED the first time they were handed a CUDA
tensor (`conformance/torch_sweep.py`). Every one had the same cause: the
module does `import numpy as np` and then calls `np.linalg.norm(x)`,
`np.stack([...])`, `np.clip(...)` on its arguments. The conformance harness
drives numpy, so the A/B never sees it; the renderer drives torch, so the
first real call dies.

The cheap fix would be to rewrite every expression against a shim. That is
also the DANGEROUS fix: those expressions are transcriptions of decompiled
shader code, they are individually verified to ~1e-16 against the reference,
and touching 132 of them to fix a plumbing problem would put every one of
those numbers back in play.

So the facade is shaped like numpy on purpose. A module is fixed by changing
ONE LINE --

    import numpy as np            ->    from .._arraylib import np

-- and not one character of any expression. The call sites keep saying
`np.linalg.norm(n, axis=-1, keepdims=True)`; `np` is now an object that
looks at what it was handed and routes accordingly.

WHAT THIS IS NOT
----------------
Not a numpy reimplementation. It covers the operations the registered impls
actually use, and anything it does not cover falls through to real numpy --
which, on a tensor, will fail exactly as loudly as it does today. That is
deliberate: a facade that silently swallowed an uncovered call would convert
a hard raise into a wrong pixel, which is a strictly worse failure than the
one being fixed.

THE CPU TRAP, WHICH IS WHY THIS WAS INVISIBLE
---------------------------------------------
On a CPU tensor, numpy does NOT raise -- it converts via `__array__` and
returns an ndarray. So the same 132 impls report only 20 raises on CPU and
75 silent "returned a numpy array". A sweep run on CPU would have declared
145/240 healthy and hidden 112 defects. The sweep therefore defaults to
CUDA, and this facade keeps the tensor world closed on both.
"""
from __future__ import annotations

import numpy as _np


def _is_t(x):
    return type(x).__module__.split(".")[0] == "torch"


def _any_t(args):
    for a in args:
        if _is_t(a):
            return True
        if isinstance(a, (list, tuple)) and _any_t(a):
            return True
    return False


def _torch():
    import torch
    return torch


# --------------------------------------------------------------------------
# per-function adapters: numpy's spelling on the left, torch's on the right
# --------------------------------------------------------------------------
def _axis_to_dim(kw):
    kw = dict(kw)
    if "axis" in kw:
        kw["dim"] = kw.pop("axis")
    if "keepdims" in kw:
        kw["keepdim"] = kw.pop("keepdims")
    return kw


def _wrap(name, torch_name=None, axis_kw=True):
    tn = torch_name or name

    def f(*a, **kw):
        if not _any_t(a):
            return getattr(_np, name)(*a, **kw)
        t = _torch()
        return getattr(t, tn)(*a, **(_axis_to_dim(kw) if axis_kw else kw))
    f.__name__ = name
    return f



def _cast(name):
    """numpy dtype-constructor that also casts tensors.

    IT MUST STILL BE A DTYPE. A plain function would work for
    `np.int64(x)` and BREAK every `arr.astype(np.int64)` in the repo,
    because astype does not call its argument -- it parses it as a dtype.
    I shipped that version for about five minutes and it turned seven
    green terms into IMPL_ERROR; the runner caught it immediately.

    So the returned object SUBCLASSES the numpy scalar type: `np.dtype()`
    still resolves it through the base, and `__new__` intercepts the
    tensor case that `Tensor.__array__` would otherwise send to
    "can't convert cuda:0 device type tensor to numpy".
    """
    base = getattr(_np, name)

    class _C(base):
        def __new__(cls, x=0, *a, **kw):
            if _is_t(x):
                return x.to(dtype=getattr(_torch(), name))
            return base(x, *a, **kw)

    _C.__name__ = _C.__qualname__ = name
    return _C


def _promote(arrs):
    # A list mixing tensors with numpy constants is the common shape here
    # (`np.stack([t, np.zeros_like(x)])`); torch.stack refuses it outright.
    t = _torch()
    ref = next(x for x in arrs if _is_t(x))
    return [x if _is_t(x) else t.as_tensor(x, dtype=ref.dtype,
                                           device=ref.device) for x in arrs]


def _stack(arrs, axis=0, **kw):
    if not _any_t([arrs]):
        return _np.stack(arrs, axis=axis, **kw)
    return _torch().stack(_promote(list(arrs)), dim=axis)


def _concatenate(arrs, axis=0, **kw):
    if not _any_t([arrs]):
        return _np.concatenate(arrs, axis=axis, **kw)
    return _torch().cat(_promote(list(arrs)), dim=axis)


def _clip(x, lo, hi, **kw):
    if not _any_t((x, lo, hi)):
        return _np.clip(x, lo, hi, **kw)
    t = _torch()
    # clamp refuses a (min=int, max=Tensor) mix, so if either bound is a
    # tensor both are promoted.
    if _is_t(lo) or _is_t(hi):
        ref = x if _is_t(x) else (lo if _is_t(lo) else hi)
        lo = lo if _is_t(lo) else t.as_tensor(lo, dtype=ref.dtype,
                                              device=ref.device)
        hi = hi if _is_t(hi) else t.as_tensor(hi, dtype=ref.dtype,
                                              device=ref.device)
    return t.clamp(x, min=lo, max=hi)


def like(const, ref):
    """A read constant, on the same backend/device/dtype as the data.

    `tensor * np.ndarray` routes through Tensor.__array__ and dies on CUDA,
    so a constant written inline in an expression is torch-hostile even
    when every np.* call around it dispatches correctly. Wrapping the
    constant is the fix; moving it to module scope is not.
    """
    if not _is_t(ref):
        return const
    t = _torch()
    dt = ref.dtype if ref.dtype.is_floating_point else t.get_default_dtype()
    return t.as_tensor(const, dtype=dt, device=ref.device)


def _unique(x, **kw):
    if not _is_t(x):
        return _np.unique(x, **kw)
    return _torch().unique(x)


def _cross(a, b, axis=-1, **kw):
    if not _any_t((a, b)):
        return _np.cross(a, b, axis=axis, **kw)
    a, b = _promote([a, b])
    return _torch().cross(a, b, dim=axis)


def _where(c, a, b):
    if not _any_t((c, a, b)):
        return _np.where(c, a, b)
    t = _torch()
    ref = next((x for x in (c, a, b) if _is_t(x)))
    if _is_t(c) and c.dtype != t.bool:
        c = c != 0
    # THE REFERENCE MAY BE THE CONDITION. When only `c` is a tensor it is
    # BOOL, so `ref.dtype.is_floating_point` is False and the branches were
    # converted with dtype=None -- numpy defaults to float64, and the
    # float64 result then met a float32 tensor downstream ("Found dtype
    # Float but expected Double" out of torch.cross). Prefer a floating
    # tensor among the operands; fall back to the default dtype, never to
    # numpy's.
    fl = next((x for x in (a, b, c)
               if _is_t(x) and x.dtype.is_floating_point), None)
    dt = fl.dtype if fl is not None else t.get_default_dtype()
    a = a if _is_t(a) else t.as_tensor(a, dtype=dt, device=ref.device)
    b = b if _is_t(b) else t.as_tensor(b, dtype=a.dtype, device=ref.device)
    return t.where(c, a, b)


def _t_or_reduce(x, axis=None):
    t = _torch()
    if axis is None:
        f = x.reshape(-1)
        acc = f[0]
        for i in range(1, f.shape[0]):
            acc = t.bitwise_or(acc, f[i])
        return acc
    # Log-depth OR along `axis`: halve and fold, so a 32-wide wave is 5
    # kernels rather than 31.
    y = x.movedim(axis, -1)
    while y.shape[-1] > 1:
        n = y.shape[-1]
        h = n // 2
        y = t.bitwise_or(y[..., :h], y[..., h:2 * h]) if n % 2 == 0 else \
            t.cat([t.bitwise_or(y[..., :h], y[..., h:2 * h]), y[..., -1:]], -1)
    return y[..., 0]


class _Ufunc:
    """A dispatching binary op that is still a numpy UFUNC where it matters.

    `.reduce` on a CUDA tensor would go through __array__ and raise, so
    bitwise_or.reduce dispatches to a fold here.

    numpy ufuncs carry methods -- `.reduce`, `.accumulate`, `.outer` -- and
    real call sites use them: `binner.py:51` is `np.bitwise_or.reduce(...)`.
    A plain function wrapper has no `.reduce`, so replacing the ufunc with
    one broke a term that had been green, which the numpy-side control
    caught in the same pass that measured the fix. Unknown attributes
    therefore delegate to the genuine ufunc.
    """

    def __init__(self, np_name, torch_name=None):
        self._np_name = np_name
        self._name = torch_name or np_name
        self._np_fn = getattr(_np, np_name)

    def __call__(self, a, b, **kw):
        if not _any_t((a, b)):
            return self._np_fn(a, b, **kw)
        t = _torch()
        ref = a if _is_t(a) else b
        a = a if _is_t(a) else t.as_tensor(a, dtype=ref.dtype,
                                           device=ref.device)
        b = b if _is_t(b) else t.as_tensor(b, dtype=ref.dtype,
                                           device=ref.device)
        return getattr(t, self._name)(a, b)

    def __getattr__(self, k):
        if k == "reduce":
            def _red(a, axis=None, **kw):
                if not _is_t(a):
                    return self._np_fn.reduce(a, axis=axis, **kw)
                if self._np_name in ("bitwise_or", "bitwise_and"):
                    if self._np_name == "bitwise_or":
                        return _t_or_reduce(a, axis)
                raise TypeError(
                    "%s.reduce has no tensor path" % self._np_name)
            return _red
        return getattr(self._np_fn, k)


def _binary(np_name, torch_name=None):
    return _Ufunc(np_name, torch_name)


def _asarray(x, *a, **kw):
    # A tensor is ALREADY an array in the sense every caller means. Passing
    # it to np.asarray is what raises on CUDA; returning it unchanged is
    # what the call site intended.
    if _is_t(x):
        return x
    return _np.asarray(x, *a, **kw)


def _co(x, ref):
    """Bring a non-tensor operand onto the tensor path.

    torch.einsum and friends reject MIXED operands outright -- "expected
    Tensor as element 1" -- so dispatching on "any operand is a tensor"
    is not enough; the others have to come with it. This is the shape a
    module-level numpy constant takes when its data argument is a tensor,
    which is most terms that fold a read constant into a dot product.
    """
    if _is_t(x) or x is None:
        return x
    t = _torch()
    return t.as_tensor(x, dtype=ref.dtype, device=ref.device)


def _einsum(sub, *ops, **kw):
    if not _any_t(ops):
        return _np.einsum(sub, *ops, **kw)
    ref = next(o for o in ops if _is_t(o))
    return _torch().einsum(sub, *[_co(o, ref) for o in ops])


def _take(a, indices, axis=None, **kw):
    if not _any_t((a, indices)):
        return _np.take(a, indices, axis=axis, **kw)
    t = _torch()
    idx = indices if _is_t(indices) else t.as_tensor(indices, device=a.device)
    return t.take(a, idx) if axis is None else t.index_select(a, axis,
                                                             idx.long())


def _repeat(a, repeats, axis=None, **kw):
    if not _is_t(a):
        return _np.repeat(a, repeats, axis=axis, **kw)
    t = _torch()
    return t.repeat_interleave(a, repeats, dim=axis)


def _broadcast_to(x, shape, **kw):
    if not _is_t(x):
        return _np.broadcast_to(x, shape, **kw)
    return _torch().broadcast_to(x, shape)


_NP2T = {}


def _astype(x, dtype, **kw):
    # `.astype` is a numpy METHOD, so the facade cannot intercept it where it
    # is written; the call sites are rewritten to `np.astype(x, dt)`. Tensors
    # spell it `.to(dtype=...)`.
    if not _is_t(x):
        return _np.asarray(x).astype(dtype, **kw)
    t = _torch()
    name = getattr(dtype, "__name__", None) or str(dtype)
    name = {"bool_": "bool", "float": "float32", "int": "int64"}.get(name, name)
    td = getattr(t, name, None)
    return x.to(dtype=td) if td is not None else x.to(dtype=dtype)


class _Linalg:
    @staticmethod
    def norm(x, *a, **kw):
        if not _is_t(x):
            return _np.linalg.norm(x, *a, **kw)
        return _torch().linalg.norm(x, *a, **_axis_to_dim(kw))


class _NpFacade:
    """`np` with a torch escape hatch. Unknown names fall through to real
    numpy untouched -- see the module docstring on why that is deliberate."""

    linalg = _Linalg()

    # elementwise unary, same name in both libraries
    sqrt = staticmethod(_wrap("sqrt", axis_kw=False))
    abs = staticmethod(_wrap("abs", axis_kw=False))
    exp = staticmethod(_wrap("exp", axis_kw=False))
    log = staticmethod(_wrap("log", axis_kw=False))
    log2 = staticmethod(_wrap("log2", axis_kw=False))
    sin = staticmethod(_wrap("sin", axis_kw=False))
    cos = staticmethod(_wrap("cos", axis_kw=False))
    tan = staticmethod(_wrap("tan", axis_kw=False))
    tanh = staticmethod(_wrap("tanh", axis_kw=False))
    sign = staticmethod(_wrap("sign", axis_kw=False))
    floor = staticmethod(_wrap("floor", axis_kw=False))
    ceil = staticmethod(_wrap("ceil", axis_kw=False))
    round = staticmethod(_wrap("round", axis_kw=False))
    isfinite = staticmethod(_wrap("isfinite", axis_kw=False))

    # reductions, axis -> dim
    sum = staticmethod(_wrap("sum"))
    mean = staticmethod(_wrap("mean"))
    prod = staticmethod(_wrap("prod"))
    cumsum = staticmethod(_wrap("cumsum"))

    # shape ops
    stack = staticmethod(_stack)
    concatenate = staticmethod(_concatenate)
    # DTYPE CASTS. `np.int64(x)` and `np.float64(x)` are numpy SCALAR
    # TYPES, not ufuncs, so `_wrap` never covered them: calling one on a
    # tensor falls through to `Tensor.__array__` and raises "can't convert
    # cuda:0 device type tensor to numpy". On a CPU tensor it SUCCEEDS,
    # which is the worst shape for a gap -- a CPU probe certifies the
    # module and the GPU run dies. Found by wiring ao.sao_hash_jitter,
    # whose integer xor hash needs the cast, into the live SAO path.
    int64 = _cast("int64")
    int32 = _cast("int32")
    float64 = _cast("float64")
    float32 = _cast("float32")
    like = staticmethod(like)
    unique = staticmethod(_unique)
    zeros_like = staticmethod(_wrap("zeros_like", axis_kw=False))
    ones_like = staticmethod(_wrap("ones_like", axis_kw=False))
    full_like = staticmethod(_wrap("full_like", axis_kw=False))

    # binary with scalar promotion
    maximum = _binary("maximum")
    minimum = _binary("minimum")
    power = _binary("power", "pow")
    arctan2 = _binary("arctan2", "atan2")

    # reductions numpy spells max/min and torch spells amax/amin when an
    # axis is given -- torch.max(x, dim) returns a (values, indices) tuple,
    # which is NOT what `np.max(x, axis=)` means, so the adapter must pick
    # amax or the call site silently gets a tuple.
    max = staticmethod(_wrap("max", "amax"))
    min = staticmethod(_wrap("min", "amin"))
    all = staticmethod(_wrap("all"))
    any = staticmethod(_wrap("any"))
    trunc = staticmethod(_wrap("trunc", axis_kw=False))
    bitwise_or = _binary("bitwise_or")
    asarray = staticmethod(_asarray)
    einsum = staticmethod(_einsum)
    take = staticmethod(_take)
    repeat = staticmethod(_repeat)
    broadcast_to = staticmethod(_broadcast_to)

    astype = staticmethod(_astype)
    clip = staticmethod(_clip)
    cross = staticmethod(_cross)
    where = staticmethod(_where)

    def __getattr__(self, k):
        # everything else -- dtypes, pi, array, asarray, einsum, ... -- is
        # real numpy. On a tensor those still fail, loudly, which is the
        # intended behaviour: an uncovered call must not be silently wrong.
        return getattr(_np, k)


np = _NpFacade()
