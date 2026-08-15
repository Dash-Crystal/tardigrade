"""Sequence-parallel camera controllers: the demo_camera recurrences as
associative scans over (B players, T ticks) -- no python step loop.

Rendering a RECORDED replay has no recurrence at all: every tick's state is
data. The only sequential structure in the export pipeline is the camera
controllers (CVKalman1D / CircEMA / Spring1D), and each is a LINEAR
recurrence y_t = a_t * y_{t-1} + b_t (scalar or 2x2), so the whole
trajectory evaluates as an associative prefix scan in O(log T) depth:

    (a2, b2) o (a1, b1) = (a2*a1, a2*b1 + b2)

- CircEMA: unwrap angles (cumsum of wrapped diffs -- itself a scan), then a
  constant-coefficient EMA scan in the unwrapped domain.
- Spring1D: semi-implicit integration is a linear TIME-VARYING 2x2
  recurrence [v; a]_t = M_t [v; a]_{t-1} + c_t. M_t depends only on
  omega_t, which depends only on the PRECOMPUTED foreshadow blend
  schedule (the fire ticks are data) -- so M_t is a tensor, not a loop.
- CVKalman1D: the covariance/gain recursion (Riccati) is data-INDEPENDENT
  -- P_t, K_t precompute once per (q, r, dt) -- leaving the state update
  x_t = (A - K_t H A) x_{t-1} + K_t z_t, a linear time-varying 2x2 scan.

Everything runs batched over B trajectories at once: 30 controllers x
2,796 steps of python (~84k interpreted iterations per view set) become a
handful of tensor ops. Equivalence vs the reference classes is exact to
float tolerance (verified in tools/test_scan_camera.py at real sizes).
"""
from __future__ import annotations

import torch


def linear_scan(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Inclusive prefix scan of y_t = a_t * y_{t-1} + b_t, y_0 = b_0.

    a, b: (..., T) scalar coefficient sequences. Returns y: (..., T).
    Blelloch-style doubling: O(log T) tensor steps, each a full-width
    elementwise op -- sequence-parallel by construction.
    """
    a = a.clone()
    b = b.clone()
    T = a.shape[-1]
    step = 1
    while step < T:
        a_prev = torch.ones_like(a)
        b_prev = torch.zeros_like(b)
        a_prev[..., step:] = a[..., :-step]
        b_prev[..., step:] = b[..., :-step]
        b = torch.where(
            torch.arange(T, device=a.device) >= step,
            a * b_prev + b, b)
        a = torch.where(
            torch.arange(T, device=a.device) >= step,
            a * a_prev, a)
        step *= 2
    return b


def linear_scan_2x2(M: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    """Inclusive scan of x_t = M_t @ x_{t-1} + c_t with x_{-1} = 0.

    M: (..., T, 2, 2), c: (..., T, 2). Returns x: (..., T, 2).
    Same doubling composition with matrix product for `a`.
    """
    M = M.clone()
    c = c.clone()
    T = M.shape[-3]
    idx = torch.arange(T, device=M.device)
    step = 1
    while step < T:
        Mp = torch.zeros_like(M)
        cp = torch.zeros_like(c)
        Mp[..., step:, :, :] = M[..., :-step, :, :]
        Mp[..., :step, :, :] = torch.eye(2, device=M.device, dtype=M.dtype)
        cp[..., step:, :] = c[..., :-step, :]
        mask = (idx >= step)
        c_new = torch.einsum("...tij,...tj->...ti", M, cp) + c
        M_new = torch.einsum("...tij,...tjk->...tik", M, Mp)
        c = torch.where(mask[..., None], c_new, c)
        M = torch.where(mask[..., None, None], M_new, M)
        step *= 2
    return c


# ---------------------------------------------------------------------------
# The three controllers, sequence-parallel, batched over B trajectories.
# ---------------------------------------------------------------------------

def unwrap_deg(x: torch.Tensor) -> torch.Tensor:
    """Angle unwrap over the last (time) axis, degrees; vectorized."""
    d = x.diff(dim=-1)
    d = (d + 180.0) % 360.0 - 180.0
    return torch.cat([x[..., :1], x[..., :1] + d.cumsum(-1)], dim=-1)


def circ_ema_scan(x_deg: torch.Tensor, alpha: float) -> torch.Tensor:
    """CircEMA over (B, T) angle tracks: unwrap -> EMA scan -> rewrap.

    Reference semantics: y_0 = x_0; y_t = y_{t-1} + alpha*wrap180(x_t - y_{t-1})
    == (1-alpha)*y_{t-1} + alpha*x_t in the unwrapped domain (exact while
    successive EMA-error stays within +-180 deg, which the wrap180 reference
    assumes identically).
    """
    u = unwrap_deg(x_deg)
    T = u.shape[-1]
    a = torch.full_like(u, 1.0 - alpha)
    b = alpha * u
    a[..., 0] = 0.0
    b[..., 0] = u[..., 0]
    y = linear_scan(a, b)
    return (y + 180.0) % 360.0 - 180.0


def spring_scan(target_deg: torch.Tensor, omega: torch.Tensor,
                dt: float) -> torch.Tensor:
    """Spring1D over (B, T): semi-implicit critically-damped spring toward a
    moving target with TIME-VARYING omega (the foreshadow-blend schedule).

    Reference step (Spring1D.step):
        err = target_t - a_{t-1}   (unwrapped domain)
        v_t = v_{t-1} + (w^2 err - 2 w v_{t-1}) dt
        a_t = a_{t-1} + v_t dt
    Linear time-varying in s = [v; a]:
        M_t = [[1-2w dt,  -w^2 dt], [dt(1-2w dt), 1 - w^2 dt^2]]
        c_t = [w^2 dt, w^2 dt^2] * target_t
    a_0 = target_0, v_0 = 0 (reference seeds on first sample).
    """
    tgt = unwrap_deg(target_deg)
    w = omega
    T = tgt.shape[-1]
    M = torch.zeros(*tgt.shape, 2, 2, dtype=tgt.dtype, device=tgt.device)
    M[..., 0, 0] = 1.0 - 2.0 * w * dt
    M[..., 0, 1] = -(w * w) * dt
    M[..., 1, 0] = dt * (1.0 - 2.0 * w * dt)
    M[..., 1, 1] = 1.0 - (w * w) * dt * dt
    c = torch.stack(((w * w) * dt * tgt,
                     (w * w) * dt * dt * tgt), dim=-1)
    # seed: t=0 output is exactly target_0 with v=0
    M[..., 0, :, :] = 0.0
    c[..., 0, 0] = 0.0
    c[..., 0, 1] = tgt[..., 0]
    s = linear_scan_2x2(M, c)
    a = s[..., 1]
    return (a + 180.0) % 360.0 - 180.0


def kalman_cv_scan(z: torch.Tensor, q: float, r: float,
                   dt: float) -> tuple:
    """CVKalman1D over (B, T) position tracks -> (pos, vel), sequence-parallel.

    The gain schedule K_t is data-independent (Riccati on P with constant
    q, r, dt): precompute K_t once serially over T CHEAP host scalars
    (T 2x2 ops, no data), then the state recursion
        x_t = (A - K_t H A) x_{t-1} + K_t z_t
    is a linear time-varying 2x2 scan over the DATA -- the only part that
    scales with batch, and it is fully parallel.
    """
    T = z.shape[-1]
    dev, dtype = z.device, z.dtype
    # Gain schedule: EXACTLY the reference's scalar recursion (CVKalman1D.
    # step), including the started-skip (t=0 emits [z,0] with NO P update)
    # and its continuous white-accel Q = q*[dt^3/3, dt^2/2; dt^2/2, dt].
    p00, p01, p10, p11 = 1e3, 0.0, 0.0, 1e3
    K0, K1 = [0.0], [0.0]                         # t=0 unused (seeded)
    for _ in range(1, T):                         # data-INDEPENDENT host math
        n00 = p00 + dt * (p10 + p01) + dt * dt * p11 + q * dt ** 3 / 3.0
        n01 = p01 + dt * p11 + q * dt ** 2 / 2.0
        n10 = p10 + dt * p11 + q * dt ** 2 / 2.0
        n11 = p11 + q * dt
        s = n00 + r
        k0, k1 = n00 / s, n10 / s
        K0.append(k0)
        K1.append(k1)
        p00, p01 = (1 - k0) * n00, (1 - k0) * n01
        p10, p11 = n10 - k1 * n00, n11 - k1 * n01
    k0t = torch.tensor(K0, dtype=dtype, device=dev)
    k1t = torch.tensor(K1, dtype=dtype, device=dev)
    # state recursion x_t = M_t x_{t-1} + K_t z_t with
    # M_t = (I - K_t H) A,  A = [[1, dt], [0, 1]],  H = [1, 0]:
    M = torch.zeros(T, 2, 2, dtype=dtype, device=dev)
    M[:, 0, 0] = 1.0 - k0t
    M[:, 0, 1] = dt * (1.0 - k0t)
    M[:, 1, 0] = -k1t
    M[:, 1, 1] = 1.0 - dt * k1t
    M = M.expand(*z.shape[:-1], T, 2, 2).clone()
    c = torch.stack((k0t * z, k1t * z), dim=-1)
    # seed: x_0 = [z_0, 0] (reference's started branch)
    M[..., 0, :, :] = 0.0
    c = c.clone()
    c[..., 0, 0] = z[..., 0]
    c[..., 0, 1] = 0.0
    s = linear_scan_2x2(M, c)
    return s[..., 0], s[..., 1]
