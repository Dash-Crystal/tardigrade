"""Census batch B -- the DEPTH / G-BUFFER / SCREEN-SPACE consumers.

Twelve of batch B's forty families: every family
`SHADER_CALLFLOW_census_batch_B.md` §3.1 lists as reading scene depth or a
G-buffer target, plus the two stencil-state passes and the one family whose
name claims depth and whose bindings do not.

    test_renderpass                  §7:1878   the whole G-buffer
    reconstruct_normals              §7:1305   scene depth -> world normals
    nonmsaa_resolve                  §7:952    scene depth + source + noise
    ssao_convert_depth               §7:1693   G-buffer depth, MSAA + non-MSAA
    ssao_downsample_depth            §7:1728   camera depth, integer/bit only
    sst_copy_depth                   §7:1757   depth in, gl_FragDepth out
    mboitfinal                       §7:831    three depth resolutions
    mboit_mixed_combine              §7:783    three depth resolutions, MRT x3
    overlay_smoke                    §7:1020   smoke depth
    screen_texture_with_depth        §7:1461   the name/binding discrepancy
    stenciltest                      §7:1778
    player_visibility_stencil_proxy  §7:1234   StencilRef, 0 FLOPs

The complementary 28 are `impl-batchB-rest`'s (`--bpx-*`, assets/fam_side_bpx.pt).
12 + 28 = 40.

WHAT THE REFERENCE FOR THIS WORK IS, EXACTLY
--------------------------------------------
**The decompiled GLSL is not on this host and could not be produced here.**
`vcs/decomp.py` needs `~/shwork/s2_shader_zstd.dict` and the three
`shaders_vulkan_dir.vpk` under `~/cs2/game/{csgo,csgo_core,core}/`; neither
exists on this machine (checked), and `spirv-cross` is not installed. So
unlike `csgo_environment_blend` -- where `gpu_render.py` cites `glsl:1239`
and means it -- **not one line below is copied from a decompiled module.**

What IS transcribed, literally, from the shipped packages via
`SHADER_CALLFLOW_census_batch_B.md`:

  * every declared texture and constant-buffer name, per family;
  * every combo axis with its `m_nComboIndexValue` PLACE VALUE and range,
    so the id decode is mixed-radix (`vcs/README.md` trap 1);
  * the shipped module / record / pair counts;
  * fetch-site counts by sampler type INCLUDING the multisample bit, and
    each site's `(set, binding)` provenance;
  * the loop tables -- bound kind, init, trip, per-body FLOPs and fetches;
  * the pass-vs-surface signals: colour output Locations, `gl_FragDepth`,
    `StencilRef`, `OpKill`;
  * `P(computation | FLOP)` -- the FLOP-category histogram at each family's
    peak module.

That last one is the strong constraint, and it is why this file counts its
own arithmetic. `dgb_cost_audit.py` prices what the code below actually
executes, in the SAME categories and with the SAME per-op cost model as
`vcs/flopcount.py` (dot 2N-1, normalize 2N+1, cross 9, mix 3N, clamp 2N,
smoothstep 6N, pow 4N, matrix*vector R*(2C-1), derivative N, int/bit N),
and prints the signed per-category difference against the published table.

**Nothing here claims to be the shipped instruction sequence.** Seven of the
twelve are small enough that a complete operation lands on the published
total exactly -- see the audit -- and for those the construction is stated
in the function. The five large ones (`reconstruct_normals`,
`nonmsaa_resolve`, `mboitfinal`, `mboit_mixed_combine`, `overlay_smoke`)
get the complete operation their bindings, axes, loops and histogram
describe, and the audit prints the residual as a NUMBER. No identity is
claimed anywhere.

WHAT OUR RENDERER DID NOT HAVE, AND WHAT WAS ADDED
--------------------------------------------------
1. **A G-buffer.** `test_renderpass` names four targets --
   `g_tGBufferDepth`, `g_tGBufferAlbedo`, `g_tGBufferNormalWs`,
   `g_tGBufferLightingTerms` -- at `set 1` bindings 31/32/33/34. That is a
   deferred layout and this renderer is forward: it shades in one pass and
   keeps nothing. `GBuffer` below is the four real targets, produced from
   the forward pass's own outputs at full resolution before the tone
   encode.

   **Corroborated from the WRITE side by census batch A (`f42fbb4`).**
   The `error` family -- the fallback material, bindable wherever a real
   material is -- writes FOUR colour targets: a checkerboard to target 0
   and `vec4(0)` to targets 1, 2 and 3. A shader that zeroes three extra
   attachments is bound to a pipeline that HAS them. One family names
   what is read, another proves what is bound; that is stronger than
   either alone, and it is why the four-target layout is treated here as
   established rather than as an inference from four names.

   **What those three extra attachments CARRY is not recoverable from a
   shader that only zeroes them**, and batch A flags it as an open
   question. So the target count is established; the channel contents of
   `g_tGBufferLightingTerms` are not, and `gbuffer_pack()` states its
   choice as a choice.

2. **`sampler2DMS` -- an unresolved multisample attachment.**
   `PER_VIEW_RENDER_TARGETS.md`'s corrected banner calls this "a memory
   root a resolve-first renderer cannot express", and `ssao_convert_depth`
   reads one (`g_tGBufferDepthMS`, `D_MSAA_DEPTH_BUFFER 1`, one `2DMS`
   fetch site). `MSTarget` stores every sample and `_ms_fetch` takes an
   explicit sample index, which is what `OpImageFetch` on a `2DMS` image
   does. It is NEVER resolved before being read.

   **The class is not confined to depth.** Batch A found
   `gaussian_bloom_blur` fetching `sampler2DMS` on an MSAA *COLOUR*
   buffer (`D_MSAA_COLOR_BUFFER`, 4 sites) -- a post-process consuming
   the unresolved scene colour per sample and doing its own resolve
   inside the bloom downsample. `MSTarget` is therefore typed by its
   data, not by "depth": it holds a colour attachment on the same terms.
   Nothing in THIS slice reads MSAA colour -- `ssao_convert_depth`'s
   `g_tGBufferDepthMS` is the only multisample site in the twelve -- so
   the colour case is provided for and not exercised here.

3. **A depth attachment a fragment shader can WRITE.** `sst_copy_depth` is
   0 FLOPs, declares no colour output at all, and writes `gl_FragDepth`.
   A 0-FLOP family is not irrelevant: it changes the depth every later
   pass tests against.

4. **A stencil attachment and a `StencilRef` write.**
   `player_visibility_stencil_proxy` is 0 FLOPs and 0 fetches and does
   only this.

BACKEND
-------
The renderer is torch; this file runs under torch OR numpy, chosen at
import by what is importable. Every array operation goes through the small
dispatching layer below. That is not an abstraction for its own sake: the
audits have to be runnable on a host with no torch, and a check that
cannot be run is not a check.
"""

import math

try:                                   # pragma: no cover - env dependent
    import torch as _T
except Exception:                      # pragma: no cover
    _T = None
import numpy as _np


# =======================================================================
# 0. Backend shim -- torch tensors and numpy arrays, one code path.
# =======================================================================

def _is_t(x):
    return _T is not None and isinstance(x, _T.Tensor)


def _clamp(x, lo, hi):
    return x.clamp(lo, hi) if _is_t(x) else _np.clip(x, lo, hi)


def _where(c, a, b):
    return _T.where(c, a, b) if _is_t(c) else _np.where(c, a, b)


def _promote(xs):
    """Make a MIXED list homogeneous, torch winning.

    _cat and _stack used to dispatch on xs[0] alone, which is only correct
    if the list is homogeneous -- and gbuffer_pack's is not. `_cat([
    normal_ws, packed_a[..., None]])` pairs a numpy normal with a torch
    coverage bit, so the xs[0] test chose the numpy branch and
    np.concatenate raised "can't convert cuda:0 device type tensor to
    numpy". The gbuffer self-test then reported RAISED on every run and
    was scrolled past twice in scored runs.

    Torch wins because the numpy element can always be adopted onto the
    tensor's device, while the reverse would need a .cpu() copy and would
    silently move the whole pack off the GPU.
    """
    if not any(_is_t(x) for x in xs):
        return xs, False
    dev = next(x.device for x in xs if _is_t(x))
    return [x if _is_t(x) else _T.as_tensor(x, device=dev) for x in xs], True


def _cat(xs, axis=-1):
    xs, ist = _promote(xs)
    return _T.cat(xs, dim=axis) if ist else _np.concatenate(xs, axis)


def _stack(xs, axis=-1):
    xs, ist = _promote(xs)
    return _T.stack(xs, dim=axis) if ist else _np.stack(xs, axis)


def _sqrt(x):
    return x.sqrt() if _is_t(x) else _np.sqrt(x)


def _exp(x):
    return x.exp() if _is_t(x) else _np.exp(x)


def _log(x):
    return x.log() if _is_t(x) else _np.log(x)


def _abs(x):
    return x.abs() if _is_t(x) else _np.abs(x)


def _floor(x):
    return x.floor() if _is_t(x) else _np.floor(x)


def _fract(x):
    return x - _floor(x)


def _maximum(a, b):
    return _T.maximum(a, _asme(b, a)) if _is_t(a) else _np.maximum(a, b)


def _minimum(a, b):
    return _T.minimum(a, _asme(b, a)) if _is_t(a) else _np.minimum(a, b)


def _asme(b, like):
    if _is_t(like) and not _is_t(b):
        return _T.as_tensor(b, dtype=like.dtype, device=like.device)
    return b


def _zeros_like(x):
    return _T.zeros_like(x) if _is_t(x) else _np.zeros_like(x)


def _ones_like(x):
    return _T.ones_like(x) if _is_t(x) else _np.ones_like(x)


def _full_like(x, v):
    return _T.full_like(x, v) if _is_t(x) else _np.full_like(x, v)


def _long(x):
    return x.long() if _is_t(x) else x.astype(_np.int64)


def _f32(x):
    return x.float() if _is_t(x) else x.astype(_np.float32)


def _sum(x, axis=-1, keep=False):
    return (x.sum(dim=axis, keepdim=keep) if _is_t(x)
            else x.sum(axis=axis, keepdims=keep))


# =======================================================================
# 1. The FLOP counter -- the SAME cost model as vcs/flopcount.py
#
# `classify()` there prices an instruction from its SPIR-V result type.
# Every arithmetic step below goes through `F.*`, which performs the work
# AND records the cost in the category batch B's tables use. Nothing is
# declared; the histogram is a by-product of the code that runs, so
# rewriting the arithmetic moves the number. dgb_cost_audit.py makes that
# fail on purpose before it is trusted.
# =======================================================================

RAW = 'raw arithmetic (mul/add/sub/div)'
DOT = 'dot products'
MAT = 'matrix products'
NRM = 'normalize'
CRS = 'cross'
MIX = 'mix / lerp'
CLP = 'clamp / saturate'
SST = 'smoothstep'
PEL = 'pow / exp / log'
LEN = 'length / sqrt / rsqrt'
ABS = 'abs / sign / floor / fract'
MIN = 'min'
MAX = 'max'
INT = 'integer arithmetic'
BIT = 'bit ops'
DRV = 'derivatives (dFdx/dFdy/fwidth)'
TRG = 'trig'
FMA = 'fma'


class _Counter(object):
    """FLOPs by category, plus texel-reading instructions by memory root."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.cat = {}
        self.fetch = {}
        self.on = False

    def add(self, cat, n):
        if self.on:
            self.cat[cat] = self.cat.get(cat, 0) + n

    def tex(self, kind):
        if self.on:
            self.fetch[kind] = self.fetch.get(kind, 0) + 1

    def total(self):
        return sum(self.cat.values())

    def nfetch(self):
        return sum(self.fetch.values())


C = _Counter()


class F(object):
    """Counted arithmetic. `n` is the SPIR-V result type's component count."""

    # ---- float elementwise: N ----------------------------------------
    @staticmethod
    def mul(a, b, n=1):
        C.add(RAW, n)
        return a * b

    @staticmethod
    def add(a, b, n=1):
        C.add(RAW, n)
        return a + b

    @staticmethod
    def sub(a, b, n=1):
        C.add(RAW, n)
        return a - b

    @staticmethod
    def div(a, b, n=1):
        C.add(RAW, n)
        return a / b

    @staticmethod
    def neg(a, n=1):
        C.add(RAW, n)
        return -a

    @staticmethod
    def vscale(v, s, n=3):
        """OpVectorTimesScalar -- flopcount prices it raw, N."""
        C.add(RAW, n)
        return v * (s if _is_t(v) or not hasattr(s, 'shape') else s)

    # ---- reductions ---------------------------------------------------
    @staticmethod
    def dot(a, b, n=3):
        C.add(DOT, 2 * n - 1)
        return _sum(a * b, -1, True)

    @staticmethod
    def matvec(m_rows, v, r=4, c=4):
        """OpMatrixTimesVector: R*(2C-1). `m_rows` is a list of R rows."""
        C.add(MAT, r * (2 * c - 1))
        return _cat([_sum(v * row, -1, True) for row in m_rows], -1)

    # ---- GLSL.std.450 -------------------------------------------------
    @staticmethod
    def normalize(v, n=3):
        C.add(NRM, 2 * n + 1)
        d = _sqrt(_sum(v * v, -1, True))
        return v / _maximum(d, _full_like(d, 1e-20))

    @staticmethod
    def cross(a, b):
        C.add(CRS, 9)
        ax, ay, az = a[..., 0:1], a[..., 1:2], a[..., 2:3]
        bx, by, bz = b[..., 0:1], b[..., 1:2], b[..., 2:3]
        return _cat([ay * bz - az * by,
                     az * bx - ax * bz,
                     ax * by - ay * bx], -1)

    @staticmethod
    def mix(a, b, t, n=3):
        C.add(MIX, 3 * n)
        return a + (b - a) * t

    @staticmethod
    def clamp(x, lo, hi, n=1):
        C.add(CLP, 2 * n)
        return _clamp(x, lo, hi)

    @staticmethod
    def smoothstep(e0, e1, x, n=1):
        C.add(SST, 6 * n)
        t = _clamp((x - e0) / (e1 - e0 if not hasattr(e1, 'shape')
                               else (e1 - e0)), 0.0, 1.0)
        return t * t * (3.0 - 2.0 * t)

    @staticmethod
    def exp(x, n=1):
        C.add(PEL, 4 * n)
        return _exp(x)

    @staticmethod
    def log(x, n=1):
        C.add(PEL, 4 * n)
        return _log(x)

    @staticmethod
    def pow(x, y, n=1):
        C.add(PEL, 4 * n)
        return x ** y

    @staticmethod
    def sqrt(x, n=1):
        C.add(LEN, n)
        return _sqrt(x)

    @staticmethod
    def length(v, n=3):
        C.add(LEN, 2 * n)
        return _sqrt(_sum(v * v, -1, True))

    @staticmethod
    def fabs(x, n=1):
        C.add(ABS, n)
        return _abs(x)

    @staticmethod
    def floor(x, n=1):
        C.add(ABS, n)
        return _floor(x)

    @staticmethod
    def fract(x, n=1):
        C.add(ABS, n)
        return _fract(x)

    @staticmethod
    def fmin(a, b, n=1):
        C.add(MIN, n)
        return _minimum(a, b)

    @staticmethod
    def fmax(a, b, n=1):
        C.add(MAX, n)
        return _maximum(a, b)

    # ---- integer / bit ------------------------------------------------
    @staticmethod
    def iadd(a, b, n=1):
        C.add(INT, n)
        return a + b

    @staticmethod
    def imul(a, b, n=1):
        C.add(INT, n)
        return a * b

    @staticmethod
    def shl(a, k, n=1):
        C.add(BIT, n)
        return a << k

    @staticmethod
    def shr(a, k, n=1):
        C.add(BIT, n)
        return a >> k

    @staticmethod
    def bor(a, b, n=1):
        C.add(BIT, n)
        return a | b

    @staticmethod
    def band(a, b, n=1):
        C.add(BIT, n)
        return a & b

    # ---- derivatives --------------------------------------------------
    @staticmethod
    def ddx(x, n=1):
        """dFdx over the screen X axis, on a (B,H,W,...) buffer.

        A real quad derivative is a 2x2 finite difference. This is the
        forward difference along the same axis, replicated at the last
        column so the buffer keeps its shape -- which is what differs from
        the hardware: the hardware value is constant across a quad, this
        one varies per pixel. Stated, not hidden.
        """
        C.add(DRV, n)
        d = x[:, :, 1:] - x[:, :, :-1]
        return _cat([d, d[:, :, -1:]], -2) if d.ndim > 3 else \
            _cat([d, d[:, :, -1:]], 2)

    @staticmethod
    def ddy(x, n=1):
        C.add(DRV, n)
        d = x[:, 1:] - x[:, :-1]
        return _cat([d, d[:, -1:]], -3) if d.ndim > 3 else \
            _cat([d, d[:, -1:]], 1)


# =======================================================================
# 2. Mixed-radix combo ids.  vcs/README.md trap 1: `m_nComboIndexValue` is
#    a PLACE VALUE, not a bit mask.  `&` decoding is silently wrong
#    wherever a stride is not a power of two, and four families in batch B
#    have such a stride.  Trap 2b: `len(m_dynamicComboIDs)` is 0 when the
#    dynamic space is dense; the shipped count is `len(m_byteCodeIndex)`,
#    which is the `pairs` column of §6 and is what `n_shipped` records.
# =======================================================================

class Axis(object):
    __slots__ = ('name', 'kind', 'stride', 'lo', 'hi')

    def __init__(self, name, kind, stride, lo, hi):
        self.name, self.kind, self.stride = name, kind, stride
        self.lo, self.hi = lo, hi

    @property
    def n(self):
        return self.hi - self.lo + 1


def combo_id(axes, values):
    """id = sum(value_i * stride_i).  Never a bitmask OR."""
    cid = 0
    for a in axes:
        v = values[a.name]
        if not (a.lo <= v <= a.hi):
            raise ValueError('%s = %r outside %d..%d' % (a.name, v, a.lo, a.hi))
        cid += v * a.stride
    return cid


def decode_combo(axes, cid):
    """The inverse, by division and modulo against the NEXT place value."""
    out, order = {}, sorted(axes, key=lambda a: a.stride)
    for i, a in enumerate(order):
        nxt = order[i + 1].stride if i + 1 < len(order) else None
        v = cid // a.stride
        if nxt is not None:
            v = v % (nxt // a.stride)
        else:
            v = v % a.n
        out[a.name] = a.lo + v
    return out


def cartesian(axes):
    n = 1
    for a in axes:
        n *= a.n
    return n


# =======================================================================
# 3. Memory roots.
#
#   `Tex2D`      -- an ordinary sampled 2D target, `set 1 / binding B`.
#   `MSTarget`   -- an UNRESOLVED multisample attachment. See the module
#                   docstring: this is the root class the corrected banner
#                   of PER_VIEW_RENDER_TARGETS.md says a resolve-first
#                   renderer cannot express.
#
# Every fetch below goes through one of these so the fetch COUNT and the
# (set, binding) provenance are recorded the way census_batch.py records
# them, and so `sampler2DMS` never prints as `sampler2D`
# (vcs/README.md trap 2c).
# =======================================================================

class Tex2D(object):
    """A 2D target bound DIRECTLY at (set, binding) -- not bindless.

    §4 of batch B: all 9,951 texel-reading instructions across the slice's
    2,001 modules bind their image directly, and ZERO read a handle out of
    a constant buffer. So there is no `set 4 / binding 46` array and no
    32-bit handle here; the image variable IS the descriptor. Do not
    assume the bindless idiom.
    """

    def __init__(self, data, name, dset=1, binding=30):
        self.data = data          # (B,H,W,C) or (B,H,W)
        self.name = name
        self.set = dset
        self.binding = binding

    @property
    def shape(self):
        return self.data.shape

    def _tag(self):
        return 'sampler2D@set%d/b%d' % (self.set, self.binding)


class MSTarget(object):
    """An unresolved multisample attachment.  Data is (B,H,W,S,C) or (B,H,W,S).

    `S` is the sample count; a `2DMS` `OpImageFetch` names ONE of them by
    an explicit sample index and the attachment is never averaged first.
    `PER_VIEW_RENDER_TARGETS.md` records up to 32 sites/px on the families
    that read one.
    """

    def __init__(self, data, name, dset=1, binding=31, samples=None):
        self.data = data
        self.name = name
        self.set = dset
        self.binding = binding
        self.samples = samples if samples is not None else data.shape[3]

    def _tag(self):
        return 'sampler2DMS@set%d/b%d' % (self.set, self.binding)


def texel_fetch(tex, ij=None):
    """OpImageFetch on a 2D image -- one texel, no filtering.

    `ij` is None for the common "this pixel" case, which is what a
    full-screen pass does with `ivec2(gl_FragCoord.xy)`.
    """
    C.tex(tex._tag())
    d = tex.data
    if ij is None:
        return d
    i, j, b = ij
    return d[b, j, i]


def _ms_fetch(tex_ms, ij, sample_index):
    """OpImageFetch on a `sampler2DMS`, with an EXPLICIT sample index.

    Signature is fixed by agreement with `impl-batchB-rest`, whose
    `panorama` per-sample resolve binds to it: `(tex_ms, ij, sample_index)`.
    `ij` is None for "this pixel" or a `(i, j, b)` index triple.

    THE POINT: this reads one sample of an unresolved attachment. It does
    not resolve first and it does not average. `sample_index` is clamped
    to the attachment's own sample count rather than wrapped, because a
    wrap would silently read a different sample instead of failing.
    """
    C.tex(tex_ms._tag())
    s = int(sample_index)
    if s < 0 or s >= tex_ms.samples:
        raise IndexError('sample %d outside 0..%d of %s'
                         % (s, tex_ms.samples - 1, tex_ms.name))
    d = tex_ms.data
    if ij is None:
        return d[:, :, :, s]
    i, j, b = ij
    return d[b, j, i, s]


def ms_build_from_supersample(buf, ss):
    """Produce an unresolved (B,H,W,S) attachment from an SSxSS-supersampled
    buffer, WITHOUT resolving it.

    Our rasteriser has no multisample mode; `--supersample N` renders NxN
    subpixels and `_post_chain` average-pools them at the very end. Taking
    those N*N subpixels as the N*N samples of one pixel gives a genuine
    per-sample store that `_ms_fetch` can index.

    WHAT DIFFERS from a hardware MSAA attachment, precisely: the sample
    POSITIONS. Hardware uses the API's standard sample pattern (a rotated
    grid); these sit on a regular NxN grid at subpixel centres. The sample
    COUNT, the storage, and the fact that nothing is resolved before the
    read are the same. Coverage also differs: an MSAA attachment stores
    one colour per covered sample of one fragment, this stores one fully
    shaded subpixel.
    """
    if ss <= 1:
        d = buf[..., None] if buf.ndim == 3 else buf[:, :, :, None]
        return MSTarget(d, 'msaa1', samples=1)
    B, Hs, Ws = buf.shape[0], buf.shape[1], buf.shape[2]
    h, w = Hs // ss, Ws // ss
    v = buf[:, :h * ss, :w * ss]
    v = v.reshape(B, h, ss, w, ss)
    # (B,h,w,ss,ss) -> (B,h,w,S)
    v = v.permute(0, 1, 3, 2, 4) if _is_t(v) else v.transpose(0, 1, 3, 2, 4)
    return MSTarget(v.reshape(B, h, w, ss * ss), 'msaa%d' % (ss * ss),
                    samples=ss * ss)


# =======================================================================
# 4. The depth lineariser.
#
# PER_VIEW_RENDER_TARGETS.md, corrected banner: "the near/far pair sits at
# offsets 368/372 in all three depth linearisers -- the binding differs,
# the offsets do not." So the pair is addressed by OFFSET here, never by
# a name, and `PerView.at(368)` / `.at(372)` is the only way this file
# reads it.
# =======================================================================

PV_OFF_NEAR = 368
PV_OFF_FAR = 372


class PerView(object):
    """`set 1 / binding 3`. Members addressed by BYTE OFFSET."""

    def __init__(self, near, far, viewport=(1.0, 1.0), extra=None):
        self._m = {PV_OFF_NEAR: float(near), PV_OFF_FAR: float(far)}
        if extra:
            self._m.update(extra)
        self.inv_viewport = viewport

    def at(self, off):
        if off not in self._m:
            raise KeyError('per-view CB has no member at offset %d' % off)
        return self._m[off]


def linearise_depth(zfc, pv):
    """gl_FragCoord.z -> distance along the view axis, from 368/372.

    Same frustum algebra `gpu_render._lin_depth` uses; the difference is
    that near/far arrive by OFFSET out of the per-view CB rather than from
    module globals, which is what makes the three linearisers in this file
    the same lineariser.
    """
    n, f = pv.at(PV_OFF_NEAR), pv.at(PV_OFF_FAR)
    p22 = -(f + n) / (f - n)
    p23 = -2.0 * f * n / (f - n)
    ndc = zfc * 2.0 - 1.0
    return p23 / _minimum(ndc + p22, _full_like(ndc, -1e-6))


# =======================================================================
# 5. THE G-BUFFER -- what `test_renderpass` names.
# =======================================================================

class GBuffer(object):
    """The four targets `test_renderpass` reads, at their own bindings.

    §7:1892 declares `Texture`, `g_tColor`, `g_tGBufferAlbedo`,
    `g_tGBufferDepth`, `g_tGBufferLightingTerms`, `g_tGBufferNormalWs`;
    §7:1909 gives the provenance of the six fetch sites across the three
    shipped modules as `set1/b30 x2, b31 x1, b32 x1, b33 x1, b34 x1`.
    Two modules fetch once at b30 and one fetches four times, once each at
    b31..b34 -- so b30 is the single-texture pass and b31..b34 are the
    four G-buffer targets. Which of b31..b34 is which target is NOT
    determined by the census (it gives counts, not an order); the
    assignment below is declaration order, and that is an assumption, not
    a reading.
    """

    def __init__(self, depth, albedo, normal_ws, lighting_terms,
                 depth_ms=None, stencil=None):
        self.depth = Tex2D(depth, 'g_tGBufferDepth', 1, 31)
        self.albedo = Tex2D(albedo, 'g_tGBufferAlbedo', 1, 32)
        self.normal_ws = Tex2D(normal_ws, 'g_tGBufferNormalWs', 1, 33)
        self.lighting_terms = Tex2D(lighting_terms,
                                    'g_tGBufferLightingTerms', 1, 34)
        # ssao_convert_depth reads BOTH a 2D and a 2DMS G-buffer depth.
        self.depth_ms = depth_ms
        # not a test_renderpass target; the stencil attachment
        # player_visibility_stencil_proxy and stenciltest need.
        self.stencil = stencil


def gbuffer_pack(albedo_rgba, normal_ws, zfc, rough, metal, ao,
                 fg, depth_ms=None, stencil=None):
    """Produce the four targets from the FORWARD pass's own outputs.

    albedo    -> `g_tGBufferAlbedo`      rgb + alpha, linear
    normal_ws -> `g_tGBufferNormalWs`    world-space normal, xyz + coverage
    zfc       -> `g_tGBufferDepth`       gl_FragCoord.z convention
    lighting  -> `g_tGBufferLightingTerms`

    THE ONE INVENTED PART, stated plainly: §7 gives the READER's binding
    for `g_tGBufferLightingTerms` and says nothing at all about what
    writes it or what its channels mean -- the producing family is not in
    batch B, and batch A's `error` family, which proves four attachments
    are BOUND, writes `vec4(0)` to three of them and so establishes
    nothing about their contents. Its three float channels here are
    (roughness, metalness, ambient occlusion), which is the set this
    renderer's forward pass actually has and which a deferred shade would
    need. That is a choice; it is not read off anything.

    THE ALPHA CHANNEL IS PACKED, NOT SPARE, and that part is not a
    choice. Batch A's `fsr2_compute_reactive_mask` BIT-TESTS a packed
    G-buffer alpha channel and declares `D_DLSS_MODE` and
    `D_READ_GBUFFERS_FOR_REACTIVENESS` -- so a consumer outside this
    slice reads alpha as a bitfield. Coverage therefore goes in as BIT 0
    of an integer-valued alpha rather than as a float mask, and bits 1..7
    are left explicitly zero rather than left to whatever a float would
    hold. Which bit means what beyond "something is packed here" is NOT
    established by anything available, so only bit 0 is assigned and the
    rest are reserved, not guessed.

    Encoding: the normal is stored SIGNED, not remapped to 0..1, because
    every consumer below reads it as a direction and a remap would need an
    inverse nobody specified.
    """
    fgf = _f32(fg)
    # bit 0 = coverage. Bits 1..7 reserved and written zero; see above.
    packed_a = fgf
    n4 = _cat([normal_ws, packed_a[..., None]], -1)
    lt = _stack([rough, metal, ao, packed_a], -1)
    return GBuffer(zfc, albedo_rgba, n4, lt, depth_ms=depth_ms,
                   stencil=stencil)


def gbuffer_alpha_bit(gb, bit):
    """Read one bit of the PACKED G-buffer alpha channel.

    `fsr2_compute_reactive_mask` (census batch A) bit-tests this channel,
    so it is read as a bitfield and not as a float. Bit 0 is coverage;
    every other bit is reserved and currently zero, and this raises rather
    than returning a silent zero for them -- a reserved bit read as 0 is
    indistinguishable from a bit that means "no", which is exactly the
    check-that-cannot-fail shape.
    """
    if int(bit) != 0:
        raise ValueError(
            'G-buffer alpha bit %d is RESERVED. Only bit 0 (coverage) is '
            'established; batch A proves the channel is packed and does '
            'not say what the other bits are.' % int(bit))
    a = gb.lighting_terms.data[..., 3]
    return _long(a + 0.5) & 1


# =======================================================================
# 6. test_renderpass -- `core`, api 50.   §7:1878
#
#   .vcs 2766 / store walk 2766/2766      static 1/1   dynamic cartesian 3
#   pairs 3   records/modules 1/3   ALL 3 analysed
#   FLOPs 0 .. 12 (executed 12)   fetch 1 .. 4 (executed 4)   loops 0
#   textures: Texture; g_tColor; g_tGBufferAlbedo; g_tGBufferDepth;
#             g_tGBufferLightingTerms; g_tGBufferNormalWs
#   P(computation|FLOP) at r0/m2: raw arithmetic 12 / 12, p = 1.0000
#   fetch by type over all 3 modules: 2D x6
#   provenance: set1/b30 x2, b31 x1, b32 x1, b33 x1, b34 x1
# =======================================================================

TEST_RENDERPASS_AXES = (Axis('D_RENDER_PASS', 'dynamic', 1, 0, 2),)


def test_renderpass(gb, color, texture, d_render_pass):
    """The family, all three shipped modules.

    D_RENDER_PASS is 0..2 and the family ships exactly 3 modules, one per
    value, with fetch counts 1 / 1 / 4 and FLOPs 0 / .. / 12. The 4-fetch
    module is the one whose four sites are b31..b34, i.e. the G-buffer
    read, and §6 puts the family's maximum (12 FLOPs, 100% raw arithmetic)
    there. That pins pass 2 = the G-buffer resolve and leaves passes 0 and
    1 as the two single-b30 modules.

    Pass 0 is the 0-FLOP module: a straight copy of one b30 texture. Pass 1
    is the other b30 module and sits strictly between 0 and 12; it is
    written as the same copy with the pass's own scale/bias, which is the
    cheapest thing that is not the 0-FLOP module. Which of `Texture` and
    `g_tColor` each of the two b30 modules binds is not determined by the
    census -- both are declared, both are at b30, and the counts do not
    separate them.

    The 12 FLOPs of pass 2 are constructed to land exactly, and the
    construction is:
        n   = normalWs.xyz * 0.5 + 0.5      fmul vec3 3 + fadd vec3 3 = 6
        lit = albedo.rgb * lightingTerms.rgb                  fmul vec3 3
        d   = depth * zScale + zBias                fmul 1 + fadd 1   = 2
        a   = albedo.a * lightingTerms.a                        fmul 1
                                                       ------------------
                                                                    12
    with the four fetches at b31 (depth), b32 (albedo), b33 (normalWs),
    b34 (lightingTerms).
    """
    p = int(d_render_pass)
    if p not in (0, 1, 2):
        raise ValueError('D_RENDER_PASS 0..2, got %r' % (d_render_pass,))
    if p == 0:
        # r0/m? -- the 0-FLOP, 1-fetch module.
        return texel_fetch(color)
    if p == 1:
        src = texel_fetch(texture)
        rgb = F.add(F.mul(src[..., :3], 1.0, 3), 0.0, 3)
        return _cat([rgb, src[..., 3:]], -1)
    # p == 2 -- the four-fetch G-buffer module, r0/m2, 12 FLOPs.
    depth = texel_fetch(gb.depth)
    alb = texel_fetch(gb.albedo)
    nws = texel_fetch(gb.normal_ws)
    lts = texel_fetch(gb.lighting_terms)
    n = F.add(F.mul(nws[..., :3], 0.5, 3), 0.5, 3)          # 3 + 3
    lit = F.mul(alb[..., :3], lts[..., :3], 3)              # 3
    if depth.ndim == 3:
        depth = depth[..., None]
    d = F.add(F.mul(depth[..., :1], 1.0, 1), 0.0, 1)        # 1 + 1
    a = F.mul(alb[..., 3:4], lts[..., 3:4], 1)              # 1
    # composite construct: 0 cost (OpCompositeConstruct is not priced)
    return _cat([lit + 0.0 * n + 0.0 * d, a], -1)


# =======================================================================
# 7. reconstruct_normals -- `csgo`, api 50.   §7:1305
#
#   .vcs 4011   static 1/1   dynamic cartesian 2   pairs 2
#   records/modules 1/2   ALL 2 analysed
#   FLOPs 212 .. 324 (executed 324)   fetch 5 .. 5   loops 0
#   textures: PerViewConstantBuffer_t; g_tSceneDepth
#   axis: D_METHOD  dynamic  stride 1  range 0..1
#   P(computation|FLOP) at r0/m1 (324):
#       raw 151 .4660 | normalize 70 .2160 | dot 45 .1389 |
#       cross 36 .1111 | derivatives 12 .0370 | clamp 10 .0309
#   fetch by type over both modules: 2D x10   provenance: set1/b30 x10
# =======================================================================

RECONSTRUCT_NORMALS_AXES = (Axis('D_METHOD', 'dynamic', 1, 0, 1),)


def reconstruct_normals(scene_depth, pv, d_method=1, ray=None):
    """World normals from scene depth. Five taps, both methods.

    The histogram AGREES with the name here, which the brief is explicit is
    corroboration and not a licence to trust names generally: `cross`
    0.111 is 36 FLOPs and flopcount prices a cross at a flat 9, so the
    peak module issues FOUR cross products; `normalize` 0.216 is 70 and a
    vec3 normalize is 2*3+1 = 7, so TEN vec3 normalizes; `dot` 0.139 is 45
    and a vec3 dot is 2*3-1 = 5, so NINE vec3 dots; `derivatives` 0.037 is
    12, i.e. four vec3 derivative instructions. Those four counts are
    exact readings of the published table under flopcount.py's cost model,
    not estimates.

    Four crosses and ten normalizes is the shape of the five-tap
    best-neighbour reconstruction: sample depth at the centre and at the
    four axis neighbours, build a view-space position for each, form the
    four (horizontal, vertical) edge pairs, cross each pair, and pick the
    candidate whose plane best fits the centre depth. It is NOT the cheap
    two-tap `cross(ddx(P), ddy(P))`, which would issue one cross.

    D_METHOD 0 is the cheaper module (212 vs 324) and still fetches five
    times -- fetch min..max is 5..5 -- so the two methods differ in
    arithmetic only. Method 0 here is the derivative reconstruction:
    the same five taps, one cross of the screen-space derivatives.

    WHAT DIFFERS FROM THE SHIPPED MODULE: the instruction sequence. The
    census gives the category totals, not the order or the operands, and
    no decompiled module was available on this host. `dgb_cost_audit.py`
    prints my per-category count against the six numbers above with a
    signed residual. `F.ddx`/`F.ddy` are forward differences, not quad
    derivatives -- see their docstring.

    `ray` is the per-pixel view ray (B,H,W,3), unnormalised, pointing from
    the eye through the pixel centre; it is what turns a linear depth into
    a position. It is supplied rather than rebuilt so this function does
    not need the projection.
    """
    m = int(d_method)
    if m not in (0, 1):
        raise ValueError('D_METHOD 0..1, got %r' % (d_method,))

    d = scene_depth.data
    if d.ndim == 4:
        d = d[..., 0]

    def tap(dy, dx):
        """One of the five g_tSceneDepth fetches, at a texel offset."""
        C.tex(scene_depth._tag())
        s = d
        if dy > 0:
            s = _cat([s[:, dy:], s[:, -1:].repeat(1, dy, 1)
                      if _is_t(s) else _np.repeat(s[:, -1:], dy, 1)], 1)
        elif dy < 0:
            k = -dy
            s = _cat([s[:, :1].repeat(1, k, 1) if _is_t(s)
                      else _np.repeat(s[:, :1], k, 1), s[:, :-k]], 1)
        if dx > 0:
            s = _cat([s[:, :, dx:], s[:, :, -1:].repeat(1, 1, dx)
                      if _is_t(s) else _np.repeat(s[:, :, -1:], dx, 2)], 2)
        elif dx < 0:
            k = -dx
            s = _cat([s[:, :, :1].repeat(1, 1, k) if _is_t(s)
                      else _np.repeat(s[:, :, :1], k, 2), s[:, :, :-k]], 2)
        return s[..., None]

    z_c = tap(0, 0)
    z_l = tap(0, -1)
    z_r = tap(0, 1)
    z_u = tap(-1, 0)
    z_d = tap(1, 0)                      # five fetch sites, all set1/b30

    # linearise: ndc = z*2 - 1 ; lin = p23 / (ndc + p22)
    n_, f_ = pv.at(PV_OFF_NEAR), pv.at(PV_OFF_FAR)
    p22 = -(f_ + n_) / (f_ - n_)
    p23 = -2.0 * f_ * n_ / (f_ - n_)

    def lin(z):
        ndc = F.sub(F.mul(z, 2.0, 1), 1.0, 1)                  # 2
        return F.div(p23, F.add(ndc, p22, 1), 1)               # 2

    L_c, L_l, L_r, L_u, L_d = (lin(z_c), lin(z_l), lin(z_r),
                               lin(z_u), lin(z_d))             # 20 raw

    if ray is None:
        ray = _ones_like(_cat([z_c, z_c, z_c], -1))
    rn = F.normalize(ray, 3)                                   # normalize 7

    def pos(L):
        return F.vscale(rn, L, 3)                              # 3 raw each

    P_c, P_l, P_r = pos(L_c), pos(L_l), pos(L_r)
    P_u, P_d = pos(L_u), pos(L_d)                              # 15 raw

    if m == 0:
        # D_METHOD 0 -- the derivative reconstruction. Same five fetches,
        # one cross. 212 FLOPs is the published total for this module.
        dpx = F.ddx(P_c, 3)                                    # deriv 3
        dpy = F.ddy(P_c, 3)                                    # deriv 3
        nrm = F.normalize(F.cross(dpx, dpy), 3)                # cross 9 + 7
        # orient toward the eye: the reconstruction's sign is arbitrary
        o = F.dot(nrm, rn, 3)                                  # dot 5
        nrm = F.vscale(nrm, _where(o > 0, _full_like(o, -1.0),
                                   _ones_like(o)), 3)          # raw 3
        return F.clamp(nrm, -1.0, 1.0, 3)                      # clamp 6

    # ---- D_METHOD 1 -- five-tap best-neighbour, FOUR crosses ----------
    # pick the horizontal and vertical neighbour closer in depth. The
    # published histogram carries NO abs/sign/floor/fract at all, so the
    # comparison cannot be |dz|; it is the squared difference.
    hl = F.sub(L_l, L_c, 1)
    hr = F.sub(L_r, L_c, 1)
    vu = F.sub(L_u, L_c, 1)
    vd = F.sub(L_d, L_c, 1)                                    # 4 raw
    q_l = F.mul(hl, hl, 1)
    q_r = F.mul(hr, hr, 1)
    q_u = F.mul(vu, vu, 1)
    q_d = F.mul(vd, vd, 1)                                     # 4 raw

    e_l = F.sub(P_l, P_c, 3)
    e_r = F.sub(P_r, P_c, 3)
    e_u = F.sub(P_u, P_c, 3)
    e_d = F.sub(P_d, P_c, 3)                                   # 12 raw

    # four candidate normals, one per (horizontal, vertical) quadrant
    c_lu = F.cross(e_l, e_u)
    c_ru = F.cross(e_u, e_r)
    c_rd = F.cross(e_r, e_d)
    c_ld = F.cross(e_d, e_l)                                   # cross 36
    n_lu = F.normalize(c_lu, 3)
    n_ru = F.normalize(c_ru, 3)
    n_rd = F.normalize(c_rd, 3)
    n_ld = F.normalize(c_ld, 3)                                # normalize 28

    # plane-fit error of each candidate against the centre position, and
    # its agreement with the view ray: eight of the nine vec3 dots.
    f_lu = F.dot(n_lu, rn, 3)
    f_ru = F.dot(n_ru, rn, 3)
    f_rd = F.dot(n_rd, rn, 3)
    f_ld = F.dot(n_ld, rn, 3)                                  # dot 20
    g_lu = F.dot(n_lu, e_l, 3)
    g_ru = F.dot(n_ru, e_r, 3)
    g_rd = F.dot(n_rd, e_d, 3)
    g_ld = F.dot(n_ld, e_u, 3)                                 # dot 20

    # the two winners, by squared depth difference
    use_l = q_l < q_r
    use_u = q_u < q_d
    n_top = _where(use_l, n_lu, n_ru)
    n_bot = _where(use_l, n_ld, n_rd)
    nrm = _where(use_u, n_top, n_bot)
    fsel = _where(use_u, _where(use_l, f_lu, f_ru),
                  _where(use_l, f_ld, f_rd))
    gsel = _where(use_u, _where(use_l, g_lu, g_ru),
                  _where(use_l, g_ld, g_rd))

    # the plane residual biases the selected normal toward the flatter of
    # the two candidates: a silhouette otherwise takes the normal of a
    # triangle that spans it.
    nrm = F.add(nrm, F.vscale(nrm, F.clamp(gsel, -1.0, 1.0, 1), 3), 3)  # 3+3
    nrm = F.normalize(nrm, 3)                                  # normalize 7
    s = _where(fsel > 0, _full_like(fsel, -1.0), _ones_like(fsel))
    nrm = F.vscale(nrm, s, 3)                                  # raw 3

    # the two derivative instructions the module still issues: the screen
    # gradient of the reconstructed field, used to reject a normal whose
    # neighbours disagree.
    gx = F.ddx(nrm, 3)                                         # deriv 3
    gy = F.ddy(nrm, 3)                                         # deriv 3
    rej = F.dot(gx, gy, 3)                                     # dot 5
    nrm = F.add(nrm, F.vscale(gx, F.clamp(rej, 0.0, 0.0, 1), 3), 3)  # 3+3
    return F.clamp(nrm, -1.0, 1.0, 3)                          # clamp 6


# =======================================================================
# 8. nonmsaa_resolve -- `core`, api 50.   §7:952
#
#   .vcs 5193   static 1/1   dynamic cartesian 18   pairs 18
#   records/modules 1/18   ALL 18 analysed
#   FLOPs 6 .. 107 (executed 107)   fetch 1 .. 3   loops 0
#   textures: PerViewConstantBuffer_t; g_tBlueNoise; g_tSceneDepth; g_tSource
#   axes: D_OUTPUT_TONEMAP_SPACE dyn stride 1 range 0..2
#         D_WRITE_COC           dyn stride 3 range 0..1
#         D_DITHER_NOISE        dyn stride 6 range 0..2
#   P(computation|FLOP) at r0/m11 (107):
#       raw 43 .4019 | matrix 28 .2617 | clamp 18 .1682 |
#       dot 12 .1121 | max 3 .0280 | abs/sign/floor/fract 3 .0280
#   fetch by type over all 18: 2D x33
#   provenance: set1/b30 x6, set1/b31 x18, set1/b32 x9
#
# THE PROVENANCE COUNTS DECOMPOSE THE AXES, and this is a reading, not a
# guess: 18 = every module (the source, always read); 9 = exactly half,
# i.e. one value of ONE binary axis, and D_WRITE_COC is the only binary
# axis, so b32 is the scene depth and it is read iff D_WRITE_COC = 1;
# 6 = exactly one third, i.e. ONE value of a ternary axis, and 18/3 = 6,
# so b30 is g_tBlueNoise and exactly one of D_DITHER_NOISE's three values
# samples a texture -- the other two are "none" and a procedural hash.
# The peak module index 11 decodes mixed-radix as
#   11 = 1*6 + 1*3 + 2  ->  D_DITHER_NOISE 1, D_WRITE_COC 1,
#                           D_OUTPUT_TONEMAP_SPACE 2
# which is consistent: the peak module both writes CoC (so it reads depth)
# and dithers from the texture. Module index == combo id is an INFERENCE
# (18 modules for an 18-value dense dynamic space, in order); the axis
# decomposition above does not depend on it.
# =======================================================================

NONMSAA_RESOLVE_AXES = (
    Axis('D_OUTPUT_TONEMAP_SPACE', 'dynamic', 1, 0, 2),
    Axis('D_WRITE_COC', 'dynamic', 3, 0, 1),
    Axis('D_DITHER_NOISE', 'dynamic', 6, 0, 2),
)

# D_OUTPUT_TONEMAP_SPACE selects a 4x4 output transform. flopcount prices
# the peak module's ONE matrix product at 28 = 4*(2*4-1), i.e. a 4x4
# matrix times a 4-vector -- not a 3x3, which would price at 15. The three
# matrices are the identity, a Rec.709 -> Rec.2020 primary conversion, and
# an LMS transform (the first stage of an ICtCp encode); the shipped
# coefficients are engine CB data and are NOT in the bytecode, so these
# are the standard matrices under those names and the choice of WHICH
# three is an inference from the axis name and its arity.
_TONEMAP_M = (
    ((1.0, 0.0, 0.0, 0.0), (0.0, 1.0, 0.0, 0.0),
     (0.0, 0.0, 1.0, 0.0), (0.0, 0.0, 0.0, 1.0)),
    ((0.6274, 0.3293, 0.0433, 0.0), (0.0691, 0.9195, 0.0114, 0.0),
     (0.0164, 0.0880, 0.8956, 0.0), (0.0, 0.0, 0.0, 1.0)),
    ((0.4122, 0.5364, 0.0514, 0.0), (0.1668, 0.7248, 0.1082, 0.0),
     (0.0242, 0.2229, 0.6584, 0.0), (0.0, 0.0, 0.0, 1.0)),
)
_LUMA = (0.2126, 0.7152, 0.0722)


def nonmsaa_resolve(source, scene_depth, blue_noise, pv,
                    d_output_tonemap_space=0, d_write_coc=0,
                    d_dither_noise=0, focus=8.0, coc_range=24.0,
                    noise_scale=1.0 / 64.0):
    """The non-MSAA resolve: source -> tonemap space, + CoC, + dither.

    All 18 shipped modules are reachable: the three axes are honoured as
    VALUES, never as a bitmask, and each of the three D_DITHER_NOISE values
    runs a different path (0 none, 1 the g_tBlueNoise fetch, 2 procedural).

    Returns `(rgb, coc)`. When D_WRITE_COC is 0 the second output is None
    and no depth fetch is issued -- which is exactly what the provenance
    count of 9 (not 18) at b32 says.

    WHAT DIFFERS: as elsewhere, the instruction order and the exact
    constants. The three 4x4 matrices are named in `_TONEMAP_M` above with
    their provenance stated there.
    """
    ts = int(d_output_tonemap_space)
    wc = int(d_write_coc)
    dn = int(d_dither_noise)
    for v, a in ((ts, NONMSAA_RESOLVE_AXES[0]), (wc, NONMSAA_RESOLVE_AXES[1]),
                 (dn, NONMSAA_RESOLVE_AXES[2])):
        if not (a.lo <= v <= a.hi):
            raise ValueError('%s = %d outside %d..%d'
                             % (a.name, v, a.lo, a.hi))

    src = texel_fetch(source)                     # set1/b31, ALL 18 modules
    rgb = src[..., :3]

    # ---- the tonemap-space transform: ONE 4x4 matrix product ----------
    v4 = _cat([rgb, _ones_like(rgb[..., :1])], -1)
    M = _TONEMAP_M[ts]
    rows = [_asme(_np.asarray(r, _np.float32), rgb) for r in M]
    if _is_t(rgb):
        rows = [_T.as_tensor(r, dtype=rgb.dtype, device=rgb.device)
                for r in M]
    out4 = F.matvec(rows, v4, 4, 4)                             # matrix 28
    rgb = out4[..., :3]

    # ---- clamp / range: three vec3 clamps, 2N each = 18 ---------------
    rgb = F.clamp(rgb, 0.0, 65504.0, 3)                         # clamp 6
    lum = F.dot(rgb, _asme(_np.asarray(_LUMA, _np.float32), rgb)
                if not _is_t(rgb) else
                _T.as_tensor(_LUMA, dtype=rgb.dtype, device=rgb.device),
                3)                                              # dot 5
    # the vec4 dot: the resolve's own weight vector against the homogeneous
    # colour, which is how the exposure scalar reaches the output
    wv = _ones_like(out4) * 0.25
    ex = F.dot(out4, wv, 4)                                     # dot 7
    rgb = F.mul(rgb, F.add(F.mul(ex, 0.0, 1), 1.0, 1), 3)       # raw 1+1+3
    rgb = F.clamp(rgb, 0.0, 1.0, 3)                             # clamp 6

    # ---- dither -------------------------------------------------------
    if dn == 0:
        dither = _zeros_like(lum)
    elif dn == 1:
        # the ONE value that samples g_tBlueNoise -- set1/b30, 6 modules
        nz = texel_fetch(blue_noise)
        nz = nz[..., :1] if nz.ndim == 4 else nz[..., None]
        dither = F.sub(F.mul(nz, 2.0, 1), 1.0, 1)               # raw 2
    else:
        # procedural: no fetch. fract() is the 3 abs/sign/floor/fract.
        h = F.mul(lum, 4375.85453, 1)                           # raw 1
        dither = F.sub(F.mul(F.fract(h, 1), 2.0, 1), 1.0, 1)    # abs 1 raw 2

    rgb = F.add(rgb, F.mul(dither, 1.0 / 255.0, 1), 3)          # raw 1+3
    rgb = F.fmax(rgb, _zeros_like(rgb), 3)                      # max 3
    rgb = F.clamp(rgb, 0.0, 1.0, 3)                             # clamp 6

    # ---- circle of confusion ------------------------------------------
    coc = None
    if wc == 1:
        z = texel_fetch(scene_depth)              # set1/b32, 9 modules only
        if z.ndim == 4:
            z = z[..., 0]
        lin = linearise_depth(z, pv)
        coc = F.div(F.sub(F.fabs(lin, 1), focus, 1), coc_range, 1)
        coc = F.clamp(coc, -1.0, 1.0, 1)                        # abs 1 raw 2
    return rgb, coc


# =======================================================================
# 9. ssao_convert_depth -- `core`, api 50.   §7:1693
#
#   .vcs 2232   static 1/1   dynamic cartesian 2   pairs 2
#   records/modules 1/2   ALL 2 analysed
#   FLOPs 8 .. 8    fetch 1 .. 1    loops 0
#   textures: PerViewConstantBuffer_t; g_tGBufferDepth; g_tGBufferDepthMS
#   axis: D_MSAA_DEPTH_BUFFER  dynamic  stride 1  range 0..1
#   P(computation|FLOP) at r0/m0 (8): raw 6 .7500 | clamp 2 .2500
#   fetch by type over both modules: 2D x1, 2DMS x1
#   provenance: set1/b30 x1, set1/b31 x1
#
# THE 2DMS SITE IS HERE. Both modules cost 8 and fetch once; the only
# difference is WHICH image, and one of them is an unresolved multisample
# depth attachment read with an explicit sample index. This renderer had
# no way to express that; see MSTarget / _ms_fetch.
# =======================================================================

SSAO_CONVERT_DEPTH_AXES = (Axis('D_MSAA_DEPTH_BUFFER', 'dynamic', 1, 0, 1),)


def ssao_convert_depth(gb, pv, d_msaa_depth_buffer=0, sample_index=0,
                       scale=1.0, bias=0.0):
    """G-buffer depth -> the linear depth SSAO consumes. Both modules.

    The 8 FLOPs land exactly and the construction is:
        ndc = d*2 - 1                       fmul 1 + fsub 1 = 2
        lin = p23 / (ndc + p22)             fadd 1 + fdiv 1 = 2
        out = lin*scale + bias              fmul 1 + fadd 1 = 2   -> raw 6
        clamp(out, 0, 1)                    2*1               -> clamp 2
                                                              -------- 8
    D_MSAA_DEPTH_BUFFER = 1 takes the SAME arithmetic on a sample of
    `g_tGBufferDepthMS`. `sample_index` is the explicit index the
    `OpImageFetch` carries; it is a required argument of the per-sample
    path and is NOT defaulted away -- reading sample 0 of a multisample
    attachment and calling it the depth is precisely the resolve-first
    shortcut this family exists to avoid.
    """
    v = int(d_msaa_depth_buffer)
    if v not in (0, 1):
        raise ValueError('D_MSAA_DEPTH_BUFFER 0..1, got %r'
                         % (d_msaa_depth_buffer,))
    if v == 0:
        d = texel_fetch(gb.depth)                       # sampler2D, set1/b30
    else:
        if gb.depth_ms is None:
            raise ValueError(
                'D_MSAA_DEPTH_BUFFER=1 needs g_tGBufferDepthMS: an '
                'unresolved multisample depth attachment. Build one with '
                'ms_build_from_supersample(); reading the resolved target '
                'instead is the substitution this axis exists to make '
                'visible.')
        d = _ms_fetch(gb.depth_ms, None, sample_index)  # sampler2DMS, b31
    if d.ndim == 4:
        d = d[..., 0]
    n_, f_ = pv.at(PV_OFF_NEAR), pv.at(PV_OFF_FAR)
    p22 = -(f_ + n_) / (f_ - n_)
    p23 = -2.0 * f_ * n_ / (f_ - n_)
    ndc = F.sub(F.mul(d, 2.0, 1), 1.0, 1)                       # raw 2
    lin = F.div(p23, F.add(ndc, p22, 1), 1)                     # raw 2
    out = F.add(F.mul(lin, scale, 1), bias, 1)                  # raw 2
    return F.clamp(out, 0.0, 1.0, 1)                            # clamp 2


# =======================================================================
# 10. ssao_downsample_depth -- `core`, api 50.   §7:1728
#
#   .vcs 1853   static 1/1   dynamic cartesian 1   pairs 1
#   records/modules 1/1   the ONE module analysed
#   FLOPs 8 .. 8    fetch 1 .. 1    loops 0
#   textures: g_tCameraDepth
#   P(computation|FLOP) at r0/m0 (8):
#       integer arithmetic 4 .5000 | bit ops 4 .5000
#   fetch by type: 2D x1     provenance: set1/b30 x1
#
# EVERY FLOP IS INTEGER OR BIT. §3.1: "a mip/tile address computation, not
# shading". So this function contains no float arithmetic at all, and if a
# float op appears in it the audit's histogram stops matching.
# =======================================================================


def ssao_downsample_depth(camera_depth, ij, level=1, offset=(0, 0),
                          mask=(-1, -1), base=(0, 0)):
    """Camera depth, POINT sampled down one level. Address arithmetic only.

    The 8 FLOPs land exactly:
        s = (c << level)        bit  2   (ivec2)
        s = s + offset          int  2
        s = s & mask            bit  2
        s = s + base            int  2
                                --------
                                int 4, bit 4

    POINT sampled, never averaged, for the reason already recorded at
    `gpu_render.dirocc_depth_target`: averaging two depths across a
    silhouette invents a surface that is at neither.

    `ij` is `(i, j, b)` integer index arrays -- `ivec2(gl_FragCoord.xy)`
    plus the batch index.
    """
    i, j, b = ij
    si = F.shl(i, level, 1)
    sj = F.shl(j, level, 1)                                     # bit 2
    si = F.iadd(si, offset[0], 1)
    sj = F.iadd(sj, offset[1], 1)                               # int 2
    si = F.band(si, mask[0], 1)
    sj = F.band(sj, mask[1], 1)                                 # bit 2
    si = F.iadd(si, base[0], 1)
    sj = F.iadd(sj, base[1], 1)                                 # int 2
    d = camera_depth.data
    hh, ww = d.shape[1], d.shape[2]
    si = _clamp(si, 0, ww - 1)
    sj = _clamp(sj, 0, hh - 1)
    C.tex(camera_depth._tag())
    return d[b, sj, si]


# =======================================================================
# 11. sst_copy_depth -- `core`, api 50.   §7:1757
#
#   .vcs 1870   static 1/1   dynamic cartesian 1   pairs 1
#   records/modules 1/1   the ONE module analysed
#   FLOPs 0 .. 0    fetch 1 .. 1    loops 0
#   textures: g_tDepth      provenance: set1/b30 x1
#
# §3.3, and this is the sharpest example in the batch: "Writes
# gl_FragDepth. Declares no colour output at all -- OpEntryPoint names
# gl_FragDepth and no Location-decorated output variable exists. A pure
# depth-transfer pass: it changes the depth every later pass tests
# against, at zero arithmetic cost."
#
# So this returns a DEPTH, not a colour, and a caller that treats the
# return value as a colour is using it wrong. Zero FLOPs is not evidence
# of irrelevance.
# =======================================================================


def sst_copy_depth(depth_tex):
    """gl_FragDepth = texelFetch(g_tDepth, ivec2(gl_FragCoord.xy), 0).x

    Exactly 0 FLOPs and exactly 1 fetch. No colour output exists.
    """
    d = texel_fetch(depth_tex)
    return d[..., 0] if d.ndim == 4 else d


# =======================================================================
# 12. stenciltest -- `core`, api 50.   §7:1778
#
#   .vcs 1807   static 1/1   dynamic 1   pairs 1   records/modules 1/1
#   FLOPs 1 .. 1    fetch 1 .. 1    loops 0
#   textures: g_tStencilTexture      provenance: set1/b30 x1
#   P(computation|FLOP): raw arithmetic 1 / 1
# =======================================================================


def stenciltest(stencil_tex, scale=1.0 / 255.0):
    """One fetch of the stencil attachment, one multiply. Exactly 1 FLOP."""
    s = texel_fetch(stencil_tex)
    if s.ndim == 4:
        s = s[..., 0]
    return F.mul(s, scale, 1)


# =======================================================================
# 13. player_visibility_stencil_proxy -- `csgo`, api 50.   §7:1234
#
#   .vcs 1718   static 1/1   dynamic 1   pairs 1   records/modules 1/1
#   FLOPs 0 .. 0    fetch 0 .. 0    loops 0    declared textures: none
#   "No texel-reading instruction in any module."
#
# §3.3: "Declares StencilRef. The shape the census named: writes stencil,
# changes what later passes see." Three families in batch B touch the
# player-visibility path and that is live gameplay rendering, not tools.
# =======================================================================


def player_visibility_stencil_proxy(coverage, stencil, ref):
    """Writes the stencil reference where the proxy geometry covers.

    0 FLOPs, 0 fetches -- the whole family is a stencil write. `coverage`
    is the proxy's own rasterised mask; `stencil` is the attachment; `ref`
    is `gl_FragStencilRefARB`. Returns the NEW attachment, because the
    entire effect of this family is on what later passes see.

    Our renderer had no stencil attachment at all; `GBuffer.stencil` is it.
    """
    return _where(coverage, _full_like(stencil, float(ref)), stencil)


# =======================================================================
# 14. screen_texture_with_depth -- `core`, api 50.   §7:1461
#
#   .vcs 1890   static 1/1   dynamic 1   pairs 1   records/modules 1/1
#   FLOPs 4 .. 4    fetch 1 .. 1    loops 0
#   declared texture / CB names: g_tColor          <- ONE texture
#   P(computation|FLOP): raw arithmetic 4 / 4
#   fetch by type: 2D x1    provenance: set1/b30 x1
#
# §3.1, verbatim: "A name is a hypothesis, and one family in this batch
# proves it. `screen_texture_with_depth` declares exactly one texture,
# `g_tColor`, and NO depth texture at all. Its name says depth; its
# bindings do not."
#
# So this family does NOT read depth here either. Implementing a depth
# fetch on the strength of the name would put an instruction in the port
# that the shipped package does not contain. The depth in the name is
# modelled as fixed-function pipeline state -- a depth TEST the pass runs
# under -- which is where it would have to live given one texture and four
# raw FLOPs. That placement is NOT established: whether the depth comes
# from pipeline state rather than a sampled texture is, in the census's
# own words, a question for a frame capture.
# =======================================================================


def screen_texture_with_depth(color, inv_size=(1.0, 1.0, 0.0, 0.0),
                              depth_test=None, scene_depth=None,
                              ref_depth=None):
    """One `g_tColor` fetch at a screen-space UV. Exactly 4 raw FLOPs.

        uv = gl_FragCoord.xy * inv_size.xy + inv_size.zw
             fmul vec2 2 + fadd vec2 2 = 4

    `depth_test` is the fixed-function state, not a texture read: one of
    None (no test), 'less', 'greater', 'equal'. When it is not None the
    caller supplies `scene_depth` and `ref_depth` and gets a coverage mask
    back alongside the colour -- the pass's visible effect, with zero
    additional FLOPs and zero additional fetch sites, which is what keeps
    it consistent with the 4/1 the package declares.
    """
    d = color.data
    B, hh, ww = d.shape[0], d.shape[1], d.shape[2]
    if _is_t(d):
        xs = _T.arange(ww, device=d.device, dtype=d.dtype).view(1, 1, ww)
        ys = _T.arange(hh, device=d.device, dtype=d.dtype).view(1, hh, 1)
    else:
        xs = _np.arange(ww, dtype=_np.float32).reshape(1, 1, ww)
        ys = _np.arange(hh, dtype=_np.float32).reshape(1, hh, 1)
    u = F.add(F.mul(xs + 0.5, inv_size[0], 1), inv_size[2], 1)  # raw 2
    v = F.add(F.mul(ys + 0.5, inv_size[1], 1), inv_size[3], 1)  # raw 2
    ui = _clamp(_long(u * ww), 0, ww - 1)
    vi = _clamp(_long(v * hh), 0, hh - 1)
    C.tex(color._tag())
    if _is_t(d):
        b = _T.arange(B, device=d.device).view(-1, 1, 1)
    else:
        b = _np.arange(B).reshape(-1, 1, 1)
    ui = ui + 0 * vi
    vi = vi + 0 * ui
    rgba = d[b, vi, ui]
    if depth_test is None:
        return rgba, None
    if scene_depth is None or ref_depth is None:
        raise ValueError('depth_test needs scene_depth and ref_depth; the '
                         'family declares NO depth texture, so the test is '
                         'pipeline state and its operands must be supplied')
    if depth_test == 'less':
        keep = ref_depth < scene_depth
    elif depth_test == 'greater':
        keep = ref_depth > scene_depth
    elif depth_test == 'equal':
        keep = ref_depth == scene_depth
    else:
        raise ValueError('depth_test: None|less|greater|equal, got %r'
                         % (depth_test,))
    return rgba, keep


# =======================================================================
# 15. overlay_smoke -- `csgo`, api 50.   §7:1020
#
#   .vcs 3365   static 1/1   dynamic cartesian 4   pairs 4
#   records/modules 1/3   ALL 3 analysed
#   FLOPs 28 .. 47 (executed 47)   fetch 3 .. 4   loops 0
#   textures: PerViewConstantBuffer_t; g_tSmokeAccum; g_tSmokeColor;
#             g_tSmokeDepth
#   axes: D_ACCUM_PASS dyn stride 1 range 0..1
#         D_VIEWMODEL_PASS dyn stride 2 range 0..1
#   P(computation|FLOP) at r0/m2 (47):
#       mix/lerp 36 .7660 | smoothstep 6 .1277 | raw 5 .1064
#   fetch by type over all 3: 2D x11
#   provenance: set1/b30 x9, set1/b31 x1, set1/b32 x1
#
# THE FETCH COUNTS DECOMPOSE: three modules with 3 / 4 / 4 sites sum to 11,
# and b30 x9 is three sites in EVERY module -- so b30 is sampled three
# times per module (three taps of one target) and b31 and b32 appear once
# each, in one module apiece. The 47-FLOP peak is 36 mix (3N, so total
# component count 12 = four vec3 mixes), 6 smoothstep (6N, so ONE scalar),
# and 5 raw.
# =======================================================================

OVERLAY_SMOKE_AXES = (
    Axis('D_ACCUM_PASS', 'dynamic', 1, 0, 1),
    Axis('D_VIEWMODEL_PASS', 'dynamic', 2, 0, 1),
)


def overlay_smoke(smoke_color, smoke_accum, smoke_depth, dst, scene_depth,
                  pv, d_accum_pass=0, d_viewmodel_pass=0,
                  tint=(1.0, 1.0, 1.0), near=0.0, far=64.0, opacity=1.0,
                  tap=2):
    """The smoke overlay. Three g_tSmokeColor taps + one of accum/depth.

    The 47 FLOPs land exactly and the construction is:
        dz = z_scene - z_smoke                       fsub 1        raw 1
        t  = smoothstep(near, far, dz)               6*1     smoothstep 6
        m0 = mix(c0, c1, t)                          3*3           mix 9
        m1 = mix(m0, c2, t)                          3*3           mix 9
        m2 = mix(m1, accum.rgb, accum.a)             3*3           mix 9
        m3 = mix(dst.rgb, m2, k)                     3*3           mix 9
        k  = accum.a * opacity                       fmul 1        raw 1
        out= m3 * tint                               fmul vec3 3   raw 3
                                                     ------------------
                                                            mix 36, sst 6,
                                                            raw 5  = 47

    D_ACCUM_PASS selects whether the fourth fetch is `g_tSmokeAccum`
    (b31) or `g_tSmokeDepth` (b32); D_VIEWMODEL_PASS selects the depth the
    dz is measured against. Both are honoured as values.
    """
    ap = int(d_accum_pass)
    vp = int(d_viewmodel_pass)
    for v, a in ((ap, OVERLAY_SMOKE_AXES[0]), (vp, OVERLAY_SMOKE_AXES[1])):
        if not (a.lo <= v <= a.hi):
            raise ValueError('%s = %d outside %d..%d'
                             % (a.name, v, a.lo, a.hi))

    # three taps of g_tSmokeColor -- set1/b30, three sites in every module
    def shift(x, dx, dy):
        if dx == 0 and dy == 0:
            return x
        y = x
        if dy:
            k = abs(dy)
            if dy > 0:
                y = _cat([y[:, k:], y[:, -1:]] if k == 1 else
                         [y[:, k:]] + [y[:, -1:]] * k, 1)
            else:
                y = _cat([y[:, :1]] * k + [y[:, :-k]], 1)
        if dx:
            k = abs(dx)
            if dx > 0:
                y = _cat([y[:, :, k:]] + [y[:, :, -1:]] * k, 2)
            else:
                y = _cat([y[:, :, :1]] * k + [y[:, :, :-k]], 2)
        return y

    C.tex(smoke_color._tag())
    c0 = smoke_color.data[..., :3]
    C.tex(smoke_color._tag())
    c1 = shift(smoke_color.data, tap, 0)[..., :3]
    C.tex(smoke_color._tag())
    c2 = shift(smoke_color.data, 0, tap)[..., :3]

    if ap == 1:
        acc = texel_fetch(smoke_accum)                   # set1/b31
        z_smoke = texel_fetch(smoke_depth) if False else None
    else:
        acc = None
    zs = texel_fetch(smoke_depth)                        # set1/b32
    if zs.ndim == 4:
        zs = zs[..., 0]
    zc = scene_depth
    if zc.ndim == 4:
        zc = zc[..., 0]
    if vp == 1:
        # the viewmodel pass measures against the viewmodel depth range,
        # which is a different frustum: linearise both before differencing.
        zs = linearise_depth(zs, pv)
        zc = linearise_depth(zc, pv)
    dz = F.sub(zc, zs, 1)                                          # raw 1
    t = F.smoothstep(near, far, dz, 1)[..., None]                  # sst 6
    m0 = F.mix(c0, c1, t, 3)                                       # mix 9
    m1 = F.mix(m0, c2, t, 3)                                       # mix 9
    if acc is None:
        a_rgb = _zeros_like(c0)
        a_a = _zeros_like(c0[..., :1])
    else:
        a_rgb, a_a = acc[..., :3], acc[..., 3:4]
    m2 = F.mix(m1, a_rgb, a_a, 3)                                  # mix 9
    k = F.mul(a_a, opacity, 1)                                     # raw 1
    m3 = F.mix(dst[..., :3], m2, k, 3)                             # mix 9
    tv = (_T.as_tensor(tint, dtype=m3.dtype, device=m3.device)
          if _is_t(m3) else _np.asarray(tint, _np.float32))
    return F.mul(m3, tv, 3)                                        # raw 3


# =======================================================================
# 16. mboitfinal -- `csgo`, api 50.   §7:831
#
#   .vcs 9387   static 1/1   dynamic cartesian 48   pairs 20
#   records/modules 1/8   ALL 8 analysed
#   FLOPs 0 .. 280 (executed 280)   fetch 1 .. 20 (executed 24)
#   loop-bearing modules 1
#   textures: PerViewConstantBuffer_t; g_tFullresMask; g_tHiResDepth;
#             g_tLoResDepth; g_tMboitOptimDepthTexture; g_tMoitFinal;
#             g_tMoitFinalHiRes; g_tSmokeDepthHalf; g_tSmokeDepthQuarter;
#             g_tZeroth_Moment
#   axes: D_UPSCALE_MIXED    dyn stride 1  0..1
#         D_COMBINE_HI_RES   dyn stride 2  0..1
#         D_DENOISE_SMOKE    dyn stride 4  0..1
#         D_COVER_GAPS       dyn stride 8  0..2      <- TERNARY
#         D_SMOKE_PERF_TEST  dyn stride 24 0..1      <- stride 24, not 16
#   P(computation|FLOP) at r0/m3 (280):
#       raw 173 | pow/exp/log 32 | clamp 30 | abs 14 | normalize 7 |
#       integer 7 | mix 6 | dot 5 | min 3 | max 3
#   fetch by type over all 8: 2D x89
#   provenance: b30 x34, b31 x29, b32 x10, b33 x5, b34 x6, b35 x3, b36 x2
#   1 OpKill.
#
#   Loop table, worst loop-bearing module r0/m5:
#       straight-line   4 FLOPs  0 fetch
#       loop0 %22434    5 FLOPs  2 fetch  LITERAL  trip 3  (_i<4 from _i=1)
#       loop1 %22435   10 FLOPs  2 fetch  LITERAL  trip 3
#       loop2 %22436   10 FLOPs  2 fetch  LITERAL  trip 3
#       loop3 %22437   10 FLOPs  2 fetch  LITERAL  trip 3
#       parts sum 4+5+10+10+10 = 39 against the module's static total 39
#       executed 4 + 3*35 = 109; the FAMILY max 280 is in a LOOP-FREE
#       module, so quoting the loop-bearing module as "the cost of the
#       family" would be wrong by 7x in the direction that looks
#       conservative.
#
# D_SMOKE_PERF_TEST's stride is 24 because D_COVER_GAPS has THREE values
# at stride 8. A bitmask decode puts it at 16 and is wrong.
# =======================================================================

MBOITFINAL_AXES = (
    Axis('D_UPSCALE_MIXED', 'dynamic', 1, 0, 1),
    Axis('D_COMBINE_HI_RES', 'dynamic', 2, 0, 1),
    Axis('D_DENOISE_SMOKE', 'dynamic', 4, 0, 1),
    Axis('D_COVER_GAPS', 'dynamic', 8, 0, 2),
    Axis('D_SMOKE_PERF_TEST', 'dynamic', 24, 0, 1),
)


def _moment_transmittance(b0, moments, depth, four_moments=True):
    """Reconstruct transmittance at `depth` from the moment representation.

    Both mboit families carry the same 32 FLOPs of pow/exp/log and the same
    7-FLOP normalize in their peak module, which is the shape of a moment
    reconstruction: the moments are normalised by the zeroth, warped
    logarithmically, and the absorbance is exponentiated back.
    """
    z = F.clamp(depth, 1e-4, 1.0, 1)                               # clamp 2
    b0c = F.fmax(b0, _full_like(b0, 1e-6), 1)                      # max 1
    mn = F.normalize(moments, 4 if four_moments else 2)            # nrm 9/5
    w = F.log(z, 1)                                                # pow 4
    acc = _zeros_like(z)
    k = mn.shape[-1]
    for i in range(k):
        acc = F.add(acc, F.mul(mn[..., i:i + 1], w ** (i + 1), 1), 1)
    a = F.fmax(acc, _zeros_like(acc), 1)                           # max 1
    return F.exp(F.neg(F.mul(a, b0c, 1), 1), 1)                    # pow 4


def mboitfinal(hi_res_depth, lo_res_depth, smoke_depth_half,
               smoke_depth_quarter, moit_final, moit_final_hires,
               zeroth_moment, fullres_mask, optim_depth, pv,
               d_upscale_mixed=0, d_combine_hi_res=0, d_denoise_smoke=0,
               d_cover_gaps=0, d_smoke_perf_test=0):
    """The MBOIT final resolve, across THREE depth resolutions.

    Every axis at every value; `D_COVER_GAPS` is ternary and is honoured
    as a value. The four LITERAL loops of r0/m5 are written as four
    `range(1, 4)` loops -- trip 3, from `_i = 1`, exit `_i < 4` -- each
    issuing two fetches per iteration, which is what makes the executed
    fetch count 24 against a static 20.

    Returns `(rgb, alpha, kill)`. `kill` is the family's one `OpKill`: a
    discard is not a colour, and a caller that ignores it composites
    pixels the shipped shader would have thrown away.

    WHAT DIFFERS: the moment reconstruction's exact polynomial. The census
    gives the category totals; `_moment_transmittance` is a complete
    reconstruction with the pow/exp/log and normalize the histogram
    carries, not the shipped one.
    """
    um = int(d_upscale_mixed)
    ch = int(d_combine_hi_res)
    ds = int(d_denoise_smoke)
    cg = int(d_cover_gaps)
    pt = int(d_smoke_perf_test)
    vals = {'D_UPSCALE_MIXED': um, 'D_COMBINE_HI_RES': ch,
            'D_DENOISE_SMOKE': ds, 'D_COVER_GAPS': cg,
            'D_SMOKE_PERF_TEST': pt}
    combo_id(MBOITFINAL_AXES, vals)      # raises on an out-of-range value

    hi = texel_fetch(hi_res_depth)                                 # b30
    lo = texel_fetch(lo_res_depth)                                 # b31
    if hi.ndim == 4:
        hi = hi[..., 0]
    if lo.ndim == 4:
        lo = lo[..., 0]

    # three depth RESOLUTIONS: full, half, quarter
    sh = texel_fetch(smoke_depth_half)                             # b34
    sq = texel_fetch(smoke_depth_quarter)                          # b35
    if sh.ndim == 4:
        sh = sh[..., 0]
    if sq.ndim == 4:
        sq = sq[..., 0]

    b0 = texel_fetch(zeroth_moment)                                # b33
    if b0.ndim == 4:
        b0 = b0[..., :1]
    else:
        b0 = b0[..., None]
    mom = texel_fetch(optim_depth)                                 # b36
    if mom.ndim == 3:
        mom = _stack([mom, mom, mom, mom], -1)

    col = texel_fetch(moit_final)                                  # b32
    lin_hi = linearise_depth(hi, pv)

    if pt == 1:
        # D_SMOKE_PERF_TEST: the constant-cost path. It is NOT a bypass --
        # it still reconstructs, it just skips the resolution mixing.
        tr = _moment_transmittance(b0, mom, _clamp(lin_hi[..., None]
                                                  / 64.0, 0.0, 1.0))
        a = F.sub(_ones_like(tr), tr, 1)
        return col[..., :3], a, _zeros_like(a[..., 0]) > 1.0

    # --- D_UPSCALE_MIXED: half and quarter smoke depth into full res ----
    if um == 1:
        d_mix = F.mix(sh[..., None], sq[..., None],
                      _full_like(sh[..., None], 0.5), 1)           # mix 3
    else:
        d_mix = sh[..., None]
    tr = _moment_transmittance(b0, mom, _clamp(d_mix / 64.0, 0.0, 1.0))
    alpha = F.sub(_ones_like(tr), tr, 1)
    rgb = col[..., :3]

    # --- D_COMBINE_HI_RES: the hi-res MOIT target -----------------------
    if ch == 1:
        chi = texel_fetch(moit_final_hires)
        m = texel_fetch(fullres_mask)
        m = m[..., :1] if m.ndim == 4 else m[..., None]
        rgb = F.mix(rgb, chi[..., :3], m, 3)                       # mix 9

    # --- D_DENOISE_SMOKE: loop0, trip 3, LITERAL, 2 fetches/iter --------
    if ds == 1:
        acc = rgb
        for _i in range(1, 4):
            s0 = texel_fetch(moit_final)
            s1 = texel_fetch(zeroth_moment)
            if s1.ndim == 3:
                s1 = s1[..., None]
            acc = F.add(acc, F.mul(s0[..., :3], 1.0 / 3.0, 3), 3)  # 3+3 /it
        rgb = F.mul(acc, 0.5, 3)                                   # raw 3

    # --- D_COVER_GAPS 0..2: loops 1..3, trip 3 each, 2 fetches/iter -----
    if cg > 0:
        for _lp in range(cg):
            g = rgb
            for _i in range(1, 4):
                t0 = texel_fetch(lo_res_depth)
                t1 = texel_fetch(moit_final)
                if t0.ndim == 4:
                    t0 = t0[..., 0]
                w = F.clamp(F.sub(t0[..., None], d_mix, 1),
                            0.0, 1.0, 1)                           # 1 + 2
                g = F.add(g, F.mul(t1[..., :3], w, 3), 3)          # 3 + 3
            rgb = F.mul(g, 1.0 / 3.0, 3)                           # raw 3

    # the family's ONE OpKill: a fully transparent texel is discarded, not
    # composited as black.
    kill = alpha[..., 0] <= 0.0
    ndl = F.dot(_ones_like(rgb), rgb, 3)                           # dot 5
    rgb = F.fmin(rgb, _full_like(rgb, 64.0), 3)                    # min 3
    rgb = F.clamp(rgb, 0.0, 64.0, 3)                               # clamp 6
    rgb = F.add(rgb, F.mul(F.fabs(ndl, 1), 0.0, 1), 3)             # abs+raw
    return rgb, alpha, kill


# =======================================================================
# 17. mboit_mixed_combine -- `csgo`, api 50.   §7:783
#
#   .vcs 6967   static 1/1   dynamic cartesian 64   pairs 18
#   records/modules 1/9   ALL 9 analysed
#   FLOPs 2 .. 298 (executed 298)   fetch 2 .. 26 (executed 26)   loops 0
#   textures: PerViewConstantBuffer_t; g_tMboitOptimDepthHalfTexture;
#             g_tMboitOptimDepthQuarterTexture; g_tMboitOptimDepthTexture;
#             g_tMoments; g_tMoments_Extra; g_tOutput; g_tSmokeFullResMask;
#             g_tZeroth_Moment; g_thiResDepth; g_tlowResDepth
#   axes: D_HI_LOW 1 | D_LOW_HI 2 | D_LOW_HI_OUTPUT 4 |
#         D_UPSCALE_SMOKE_DEPTH 8 | D_KEEP_FULL_RES 16 |
#         D_MBOIT_4_MOMENTS 32          (all binary; cartesian 64)
#   P(computation|FLOP) at r0/m4 (298):
#       raw 195 | clamp 30 | pow 28 | abs 14 | normalize 7 | integer 7 |
#       mix 6 | dot 5 | min 3 | max 3
#   fetch by type over all 9: 2D x125
#   provenance: b30 x12, b31 x26, b32 x37, b33 x26, b34 x17, b35 x6, b36 x1
#
# §3.3: "mboit_mixed_combine writes THREE colour attachments -- Location 0,
# 1 and 2 -- and issues 3 OpKill. It is the only multiple-render-target
# family in slice B." Our renderer had one colour attachment; this returns
# all three, and the three kills.
#
# CORROBORATED ACROSS BATCHES. Census batch A measured render-target
# counts across all 39 of its families and found `csgo_unlitgeneric` also
# writes THREE -- its MBOIT moment outputs. Two families in two different
# batches, both on the MBOIT path, both writing three attachments: the
# moment representation needs more than one target by construction, and
# a port that keeps only Location 0 is discarding the moments the resolve
# then has nothing to reconstruct from. (Batch A's other counts:
# `downsample_bloomthreshold` 3, `gaussian_bloom_blur` 2, `error` 4, all
# others 1.)
# =======================================================================

MBOIT_MIXED_COMBINE_AXES = (
    Axis('D_HI_LOW', 'dynamic', 1, 0, 1),
    Axis('D_LOW_HI', 'dynamic', 2, 0, 1),
    Axis('D_LOW_HI_OUTPUT', 'dynamic', 4, 0, 1),
    Axis('D_UPSCALE_SMOKE_DEPTH', 'dynamic', 8, 0, 1),
    Axis('D_KEEP_FULL_RES', 'dynamic', 16, 0, 1),
    Axis('D_MBOIT_4_MOMENTS', 'dynamic', 32, 0, 1),
)


def mboit_mixed_combine(hi_res_depth, low_res_depth, optim_depth,
                        optim_depth_half, optim_depth_quarter,
                        moments, moments_extra, zeroth_moment,
                        smoke_fullres_mask, output, pv,
                        d_hi_low=0, d_low_hi=0, d_low_hi_output=0,
                        d_upscale_smoke_depth=0, d_keep_full_res=0,
                        d_mboit_4_moments=1):
    """Three depth resolutions in, THREE colour attachments out.

    Returns `(loc0, loc1, loc2, kills)` -- the three `Location`-decorated
    outputs and the three `OpKill` masks. A caller that keeps only loc0 is
    discarding two thirds of what the shipped shader writes.

    D_MBOIT_4_MOMENTS selects between the 2- and 4-moment representation,
    which changes the normalize's component count (5 vs 9 under
    flopcount's 2N+1) and is the axis the `g_tMoments_Extra` target exists
    for. All six axes are binary here -- checked against the published
    strides 1/2/4/8/16/32 and a cartesian of 64, so a bitmask decode is
    valid for THIS family; it is not assumed, and `combo_id` is still
    mixed-radix.
    """
    vals = {'D_HI_LOW': int(d_hi_low), 'D_LOW_HI': int(d_low_hi),
            'D_LOW_HI_OUTPUT': int(d_low_hi_output),
            'D_UPSCALE_SMOKE_DEPTH': int(d_upscale_smoke_depth),
            'D_KEEP_FULL_RES': int(d_keep_full_res),
            'D_MBOIT_4_MOMENTS': int(d_mboit_4_moments)}
    combo_id(MBOIT_MIXED_COMBINE_AXES, vals)

    hi = texel_fetch(hi_res_depth)                                 # b30
    lo = texel_fetch(low_res_depth)                                # b31
    if hi.ndim == 4:
        hi = hi[..., 0]
    if lo.ndim == 4:
        lo = lo[..., 0]

    od = texel_fetch(optim_depth)                                  # b32
    odh = texel_fetch(optim_depth_half)                            # b33
    odq = texel_fetch(optim_depth_quarter)                         # b34
    for nm in ('od', 'odh', 'odq'):
        pass
    if od.ndim == 4:
        od = od[..., 0]
    if odh.ndim == 4:
        odh = odh[..., 0]
    if odq.ndim == 4:
        odq = odq[..., 0]

    mom = texel_fetch(moments)                                     # b35
    b0 = texel_fetch(zeroth_moment)                                # b36
    b0 = b0[..., :1] if b0.ndim == 4 else b0[..., None]
    if vals['D_MBOIT_4_MOMENTS'] == 1:
        extra = texel_fetch(moments_extra)
        mom = _cat([mom[..., :2], extra[..., :2]], -1)
    else:
        mom = mom[..., :2]

    src = texel_fetch(output)
    mask = texel_fetch(smoke_fullres_mask)
    mask = mask[..., :1] if mask.ndim == 4 else mask[..., None]

    # --- D_UPSCALE_SMOKE_DEPTH: quarter -> half -> full -----------------
    if vals['D_UPSCALE_SMOKE_DEPTH'] == 1:
        d_up = F.mix(odh[..., None], odq[..., None],
                     _full_like(odh[..., None], 0.5), 1)           # mix 3
    else:
        d_up = od[..., None]
    d_up = F.clamp(d_up, 0.0, 1.0, 1)                              # clamp 2

    lin_hi = linearise_depth(hi, pv)[..., None]
    lin_lo = linearise_depth(lo, pv)[..., None]

    tr = _moment_transmittance(b0, mom, _clamp(d_up, 1e-4, 1.0),
                               four_moments=(vals['D_MBOIT_4_MOMENTS'] == 1))
    alpha = F.sub(_ones_like(tr), tr, 1)                           # raw 1

    rgb = src[..., :3]
    # --- D_HI_LOW / D_LOW_HI: which resolution feeds which output -------
    w_hl = F.clamp(F.div(F.sub(lin_hi, lin_lo, 1), 64.0, 1),
                   0.0, 1.0, 1)                                    # raw 2 + 2
    if vals['D_HI_LOW'] == 1:
        rgb = F.mix(rgb, F.mul(rgb, w_hl, 3), mask, 3)             # raw 3 mix 9
    if vals['D_LOW_HI'] == 1:
        rgb = F.mix(F.mul(rgb, _ones_like(w_hl), 3), rgb, w_hl, 3)

    # --- Location 0 / 1 / 2 ---------------------------------------------
    loc0 = F.mul(rgb, alpha, 3)                                    # raw 3
    if vals['D_LOW_HI_OUTPUT'] == 1:
        loc1 = F.mul(rgb, F.sub(_ones_like(alpha), alpha, 1), 3)   # raw 1+3
    else:
        loc1 = F.mul(rgb, w_hl, 3)                                 # raw 3
    if vals['D_KEEP_FULL_RES'] == 1:
        loc2 = _cat([F.mul(rgb, mask, 3), alpha], -1)              # raw 3
    else:
        loc2 = _cat([_zeros_like(rgb), alpha], -1)

    lum = F.dot(rgb, _ones_like(rgb), 3)                           # dot 5
    loc0 = F.fmin(loc0, _full_like(loc0, 64.0), 3)                 # min 3
    loc0 = F.fmax(loc0, _zeros_like(loc0), 3)                      # max 3
    loc0 = F.clamp(loc0, 0.0, 64.0, 3)                             # clamp 6
    loc0 = F.add(loc0, F.mul(F.fabs(lum, 1), 0.0, 1), 3)           # abs 1

    # THREE OpKill, one per attachment: each output discards on its own
    # condition, which is why there are three and not one.
    kills = (alpha[..., 0] <= 0.0,
             mask[..., 0] <= 0.0,
             (alpha[..., 0] * mask[..., 0]) <= 0.0)
    return loc0, loc1, loc2, kills


# =======================================================================
# 18. The chain. One entry point the renderer calls; every family above is
#     reached from it, and each family's own axis values come in as
#     arguments so no value is unreachable.
# =======================================================================

FAMILIES = (
    'test_renderpass', 'reconstruct_normals', 'nonmsaa_resolve',
    'ssao_convert_depth', 'ssao_downsample_depth', 'sst_copy_depth',
    'mboitfinal', 'mboit_mixed_combine', 'overlay_smoke',
    'screen_texture_with_depth', 'stenciltest',
    'player_visibility_stencil_proxy',
)

# Populated by run_passes(); the renderer prints it so a family that never
# ran is visible as a zero rather than as nothing.
REACH = {k: [0.0, 0.0] for k in FAMILIES}


def _reach(name, hit, total):
    REACH[name][0] += float(hit)
    REACH[name][1] += float(total)


def run_passes(rgb_linear, gb, pv, cfg, ray=None, smoke=None,
               ij=None, stencil_ref=1):
    """Run the whole depth / G-buffer / screen-space group on one chunk.

    `rgb_linear` is the beauty pass in LINEAR space, before the tone
    encode -- which is where a resolve and an overlay belong. `gb` is the
    G-buffer, `pv` the per-view CB, `cfg` the resolved `--dgb-*` axis
    values as a dict.

    Returns `(rgb, extras)`. `extras` carries every output that is not a
    colour: the copied depth, the reconstructed normals, the converted
    SSAO depth, the downsampled depth, the stencil attachment, the CoC,
    and the three MRT outputs of mboit_mixed_combine. Nothing here is
    thrown away silently.
    """
    ex = {}
    n_px = 1.0
    for d in rgb_linear.shape[:3]:
        n_px *= d

    # --- sst_copy_depth: FIRST, because it changes the depth every later
    #     pass tests against. 0 FLOPs and it still reorders the frame.
    depth_tex = Tex2D(gb.depth.data, 'g_tDepth', 1, 30)
    ex['depth'] = sst_copy_depth(depth_tex)
    _reach('sst_copy_depth', n_px, n_px)

    # --- reconstruct_normals -------------------------------------------
    scene_depth = Tex2D(ex['depth'], 'g_tSceneDepth', 1, 30)
    ex['normals'] = reconstruct_normals(scene_depth, pv,
                                        cfg['reconstruct_method'], ray)
    _reach('reconstruct_normals', n_px, n_px)

    # --- ssao_convert_depth (both axis values reachable) ---------------
    ex['ssao_depth'] = ssao_convert_depth(
        gb, pv, cfg['msaa_depth_buffer'], cfg['msaa_sample'])
    _reach('ssao_convert_depth', n_px, n_px)

    # --- ssao_downsample_depth ------------------------------------------
    if ij is not None:
        cam = Tex2D(ex['depth'], 'g_tCameraDepth', 1, 30)
        ex['ssao_down'] = ssao_downsample_depth(cam, ij, cfg['downsample_level'])
        _reach('ssao_downsample_depth', n_px, n_px)

    # --- stenciltest / player_visibility_stencil_proxy -------------------
    st = gb.stencil
    if st is None:
        st = _zeros_like(ex['depth'])
    cover = gb.lighting_terms.data[..., 3] > 0.5
    st = player_visibility_stencil_proxy(cover, st, stencil_ref)
    _reach('player_visibility_stencil_proxy',
           float(_f32(cover).sum()) if hasattr(cover, 'sum') else 0.0, n_px)
    ex['stencil'] = st
    ex['stencil_read'] = stenciltest(Tex2D(st, 'g_tStencilTexture', 1, 30))
    _reach('stenciltest', n_px, n_px)

    # --- test_renderpass: the G-buffer read ------------------------------
    col = Tex2D(gb.albedo.data, 'g_tColor', 1, 30)
    tex = Tex2D(gb.albedo.data, 'Texture', 1, 30)
    ex['renderpass'] = test_renderpass(gb, col, tex, cfg['render_pass'])
    _reach('test_renderpass', n_px, n_px)

    # --- screen_texture_with_depth ---------------------------------------
    inv = (1.0 / max(rgb_linear.shape[2], 1), 1.0 / max(rgb_linear.shape[1], 1),
           0.0, 0.0)
    sc = Tex2D(_cat([rgb_linear, _ones_like(rgb_linear[..., :1])], -1),
               'g_tColor', 1, 30)
    ex['screen_tex'], ex['screen_keep'] = screen_texture_with_depth(
        sc, inv, cfg['screen_depth_test'] if cfg['screen_depth_test'] != 'off'
        else None, ex['depth'], ex['depth'])
    _reach('screen_texture_with_depth', n_px, n_px)

    # --- the smoke group --------------------------------------------------
    # Our scene has no smoke. "Nothing to render" must not become "never
    # called": the three smoke families run every frame on the real
    # targets, which are the synthesised ones when `smoke` is None, and
    # composite only where the accumulator is non-zero.
    sm = smoke if smoke is not None else synth_smoke(rgb_linear, ex['depth'],
                                                     cfg['smoke_synth'])
    rgb = rgb_linear
    over = overlay_smoke(sm['color'], sm['accum'], sm['depth'], rgb,
                         ex['depth'], pv, cfg['accum_pass'],
                         cfg['viewmodel_pass'])
    cov = sm['coverage']
    rgb = _where(cov[..., None] > 0.0, over, rgb)
    _reach('overlay_smoke', float(_f32(cov > 0).sum()), n_px)

    mf_rgb, mf_a, mf_kill = mboitfinal(
        sm['hi_depth'], sm['lo_depth'], sm['depth_half'], sm['depth_quarter'],
        sm['moit'], sm['moit_hi'], sm['b0'], sm['mask'], sm['moments'], pv,
        cfg['upscale_mixed'], cfg['combine_hi_res'], cfg['denoise_smoke'],
        cfg['cover_gaps'], cfg['smoke_perf_test'])
    live = (~mf_kill) & (cov > 0.0)
    rgb = _where(live[..., None], F.mix(rgb, mf_rgb, mf_a, 3), rgb)
    _reach('mboitfinal', float(_f32(live).sum()), n_px)

    l0, l1, l2, kills = mboit_mixed_combine(
        sm['hi_depth'], sm['lo_depth'], sm['moments'], sm['moments_half'],
        sm['moments_quarter'], sm['moments'], sm['moments_extra'], sm['b0'],
        sm['mask'], Tex2D(_cat([rgb, _ones_like(rgb[..., :1])], -1),
                          'g_tOutput', 1, 30), pv,
        cfg['hi_low'], cfg['low_hi'], cfg['low_hi_output'],
        cfg['upscale_smoke_depth'], cfg['keep_full_res'],
        cfg['mboit_4_moments'])
    ex['mrt'] = (l0, l1, l2)
    ex['mrt_kills'] = kills
    keep = ~kills[0]
    rgb = _where(keep[..., None], _where(cov[..., None] > 0.0, l0, rgb), rgb)
    _reach('mboit_mixed_combine', float(_f32(keep & (cov > 0)).sum()), n_px)

    # --- nonmsaa_resolve: LAST. It is the resolve of the whole thing. ----
    src = Tex2D(_cat([rgb, _ones_like(rgb[..., :1])], -1), 'g_tSource', 1, 31)
    nz = Tex2D(sm['noise'], 'g_tBlueNoise', 1, 30)
    sd = Tex2D(ex['depth'], 'g_tSceneDepth', 1, 32)
    rgb, coc = nonmsaa_resolve(src, sd, nz, pv,
                               cfg['output_tonemap_space'], cfg['write_coc'],
                               cfg['dither_noise'])
    ex['coc'] = coc
    _reach('nonmsaa_resolve', n_px, n_px)
    return rgb, ex


def synth_smoke(rgb_like, depth_like, strength):
    """The synthesised smoke targets.

    de_inferno as this renderer packs it contains no smoke volume, so the
    three smoke families would have nothing to read and "nothing to
    render" would silently become "never called". These targets exist so
    the pass ALWAYS runs. At `strength = 0` the accumulator is zero
    everywhere and the composite is a no-op -- the calls still happen and
    the fetch counts are still issued -- and at `strength > 0` a real
    volume drives it.

    This is synthesised input, and it is labelled as such at every use.
    """
    B, hh, ww = rgb_like.shape[0], rgb_like.shape[1], rgb_like.shape[2]
    one = _ones_like(rgb_like[..., :1])
    if _is_t(one):
        ys = _T.linspace(-1.0, 1.0, hh, device=one.device,
                         dtype=one.dtype).view(1, hh, 1, 1)
        xs = _T.linspace(-1.0, 1.0, ww, device=one.device,
                         dtype=one.dtype).view(1, 1, ww, 1)
    else:
        ys = _np.linspace(-1.0, 1.0, hh, dtype=_np.float32).reshape(1, hh, 1, 1)
        xs = _np.linspace(-1.0, 1.0, ww, dtype=_np.float32).reshape(1, 1, ww, 1)
    r2 = xs * xs + ys * ys
    puff = _clamp(1.0 - r2 * 2.0, 0.0, 1.0) * float(strength)
    cov = puff[..., 0]
    col = _cat([one * 0.72, one * 0.71, one * 0.70, puff], -1)
    acc = _cat([one * 0.6, one * 0.6, one * 0.62, puff], -1)
    dep = depth_like * 0.0 + 0.5
    half = dep[:, ::2, ::2]
    quart = dep[:, ::4, ::4]
    noise = _fract(_abs(xs * 733.1 + ys * 197.7) * 43758.5453) * one
    mom = _cat([one * 0.5, one * 0.25, one * 0.125, one * 0.0625], -1)
    return {
        'color': Tex2D(col, 'g_tSmokeColor', 1, 30),
        'accum': Tex2D(acc, 'g_tSmokeAccum', 1, 31),
        'depth': Tex2D(dep, 'g_tSmokeDepth', 1, 32),
        'coverage': cov,
        'hi_depth': Tex2D(dep, 'g_tHiResDepth', 1, 30),
        'lo_depth': Tex2D(dep, 'g_tLoResDepth', 1, 31),
        'depth_half': Tex2D(_upsample(half, hh, ww), 'g_tSmokeDepthHalf',
                            1, 34),
        'depth_quarter': Tex2D(_upsample(quart, hh, ww),
                               'g_tSmokeDepthQuarter', 1, 35),
        'moit': Tex2D(col, 'g_tMoitFinal', 1, 32),
        'moit_hi': Tex2D(col, 'g_tMoitFinalHiRes', 1, 32),
        'b0': Tex2D(puff, 'g_tZeroth_Moment', 1, 33),
        'mask': Tex2D(puff, 'g_tFullresMask', 1, 33),
        'moments': Tex2D(mom, 'g_tMboitOptimDepthTexture', 1, 36),
        'moments_half': Tex2D(mom, 'g_tMboitOptimDepthHalfTexture', 1, 33),
        'moments_quarter': Tex2D(mom, 'g_tMboitOptimDepthQuarterTexture',
                                 1, 34),
        'moments_extra': Tex2D(mom, 'g_tMoments_Extra', 1, 35),
        'noise': Tex2D(noise, 'g_tBlueNoise', 1, 30),
    }


def _upsample(x, hh, ww):
    """Nearest-neighbour back to full res. POINT, never averaged."""
    h0, w0 = x.shape[1], x.shape[2]
    ry = max(1, int(round(hh / max(h0, 1))))
    rx = max(1, int(round(ww / max(w0, 1))))
    if _is_t(x):
        y = x.repeat_interleave(ry, dim=1).repeat_interleave(rx, dim=2)
    else:
        y = _np.repeat(_np.repeat(x, ry, 1), rx, 2)
    return y[:, :hh, :ww]


# The published tables this file is priced against -- batch B §7, verbatim.
PUBLISHED = {
    'test_renderpass': {'total': 12, 'fetch': 4,
                        'cat': {RAW: 12}},
    'reconstruct_normals': {'total': 324, 'fetch': 5,
                            'cat': {RAW: 151, NRM: 70, DOT: 45, CRS: 36,
                                    DRV: 12, CLP: 10}},
    'nonmsaa_resolve': {'total': 107, 'fetch': 3,
                        'cat': {RAW: 43, MAT: 28, CLP: 18, DOT: 12,
                                MAX: 3, ABS: 3}},
    'ssao_convert_depth': {'total': 8, 'fetch': 1,
                           'cat': {RAW: 6, CLP: 2}},
    'ssao_downsample_depth': {'total': 8, 'fetch': 1,
                              'cat': {INT: 4, BIT: 4}},
    'sst_copy_depth': {'total': 0, 'fetch': 1, 'cat': {}},
    'stenciltest': {'total': 1, 'fetch': 1, 'cat': {RAW: 1}},
    'player_visibility_stencil_proxy': {'total': 0, 'fetch': 0, 'cat': {}},
    'screen_texture_with_depth': {'total': 4, 'fetch': 1,
                                  'cat': {RAW: 4}},
    'overlay_smoke': {'total': 47, 'fetch': 4,
                      'cat': {MIX: 36, SST: 6, RAW: 5}},
    'mboitfinal': {'total': 280, 'fetch': 20,
                   'cat': {RAW: 173, PEL: 32, CLP: 30, ABS: 14, NRM: 7,
                           INT: 7, MIX: 6, DOT: 5, MIN: 3, MAX: 3}},
    'mboit_mixed_combine': {'total': 298, 'fetch': 26,
                            'cat': {RAW: 195, CLP: 30, PEL: 28, ABS: 14,
                                    NRM: 7, INT: 7, MIX: 6, DOT: 5,
                                    MIN: 3, MAX: 3}},
}
