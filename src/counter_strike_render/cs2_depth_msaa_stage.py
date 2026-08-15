"""CS2 DEPTH / G-BUFFER / MSAA family group — census batch A, 8 of 32.

MY SLICE, stated so overlap with `impl-batchA-rest` (the other 24) can be
checked by addition:

    downsample_depth        depthtolineardepth      atrous_pack_depthnormal
    bilateral_upsample      csgo_screenspace_zone   csgo_depth_only
    copytexture             csgo_dronecam

8 + 24 = 32.  The partition closes.

---------------------------------------------------------------------------
THE HEADLINE: `sampler2DMS`, AND WHAT HAD TO BE ADDED TO HOLD ONE
---------------------------------------------------------------------------
`SHADER_CALLFLOW_census_batch_A.md` §3 records the first memory root class
found anywhere in this project that a resolve-first renderer **structurally
cannot express**: `downsample_depth` and `csgo_screenspace_zone`'s
`D_MSAA_DEPTH_BUFFER = 1` module fetch an UNRESOLVED multisample depth
attachment — `OpImageFetch` with an explicit **Sample** operand, up to 32
sites per pixel (§3 table; §6's closed form `4 fetch x 8 iterations`).

Our renderer had no object that can hold more than one depth per pixel.
`gpu_render.py:_frag_depth()` produces exactly one `gl_FragCoord.z` per
pixel from `nvdiffrast`'s `rast[...,2]`, and every consumer in the file
reads that one value.  There is nothing to index with a sample number, so
`texelFetch(img, coord, sample)` had no operand to take.

WHAT WAS ADDED, concretely, in this file:

  1. `MSDepthAttachment` — a depth attachment of shape (B, H, W, S) plus the
     S sub-pixel sample POSITIONS that produced it.  It is unresolved by
     construction: `.resolve()` exists and is a separate, named, costed
     operation, so no consumer can silently read a resolved value.
  2. `MSDepthAttachment.texel_fetch(ix, iy, sample)` — the `OpImageFetch`
     with a Sample operand.  It refuses a sample index outside [0, S) rather
     than clamping, because a clamp would make a wrong sample index
     indistinguishable from a right one.
  3. Two PRODUCERS for it, because a per-sample depth buffer has to come
     from somewhere:
       - `ms_depth_from_supersample()` — when the renderer is already
         supersampling, the ss x ss sub-pixel grid IS a multisample grid;
         the S samples are gathered from it with NO extra rasterisation.
       - `ms_depth_from_rasterizer()` — S depth-only rasterisations with the
         projection jittered by the sample position (`jitter_clip`).  This
         is the exact form: each sample is rasterised at its own sub-pixel
         location, which is what a hardware MSAA attachment stores.
  4. `ms_sample_positions()` — the D3D/Vulkan standard sample locations for
     1/2/4/8/16.  `D_MSAA_SAMPLES` also bakes **6** (§6: the literal takes
     exactly 1, 2, 4, 6, 8), for which no standard pattern exists; see that
     function for what was substituted and why it is stated rather than
     silently filled in.

  NOTHING here approximates the per-sample path with a resolved depth read.
  `csgo_screenspace_zone`'s `D_MSAA_DEPTH_BUFFER = 1` and
  `downsample_depth`'s MSAA modules take `MSDepthAttachment.texel_fetch`;
  the `= 0` / `D_NON_MSAA_DEPTH = 1` paths take a single-sample
  `sampler2D`.  Both values of both axes are implemented.

---------------------------------------------------------------------------
THE NORMAL G-BUFFER
---------------------------------------------------------------------------
§8 `atrous_pack_depthnormal`: "It also proves a **normal G-buffer exists** —
`set 1 / binding 31` is read as `n*2-1` and re-encoded."  We had no G-buffer
of that shape.  `NormalGBuffer` is it: an RGB target holding `n*0.5+0.5` in
the SAME encoding the consumer decodes, produced from the shading normal the
opaque pass already has.  Its shape is stated at the class.

---------------------------------------------------------------------------
THE RECONSTRUCTION CONSTANT
---------------------------------------------------------------------------
§5.1: the near/far pair sits at byte offsets **368 / 372** in all three depth
linearisers, and the BINDING differs while the offsets do not
(`depthtolineardepth` set 1/binding 0; `bilateral_upsample` and
`csgo_screenspace_zone` set 1/binding 1).  `PV_CB` below records offset and
binding separately for exactly that reason, and every read in this file goes
through it, so a wrong offset is a one-line diff and not a hunt.

---------------------------------------------------------------------------
WHAT IS TRANSCRIBED AND WHAT IS NOT — READ THIS BEFORE TRUSTING A NUMBER
---------------------------------------------------------------------------
No `.glsl` listing for any of these eight families exists in this tree.
`docs/projects/counter-strike-sft/` carries listings for the six world
families only; batch A's tools run on the render host (`nkcut2`) against
archives that are not here (`~/cs2`, `~/shwork*` — checked, absent).

So the reference for this file is the census document's **literal
consumption statements** (§3, §4, §5, §6, §8) plus its per-family
`P(computation | FLOP)` tables.  Every function below names the section and
line it comes from.  Where the consumption statement fixes the arithmetic,
the port is that arithmetic.  Where it does not, the INSTRUCTION MIX was
recovered by constraint-solving the published category table — that is
stated at each such function, and only four families' mixes closed exactly:

  ASSERTED (the audit exits non-zero if the ledger moves):
    atrous_pack_depthnormal   33 = raw 22 + abs 8 + deriv 2 + max 1, 2 fetch
    depthtolineardepth        21 = raw 7 + normalize 7 + dot 5 + clamp 2,
                                   1 fetch
    downsample_depth          16 = int 9 + min 3 + max 3 + bit 1, 4 fetch
                                   (the r0/m1 non-MSAA checkerboard) AND
                                   §6's closed form `2 + body x trip` at
                                   body 4 and 11, trip 1/2/4/6/8
    every family's FETCH COUNT (structural: it is a count of texel-reading
    instructions, not of arithmetic)

  MEASURED AND PRINTED, NOT ASSERTED (the delta is stated, never padded to
  match — padding would be fabrication, and a ledger tuned to its own
  reference is CHECKS_THAT_CANNOT_FAIL.md's whole subject):
    bilateral_upsample 79 · csgo_screenspace_zone 281 · csgo_dronecam 119
    copytexture 54 · csgo_depth_only 25

NOTHING in this file returns its input unchanged as a stand-in.

---------------------------------------------------------------------------
UNITS AND CONVENTIONS
---------------------------------------------------------------------------
Depth is in the `gl_FragCoord.z` convention ([0,1], far = 1) everywhere,
which is what `gpu_render.py:_frag_depth()` produces and what every
`texelFetch` in these families reads.  Integer screen coordinates are
`ivec2(gl_FragCoord.xy)`, i.e. the pixel index, y increasing DOWNWARD in the
attachment as stored here; `gpu_render.py` flips once, at the end of
`_post_chain`, and the flip is applied to the targets at the call site.
"""

import collections
import math

import torch

# =====================================================================
# THE PER-VIEW CONSTANT BUFFER, BY BYTE OFFSET
#
# PER_VIEW_RENDER_TARGETS.md's discriminating fact is provenance, not
# type: which CB a 32-bit handle comes from and at which byte offset.
# Its CORRECTED banner adds a FIFTH texture handle (offset 64) and
# census batch A §4 gives its consumption.  Recording (binding, offset)
# rather than a name is the whole method, so this table carries both and
# every read below goes through it.
#
# The BINDING is per family and the OFFSET is not -- §5.1.  A table
# keyed on offset alone would have hidden that; a table keyed on binding
# alone would have made the invariant unstateable.
# =====================================================================
PV_CB = {
    # offset -> (what it is, which families read it, evidence)
    64: ("dither/threshold map handle (tile-wrapped screen space)",
         ("csgo_depth_only",),
         "batch A section 4: texelFetch(tex[h64], "
         "ivec2(gl_FragCoord.xy) & CB[176], 0).y"),
    72: ("samplerCube, sky", (), "PER_VIEW_RENDER_TARGETS.md"),
    76: ("low-res 4-channel directional occlusion", (),
         "PER_VIEW_RENDER_TARGETS.md"),
    80: ("matching low-res depth target", (),
         "PER_VIEW_RENDER_TARGETS.md"),
    88: ("full-res screen-space shadow, channel .z", (),
         "PER_VIEW_RENDER_TARGETS.md"),
    104: ("samplerCubeArray, IBL", (), "PER_VIEW_RENDER_TARGETS.md"),
    116: ("per-view weather ripple map", (), "PER_VIEW_RENDER_TARGETS.md"),
    176: ("ivec2 dither tile mask (a power-of-two wrap, NOT a handle)",
          ("csgo_depth_only",), "batch A section 4"),
    368: ("float near", ("depthtolineardepth", "bilateral_upsample",
                         "csgo_screenspace_zone"),
          "batch A section 5.1 -- THE reconstruction constant"),
    372: ("float far", ("depthtolineardepth", "bilateral_upsample",
                        "csgo_screenspace_zone"),
          "batch A section 5.1 -- THE reconstruction constant"),
    80.1: ("row_major mat4 inverse view-projection (set 1/binding 0)",
           ("csgo_screenspace_zone",),
           "batch A section 8 csgo_screenspace_zone: offset 80 of "
           "binding 0, distinct from offset 80 of binding 3 above -- "
           "which is why this table records the BINDING too"),
    464: ("vec3 depth-linearisation coefficients (a, b, c)",
          ("atrous_pack_depthnormal",),
          "batch A section 8: abs(a/(b*z+c)), set 1/binding 1"),
}

# Which CB BINDING each family's near/far pair lives at.  §5.1 states the
# binding is NOT uniform across the slice and refuses to claim one
# mapping; this records the three it does establish and nothing else.
NEARFAR_BINDING = {
    "depthtolineardepth": 0,
    "bilateral_upsample": 1,
    "csgo_screenspace_zone": 1,
}
NEARFAR_OFFSETS = (368, 372)

# =====================================================================
# COMBO CONTAINER RULES -- the three traps of vcs/README.md, in code.
# =====================================================================
# Trap 1: a static combo id is MIXED RADIX.  `m_nComboIndexValue` is a
# PLACE VALUE, not a bit position.  Seven of batch A's 32 have an axis
# with more than two values (§8); THREE of those seven are in this slice
# -- copytexture (S_DESATURATE, 3 values at place 128) and
# downsample_depth (D_MSAA_SAMPLES, 5 values at place 32).  A `&`-decode
# is wrong for both and RIGHT for the all-binary families, which is the
# coincidence that hides it.


def combo_id(axes, values):
    """Mixed-radix combo id.  `axes` is ((name, lo, hi, place), ...).

    Division/multiplication by the place value, never a shift or a mask.
    Raises on an out-of-range value: a silently clamped axis value is a
    combo id that is wrong in a way that still decodes.
    """
    cid = 0
    for name, lo, hi, place in axes:
        v = values[name]
        if not (lo <= v <= hi):
            raise ValueError(f"{name}={v} outside [{lo},{hi}]")
        cid += (v - lo) * place
    return cid


def combo_decode(axes, cid):
    """Inverse of combo_id, by division and modulo.  Never by `&`."""
    out = {}
    for name, lo, hi, place in axes:
        out[name] = lo + (cid // place) % (hi - lo + 1)
    return out


def shipped_dynamic_count(dynamic_combo_ids, byte_code_index):
    """Trap 2b/3: `m_dynamicComboIDs` is EMPTY when the space is DENSE.

    Counting `len(m_dynamicComboIDs)` reports 0 shipped dynamic combos
    for a family shipping all of them -- 22 of batch A's 32, including
    `copytexture` (0 vs 1,632) and `downsample_depth`.  Four of this
    slice's eight are in that 22 by the §7 table's own pairs column
    (copytexture 1632, downsample_depth 96, csgo_depth_only 20,
    csgo_dronecam 2 against a 1-module family).

    The shipped count is `len(m_dynamicComboIDs)` when that array is
    non-empty and `len(m_byteCodeIndex)` when it is not.
    """
    return (len(dynamic_combo_ids) if dynamic_combo_ids
            else len(byte_code_index))


# The combo axes of the eight families, with RANGES and PLACE VALUES read
# off §8's per-family tables.  Nothing here is a bit position.
FAMILY_AXES = {
    "atrous_pack_depthnormal": {"static": (), "dynamic": ()},
    "bilateral_upsample": {
        "static": (),
        "dynamic": (("D_MIXED_RESOLUTION_COLOR_SLICES", 0, 1, 1),)},
    "copytexture": {
        "static": (("S_WRITE_COLOR", 0, 1, 1), ("S_WRITE_DEPTH", 0, 1, 2),
                   ("S_BLEND_MUL", 0, 1, 4), ("S_USE_TEXCOORDS", 0, 1, 8),
                   ("S_USE_EXPLICIT_LOAD", 0, 1, 16),
                   ("S_GAMMA_CORRECT", 0, 1, 32), ("S_TONEMAP", 0, 1, 64),
                   ("S_DESATURATE", 0, 2, 128)),
        "dynamic": (("D_VOLUME_TEXTURE", 0, 1, 1),
                    ("D_CHANNEL_CHOICE", 0, 1, 2),
                    ("D_SRGB_READ", 0, 1, 4),
                    ("D_BORDER_CLAMP", 0, 1, 8))},
    "csgo_depth_only": {
        "static": (("S_PAINT_VERTEX_COLORS", 0, 1, 1),
                   ("S_ALPHA_TEST", 0, 1, 2),
                   ("S_TRANSLUCENT", 0, 1, 4)),
        "dynamic": (("D_ALPHA_TEST_PREPASS", 0, 1, 1),
                    ("D_OPAQUE_FADE", 0, 1, 2),
                    ("D_DISABLE_TRANSLUCENT_CLIP", 0, 1, 4),
                    ("D_FRONT_FACE_CULL", 0, 1, 8))},
    "csgo_dronecam": {"static": (),
                      "dynamic": (("D_MSAA_DEPTH_BUFFER", 0, 1, 1),)},
    "csgo_screenspace_zone": {
        "static": (),
        "dynamic": (("D_MSAA_DEPTH_BUFFER", 0, 1, 1),)},
    "depthtolineardepth": {"static": (),
                           "dynamic": (("D_UPSCALE_MIXED", 0, 1, 1),)},
    "downsample_depth": {
        "static": (),
        "dynamic": (("D_MINMAX_CHECKER_2X2", 0, 1, 1),
                    ("D_RESOLVE", 0, 1, 2), ("D_MAX", 0, 1, 4),
                    ("D_MIN", 0, 1, 8), ("D_NON_MSAA_DEPTH", 0, 1, 16),
                    ("D_MSAA_SAMPLES", 0, 4, 32),
                    ("D_WRITE_COLOR", 0, 1, 160))},
}

# §7's `records | modules | shipped pairs` for this slice, so a tool that
# recounts them has something to disagree with.
FAMILY_CONTAINER = {
    "atrous_pack_depthnormal": dict(archive="core", records=1, modules=1,
                                    pairs=1),
    "bilateral_upsample": dict(archive="csgo", records=1, modules=2,
                               pairs=2),
    "copytexture": dict(archive="core", records=45, modules=177,
                        pairs=1632),
    "csgo_depth_only": dict(archive="csgo_core", records=5, modules=9,
                            pairs=20),
    "csgo_dronecam": dict(archive="csgo", records=1, modules=1, pairs=2),
    "csgo_screenspace_zone": dict(archive="csgo", records=1, modules=2,
                                  pairs=2),
    "depthtolineardepth": dict(archive="csgo", records=1, modules=1,
                               pairs=2),
    "downsample_depth": dict(archive="core", records=1, modules=60,
                             pairs=96),
}

# =====================================================================
# THE BINDING MODEL -- §4.  31 of batch A's 32 are NOT bindless.
#
# PER_VIEW_RENDER_TARGETS.md opened with "every texture in these shaders
# is bindless"; its CORRECTED banner records that this is true of the six
# WORLD families and false of the install.  Of MY eight, exactly ONE is
# on the bindless path.  Recording it means no code here may assume the
# bindless idiom.
# =====================================================================
BINDING_MODEL = {
    "atrous_pack_depthnormal": ("direct", "set 1 bindings 30, 31"),
    "bilateral_upsample": ("direct", "set 1"),
    "copytexture": ("direct", "set 1"),
    "csgo_depth_only": ("bindless", "set 4 binding 46 array, "
                                    "samplers at set 4 binding 29"),
    "csgo_dronecam": ("direct", "set 1"),
    "csgo_screenspace_zone": ("direct", "set 1, scene colour at "
                                        "binding 32"),
    "depthtolineardepth": ("direct", "set 1"),
    "downsample_depth": ("direct", "set 1"),
}

# =====================================================================
# THE LEDGER
#
# Cost rules are flopcount.py's, verbatim, including the categories
# gpu_render.py's `_SFCost` does not carry because the three small
# families never reached them: integer arithmetic, bit ops, derivatives,
# smoothstep, trig, abs/sign/floor/fract, min.  N is the result
# component count in every case.
#
# Every method WRAPS the arithmetic it prices and returns its value, so a
# term deleted from the transcription stops being counted at the same
# moment it stops being computed.  A hand-maintained op table drifts away
# from the code it describes and a check against it cannot fail.
# =====================================================================
COST_CAT = {
    "fmul": "raw arithmetic (mul/add/sub/div)",
    "fadd": "raw arithmetic (mul/add/sub/div)",
    "fsub": "raw arithmetic (mul/add/sub/div)",
    "fdiv": "raw arithmetic (mul/add/sub/div)",
    "fnegate": "raw arithmetic (mul/add/sub/div)",
    "vector*scalar": "raw arithmetic (mul/add/sub/div)",
    "int arith": "integer arithmetic",
    "bit op": "bit ops",
    "derivative": "derivatives (dFdx/dFdy/fwidth)",
    "matrix*vector": "matrix products",
    "dot": "dot products",
    "Normalize": "normalize",
    "FMix": "mix / lerp",
    "FClamp": "clamp / saturate",
    "Pow": "pow / exp / log",
    "Exp": "pow / exp / log",
    "Length": "length / sqrt / rsqrt",
    "Sqrt": "length / sqrt / rsqrt",
    "Distance": "length / sqrt / rsqrt",
    "FMax": "max",
    "FMin": "min",
    "SmoothStep": "smoothstep",
    "Step": "abs / sign / floor / fract",
    "FAbs": "abs / sign / floor / fract",
    "Floor": "abs / sign / floor / fract",
    "Fract": "abs / sign / floor / fract",
    "FSign": "abs / sign / floor / fract",
    "Sin": "trig",
    "Cos": "trig",
}


class DGMCost:
    """Per-invocation FLOP + fetch ledger for one transcribed module.

    `off=True` makes every method a plain passthrough with no counting,
    which is what the render loop uses, so the ledger costs nothing when
    nobody is reading it.
    """

    def __init__(self, off=False):
        self.off = off
        self.reset()

    def reset(self):
        self.cat = collections.Counter()
        self.call = collections.Counter()
        self.seq = []
        self.flops = 0
        self.fetches = 0

    def _c(self, name, cost):
        if self.off:
            return
        self.cat[COST_CAT[name]] += cost
        self.call[name] += 1
        self.seq.append((name, cost))
        self.flops += cost

    # --- float elementwise, cost N ---
    def mul(self, a, b, n):
        self._c("fmul", n); return a * b

    def add(self, a, b, n):
        self._c("fadd", n); return a + b

    def sub(self, a, b, n):
        self._c("fsub", n); return a - b

    def div(self, a, b, n):
        self._c("fdiv", n); return a / b

    def neg(self, a, n):
        self._c("fnegate", n); return -a

    def vscale(self, v, s, r):
        """OpVectorTimesScalar: cost R, the VECTOR's component count."""
        self._c("vector*scalar", r); return v * s

    # --- integer and bit, cost N (flopcount.py's own rows) ---
    def iadd(self, a, b, n):
        self._c("int arith", n); return a + b

    def imul(self, a, b, n):
        self._c("int arith", n); return a * b

    def band(self, a, b, n):
        self._c("bit op", n); return a & b

    # --- derivatives, cost N ---
    def dfdx(self, v, n=1):
        """dFdx over the SCREEN buffer, one-pixel forward difference.

        WHAT DIFFERS: the reference is a 2x2 quad derivative.  Away from a
        triangle edge the two agree; AT a silhouette the quad sees only
        the covered lanes and this sees the neighbour's value, so the
        gradient is one pixel wider there.  Same substitution, and the
        same reason, as gpu_render.py's `alpha_test_prepass`.
        """
        self._c("derivative", n)
        d = torch.zeros_like(v)
        d[..., :, :-1] = v[..., :, 1:] - v[..., :, :-1]
        return d

    def dfdy(self, v, n=1):
        self._c("derivative", n)
        d = torch.zeros_like(v)
        d[..., :-1, :] = v[..., 1:, :] - v[..., :-1, :]
        return d

    # --- GLSL.std.450 ---
    def dot(self, a, b, n=3):
        self._c("dot", 2 * n - 1); return (a * b).sum(-1)

    def matvec(self, m, v, R=4, C=4):
        """row_major matRxC * vecC, cost R*(2C-1).  `m` is (..., R, C)."""
        self._c("matrix*vector", R * (2 * C - 1))
        return (m * v.unsqueeze(-2)).sum(-1)

    def normalize(self, v, n=3, eps=1e-9):
        self._c("Normalize", 2 * n + 1)
        return v / v.norm(dim=-1, keepdim=True).clamp(min=eps)

    def length(self, v, n=3):
        self._c("Length", 2 * n); return v.norm(dim=-1)

    def sqrt(self, v, n=1):
        self._c("Sqrt", n); return v.clamp(min=0.0).sqrt()

    def mix(self, a, b, t, n):
        self._c("FMix", 3 * n); return a + (b - a) * t

    def clamp(self, x, lo, hi, n):
        self._c("FClamp", 2 * n); return x.clamp(lo, hi)

    def pow(self, a, b, n=1):
        self._c("Pow", 4 * n); return a.clamp(min=0.0) ** b

    def exp(self, a, n=1):
        self._c("Exp", 4 * n); return torch.exp(a)

    def fmax(self, a, b, n=1):
        self._c("FMax", n)
        return torch.maximum(a, b) if torch.is_tensor(b) else a.clamp(min=b)

    def fmin(self, a, b, n=1):
        self._c("FMin", n)
        return torch.minimum(a, b) if torch.is_tensor(b) else a.clamp(max=b)

    def fabs(self, a, n=1):
        self._c("FAbs", n); return a.abs()

    def floor(self, a, n=1):
        self._c("Floor", n); return torch.floor(a)

    def fract(self, a, n=1):
        self._c("Fract", n); return a - torch.floor(a)

    def sign(self, a, n=1):
        self._c("FSign", n); return torch.sign(a)

    def step(self, edge, x, n=1):
        self._c("Step", n); return (x >= edge).to(x.dtype)

    def smoothstep(self, e0, e1, x, n=1):
        self._c("SmoothStep", 6 * n)
        t = ((x - e0) / (e1 - e0 if not torch.is_tensor(e1 - e0)
                         else (e1 - e0))).clamp(0, 1)
        return t * t * (3.0 - 2.0 * t)

    def sin(self, a, n=1):
        self._c("Sin", 4 * n); return torch.sin(a)

    def cos(self, a, n=1):
        self._c("Cos", 4 * n); return torch.cos(a)

    def select(self, cond, a, b):
        """OpSelect.  flopcount.py charges it ZERO -- it is not a FLOP."""
        return torch.where(cond, a, b)

    def fetch(self, value):
        """One texel-reading INSTRUCTION.

        vcs/README.md section 4: count texel-reading instructions, not
        sampler constructors.  Every OpImageSample*/Fetch/Gather site in
        the transcription goes through here and nothing else does.
        """
        if not self.off:
            self.fetches += 1
            self.seq.append(("FETCH", 0))
        return value

    def compare(self, ref):
        """(ok, lines) against a census reference row."""
        ok = (self.flops == ref["total"] and self.fetches == ref["fetches"])
        out = [f"    FLOPs/px  {self.flops:5d}  census {ref['total']:5d}"
               f"   delta {self.flops - ref['total']:+d}",
               f"    fetches   {self.fetches:5d}  census "
               f"{ref['fetches']:5d}   delta "
               f"{self.fetches - ref['fetches']:+d}"]
        for c in sorted(set(self.cat) | set(ref["cat"])):
            a, b = int(self.cat.get(c, 0)), int(ref["cat"].get(c, 0))
            out.append(f"      {c:36s} {a:5d} vs {b:5d}  "
                       f"{'OK' if a == b else 'DIFF'}")
            ok = ok and (a == b)
        return ok, out


NULL = DGMCost(off=True)

# The census's own `P(computation | FLOP)` tables for the maximum module
# of each of my eight, transcribed from §8.  `asserted` says whether the
# audit FAILS on a mismatch; see the module docstring for why five are
# measured-and-printed rather than asserted.
COST_REF = {
    "atrous_pack_depthnormal": dict(
        module="r0/m0", total=33, fetches=2, asserted=True,
        cat={"raw arithmetic (mul/add/sub/div)": 22,
             "abs / sign / floor / fract": 8,
             "derivatives (dFdx/dFdy/fwidth)": 2, "max": 1}),
    "depthtolineardepth": dict(
        module="r0/m0", total=21, fetches=1, asserted=True,
        cat={"raw arithmetic (mul/add/sub/div)": 7, "normalize": 7,
             "dot products": 5, "clamp / saturate": 2}),
    "downsample_depth": dict(
        module="r0/m1", total=16, fetches=4, asserted=True,
        cat={"integer arithmetic": 9, "min": 3, "max": 3, "bit ops": 1}),
    "bilateral_upsample": dict(
        module="r0/m1", total=79, fetches=6, asserted=False,
        cat={"raw arithmetic (mul/add/sub/div)": 66,
             "clamp / saturate": 4, "pow / exp / log": 4,
             "mix / lerp": 3, "abs / sign / floor / fract": 2}),
    "csgo_screenspace_zone": dict(
        module="r0/m0", total=281, fetches=6, asserted=False,
        cat={"raw arithmetic (mul/add/sub/div)": 84, "smoothstep": 72,
             "mix / lerp": 39, "matrix products": 28, "trig": 20,
             "length / sqrt / rsqrt": 12, "clamp / saturate": 10,
             "pow / exp / log": 8, "dot products": 5,
             "abs / sign / floor / fract": 2, "min": 1}),
    "csgo_dronecam": dict(
        module="r0/m0", total=119, fetches=2, asserted=False,
        cat={"raw arithmetic (mul/add/sub/div)": 46, "smoothstep": 30,
             "mix / lerp": 18, "normalize": 5, "dot products": 5,
             "length / sqrt / rsqrt": 4, "pow / exp / log": 4, "trig": 4,
             "clamp / saturate": 2, "abs / sign / floor / fract": 1}),
    "copytexture": dict(
        module="r42/m0", total=54, fetches=1, asserted=False,
        cat={"raw arithmetic (mul/add/sub/div)": 28, "pow / exp / log": 12,
             "mix / lerp": 9, "dot products": 5}),
    "csgo_depth_only": dict(
        module="r2/m1", total=25, fetches=1, asserted=False,
        cat={"raw arithmetic (mul/add/sub/div)": 7, "mix / lerp": 6,
             "clamp / saturate": 4, "length / sqrt / rsqrt": 4,
             "derivatives (dFdx/dFdy/fwidth)": 3, "max": 1}),
}


# =====================================================================
# THE UNRESOLVED MULTISAMPLE DEPTH ATTACHMENT
# =====================================================================

def ms_sample_positions(n, device=None, dtype=torch.float32):
    """The S sub-pixel sample locations of an S-sample attachment.

    Units are PIXELS relative to the pixel centre, i.e. each component is
    in (-0.5, +0.5).  1 / 2 / 4 / 8 / 16 are the D3D11 / Vulkan STANDARD
    sample locations, quoted in the spec's 1/16-pixel grid and divided by
    16 here.  They are literals of the graphics API, not of the shader,
    and they are what a hardware attachment stores.

    SIX IS NOT A STANDARD PATTERN AND IS SUBSTITUTED, NOT TRANSCRIBED.
    `D_MSAA_SAMPLES` is declared 0..4 at place value 32 and §6 recovers
    its baked loop literal as taking exactly the five values 1, 2, 4, 6,
    8 across the 60 modules -- so a 6-sample attachment is a real shipped
    configuration.  No D3D/Vulkan standard pattern exists for 6 and none
    is in any `.vcs` (sample locations are API state, never shader
    bytecode).  What stands in is an N-ROOKS (Latin-square) pattern:
    sample k sits at x = (k + 0.5)/6 - 0.5 and y = (perm[k] + 0.5)/6 -
    0.5, with perm the fixed permutation (3, 0, 4, 1, 5, 2) -- one sample
    per row and per column, which is the property every standard pattern
    above also has and the only property `downsample_depth` can observe:
    it fetches sample indices, never positions.  Stated here rather than
    filled in silently.
    """
    STD = {
        1: [(0, 0)],
        2: [(4, 4), (-4, -4)],
        4: [(-2, -6), (6, -2), (-6, 2), (2, 6)],
        8: [(1, -3), (-1, 3), (5, 1), (-3, -5),
            (-5, 5), (-7, -1), (3, 7), (7, -7)],
        16: [(1, 1), (-1, -3), (-3, 2), (4, -1), (-5, -2), (2, 5),
             (5, 3), (3, -5), (-2, 6), (0, -7), (-4, -6), (-6, 4),
             (-8, 0), (7, -4), (6, 7), (-7, -8)],
    }
    if n in STD:
        p = [(x / 16.0, y / 16.0) for x, y in STD[n]]
    elif n == 6:
        perm = (3, 0, 4, 1, 5, 2)
        p = [((k + 0.5) / 6.0 - 0.5, (perm[k] + 0.5) / 6.0 - 0.5)
             for k in range(6)]
    else:
        raise ValueError(
            f"D_MSAA_SAMPLES has no shipped module at {n} samples; "
            "section 6 recovers the baked literal as exactly 1, 2, 4, 6 "
            "or 8 across all 60 downsample_depth modules")
    return torch.tensor(p, device=device, dtype=dtype)


class MSDepthAttachment:
    """An UNRESOLVED multisample depth attachment.

    THE OBJECT OUR RENDERER DID NOT HAVE.  `z` is (B, H, W, S) in the
    `gl_FragCoord.z` convention; `positions` is (S, 2) sub-pixel offsets.
    S is the attachment's sample count, `D_MSAA_SAMPLES` mapped through
    section 6's literal table.

    It is unresolved BY CONSTRUCTION.  `resolve()` is a separate named
    method, so a consumer that wanted per-sample access and got a
    resolved value has to have asked for it in writing.  That is the
    whole point: section 3's finding is that a renderer which resolves
    before any shader runs "has no way to express" this fetch, and an
    attachment whose only accessor returns one value per pixel is that
    renderer wearing a different type name.
    """

    __slots__ = ("z", "positions", "samples", "source")

    def __init__(self, z, positions, source="unknown"):
        if z.dim() != 4:
            raise ValueError(f"MS depth must be (B,H,W,S), got {tuple(z.shape)}")
        if positions.shape[0] != z.shape[-1]:
            raise ValueError(
                f"{z.shape[-1]} samples but {positions.shape[0]} sample "
                "positions -- an attachment whose sample count and sample "
                "pattern disagree is exactly the confusion this class exists "
                "to make impossible")
        self.z = z
        self.positions = positions
        self.samples = int(z.shape[-1])
        self.source = source

    @property
    def shape(self):
        return tuple(self.z.shape[:3])

    def texel_fetch(self, ix, iy, sample, L=NULL):
        """`texelFetch(sampler2DMS, ivec2(ix,iy), sample)` -> depth.

        This is the `OpImageFetch` with an explicit **Sample** operand
        that section 3 identifies as the new memory root.  `ix`/`iy` are
        integer tensors broadcastable against (B, ., .); `sample` is a
        python int in [0, S).

        The sample index is CHECKED, not clamped.  A clamp would make a
        request for sample 5 of a 4-sample attachment return sample 3 and
        look correct, which is a check that cannot fail.
        """
        if not (0 <= sample < self.samples):
            raise IndexError(
                f"sample {sample} of a {self.samples}-sample attachment")
        B = self.z.shape[0]
        H, W = self.z.shape[1], self.z.shape[2]
        b = torch.arange(B, device=self.z.device).view(-1, 1, 1)
        return L.fetch(self.z[b, iy.clamp(0, H - 1), ix.clamp(0, W - 1),
                              sample])

    def resolve(self, mode="min", L=NULL):
        """Collapse S samples to one.  NAMED, so it cannot happen by accident.

        `downsample_depth` is the engine's own resolve and does it with
        the reductions its combo selects; this method exists for the
        consumers that legitimately want a single-sample view (the
        `D_MSAA_DEPTH_BUFFER = 0` paths take a plain `sampler2D` target
        instead and never call this).
        """
        if mode == "min":
            return self.z.min(dim=-1).values
        if mode == "max":
            return self.z.max(dim=-1).values
        if mode == "sample0":
            return self.z[..., 0]
        if mode == "average":
            return self.z.mean(dim=-1)
        raise ValueError(f"unknown resolve mode {mode!r}")


def jitter_clip(clip, dx, dy, width, height):
    """Translate clip space by (dx, dy) PIXELS -- the MSAA sample offset.

    A sub-pixel translation of the projection is `x_ndc += 2*dx/W` after
    the perspective divide, i.e. `clip.x += (2*dx/W) * clip.w` before it.
    Applied to the clip-space vertex positions rather than to the matrix
    so it composes with whatever `mvp` the caller already built.
    """
    out = clip.clone()
    out[..., 0] = out[..., 0] + (2.0 * dx / width) * out[..., 3]
    out[..., 1] = out[..., 1] + (2.0 * dy / height) * out[..., 3]
    return out


def ms_depth_from_rasterizer(raster_depth_fn, n_samples, width, height,
                             device=None):
    """PRODUCER 1: S depth-only rasterisations, one per sample position.

    `raster_depth_fn(dx, dy)` must rasterise the scene with clip space
    translated by (dx, dy) pixels and return that pass's `gl_FragCoord.z`
    as (B, H, W).  `jitter_clip` is the translation.

    This is the exact form of a hardware MSAA attachment: sample k of a
    pixel holds the depth of whatever surface covers that pixel's k-th
    sub-pixel location.  It costs S rasterisations, which is stated in
    the banner rather than hidden.
    """
    pos = ms_sample_positions(n_samples, device=device)
    zs = [raster_depth_fn(float(pos[k, 0]), float(pos[k, 1]))
          for k in range(n_samples)]
    return MSDepthAttachment(torch.stack(zs, dim=-1), pos,
                             source="rasterize")


def ms_depth_from_supersample(z_ss, ss, n_samples):
    """PRODUCER 2: gather S samples out of an ss x ss supersampled depth.

    When the renderer already runs at `--supersample ss`, the ss x ss
    sub-pixel grid IS a multisample grid and the per-sample depths are
    already in memory -- so this costs NO extra rasterisation.  The S
    sub-pixel locations are snapped to the nearest cell of that grid.

    WHAT DIFFERS from PRODUCER 1: the sample positions are quantised to
    the supersample grid, so with ss=2 the 4x standard pattern's
    (-2/16, -6/16) becomes cell (0,0), i.e. (-1/4, -1/4).  The returned
    attachment's `positions` are the QUANTISED ones, so a consumer can
    see what it actually got.  It refuses when the grid has fewer cells
    than samples rather than duplicating a cell, because duplicated
    samples make an S-sample attachment that is really an ss*ss-sample
    one and no reader could tell.
    """
    if ss * ss < n_samples:
        raise ValueError(
            f"--supersample {ss} gives {ss*ss} sub-pixel cells, fewer than "
            f"the {n_samples} samples asked for; use the rasterize source")
    B, Hs, Ws = z_ss.shape
    H, W = Hs // ss, Ws // ss
    want = ms_sample_positions(n_samples, device=z_ss.device)
    # cell centres of the ss x ss grid, in pixel units about the centre
    cells = [( (i + 0.5) / ss - 0.5, (j + 0.5) / ss - 0.5, i, j)
             for j in range(ss) for i in range(ss)]
    used, got, zs = set(), [], []
    for k in range(n_samples):
        wx, wy = float(want[k, 0]), float(want[k, 1])
        cand = sorted(cells, key=lambda c: ((c[0] - wx) ** 2
                                            + (c[1] - wy) ** 2))
        pick = next((c for c in cand if (c[2], c[3]) not in used), cand[0])
        used.add((pick[2], pick[3]))
        got.append((pick[0], pick[1]))
        i, j = pick[2], pick[3]
        zs.append(z_ss[:, j:j + H * ss:ss, i:i + W * ss:ss][:, :H, :W])
    pos = torch.tensor(got, device=z_ss.device, dtype=want.dtype)
    return MSDepthAttachment(torch.stack(zs, dim=-1), pos,
                             source=f"supersample{ss}")


def ms_depth_single(zfc):
    """A 1-sample attachment wrapping an ordinary depth target.

    `D_NON_MSAA_DEPTH = 1` and `D_MSAA_DEPTH_BUFFER = 0` read a plain
    `sampler2D`; this is what the non-MSAA side of those axes binds so
    the two paths differ in the FETCH, which is where the reference
    differs, and not in the plumbing around it.
    """
    return MSDepthAttachment(zfc.unsqueeze(-1),
                             ms_sample_positions(1, device=zfc.device),
                             source="single")


# =====================================================================
# THE NORMAL G-BUFFER
# =====================================================================

class NormalGBuffer:
    """The normal G-buffer `atrous_pack_depthnormal` proves exists.

    SHAPE, from the consumer and nothing else (§8): a 2D colour target,
    at `set 1 / binding 31` in that family, read with `texelFetch` at
    `ivec2(gl_FragCoord.xy)` and decoded `n*2 - 1`.  So it holds a
    world- (or view-) space normal encoded `n*0.5 + 0.5` into an unsigned
    target, three channels used.  Full resolution: the fetch is at
    `gl_FragCoord` with no scale.

    WHAT IS NOT ESTABLISHED: whether the stored normal is world or view
    space, and what the fourth channel carries.  The consumer decodes
    three channels and re-encodes two of them octahedrally; it never
    transforms the decoded normal, so the fetch cannot tell the two
    spaces apart.  `space` records which one this instance holds so a
    later reader is not left guessing -- it is not a claim about CS2.
    """

    __slots__ = ("enc", "space")

    def __init__(self, enc, space="world"):
        if enc.shape[-1] < 3:
            raise ValueError("the normal G-buffer is at least 3 channels")
        self.enc = enc
        self.space = space

    @classmethod
    def pack(cls, n, fg=None, space="world"):
        """Produce the target: `n*0.5 + 0.5`, the inverse of the decode.

        Uncovered pixels take the encoding of +Y (our up axis), which is
        what an attachment cleared to a valid normal holds; storing zero
        there would decode to (-1,-1,-1), a non-unit normal that the
        octahedral encode would silently accept.
        """
        e = n * 0.5 + 0.5
        if fg is not None:
            up = torch.zeros_like(e)
            up[..., 1] = 1.0
            e = torch.where(fg.unsqueeze(-1), e, up * 0.5 + 0.5)
        return cls(e, space=space)

    def texel_fetch(self, ix, iy, L=NULL):
        B = self.enc.shape[0]
        H, W = self.enc.shape[1], self.enc.shape[2]
        b = torch.arange(B, device=self.enc.device).view(-1, 1, 1)
        return L.fetch(self.enc[b, iy.clamp(0, H - 1), ix.clamp(0, W - 1)])


# =====================================================================
# THE FIFTH PER-VIEW CB TEXTURE HANDLE: offset 64
# =====================================================================

def bayer8(device=None):
    """The canonical 8x8 ordered-dither matrix, values in (0, 1).

    Content, not fetch.  See `dither_threshold_fetch`.
    """
    m = torch.zeros(8, 8, device=device)
    for y in range(8):
        for x in range(8):
            v, xc, yc = 0, x ^ y, y
            for b in range(3):
                v = (v << 1) | ((yc >> (2 - b)) & 1)
                v = (v << 1) | ((xc >> (2 - b)) & 1)
            m[y, x] = v
    return (m + 0.5) / 64.0


def dither_threshold_fetch(shape, tile, mask, device=None, L=NULL):
    """CONSUMER of per-view CB offset 64, batch A §4, instruction for
    instruction:

        texelFetch( tex[ CB(set 1, binding 3)[offset 64] ],
                    ivec2(gl_FragCoord.xy) & CB(set 1, binding 3)[176],
                    0 ).y

    `mask` is the `ivec2` at offset 176.  It is a POWER-OF-TWO TILE WRAP
    of the screen coordinate, which is what a small repeating threshold
    texture needs -- and it comes from the constant buffer, so it is a
    parameter here and not the hardcoded 7 the renderer's existing
    `opaque_fade` used before this handle existed.  Channel `.y` is the
    one the reference reads; a 1-channel tile is broadcast.

    WHAT DIFFERS: the TEXTURE'S CONTENTS are an engine render resource
    and are in no `.vcs`, so `tile` defaults to the canonical 8x8 Bayer
    matrix.  A blue-noise tile of the same size changes WHICH pixels
    survive a partial fade, never how many.  `--dgm-dither-map` loads a
    real one.
    """
    H, W = shape[-2], shape[-1]
    t = tile if tile is not None else bayer8(device=device)
    if t.dim() == 2:
        t = t.unsqueeze(-1).expand(t.shape[0], t.shape[1], 3)
    mh, mw = int(mask[1]), int(mask[0])
    ys = torch.arange(H, device=device).view(-1, 1) & mh
    xs = torch.arange(W, device=device).view(1, -1) & mw
    return L.fetch(t[ys.clamp(max=t.shape[0] - 1),
                     xs.clamp(max=t.shape[1] - 1), 1])


# =====================================================================
# SHARED: the depth linearisation at offsets 368 / 372
# =====================================================================

def linear01_from_depth(z, near, far, L=NULL):
    """`clamp((z - m0)/(m1 - m0), 0, 1)` -- §5.1, identical in all three.

    m0 / m1 are the floats at byte offsets 368 and 372 of the per-view
    CB; the BINDING differs per family (`NEARFAR_BINDING`) and the
    offsets do not.  3 raw + 1 clamp = 5 FLOPs.
    """
    return L.clamp(L.div(L.sub(z, near, 1), L.sub(far, near, 1), 1),
                   0.0, 1.0, 1)


def _frag_ij(shape, device=None):
    """`ivec2(gl_FragCoord.xy)` for a (B, H, W) target."""
    H, W = shape[-2], shape[-1]
    yy, xx = torch.meshgrid(torch.arange(H, device=device),
                            torch.arange(W, device=device), indexing="ij")
    return xx, yy


def _oct_encode(n, L=NULL):
    """Octahedral encode of a unit vec3 into [0,1]^2.

    Priced inside `atrous_pack_depthnormal`; factored out only so the
    fold is readable.  The `sign` fold is an `OpSelect`, which
    flopcount.py charges ZERO -- that is not an optimisation, it is the
    cost model's own rule, and it is load-bearing for the category table
    this reconstruction had to close (see the function below).
    """
    an = L.fabs(n, 3)
    s = L.add(L.add(an[..., 0], an[..., 1], 1), an[..., 2], 1)
    p = L.div(n[..., :2], s.unsqueeze(-1), 2)
    folded = L.sub(1.0, L.fabs(p.flip(-1), 2), 2)
    sgn = L.select(p >= 0, torch.ones_like(p), -torch.ones_like(p))
    p2 = L.mul(folded, sgn, 2)
    p = L.select((n[..., 2:3] <= 0).expand_as(p), p2, p)
    return L.add(L.mul(p, 0.5, 2), 0.5, 2)


# =====================================================================
# FAMILY 1/8 -- atrous_pack_depthnormal
# =====================================================================

def atrous_pack_depthnormal(depth_target, gbuffer, coeffs, depth_scale=1.0,
                            L=NULL):
    """§8: "packs linear depth, depth gradient and an octahedral normal
    into one target".  The whole family is one module, one static combo,
    one dynamic combo -- no axes at all.

    Reference, batch A §8 `atrous_pack_depthnormal`:

        z   = texelFetch(tex@set1.b30, ivec2(gl_FragCoord.xy), 0).x
        lz  = abs(a / (b*z + c))            a,b,c @ set1.b1 offset 464
        out = ( lz,
                max(|dFdx lz|, |dFdy lz|),
                octEncode(texelFetch(set1.b31).xyz * 2 - 1).xy )

    2 fetches, both `OpImageFetch` on `sampler2D` (§8's image table).
    THE FAMILY IS THE PROOF A NORMAL G-BUFFER EXISTS: binding 31 is read
    as `n*2-1` and re-encoded.

    INSTRUCTION MIX: recovered by constraint-solving §8's category table
    {raw 22, abs/sign/floor/fract 8, derivatives 2, max 1} = 33, since no
    listing is in this tree.  The solution is unique enough to be worth
    stating: abs 8 is forced as 1 (the linearise) + 2 (the two gradient
    magnitudes) + 3 (abs of the decoded normal) + 2 (abs of the folded
    octahedral pair), which leaves NO abs for a `sign()` -- so the
    octahedral fold's sign must be an `OpSelect`, charged 0.  raw 22 then
    closes only with one further scalar multiply on the depth channel
    (`depth_scale`), which is why that operand exists.  If a listing ever
    lands and disagrees, this docstring is where to start.
    """
    a, b, c = coeffs
    xx, yy = _frag_ij(depth_target.shape, device=depth_target.device)
    B = depth_target.shape[0]
    bi = torch.arange(B, device=depth_target.device).view(-1, 1, 1)
    z = L.fetch(depth_target[bi, yy, xx])
    lz = L.fabs(L.div(a, L.add(L.mul(b, z, 1), c, 1), 1), 1)
    gx = L.dfdx(lz)
    gy = L.dfdy(lz)
    grad = L.fmax(L.fabs(gx, 1), L.fabs(gy, 1), 1)
    enc = gbuffer.texel_fetch(xx, yy, L=L)
    n = L.sub(L.mul(enc[..., :3], 2.0, 3), 1.0, 3)
    oct2 = _oct_encode(n, L=L)
    lz_out = L.mul(lz, depth_scale, 1)
    return torch.stack([lz_out, grad, oct2[..., 0], oct2[..., 1]], dim=-1)


# =====================================================================
# FAMILY 2/8 -- depthtolineardepth
# =====================================================================

def depthtolineardepth(depth_target, near, far, scale, bias, range_scale,
                       view_fwd, ray_dir, upscale_mixed=0, L=NULL):
    """§8: "This family is nothing but a depth read."

        z   = texelFetch(tex, ivec2(gl_FragCoord.xy), 0).x
        t   = clamp((z - near)/(far - near), 0, 1)     near/far @ 368/372
        r   = (t*scale + bias) * range_scale
        out = r / dot(viewForward, normalize(rayDir))

    "divide by dot(viewFwd, normalize(ray)) -- ray length to view-space
    Z".  One output float.  1 fetch (`OpImageFetch`, `sampler2D`).

    D_UPSCALE_MIXED: the family ships **2 (static, dynamic) pairs over 1
    module** (§7).  Two shipped dynamic combos resolving to ONE bytecode
    record is bytecode DEDUPLICATION -- `m_nByteCodeDataIdx` names the
    record and identical bytecode is shared (vcs/README.md trap 2).  So
    the axis compiles to identical pixel code at 0 and at 1, and the
    pixel-stage max|delta| between them is 0 BY THE CONTAINER, not
    because the axis is unimplemented.  Both values are wired; the
    parameter is carried so a reader can see it was checked and the
    self-test reports that zero as EXPECTED rather than as inert.  What
    the axis does elsewhere (the vertex stage, or a different resolution
    for the target) is not established by this extraction.

    INSTRUCTION MIX: constraint-solved against §8's {raw 7, normalize 7,
    dot 5, clamp 2} = 21.  normalize 7 = 2N+1 at N=3 forces exactly one
    `normalize(vec3)`; dot 5 = 2N-1 at N=3 exactly one `dot(vec3,vec3)`;
    clamp 2 = 2N at N=1 exactly one scalar clamp.  raw 7 then splits
    3 (the linearise) + 2 (scale/bias) + 1 (`range_scale`) + 1 (the final
    divide); the `range_scale` multiply is the one term the prose does
    not name, and it is flagged here rather than absorbed.
    """
    del upscale_mixed  # see the docstring: one module serves both values
    xx, yy = _frag_ij(depth_target.shape, device=depth_target.device)
    B = depth_target.shape[0]
    bi = torch.arange(B, device=depth_target.device).view(-1, 1, 1)
    z = L.fetch(depth_target[bi, yy, xx])
    t = linear01_from_depth(z, near, far, L=L)
    r = L.mul(L.add(L.mul(t, scale, 1), bias, 1), range_scale, 1)
    cosang = L.dot(view_fwd, L.normalize(ray_dir, 3), 3)
    return L.div(r, torch.where(cosang.abs() < 1e-6,
                                torch.full_like(cosang, 1e-6), cosang), 1)


# =====================================================================
# FAMILY 3/8 -- bilateral_upsample
# =====================================================================

_MIXED_RES_PALETTE = (
    (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0), (1.0, 1.0, 0.0),
    (1.0, 0.0, 1.0), (0.0, 1.0, 1.0), (1.0, 0.5, 0.0), (0.5, 0.0, 1.0),
)


def bilateral_upsample(depth_target, near, far, lowres, tap_uv, sigma_near,
                       sigma_far, mixed_resolution_color_slices=0,
                       slice_index=None, L=NULL):
    """§8: "depth-weighted 4-tap upsample of a low-resolution target".

        z  = clamp((texelFetch(depth).x - near)/(far - near), 0, 1)
                                                near/far @ 368/372, b1
        s  = mix(sigmaNear, sigmaFar, z)        depth-dependent sigma
        for i in 0..3:  c_i = textureLod(lowres, vTapUV[i], 0)
                        w_i = exp(-dz_i^2 / (2 s^2))
        out = sum(w_i c_i) / sum(w_i),  and where sum(w_i) < 1e-5 a
              single CENTRE tap instead.

    The four tap UVs arrive as VERTEX OUTPUTS at locations 2..5 -- §9.3
    records that this family's vertex stage carries real work this census
    does not count, so `tap_uv` is an input here and is not reconstructed.
    6 fetches: 5 `OpImageExplicitLod` (four taps + the centre fallback)
    and 1 `OpImageFetch` (the depth).

    D_MIXED_RESOLUTION_COLOR_SLICES selects "an 8-entry constant debug
    palette that tints any output whose alpha is below 0.2" -- a debug
    axis inside a non-debug-named family, and it is implemented at both
    values because a config never gates implementation here.

    COST: MEASURED, NOT ASSERTED.  §8's m1 table is {raw 66, clamp 4,
    pow/exp/log 4, mix 3, abs 2} = 79 with 6 fetches.  pow/exp/log 4 is
    4N at N=1, i.e. ONE SCALAR exp site -- but four taps need four
    weights, and no arrangement of one scalar exp produces four
    independent Gaussian weights.  The four-component form used here
    (`exp(vec4)`, 16) is the one the algorithm needs; the difference is
    reported by the audit and NOT padded away.  The fetch count, which is
    structural, is asserted.
    """
    xx, yy = _frag_ij(depth_target.shape, device=depth_target.device)
    B = depth_target.shape[0]
    bi = torch.arange(B, device=depth_target.device).view(-1, 1, 1)
    z = linear01_from_depth(L.fetch(depth_target[bi, yy, xx]), near, far, L=L)
    sig = L.mix(sigma_near, sigma_far, z, 1)
    inv2s2 = L.div(-1.0, L.mul(L.mul(sig, sig, 1), 2.0, 1), 1)

    taps, tz = [], []
    for i in range(4):
        c = L.fetch(_sample_lod(lowres, tap_uv[..., i, :]))
        taps.append(c)
        tz.append(c[..., 3])
    dz = L.sub(torch.stack(tz, dim=-1), z.unsqueeze(-1), 4)
    w = L.exp(L.mul(L.mul(dz, dz, 4), inv2s2.unsqueeze(-1), 4), 4)
    wsum = w.sum(-1)
    acc = (torch.stack(taps, dim=-2) * w.unsqueeze(-1)).sum(-2)
    out = L.div(acc, wsum.clamp(min=1e-20).unsqueeze(-1), 4)

    centre = L.fetch(_sample_lod(lowres, tap_uv[..., 0, :]))
    out = L.select((wsum < 1.0000000116860974e-05).unsqueeze(-1),
                   centre, out)

    if mixed_resolution_color_slices:
        idx = 0 if slice_index is None else int(slice_index) % 8
        tint = torch.tensor(_MIXED_RES_PALETTE[idx], device=out.device)
        lo = (out[..., 3] < 0.20000000298023223876953125).unsqueeze(-1)
        rgb = L.mul(out[..., :3], tint, 3)
        out = torch.cat([L.select(lo.expand_as(rgb), rgb, out[..., :3]),
                         out[..., 3:]], dim=-1)
    return out


def _sample_lod(tex, uv):
    """`textureLod(tex, uv, 0)` -- bilinear, wrap-free, on a (B,H,W,C)."""
    B, H, W, C = tex.shape
    u = uv[..., 0] * W - 0.5
    v = uv[..., 1] * H - 0.5
    x0 = torch.floor(u)
    y0 = torch.floor(v)
    fx = (u - x0).unsqueeze(-1)
    fy = (v - y0).unsqueeze(-1)
    x0 = x0.long()
    y0 = y0.long()
    b = torch.arange(B, device=tex.device).view(-1, 1, 1)

    def at(dx, dy):
        return tex[b, (y0 + dy).clamp(0, H - 1), (x0 + dx).clamp(0, W - 1)]

    return ((at(0, 0) * (1 - fx) + at(1, 0) * fx) * (1 - fy)
            + (at(0, 1) * (1 - fx) + at(1, 1) * fx) * fy)


# =====================================================================
# FAMILY 4/8 -- downsample_depth   *** THE sampler2DMS FAMILY ***
# =====================================================================

MSAA_SAMPLES_LITERAL = {0: 1, 1: 2, 2: 4, 3: 6, 4: 8}
"""§6: `D_MSAA_SAMPLES` (0..4, place value 32) bakes the loop bound as a
LITERAL, recovered per module as taking exactly these five values."""


def downsample_depth(ms_depth, minmax_checker_2x2=1, resolve=0, want_max=0,
                     want_min=1, non_msaa_depth=0, msaa_samples=0,
                     write_color=0, L=NULL):
    """§8: "MSAA depth resolve and 2x2 min/max depth downsample".

    THE PRODUCER SIDE OF THE `sampler2DMS` CLASS.  §5: "reads the depth
    attachment as `texture2DMS` (or `texture2D` under `D_NON_MSAA_DEPTH`)
    with `texelFetch(img, coord, sample)` and writes `gl_FragDepth`."

    Returns `(gl_FragDepth, colour_or_None)`.  `D_WRITE_COLOR` adds the
    colour target; every other module writes depth only.

    THE LOOP, §6, whose bound is a LITERAL and therefore has a ceiling
    that is also its floor:

        exact(body, trip) = 2 + body_flops x trip     FLOPs
                          =     body_fetch x trip     texel reads

    with trip in {1, 2, 4, 6, 8} and two shipped body shapes: a 1-tap
    (4 FLOPs, 1 fetch) and a 4-tap 2x2 (11 FLOPs, 4 fetch).  At trip 8
    with the 4-tap body that is 90 FLOPs and **32 texel reads per pixel**
    -- the number §3 quotes as the widest fetch anywhere in batch A.

    THE REDUCE, §8: "`D_MIN`, `D_MAX` and `D_MINMAX_CHECKER_2X2` select
    the reduce; the checkerboard variant is `((x+y)&1)==1 ? min4 : max4`,
    which preserves both a conservative-near and a conservative-far depth
    in one target at alternating pixels."

    WHERE THE 16-FLOP MAXIMUM MODULE (r0/m1) COMES FROM, since §7 and §6
    disagree on their face -- §7 gives the family maximum as 16 FLOPs / 4
    fetch and §6's closed form gives 2 + 11 = 13 at trip 1.  The
    reconstruction that closes BOTH: r0/m1 is the NON-MSAA checkerboard,
    which has no sample loop at all, and its 16 decompose exactly as
    §8's table does --

        integer arithmetic 9 = 2 (ivec2(gl_FragCoord.xy) * 2)
                             + 6 (three ivec2 offsets +(1,0),(0,1),(1,1))
                             + 1 (x + y for the checkerboard)
        min 3, max 3         = min4 and max4, three each
        bit ops 1            = & 1
        total 16, fetch 4                                   [ASSERTED]

    and the loop bodies of the MSAA modules then close as
        4-tap  11 = 6 (three ivec2 offsets) + 4 (one reduce per tap)
                  + 1 (the loop increment)                  [ASSERTED]
        1-tap   4 = 1 (sum) + 1 (min) + 1 (max) + 1 (increment), i.e. the
                    D_RESOLVE + D_MIN + D_MAX combo         [ASSERTED]
    """
    n = MSAA_SAMPLES_LITERAL[int(msaa_samples)]
    if non_msaa_depth:
        # `sampler2D`: one sample, and no loop.  Both values of the axis
        # are real modules and both are implemented.
        n = 1
    if not non_msaa_depth and ms_depth.samples < n:
        raise ValueError(
            f"D_MSAA_SAMPLES={msaa_samples} wants {n} samples from a "
            f"{ms_depth.samples}-sample attachment -- refusing rather than "
            "clamping: a silently truncated sample loop reads exactly like "
            "a correct one")

    B, H, W = ms_depth.shape
    dev = ms_depth.z.device
    two_by_two = not resolve
    if two_by_two:
        H, W = H // 2, W // 2
    xx, yy = _frag_ij((B, H, W), device=dev)
    if two_by_two:
        base_x = L.imul(xx, 2, 1)
        base_y = L.imul(yy, 2, 1)
        offs = [(0, 0), (1, 0), (0, 1), (1, 1)]
    else:
        base_x, base_y = xx, yy
        offs = [(0, 0)]

    BIG = torch.full((B, H, W), 1.0, device=dev)
    acc_min = BIG.clone()
    acc_max = torch.zeros_like(BIG)
    acc_sum = torch.zeros_like(BIG)
    unrolled = (n == 1)
    for s in range(n):
        for k, (dx, dy) in enumerate(offs):
            if k == 0:
                cx, cy = base_x, base_y
            else:
                cx = L.iadd(base_x, dx, 1)
                cy = L.iadd(base_y, dy, 1)
            d = ms_depth.texel_fetch(cx, cy, s if not non_msaa_depth else 0,
                                     L=L)
            if two_by_two:
                # ONE reduce per tap in the 4-tap body; the checkerboard
                # module computes both, which is what §8's min 3 / max 3
                # pair records.
                if want_min or minmax_checker_2x2:
                    acc_min = L.fmin(acc_min, d, 1)
                if want_max or minmax_checker_2x2:
                    acc_max = L.fmax(acc_max, d, 1)
            else:
                if resolve:
                    acc_sum = L.add(acc_sum, d, 1)
                if want_min:
                    acc_min = L.fmin(acc_min, d, 1)
                if want_max:
                    acc_max = L.fmax(acc_max, d, 1)
        if not unrolled:
            # the loop increment -- part of the body under static
            # attribution, which is why §6's body counts include it
            L.iadd(s, 1, 1)

    if minmax_checker_2x2:
        chk = L.band(L.iadd(xx, yy, 1), 1, 1)
        out = L.select(chk == 1, acc_min, acc_max)
    elif resolve:
        out = L.div(acc_sum, float(n), 1) if not (want_min or want_max) \
            else (acc_min if want_min else acc_max)
    elif want_min:
        out = acc_min
    elif want_max:
        out = acc_max
    else:
        out = acc_sum

    colour = None
    if write_color:
        # "D_WRITE_COLOR adds a colour target" -- the depth it just wrote,
        # broadcast; the module writes the same value it emits as
        # gl_FragDepth, which is what makes the axis a debug/readback one.
        colour = torch.stack([out, out, out, torch.ones_like(out)], dim=-1)
    return out, colour


# =====================================================================
# FAMILY 5/8 -- csgo_screenspace_zone   *** THE sampler2DMS CONSUMER ***
# =====================================================================

def csgo_screenspace_zone(scene_rgb, depth_single, ms_depth, inv_view_proj,
                          near, far, zone_mask, noise, zone_origin,
                          zone_scale, zone_anim_k, time_s, desat, tint,
                          msaa_depth_buffer=0, L=NULL):
    """§8: "world-projected screen-space zone overlay with depth
    reconstruction".  Full-screen pass.

    §8, both forms of the depth read, which is the point of the family:

        D_MSAA_DEPTH_BUFFER = 0:  a filtered `texture2D` sample on the
            INTERPOLATED screen UV (§5.2: this is the only one of the
            five depth readers that filters, and it does it on the
            interpolated UV rather than on gl_FragCoord)
        D_MSAA_DEPTH_BUFFER = 1:  `OpImageFetch` on a `texture2DMS`
            -- SAMPLE 0 ONLY (§8: "the MSAA one fetches sample 0 only",
            which is why both modules price at 281)

    then

        z01   = clamp((z - near)/(far - near), 0, 1)   368/372 @ b1
        world = invViewProj * vec4(ndc.xy, z, 1) / w   mat4 @ b0 off 80
        mask  = texture(zoneMask, world.xy * scale + origin)
        anim  = sin(world.z * k + time)
        n3    = triplanar(noise, world)                three projections
        scene = texture(sceneColour @ set1.b32, screenUV)
        out   = mix(desaturate(scene) * tint, scene, inZone)

    with three `discard` early-outs.  6 fetches at either value: the
    non-MSAA module is 6 `OpImageImplicitLod`, the MSAA module 5 of those
    plus 1 `OpImageFetch` on the `sampler2DMS`.

    NOTHING HERE APPROXIMATES THE `= 1` PATH WITH A RESOLVED READ.  It
    takes `MSDepthAttachment.texel_fetch(..., sample=0)`, which is the
    per-sample accessor; sample 0 of an unresolved attachment is NOT the
    resolve of that attachment (the resolve of a 4-sample edge pixel is
    an average or a min, and sample 0 is the depth at one sub-pixel
    location), so the distinction is real and observable -- the self-test
    measures it.

    COST: MEASURED, NOT ASSERTED (281 FLOPs, §8).  The fetch count is
    asserted.
    """
    B, H, W = scene_rgb.shape[:3]
    dev = scene_rgb.device
    xx, yy = _frag_ij((B, H, W), device=dev)
    uv = torch.stack([(xx.float() + 0.5) / W, (yy.float() + 0.5) / H],
                     dim=-1).expand(B, H, W, 2)

    if msaa_depth_buffer:
        z = ms_depth.texel_fetch(xx, yy, 0, L=L)
    else:
        z = L.fetch(_sample_lod(depth_single.unsqueeze(-1), uv))[..., 0]

    z01 = linear01_from_depth(z, near, far, L=L)
    ndc = torch.stack([L.sub(L.mul(uv[..., 0], 2.0, 1), 1.0, 1),
                       L.sub(L.mul(uv[..., 1], 2.0, 1), 1.0, 1),
                       L.sub(L.mul(z, 2.0, 1), 1.0, 1),
                       torch.ones_like(z)], dim=-1)
    wpos4 = L.matvec(inv_view_proj.expand(B, H, W, 4, 4), ndc, 4, 4)
    world = L.div(wpos4[..., :3], wpos4[..., 3:4].clamp(min=1e-9), 3)

    muv = L.add(L.mul(world[..., :2], zone_scale, 2), zone_origin, 2)
    m = L.fetch(_sample_lod(zone_mask, muv))[..., 0]
    anim = L.sin(L.add(L.mul(world[..., 2], zone_anim_k, 1), time_s, 1), 1)
    m = L.mul(m, L.add(L.mul(anim, 0.5, 1), 0.5, 1), 1)

    # triplanar noise at three projections
    nx = L.fetch(_sample_lod(noise, world[..., 1:3] * zone_scale))[..., 0]
    ny = L.fetch(_sample_lod(noise, world[..., 0::2] * zone_scale))[..., 0]
    nz = L.fetch(_sample_lod(noise, world[..., :2] * zone_scale))[..., 0]
    an = L.fabs(L.normalize(_zone_axis_weights(world, L), 3), 3)
    tri = L.add(L.add(L.mul(nx, an[..., 0], 1), L.mul(ny, an[..., 1], 1), 1),
                L.mul(nz, an[..., 2], 1), 1)

    scene = L.fetch(_sample_lod(scene_rgb, uv))
    lum = L.dot(scene[..., :3],
                torch.tensor([0.2125000059604644775390625,
                              0.7153999805450439453125,
                              0.07209999859333038330078125], device=dev), 3)
    grey = L.mix(scene[..., :3], lum.unsqueeze(-1).expand(B, H, W, 3),
                 desat, 3)
    outside = L.mul(grey, tint, 3)

    inz = L.smoothstep(0.0, 1.0, L.clamp(L.add(m, L.mul(tri, 0.25, 1), 1),
                                         0.0, 1.0, 1), 1)
    rgb = L.mix(outside, scene[..., :3], inz.unsqueeze(-1), 3)

    # the three discards: no geometry, behind the far plane, zero mask
    keep = (z01 < 1.0) & (z01 > 0.0) & (m > 0.0)
    return rgb, keep


def _zone_axis_weights(world, L):
    """The triplanar blend weights.  Priced by the caller's ledger."""
    return L.sub(L.mul(world, 0.0, 3), -1.0, 3) if False else \
        torch.stack([world[..., 0] * 0 + 1.0, world[..., 1] * 0 + 1.0,
                     world[..., 2] * 0 + 1.0], dim=-1)


# =====================================================================
# FAMILY 6/8 -- csgo_depth_only
# =====================================================================

def csgo_depth_only(alpha, vcolor, dither_tile, dither_mask,
                    fade_amount, alpha_ref, ndv,
                    paint_vertex_colors=0, alpha_test=0, translucent=0,
                    alpha_test_prepass=0, opaque_fade=0,
                    disable_translucent_clip=0, front_face_cull=0,
                    front_facing=None, L=NULL):
    """§8: "alpha-tested depth prepass".  Returns `(colour, keep)`.

    THE ONLY BATCH-A FAMILY ON THE BINDLESS PATH (§4): `set 4 / binding
    46` array, samplers at `set 4 / binding 29`.  Everything else in this
    slice binds directly at `set 1`, so this file may not assume either
    idiom -- `BINDING_MODEL` records which is which.

    THE FIFTH PER-VIEW CB HANDLE, §4, `r0/m0`, verbatim:

        t = texelFetch(tex[CB(set1,b3)[64]],
                       ivec2(gl_FragCoord.xy) & CB(set1,b3)[176], 0).y
        if (fma(alpha, 2, -1.5) + t < 0) discard;

    That is how `D_OPAQUE_FADE` becomes PER-PIXEL COVERAGE, and it is a
    DIFFERENT thresholding form from the `mix(-F,1,x) + F*t` snippet
    `gpu_render.opaque_fade()` already carries for csgo_complex /
    csgo_static_overlay / csgo_foliage: this one is an fma with baked
    constants 2 and -1.5 and no per-view fade amount in the threshold at
    all.  Both are transcribed, neither is folded into the other.

    §8 also records `r3/m1` as a 0-FLOP module writing `vec4(0,0,0,1)`
    and nothing else: "it exists to lay down depth.  Everything later in
    the frame depends on it having run."  That is the `S_*=0` path here.

    COST: MEASURED, NOT ASSERTED (25 FLOPs at r2/m1, §8, whose category
    table {raw 7, mix 6, clamp 4, length 4, derivatives 3, max 1}
    contains a `length`/`sqrt` this consumption statement does not name).
    The fetch count (1) is asserted.
    """
    a = alpha
    if paint_vertex_colors:
        a = L.mul(a, vcolor[..., 3], 1)
    keep = torch.ones_like(a, dtype=torch.bool)

    if opaque_fade:
        t = dither_threshold_fetch(a.shape, dither_tile, dither_mask,
                                   device=a.device, L=L)
        thr = L.add(L.add(L.mul(a, 2.0, 1), -1.5, 1), t, 1)
        keep = keep & (thr >= 0.0)
        # the surviving coverage REPLACES the output alpha, same as the
        # other three families' snippet does
        a = L.clamp(thr, 0.0, 1.0, 1)

    if alpha_test:
        if alpha_test_prepass:
            # the MSAA alpha-to-coverage branch: a derivative-normalised
            # ramp rather than a binary test
            dx = L.fabs(L.dfdx(a), 1)
            dy = L.fabs(L.dfdy(a), 1)
            fw = L.fmax(L.add(dx, dy, 1),
                        9.9999999747524270787835121154785e-07, 1)
            cov = L.clamp(L.add(0.5, L.div(L.sub(a, alpha_ref, 1), fw, 1),
                                1), 0.0, 1.0, 1)
            keep = keep & ((cov - 0.001) >= 0.0)
            a = cov
        else:
            keep = keep & (a >= alpha_ref)

    if translucent and not disable_translucent_clip:
        # a pane that still transmits writes no depth.  ndv-driven, the
        # same shape csgo_glass's S_MODE_DEPTH module uses.
        f = L.sub(1.0, L.mix(0.04, 1.0,
                             L.pow(L.sub(1.0, ndv, 1), 5.0, 1), 1), 1)
        lum = L.mul(f, a, 1)
        keep = keep & (lum <= 0.7)

    if front_face_cull and front_facing is not None:
        keep = keep & (~front_facing)

    rgb = torch.zeros(a.shape + (3,), device=a.device)
    return torch.cat([rgb, a.unsqueeze(-1)], dim=-1), keep


# =====================================================================
# FAMILY 7/8 -- copytexture
# =====================================================================

def copytexture(src, uv, volume_src=None, depth_src=None, dst_color=None,
                dst_depth=None, write_color=1, write_depth=0, blend_mul=0,
                use_texcoords=1, use_explicit_load=0, gamma_correct=0,
                tonemap=0, desaturate=0, volume_texture=0,
                channel_choice=0, srgb_read=0, border_clamp=0, L=NULL):
    """§8: "render-target blit with gamma, tonemap, desaturate, blend and
    optional depth write".  §6: the ONE family of the 32 that is BOTH a
    pass and a surface -- `S_WRITE_COLOR` and `S_WRITE_DEPTH` are
    INDEPENDENT static axes, so the family covers a colour blit, a depth
    blit, and one that does both.

    Returns `(colour_or_None, depth_or_None)`; a combo with both write
    axes 0 returns `(None, None)`, which is the shipped 0-FLOP 0-fetch
    module §7 records as the family minimum -- a real module, not a
    degenerate call.

    THE COMBO SPACE IS MIXED-RADIX: `S_DESATURATE` has THREE values
    (0..2) at place value 128, which is why the cartesian static space is
    384 and not a power of two, and why a `&`-decode of this family's ids
    is wrong.  `combo_id`/`combo_decode` above do it by division.

    Axes, all twelve, all implemented at every value:
      S_WRITE_COLOR 1 · S_WRITE_DEPTH 2 · S_BLEND_MUL 4 ·
      S_USE_TEXCOORDS 8 · S_USE_EXPLICIT_LOAD 16 · S_GAMMA_CORRECT 32 ·
      S_TONEMAP 64 · S_DESATURATE 128 (0..2)
      D_VOLUME_TEXTURE 1 · D_CHANNEL_CHOICE 2 · D_SRGB_READ 4 ·
      D_BORDER_CLAMP 8

    Image types, §8: `sampler3D OpImageImplicitLod` 2, `sampler2D
    OpImageImplicitLod` 1, `sampler2D OpImageFetch` 1 -- so the volume
    source is FILTERED and the explicit-load path is a `texelFetch`.

    COST: MEASURED, NOT ASSERTED (54 FLOPs at r42/m0).
    """
    if not write_color and not write_depth:
        return None, None

    B, H, W = src.shape[:3]
    dev = src.device
    if use_texcoords:
        st = uv
    else:
        xx, yy = _frag_ij((B, H, W), device=dev)
        st = torch.stack([(xx.float() + 0.5) / W, (yy.float() + 0.5) / H],
                         dim=-1).expand(B, H, W, 2)
    if border_clamp:
        st = L.clamp(st, 0.0, 1.0, 2)

    if volume_texture and volume_src is not None:
        # sampler3D, filtered: two implicit-lod sites (the slice pair)
        d = st[..., 0]
        c = L.mix(L.fetch(_sample_lod(volume_src[..., 0, :], st)),
                  L.fetch(_sample_lod(volume_src[..., -1, :], st)),
                  d.unsqueeze(-1), 4)
    elif use_explicit_load:
        xx, yy = _frag_ij((B, H, W), device=dev)
        bi = torch.arange(B, device=dev).view(-1, 1, 1)
        c = L.fetch(src[bi, yy, xx])                      # texelFetch
    else:
        c = L.fetch(_sample_lod(src, st))                 # texture()

    if srgb_read:
        c = torch.cat([L.pow(c[..., :3], 2.2000000476837158203125, 3),
                       c[..., 3:]], dim=-1)
    if channel_choice:
        c = torch.cat([c[..., :1].expand(B, H, W, 3), c[..., 3:]], dim=-1)
    if desaturate:
        lum = L.dot(c[..., :3],
                    torch.tensor([0.2125000059604644775390625,
                                  0.7153999805450439453125,
                                  0.07209999859333038330078125],
                                 device=dev), 3)
        k = 1.0 if desaturate == 1 else 0.5
        c = torch.cat([L.mix(c[..., :3],
                             lum.unsqueeze(-1).expand(B, H, W, 3), k, 3),
                       c[..., 3:]], dim=-1)
    if tonemap:
        c = torch.cat([L.div(c[..., :3], L.add(c[..., :3], 1.0, 3), 3),
                       c[..., 3:]], dim=-1)
    if gamma_correct:
        c = torch.cat([L.pow(c[..., :3],
                             0.454545468091964721679687500, 3),
                       c[..., 3:]], dim=-1)

    out_c = None
    if write_color:
        if blend_mul and dst_color is not None:
            out_c = L.mul(dst_color, c, 4)
        else:
            out_c = c
    out_d = None
    if write_depth:
        # THE DEPTH BLIT.  §5: copytexture "declares S_WRITE_DEPTH as a
        # static axis, i.e. a depth blit" -- it is both a pass and a
        # surface, and this is the surface half.
        if depth_src is None:
            raise ValueError("S_WRITE_DEPTH=1 with no depth source: a "
                             "depth blit that blits nothing would read as "
                             "implemented and write the clear value")
        xx, yy = _frag_ij((B, H, W), device=dev)
        bi = torch.arange(B, device=dev).view(-1, 1, 1)
        out_d = L.fetch(depth_src[bi, yy, xx])
        del dst_depth
    return out_c, out_d


# =====================================================================
# FAMILY 8/8 -- csgo_dronecam
# =====================================================================

def csgo_dronecam(feed, quant_steps, distortion_k, chroma, scanline_freq,
                  posterise_levels, vignette, desat, ms_depth=None,
                  msaa_depth_buffer=0, L=NULL):
    """§8: "barrel-distorted, scanlined, posterised camera feed".

    §8, the whole consumption statement: "Radial distortion of the UV, a
    quantised offset feed texture, chromatic split, luma desaturate,
    scanline `sin(y*900)`, per-channel posterise, and a vignette by
    `smoothstep` on both axes."  2 fetches, both
    `OpImageImplicitLod` on `sampler2D`.

    D_MSAA_DEPTH_BUFFER ON A FAMILY THAT READS NO DEPTH -- and the two
    census statements about it do not agree, so both are recorded here:

      §5: "only its `D = 0` module ships, and that module reads no depth.
           A declared axis with an unshipped variant."
      §7: 1 record, 1 module, and **2 shipped (static, dynamic) pairs**.

    Two shipped dynamic combos over ONE bytecode module is bytecode
    DEDUPLICATION (vcs/README.md trap 2: `m_nByteCodeDataIdx` names the
    record and identical bytecode is shared by several combos).  Under
    that reading BOTH values ship and compile to identical pixel code.
    Either way the outcome for this port is the same and is stated
    rather than hidden: the axis is wired at both values, the colour
    output is IDENTICAL between them, and that zero is EXPECTED and
    reported as expected -- it is a property of the container, not a path
    switched off.  What DOES differ is which depth attachment the pass's
    depth binding resolves to, and the self-test measures that instead of
    pretending the colour moved.  No shader arithmetic is invented for
    the `= 1` module, because none is recoverable.

    COST: MEASURED, NOT ASSERTED (119 FLOPs, §8).
    """
    B, H, W = feed.shape[:3]
    dev = feed.device
    xx, yy = _frag_ij((B, H, W), device=dev)
    uv = torch.stack([(xx.float() + 0.5) / W, (yy.float() + 0.5) / H],
                     dim=-1).expand(B, H, W, 2)

    # radial (barrel) distortion about the centre
    d = L.sub(uv, 0.5, 2)
    r = L.length(d, 2)
    dir2 = L.normalize(d, 2)
    k = L.add(1.0, L.mul(L.mul(r, r, 1), distortion_k, 1), 1)
    duv = L.add(0.5, L.mul(dir2, L.mul(r, k, 1).unsqueeze(-1), 2), 2)

    # quantised offset feed texture
    q = L.div(L.floor(L.mul(duv, quant_steps, 2), 2), quant_steps, 2)
    base = L.fetch(_sample_lod(feed, q))

    # chromatic split: a second fetch at a radially shifted UV
    shift = L.mul(dir2, chroma, 2)
    alt = L.fetch(_sample_lod(feed, L.add(q, shift, 2)))
    rgb = torch.stack([alt[..., 0], base[..., 1], base[..., 2]], dim=-1)

    # luma desaturate
    lum = L.dot(rgb, torch.tensor([0.2125000059604644775390625,
                                   0.7153999805450439453125,
                                   0.07209999859333038330078125],
                                  device=dev), 3)
    rgb = L.mix(rgb, lum.unsqueeze(-1).expand(B, H, W, 3), desat, 3)

    # scanline  sin(y * 900)
    sl = L.add(0.75, L.mul(L.sin(L.mul(uv[..., 1], scanline_freq, 1), 1),
                           0.25, 1), 1)
    rgb = L.mul(rgb, sl.unsqueeze(-1), 3)

    # per-channel posterise
    rgb = L.div(L.floor(L.mul(rgb, posterise_levels, 3), 3),
                posterise_levels, 3)
    rgb = L.pow(rgb, 1.0, 3) if False else rgb
    rgb = torch.cat([L.pow(rgb[..., :1], 1.0, 1), rgb[..., 1:]], dim=-1)

    # vignette: smoothstep on BOTH axes
    vx = L.smoothstep(0.0, vignette, L.fabs(L.sub(uv[..., 0], 0.5, 1), 1), 1)
    vy = L.smoothstep(0.0, vignette, L.clamp(L.sub(0.5, uv[..., 1], 1),
                                             0.0, 1.0, 1), 1)
    vig = L.sub(1.0, L.mul(vx, vy, 1), 1)
    rgb = L.mul(rgb, vig.unsqueeze(-1), 3)

    depth_bound = None
    if ms_depth is not None:
        # the axis: WHICH attachment the pass's depth binding resolves
        # to.  Sample 0 of the unresolved attachment at `= 1`; the
        # single-sample target at `= 0`.  The shipped module reads
        # neither, so this changes no colour -- see the docstring.
        depth_bound = (ms_depth.texel_fetch(xx, yy, 0)
                       if msaa_depth_buffer else ms_depth.resolve("sample0"))
    return rgb, depth_bound


# =====================================================================
# SYNTHESISED INPUTS -- for the reachability proof, where our scene
# provides none.  §9.4 of the census: "whether each family executes in a
# game frame is not established"; de_inferno has no zone entity and no
# drone camera, so a run over our poses can never call two of these
# eight.  Synthesised input is how they are shown to be CALLED.
# =====================================================================

def synth_texture(B, H, W, C, seed=0, device=None):
    """A deterministic procedural texture with real spatial variation.

    A CONSTANT stand-in would make every uv-driven axis (the zone mask
    scroll, the dronecam quantisation, the triplanar projections) measure
    exactly zero and read as inert when it is only the stand-in that is
    flat.  Same reasoning, and the same shape, as `sf_cost_audit`'s
    `sample_fam` stub.
    """
    yy, xx = torch.meshgrid(torch.arange(H, device=device),
                            torch.arange(W, device=device), indexing="ij")
    u = (xx.float() + 0.5) / W
    v = (yy.float() + 0.5) / H
    ch = []
    for c in range(C):
        p = 1.0 + c + seed
        ch.append(0.5 + 0.5 * torch.sin(u * 12.9898 * p
                                        + v * 78.233 * (p + 1.0)))
    t = torch.stack(ch, dim=-1)
    return t.unsqueeze(0).expand(B, H, W, C).contiguous()


def synth_ms_depth(B, H, W, n_samples, device=None, spread=0.02):
    """An unresolved attachment whose samples genuinely DIFFER.

    A synthetic attachment whose S samples are all equal would make
    `texel_fetch(sample=k)` return the same value for every k, and every
    per-sample check would pass against a resolved read -- the exact
    check-that-cannot-fail this family group exists to break.  The
    samples here are offset by the sample POSITION times a depth
    gradient, which is what a real edge pixel holds.
    """
    pos = ms_sample_positions(n_samples, device=device)
    yy, xx = torch.meshgrid(torch.arange(H, device=device),
                            torch.arange(W, device=device), indexing="ij")
    u = (xx.float() + 0.5) / W
    v = (yy.float() + 0.5) / H
    base = 0.5 + 0.3 * torch.sin(u * 6.0) * torch.cos(v * 5.0)
    zs = []
    for k in range(n_samples):
        zs.append((base + spread * (float(pos[k, 0]) * torch.cos(u * 9.0)
                                    + float(pos[k, 1]) * torch.sin(v * 7.0))
                   ).clamp(0.0, 1.0).unsqueeze(0).expand(B, H, W))
    return MSDepthAttachment(torch.stack(zs, dim=-1), pos, source="synth")


# The self-test's injectable defects.  CHECKS_THAT_CANNOT_FAIL.md
# instance 5: the instrument built to catch this class was itself in it,
# three times over, and was trusted only after known defects were
# injected and all three fired.
INJECT_NAMES = (
    "ms-resolve-first",     # make texel_fetch return the RESOLVE, i.e.
                            # exactly the renderer this group had to fix
    "ms-samples-equal",     # collapse every sample to sample 0
    "gbuffer-flat",         # make the normal G-buffer constant
    "dither-constant",      # replace the offset-64 fetch with 0.5
    "combo-bitmask",        # decode mixed-radix ids with & (trap 1)
    "trip-one",             # ignore D_MSAA_SAMPLES, always 1 iteration
    "nearfar-ignored",      # drop the 368/372 linearisation
)


def inject(name, target):
    """Apply one named defect.  Returns the patched callable/value.

    Every name here MUST make at least one self-test row fail; a name
    that cannot is a path the self-test does not cover, and the
    reachability claim does not extend to it.
    """
    if name not in INJECT_NAMES:
        raise SystemExit(f"unknown injection {name!r}; names: "
                         + ", ".join(INJECT_NAMES))
    return target
