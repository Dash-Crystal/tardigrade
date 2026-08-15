"""Batch-3 screen-space FILTER families, plus the one real BINNER.

Every function here is transcribed from decompiled GLSL that this repo can
now regenerate locally:

    python3 iji_model/counter_strike_render/vcs/extract_fam.py <family> --out .scratch-fbx

Line references in the docstrings are into
`.scratch-fbx/<family>/<family>_r<record>m<module>.glsl` from that command,
which is byte-for-byte reproducible: the record walk consumes 100.00% of
every .vcs and container-declared module size equalled the SPIR-V
instruction walk on 144/144 modules.

WHY A SEPARATE FILE. `gpu_render.py` is 16,846 lines and sixteen agents are
appending to it concurrently; four merges on 2026-08-07 needed hand
resolution because two agents appended to the same anchor and git spliced
one literal's tail onto another's head. Everything that can live outside
that file does. What gpu_render.py gains is one contiguous import + flag +
call block, at the very end of each list.

CLASSIFICATION, AND THE ONE PLACE THE BRIEF WAS WRONG.

`SHADER_CALLFLOW_loop_families_batch3.md` sec 0b defines a three-way test:
MARCH if max loop nesting depth >= 2; BINNER if depth <= 1 and some region
contains `findLSB`; FILTER if depth <= 1 and none does. Measured over all
modules of all families here:

    family                          loops  findLSB  class
    inferno                           2       2     BINNER
    lpv_debug_grid                    0       0     neither -- see below
    everything else in this file      1-2     0     FILTER

`lpv_debug_grid` was briefed to me as a binner with "44 fetch sites, 2,588
FLOPs/px, 8 of 9 regions binner pairs". Its actual bytecode -- all five
modules of all three stages, 292 lines total -- has ZERO loops, ZERO
findLSB and ZERO texture fetches. It is a ray-sphere intersection that
writes the hit normal as colour. Writing the clustered-light loop I was
asked for would have put ~2,588 FLOPs/px of invented lighting into the
renderer. It is implemented below as what it is, and the discrepancy is
recorded rather than smoothed over.

ON THE RISK THE BRIEF NAMES -- a binner written WITHOUT `findLSB` would be
misclassified as a filter, silently dropping a family from the lit set.
Every family here was swept for the other ways to walk a cull bitmask:
`findMSB`, `bitCount`, `(x >> i) & 1` shift-test loops, `x &= x - 1`,
`uvec4` word indexing by `>> 5`, and `atomicOr/And/Add`. The only hit
anywhere is inferno's `x & (x - 1)`, and inferno carries `findLSB`
regardless. So within these families the split does not rest on `findLSB`
alone. That is a statement about these families and not about the other 23
of the 41.

ONE CLASSIFICATION THAT DOES NOT FIT ITS LABEL. `csgo_volume_viewer` is
called a FILTER because its single loop is not nested -- and its body
advances a position along a ray by a fixed step and samples a texture3D 64
times. It is a ray march with an unnested loop. Nesting depth does not
separate "march" from "filter" as cleanly as the three-way test implies,
and the honest reading is that `csgo_volume_viewer` is a march that the
depth metric cannot see. It is transcribed exactly as written; no
ray-marching structure has been imported into anything else.

NO DEFAULT HERE DISABLES A PATH. Every pass runs at every value of every
axis it declares; quality levels never gate implementation
(CHECKS_THAT_CANNOT_FAIL.md, standing rule).
"""
import torch

__all__ = [
    "B3_FAMILIES", "B3_CLASS", "b3_tex2d",
    "b3_blur", "b3_general_filter", "b3_convolve_environment_map",
    "b3_terrain_composite_slope", "b3_outlines", "b3_volume_viewer",
    "b3_msaa_resolve", "b3_visualize_depth", "b3_lpv_debug_grid",
    "b3_inferno",
]


# --------------------------------------------------------------------
# sampling helpers
# --------------------------------------------------------------------

def b3_tex2d(img, uv, lod=None):
    """Bilinear `texture(sampler2D, uv)` with clamp-to-edge.

    `img` (B,H,W,C), `uv` (B,H,W,2) in [0,1] with v DOWN the image, which
    is the Vulkan convention these shaders are compiled for. `lod` is
    accepted and applied as a box pre-filter (2**lod averaging) so that
    `textureLod(..., float(k))` calls transcribe honestly rather than
    silently collapsing to lod 0 -- a mip argument dropped on the floor
    is exactly the kind of quiet no-op this project keeps finding.
    """
    B, H, W, C = img.shape
    src = img
    if lod is not None:
        k = int(lod)
        if k > 0:
            f = min(2 ** k, max(1, min(H, W)))
            src = torch.nn.functional.avg_pool2d(
                img.permute(0, 3, 1, 2), kernel_size=f, stride=f,
                ceil_mode=True).permute(0, 2, 3, 1)
    g = uv * 2.0 - 1.0
    out = torch.nn.functional.grid_sample(
        src.permute(0, 3, 1, 2), g, mode="bilinear",
        align_corners=False, padding_mode="border")
    return out.permute(0, 2, 3, 1)


def b3_tex3d(vol, p):
    """`textureLod(sampler3D, p, 0.0)`; `vol` (B,D,H,W,C), `p` in [0,1]^3.

    grid_sample wants a 5-D grid (N, D_out, H_out, W_out, 3) whose last
    axis is ordered (x, y, z) against input axes (D, H, W) -- i.e. the
    REVERSE of the volume's index order. Flattening the arbitrary leading
    shape of `p` into W_out keeps this correct for any caller rank.
    """
    Bv, D, H, W, C = vol.shape
    lead = p.shape[:-1]
    n = 1
    for s in lead[1:]:
        n *= s
    g = (p.reshape(Bv, 1, 1, n, 3) * 2.0 - 1.0)
    out = torch.nn.functional.grid_sample(
        vol.permute(0, 4, 1, 2, 3), g,
        mode="bilinear", align_corners=False, padding_mode="border")
    return out.permute(0, 2, 3, 4, 1).reshape(*lead, C)


def _ss(a, b, x):
    """GLSL smoothstep with no ordering assumption on (a, b).

    inferno:111 calls it with a > b three times and outlines:55 once, so
    the descending form is REACHED, not hypothetical.
    """
    d = b - a
    d = torch.where(d.abs() < 1e-20, torch.full_like(d, 1e-20), d) \
        if torch.is_tensor(d) else (d if abs(d) > 1e-20 else 1e-20)
    t = ((x - a) / d).clamp(0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


# --------------------------------------------------------------------
# blur -- core/shaders/vfx/blur_vulkan_50_ps.vcs, 5 records / 9 modules
# --------------------------------------------------------------------
# Four separable Gaussian kernels x two directions, plus a passthrough.
# r0m0 is the passthrough (:11, no loop, no weights). r1..r4 are the four
# kernels; mK=0 steps in x scaled by CB(set1,b0)._m0.x and mK=1 steps in y
# scaled by _m0.y -- blur_r1m0:43 vs blur_r1m1:43, identical but for the
# vec2 constructor. Every literal below is at the full printed precision
# of the module's `const float[]` declarations; they are float32 bit
# patterns and rounding them would be a silent change of kernel.
B3_BLUR_KERNELS = {
    # r1: 5 taps  (blur_r1m0:15-16)
    5: ([-3.0, -1.1824250221252441, 0.0, 1.1824250221252441, 3.0],
        [0.00443299999460578, 0.29604199528694153, 0.3990499973297119,
         0.29604199528694153, 0.00443299999460578]),
    # r2: 5 taps, wider  (blur_r2m0:15-16)
    55: ([-3.096215009689331, -1.276877999305725, 0.0,
          1.276877999305725, 3.096215009689331],
         [0.01982700079679489, 0.3205609917640686, 0.3192239999771118,
          0.3205609917640686, 0.01982700079679489]),
    # r3: 7 taps  (blur_r3m0:15-16)
    7: ([-5.142348766326904, -3.2417960166931152, -1.3799420595169067,
         0.0, 1.3799420595169067, 3.2417960166931152, 5.142348766326904],
        [0.004486999940127134, 0.06918500363826752, 0.31232500076293945,
         0.22800500690937042, 0.31232500076293945, 0.06918500363826752,
         0.004486999940127134]),
    # r4: 13 taps  (blur_r4m0:15-16)
    13: ([-11.251852035522461, -9.289172172546387, -7.329586029052734,
          -5.37268590927124, -3.417910099029541, -1.4645570516586304, 0.0,
          1.4645570516586304, 3.417910099029541, 5.37268590927124,
          7.329586029052734, 9.289172172546387, 11.251852035522461],
         [0.0005339999916031957, 0.0037330000195652246,
          0.018004000186920166, 0.05992799997329712, 0.13774000108242035,
          0.21867699921131134, 0.12276499718427658, 0.21867699921131134,
          0.13774000108242035, 0.05992799997329712, 0.018004000186920166,
          0.0037330000195652246, 0.0005339999916031957]),
}
# The two 5-tap kernels are DIFFERENT kernels, not a duplicate: r1 sums to
# 0.99999999 with a 0.399 centre, r2 sums to 0.99999999 with a 0.319
# centre and wider taps. Keying them 5 and 55 keeps both; collapsing them
# would drop a shipped variant on the grounds that a count matched.
B3_BLUR_TAPS = (5, 55, 7, 13)


def b3_blur(img, uv, step_xy, taps=5, axis="x"):
    """blur_r{1..4}m{0,1}:29-49. Separable Gaussian, fixed tap loop.

    `step_xy` is CB(set 1, binding 0)._m0: .x used by the mK=0 modules,
    .y by mK=1. `taps` selects the record; `axis` selects the module.
    Passing taps=0 gives r0m0, the passthrough at :11 -- which is a real
    shipped module, not a disabled path.
    """
    if taps == 0:                                          # r0m0:11
        return b3_tex2d(img, uv, lod=0)
    off, wts = B3_BLUR_KERNELS[taps]
    acc = torch.zeros_like(img)
    s = step_xy[..., 0:1] if axis == "x" else step_xy[..., 1:2]
    for o, w in zip(off, wts):                             # :37-48
        d = torch.zeros_like(uv)
        if axis == "x":
            d[..., 0:1] = o * s                            # :43 vec2(o*s, 0)
        else:
            d[..., 1:2] = o * s                            # :43 vec2(0, o*s)
        acc = acc + b3_tex2d(img, uv + d) * w
    return acc                                             # :49


# --------------------------------------------------------------------
# general_filter -- core/.../general_filter_vulkan_50_ps.vcs, 4 rec / 8 mod
# --------------------------------------------------------------------

def b3_general_filter(img, uv, taps, cmat, n=None):
    """general_filter_r0m0:33-54. Variable-tap filter + colour matrix.

    `taps` is CB(set 1, binding 2)._m2, a vec4[32]: .xy is the uv offset
    and .z the weight (:48). `.w` is never read by the module. `n` is
    int(CB._m0.x) at :35 and defaults to the full declared 32 -- NOT to a
    smaller number, because a default that shortens the loop is a default
    that quietly disables taps.

    `cmat` is CB._m1, a row_major mat4 at offset 32 applied to the
    accumulated RGBA at :54. The multiply is `vec4 * mat4`, i.e. the
    vector is on the LEFT, which in GLSL is a row-vector product and
    therefore `acc @ cmat` and not `cmat @ acc`.
    """
    k = int(taps.shape[0]) if n is None else int(n)
    k = max(0, min(k, int(taps.shape[0])))
    acc = torch.zeros_like(img)
    for i in range(k):                                     # :42-53
        d = taps[i, :2].to(uv.dtype).view(*([1] * (uv.dim() - 1)), 2)
        acc = acc + b3_tex2d(img, uv + d, lod=0) * float(taps[i, 2])
    return acc @ cmat.to(acc.dtype)                        # :54


# --------------------------------------------------------------------
# convolve_environment_map -- core/.../convolve_environment_map_..._ps.vcs
# --------------------------------------------------------------------

def b3_convolve_environment_map(cube_fn, dirs, samples, weights, n=None):
    """convolve_environment_map_r0m0:25-53. Cube convolution, 1 loop.

    `dirs` (...,3) is the interpolated direction, `samples` is CB._m0 (a
    vec4[2048]: xyz = tangent-space direction, w = cube LOD) and
    `weights` is CB._m1 with the scalar in .x. `n` is CB(set1,b0)._m0, an
    int at byte offset 4 (:10), defaulting to the full 2048.

    Tangent frame at :29-30. Note the `mix` sense: `mix(vec3(1,0,0),
    vec3(0,0,1), bvec3(abs(n.z) < 0.999000012874603271484375))` selects
    the SECOND argument when the condition is TRUE, so the up vector is
    vec3(0,0,1) for MOST directions and vec3(1,0,0) only near the poles.
    Reading that backwards flips the frame everywhere except the poles,
    which is why it is spelled out here.

    `cube_fn(d, lod)` samples the cube. Alpha is initialised to 1.0 at
    :32 and never written, so it stays 1.0 (:53).
    """
    nz = dirs[..., 2].abs()
    up = torch.where(
        (nz < 0.999000012874603271484375).unsqueeze(-1),
        torch.tensor([0.0, 0.0, 1.0], device=dirs.device,
                     dtype=dirs.dtype).expand_as(dirs),
        torch.tensor([1.0, 0.0, 0.0], device=dirs.device,
                     dtype=dirs.dtype).expand_as(dirs))
    nrm = dirs / dirs.norm(dim=-1, keepdim=True).clamp(min=1e-9)
    t = torch.cross(up, nrm, dim=-1)                        # :29
    t = t / t.norm(dim=-1, keepdim=True).clamp(min=1e-9)
    b = torch.cross(nrm, t, dim=-1)                         # :30
    k = int(samples.shape[0]) if n is None else int(n)
    k = max(0, min(k, int(samples.shape[0])))
    acc = torch.zeros_like(dirs)
    for i in range(k):                                      # :36-52
        s = samples[i]
        d = t * float(s[0]) + b * float(s[1]) + nrm * float(s[2])
        acc = acc + (cube_fn(d, float(s[3]))
                     * float(weights[i, 0])).clamp(min=0.0)  # :42 max(,0)
    return acc, torch.ones_like(acc[..., 0])                # :32/:53


# --------------------------------------------------------------------
# tools_terrain_composite_slope
# --------------------------------------------------------------------
# The 8-neighbour ring, in the module's own order (:5).
B3_SLOPE_RING = [(-1.0, -1.0), (0.0, -1.0), (1.0, -1.0), (1.0, 0.0),
                 (1.0, 1.0), (0.0, 1.0), (-1.0, 1.0), (-1.0, 0.0)]


def b3_terrain_composite_slope(img, uv, hscale, wscale):
    """tools_terrain_composite_slope_r0m0:19-44. 8-neighbour max slope.

    `hscale` is CB(set1,b0,scalar)._m0, a vec3; `wscale` is _m1, a vec2.
    Note the `.zyx` swizzle on BOTH fetches (:22, :36) -- the height lives
    in the texture's BLUE channel, so sampling `.xyz` reads the wrong
    component and produces a plausible-looking wrong answer.

    Per neighbour the module forms
        d = abs(centre - (vec3(texel * wscale * off, 0) + neighbour))
    and accumulates `max(acc, d.z*d.z / dot(d,d))` (:38). d.z is the
    height difference and d.xy the world-space step, so the ratio is
    cos^2 of the angle from horizontal. Output is that scalar in rgb with
    alpha 1 (:44).

    `textureSize` is re-evaluated inside the loop at :34 in the original.
    It is loop-invariant, so it is hoisted here; that changes the FLOP
    count and not the value.
    """
    B, H, W, _ = img.shape
    texel = torch.tensor([1.0 / W, 1.0 / H], device=img.device,
                         dtype=img.dtype)                   # :35
    hs = hscale.to(img.dtype).view(*([1] * (img.dim() - 1)), 3)
    ws = wscale.to(img.dtype).view(*([1] * (uv.dim() - 1)), 2)
    centre = b3_tex2d(img, uv)[..., [2, 1, 0]] * hs         # :21-22 .zyx
    acc = torch.zeros_like(centre[..., 0])
    for ox, oy in B3_SLOPE_RING:                            # :28-43
        o = torch.tensor([ox, oy], device=img.device, dtype=img.dtype)
        nb = b3_tex2d(img, uv + texel * o)[..., [2, 1, 0]] * hs  # :36 .zyx
        disp = torch.zeros_like(nb)
        disp[..., :2] = texel * ws * o                      # :36 vec3(..,0)
        d = (centre - (disp + nb)).abs()
        num = d[..., 2] * d[..., 2]
        acc = torch.maximum(acc, num / (d * d).sum(-1).clamp(min=1e-20))
    return acc.unsqueeze(-1).expand(*acc.shape, 3), torch.ones_like(acc)


# --------------------------------------------------------------------
# outlines -- csgo/shaders/vfx/outlines_vulkan_50_ps.vcs
# --------------------------------------------------------------------
B3_OUTLINE_TAPS = [(2.0, 1.0), (1.0, -2.0), (-2.0, -1.0), (-1.0, 2.0),
                   (0.0, 0.0)]                              # :3


def b3_outlines(img, frag_xy, inv_viewport, width):
    """outlines_r0m0:20-60. 5-tap rotated cross, edge from a black count.

    `inv_viewport` is CB(set 1, binding 1) at byte offset 352, `.xy`
    (:7/:22); `width` is CB(set 1, binding 0)._m0 (:12/:25).

    The count at :42 is `step(c.x+c.y+c.z, 0.0)`, i.e. it counts taps
    whose RGB sums to <= 0 -- BLACK taps. `edge = abs(count/5 - 0.5)*2`
    is 0 when exactly half the taps are black (a real silhouette) and 1
    when all or none are, so the discard at :51-54 (`0.999 - edge < 0`)
    throws away uniform interior AND uniform exterior.

    Returns (rgb, alpha, keep). `keep` is the discard mask; the caller
    must apply it. Returning a colour and dropping the mask would make
    the pass paint over the whole frame.
    """
    uv = frag_xy * inv_viewport                             # :22
    r = (uv - 0.5).norm(dim=-1)                             # :24
    spread = width * (1.0 - r)                              # :25 mix(w,0,r)
    acc = torch.zeros_like(img)
    cnt = torch.zeros_like(r)
    for tx, ty in B3_OUTLINE_TAPS:                          # :34-48
        o = torch.stack([spread * tx, spread * ty], dim=-1)
        c = b3_tex2d(img, uv + o)
        acc = acc + c
        s = c[..., 0] + c[..., 1] + c[..., 2]
        cnt = cnt + (s <= 0.0).to(cnt.dtype)                # :42 step(s,0)
    avg = acc * 0.20000000298023223876953125                # :49
    edge = ((cnt * 0.20000000298023223876953125) - 0.5).abs() * 2.0  # :50
    keep = (0.999000012874603271484375 - edge) >= 0.0       # :51
    vig = _ss(-0.20000000298023223876953125,
              0.4000000059604644775390625,
              0.60000002384185791015625 - r)                # :55
    rgb = avg[..., :3] * ((1.0 - edge) * 4.0).unsqueeze(-1) \
        * vig.unsqueeze(-1)
    return rgb, avg[..., 3], keep                           # :60


# --------------------------------------------------------------------
# csgo_volume_viewer -- csgo_core/.../csgo_volume_viewer_..._ps.vcs
# --------------------------------------------------------------------

def b3_volume_viewer(vol, wpos, eye):
    """csgo_volume_viewer_r0m0:14-91.

    Classified FILTER by the batch-3 three-way test because its one loop
    is not nested. It is a RAY MARCH: 64 fixed steps of 0.02 along the
    view ray through a texture3D, accumulating, with an early break when
    the position leaves the unit cube (:36-85). Recorded in the module
    docstring; transcribed here exactly as written.

    `p` starts at `(wpos + 16.0) * 0.03125` (:20), which maps the world
    box [-16, 16] to [0, 1]. The divisor at :91 is the loop counter AFTER
    its increment (`_3048 = _17017 + 1` at :28), so a march that breaks
    immediately divides by 1 and never by 0.
    """
    d = wpos - eye
    d = d / d.norm(dim=-1, keepdim=True).clamp(min=1e-9)     # :16
    p = (wpos + 16.0) * 0.03125                              # :20
    acc = torch.zeros_like(wpos)
    out = torch.zeros_like(wpos)
    n = torch.ones(*wpos.shape[:-1], device=wpos.device, dtype=wpos.dtype)
    live = torch.ones(*wpos.shape[:-1], dtype=torch.bool, device=wpos.device)
    for i in range(64):                                      # :26-89
        cnt = float(i + 1)                                   # :28
        if i + 1 >= 64:                                      # :29 !(i+1<64)
            out = torch.where(live.unsqueeze(-1), acc, out)
            n = torch.where(live, torch.full_like(n, cnt), n)
            break
        nxt_acc = acc + b3_tex3d(vol, p)[..., :3]            # :34
        p2 = p + d * 0.0199999995529651641845703125          # :35
        oob = (p2 < 0.0).any(-1) | (p2 > 1.0).any(-1)        # :36-80
        stop = live & oob
        out = torch.where(stop.unsqueeze(-1), nxt_acc, out)  # :83
        n = torch.where(stop, torch.full_like(n, cnt), n)
        live = live & ~oob
        acc = torch.where(live.unsqueeze(-1), nxt_acc, acc)
        p = torch.where(live.unsqueeze(-1), p2, p)
    return out / n.unsqueeze(-1).clamp(min=1.0), \
        torch.ones_like(n)                                   # :91


# --------------------------------------------------------------------
# msaa_resolve -- core/shaders/vfx/msaa_resolve_vulkan_50_ps.vcs
# --------------------------------------------------------------------
# 72 modules = {2, 4, 8 samples} x {plain, depth-plane-alpha} x 12. The
# 12 is not a shading difference in any module pair inspected; the two
# SHAPES are what differ, and both are implemented.
B3_MSAA_SAMPLES = (2, 4, 8)


def b3_msaa_resolve(ms, frag_off, plane=None):
    """msaa_resolve_r0m0:15-40 (plain) and r0m12:29-58 (depth-plane).

    `ms` is (B,H,W,S,C): S per-sample values. The fetch at :28 is
    `texelFetch(tex, ivec2(gl_FragCoord.xy - CB._m0.xy), s)` -- the
    integer offset `frag_off` is subtracted from the pixel coordinate,
    not added. RGB is clamped to [0, 65504] (:29), which is the largest
    finite float16, and each sample is weighted 1/S (:34).

    ALPHA IN THE PLAIN VARIANT. At r0m0:34 the module writes
    `vec4(rgb*0.5, _3401.w*0.5)`, so alpha resolves normally. At
    r0m12:34 the fourth component is the file-scope `vec4 _3;` declared
    at :4 and never assigned -- an UNDEFINED value that spirv-cross
    surfaced rather than hid. Alpha is taken as 0 there and the whole
    alpha is overwritten at :58 anyway; this is stated because silently
    substituting the plain variant's `.w*0.5` would be inventing a value
    the module does not have.

    `plane`, when given, is the r0m12 shape: a dict with `inv_proj`
    (the vec4-rows of CB._m0), `plane` (CB._m1), `ramp` (CB._m2),
    `inv_viewport` (CB(b1) offset 336), `znear` (offset 368) and `zfar`
    (offset 372), plus `depth_ms` (B,H,W,S). Alpha then becomes the max
    of two clamped linear ramps of a plane distance (:58).
    """
    S = ms.shape[3]
    w = 1.0 / float(S)
    rgb = ms[..., :3].clamp(0.0, 65504.0)                    # :29
    acc = (rgb * w).sum(dim=3)                               # :34
    if plane is None:
        a = (ms[..., 3] * w).sum(dim=3)                      # r0m0:34
        return acc, a
    dmin = plane["depth_ms"].min(dim=3).values               # r0m12:41 min
    uv = plane["frag_xy"] * plane["inv_viewport"]            # :56
    z = ((dmin - plane["znear"])
         / (plane["zfar"] - plane["znear"])).clamp(0.0, 1.0)  # :57
    ndc = torch.stack([uv[..., 0] * 2.0 - 1.0,
                       (1.0 - uv[..., 1]) * 2.0 - 1.0,
                       z, torch.ones_like(z)], dim=-1)        # :57
    h = ndc @ plane["inv_proj"].to(ndc.dtype)                 # :57
    wp = h[..., :3] / h[..., 3:4].clamp(min=1e-20)
    dist = (wp * plane["plane"][:3].to(wp.dtype)).sum(-1) \
        + float(plane["plane"][3])                            # :58 dot(..,1)
    r = plane["ramp"]
    a = torch.maximum((dist * float(r[0]) + float(r[1])).clamp(0.0, 1.0),
                      (dist * float(r[2]) + float(r[3])).clamp(0.0, 1.0))
    return acc, a                                             # :58


# --------------------------------------------------------------------
# visualize_depth -- core/shaders/vfx/visualize_depth_vulkan_50_ps.vcs
# --------------------------------------------------------------------
B3_VISDEPTH_AXIS = [(1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)]
B3_VISDEPTH_MODES = (0, 1, 2, 3)


def b3_visualize_depth(tex, uv, lod=0, mode=0):
    """visualize_depth_r0m0:18-83. Four modes on CB(set1,b0)._m1 (:9).

    mode 0 (:23)  fract(textureLod(tex, uv, lod).x * 32.0) in all four
                  channels -- alpha included, which is why it is a vec4
                  broadcast and not vec3 + 1.
    mode 1 (:30)  walk mip 9 down to 0 and take the FIRST level whose
                  fract(.x) > 0; colour = axis[level % 3] scaled by
                  fract(textureLod(tex, uv, 0).x * 64.0). No hit -> black.
                  Note the scale is 64 here and 32 in the other modes.
    mode 2 (:56)  sum over mips 0..9 of axis[i % 3] * fract(mip_i * 32.0).
    mode 3 (:77)  black, alpha 1. This is the `else`, a real shipped
                  branch, not a fallback I invented.

    `lod` is CB._m0, an int, and is read ONLY by mode 0 (:23).
    All four modes are implemented; none is gated.
    """
    ax = torch.tensor(B3_VISDEPTH_AXIS, device=tex.device, dtype=tex.dtype)
    if mode == 0:
        v = b3_tex2d(tex, uv, lod=lod)[..., 0]
        f = torch.frac(v * 32.0)
        return f.unsqueeze(-1).expand(*f.shape, 3), f        # :23
    if mode == 1:
        rgb = torch.zeros(*uv.shape[:-1], 3, device=tex.device,
                          dtype=tex.dtype)
        found = torch.zeros(*uv.shape[:-1], dtype=torch.bool,
                            device=tex.device)
        for lv in range(9, -1, -1):                          # :33-48
            hit = (torch.frac(b3_tex2d(tex, uv, lod=lv)[..., 0]) > 0.0)
            take = hit & ~found
            c = ax[lv - 3 * (lv // 3)]                       # :42
            rgb = torch.where(take.unsqueeze(-1), c.expand_as(rgb), rgb)
            found = found | hit
        s = torch.frac(b3_tex2d(tex, uv, lod=0)[..., 0] * 64.0)  # :49
        return rgb * s.unsqueeze(-1), torch.ones_like(s)
    if mode == 2:
        acc = torch.zeros(*uv.shape[:-1], 3, device=tex.device,
                          dtype=tex.dtype)
        for i in range(10):                                  # :61-72
            f = torch.frac(b3_tex2d(tex, uv, lod=i)[..., 0] * 32.0)
            acc = acc + ax[i - 3 * (i // 3)] * f.unsqueeze(-1)
        return acc, torch.ones_like(acc[..., 0])             # :73
    z = torch.zeros(*uv.shape[:-1], 3, device=tex.device, dtype=tex.dtype)
    return z, torch.ones_like(z[..., 0])                     # :77


# --------------------------------------------------------------------
# lpv_debug_grid -- core/shaders/vfx/lpv_debug_grid_vulkan_50_ps.vcs
# --------------------------------------------------------------------

def b3_lpv_debug_grid(wpos, eye, sphere, jitter=None, mode=0):
    """lpv_debug_grid_r0m0:19-38 (mode 0) and r0m1:6-9 (mode 1).

    THIS FAMILY IS NOT A BINNER. It has no loop, no findLSB and no
    texture fetch in any of its five modules across vs/gs/ps. See the
    module docstring.

    mode 0 (r0m0) is a ray-sphere intersection:
        rd  = normalize((wpos + jitter) - eye)              :19
        oc  = eye - sphere.xyz                              :20
        b   = dot(rd, oc)                                   :21
        h   = b*b - (dot(oc,oc) - sphere.w*sphere.w)        :22
        if (h < 0.0) discard                                :23-26
        t   = -b - (h > 0.0 ? sqrt(h) : 0.0)                :27-34
        rgb = 0.5 + normalize((eye + rd*t) - sphere.xyz)*0.5 :36
    mode 1 (r0m1) is the same encoding of an interpolated normal, with
    the sign flipped on back faces:
        rgb = 0.5 + (n * (gl_FrontFacing ? 1.0 : -1.0)) * 0.5

    Note `h > 0.0` at :29 and `h < 0.0` at :23 are separate tests, so
    exactly h == 0.0 takes the tangent path with sqrt omitted. That is
    the module's own branch structure and it is preserved.

    Returns (rgb, alpha, keep); `keep` is the discard mask from :23.
    """
    if mode == 1:                                            # r0m1
        n = wpos / wpos.norm(dim=-1, keepdim=True).clamp(min=1e-9)
        sgn = 1.0 if jitter is None else float(jitter)
        rgb = 0.5 + (n * sgn) * 0.5
        return rgb, torch.ones_like(rgb[..., 0]), \
            torch.ones(*rgb.shape[:-1], dtype=torch.bool, device=rgb.device)
    j = torch.zeros_like(wpos) if jitter is None else jitter
    rd = (wpos + j) - eye                                    # :19
    rd = rd / rd.norm(dim=-1, keepdim=True).clamp(min=1e-9)
    c = sphere[..., :3]
    rr = sphere[..., 3]
    oc = eye - c                                             # :20
    b = (rd * oc).sum(-1)                                    # :21
    h = b * b - ((oc * oc).sum(-1) - rr * rr)                # :22
    keep = h >= 0.0                                          # :23
    t = -b - torch.where(h > 0.0, h.clamp(min=0.0).sqrt(),
                         torch.zeros_like(h))                # :27-34
    hit = eye + rd * t.unsqueeze(-1)
    nn = hit - c
    nn = nn / nn.norm(dim=-1, keepdim=True).clamp(min=1e-9)
    return 0.5 + nn * 0.5, torch.ones_like(b), keep          # :36


# --------------------------------------------------------------------
# inferno -- csgo/shaders/vfx/inferno_vulkan_50_ps.vcs   *** BINNER ***
# --------------------------------------------------------------------

def b3_inferno(uv, frag_depth, view_ray, eye, fwd, blobs, groups,
               group_mask, blob_mask, now, znear, zfar, depth_ab,
               noise_fn, fire_rgb, fire_scale):
    """inferno_r0m0:58-137. The only genuine binner in this slice.

    THE CULL STRUCTURE, which is what makes this a binner and not a
    filter. The screen is a 4x4 tile grid (:60):

        tile = dot(floor(uv * 4.0), vec2(1.0, 4.0))     0..15
        word = tile / 4u ; comp = tile % 4u             -> uvec4[4]

    `group_mask` is CB._m3, a uvec4[4] = 16 uints, one per tile: bit g
    set means group g touches this tile. Empty tile discards at :63-66.

    OUTER loop :73-132 peels that mask with `findLSB` (:79) and
    `x & (x - 1)` (:80) -- the two idioms the BINNER/FILTER split turns
    on, both present.
    INNER loop :91-115 peels `blob_mask` = CB._m2, a uvec4[64] = 256
    uints indexed by `(g*16 + tile)`, whose bit b means blob b of group g
    touches this tile. Blob index is `g*16 + b` into CB._m0, a vec4[256]
    -- 16 groups x 16 blobs, which is exactly the declared array size.

    THE SHADING.
      world position (:67) is reconstructed from the depth buffer:
          t   = clamp((depth - znear)/(zfar - znear), 0, 1)
          wp  = eye + ray * (1 / ((t*a + b) * dot(fwd, ray)))
      with (a, b) = `depth_ab` = CB(set1,b1) offset 256 .zw. The
      reciprocal makes (t*a + b) a RECIPROCAL depth, so the whole factor
      is z_view / cos(angle to forward) -- distance along the ray.
      per blob (:100-111):
          d      = wp - blob.xyz ;  d.z *= 2.0     <- vertical squash
          age    = now - blob.w                    <- radius IS the age
          sd     = length(d) - age
          skip if sd >= 100.0
          v      = smoothstep(100, 50, sd)               descending
                 * smoothstep(0, 1, age)
                 * smoothstep(6.5, 5.5, blob.w - group.x)   descending
          inner  = max(inner, v)                   <- MAX, not sum
      per group (:116-129):
          gage   = now - group.x
          a1     = inner * smoothstep(0, 0.5, gage)
          if a1 > 0:  wp.z += 100.0 * smoothstep(7.0, 0.0, gage)
                      n = noise(wp * 0.007000000216066837310791015625)
                      v = clamp((n.y + a1) * a1, 0, 1)
                          * (0.85000002384185791015625
                             + n.x * 0.100000001490116119384765625)
          else:       v = a1
          outer  = max(outer, v * smoothstep(20.0, 6.0, gage))
      discard if outer - 1e-4 < 0 (:133); output
          vec4(fire_rgb, outer) * fire_scale                (:137)
      -- the scale multiplies ALPHA as well as RGB.

    Masks are uint32 tensors. `blobs` is (G*16, 4), `groups` is (G, 4).
    """
    tile = (uv * 4.0).floor()
    tile = (tile[..., 0] + tile[..., 1] * 4.0).long().clamp(0, 15)  # :60
    gm = group_mask.reshape(-1)[tile]                        # :63 uvec4[4]
    keep = gm != 0

    t = ((frag_depth - znear) / (zfar - znear)).clamp(0.0, 1.0)  # :67
    denom = (t * depth_ab[0] + depth_ab[1]) * (fwd * view_ray).sum(-1)
    wp = eye + view_ray * (1.0 / denom.clamp(min=1e-20)).unsqueeze(-1)

    outer = torch.zeros_like(t)
    G = int(groups.shape[0])
    for g in range(G):                                       # :73-132
        active = ((gm >> g) & 1).bool()                      # findLSB peel
        if not bool(active.any()):
            continue
        gage = now - groups[g, 0]                            # :81
        bm = blob_mask.reshape(-1)[(g * 16 + tile).clamp(
            0, blob_mask.numel() - 1)]                       # :90
        inner = torch.zeros_like(t)
        for b in range(16):                                  # :91-115
            on = ((bm >> b) & 1).bool()
            if not bool(on.any()):
                continue
            bl = blobs[g * 16 + b]                           # :99-100
            d = wp - bl[:3]
            d = torch.stack([d[..., 0], d[..., 1], d[..., 2] * 2.0], -1)
            age = now - bl[3]                                # :102
            sd = d.norm(dim=-1) - age                        # :103
            v = (_ss(100.0, 50.0, sd) * _ss(0.0, 1.0, age)
                 * _ss(6.5, 5.5, bl[3] - groups[g, 0]))      # :111
            v = torch.where((sd < 100.0) & on, v, torch.zeros_like(v))
            inner = torch.maximum(inner, v)                  # :111 max
        a1 = inner * _ss(0.0, 0.5, gage)                     # :116
        wpn = torch.stack(
            [wp[..., 0], wp[..., 1],
             wp[..., 2] + 100.0 * _ss(7.0, 0.0, gage)], -1)  # :120
        n = noise_fn(wpn * 0.007000000216066837310791015625)  # :121
        vv = ((n[..., 1] + a1) * a1).clamp(0.0, 1.0) \
            * (0.85000002384185791015625
               + n[..., 0] * 0.100000001490116119384765625)  # :122
        vv = torch.where(a1 > 0.0, vv, a1)                   # :118/:126
        cand = vv * _ss(20.0, 6.0, gage)                     # :128
        outer = torch.maximum(outer, torch.where(
            active, cand, torch.zeros_like(cand)))           # :128 max
    keep = keep & ((outer - 9.9999997473787516355514526367188e-05)
                   >= 0.0)                                   # :133
    rgb = fire_rgb.expand(*outer.shape, 3) * fire_scale      # :137
    return rgb, outer * fire_scale, keep


# --------------------------------------------------------------------
# registry
# --------------------------------------------------------------------
# Resolved from the bytecode, not from the census row. `loops` and
# `findLSB` are counts over the busiest module of the family.
B3_CLASS = {
    "blur":                          ("FILTER", 1, 0),
    "general_filter":                ("FILTER", 1, 0),
    "convolve_environment_map":      ("FILTER", 1, 0),
    "tools_terrain_composite_slope": ("FILTER", 1, 0),
    "outlines":                      ("FILTER", 1, 0),
    "csgo_volume_viewer":            ("FILTER", 1, 0),
    "msaa_resolve":                  ("FILTER", 1, 0),
    "visualize_depth":               ("FILTER", 2, 0),
    "lpv_debug_grid":                ("NO-LOOP", 0, 0),
    "inferno":                       ("BINNER", 2, 2),
}

B3_FAMILIES = {
    "blur": b3_blur,
    "general_filter": b3_general_filter,
    "convolve_environment_map": b3_convolve_environment_map,
    "tools_terrain_composite_slope": b3_terrain_composite_slope,
    "outlines": b3_outlines,
    "csgo_volume_viewer": b3_volume_viewer,
    "msaa_resolve": b3_msaa_resolve,
    "visualize_depth": b3_visualize_depth,
    "lpv_debug_grid": b3_lpv_debug_grid,
    "inferno": b3_inferno,
}
