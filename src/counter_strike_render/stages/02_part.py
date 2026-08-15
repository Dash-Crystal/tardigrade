parser.add_argument("--cs2ao-zprepass-normals", type=int, default=0,
                    choices=(0, 1),
                    help="D_Z_PREPASS_OUTPUTS_NORMALS, ssao.vfx dynamic "
                         "place value 1. 0 (r0m0:38-46) RECONSTRUCTS the "
                         "normal from four neighbour ray-distance taps "
                         "with shortest-edge selection "
                         "-- mix(dx+,dx-, len(dx-)<len(dx+)) -- then "
                         "-normalize(cross(ddx,ddy)). 1 (r0m1:39-43) "
                         "reads it from a texture at set1/binding31 and "
                         "shifts the distance buffer to binding 32.")
parser.add_argument("--cs2ao-read-normal-tex", type=int, default=0,
                    choices=(0, 1),
                    help="D_READ_NORMAL_FROM_TEXTURE, "
                         "ssao_scalable_ambient_obscurance dynamic place "
                         "value 1. 0 (r0m0:29-31) takes the camera-space "
                         "normal as normalize(cross(dFdx(C),dFdy(C))). 1 "
                         "(r0m1:35-37) reads set1/binding31, decodes "
                         "(rgb-0.5)*2, multiplies by the row-major mat4 "
                         "at set1/binding1 offset 128, and NEGATES Y.")
parser.add_argument("--cs2ao-box2x2", type=int, default=0,
                    choices=(0, 1),
                    help="D_2X2_BOX_FILTER, "
                         "ssao_scalable_ambient_obscurance dynamic place "
                         "value 2. Adds the cross-quad 2x2 resolve at "
                         "r0m2:60-79: A -= dFdx(A)*((x&1)-0.5) gated on "
                         "abs(dFdx(z))<1.0, then the same in y. The gate "
                         "is what stops it averaging across a silhouette.")
parser.add_argument("--cs2ao-red-only", type=int, default=0,
                    choices=(0, 1),
                    help="S_OUTPUT_RED_CHANNEL_ONLY, "
                         "ssao_scalable_ambient_obscurance static place "
                         "value 1, 2 static combos. 0 packs the bilateral "
                         "blur's depth key into .y as "
                         "clamp(abs(z)*2e-4,0,1) (r0m0:60); 1 writes 0.0 "
                         "there (r1m0:60) and the blur then has no key. "
                         "Not a quality level -- it changes what the NEXT "
                         "pass can do.")
parser.add_argument("--cs2ao-msaa-depth", type=int, default=0,
                    choices=(0, 1),
                    help="D_MSAA_DEPTH_BUFFER (ssao dynamic place value "
                         "2; also ssao_convert_depth pv 1, denoise_blur "
                         "pv 2, blur_with_depth pv 8). 1 switches the "
                         "depth read from sampler2D+textureLodOffset to "
                         "texture2DMS+texelFetch(...,0) (ssao r0m2:23,35). "
                         "Our depth target is single-sample, so at 1 the "
                         "four OFFSET neighbour taps collapse onto the "
                         "centre texel exactly as they do in the "
                         "reference (r0m2:40-43 fetch _11527 with NO "
                         "offset for all four) -- that collapse is the "
                         "reference's own behaviour, not our "
                         "approximation of it. PRINTED when engaged.")
parser.add_argument("--cs2ao-bilateral", type=int, default=1,
                    choices=(0, 1),
                    help="run ssao_bilateral_blur after the estimator. "
                         "Default 1: the estimator is deliberately "
                         "under-sampled and the blur is a separate shader "
                         "in CS2 precisely because it materially changes "
                         "the result, so a 0 default would report the "
                         "subsystem as run while omitting a stage.")
parser.add_argument("--cs2ao-radius", type=float, default=1.0,
                    help="SAO world-space obscurance radius, metres. Sets "
                         "the reference's _m2.w (projScale*r), _m3.x "
                         "(1/r^2) and _m2.x (r, the estimator's "
                         "normaliser at r0m0:60).")
parser.add_argument("--cs2ao-bias", type=float, default=0.01,
                    help="SAO _5618._m4, the metric bias subtracted from "
                         "dot(v,n) at r0m0:54.")
parser.add_argument("--cs2ao-intensity", type=float, default=1.0,
                    help="SAO _5618._m5, the numerator of the obscurance "
                         "normalisation at r0m0:60.")
parser.add_argument("--cs2ao-phase", type=float, default=0.0,
                    help="SAO _5618._m6, the per-frame spiral phase "
                         "added to the hash at r0m0:28. The hash itself "
                         "is float((3*x)^(y+x*y))*8.0 -- note 8.0, where "
                         "the published paper and this file's previous "
                         "fitted stand-in both use 10.0.")
parser.add_argument("--cs2ao-normal-bias", type=float, default=0.02,
                    help="ssao.vfx _5618._m1: the sample origin is "
                         "P + N*(rayDistance*this), i.e. the normal "
                         "offset scales with DISTANCE (r0m0:47), which is "
                         "why it does not need a per-pixel depth-slope "
                         "term.")
parser.add_argument("--aoproxy-splat", type=int, default=1,
                    choices=(0, 1),
                    help="run aoproxy_splat as the offset-76 producer. "
                         "THIS SETTLES AN OPEN QUESTION: "
                         "PER_VIEW_RENDER_TARGETS.md records that the "
                         "producers of offsets 76/80 'are not among the "
                         "six extracted shaders, so their algorithms are "
                         "unknown', and dirocc_target() in this file says "
                         "the same. aoproxy_splat IS that producer -- it "
                         "writes a 4-channel target, one channel per "
                         "engine ambient-basis direction, from "
                         "_4083._m0._m0[0..3] (r0m0:74), which is the "
                         "same four-vector array the offset-76 consumer "
                         "resolves against. Default 1 so the transcribed "
                         "producer, not the fitted stand-in, is what "
                         "runs.")
parser.add_argument("--aoproxy-reverse-depth", type=int, default=0,
                    choices=(0, 1),
                    help="S_REVERSE_DEPTH_BUFFER, aoproxy_splat static "
                         "place value 1, 2 static combos. 0 (r0m0:49) "
                         "reconstructs 1/((d'*_m0.z + _m0.w)*dot(m4,ray)); "
                         "1 (r1m0:49) reconstructs _m2/(d'*dot(m4,ray)) "
                         "-- a different expression, not a sign flip.")
parser.add_argument("--aoproxy-proxies", default=None,
                    help="proxy-volume JSON: {'spheres':[{'xform':4x3, "
                         "...}], 'boxes':[{'xform':4x3,'ext':[x,y,z]}]}. "
                         "de_inferno ships no extracted proxy set, so "
                         "with no file the reachability self-test "
                         "SYNTHESISES proxies from the scene bounds and "
                         "says so. The mat4x3 array is dimensioned [106] "
                         "in the bytecode (r0m0:11), which bounds the "
                         "proxy count per draw.")
parser.add_argument("--atrous-passes", type=int, default=3,
                    help="how many atrous_filter iterations to run. Each "
                         "pass doubles the tap stride _5618._m1 "
                         "(r0m0:101), which is what makes it a-trous.")
parser.add_argument("--atrous-sigma-z", type=float, default=1.0,
                    help="atrous_filter _5618._m2, the depth-gradient "
                         "scale at r0m0:74. The full denominator is "
                         "m2*clamp(gradZ,0.01,1000)*stepSize*|offset|.")
parser.add_argument("--atrous-sigma-l", type=float, default=4.0,
                    help="atrous_filter _5618._m4, multiplying "
                         "sqrt(prefiltered variance) in the luminance "
                         "weight at r0m0:73.")
parser.add_argument("--atrous-sigma-n", type=float, default=128.0,
                    help="atrous_filter _5618._m3, the exponent of "
                         "pow(clamp(dot(n_c,n_t),0,1), this) at "
                         "r0m0:145.")
parser.add_argument("--atrous-first-pass", type=int, default=0,
                    choices=(0, 1),
                    help="D_FIRST_PASS, atrous_filter dynamic place "
                         "value 1.")
parser.add_argument("--atrous-final-pass", type=int, default=0,
                    choices=(0, 1),
                    help="D_FINAL_PASS, atrous_filter dynamic place "
                         "value 2.")
parser.add_argument("--atrous-baked-shadows", type=int, default=0,
                    choices=(0, 1),
                    help="D_BAKED_SHADOWS, atrous_filter dynamic place "
                         "value 4.")
parser.add_argument("--atrous-baked-ao", type=int, default=0,
                    choices=(0, 1),
                    help="D_BAKED_AO, atrous_filter dynamic place value "
                         "8. atrous_filter is the only family in this "
                         "slice with an EMPTY static array and exactly "
                         "one static combo -- the dynamic-only shape the "
                         "smoke/MBOIT document found. Four of my nine "
                         "have it; five do NOT.")
parser.add_argument("--dnb-kernel", type=int, default=2, choices=(0, 1, 2),
                    help="S_BOX_KERNEL, denoise_blur static place value "
                         "1, m_nMax 2, 3 static combos in 3 records. 0 is "
                         "a PURE PASSTHROUGH -- texelFetch and write, no "
                         "loop at all (r0m0:10). 1 is 5 taps at "
                         "{-6,-3,0,3,6} (r1m0:16). 2 is 7 taps at "
                         "{-9,-6,-3,0,3,6,9} (r2m0:16). Default 2 = the "
                         "widest; note that 0 being a no-op means a "
                         "0 default would be a path that quietly does "
                         "nothing.")
parser.add_argument("--dnb-blur-y", type=int, default=0, choices=(0, 1),
                    help="D_BLUR_Y, denoise_blur dynamic place value 1. "
                         "0 offsets in x (r1m0:60), 1 in y (r1m1:60).")
parser.add_argument("--dnb-depth-sigma", type=float, default=2.0,
                    help="denoise_blur _5618._m1: w = 1 - clamp(|z_t - "
                         "z_c|/this, 0, 1) at r1m0:62.")
parser.add_argument("--bwd-kernel", type=int, default=4,
                    choices=(0, 1, 2, 3, 4),
                    help="S_GAUSSIAN_KERNEL, blur_with_depth static place "
                         "value 1, m_nMax 4, and with S_SRGB_READ (place "
                         "value 5) that is 10 static combos -- MIXED "
                         "RADIX, stride 5, exactly the vcs/README.md trap. "
                         "0 has no kernel table at all; 1 and 2 are 5-tap "
                         "(r1m0:16, r2m0:16 -- DIFFERENT offsets and "
                         "weights, not a rescale); 3 is 7-tap (r3m0:16); "
                         "4 is 13-tap (r4m0:16). Offsets are fractional "
                         "because each tap is a bilinear-weighted PAIR.")
parser.add_argument("--bwd-srgb-read", type=int, default=0, choices=(0, 1),
                    help="S_SRGB_READ, blur_with_depth static place value "
                         "5.")
parser.add_argument("--bwd-blur-y", type=int, default=0, choices=(0, 1),
                    help="D_BLUR_Y, blur_with_depth dynamic place value "
                         "1: offset along x (r1m0:70) or y (r1m1:70).")
parser.add_argument("--bwd-disable-alpha-write", type=int, default=0,
                    choices=(0, 1),
                    help="D_DISABLE_ALPHA_WRITE, blur_with_depth dynamic "
                         "place value 2.")
parser.add_argument("--bwd-half-res", type=int, default=0, choices=(0, 1),
                    help="D_USE_HALF_RES, blur_with_depth dynamic place "
                         "value 4.")
parser.add_argument("--bwd-plane", default="0,1,0,0",
                    help="blur_with_depth _5618._m4, the world-space "
                         "PLANE the blur keys on. The key is not depth "
                         "directly: r1m0:53-54 unprojects NDC+linear "
                         "depth by the inverse view-projection at "
                         "_5618._m5 and evaluates "
                         "clamp(m3.x*dot(plane, vec4(P,1)) + m3.y, 0, 1) "
                         "* m3.z. STRING-VALUED (four comma-separated "
                         "floats): never tested for truthiness.")
parser.add_argument("--bwd-plane-scale", type=float, default=0.02,
                    help="blur_with_depth _5618._m3.x.")
parser.add_argument("--bwd-plane-bias", type=float, default=0.0,
                    help="blur_with_depth _5618._m3.y.")
parser.add_argument("--cs2filt-selftest", choices=("on", "off"), default="on",
                    help="direct-call reachability proof for all seven "
                         "families at every value of every combo axis "
                         "they declare, on synthesised input. Prints per "
                         "family and per axis value how many pixels "
                         "executed and the max|delta| against the "
                         "neighbouring value, so an axis that changes "
                         "nothing shows up as a zero rather than as "
                         "silence. STRING-VALUED -- 'off' is truthy, so "
                         "this is compared to 'on', never tested bare.")
parser.add_argument("--cs2filt-selftest-inject", default=None,
                    help="break one named path on purpose so the "
                         "self-test can be made to FAIL. Names: "
                         "sao-spiral-frozen, sao-no-pow, "
                         "bilateral-ignores-key, ssao-kernel-single, "
                         "atrous-no-stride, dnb-kernel-flat, "
                         "bwd-kernel-flat, aoproxy-no-boxes, "
                         "combo-bitmask. The last decodes "
                         "blur_with_depth's static combo with & instead "
                         "of mixed-radix division, which must move the "
                         "reported ids because S_SRGB_READ sits at "
                         "stride 5. STRING-VALUED.")
# ------------------- END APPEND-ONLY BLOCK (impl-ssao-filters) ----------


def _parser_conflict_selfcheck(verbose=False):
    """Replay every add_argument against a FRESH ArgumentParser.

    A P0 shipped on 2026-08-06 (fixed in efb652a): two agents added
    `--secondary-uv` in different files, git merged both cleanly, and
    argparse raised ArgumentError at IMPORT time -- so the renderer could
    not build its parser for ANY invocation, not just the one that used
    the flag. Reading the diff does not catch that; only replaying the
    parser does.

    The replay is over `parser._actions`, which is the parser argparse
    actually built, so it cannot drift from the definitions the way a
    hand-maintained list of flag names would.

    CAN THIS CHECK FAIL? Yes, and it was made to. Run with
    IJI_PARSER_SELFCHECK_INJECT=--fam-side in the environment: that
    re-registers an option string that already exists, the replay raises
    argparse.ArgumentError, and this function reports 1 conflict and
    exits non-zero. Without the injection it reports 0 over
    len(parser._actions) actions. A check that has never been made to
    fail is CHECKS_THAT_CANNOT_FAIL.md's whole subject.
    """
    import argparse as _ap
    probe = _ap.ArgumentParser(add_help=False)
    seen, conflicts, n = {}, [], 0
    inject = os.environ.get("IJI_PARSER_SELFCHECK_INJECT", "")
    for a in parser._actions:
        if not a.option_strings:
            continue
        n += 1
        kw = {}
        if a.nargs == 0:
            kw["action"] = ("store_true" if a.const is True
                            else "store_false" if a.const is False
                            else "store_const")
            if kw["action"] == "store_const":
                kw["const"] = a.const
        else:
            kw["nargs"] = a.nargs
            kw["type"] = a.type
            kw["choices"] = a.choices
        kw["default"] = a.default
        kw["dest"] = a.dest
        try:
            probe.add_argument(*a.option_strings, **kw)
        except _ap.ArgumentError as e:
            conflicts.append((a.option_strings, str(e)))
        for s in a.option_strings:
            if s in seen and seen[s] is not a:
                conflicts.append((s, f"also defined for --{seen[s].dest}"))
            seen[s] = a
    if inject:
        try:
            probe.add_argument(inject, action="store_true",
                               dest="_selfcheck_injected")
        except _ap.ArgumentError as e:
            conflicts.append((inject, "INJECTED: " + str(e)))
    print(f"parser self-check: {n} option actions replayed, "
          f"{len(conflicts)} conflicts", flush=True)
    for c in conflicts:
        print(f"  CONFLICT {c[0]}: {c[1]}", flush=True)
    if conflicts:
        raise SystemExit("parser has conflicting option strings; the "
                         "renderer would fail to build its parser for "
                         "EVERY invocation")
    if verbose:
        for s in sorted(seen):
            print("   ", s, flush=True)
    return n


# Run BEFORE parse_args: the whole point is to prove the parser can be
# BUILT, which must be answerable without supplying --world/--out/ticks.
if os.environ.get("IJI_PARSER_SELFCHECK") or \
        os.environ.get("IJI_PARSER_SELFCHECK_INJECT"):
    _parser_conflict_selfcheck(
        verbose=os.environ.get("IJI_PARSER_SELFCHECK") == "list")
    raise SystemExit(0)

# --- the volumetric smoke subsystem and the MBOIT RESOLVE ---------------
# Seven families -- smoke_volume, smoke_volume_depth, smoke_volume_mask,
# write_smoke_depth_water_reflection, overlay_smoke, mboit_mixed_combine,
# mboitfinal -- all of them DYNAMIC-ONLY (one static combo, empty static
# array, every axis selected per draw). They are PASSES the engine
# schedules, not materials, so they are not behind a material-family
# dispatch: `smoke_mboit.smoke_pass()` is called from the render path.
#
# Every flag it declares is namespaced --smoke-* or --mboit-*, and
# `iji_model/counter_strike_render/parser_replay.py` replays this whole parser plus
# that module's block against a real ArgumentParser and reports the
# option-string conflict count.
import smoke_mboit                                            # noqa: E402
from passes import lighting as _lighting                      # noqa: E402
smoke_mboit.add_arguments(parser)

# =======================================================================
# THE FOUR MODULES FROM THE LAST SWEEP THAT WERE WRITTEN AND NEVER
# IMPORTED. ~6,100 lines of transcribed shader code that no invocation
# could reach, which is the "present, resident and unreachable" shape
# this project keeps finding. Imported and wired here so they RUN.
#
# They carry no add_arguments()/configure() hooks -- they are plain
# function libraries -- so their flags are declared here.
import dgb_passes                                              # noqa: E402
import cs2_depth_msaa_stage as _msaa                           # noqa: E402
import fam_loopfam_b as _lfb                                   # noqa: E402
import sf_batch3_filters as _b3                                # noqa: E402

# Reach counters for the four. A module that runs and reports nothing is
# indistinguishable from one that never ran, which is the whole failure
# class this project has been cataloguing.
MSAA_STATE = {"built": 0}
GBUF_STATE = {"packed": 0}
B3_STATE = {"ran": 0}
LFB_STATE = {}


def _screen_uv_like(t):
    """(...,H,W,C) -> (...,H,W,2) normalised screen UV, on t's device."""
    h, w = t.shape[-3], t.shape[-2]
    ys = torch.arange(h, device=t.device, dtype=torch.float32)
    xs = torch.arange(w, device=t.device, dtype=torch.float32)
    gy, gx = torch.meshgrid(ys, xs, indexing="ij")
    uv = torch.stack([(gx + 0.5) / w, (gy + 0.5) / h], dim=-1)
    return uv.expand(*t.shape[:-3], h, w, 2)


def _mod_err(exc):
    """One-line failure summary for a newly-wired module, WITH the site.

    `repr(exc)[:120]` was hiding the only part that identifies the bug:
    a TypeError's message names a type but never says which call
    produced it, and the truncation cut the rest. The innermost frame is
    the whole diagnosis -- with it, "can't convert cuda:0 tensor to
    numpy" becomes a named line in a named file.
    """
    _tb = traceback.extract_tb(exc.__traceback__)
    # The DEEPEST frame in our own code, not the deepest frame overall.
    # These failures surface inside torch, so the innermost frame reads
    # "_tensor.py:1253 in __array__()" -- true, and useless: it names the
    # conversion, never the call that asked for it. The last frame
    # outside site-packages is the line a person can actually go and fix.
    _ours = [f for f in _tb if "site-packages" not in f.filename]
    _f = (_ours or _tb or [None])[-1]
    _at = (f"  <- {os.path.basename(_f.filename)}:{_f.lineno} "
           f"in {_f.name}()" if _f else "")
    return f"{exc!r}"[:400] + _at


# G-buffer inputs the forward path does not separately carry. They are
# filled by the shading pass where it has them and left as neutral
# constants where it does not -- stated, not silently zero.
_GB_NRM = _GB_ROUGH = _GB_METAL = _GB_AO = None

# --lfb-families is already declared by the loop-family block below;
# the parser replay caught the duplicate the moment it was added, which
# is the check that a P0 shipped for want of earlier today.
parser.add_argument("--dgb-gbuffer", type=int, default=1,
                    help="pack the deferred G-buffer (albedo / world "
                         "normal / depth / lighting-terms) alongside the "
                         "forward frame. The reference is a 4-attachment "
                         "MRT pass; this renderer is forward, so the "
                         "targets are built and carried, not consumed.")
parser.add_argument("--msaa-samples", type=int, default=4,
                    help="unresolved multisample depth. >1 builds the "
                         "per-sample attachment sampler2DMS needs -- the "
                         "one memory root class a resolve-first renderer "
                         "structurally cannot express. Defaults to 4, NOT "
                         "1: the block is gated on >1, so a default of 1 "
                         "left the module resident and unreachable at "
                         "every default invocation, which is the state "
                         "this renderer is explicitly not meant to be in. "
                         "Set 1 to switch the attachment off.")
parser.add_argument("--b3-filters", type=int, default=1,
                    help="batch-3 screen-space filter kernels (blur, "
                         "general convolve, outlines, terrain slope, "
                         "volume viewer). Fixed-tap kernels, NOT cluster "
                         "loops -- 25 of the 41 loop families are this "
                         "shape and writing a binner for them is the "
                         "wrong construct.")
# =======================================================================
# The five loop-carrying families of batch B -- csgo_textile_layer,
# cables, csgo_simple_2way_blend, csgo_water, grasstile.
#
# EVERY option string below is prefixed `--lfb-`. Two agents each declared
# `--secondary-uv` on 2026-08-07, git merged both cleanly, and argparse
# raised at import so the renderer could not build its parser for ANY
# invocation. Namespacing makes that collision impossible to reach by
# accident, and `check_parser_flags.py` replays every add_argument in this
# file against a real ArgumentParser so it cannot happen silently again.
#
# Each static axis is an explicit int with choices (0, 1) -- never a
# store_true -- because the SHIPPED default of every one of these axes is
# its m_nMin = 0, and a store_true would make "0" unexpressible while
# reading as if the feature were merely off. The defaults below follow the
# shader; both values of every axis are implemented, and
# `--lfb-reach` executes every axis at BOTH values so that following the
# shader's defaults can never leave a path unreachable.
#
# NOTE for consumers: --lfb-families is STRING-valued and its "off" is
# TRUTHY. Test it with `!= "off"`, never with bare `if`.
parser.add_argument("--lfb-families", default="off",
                    help="which of the five batch-B loop families to shade: "
                         "a comma list of textile,cables,twoway,water,grass "
                         "or 'all', or 'off'. STRING-VALUED: \"off\" is "
                         "truthy, so consumers must compare != \"off\".")
parser.add_argument("--lfb-reach", action="store_true",
                    help="DIAGNOSTIC + reachability proof: run every one of "
                         "the five families' entry points over a synthesised "
                         "surface at EVERY value of EVERY combo axis, report "
                         "which executed and max|delta| against the axis "
                         "suppressed, and exit non-zero if any axis moves "
                         "nothing. de_inferno has ZERO materials in all five "
                         "families (663/663 decompiled, a self-summing "
                         "partition), so without this the entry points would "
                         "be defined and never called.")
parser.add_argument("--lfb-reach-inject", default=None,
                    help="break one named path so --lfb-reach can be made to "
                         "FAIL ON PURPOSE. Names: water-fresnel-flat, "
                         "water-flow-noop, cables-tint-noop, "
                         "twoway-blend-frozen, grass-a2c-noop, "
                         "textile-aniso-isotropic. 'all' lists them, exits.")
# --- csgo_water: 9 static axes, 200 combos, 192 records -----------------
parser.add_argument("--lfb-water-flow-normals", type=int, choices=(0, 1),
                    default=1,
                    help="S_FLOW_NORMALS. 1 = the two-phase flow-advected "
                         "normal cycle (flow_normals1_c1.glsl:303); 0 = "
                         "three scrolling taps averaged by a literal "
                         "0.33000001311302185 (flow_normals0_c0.glsl:303). "
                         "DEFAULT 1, against the shader's m_nMin of 0, "
                         "because value 0 reads three UV sets our vertex "
                         "stage does not emit -- see the SUBST note.")
parser.add_argument("--lfb-water-fresnel", type=int, choices=(0, 1),
                    default=0,
                    help="S_FRESNEL. 0 = reflection weight is the constant "
                         "g_flReflectance; 1 = Schlick, F0 + (1-F0)*"
                         "(1-saturate(dot(-V,N)))^5 (fresnel1_c8.glsl:327 "
                         "against fresnel0_c0.glsl:325).")
parser.add_argument("--lfb-water-lightmap-fog", type=int, choices=(0, 1),
                    default=0,
                    help="S_LIGHTMAP_WATER_FOG. 1 multiplies the deep "
                         "colour g_vWaterFogColor by the surface's own "
                         "indirect diffuse (lightmap_water_fog1_c16.glsl"
                         ":856); 0 leaves it a flat constant. Costs "
                         "560->1352 FLOPs and 8->20 fetch sites.")
parser.add_argument("--lfb-water-disable-refraction", type=int,
                    choices=(0, 1), default=0,
                    help="S_DISABLE_REFRACTION. 1 removes the scene-colour "
                         "read, the scene-DEPTH read and the whole depth "
                         "fade; base colour becomes flat g_vRefractionTint "
                         "and the reflection loses both the shoreline gate "
                         "and the clamp(2*fade) factor -- the module has a "
                         "literal 1.0 there "
                         "(disable_refraction1_c32.glsl:302-304).")
parser.add_argument("--lfb-water-animated-normals", type=int, choices=(0, 1),
                    default=0,
                    help="S_ANIMATED_NORMALS. 1 turns g_tNormal from a "
                         "texture2D into a texture2DArray and lerps every "
                         "normal tap between array layers floor/ceil of "
                         "(time+g_flNormalMapAnimationTimeOffset)/"
                         "g_flNormalMapAnimationTimePerFrame, both wrapped "
                         "by x-depth*trunc(x/depth) "
                         "(animated_normals1_c64.glsl:307-320).")
parser.add_argument("--lfb-water-flow-color", type=int, choices=(0, 1),
                    default=0,
                    help="S_FLOW_COLOR. 1 adds a second two-phase cycle on "
                         "g_tColor (its own period/scale/scroll, UV scaled "
                         "0.235 AFTER the offsets, weights CROSS-assigned "
                         "and SUMMED) whose alpha is a foam mask -- AND "
                         "changes the reflection composite from ADD to MIX "
                         "(flow_color1_c129.glsl against ...0_c1.glsl).")
parser.add_argument("--lfb-water-quality", type=int, choices=(0, 1),
                    default=0,
                    help="S_SHADER_QUALITY. 0 = ONE cube-array tap at layer "
                         "literal 0 (hader_quality0_c0.glsl:325); 1 = the "
                         "accumulated parallax-corrected per-probe IBL loop "
                         "normalised by its own weight "
                         "(hader_quality1_c256.glsl:448). Same fetch-SITE "
                         "count (8), 560->707 FLOPs. Both implemented.")
parser.add_argument("--lfb-water-tools-vis", type=int, choices=(0, 1),
                    default=0,
                    help="S_MODE_TOOLS_VIS (csgo_water). 1 overrides the "
                         "output with the tools visualiser.")
parser.add_argument("--lfb-water-mode-depth", type=int, choices=(0, 1),
                    default=0,
                    help="S_MODE_DEPTH (csgo_water). RECORDED FACT: of the "
                         "200 shipped combos, only S_MODE_DEPTH=0 has a "
                         "bytecode record -- the depth-only variant ships "
                         "no pixel module in this family.")
# --- csgo_textile_layer: 3 static axes, 6 combos, 6 records -------------
parser.add_argument("--lfb-textile-backwards-compat", type=int,
                    choices=(0, 1), default=0,
                    help="S_BACKWARDS_COMPATIBILITY (csgo_textile_layer).")
parser.add_argument("--lfb-textile-aniso-gloss", type=int, choices=(0, 1),
                    default=0,
                    help="S_ANISOTROPIC_GLOSS (csgo_textile_layer). Splits "
                         "the specular roughness along the fabric tangent. "
                         "2586->2721 FLOPs at equal fetch count.")
parser.add_argument("--lfb-textile-tools-vis", type=int, choices=(0, 1),
                    default=0, help="S_MODE_TOOLS_VIS (csgo_textile_layer).")
parser.add_argument("--lfb-textile-cube-static", type=int, choices=(0, 1),
                    default=1,
                    help="D_SPECULAR_CUBE_MAP_STATIC (csgo_textile_layer), "
                         "a DYNAMIC axis.")
# --- cables: 8 static axes, 40 combos, 15 records -----------------------
parser.add_argument("--lfb-cables-clamp-min-radius", type=int, choices=(0, 1),
                    default=0,
                    help="S_CLAMP_MIN_RADIUS (cables). RECORDED FACT: all "
                         "16 live pairs share one bytecode record -- this "
                         "axis changes NO pixel bytecode, it is a "
                         "vertex-stage axis. The flag exists so the fact is "
                         "expressible, and it is a no-op in the PS by "
                         "MEASUREMENT, not by omission.")
parser.add_argument("--lfb-cables-mode-depth", type=int, choices=(0, 1),
                    default=0,
                    help="S_MODE_DEPTH (cables). 1 is a 0-FLOP, 0-fetch "
                         "empty pixel shader.")
parser.add_argument("--lfb-cables-wireframe", type=int, choices=(0, 1),
                    default=0,
                    help="S_MODE_TOOLS_WIREFRAME (cables). 1847->278 FLOPs, "
                         "43->0 fetch sites.")
parser.add_argument("--lfb-cables-shading-complexity", type=int,
                    choices=(0, 1), default=0,
                    help="S_MODE_TOOLS_SHADING_COMPLEXITY (cables). "
                         "1847->4 FLOPs, 43->0 fetch sites.")
parser.add_argument("--lfb-cables-translucent", type=int, choices=(0, 1),
                    default=0, help="S_TRANSLUCENT (cables).")
parser.add_argument("--lfb-cables-alpha-test", type=int, choices=(0, 1),
                    default=0, help="S_ALPHA_TEST (cables).")
parser.add_argument("--lfb-cables-tint-mask", type=int, choices=(0, 1),
                    default=0, help="S_TINT_MASK (cables), g_tTintMask.")
parser.add_argument("--lfb-cables-tools-vis", type=int, choices=(0, 1),
                    default=0, help="S_MODE_TOOLS_VIS (cables).")
parser.add_argument("--lfb-cables-quad-overdraw", type=int, choices=(0, 1),
                    default=0,
                    help="D_QUAD_OVERDRAW (cables), a DYNAMIC axis unique "
                         "to this family among the five.")
# --- csgo_simple_2way_blend: 4 static axes, 12 combos, 12 records -------
parser.add_argument("--lfb-twoway-tint-masks", type=int, choices=(0, 1),
                    default=0,
                    help="S_ENABLE_TINT_MASKS (csgo_simple_2way_blend). A "
                         "DIFFERENT family from csgo_environment_blend and "
                         "from csgo_simple_3layer_parallax; transcribed "
                         "from its own bytecode.")
parser.add_argument("--lfb-twoway-quality", type=int, choices=(0, 1),
                    default=0,
                    help="S_SHADER_QUALITY (csgo_simple_2way_blend). "
                         "1545->1885 FLOPs and 19->45 fetch sites -- the "
                         "largest fetch jump any quality axis makes in the "
                         "five. Both implemented.")
parser.add_argument("--lfb-twoway-tools-vis", type=int, choices=(0, 1),
                    default=0,
                    help="S_MODE_TOOLS_VIS (csgo_simple_2way_blend).")
parser.add_argument("--lfb-twoway-ao", type=int, choices=(0, 1), default=0,
                    help="S_ENABLE_AO (csgo_simple_2way_blend).")
parser.add_argument("--lfb-twoway-opaque-fade", type=int, choices=(0, 1),
                    default=0,
                    help="D_OPAQUE_FADE (csgo_simple_2way_blend), DYNAMIC.")
parser.add_argument("--lfb-twoway-cube-static", type=int, choices=(0, 1),
                    default=1,
                    help="D_SPECULAR_CUBE_MAP_STATIC "
                         "(csgo_simple_2way_blend), DYNAMIC.")
# --- grasstile: 4 static axes, 16 combos, 5 records ---------------------
parser.add_argument("--lfb-grass-tools-vis", type=int, choices=(0, 1),
                    default=0, help="S_MODE_TOOLS_VIS (grasstile).")
parser.add_argument("--lfb-grass-wireframe", type=int, choices=(0, 1),
                    default=0,
                    help="S_MODE_TOOLS_WIREFRAME (grasstile). 1 collapses "
                         "to a 0-FLOP, 0-fetch module.")
parser.add_argument("--lfb-grass-shading-complexity", type=int,
                    choices=(0, 1), default=0,
                    help="S_MODE_TOOLS_SHADING_COMPLEXITY (grasstile). "
                         "RECORDED FACT: unlike cables', this axis changes "
                         "NO grasstile pixel bytecode -- 8 live pairs, 0 "
                         "distinct records. The same axis name is a real "
                         "code change in one family and a dedup in another.")
parser.add_argument("--lfb-grass-alpha-to-coverage", type=int, choices=(0, 1),
                    default=0,
                    help="S_ALPHA_TO_COVERAGE (grasstile), a STATIC axis "
                         "distinct from the DYNAMIC D_ALPHA_TEST below.")
parser.add_argument("--lfb-grass-alpha-test", type=int, choices=(0, 1),
                    default=0,
                    help="D_ALPHA_TEST (grasstile), DYNAMIC. Two different "
                         "alpha mechanisms; both implemented.")

# ======================================================================
# csgo_weapon (csgo archive) and csgo_legs_prepass (csgo_core archive)
# ======================================================================
# Two families that NO map-material scan can reach: de_inferno's world
# materials contain no weapon and no first-person legs, so absence here is
# a USAGE fact about one map, not a capability fact about the renderer.
# Reachability is therefore demonstrated with --viewmodel-synth, which
# submits a synthesised draw through the same dispatch a packed weapon
# would take.
#
# THE PASS STRUCTURE IS THE POINT.  A first-person viewmodel is not a
# model placed near the camera: it is its own pass, with its own FOV and
# its own depth range, composited over the world so it cannot clip into
# world geometry.  csgo_legs_prepass existing as a separate single-combo
# shader carrying its own g_flFirstpersonLegsFade is the direct evidence.
# --viewmodel-depth-range keeps the collapsed form available as a
# DIAGNOSTIC so the separation can be measured, never as the default.
#
# Every flag below is namespaced --weapon-* / --viewmodel-*; no flag here
# is a bare word that another family could plausibly also want.  Every
# string-valued flag is compared against its own literal, never tested for
# truthiness -- "off" is truthy.
parser.add_argument("--weapon", action="store_true",
                    help="csgo_weapon (csgo/shaders_vulkan_dir.vpk), 180 "
                         "materials, 672 static combos over 12 binary "
                         "axes x 11,520 shipped (static, dynamic) pairs, "
                         "2,326 FLOPs/px and 44 fetch sites at the first "
                         "live combo. Shades the viewmodel/world-model "
                         "weapon skin inside the VIEWMODEL PASS; needs "
                         "--viewmodel. Transcribed from "
                         "csgo_weapon__enable_adjustments0_c0.glsl (record "
                         "0 of the store) and the per-axis modules named "
                         "on each axis flag below.")
parser.add_argument("--weapon-adjustments", action="store_true",
                    help="S_ENABLE_ADJUSTMENTS (stride 1): the two "
                         "view-angle colour adjustments that ride on the "
                         "mask texture's .z channel -- a hue ROTATION "
                         "about the (0.577,0.577,0.577) axis by "
                         "g_flHueShift*(1-dot(V,N))*mask.z, blended by "
                         "pow(saturation,0.125) (baseline glsl:396-412), "
                         "and a 6-segment rainbow ramp indexed by "
                         "fract((dot(V,N)+dot(V,sunDir))*k+phase)*6 "
                         "(baseline glsl:464-519). BOTH are compiled into "
                         "every module and gated by their own uniform, so "
                         "the static axis does not change the bytecode.")
parser.add_argument("--weapon-tint-mask", action="store_true",
                    help="S_TINT_MASK (stride 32, record 12): albedo "
                         "becomes mix(C, C*vColorTint, tintMask.x) instead "
                         "of the baseline's unconditional C*vColorTint "
                         "(csgo_weapon__tint_mask1_c32.glsl:310-313).")
parser.add_argument("--weapon-self-illum", action="store_true",
                    help="S_SELF_ILLUM (stride 256, record 48): an "
                         "emissive map sampled at uv + fract(scroll*time) "
                         "and multiplied by g_vSelfIllumTint, added to the "
                         "output "
                         "(csgo_weapon__self_illum1_c256.glsl:371-372).")
parser.add_argument("--weapon-glitter", action="store_true",
                    help="S_GLITTER (stride 64, record 24): the candy/"
                         "sparkle flake layer. Decodes a second normal "
                         "map at a 1.75x (2.5x under the per-view flag) "
                         "UV scale, reflects the view vector through it, "
                         "takes sin(R*mix(12,5.6,spread)) and thresholds "
                         "at mix(0.99,0.80,spread); perturbs albedo, "
                         "roughness, metalness and the tangent normal, and "
                         "adds its own radiance term "
                         "(csgo_weapon__glitter1_c64.glsl:424-466, "
                         "composited at :1305 and :1317-1321).")
parser.add_argument("--weapon-sfx-mask", action="store_true",
                    help="S_ENABLE_SFX_MASK (stride 128). MEASURED: combo "
                         "128 and combo 0 share bytecode record 0 "
                         "BYTE-FOR-BYTE, so this static axis changes no "
                         "pixel-shader instruction -- the whole effect is "
                         "compiled into every module and gated by "
                         "g_flSfxAmount*perView, and the axis only selects "
                         "the material binding. The path is the scrolling "
                         "flow-map UV distortion at "
                         "csgo_weapon__enable_adjustments0_c0.glsl:240-276 "
                         "and its albedo/roughness/normal blend at "
                         ":372-393.")
parser.add_argument("--weapon-stickers", action="store_true",
                    help="S_STICKERS (stride 512, record 96): 3,565 lines "
                         "of added pixel shader over the baseline, five "
                         "independent sticker slots at a 27-uniform "
                         "stride (bases m28/m55/m82/m109/m136 in the "
                         "flattened material CB), 81 fetch call-sites. "
                         "Per slot: rotate+scale the sticker UV set, "
                         "reject outside [0,1], alpha = "
                         "clamp(tex.w*12.75,0,1)*wearMask, an LOD>3 "
                         "parallax-offset second tap, a sticker normal "
                         "blended as normalize(baseN + stickerN*a*2), "
                         "roughness min(pow(baseAO,0.75), a*0.5+0.5), and "
                         "an optional holo-foil glitter kernel "
                         "(csgo_weapon__stickers1_c512.glsl:571-4135).")
parser.add_argument("--weapon-sticker-slots", type=int, default=5,
                    help="how many of the five shipped sticker slots to "
                         "evaluate. 5 is the shipped count; lower values "
                         "are a DIAGNOSTIC for isolating one slot, not a "
                         "quality setting.")
parser.add_argument("--weapon-opaque-refract", action="store_true",
                    help="S_OPAQUE_REFRACT (stride 2048, record 218). "
                         "MEASURED: record 218's SPIR-V is byte-identical "
                         "to record 0's (md5 994555a5cd4ff5be... on both), "
                         "so the static axis alone emits no instruction; "
                         "the refraction only appears with "
                         "--weapon-bgrefract, which is the dynamic axis "
                         "D_USE_BGREFRACT. This flag selects the material "
                         "side of the pair.")
parser.add_argument("--weapon-bgrefract", default="off",
                    choices=("off", "sharp", "blur"),
                    help="D_USE_BGREFRACT (dynamic stride 32). Adds "
                         "exactly 6 fetch sites over the baseline module: "
                         "5 on the background render target and 1 on a "
                         "refraction mask. uv = FragCoord*invViewport + "
                         "vec2(dot(cross(camFwd,camUp),N), dot(camUp,N))*"
                         "g_flRefractScale; 'blur' additionally averages "
                         "the centre tap with four corner taps at "
                         "+-invViewport*blur*4 and multiplies by 0.2 "
                         "(csgo_weapon__DYN_use_bgrefract1_d32.glsl:"
                         "4409-4426). Per SHADER_CALLFLOW_nonworld_"
                         "families.md section 4 this is an ENGINE RENDER "
                         "TARGET read through a bindless 2D handle from a "
                         "per-view CB -- NOT a scene-colour sampler, and "
                         "no new memory root class.")
parser.add_argument("--weapon-alpha-test", action="store_true",
                    help="S_ALPHA_TEST (stride 4, record 4): the output "
                         "alpha becomes a computed cutout value instead of "
                         "the interpolated vertex fade "
                         "(csgo_weapon__alpha_test1_c4.glsl, tail).")
parser.add_argument("--weapon-translucent", action="store_true",
                    help="S_TRANSLUCENT (stride 8, record 8).")
parser.add_argument("--weapon-additive-blend", action="store_true",
                    help="S_ADDITIVE_BLEND (stride 16). MEASURED: combo "
                         "16 alone ships NO module; the lowest shipped "
                         "combo carrying it is 24 = 8+16, so additive "
                         "implies translucent in the shipped set. It "
                         "multiplies the output alpha by the fog "
                         "attenuation, because an additive blend that "
                         "ignores fog never fades "
                         "(csgo_weapon__additive_blend1_c24.glsl).")
parser.add_argument("--weapon-mode-depth", action="store_true",
                    help="S_MODE_DEPTH (stride 1024, record 192): a "
                         "26-line depth-only module. Screen-door discards "
                         "on fma(vertexFade,2,-1.5)+dither.y < 0 and "
                         "writes vec4(0,0,0,1) "
                         "(csgo_weapon__mode_depth1_c1024.glsl, complete). "
                         "Replaces the beauty shade, exactly as the combo "
                         "does in the engine.")
parser.add_argument("--weapon-tools-vis", action="store_true",
                    help="S_MODE_TOOLS_VIS (stride 2, record 2).")
parser.add_argument("--weapon-mouse-trace", default="off",
                    choices=("off", "on"),
                    help="D_MOUSE_TRACE_COORD (dynamic stride 16), an "
                         "axis with no world-family analogue. MEASURED: "
                         "it adds NO fetch site. It adds an interpolated "
                         "vec2 at location 1 -- which renumbers every "
                         "later location -- read once as a texture "
                         "coordinate (glsl:646), and it adds imageStore "
                         "writes of the traced hit's local x/y/z and a hit "
                         "flag into a small readback image (glsl:6862-"
                         "6883). It is a CPU-readback side channel for "
                         "what the crosshair is over, not a shading term.")
parser.add_argument("--weapon-baked-lighting", default="none",
                    choices=("none", "vertex-stream", "probe", "lightmap"),
                    help="D_BAKED_LIGHTING_FROM_VERTEX_STREAM / _PROBE / "
                         "_LIGHTMAP (dynamic strides 1, 2, 4). Selects "
                         "which baked source feeds the weapon's indirect "
                         "diffuse; 'probe' is the largest module of the "
                         "four (8,671 decompiled lines against 6,695).")
parser.add_argument("--weapon-specular-cube-static", action="store_true",
                    help="D_SPECULAR_CUBE_MAP_STATIC (dynamic stride 8): "
                         "a single pre-selected cubemap instead of the "
                         "runtime cubemap binner loop.")
parser.add_argument("--weapon-dissolve-origin", type=float, nargs=3,
                    default=None, metavar=("X", "Y", "Z"),
                    help="g_vDissolveOrigin.xyz, world metres. The tail "
                         "of EVERY csgo_weapon module carries a proximity "
                         "screen-door: f = 1 - smoothstep(6,2,|P-origin|), "
                         "threshold = mix(f*0.51+0.5, f*0.91+0.1, "
                         "smoothstep(0.1,1,strength)), discard where "
                         "dither.y < threshold "
                         "(baseline glsl:1352-1358). Unset = the term is "
                         "OFF because its strength is 0, which is what the "
                         "shader's own `if (w > 0)` does.")
parser.add_argument("--weapon-dissolve-strength", type=float, default=0.0,
                    help="g_vDissolveOrigin.w. 0 = the shader's own gate "
                         "is closed, matching an unset origin.")
parser.add_argument("--weapon-visibility-stencil", default="off",
                    choices=("off", "proxy"),
                    help="player_visibility_stencil_proxy: a stencil-proxy "
                         "path with no world-family counterpart "
                         "(player_visibility{,_stencil_proxy}.vcs in the "
                         "csgo archive, player_visibility_data.vcs in "
                         "csgo_core). This renderer has no stencil buffer, "
                         "so 'proxy' builds the equivalent COVERAGE MASK "
                         "from the viewmodel pass's own depth buffer and "
                         "uses it to keep the weapon out of the "
                         "player-occluded region. Stated difference: a "
                         "coverage mask, not a hardware stencil.")
parser.add_argument("--weapon-axes-reach", action="store_true",
                    help="print, per csgo_weapon combo axis, what "
                         "fraction of the viewmodel pass's covered pixels "
                         "took that path. An axis at 0%% says so out loud.")
parser.add_argument("--viewmodel", action="store_true",
                    help="run the FIRST-PERSON VIEWMODEL PASS: a second "
                         "raster with its OWN projection (--viewmodel-"
                         "hfov), its OWN VIEWPORT DEPTH RANGE (the "
                         "reference's [0, 0.1] minDepth/maxDepth, READ from "
                         "the wallA2 capture) "
                         "and its OWN depth buffer, composited over the "
                         "world frame WITHOUT testing world depth -- which "
                         "is what stops a viewmodel clipping into "
                         "geometry. Without this flag --weapon and "
                         "--viewmodel-legs-prepass have no pass to run in "
                         "and the renderer refuses rather than silently "
                         "shading nothing.")
parser.add_argument("--viewmodel-hfov", type=float, default=54.0,
                    help="horizontal FOV of the VIEWMODEL projection, "
                         "degrees, independent of --hfov. The engine keeps "
                         "this in a per-view constant buffer, not in the "
                         "shader, so this value is a RENDERER PARAMETER "
                         "with no bytecode source; what IS from the "
                         "bytecode is that the viewmodel is a separate "
                         "view at all.")
# --viewmodel-near / --viewmodel-far are DELETED. They expressed the
# viewmodel's depth compression as a second frustum in METRES (0.01 .. 4.0),
# which was a plausible reconstruction and is not the reference's mechanism:
# CS2 keeps the world frustum and compresses via the VkViewport
# minDepth/maxDepth, READ from the wallA2 capture as [0, 0.10000000149011612].
# See VM_RANGE below and W4_PROVENANCE_AUDIT.md sec 5-6. A validated
# replacement leaves ONE mechanism -- keeping both would let a run silently
# use the knob nobody meant.
parser.add_argument("--viewmodel-offset", type=float, nargs=3,
                    default=(0.09, -0.075, -0.55),
                    metavar=("RIGHT", "UP", "FORWARD"),
                    help="the viewmodel transform's translation in VIEW "
                         "space, metres (right, up, forward; forward is "
                         "-Z). Engine data, not shader data. MEASURED: at "
                         "(0.12, -0.10, -0.35) every visible face of the "
                         "synthesised proxy sits past 66 degrees of "
                         "incidence, and D_USE_BGREFRACT's edge term "
                         "1 - clamp((dot(camFwd,N)+1)/(1-g_flRefractEdge)) "
                         "is exactly 0 there at the shipped edge of 0.4 -- "
                         "the axis reported 0%% of pixels. At -0.55 m the "
                         "proxy spans face-on to grazing, so the term is "
                         "PARTITIONED rather than saturated at either "
                         "end. This is the synthesised proxy moving, not "
                         "the shader term changing.")
parser.add_argument("--viewmodel-scale", type=float, default=1.0,
                    help="uniform scale of the viewmodel transform.")
parser.add_argument("--viewmodel-roll", type=float, default=0.0,
                    help="roll of the viewmodel transform about the view "
                         "forward axis, degrees.")
parser.add_argument("--viewmodel-model", default=None,
                    help="a .pt bundle from extract_viewmodel.py: the REAL "
                         "weapon mesh, in the same 8-tuple "
                         "vm_synth_geometry() returns. Takes precedence "
                         "over --viewmodel-synth, which fabricates boxes. "
                         "The pass, its side-table columns and its shading "
                         "are unchanged -- only the vertex source differs, "
                         "which vm_synth_geometry's own docstring already "
                         "names as the whole of the gap.")
parser.add_argument("--vm-anim", choices=("on", "off"), default="on",
                    help="time-indexed clip playback for the viewmodel. "
                         "`off` holds the bundle's bind pose and is the "
                         "CONTROL arm: an A/B on one tick isolates what "
                         "playback changed, which a comparison across ticks "
                         "cannot, because the camera moves too. Also what "
                         "the ablation harness masks against.")
parser.add_argument("--vm-selftest-drop-class", default="none",
                    choices=("none", "weapon", "arms", "legs"),
                    help="CALIBRATION: drop one viewmodel class from the "
                         "composite, so the class assertion can be SEEN to "
                         "fire. An assertion never observed firing is not "
                         "evidence that it works -- it is a claim -- and "
                         "this lane has already shipped one check that was "
                         "vacuous for exactly that reason. Dropping `weapon` "
                         "reproduces the invisible gun the owner caught in "
                         "combat_closing; the run must then print CLASS "
                         "DEFECT and exit 3. Renders a deliberately wrong "
                         "frame and says so, so it can never be mistaken "
                         "for a real arm.")
parser.add_argument("--viewmodel-synth", default="off",
                    choices=("off", "weapon", "legs", "both"),
                    help="submit a SYNTHESISED draw into the viewmodel "
                         "pass. de_inferno's world pack contains no weapon "
                         "and no first-person legs, so 'nothing to render' "
                         "would otherwise be indistinguishable from 'never "
                         "called'. The synthesised geometry goes through "
                         "the SAME dispatch, the SAME side-table columns "
                         "and the SAME shading functions a packed weapon "
                         "would; only the vertex source differs, and that "
                         "is stated rather than hidden.")
parser.add_argument("--viewmodel-legs-prepass", action="store_true",
                    help="csgo_legs_prepass (csgo_core/shaders_vulkan_dir"
                         ".vpk), 1 static combo, 1 bytecode record, 1 "
                         "module, 42 FLOPs, 1 fetch -- transcribed "
                         "COMPLETE from all 34 lines of "
                         "csgo_legs_prepass__only_m0.glsl. Runs inside the "
                         "viewmodel pass BEFORE the weapon, writes "
                         "viewmodel depth and the constant colour "
                         "vec4(0.01,0.01,0.01,1) it emits at :32.")
parser.add_argument("--viewmodel-legs-fade", type=float, default=60.0,
                    help="g_flFirstpersonLegsFade -- set 1 / binding 0 / "
                         "offset 8 in csgo_legs_prepass. It is a DISTANCE: "
                         "fade = smoothstep(F-35, F, |camPos - wpos|), so "
                         "the legs dissolve as the camera closes to within "
                         "35 units of the fragment (:28).")
parser.add_argument("--viewmodel-legs-eye-z", type=float, default=55.0,
                    help="the (0,0,55) constant in the legs cone test "
                         "(:28), Source units above the model origin. "
                         "Literal in the shader; exposed because our "
                         "synthesised legs are not at Valve's scale.")
parser.add_argument("--viewmodel-legs-cone", type=float, default=0.8,
                    help="the step(0.8, ...) threshold in the legs cone "
                         "test (:28): keep only fragments whose direction "
                         "from the eye point is within acos(0.8) = 36.87 "
                         "degrees of straight down. Literal in the shader.")
parser.add_argument("--viewmodel-dither", default="bayer",
                    choices=("bayer", "hash", "white"),
                    help="which tile fills the role of the engine's "
                         "screen-door noise texture. csgo_legs_prepass "
                         "reads it as a DEDICATED texture2D at set 1 / "
                         "binding 30 (:20); every csgo_weapon module reads "
                         "it BINDLESSLY through a per-view CB handle, and "
                         "both index it as FragCoord.xy & mask, so the "
                         "tile is power-of-two and the .y channel is the "
                         "one used. STATED DIFFERENCE: the engine's tile "
                         "is an asset we do not have, so this synthesises "
                         "a tile of the same role and dimensions. 'bayer' "
                         "is an ordered dither, 'hash' a per-texel hash, "
                         "'white' uniform noise.")
parser.add_argument("--viewmodel-dither-size", type=int, default=64,
                    help="edge of the synthesised dither tile. Must be a "
                         "power of two: the shader masks with & (n-1), so "
                         "a non-power-of-two would alias silently.")
parser.add_argument("--viewmodel-depth-range", default="own",
                    choices=("own", "world"),
                    help="'own' (default, and the structure the evidence "
                         "supports) gives the viewmodel pass its own "
                         "projection, its own near/far and its own depth "
                         "buffer, composited over the world by COVERAGE "
                         "with no comparison against world depth. 'world' "
                         "is a DIAGNOSTIC that collapses the viewmodel "
                         "into the world pass's single camera and single "
                         "depth range, so the two can be differenced and "
                         "the separation measured. 'world' is never the "
                         "default -- collapsing the pass is the failure "
                         "this axis exists to make visible.")
parser.add_argument("--viewmodel-reach", action="store_true",
                    help="print the viewmodel pass's coverage and the "
                         "legs prepass's discard rate. A pass that covers "
                         "zero pixels has to say so.")
parser.add_argument("--viewmodel-parser-selftest", action="store_true",
                    help="replay every add_argument() call in this file "
                         "against a FRESH argparse.ArgumentParser and "
                         "report the conflict count, then exit. Two agents "
                         "declaring the same flag in different files "
                         "merges clean in git and then raises at import, "
                         "so the parser cannot be built for ANY "
                         "invocation. Use --viewmodel-parser-selftest-"
                         "inject to make it fail on purpose.")
parser.add_argument("--viewmodel-parser-selftest-inject", default="none",
                    choices=("none", "duplicate", "duplicate-foreign"),
                    help="inject a known defect into the parser self-test "
                         "so a clean result is distinguishable from a "
                         "self-test that never ran. 'duplicate' re-adds "
                         "one of THIS file's own flags; "
                         "'duplicate-foreign' re-adds a flag this file did "
                         "not declare. Both must be seen to FAIL before a "
                         "clean run means anything "
                         "(CHECKS_THAT_CANNOT_FAIL.md).")

# =======================================================================
# OTHER PLAYERMODELS -- the enemies and teammates standing in the world
# =======================================================================
# THE GAP THIS CLOSES. Every ground-truth frame this project scores was
# captured from inside a live round, and a live round has other players in
# it. This renderer has drawn none of them, ever: the csgo_character
# shading family is fully written (char_normal_decode, char_frame,
# char_aniso_*, the hair/cloth/retro lobes, the --char-* axes) and has
# never had one character VERTEX to run on, because de_inferno's world pack
# contains no csgo_character material and no player geometry. So a human
# silhouette in the GT frame scored against empty world behind it.
#
# The two halves come from different places and both are READ:
#   GEOMETRY  extract_playermodel.py, from the depot's own character
#             .vmdl_c, in MODEL space / SOURCE units (bind pose -- animation
#             is not on the wire, proto 14150 deleted DEM_AnimationData).
#   PLACEMENT the .dem, per tick, per entity: X/Y/Z/yaw/is_alive/team.
#
# ABSENT BY DEFAULT AND LOUD ABOUT IT. With no --playermodels the pass
# prints a GAP line per frame rather than rendering an empty world that
# looks finished, which is the state this whole flag exists to end.
parser.add_argument("--playermodels", default=None,
                    help="entity source for the OTHER players in frame: "
                         "either a .dem (parsed with demoparser2, the same "
                         "dependency harness/cs2_demo_camera.py uses) or a "
                         "rows JSON of schema "
                         "iji/cs2-demo-playermodels/v1. Absent -> the "
                         "feature is OFF and every frame prints a GAP line "
                         "saying so.")
parser.add_argument("--playermodel-bundles", default=None,
                    help="directory of <team>.pt bundles from "
                         "extract_playermodel.py -- `ct.pt` and `t.pt`. An "
                         "entity whose team has no bundle is NOT drawn and "
                         "NOT substituted with the other team's model: a "
                         "CT wearing a T body scores as a near miss rather "
                         "than as the miss it is.")
parser.add_argument("--playermodel-agent-map", default=None,
                    help="JSON {agent key: bundle basename} binding the "
                         "per-player agent READ off the wire to a bundle in "
                         "--playermodel-bundles. The agent arrives as "
                         "CCSPlayerPawn.CBodyComponentBaseAnimGraph."
                         "m_hModel, a model HANDLE and not a name -- "
                         "demoparser2 exposes no handle -> "
                         "agents/models/<name> table -- so the binding is "
                         "the operator's, made once the join is known. "
                         "Without it every entity falls back to its TEAM "
                         "bundle and the frame line says so per entity. "
                         "Extract more agents with extract_playermodel.py "
                         "and add them here; nothing in the pass changes.")
parser.add_argument("--playermodel-exclude-steamid", default=None,
                    help="comma-separated steamids to leave out. The "
                         "RECORDING player is excluded automatically by the "
                         "steamid --camera-json carries (cs2_demo_camera "
                         "writes it), because that entity IS the camera and "
                         "drawing it puts a head inside the near plane. "
                         "This flag is for anyone else.")
parser.add_argument("--pm-anim", choices=("on", "off"), default="on",
                    help="third-person clip playback for the OTHER players. "
                         "`off` holds every entity at its bundle's BIND "
                         "pose, which is what this pass did before the "
                         "skinning landed and is therefore the CONTROL arm: "
                         "an on/off A/B on ONE tick isolates what playback "
                         "moved, which a comparison across ticks cannot "
                         "because the players and the camera both move. "
                         "`off` is also what conformance/anim_driven.py "
                         "masks against.")
parser.add_argument("--playermodel-clips", default=None,
                    help="the world-clip sidecar (`agents.clips.pt`, schema "
                         "iji/model-clips/v2) every character shares -- all "
                         "82 bundles are skinned to the SAME "
                         "animation/skeletons/characters/worldmodel.vnmskel, "
                         "so one sidecar drives all of them. Default: "
                         "`agents.clips.pt` inside --playermodel-bundles, "
                         "and its absence is printed rather than silently "
                         "leaving every player T-posed.")
parser.add_argument("--playermodel-weapons", default=None,
                    help="directory of MODEL-SPACE weapon bundles for the "
                         "THIRD-PERSON hand, built with "
                         "extract_playermodel.py --allow-stub over "
                         "weapons/models/<w>/<w>.vmdl_c. NOT vm_bundles: "
                         "those are view-space with a placement already "
                         "baked in and are refused by the `space` field. "
                         "Absent -> allies are posed holding nothing, and "
                         "every frame line says so per entity.")
parser.add_argument("--playermodel-weapon-map", default=None,
                    help="JSON {active_weapon_name: bundle basename} -- the "
                         "demo says `AK-47`, the depot says "
                         "`weapon_rif_ak47`, and no table joins them that "
                         "this lane can READ, so the operator supplies it. "
                         "A normalised direct hit (AK-47 -> ak47) is tried "
                         "first; an unmapped weapon is NOT drawn and NOT "
                         "substituted with another model, and the entity's "
                         "own line names it.")

# =======================================================================
# NON-WORLD SHADER FAMILIES: csgo_character / csgo_eyeball / csgo_customglove
# =======================================================================
# SHADER_CALLFLOW_nonworld_families.md.  Every flag below is namespaced
# --char-* / --eyeball-* / --glove-* so it cannot collide with another
# family's flag; `parser_replay.py` replays the whole file's add_argument
# list against a real ArgumentParser and reports duplicates, because on
# 2026-08-07 two agents declared the same name and argparse then raised
# during import, killing every invocation.
#
# NONE of the three families is a MAP material, so a de_inferno material
# scan contains zero of them.  That is a USAGE fact.  --char-assign exists
# so the capability is still reachable on the only map this project packs;
# it is loud about what it reassigns and it is NOT on by default.
parser.add_argument("--char", action="store_true",
                    help="csgo_character (csgo_core archive; 314 model "
                         "materials, 5,904 static combos / 68,240 shipped "
                         "pairs, 3,083 FLOPs/px, 49 fetch sites, 9 loops -- "
                         "the largest combo space of any CS2 pixel shader "
                         "found). Turns on the family's surface path; the "
                         "individual axes below select which of its combos "
                         "are evaluated.")
parser.add_argument("--char-assign", choices=("family", "vertexlit",
                                              "complex", "all"),
                    default="family",
                    help="which materials the csgo_character/eyeball/glove "
                         "paths run on. 'family' (default) = only materials "
                         "whose .vmat names the family, which on de_inferno "
                         "is ZERO because these are MODEL materials and the "
                         "pack is a MAP export -- the run then prints that "
                         "the paths reached no pixels and names this flag. "
                         "'vertexlit' reassigns csgo_vertexlitgeneric "
                         "materials (this project's own models/ inference, "
                         "CHECKS_THAT_CANNOT_FAIL.md 'When the better test "
                         "dies'), 'complex' reassigns csgo_complex, 'all' "
                         "every material. Reassignment is printed with its "
                         "material and face counts.")
parser.add_argument("--char-side", default=None,
                    help="path to a side table built by the CURRENT "
                         "fast_pack_fam.py carrying the non-world columns "
                         "and textures. Defaults to --fam-side. Kept "
                         "separate so a character pack is never written "
                         "over assets/fam_side.pt.")
parser.add_argument("--char-sss", action="store_true",
                    help="S_SUBSURFACE_SCATTERING. Pre-integrated skin: the "
                         "diffuse is looked up in g_tDiffuseFalloff by "
                         "(N.L, curvature). No world family has any "
                         "transmission model beyond csgo_complex's "
                         "S_TRANSMISSIVE_BACKFACE_NDOTL, which is a wrap "
                         "term, not scattering.")
parser.add_argument("--char-curvature", choices=("vertex", "derivative",
                                                 "off", "auto"),
                    default="auto",
                    help="S_USE_PER_VERTEX_CURVATURE. 'auto' (the default) "
                         "is 'vertex' when a csgo_character/eyeball/glove "
                         "path is actually selected and 'off' otherwise, "
                         "and prints which it chose. It defaults to 'auto' "
                         "rather than 'vertex' because a default-on value "
                         "here is read through CHAR_AXES -> NONWORLD -> "
                         "FAMILY, which made --fam-side a hard error for "
                         "EVERY invocation of this file, including the ones "
                         "that never asked for a character. "
                         "'vertex' = the "
                         "Curvature vertex stream the csgo_character vs "
                         "passes straight through to the ps "
                         "(m_vsInputSignatureArray names it 'Curvature' / "
                         "flSSSCurvature); our pack has no such stream, so "
                         "it is derived per vertex from the mesh and that "
                         "substitution is printed. 'derivative' takes it "
                         "from the screen-space normal derivative instead, "
                         "which is the S_USE_PER_VERTEX_CURVATURE=0 form. "
                         "Test with != 'off'; the string 'off' is truthy.")
parser.add_argument("--char-aniso", action="store_true",
                    help="S_ANISOTROPIC_GLOSS on csgo_character. The axis "
                         "csgo_complex declares and NO de_inferno material "
                         "selects -- so --aniso-gloss was written against a "
                         "path nothing exercised. Here it is live.")
parser.add_argument("--char-aniso-tangents",
                    choices=("uv", "spherical", "auto"),
                    default="auto",
                    help="S_SPHERICAL_PROJECTED_ANISOTROPIC_TANGENTS. "
                         "'spherical' builds the tangent by projecting the "
                         "surface point onto a sphere about the model "
                         "origin, which is the axis; 'uv' keeps the UV "
                         "tangent, i.e. the axis off. 'auto' (the default) "
                         "is 'spherical' when a csgo_character/eyeball/"
                         "glove path is actually selected and 'uv' "
                         "otherwise, and prints which it chose -- see "
                         "--char-curvature for why neither defaults on.")
parser.add_argument("--char-hair", action="store_true",
                    help="S_ANISOTROPIC_HAIR: the shifted specular pair "
                         "plus g_flHairTransmission.")
parser.add_argument("--char-cloth", action="store_true",
                    help="F_CLOTH_SHADING (g_bClothShading): the sheen "
                         "lobe, scaled by g_flSheenScale and tinted by "
                         "g_flSheenTintColor.")
parser.add_argument("--char-iridescence", action="store_true",
                    help="S_IRIDESCENCE: a hue-shifted, thickness-masked "
                         "Fresnel term.")
parser.add_argument("--char-retro-reflective", action="store_true",
                    help="S_RETRO_REFLECTIVE: the lobe folded back toward "
                         "the view direction.")
parser.add_argument("--char-patches", action="store_true",
                    help="S_PATCHES: three independently transformed patch "
                         "decals, each with its own backing layer, scale, "
                         "rotation, squash and highlight time.")
parser.add_argument("--char-decals", action="store_true",
                    help="S_DECAL_TEXTURE + D_DECALS_ENABLED on "
                         "csgo_character -- decal projection onto SKINNED "
                         "geometry, which no world family has.")
parser.add_argument("--char-blood", action="store_true",
                    help="the blood layer: g_tBloodMask gates g_tColorBlood "
                         "and g_tNormalBlood over the base surface.")
parser.add_argument("--char-adjustments", action="store_true",
                    help="S_ENABLE_ADJUSTMENTS: hue / saturation / "
                         "brightness / contrast on the albedo, plus the "
                         "roughness brightness-contrast pair and its "
                         "g_bMaskRoughnessAdjustmentsByTintMask gate.")
parser.add_argument("--char-detail", action="store_true",
                    help="F_DETAIL_TEXTURE on csgo_character: g_tDetail "
                         "under g_nDetailBlendMode, with its own "
                         "hue/sat/brightness/contrast set.")
parser.add_argument("--char-eyes", action="store_true",
                    help="S_EYEBALLS inside csgo_character (as opposed to "
                         "the separate csgo_eyeball family): the iris and "
                         "pupil discs on g_tEyeAlbedo1, masked by "
                         "g_tEyeMask1.")
parser.add_argument("--char-tint-mask", action="store_true",
                    help="S_TINT_MASK on csgo_character.")
parser.add_argument("--char-alpha-test", action="store_true",
                    help="S_ALPHA_TEST on csgo_character, using the "
                         "material's own g_flAlphaTestReference.")
parser.add_argument("--char-translucent", action="store_true",
                    help="S_TRANSLUCENT on csgo_character: moves the "
                         "family's faces into the BLEND class so they "
                         "composite -- and so --mboit's two-pass moment "
                         "path shades them, which is how 'MBOIT on player "
                         "geometry' composes without a second entry point.")
parser.add_argument("--char-additive", action="store_true",
                    help="S_ADDITIVE_BLEND on csgo_character.")
parser.add_argument("--char-invulnerability", action="store_true",
                    help="g_cInvulnerabilityColor * g_flSpawnInvulnerability "
                         "-- the spawn-protection tint.")
parser.add_argument("--char-supports-decals", action="store_true",
                    help="S_SUPPORTS_DECALS: the combo that makes the "
                         "surface a decal RECEIVER (distinct from "
                         "S_DECAL_TEXTURE, which applies one).")
parser.add_argument("--char-mode-depth", action="store_true",
                    help="S_MODE_DEPTH on csgo_character -- the depth-only "
                         "variant, which shades nothing.")
parser.add_argument("--char-tools-vis", type=int, default=0,
                    help="S_MODE_TOOLS_VIS on csgo_character, i.e. "
                         "g_nToolsVisMode. NOT a small enum: the shipped "
                         "module tests THIRTY distinct values -- 1,2,3,4,5,"
                         "6,8,10..25,30,31,60,65,66,80,81 -- counted from "
                         "the equality comparisons against that uniform in "
                         "the decompiled bytecode. 0 is off, and 0 is FALSY "
                         "so a bare truthiness test on this flag is "
                         "correct, unlike the string-valued flags.")
parser.add_argument("--char-baked-lighting",
                    choices=("inherit", "vertex-stream", "probe", "lightmap"),
                    default="inherit",
                    help="D_BAKED_LIGHTING_FROM_VERTEX_STREAM / _PROBE / "
                         "_LIGHTMAP for csgo_character specifically. "
                         "'inherit' takes --baked-lighting. Test with "
                         "!= 'inherit'.")
parser.add_argument("--char-visibility-stencil", action="store_true",
                    help="the player_visibility / "
                         "player_visibility_stencil_proxy pass: a "
                         "stencil-proxy occlusion term with no world-family "
                         "counterpart. g_flPlayerOcclusionAmount and "
                         "g_flWeaponOcclusionAmount fade the surface where "
                         "the proxy says the player is behind geometry.")
parser.add_argument("--char-skinning", action="store_true",
                    help="the csgo_character VERTEX path: four-influence "
                         "linear blend skinning off BLENDINDICES / "
                         "BLENDWEIGHT against a bone palette, transcribed "
                         "from csgo_character_vulkan_50_vs. THE FIRST "
                         "SKINNED VERTEX PATH IN THIS RENDERER.")
parser.add_argument("--char-bones", type=int, default=12,
                    help="bone count for the substituted palette. The .glb "
                         "world pack carries no skeleton, so the binding is "
                         "derived from the mesh (cs2_vertex_stage."
                         "character_skin_binding) and this is its "
                         "resolution.")
parser.add_argument("--char-bone-amp", type=float, default=0.0,
                    help="pose amplitude of the substituted bone palette. "
                         "0.0 = every bone exactly the identity, so the "
                         "skinning path still RUNS and returns the bind "
                         "pose -- a zero delta then means 'bind pose "
                         "requested', never 'the path did not run'.")
parser.add_argument("--char-bone-rate", type=float, default=0.0,
                    help="angular rate of the substituted bone palette, "
                         "rad/s.")
parser.add_argument("--char-selftest", action="store_true",
                    help="run every csgo_character / csgo_eyeball / "
                         "csgo_customglove path at ITS OWN DEFAULTS on a "
                         "synthetic pixel batch, print max|delta| per path "
                         "against that path suppressed, and EXIT NON-ZERO "
                         "if any selected path is dead. Reachability proof, "
                         "not an image metric.")
parser.add_argument("--char-selftest-inject", default=None,
                    help="deliberately break one named path so "
                         "--char-selftest can be made to FAIL ON PURPOSE. "
                         "Names: sss-flat, aniso-isotropic, hair-dead, "
                         "cloth-zero, irid-zero, retro-zero, patches-clear, "
                         "decal-zero, adjust-identity, eye-flat, "
                         "glove-passthrough, skin-identity. 'all' lists "
                         "them and exits.")
parser.add_argument("--eyeball", action="store_true",
                    help="csgo_eyeball (csgo archive; 1,922 FLOPs/px, 39 "
                         "fetch sites, 8 loops, 2 static combos). A "
                         "SEPARATE FAMILY from csgo_character's S_EYEBALLS "
                         "combo, not the same shader.")
parser.add_argument("--eyeball-parallax", action="store_true",
                    help="the cornea/iris parallax: the view ray is walked "
                         "to the iris plane inside the eyeball sphere of "
                         "radius g_flEyeBallRadius1 before the iris disc is "
                         "sampled.")
parser.add_argument("--eyeball-cubemap-static", action="store_true",
                    help="D_SPECULAR_CUBE_MAP_STATIC -- the eyeball's own "
                         "dynamic axis, which selects a static cube for the "
                         "corneal reflection instead of the binned list.")
parser.add_argument("--glove", action="store_true",
                    help="csgo_customglove (csgo_core archive; 1,621 "
                         "FLOPs/px, 28 fetch sites, ALL 2D -- no shadow, no "
                         "probe, no IBL fetch at all, which is why it needs "
                         "its own path rather than a combo of another).")
parser.add_argument("--glove-aniso", action="store_true",
                    help="S_ANISOTROPIC_GLOSS on csgo_customglove.")
parser.add_argument("--glove-tint-id", action="store_true",
                    help="S_TINT_ID: g_tTintId indexes the g_vId*Color "
                         "palette, so one texture drives up to eight tint "
                         "regions.")
parser.add_argument("--glove-backcompat", action="store_true",
                    help="S_BACKWARDS_COMPATIBILITY: the pre-finish-system "
                         "path, which reads g_tSurface directly instead of "
                         "compositing the substrate/surface/damage/grime "
                         "layer stack.")
parser.add_argument("--eyeball-centre", type=_f3, default="16,16,0",
                    help="the csgo_eyeball sphere centre. The shipped "
                         "bytecode uses the LITERAL float3(16,16,0) in "
                         "every static and dynamic combo of the family "
                         "(eyeball_ps/r1_m2.metal:346,367) and no per-eye "
                         "centre uniform or varying reaches the pixel "
                         "shader at all -- so the literal is the default "
                         "here, and it is exposed because that literal "
                         "cannot be right for two eyes on a moving player.")
parser.add_argument("--char-lobe-radius", type=float, default=0.0,
                    help="the light-source radius floor the shipped "
                         "csgo_character clamps its roughness pair up to "
                         "(r5400_m0:1347, `rough2 = max(rough2, "
                         "lightSourceRadius)`; the sun uses "
                         "_5538._m18.w and each punctual light its own "
                         "_5570[i]._m11). The renderer has no per-light "
                         "radius, so this is the one value for all of "
                         "them; 0.0 is the no-op.")
parser.add_argument("--glove-wear", type=float, default=-1.0,
                    help="override g_fWearProgress in [0,1]. -1 = per "
                         "material, the ground-truth path and the default.")

# ======================================================================
# NON-WORLD FAMILIES: csgo_projected_decals + spritecard (csgo_core).
#
# Every flag they add is declared inside fam_nonworld.add_arguments and
# is namespaced --decal-* / --sprite-*. `--nonworld-selftest` REPLAYS
# the whole parser (this one, with these flags on it) against a fresh
# ArgumentParser and reports the conflict count -- because a P0 broke
# main when two files declared the same flag, git merged clean, and
# argparse then raised at IMPORT so no invocation could build a parser.
# ======================================================================
import fam_nonworld as _nw                                    # noqa: E402
# Runtime output statistics per term -- mean, sd, and FRACTION AT
# IDENTITY. Its own module so the conformance registry and any offline
# driver can import it without importing this file, which parses argv at
# import and is therefore not importable.
import term_stats as _ts                                      # noqa: E402
# A NON-SHADOWABLE HANDLE ON THE SAME MODULE.
#
# render_window binds a LOCAL `_ts` -- a tangent-space normal,
# `_ts = torch.cat([nmap[..., :2] * 2.0 - 1.0, ...])` -- and under Python's
# scoping rules that makes `_ts` a local for the WHOLE function body. Every
# reference to the module EARLIER in that function is then an
# UnboundLocalError rather than a module access:
#
#   UnboundLocalError: cannot access local variable '_ts'
#   render_window, at the customglove.alpha observe
#
# It fires only on the branch that reaches it, which is why it shipped: the
# render job that caught it ran ten arms and only the one with a non-zero
# csgo_customglove population died. Both names are legitimate -- a
# tangent-space normal is `_ts` and so is term_stats -- so the collision is
# the accident, not either name.
#
# Long-lived functions reference THIS alias, so a future local called `_ts`
# (this file already has two) cannot reach through and silently disable a
# measurement. Renaming the local instead would fix today's crash and leave
# the trap armed for the next one.
_TERMSTATS = _ts
_nw.add_arguments(parser)
import muzzle_flash as _mz                                    # noqa: E402
import anim_script as _anim                                   # noqa: E402
import skin_anim as _skin                                     # noqa: E402
_mz.add_arguments(parser)
parser.add_argument("--nonworld-selftest", action="store_true",
                    help="replay every add_argument on this parser "
                         "against a real ArgumentParser, recount "
                         "spritecard's 708/19 from the committed "
                         "reference GLSL, and exit. Reports max|delta| "
                         "for the depth-buffer world reconstruction when "
                         "a world is loaded.")
parser.add_argument("--nonworld-selftest-inject", default=None,
                    choices=("dup-flag", "depth-remap", "decal-dead",
                             "sprite-dead"),
                    help="break one named path on purpose so the "
                         "non-world self-test can be seen to FAIL. "
                         "Until each has fired, a clean run and a run "
                         "that never checked anything are the same "
                         "output (CHECKS_THAT_CANNOT_FAIL.md).")

# =====================================================================
# BATCH-2 LOOP FAMILIES -- six families, transcribed from the shipped
# .vcs bytecode.  SHADER_CALLFLOW_loop_families_batch2.md is the
# enumeration these come from.
#
# EVERY FLAG IN THIS BLOCK IS NAMESPACED `--b2-`.  Two agents declaring
# the same flag in two files merged clean this morning and argparse then
# raised at import, so the renderer could not build its parser for ANY
# invocation.  A private prefix is the only thing that makes that
# collision impossible rather than unlikely, and `--b2-parser-replay`
# below re-plays every add_argument in this file against a fresh
# ArgumentParser and prints the conflict count.
#
# All six are the STANDARD BITMASK-BINNER shape at nesting depth <= 1.
# None is a march -- csgo_simple_liquid's 18 loops are nine ordinary
# binner pairs, and a high loop count is not a march; nesting depth is.
# No ray-marching structure is imported into any of them.
# =====================================================================
parser.add_argument("--b2-families", default="off",
                    help="comma list of the batch-2 families to run, "
                         "'all', or 'off'. Names: sticker, glove, simple, "
                         "fourway, liquid, customweapon. THIS IS A STRING: "
                         "'off' is TRUTHY in Python, so it is resolved to "
                         "a set exactly once, at startup, by "
                         "_b2_family_set(); nothing downstream tests it "
                         "with bare truthiness.")
parser.add_argument("--b2-side", default=None,
                    help="batch-2 side table, built by fast_pack_fam.py "
                         "--b2-out. A DISTINCT FILE from --fam-side: it is "
                         "never written over assets/fam_side.pt, which "
                         "every other agent is scoring against.")
parser.add_argument("--b2-reach", action="store_true",
                    help="synthesise one draw per family whose materials "
                         "this pack does not carry, so 'nothing to render' "
                         "can never be reported as 'never called'. Prints "
                         "the pixel count each entry point wrote.")
parser.add_argument("--b2-selftest", action="store_true",
                    help="run every entry point at EVERY value of EVERY "
                         "combo axis and print max|delta| between the arms. "
                         "Use --b2-selftest-inject to make it fail on "
                         "purpose; a self-test that has never been made to "
                         "fail is not a check (CHECKS_THAT_CANNOT_FAIL.md).")
parser.add_argument("--b2-selftest-inject", default=None,
                    help="deliberately break one named batch-2 path so "
                         "--b2-selftest FAILS ON PURPOSE. Names: "
                         "sticker-frames-collapse, holomask-passthrough, "
                         "glove-deriv-zero, fourway-weights-uniform, "
                         "liquid-depth-dead, "
                         "cw-style-collapse. 'all' lists them and exits.")
# --- csgo_weapon_sticker: 7,127 FLOPs/px, 82 fetch, 13 loops, depth 1 --
# the most expensive pixel shader measured anywhere in CS2 so far, and
# 198 `cross` -- multiple tangent frames, stickers layered over a surface
# that has its own.
parser.add_argument("--b2-sticker-foil-layers", type=int, default=2,
                    help="foil/iridescence sub-layers inside the ONE "
                         "sticker layer. MEASURED, not assumed: the "
                         "reference has a single sticker over the base "
                         "(glsl:423 gates the whole path on _5618._m26) "
                         "and packs TWO foil UV sets in one vec4 at "
                         "glsl:556 -- layer A at stickerUV*2.5*scale, "
                         "layer B at (stickerUV+0.5)*2.5*scale.")
parser.add_argument("--b2-holomask-dxt", type=int, choices=(0, 1), default=0,
                    help="S_HOLOMASK_USE_DXT. MEASURED: the two static "
                         "combos compile to BYTE-IDENTICAL SPIR-V (s1_d2 "
                         "and s3_d2 have the same module md5), so this "
                         "axis selects no code in this family. The choice "
                         "it names is made at RUNTIME instead, by the "
                         "dual-sampler lerp at glsl:711-714 and the LUT "
                         "sampler select at glsl:945 -- both implemented "
                         "and both driven by --b2-holomask-lut-sampler. "
                         "The flag is kept so the axis is representable "
                         "at both its values and the finding is "
                         "discoverable, not silently dropped.")
parser.add_argument("--b2-holomask-lut-sampler", type=int, choices=(0, 1),
                    default=0,
                    help="_5618._m42 (glsl:945), the runtime holo-gradient "
                         "LUT sampler select that S_HOLOMASK_USE_DXT was "
                         "expected to be. This is the axis that actually "
                         "changes the image.")
parser.add_argument("--b2-sticker-mode-depth", type=int, choices=(0, 1),
                    default=0,
                    help="S_MODE_DEPTH on csgo_weapon_sticker. The depth "
                         "variant is a real 1,346-FLOP / 27-fetch module "
                         "(measured), not an empty pass.")
parser.add_argument("--b2-sticker-wear", type=float, default=0.0,
                    help="sticker wear scalar, 0..1.")
# --- csgo_customglove_preview: 6,385 FLOPs/px, 104 fetch (WIDEST) -----
parser.add_argument("--b2-glove-tint-id", type=int, default=1,
                    help="S_TINT_ID. Both values implemented.")
parser.add_argument("--b2-glove-spec-cube-static", type=int, choices=(0, 1),
                    default=0,
                    help="D_SPECULAR_CUBE_MAP_STATIC on the glove. "
                         "Measured: 6,385 FLOPs at 0 and 6,195 at 1, so "
                         "the axis is a REPLACEMENT of the probe path, not "
                         "an addition to it.")
parser.add_argument("--b2-glove-deriv", choices=("analytic", "off"),
                    default="analytic",
                    help="the 142 dFdx/dFdy/fwidth ops -- the heaviest "
                         "derivative use of any CS2 family measured. "
                         "'analytic' runs the screen-space filter width; "
                         "'off' is a DIAGNOSTIC that isolates its "
                         "contribution. CHOICES, not a bare string tested "
                         "for truth.")
# --- simple: 5,196 FLOPs/px, 54 fetch, 11 loops ----------------------
parser.add_argument("--b2-simple-baked", choices=("vertex", "probe",
                                                  "lightmap", "none"),
                    default="probe",
                    help="the three D_BAKED_LIGHTING_FROM_* axes are "
                         "MUTUALLY EXCLUSIVE in the shipped enumeration "
                         "(8 dynamic ids in the space, only 4 live), so "
                         "they are one 4-valued choice here rather than "
                         "three independent flags that could select a "
                         "combo Valve does not ship.")
# --- csgo_lightmapped_4wayblend: 5,145 FLOPs/px, 56 fetch, 7 loops ---
parser.add_argument("--b2-fourway-detail", type=int, choices=(0, 1),
                    default=1, help="S_DETAILTEXTURE.")
parser.add_argument("--b2-fourway-spec-direct", type=int, choices=(0, 1),
                    default=1, help="S_SPECULAR_DIRECT.")
parser.add_argument("--b2-fourway-spec-indirect", type=int, choices=(0, 1),
                    default=1, help="S_SPECULAR_INDIRECT.")
parser.add_argument("--b2-fourway-quality", type=int, choices=(0, 1),
                    default=1,
                    help="S_SHADER_QUALITY. Both BRDFs are written; a "
                         "quality level never gates implementation.")
parser.add_argument("--b2-fourway-spec-cube-static", type=int,
                    choices=(0, 1), default=0,
                    help="D_SPECULAR_CUBE_MAP_STATIC on the 4-way blend.")
parser.add_argument("--b2-fourway-baked", choices=("vertex", "probe",
                                                   "lightmap", "none"),
                    default="probe", help="the three baked-lighting axes.")
# --- csgo_simple_liquid: 4,997 FLOPs/px, 82 fetch, 18 loops ----------
# 54 sampler2DShadow -- exactly double the 27 the others share. MEASURED
# CAUSE (glsl:1167-1181 vs :529): the whole binner + cascade runs TWICE,
# once at gl_FragCoord and once at an explicitly RE-PROJECTED screen
# position, gated by the per-view int _5037._m2.x.
parser.add_argument("--b2-liquid-foam", type=int, choices=(0, 1), default=1,
                    help="S_FOAM.")
parser.add_argument("--b2-liquid-no-liquid", type=int, choices=(0, 1),
                    default=0, help="S_NO_LIQUID.")
parser.add_argument("--b2-liquid-opaque-refract", type=int, choices=(0, 1),
                    default=1, help="S_OPAQUE_REFRACT.")
parser.add_argument("--b2-liquid-test-values", type=int, choices=(0, 1),
                    default=0, help="S_USE_TEST_VALUES.")
parser.add_argument("--b2-liquid-bgrefract", type=int, choices=(0, 1),
                    default=1,
                    help="D_USE_BGREFRACT. MEASURED to change the shipped "
                         "module on 192/192 comparable combo pairs.")
parser.add_argument("--b2-liquid-opaque-fade", type=int, choices=(0, 1),
                    default=1,
                    help="D_OPAQUE_FADE. MEASURED to change the module on "
                         "336/336 pairs.")
parser.add_argument("--b2-liquid-use-depth", type=int, choices=(0, 1),
                    default=1,
                    help="D_USE_DEPTH. Implemented at both values. NOTE "
                         "the measurement: this axis selects BYTE-IDENTICAL "
                         "bytecode on all 288 comparable combo pairs, so "
                         "the shipped modules do not branch on it -- the "
                         "scene-depth read they DO carry is the per-view "
                         "offset-80 target and is unconditional.")
parser.add_argument("--b2-liquid-msaa-depth", type=int, choices=(0, 1),
                    default=1,
                    help="D_MSAA_DEPTH. Same measurement: identical "
                         "bytecode on all 96 comparable pairs.")
parser.add_argument("--b2-liquid-depth-source", choices=("frag", "lin"),
                    default="frag",
                    help="which of the renderer's two exposed depths the "
                         "liquid's scene-depth read consumes: 'frag' = "
                         "_frag_depth (the raster z the reference "
                         "subtracts as gl_FragCoord.zzzz), 'lin' = "
                         "_lin_depth (metres along the view axis). BOTH "
                         "axes at both values, so this crosses with "
                         "--b2-liquid-use-depth / --b2-liquid-msaa-depth "
                         "for the four combinations.")
parser.add_argument("--b2-liquid-second-pass", type=int, choices=(0, 1),
                    default=0,
                    help="NOT IMPLEMENTED, and accepted only so an "
                         "existing command line does not break. The "
                         "per-view enable _5037._m2.x (glsl:1166) runs "
                         "the binner and the 27-tap cascade a SECOND time "
                         "at the re-projected position, which is why this "
                         "family has 54 sampler2DShadow sites. What stood "
                         "here re-ran both at the SAME position -- the "
                         "re-projection's return value was never read -- "
                         "with baked lighting recomputed from identical "
                         "arguments and a blend weight of 0, so it could "
                         "not change a pixel while counting 2,048 "
                         "executions per audit. It is deleted rather than "
                         "left reporting itself as live. Passing 1 raises "
                         "instead of quietly doing nothing.")
# --- csgo_customweapon: 1,343 FLOPs/px sampled, 21 fetch, ZERO loops --
# NOT A LIT SURFACE.  Zero loops, zero shadow, zero probe, zero IBL,
# zero reflect, zero cross, zero derivatives; 21 sampler2D and nothing
# else over 1,152 static combos whose axes are all texture composition.
# It is a WEAPON-FINISH COMPOSITOR and is deliberately NOT wired into
# the light binner.
parser.add_argument("--b2-cw-paint-style", type=int, default=-1,
                    help="S_PAINT_STYLE, the one NON-BINARY axis in these "
                         "six: 0..8, place value 1, so the combo id is "
                         "MIXED RADIX and must be decoded with // and %% "
                         "and never with &. -1 = take it from the "
                         "material.")
parser.add_argument("--b2-cw-override-normal", type=int, choices=(0, 1),
                    default=0, help="S_OVERRIDE_NORMAL.")
parser.add_argument("--b2-cw-use-all-masks", type=int, choices=(0, 1),
                    default=0, help="S_USE_ALL_MASKS.")
parser.add_argument("--b2-cw-roughness-texture", type=int, choices=(0, 1),
                    default=0, help="S_ROUGHNESS_TEXTURE.")
parser.add_argument("--b2-cw-pearlescence-mask", type=int, choices=(0, 1),
                    default=0, help="S_PEARLESCENCE_MASK.")
parser.add_argument("--b2-cw-metalness-texture", type=int, choices=(0, 1),
                    default=0, help="S_METALNESS_TEXTURE.")
parser.add_argument("--b2-cw-separate-channels", type=int, choices=(0, 1),
                    default=0, help="S_SEPARATE_CHANNEL_INPUTS.")
