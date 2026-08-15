"""The constant tables the loop families ship, as this renderer holds them.

THE VALUES BELOW ARE MACHINE-EMITTED, not hand-typed. The block between the
BEGIN/END markers is written by

    python3 -m counter_strike_render.vcs.gen_loopfam_tables --write

out of the decompiled modules under
`docs/projects/counter-strike-sft/glsl_loopfam/`, whose sha256 are pinned in
`conformance/reference_sources.json`; the generator REFUSES before reading
if any artifact no longer matches its pin. `--check` diffs the committed
block against a fresh parse and exits 1 on any difference, so "tables.py is
the generator's output" is asserted rather than assumed.

WHY THAT MATTERS FOR THE A/B. The conformance terms for these tables parse
the .glsl on the REFERENCE side, which closes the same-reader gap there:
the comparison is our constants against THE ARTIFACT, not against a second
reading of it. If the IMPL side were hand-typed the pair would still be
checking a hand-copy, and a 27-float kernel is a real transcription risk.
With the generator in the loop the A/B checks THE GENERATOR against the
artifact -- the honest boundary, because a generator that misreads 27
floats misreads them reproducibly and the test says so.

MEASURED SENSITIVITY of that check: a +1e-4 perturbation of one weight goes
red on BOTH families that share the copy; +1e-7 does not fire, being under
the registry's 1e-6 epsilon. So it catches transcription and edit errors at
1e-6 and above and does NOT catch a float32 last-digit slip.

EACH LITERAL KEEPS ITS SOURCE TEXT. `repr(float)` would print the shortest
round-tripping form -- 0.3990499973297119 for a value the artifact writes as
0.3990499973297119140625. Both parse to the same double; only one can be
diffed against the shader by eye, and provenance is why this file exists.

None of these is replaced by its closed form. Several are recognisably
rounded (1/6, 1/12, cos 5 degrees) and the bytecode stores the float32; the
closed form is a different number.
"""
from .._arraylib import np  # dispatches to torch when handed tensors

# --- BEGIN GENERATED TABLES ---
# Emitted by src/counter_strike_render/vcs/gen_loopfam_tables.py from the
# decompiled modules under docs/projects/counter-strike-sft/
# glsl_loopfam/, whose sha256 are pinned in
# conformance/reference_sources.json. DO NOT HAND-EDIT: run
#   python3 -m counter_strike_render.vcs.gen_loopfam_tables --check
# which diffs this block against a fresh parse and exits 1 on any
# difference.

# ssao_r0m0.glsl:4  _1213 -- the 9 hemisphere kernel vectors; NOT unit length (0.0047 to 0.985), so
# the falloff lives in the magnitudes
SSAO_KERNEL_9 = np.array([
    [0.13390399515628814697265625, 0.0365430004894733428955078125, 0.049350000917911529541015625],
    [-0.12978200614452362060546875, -0.0326079986989498138427734375, 0.098058998584747314453125],
    [-0.1944600045680999755859375, -0.4137530028820037841796875, 0.8764560222625732421875],
    [-0.0387840010225772857666015625, 0.43998301029205322265625, 0.0596049986779689788818359375],
    [-0.118508003652095794677734375, 0.001674000057391822338104248046875, -0.011568999849259853363037109375],
    [0.0038580000400543212890625, -0.00258699990808963775634765625, -0.00047200001426972448825836181640625],
    [0.007021999917924404144287109375, 0.000399000011384487152099609375, -0.0054720002226531505584716796875],
    [-0.16810800135135650634765625, -0.2530579864978790283203125, -0.224711000919342041015625],
    [0.0112229995429515838623046875, -0.254115998744964599609375, -0.4668669998645782470703125]])

# ssao_bilateral_blur_r0m0.glsl:16  _487 -- 5 spatial weights; sum 0.5765, normalised at use, not at authoring
BILATERAL_5 = np.array([
    0.15317000448703765869140625,
    0.14489300549030303955078125,
    0.12264899909496307373046875,
    0.092901997268199920654296875,
    0.06296999752521514892578125])

# atrous_filter_r0m1.glsl:5  _2569 -- the 2x2 variance prefilter -- the (0.5, 0.25) outer product, stored
ATROUS_VARIANCE_2X2 = np.array([
    [0.25, 0.125],
    [0.125, 0.0625]])

# atrous_filter_r0m1.glsl:6  _1999 -- float32 (1, 2/3, 1/6), UNNORMALISED
ATROUS_TAP_3 = np.array([
    1.0,
    0.666666686534881591796875,
    0.16666667163372039794921875])

# blur_r6m2.glsl:16  _302 -- 5 BILINEAR tap positions; blur_with_depth:17 declares the same five
GAUSS5_OFFSETS = np.array([
    -3.0,
    -1.182425022125244140625,
    0.0,
    1.182425022125244140625,
    3.0])

# blur_r6m2.glsl:17  _754 -- 5 weights summing to 0.9999999878928065; renormalising would be a fit
GAUSS5_WEIGHTS = np.array([
    0.0044329999946057796478271484375,
    0.2960419952869415283203125,
    0.3990499973297119140625,
    0.2960419952869415283203125,
    0.0044329999946057796478271484375])

# denoise_blur_r1m0.glsl:16  _446 -- INTEGER texelFetch offsets, stride 3, spanning 13 texels
DENOISE_OFFSETS_5 = np.array([
    -6,
    -3,
    0,
    3,
    6], dtype=np.int64)

# outlines_r0m1.glsl:3  _2528 -- a ROTATED cross plus the centre -- four taps at distance sqrt(5), each
# the previous rotated 90 degrees, so no axis is preferred
OUTLINE_TAPS_5 = np.array([
    [2.0, 1.0],
    [1.0, -2.0],
    [-2.0, -1.0],
    [-1.0, 2.0],
    [0.0, 0.0]])

# panorama_alpha_r0m0.glsl:3  _1374 -- the 8-neighbour ring; THREE entries are SPLATS in the source
PANORAMA_RING_8 = np.array([
    [-1.0, -1.0],
    [-1.0, 0.0],
    [-1.0, 1.0],
    [0.0, 1.0],
    [1.0, 1.0],
    [1.0, 0.0],
    [1.0, -1.0],
    [0.0, -1.0]])

# tools_terrain_composite_slope_r0m0.glsl:5  _863 -- the same eight offsets AS A SET as PANORAMA_RING_8, DIFFERENT order --
# not interchangeable in code that indexes by i
SLOPE_RING_8 = np.array([
    [-1.0, -1.0],
    [0.0, -1.0],
    [1.0, -1.0],
    [1.0, 0.0],
    [1.0, 1.0],
    [0.0, 1.0],
    [-1.0, 1.0],
    [-1.0, 0.0]])

# player_visibility_r0m1.glsl:4  _2034 -- 16 weights summing to 1.0000000074505806 -- the only kernel here
# within 1e-8 of unity, but NOT exactly 1
PLAYER_VIS_WEIGHTS_16 = np.array([
    0.0228166244924068450927734375,
    0.0483712442219257354736328125,
    0.0483712442219257354736328125,
    0.0228166244924068450927734375,
    0.0483712442219257354736328125,
    0.0994804799556732177734375,
    0.130440890789031982421875,
    0.07933165132999420166015625,
    0.0483712442219257354736328125,
    0.0994804799556732177734375,
    0.130440890789031982421875,
    0.07933165132999420166015625,
    0.0228166244924068450927734375,
    0.0483712442219257354736328125,
    0.0483712442219257354736328125,
    0.0228166244924068450927734375])

# visualize_depth_r0m0.glsl:3  _2728 -- red, green, blue; indexed i mod 3
VISDEPTH_COLOURS_A = np.array([
    [1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0],
    [0.0, 0.0, 1.0]])

# visualize_depth_r0m0.glsl:4  _2729 -- IDENTICAL to _2728 in the artifact. Two declarations, two terms, two
# copies -- deduplicating stops detecting the day they diverge
VISDEPTH_COLOURS_B = np.array([
    [1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0],
    [0.0, 0.0, 1.0]])

# generic_r69m3.glsl:48  _2918 -- the 4x4 identity written out; kept as a table, not np.eye, because the
# term exists to check that the file says what we say
IDENTITY_ROWS_4 = np.array([
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, 1.0]])

# --- END GENERATED TABLES ---


# ---------------------------------------------------------------------------
# Row accessors -- the single read path
# ---------------------------------------------------------------------------
# These were monkey-patched onto this module by the conformance terms file,
# which meant the renderer could not call them and the whole module read
# DEAD in the wiring audit while its constants were plainly in use. Defined
# here instead, and the renderer indexes through them, so one function is
# both what the A/B measures and what the frame runs.
def _row(table, index):
    """One row of a read table.

    A DEVICE INDEX MOVES THE TABLE, NOT THE OTHER WAY. Indexing a numpy
    table with a CUDA tensor routes through Tensor.__array__ and raises;
    copying the index to host would work and would serialise the frame on
    a host round-trip per lookup. The table is a handful of floats, so it
    goes to the device instead.
    """
    if hasattr(index, "device"):
        return np.like(table, index * 1.0)[np.int64(index)]
    return table[int(index)]

def row_ssao_kernel_9(index):
    return _row(SSAO_KERNEL_9, index)

def row_bilateral_5(index):
    return _row(BILATERAL_5, index)

def row_atrous_variance_2x2(index):
    return _row(ATROUS_VARIANCE_2X2, index)

def row_atrous_tap_3(index):
    return _row(ATROUS_TAP_3, index)

def row_gauss5_offsets(index):
    return _row(GAUSS5_OFFSETS, index)

def row_gauss5_weights(index):
    return _row(GAUSS5_WEIGHTS, index)

def row_denoise_offsets_5(index):
    return _row(DENOISE_OFFSETS_5, index)

def row_outline_taps_5(index):
    return _row(OUTLINE_TAPS_5, index)

def row_panorama_ring_8(index):
    return _row(PANORAMA_RING_8, index)

def row_slope_ring_8(index):
    return _row(SLOPE_RING_8, index)

def row_player_vis_weights_16(index):
    return _row(PLAYER_VIS_WEIGHTS_16, index)

def row_visdepth_colours_a(index):
    return _row(VISDEPTH_COLOURS_A, index)

def row_visdepth_colours_b(index):
    return _row(VISDEPTH_COLOURS_B, index)

def row_identity_rows_4(index):
    return _row(IDENTITY_ROWS_4, index)

