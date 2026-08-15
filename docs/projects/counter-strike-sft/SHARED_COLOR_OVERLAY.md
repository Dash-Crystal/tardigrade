# `g_tSharedColorOverlay` — what the draw call actually does

Read from the shipping shader on 2026-08-09. Until this date the overlay's
consumption expression **did not exist in this repo**: it lives only in
static combo **258** of `csgo_environment_blend_vulkan_50_ps.vcs`, and
`SHADER_CALLFLOW_csgo_environment_blend.md` §6.1 records that combo 258
"was not extracted". The pinned `csgo_environment_blend_ps.glsl` is combo
**0** — every static feature off, `S_SHARED_COLOR_OVERLAY` among them —
so grepping it for the overlay could not have found a positive, and did
not.

Combo 258 is now emitted, pinned as
`csgo_environment_blend_ps_c258_d0.glsl` (2,007 lines, sha256
`d44253733fcc7aa0e13f703ed04739af3af61f473d3be33efd251bb042aab4ff`), and
reproducible with `harness/gpu_render/vcs/eb_combo.py`. Combos 0 and 256
come out of the same run at the sha256s already on record
(`af79406e…`, `6cb33c49…`), which is what validates the index chain.

## Naming: byte offsets, never `_mN`

spirv-cross renumbers the material block per combo. The join below is
read from the shader's own `m_allVars` / `m_dynamicComboVars`, which pack
`(variableIndex << 16) | registerOffset` with the offset in **4-byte
words** and `0xFFFF` for "not laid out"; `word * 4` is the
`layout(offset = N)` byte offset in the GLSL.

| byte | c258 member | name |
|---|---|---|
| 36 | `_m4` | `g_tColor1` |
| 420 | `_m17` | `g_tNormal1` |
| 424 | `_m18` | `g_tHeight1` |
| 1036 | `_m80` | `g_tSharedColorOverlay` |
| 1040 | `_m81` | `g_nColorOverlayUVSet` |
| 1048 / 1052 | `_m82` / `_m83` | `g_bColorOverlayLayer1` / `…Layer2` |
| 1064 / 1068 | `_m84` / `_m85` | `g_bColorOverlayMaskLayer1` / `…Layer2` |
| 1100 | `_m86` | `g_flOverlayDarknessContrast` |
| 1104 | `_m87` | `g_flOverlayBrightnessContrast` |
| 1116 | `_m88` | `g_vOverlayTexCoordScale` |

The three control-check members are independently confirmed by their
fetch sites: 36 is the albedo index, 420 the normal index, 424 the index
the blend family samples its tint mask from.

## The expression — `csgo_environment_blend_ps_c258_d0.glsl:1188-1221`

`w` is the layer-2 blend weight (`_6253`), `wInv = clamp(1-w)` (`_24399`),
`tm1`/`tm2` the per-layer tint-mask values (`_24803`, `_6617`).

```glsl
// :1188  strength
float k = ((g_bColorOverlayMaskLayer1 != 0) ? tm1 : 1.0) * (float(g_bColorOverlayLayer1 != 0) * wInv)
        + ((g_bColorOverlayMaskLayer2 != 0) ? tm2 : 1.0) * (float(g_bColorOverlayLayer2 != 0) * w);
if (k > 0.0) {                                                   // :1190
    vec4 t;
    if (g_nColorOverlayUVSet == 0)                               // :1192  Biplanar
        t = textureGrad(g_tSharedColorOverlay, uvA * g_vOverlayTexCoordScale) * wA
          + textureGrad(g_tSharedColorOverlay, uvB * g_vOverlayTexCoordScale) * wB;
    else                                                         // :1209  UV1 / UV2
        t = texture(g_tSharedColorOverlay, vOverlayTexCoord.xy);
    vec3 s = (t * 2.0 - 1.0).xyz;                                // :1211  SIGNED
    vec3 f = max(vec3(0.0),                                      // :1212
                 (1.0 - pow(1.0 - max(vec3(0.0), s), vec3(B))) * B
               + (pow(1.0 + min(vec3(0.0), s), vec3(D)) - 1.0) * D
               + 1.0);
    rgb = rgb * mix(vec3(1.0), f, vec3(k));                      // :1217
}
```

`B = g_flOverlayBrightnessContrast`, `D = g_flOverlayDarknessContrast`.

**It is a MULTIPLY by a factor around 1.0, not a Photoshop overlay.**
Endpoints, evaluated: `s = 0` (mid-grey texel) gives `f = 1` exactly, so
a mid-grey page is the identity and the pack's `AUX_HALF` sentinel is the
right no-op; `s = +1` gives `f = 1 + B`; `s = -1` gives `f = 1 - D`. On
`inferno_plaster_facade_01_basic` (B 0.20, D 0.65) the overlay can scale
albedo over `[0.35, 1.20]`.

`B` and `D` are **exponents as well as scales** — each appears twice in
its own half. Treating them as blend weights, which the renderer's
`shared_overlay()` did, is a different function of the same two numbers.

Applied to `_13381`: the layer-blended albedo after the
`S_BLEND_EFFECTS_2` colour terms, **before** the vertex-colour tint at
`:1223`. Both are pure multiplies on rgb, so their order does not change
the result.

## The two enum-derived booleans

`g_nColorOverlayMode` and `g_nColorOverlayTintMask` are **not** uniform
members — they are `m_sourceType = 6` expression sources that produce the
`g_b…` booleans above. Their compiled expressions are 36 bytes and differ
across Layer1/2/3 in exactly one float constant (`1.0` / `2.0` / `3.0`),
which with the declared enum labels

    g_nColorOverlayMode      0=All Layers, 1=Layer1, 2=Layer2, 3=Layer3
    g_nColorOverlayTintMask  0=Mask All, 1..3=Mask LayerN, 4=Unmasked

gives

    g_bColorOverlayLayerN     = (mode == 0) || (mode == N)
    g_bColorOverlayMaskLayerN = (tintMask == 0) || (tintMask == N)

That is a **derivation from the enum labels plus the one-constant delta**,
not a decode of the expression VM. It is the only reading consistent with
both, and the falsifier is cheap: an expression decoder, or a capture.

Consequence worth stating plainly: `g_nColorOverlayTintMask = 4` means
**UNMASKED**. `gpu_render.py`'s `shared_overlay()` read the value 4 as
"gate the overlay by the tint mask" — the exact opposite of what the
material asks for, on the 8 of 11 de_inferno materials that set it.

## What de_inferno's 11 overlay materials actually set

3 distinct pages (`inferno_wall_overlay_03` on 9, `_01` and `_04` on one
each). `g_nColorOverlayUVSet` is **unset on all 11**, so all 11 take the
default `2 = UV2` and the single-fetch branch at `:1209`; the biplanar
two-tap branch is dead on this map. `g_vOverlayTexCoordScale` is `[1,1]`
and `g_flOverlayTexCoordRotation` `0` on all 11, so the vertex stage's
overlay transform is the identity and the fetch is at plain UV2.
`g_nColorOverlayMode` is 1 on 10 and 0 on one; `g_nColorOverlayTintMask`
is 4 on 8 and 0 on 3. `B` spans 0.20–0.868 and `D` 0.60–1.30 — real
per-material numbers, read from the vmat, never fitted.

For `inferno_plaster_facade_01_basic` (mode 1, tintMask 4) the strength
reduces to `k = 1 - w`: full overlay on layer 1, none on layer 2.
