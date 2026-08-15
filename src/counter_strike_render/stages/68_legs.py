

# ----------------------------------------------------------------------
# csgo_legs_prepass -- the WHOLE module, all 34 lines
# ----------------------------------------------------------------------
def legs_prepass_shade(wpos, lpos, eye, covered, x):
    """csgo_legs_prepass_vulkan_50_ps.vcs, 1 combo, 1 record, 1 module.

    The entire shader body is one line (csgo_legs_prepass__only_d0:28) plus
    one constant store (:32). Transcribed complete:

        fade = smoothstep(F - 35.0, F,
                          length(camPos - (worldPos + viewOrigin)))
             * step(0.8, dot(normalize(localPos - vec3(0,0,55)),
                             vec3(0,0,-1)));
        if (fma(fade * 3.0, 2.0, -1.5) + dither.y < 0.0) discard;
        out = vec4(0.01, 0.01, 0.01, 1.0);

    Two literal facts worth naming because they are easy to invert:

      * F is a DISTANCE, not an opacity. smoothstep(F-35, F, dist) is 0
        when the camera is CLOSER than F-35 to the fragment and 1 when it
        is further than F, so the legs dissolve as the camera closes on
        them -- which is what a first-person legs pass needs, since the
        thighs pass through the near plane when you look down.
      * The cone test is anchored at the model's eye height
        (0, 0, 55 Source units) and points straight DOWN (0, 0, -1) with
        a cos threshold of 0.8, i.e. acos(0.8) = 36.87 degrees. Only the
        part of the legs you can see by looking down survives.

    `fade * 3.0` before the screen-door means keep-probability is
    clamp(6*fade - 0.5, 0, 1): fade <= 1/12 always discarded, fade >= 0.25
    always kept.

    Returns (rgb, keep) -- rgb is the shader's own constant.
    """
    F = _e2(x, "legs_fade_dist")
    cone_cos = _e2(x, "legs_cone_cos")
    eye_z = _e2(x, "legs_eye_z")
    # `/ S` converts to SOURCE UNITS. g_flFirstpersonLegsFade and the
    # literal 35.0 are both in the shader's world units, and this
    # renderer's world is in metres (S = 0.0254 m/unit). The same
    # conversion is already used for the fog distance elsewhere in this
    # file. Scaling the LITERAL instead would have corrupted the
    # transcription; converting the input keeps it verbatim.
    dist = (eye[:, None, None, :] - wpos).norm(dim=-1) / S
    # csgo_legs_prepass__only_d0:28, the whole varying part of the module.
    # cone_cos and eye_z arrive as material columns whose defaults ARE the
    # shader's literals (0.8, 55.0), so this call is the reference.
    fade = wpn.legs_fade(F, dist, lpos, cone_cos=cone_cos, eye_z=eye_z)
    cone = (fade > 0).float()
    keep = vm_screendoor(fade * 3.0, covered)
    rgb = torch.full(wpos.shape[:-1] + (3,), 0.00999999977648258209228515625,
                     device=wpos.device, dtype=torch.float32)
    # denominators are the pass's own covered pixels; a mask that is not
    # AND-ed with `covered` counts pixels the pass never touched, which is
    # how the first version of this report printed 597%.
    n = int(covered.sum())
    _vm_reach("legs_cone", covered & (cone > 0), n)
    _vm_reach("legs_faded_out", covered & (fade * 3.0 < 0.25 / 3.0), n)
    _vm_reach("legs_kept", keep, n)
    return rgb, keep


# ----------------------------------------------------------------------
# csgo_weapon -- the capability set
# ----------------------------------------------------------------------
_LUMA = (0.2125000059604644775390625, 0.7153999805450439453125,
         0.07209999859333038330078125)


def _luma(c):
    w = torch.tensor(_LUMA, device=c.device, dtype=c.dtype)
    return (c * w).sum(-1)


def _octa_normal(a, b):
    """The two-channel normal decode every csgo_weapon module uses.

    reference/csgo_weapon/csgo_weapon__allon_c3041_d2.glsl:540-542. The
    expression itself is `wpn.octa_normal_decode`, which the conformance
    registry A/Bs against the decompiled line range
    (csgo_weapon.t01.octa_normal_decode). This wrapper exists only to keep
    the renderer's two-scalar call site.
    """
    return wpn.octa_normal_decode(a, b)


def weapon_sfx_mask(uv, wpos, lpos, nrm_ts, albedo, rough, tsec, x, tex):
    """S_ENABLE_SFX_MASK -- the scrolling flow-map distortion.

    MEASURED FIRST, because it changes what this function IS: combo 128
    (S_ENABLE_SFX_MASK=1) and combo 0 resolve to bytecode record 0 --
    the SAME record, so the axis emits no pixel-shader instruction. The
    whole effect is compiled into every module and gated at runtime by
    `g_flSfxAmount * perViewScale > 0` (allon_c3041_d2:451-452). The static
    axis selects the material binding, not the code path.

    The path itself, baseline UNSOURCED-glsl:242-267 and :372-393:

        uvFlow = uv * 2.5, x3 where the UV density test
                 |cross(dFdx(uv),dFdy(uv))| / |cross(dFdx(P),dFdy(P))|
                 < 0.002 (a coarse-UV surface gets a finer flow map)
        F      = texture(flowMap, uvFlow + P.xy * 1e-4)
        flow   = F.xy * 2 - 1 ; flow.y = -flow.y
        phase  = F.z + localPos.x * 0.01
        wave   = clamp(((amt*0.25 - fract(phase + speed*0.1*k*k)) * 5)
                       / (k + 0.001), 0, 1)
                 * clamp((N.z + 0.75) * 4, 0, 1)
        uv    += flow * -0.02 * wave * F.w

    and then, with s = wave * F.w:

        albedo  = mix(A, pow(A, 1.6) * 0.6, edge)     edge from mask.x
        rough   = mix(rough, 0.1, mask.x * clamp(...))
        normal  = mix(..., flow-space normal, s)

    `k` is the per-view SFX scale from the per-view CB (_5037._m13 at
    allon_c3041_d2:451), which is engine data; --weapon-sfx-mask supplies it as 1.
    """
    amt = _e2(x, "wpn_sfx_amount")
    spd = _e2(x, "wpn_sfx_speed")
    k = 1.0  # the per-view SFX scale, engine CB data
    a = amt * k
    live = a > 0.0
    if not bool(live.any()):
        return albedo, rough, nrm_ts, uv, torch.zeros_like(a)
    F = tex["flow"]
    uvf = uv * 2.5 + wpos[..., :2] * 9.9999997473787516355514526367188e-05
    Fs = vm_sample(F, uvf)
    flow = Fs[..., :2] * 2.0 - 1.0
    flow = torch.stack([flow[..., 0], -flow[..., 1]], dim=-1)
    fw = Fs[..., 3]
    phase = Fs[..., 2] + lpos[..., 0] * 0.00999999977648258209228515625
    t = tsec.view(-1, 1, 1) if torch.is_tensor(tsec) else float(tsec)
    wave = (((a * 0.25 - torch.frac(phase + spd * 0.1 * k * k * t)) * 5.0)
            / (a + 0.001000000047497451305389404296875)).clamp(0.0, 1.0)
    wave = wave * ((nrm_ts[..., 2] + 0.75) * 4.0).clamp(0.0, 1.0)
    uv2 = uv + flow * (-0.0199999995529651641845703125) * wave.unsqueeze(-1) \
        * fw.unsqueeze(-1)
    s = wave * fw
    a01 = a.clamp(0.0, 1.0)
    edge = ((s + a01 * 0.5).clamp(0.0, 1.0)).unsqueeze(-1)
    albedo = albedo + (albedo.clamp(min=1e-6).pow(1.60000002384185791015625)
                       * 0.60000002384185791015625 - albedo) * edge
    rough = rough + (0.100000001490116119384765625 - rough) \
        * (s * 4.0 + a * 0.4000000059604644775390625).clamp(0.0, 1.0)
    fz = (1.0 - (flow * flow).sum(-1).clamp(0.0, 1.0)).clamp(min=0.0).sqrt()
    n_flow = torch.cat([flow, fz.unsqueeze(-1)], dim=-1)
    nrm_ts = nrm_ts + (n_flow - nrm_ts) * s.unsqueeze(-1)
    nrm_ts = nrm_ts / nrm_ts.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    return albedo, rough, nrm_ts, uv2, s


def weapon_adjustments(albedo, view, nrm_w, sun_dir, mask_z, x):
    """S_ENABLE_ADJUSTMENTS -- the two view-angle colour adjustments.

    Both are compiled into every module and gated by their own uniform.

    (a) hue ROTATION, allon_c3041_d2:709-723. Rodrigues about the grey
        axis vec3(0.57735002040863037109375):

            th  = hue * (1 - dot(V, N)) * mask.z
            rot = C*cos(th) + cross(g, C)*sin(th) + g*dot(g,C)*(1-cos(th))
            sat = (max(C) - min(C)) / max(C),  0 when max(C) == 0
            out = mix(vec3(luma(C)), rot, pow(sat, 0.125))

        The pow(sat, 0.125) blend does the OPPOSITE of what it looks
        like, and the comment here used to say the opposite: an EIGHTH
        root is ~0.84 at saturation 0.25 and ~0.56 at 0.01, so a
        near-grey pixel takes MOST of the rotation. Only an exactly grey
        pixel (sat == 0, the branch at :714-717) is exempt. The curve is
        a hard cutoff at zero, not a soft protection of desaturated
        pixels.

    (b) the 6-segment rainbow ramp, allon_c3041_d2:5759-5800:

            h   = fract((dot(V,N) + dot(V,sunDir)) * freq + phase) * 6
            i   = floor(h), f = h - i, and the six segments are
                  (1,f,0) (1-f,1,0) (0,1,f) (0,1-f,1) (f,0,1) (1,0,1-f)
            out = clamp(mix(C, normalize(max(ramp,1e-3))
                            * min(L/luma, 3L*max(ramp)), amount * mask.z),
                        0, 1)

        where L is the incoming luminance -- the ramp is renormalised to
        preserve it rather than replacing brightness with hue.
    """
    hue = _e2(x, "wpn_hue_shift")
    C = albedo
    # allon_c3041_d2:709, :714-721, :723 -- angle, saturation, rotation.
    # `view` is already the normalised fragment-to-camera direction, which
    # is what the line's `normalize(camPos - worldPos)` produces.
    th = wpn.hue_angle(hue, view, nrm_w, mask_z)
    out = wpn.hue_rotate(C, th, wpn.hsv_saturation(C))
    out = torch.where((hue != 0.0).unsqueeze(-1), out, C)
    # --- (b) the ramp
    freq = _e2(x, "wpn_ramp_freq")
    phase = _e2(x, "wpn_ramp_phase")
    amount = _e2(x, "wpn_ramp_amount")
    # STATED DIFFERENCE, and it is in the PHASE, not the ramp. The ramp
    # itself is transcribed (allon_c3041_d2:5760-5800, term t27). What
    # feeds it at :5759 is
    #     fract((dot(a, N) + dot(a, normalize(<a luma-weighted difference
    #            of four sampled colours, plus a per-view vector>)))
    #           * freq + phase) * 6.0
    # and this uses `dot(view, N) + dot(view, sun_dir)` in place of that
    # inner normalize. The ramp is conformant; the coordinate it is
    # sampled at is NOT, and saying so here is the difference between a
    # stated difference and a claim.
    h = torch.frac(((view * nrm_w).sum(-1)
                    + (view * sun_dir).sum(-1)) * freq + phase) * 6.0
    ramp = wpn.rainbow_ramp(h)
    L = _luma(out)
    rn = ramp.clamp(min=0.001000000047497451305389404296875)
    rn = rn / rn.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    scale = torch.minimum(L / _luma(rn).clamp(min=1e-6),
                          3.0 * L * ramp.max(-1).values)
    w2 = (amount * mask_z).unsqueeze(-1)
    out2 = (out * (1.0 - w2) + rn * scale.unsqueeze(-1) * w2).clamp(0.0, 1.0)
    return torch.where((amount != 0.0).unsqueeze(-1), out2, out)


def weapon_glitter(uv, view, T, Bt, N, nrm_ts, albedo, rough, metal,
                   mask_z, x, tex, per_view_hi):
    """S_GLITTER (stride 64, record 24).

    UNSOURCED csgo_weapon__glitter1_c64.glsl:424-466, composited at :1305 / :1317.

        amt = mask.z * min(1, g_flGlitterScale)
        gUV = uv * ((perViewHi ? 2.5 : 1.75) * uvScale * scale)
        fw  = max(dFdx(gUV), dFdy(gUV))
        gN  = octaDecode(texture(glitterNormal, gUV))     .w is its mask
        gNz = gN * gN.z
        R   = reflect(V, normalize(T*(gNz+n).x + B*(gNz+n).y + N*(gNz+n).z))
        s   = sin(R * mix(12.0, 5.6, spread))
        hi  = mix(0.99, 0.80, spread) ; inv = 1/(1-hi)
        p   = max(0, s - hi) * inv ; q = max(0, -s - hi) * inv
        C   = pow(dot(clamp(-s,0,1), LUMA), 4)
            + dot(clamp(0.15 - s, 0, 1), LUMA) * max(0, balance)
            + (p + pow(p.yzx + q*max(0,-balance),
                       vec3(4 - 3.5*max(0, spread)))) * 4 * max(0, 1-balance)
        m   = gN.w ; strength = amt * m * |1 - gNz.z|
        glitterRGB = ((C*0.05 + C*(normalize(max(3e-4, albedo))*1.06)*0.95)
                      * m) * dot(V, N) * scale * mask.z
        rough  *= 1 - strength*0.25
        albedo *= mix(1 + strength*2.5, 1, smoothstep(0, 0.8, luma(albedo)))
        metal   = max(metal, strength*0.5*amt)
        n      += gNz * (0.04*amt * clamp(1 - min(fw.x,fw.y)*40, 0, 1))

    and the radiance is ADDED to the indirect term at :1321, not blended
    into albedo -- a sparkle that survives being in shadow is wrong, and
    the shader avoids it by multiplying through the diffuse product.

    `dFdx/dFdy` of the glitter UV is the anti-aliasing guard: when a flake
    is smaller than a pixel the normal perturbation is faded out. This
    renderer has no per-pixel derivative at this point, so the footprint
    is taken from the screen-space UV difference of neighbouring pixels
    with torch.diff, which is the same quantity computed one pixel later.
    """
    scale = _e2(x, "wpn_glitter_scale")
    uvs = _e2(x, "wpn_glitter_uv")
    spread = _e2(x, "wpn_glitter_spread")
    bal = _e2(x, "wpn_glitter_balance")
    amt = mask_z * scale.clamp(max=1.0)
    live = amt != 0.0
    if not bool(live.any()):
        z = torch.zeros_like(albedo)
        return z, albedo, rough, metal, nrm_ts
    gs = ((2.5 if per_view_hi else 1.75) * uvs * scale).unsqueeze(-1)
    gUV = uv * gs
    dx = torch.zeros_like(gUV)
    dy = torch.zeros_like(gUV)
    dx[:, :, :-1] = gUV[:, :, 1:] - gUV[:, :, :-1]
    dy[:, :-1] = gUV[:, 1:] - gUV[:, :-1]
    fw = torch.maximum(dx, dy)
    G = vm_sample(tex["glitter"], gUV)
    gN = _octa_normal(G[..., 0], G[..., 1])
    gNz = gN * gN[..., 2:3]
    comb = gNz + nrm_ts
    W = T * comb[..., 0:1] + Bt * comb[..., 1:2] + N * comb[..., 2:3]
    W = W / W.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    R = view - 2.0 * (view * W).sum(-1, keepdim=True) * W
    s = torch.sin(R * (12.0 + (5.599999904632568359375 - 12.0)
                       * spread).unsqueeze(-1))
    hi = (0.9900000095367431640625
          + (0.800000011920928955078125 - 0.9900000095367431640625) * spread)
    inv = 1.0 / (1.0 - hi).clamp(min=1e-6)
    p = (s - hi.unsqueeze(-1)).clamp(min=0.0) * inv.unsqueeze(-1)
    q = (-s - hi.unsqueeze(-1)).clamp(min=0.0) * inv.unsqueeze(-1)
    pb = bal.clamp(min=0.0)
    term1 = _luma((-s).clamp(0.0, 1.0)).pow(4.0).unsqueeze(-1).expand_as(s)
    term2 = (_luma((0.14999997615814208984375 - s).clamp(0.0, 1.0))
             * pb).unsqueeze(-1).expand_as(s)
    ex = (4.0 - 3.5 * spread.clamp(min=0.0)).unsqueeze(-1)
    inner = (q.roll(0, -1) * (-bal).clamp(min=0.0).unsqueeze(-1))
    term3 = (p + (p.roll(-1, -1) + inner).clamp(min=0.0).pow(ex)) * 4.0 \
        * (1.0 - bal).clamp(min=0.0).unsqueeze(-1)
    C = term1 + term2 + term3
    m = G[..., 3]
    strength = amt * m * (1.0 - gNz[..., 2]).abs()
    an = albedo.clamp(min=0.0003000000142492353916168212890625)
    an = an / an.norm(dim=-1, keepdim=True).clamp(min=1e-8) * 1.059999942779541015625
    ndv = (view * N).sum(-1)
    glit = ((C * 0.0500000007450580596923828125 + C * an
             * 0.949999988079071044921875) * m.unsqueeze(-1)) \
        * ndv.unsqueeze(-1) * scale.unsqueeze(-1) * mask_z.unsqueeze(-1)
    rough = rough * (1.0 - strength * 0.25)
    lt = _luma(albedo)
    t01 = (lt / 0.800000011920928955078125).clamp(0.0, 1.0)
    sm = t01 * t01 * (3.0 - 2.0 * t01)
    albedo = albedo * ((1.0 + strength * 2.5) * (1.0 - sm)
                       + 1.0 * sm).unsqueeze(-1)
    metal = torch.maximum(metal, strength * 0.5 * amt)
    aa = (1.0 - torch.minimum(fw[..., 0], fw[..., 1]) * 40.0).clamp(0.0, 1.0)
    nrm_ts = nrm_ts + gNz * (0.039999999105930328369140625 * amt * aa
                             ).unsqueeze(-1)
    nrm_ts = nrm_ts / nrm_ts.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    return glit, albedo, rough, metal, nrm_ts


def weapon_stickers(uv_s, view, T, Bt, N, nrm_ts, albedo, rough, x, tex,
                    cam_up, cam_fwd,
                    n_slots):
    """S_STICKERS (stride 512, record 96) -- FIVE slots.

    The axis adds 3,565 decompiled lines and 81 fetch call-sites over the
    baseline. The added region is five near-identical bodies at a
    27-uniform stride in the flattened material CB -- bases m28, m55, m82,
    m109, m136, with the per-slot sticker albedo at base+9 (m37, m64, m91,
    m118, m145) and the per-slot sticker normal at base+4. The five slot
    bodies begin at UNSOURCED-glsl:752, :1455, :2158, :2861, :3564.

    One slot, transcribed (UNSOURCED-glsl:671-760, slot 0):

        wear = texture(wearMask, uv).x                          (:571)
        gate = wear > 0 && stickerUV.z > -1                      (:574-576)
        p    = ((stickerUV.zw - 0.5) - offset) * |scale|.x       (:671)
        th   = rotation * 6.28318023681640625                    (:672)
        q    = rot2(p, th) + 0.5                                 (:677)
        reject if q outside [0,1]                                (:680-688)
        A    = texture(stickerAlbedo, clamp(q,0,1))              (:710)
        a    = clamp(A.w * 12.75, 0, 1) * wear                   (:712)
        lod  = textureQueryLod(stickerAlbedo, q).x               (:717)
        near = 1 - clamp(lod - 3, 0, 1)                          (:719)
        if near > 0:  a = mix(a, max(a, clamp(texture(stickerAlbedo,
                      q - viewTangentOffset*0.01).w * 12.75, 0, 1)
                      * wear * 0.7), near)                       (:727)
        sN   = octaDecode(texture(stickerNormal, q)); sN.y = -sN.y (:744-750)
        bN   = octaDecode(textureLod(baseNormal, uv, max(lod,3))) (:752-758)
        n    = normalize(bN + sN * a * 2)                        (:759)
        rgh  = min(pow(textureLod(baseAO, uv, lod).x, 0.75),
                   a*0.5 + 0.5)                                  (:760)

    The 12.75 is 255/20: a sticker's alpha channel is authored with a
    1/12.75 = 0.0784 soft edge and the shader hardens it. The `q - view
    tangent offset * 0.01` tap at :727 is a parallax: the sticker sits on
    a decal layer above the surface, so at close range it shifts against
    the view direction, and `near` fades that off past LOD 3.

    STATED DIFFERENCE: textureQueryLod has no equivalent here, so `lod` is
    taken from log2 of the screen-space sticker-UV footprint, which is the
    quantity textureQueryLod returns for an isotropic footprint. The five
    slots share one synthesised sticker sheet under --viewmodel-synth
    (there is no per-slot sticker asset in this pack); each slot gets its
    own transform, so the five bodies are five distinct evaluations.
    """
    wear = vm_sample(tex["wear"], uv_s[..., :2])[..., 0]
    gate = (wear > 0.0) & (uv_s[..., 2] > -1.0)
    scale = _e2(x, "wpn_stk_scale")
    rot = _e2(x, "wpn_stk_rot")
    ou = _e2(x, "wpn_stk_off_u")
    ov = _e2(x, "wpn_stk_off_v")
    holo = _e2(x, "wpn_stk_holo")
    n_slots = int(max(0, min(n_slots, int(_e2(x, "wpn_sticker_slots")
                                          .max().item() if x.numel() else 5))))
    acc_a = torch.zeros_like(wear)
    for k in range(n_slots):
        # each slot has its OWN 27-uniform block; on a synthesised draw
        # the per-slot transform is derived deterministically from the
        # slot index so the five bodies are five distinct evaluations
        # rather than five copies of one.
        sk = scale * (1.0 + 0.17 * k)
        rk = rot + k * 0.2
        # allon_c3041_d2:829-835 -- the `.zw` texcoord pair, abs() on the
        # scale, and 6.28318023681640625 (NOT float32 2*pi) for turns.
        off_k = torch.stack([ou + 0.11 * k, ov - 0.07 * k], dim=-1)
        q = wpn.sticker_uv(uv_s[..., 2:4], off_k, sk, rk)
        # :838-845 is a CLAMP-INEQUALITY, not `q >= 0 && q <= 1`. The two
        # agree everywhere except NaN, which this form rejects and the
        # comparison form keeps.
        inside = (~wpn.sticker_outside(q)) & gate
        if not bool(inside.any()):
            _wpn_reach(f"stickers.slot{k}", inside, inside.numel())
            continue
        qc = q.clamp(0.0, 1.0)
        A = vm_sample(tex["sticker"], qc)
        a = wpn.sticker_alpha(A[..., 3], wear)              # :869-870
        # LOD from the screen-space footprint of the sticker UV
        dx = torch.zeros_like(qc)
        dx[:, :, :-1] = qc[:, :, 1:] - qc[:, :, :-1]
        foot = (dx.norm(dim=-1) * tex["sticker"].shape[-2]).clamp(min=1e-6)
        lod = torch.log2(foot)
        near = wpn.sticker_lod_fade(lod)                    # :876-877
        # allon_c3041_d2:882-885. THIS WAS THE WRONG BASIS: the parallax
        # was built from the TANGENT frame, (dot(V,B), dot(V,T)), and was
        # not rotated. The module builds it from the same two PER-VIEW
        # vectors the refraction uses -- cross(a,b) against the NEGATED
        # normal for x, b against the normal for y -- and then rotates it
        # by the slot's OWN angle, so the holo shift follows the decal
        # rather than the screen.
        qp = wpn.sticker_holo_uv(qc, cam_fwd, cam_up, N, rk)
        Ap = vm_sample(tex["sticker"], qp)
        a = wpn.sticker_holo_alpha(
            a, wpn.sticker_alpha(Ap[..., 3], wear), near)       # :885
        a = torch.where(inside, a, torch.zeros_like(a))
        sN = _octa_normal(A[..., 0] * 0.0 + A[..., 0], A[..., 1])
        sN = torch.stack([sN[..., 0], -sN[..., 1], sN[..., 2]], dim=-1)
        nrm_ts = nrm_ts + sN * (a * 2.0).unsqueeze(-1)
        nrm_ts = nrm_ts / nrm_ts.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        albedo = albedo + (A[..., :3] - albedo) * a.unsqueeze(-1)
        rough = torch.minimum(rough.clamp(min=1e-6).pow(0.75), a * 0.5 + 0.5)
        if bool((holo > 0).any()):
            # the holo-foil kernel at :789-791: the same sin(reflect)
            # sparkle as S_GLITTER but with the constants FROZEN --
            # 12.0, 0.99, 0.25, 0.75 -- rather than driven by uniforms.
            R = view - 2.0 * (view * N).sum(-1, keepdim=True) * N
            sg = torch.sin(R * 12.0)
            p2 = (sg - 0.9900000095367431640625).clamp(min=0.0) \
                * 100.00009918212890625
            Ch = (_luma((-sg).clamp(0.0, 1.0)).pow(4.0).unsqueeze(-1)
                  + (_luma((0.14999997615814208984375 - sg).clamp(0.0, 1.0))
                     * 0.25).unsqueeze(-1)
                  + (p2 + p2.roll(-1, -1).pow(4.0)) * 4.0 * 0.75)
            albedo = albedo + Ch * (a * holo).unsqueeze(-1)
        acc_a = torch.maximum(acc_a, a)
        _wpn_reach(f"stickers.slot{k}", inside & (a > 0), inside.numel())
    return albedo, rough, nrm_ts, acc_a


def weapon_opaque_refract(frag_xy, view, Wn, cam_up, cam_fwd, albedo,
                          uv, x, tex, viewport, sticker_a=None):
    """S_OPAQUE_REFRACT + D_USE_BGREFRACT.

    THIS WAS TRANSCRIBED FROM A MODULE THAT DOES NOT CONTAIN IT. db11cafd
    read the refract path out of static combo 2048 at dynamic combo 2. That
    module's SPIR-V has sha256
    838cd370e8c6095ec76a973f9202437a279c125f636e3805d7c83821257c9b49 --
    byte for byte the BASELINE's, combo 0. S_OPAQUE_REFRACT emits no
    instruction of its own; it is the PERMISSION, and D_USE_BGREFRACT is
    the code. That dynamic axis ships only under S_OPAQUE_REFRACT, 128
    statics x 16 dynamic ids = the 2,048 pairs the container reports.

    Re-transcribed from the two committed modules that DO contain it:

        reference/csgo_weapon/csgo_weapon__allon_c3041_bgr1_d34.glsl
            :5534   the screen-space UV offset
            :5538-5543  the five-tap blur
            :5549   the mask fetch
            :5550   the blend
        reference/csgo_weapon/csgo_weapon__opaque_refract1_c2048_bgr1_d34.glsl
            :512, :516-521, :527, :528 -- the same expressions under
            DIFFERENT member numbers (_m29/_m30/_m32 against
            _m201/_m202/_m204) and different per-view members for the same
            view vectors. Two combos of one family, incompatible _mN
            identity: the cross-check that these expressions belong to the
            AXIS and not to the all-on combo.

    TWO DIVERGENCES THE RE-READ FOUND in the version written blind:

      * a `.clamp(min=1e-3)` on the edge denominator. The reference divides
        by `1.0 - edgeK` with no guard: at edgeK == 1 it divides by zero,
        the clamp takes the inf to 1, and the edge term becomes 0. The
        guard changes the answer at exactly the value the shader handles by
        construction. Removed.
      * the `(1.0 - stickerAccum.x)` factor. The all-on module carries it
        at :5550 and the sticker-free module does not, so it is an
        S_STICKERS INTERACTION rather than part of the refract term. It is
        now passed in and defaults to absent.

    `tint` is a FLOAT (`float _m209` / `float _m37`), not a colour. The
    shimmer was reading this same scalar column for its vec3 tint.

    `bgTarget` is an ENGINE RENDER TARGET through a bindless 2D handle from
    a per-view CB -- 331 sampled modules of these families declare zero
    SubpassData and zero OpImageRead/Write, so no new memory root class is
    introduced here, and none is added.
    """
    if args.weapon_bgrefract == "off":
        return albedo, torch.zeros(albedo.shape[:-1], device=albedo.device)
    scale = _e2(x, "wpn_refract_scale")
    edgeK = _e2(x, "wpn_refract_edge")
    amount = _e2(x, "wpn_refract_amount")
    contrast = _e2(x, "wpn_refract_contrast")
    tint = _e2(x, "wpn_refract_tint")
    blur = _e2(x, "wpn_refract_blur")
    inv = torch.tensor([1.0 / viewport[0], 1.0 / viewport[1]],
                       device=albedo.device)
    # :5534 -- cross(view_a, view_b) for x, view_b for y, and view_a again
    # in the edge term below. The renderer's (cam_fwd, cam_up) fill those
    # two roles.
    uv_r = wpn.refract_screen_uv(frag_xy, inv, Wn, cam_fwd, cam_up, scale)
    bg = vm_sample(tex["bgrt"], uv_r)[..., :3]
    if args.weapon_bgrefract == "blur":
        # :5540 -- the radius is invScreen.xy * (blur * 4), i.e. render-
        # target texels, and the four signed offsets are (++, -+, +-, --).
        s = (blur * 4.0).unsqueeze(-1)
        taps = []
        for sx, sy in ((1, 1), (-1, 1), (1, -1), (-1, -1)):
            sgn = torch.stack([torch.full_like(s[..., 0], sx),
                               torch.full_like(s[..., 0], sy)], -1)
            taps.append(vm_sample(tex["bgrt"],
                                  uv_r + inv * sgn * s)[..., :3])
        bg = wpn.refract_blur5(bg, *taps)
    mask = vm_sample(tex["refract_mask"], uv)[..., 0]   # :5549
    stk = (torch.zeros_like(mask) if sticker_a is None else sticker_a)
    out = wpn.refract_blend(albedo, bg, mask, cam_fwd, Wn, edgeK, amount,
                            contrast, tint, sticker_a=stk)           # :5550
    edge = ((((cam_fwd * Wn).sum(-1) - (-1.0))
             / (1.0 - edgeK)).clamp(0.0, 1.0) * (-1.0)) + 1.0
    return out, ((edge * mask) * amount) * (1.0 - stk)


def vm_sample(tex, uv):
    """Wrap-repeat bilinear tap on a (H,W,C) tensor.

    The same convention as sample_textures()/sample_aux(): the UV is taken
    mod 1 and filtered bilinearly. There is no mip chain on the viewmodel
    textures, so this is a level-0 tap and says so; --weapon-stickers
    computes its own LOD from the screen-space footprint where the shader
    calls textureQueryLod.
    """
    Ht, Wt = tex.shape[0], tex.shape[1]
    u = (uv[..., 0] % 1.0) * Wt - 0.5
    v = (uv[..., 1] % 1.0) * Ht - 0.5
    x0 = torch.floor(u)
    y0 = torch.floor(v)
    fx = (u - x0).unsqueeze(-1)
    fy = (v - y0).unsqueeze(-1)
    x0 = x0.long() % Wt
    y0 = y0.long() % Ht
    x1 = (x0 + 1) % Wt
    y1 = (y0 + 1) % Ht
    a = tex[y0, x0]
    b = tex[y0, x1]
    c = tex[y1, x0]
    d = tex[y1, x1]
    return (a * (1 - fx) + b * fx) * (1 - fy) + (c * (1 - fx) + d * fx) * fy


# ----------------------------------------------------------------------
# the viewmodel's data dependencies
# ----------------------------------------------------------------------
_VM_TEX = None
_VM_TEX_OVERRIDE = None      # real pages, per slot, from the model bundle
_VM_TEX_PROV = None          # what each slot actually sampled


def vm_textures():
    """The texture set csgo_weapon reads, supplied.

    csgo_weapon's baseline module samples, at the shared UV:
        m1  colour    (allon_c3041_d2:526)  .rgb albedo, times vColor
        m16 tint mask (allon_c3041_d2:527)  .x   gates the vColor tint at :529
        m2  mask      (allon_c3041_d2:532)  .x roughness :535, .y METALNESS :533
                                  lerp, .z the ADJUSTMENT + GLITTER mask
        m15 normal    (allon_c3041_d2:537)  two-channel octahedral, :540-542
    plus, per axis: a flow map (SFX), a glitter normal whose .w is its own
    mask (S_GLITTER), a wear mask + sticker sheet + sticker normal
    (S_STICKERS), a refraction mask and the background render target
    (D_USE_BGREFRACT), a tint mask (S_TINT_MASK) and an emissive
    (S_SELF_ILLUM).

    de_inferno's pack contains NONE of these -- it has no weapon in it, so
    it has no weapon texture in it either. Under --viewmodel-synth they
    are synthesised deterministically from a fixed seed. STATED
    DIFFERENCE: what is reproduced is the CHANNEL LAYOUT each fetch site
    indexes, so every axis has a live input at the channel the bytecode
    reads; the images are not Valve's and are not claimed to be.
    """
    global _VM_TEX
    if _VM_TEX is not None:
        return _VM_TEX
    n = 256
    g = torch.Generator().manual_seed(0x57C1)
    yy, xx = torch.meshgrid(torch.arange(n, dtype=torch.float32),
                            torch.arange(n, dtype=torch.float32),
                            indexing="ij")
    u, v = xx / n, yy / n

    def _r(*shape):
        return torch.rand(*shape, generator=g)

    checker = (((xx // 16).long() + (yy // 16).long()) % 2).float()
    # colour: a two-tone panel with a plate seam, so albedo is not flat
    col = torch.stack([0.22 + 0.35 * checker,
                       0.24 + 0.22 * checker,
                       0.26 + 0.14 * checker], dim=-1)
    col = col * (0.85 + 0.3 * torch.sin(u * 25.1).abs()).unsqueeze(-1)
    # mask.x roughness, .y remap lerp, .z the adjustment/glitter mask
    msk = torch.stack([0.25 + 0.55 * torch.sin(u * 11.0).abs(),
                       0.5 + 0.5 * torch.cos(v * 7.0),
                       (torch.sin(u * 6.0) * torch.cos(v * 6.0))
                       .clamp(0.0, 1.0)], dim=-1)
    msk = torch.cat([msk, torch.ones_like(msk[..., :1])], dim=-1)
    # a two-channel octahedral normal: the encode is the exact inverse of
    # _octa_normal(), so a flat texel decodes to (0,0,1).
    bump = 0.25 * torch.sin(u * 40.0) * torch.sin(v * 40.0)
    e_a = (bump + 1.00392162799835205078125) * 0.5
    e_b = (1.00392162799835205078125 - bump) * 0.5
    nrm = torch.stack([e_a, e_b, torch.zeros_like(u),
                       torch.ones_like(u)], dim=-1)
    ao = torch.stack([0.6 + 0.4 * checker] * 3 + [torch.ones_like(u)],
                     dim=-1)
    _VM_TEX = {
        "colour": col.to(device),
        "mask": msk.to(device),
        "normal": nrm.to(device),
        "ao": ao.to(device),
        # SFX flow map: .xy the flow vector, .z the phase, .w its mask
        "flow": torch.stack([0.5 + 0.5 * torch.sin(u * 9.0),
                             0.5 + 0.5 * torch.cos(v * 9.0),
                             torch.frac(u * 3.0 + v * 2.0),
                             (0.3 + 0.7 * checker)], dim=-1).to(device),
        # glitter: octahedral flake normal in .xy, flake mask in .w
        "glitter": torch.stack([_r(n, n), _r(n, n), torch.zeros_like(u),
                                (_r(n, n) > 0.86).float()],
                               dim=-1).to(device),
        # sticker sheet: .rgb the print, .w the authored alpha the shader
        # hardens with *12.75
        "sticker": torch.stack([
            (0.9 * ((u - 0.5) ** 2 + (v - 0.5) ** 2 < 0.16).float()),
            (0.35 + 0.5 * torch.sin(u * 18.0).abs()),
            (0.15 + 0.7 * torch.cos(v * 14.0).abs()),
            ((u - 0.5) ** 2 + (v - 0.5) ** 2 < 0.16).float() * 0.5,
        ], dim=-1).to(device),
        # wear mask: .x, the multiplier on every sticker's alpha
        "wear": torch.stack([0.55 + 0.45 * torch.sin(u * 5.0
                                                     + v * 3.0).abs()] * 4,
                            dim=-1).to(device),
        "refract_mask": torch.stack([(0.2 + 0.8 * checker)] * 4,
                                    dim=-1).to(device),
        "tint_mask": torch.stack([(0.5 + 0.5 * torch.sin(v * 13.0))] * 4,
                                 dim=-1).to(device),
        "emissive": torch.stack([torch.ones_like(u) * 0.9,
                                 torch.ones_like(u) * 0.35,
                                 torch.ones_like(u) * 0.1,
                                 torch.ones_like(u)], dim=-1).to(device),
        # the ENGINE RENDER TARGET D_USE_BGREFRACT reads. It is filled per
        # frame from the world frame this pass composites over, so the
        # refraction shows the actual background rather than a constant;
        # this entry is only the fallback shape.
        "bgrt": torch.zeros((n, n, 4)).to(device),
    }
    # --- REAL PAGES OVERRIDE THE SYNTHESISED ONES, PER SLOT ------------
    # Whole-set synthesis was a clean state: the docstring above says the
    # images are not Valve's and nothing samples anything else. The
    # PARTIAL state is the dangerous one -- a real albedo beside four
    # synthesised channels makes any measurement ambiguous between "the
    # real page helped" and "the synthetic ones hurt", and turns that
    # docstring into a claim that is false in one direction and true in
    # the rest. So the provenance is per SLOT and it is a value, not a
    # comment: the render prints it and the bundle carries it.
    for _s, _p in (_VM_TEX_OVERRIDE or {}).items():
        if _s not in _VM_TEX:
            print(f"viewmodel texture: bundle supplies slot {_s!r}, which "
                  f"csgo_weapon does not sample -- IGNORED, not silently "
                  f"added", flush=True)
            continue
        # Channel count must match the slot it replaces. The albedo slot is
        # 3-channel and a decoded .vtex_c is RGBA, and without this the
        # mismatch surfaced 300 lines later as "size of tensor a (4) must
        # match tensor b (3)" inside the diffuse term -- a shape error
        # wearing a lighting error's clothes. Conformed HERE, at the
        # boundary, and the narrowing is stated in the provenance rather
        # than done quietly.
        _want = _VM_TEX[_s].shape[-1]
        _pf = _p.to(device).float() / 255.0
        if _pf.shape[-1] > _want:
            _VM_TEX_PROV[_s] = (_VM_TEX_PROV.get(_s, "") +
                                f", {_pf.shape[-1]}ch sliced to {_want}ch "
                                f"to match the slot")
            _pf = _pf[..., :_want]
        elif _pf.shape[-1] < _want:
            raise SystemExit(
                f"--viewmodel-model supplies slot {_s!r} with "
                f"{_pf.shape[-1]} channels; csgo_weapon samples {_want}. "
                f"Padding it would invent the missing channel, and every "
                f"term that reads it would be measuring the invention.")
        _VM_TEX[_s] = _pf
    for _s in sorted(_VM_TEX):
        _pr = (_VM_TEX_PROV or {}).get(_s)
        print(f"viewmodel texture: {_s:12s} "
              + (_pr if _pr else
                 f"SYNTHESISED seed 0x57C1 {n}x{n} -- channel layout only, "
                 f"not Valve's image"), flush=True)
    return _VM_TEX


def vm_synth_geometry(kind):
    """A synthesised viewmodel draw, in VIEW space, metres.

    de_inferno's world pack contains no weapon and no first-person legs,
    so 'nothing to render' would be indistinguishable from 'never called'.
    This submits geometry through the SAME dispatch, the SAME side-table
    columns and the SAME shading functions a packed weapon would take;
    only the vertex source differs, and that is the whole of the
    difference.

    Returns (pos (N,3) view-space metres, uv (N,4), nrm (N,3), tan (N,3),
             loc (N,3) MODEL-LOCAL in SOURCE UNITS, tri (T,3) int32,
             fam (T,) long, is_legs (N,) bool).

    uv.xy is the surface UV and uv.zw is the STICKER UV SET -- csgo_weapon
    reads them as separate channels of one attribute (UNSOURCED-glsl:671 uses .zw
    where :308 uses .xy).

    `loc` is separate from `pos` and is in SOURCE UNITS because two
    transcribed terms are written in that space and only make sense there:
    csgo_legs_prepass's cone test is anchored at vec3(0, 0, 55) -- the
    model's eye height in Source units -- and csgo_weapon's SFX phase adds
    localPos.x * 0.01. Feeding metres into either would make the term
    degenerate rather than wrong-looking, which is worse: the cone test
    would pass everywhere.
    """
    P, UV, NR, TN, LOC, TRI, FAM, LEG = [], [], [], [], [], [], [], []

    def _box(cx, cy, cz, sx, sy, sz, fam_id, loc_c=None, loc_s=None):
        base = len(P)
        c = torch.tensor([cx, cy, cz])
        s = torch.tensor([sx, sy, sz])
        lc = torch.tensor(loc_c) if loc_c is not None else c / S
        ls = torch.tensor(loc_s) if loc_s is not None else s / S
        # 6 faces x 4 verts, so each face carries its own normal and UV
        axes = [(0, 1, 2), (0, 1, 2), (1, 2, 0), (1, 2, 0), (2, 0, 1),
                (2, 0, 1)]
        signs = [1, -1, 1, -1, 1, -1]
        for f, ((a0, a1, a2), sg) in enumerate(zip(axes, signs)):
            for du, dv in ((-1, -1), (1, -1), (1, 1), (-1, 1)):
                p = torch.zeros(3)
                p[a0] = du
                p[a1] = dv
                p[a2] = sg
                P.append(c + p * s)
                LOC.append(lc + p * ls)
                LEG.append(fam_id == 1)
                n = torch.zeros(3)
                n[a2] = sg
                NR.append(n)
                t = torch.zeros(3)
                t[a0] = 1.0
                TN.append(t)
                UV.append(torch.tensor([(du + 1) * 0.5, (dv + 1) * 0.5,
                                        (du + 1) * 0.5, (dv + 1) * 0.5]))
            q = base + f * 4
            TRI.append([q, q + 1, q + 2])
            TRI.append([q, q + 2, q + 3])
            FAM.append(fam_id)
            FAM.append(fam_id)

    if kind in ("weapon", "both"):
        # a weapon-shaped proxy: receiver, barrel, magazine
        _box(0.0, 0.0, 0.0, 0.055, 0.040, 0.150, 0)
        _box(0.0, 0.018, -0.230, 0.020, 0.020, 0.110, 0)
        _box(0.0, -0.075, 0.020, 0.028, 0.055, 0.045, 0)
    if kind in ("legs", "both"):
        # Two thigh proxies. Placed to be INSIDE the viewmodel frustum
        # rather than where a real player's thighs sit, because a camera
        # looking straight ahead sees no legs at all and a pass that
        # covers 0 px cannot be told from a pass that never ran.
        #
        # Their MODEL-LOCAL coordinates are the ones that matter: z spans
        # 20..50 Source units against the shader's 55-unit eye anchor, so
        # the cone test step(0.8, dot(normalize(loc - (0,0,55)), (0,0,-1)))
        # PARTITIONS the proxy -- the upper thigh (z near 50) falls
        # outside acos(0.8) and the lower part inside. A local frame that
        # made the whole proxy pass would be a check that cannot fail.
        # They are also placed to STRADDLE the fade band. The transcribed
        # term is smoothstep(F-35, F, dist) with dist in Source units, and
        # the screen-door then keeps with probability clamp(6*fade-0.5,0,1)
        # -- so fade <= 1/12 is always discarded and fade >= 0.25 always
        # kept, and the interesting window is dist in about 31..36 units.
        # At the default F = 60 that is 0.79..0.90 m, so a proxy spanning
        # roughly 0.69..1.03 m from the eye is partitioned three ways by
        # its own fade: discarded, dithered, kept. Geometry that sat
        # wholly on one side would make the fade term unfalsifiable.
        _box(-0.10, -0.20, -0.82, 0.060, 0.160, 0.140, 1,
             loc_c=(-11.0, 0.0, 35.0), loc_s=(6.0, 6.0, 15.0))
        _box(0.10, -0.20, -0.82, 0.060, 0.160, 0.140, 1,
             loc_c=(11.0, 0.0, 35.0), loc_s=(6.0, 6.0, 15.0))
    if not P:
        return None
    return (torch.stack(P).to(device),
            torch.stack(UV).to(device),
            torch.stack(NR).to(device),
            torch.stack(TN).to(device),
            torch.stack(LOC).to(device),
            torch.tensor(TRI, dtype=torch.int32, device=device),
            torch.tensor(FAM, dtype=torch.long, device=device),
            torch.tensor(LEG, dtype=torch.bool, device=device))


def vm_transform(pos, is_legs):
    """The viewmodel transform: roll, scale, then the view-space offset.

    Engine data, not shader data -- the shader is handed an already-posed
    vertex. Supplied by --viewmodel-offset / --viewmodel-scale /
    --viewmodel-roll so the pass has a real transform rather than an
    identity that would read as implemented.

    It applies to the WEAPON only. The first-person legs are part of the
    PLAYER model, not the viewmodel: they share the pass and the depth
    range but not the weapon's bob/sway transform, which is why
    csgo_legs_prepass is a separate shader on separate geometry rather
    than a combo of csgo_weapon.
    """
    th = math.radians(args.viewmodel_roll)
    c, s = math.cos(th), math.sin(th)
    R = torch.tensor([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]],
                     device=pos.device, dtype=pos.dtype)
    off = torch.tensor(args.viewmodel_offset, device=pos.device,
                       dtype=pos.dtype)
    posed = (pos * args.viewmodel_scale) @ R.T + off
    return torch.where(is_legs.unsqueeze(-1), pos, posed)


_WPN_DEFAULTS_PRINTED = False


def _e2_has(x, key):
    """Does the per-pixel parameter block carry this column at all.

    EXT2 is the column map `_e2` indexes; a key it does not hold is a
    column this renderer never packed, and the caller substitutes a stated
    default rather than raising.
    """
    return key in EXT2


def _vm_per_view_scale():
    """`perViewScale` of the SFX gate product (allon_c3041_d2:451).

    The reference reads it per view; this renderer has no per-view scalar
    for it, so it is 1.0 and the gate is then the sfx amount alone.
    """
    return 1.0


def _wpn_defaults_banner(x):
    """Name the weapon inputs this renderer cannot supply, at RUNTIME."""
    global _WPN_DEFAULTS_PRINTED
    if _WPN_DEFAULTS_PRINTED:
        return
    _WPN_DEFAULTS_PRINTED = True
    miss = [k for k in ("wpn_si_tint_r", "wpn_si_albedo_blend",
                        "wpn_sfx_wipe_width") if not _e2_has(x, k)]
    print("[weapon] self_illum + the four SFX-flow terms are LIVE.")
    print("[weapon]   perViewScale = 1.0    no per-view scalar here; the "
          "gate is the sfx amount alone")
    if miss:
        print("[weapon]   DEFAULTED columns (tint=white, blend=0, "
              "wipe_width=0.25): %s" % ", ".join(miss))
        print("[weapon]   Those terms RUN; the values they run on are "
              "stand-ins.")
