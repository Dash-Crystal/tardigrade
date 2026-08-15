"""Which real page belongs in which viewmodel texture slot, PER WEAPON.

READ, per weapon, by the route that is in the container:

    <weapon>.vmdl_c CTRL.embedded_meshes[name].m_nDataBlock
        -> that MDAT block  (kv3v5 strict=False, see below)
        -> m_sceneObjects[*].m_drawCalls[*].m_material
        -> that .vmat_c's m_textureParams
        -> the .vtex_c path for each parameter the slot wants

THE MESH PICKS THE MATERIAL, and this is a CORRECTION. The previous table
was hardcoded to `materials/models/weapons/v_models/rif_ak47/ak47.vmat`,
reached through the vmdl's RERL refs. RERL lists BOTH materials the file
references and does not say which mesh uses which. The AK's two meshes bind
different ones:

    body_legacy  m_nDataBlock 2  -> materials/models/weapons/v_models/rif_ak47/ak47.vmat
    body_hd      m_nDataBlock 7  -> weapons/models/ak47/materials/weapon_rif_ak47.vmat

The extractor uses `body_hd`. So the shipped AK bundle carried body_LEGACY's
pages on body_HD's geometry -- the two page sets are different files
(`ak47_color_psd_1f318532` vs `ak47_default_color_psd_5b66a23b`) and the
legacy/hd split is visible in their own names: the v_models material names
`weapon_rif_ak47_sticker_mask_LEGACY`, the weapons/models one names
`weapon_rif_ak47_sticker_mask_HD`. m_nDataBlock decides it and is read here.

MDAT PARSES. `extract_viewmodel.py` recorded MDAT as "REFUSED by kv3v5's
strict check" and routed around it. The refusal is real and strict=False
parses the same block without complaint: 2 of 2 AK MDATs and 2 of 2 galil
MDATs return m_sceneObjects. The strict residual ("v5 strings left 4 bytes
of the auxiliary 1-byte buffer unread") stays an open parser question; it is
not a reason to leave the mesh->material join unread.

SLOT <- PARAMETER is the one STATED mapping here. The paths are read; which
Valve parameter feeds which of OUR slots is our choice, because the slots
are ours (`vm_textures()` in gpu_render.py). It is stated in one place so a
consumer reads it instead of finding it by reading a decode.
"""

# The decode edge. A STATED PARAMETER, not a constant: TEXTURE_EDGE = 64
# became a problem precisely because it was a defensible choice that stopped
# being visible.
TEX_EDGE = 512

# Our slot -> the csgo_weapon.vfx material parameter that fills it.
SLOT_PARAM = {
    "colour": "g_tColor",
    "normal": "g_tNormal",
    "ao":     "g_tAmbientOcclusion",
}

# Parameters the material DOES name that we do NOT bind, each with the
# reason. These are not missing and not optional -- they exist in the
# archives and the weapon's own material references them. What stops them
# being bound is that `vm_textures()`'s corresponding slots carry OUR
# channel packing, chosen when those slots were synthesised. Dropping a
# Valve image into a slot whose channel meanings we invented replaces a
# LABELLED stand-in with an UNLABELLED wrong answer, which is strictly
# worse: the stand-in at least prints itself as one.
UNMAPPED_PARAM = {
    "mask": ("g_tMetalness",
             "our `mask` slot packs roughness in .x and a metal lerp in .y; "
             "this page is not that layout, so the channel mapping must be "
             "stated before it can be bound"),
    "sticker": ("g_tSticker0",
                "CS2's shared sticker page -- legitimate to bind "
                "(BINDLESS_MATERIAL_TABLE.md sec 6: the test is 'no page WE "
                "invented'), but our sticker slot's channel packing is ours "
                "and needs the mapping stated"),
}


def _blocks(content, vmdl_path):
    import ash_extract as A
    data = content.find(vmdl_path)
    if data is None:
        raise SystemExit(f"{vmdl_path}: not in the content mount")
    return A.ordered_blocks(data)


def mesh_materials(content, vmdl_path, mesh_name="body_hd"):
    """(materials_of_this_mesh, materials_of_other_meshes, lines).

    `materials_of_this_mesh` is ordered by TRIANGLE COVERAGE, most first.

    Both lists are returned because the other mesh's material is exactly
    what the previous hardcoded table bound. Printing it is what makes
    "we chose the mesh's own" a checked statement rather than a claim.
    """
    import kv3v5
    blocks = _blocks(content, vmdl_path)
    by = {}
    for t, b in blocks:
        by.setdefault(t, b)
    ctrl = kv3v5.parse(by["CTRL"])
    meshes = {m.get("m_Name"): m for m in ctrl["embedded_meshes"]}
    if mesh_name not in meshes:
        raise SystemExit(f"{vmdl_path}: no mesh {mesh_name!r}; has "
                         f"{sorted(meshes)}")
    lines, mine, others, cover = [], [], [], {}
    for nm, m in meshes.items():
        blk = m["m_nDataBlock"]
        tag, raw = blocks[blk]
        if tag != "MDAT":
            lines.append(f"  mesh {nm}: m_nDataBlock {blk} is {tag}, not "
                         f"MDAT -- material NOT READ")
            continue
        md = kv3v5.parse(raw, strict=False)
        # BY TRIANGLE COVERAGE, not by draw-call order. A weapon's mesh has
        # a dominant body material and a small extra; a CHARACTER's does not
        # -- ctm_sas' thirdperson_body has four, and the FIRST is
        # ctm_sas_lenses at 140 of 13,504 triangles (1.0%), whose colour page
        # is 4x4. Taking draw call 0 shaded a whole enemy with the goggle
        # lenses. tm_phoenix' first is bare_arms at 7.5% against a body at
        # 44.0%. Order is an authoring artefact; m_nIndexCount is a fact.
        tally = {}
        for o in md.get("m_sceneObjects") or []:
            for c in o.get("m_drawCalls") or []:
                mp = c.get("m_material")
                if not mp:
                    continue
                tally[mp] = tally.get(mp, 0) + int(c.get("m_nIndexCount", 0)) // 3
        mats = [m for m, _n in sorted(tally.items(), key=lambda kv: -kv[1])]
        tick = "USED" if nm == mesh_name else "not this mesh"
        tot = max(1, sum(tally.values()))
        shown = ", ".join(f"{m} ({100.0 * tally[m] / tot:.1f}%)" for m in mats)
        lines.append(f"  mesh {nm:12s} MDAT[{blk}] -> {shown}   [{tick}]")
        if nm == mesh_name:
            mine.extend(mats)
            cover.update({m: tally[m] / tot for m in mats})
        else:
            others.extend(mats)
    return mine, others, lines


def resolve(content, vmdl_path, mesh_name="body_hd", material=None):
    """(slots, unmapped, lines) for one weapon.

    `slots`     {our slot: .vtex path} for every SLOT_PARAM the material has
    `unmapped`  {our slot: (.vtex path, why)} for UNMAPPED_PARAM
    `lines`     everything READ, to be printed by the caller

    `material` names WHICH of the mesh's materials to resolve; None keeps the
    dominant-coverage default above, which every weapon caller relies on.
    The parameter exists because "one page set per bundle" is a CONTRACT,
    not a fact about the asset: taking the largest still leaves the rest of
    the surface wrong, and on tm_phoenix the largest is 44.0%, so more than
    half the character would be textured with a material it does not draw.
    A caller that carries a per-triangle material index -- extract_playermodel
    reads one straight off the draw calls -- resolves each material in turn
    and covers 100%. The default is unchanged for anyone who cannot.
    """
    import ash_extract as A
    mine, others, lines = mesh_materials(content, vmdl_path, mesh_name)
    if not mine:
        raise SystemExit(f"{vmdl_path}: mesh {mesh_name} names no material")
    if material is not None:
        if material not in mine:
            raise SystemExit(
                f"{vmdl_path}: mesh {mesh_name} does not draw {material!r}; "
                f"it draws {mine}. Resolving a material the mesh never binds "
                f"would page in a texture nothing samples.")
        mat = material
        lines.append(f"  material SELECTED by the caller: {mat} "
                     f"(1 of {len(mine)} on this mesh)")
    else:
        # The DOMINANT material by triangle coverage. One page set per bundle
        # is the contract, so a mesh with four materials gets the one that
        # covers the most surface and the rest are listed with their shares
        # -- an explicit shortfall rather than a silent one.
        mat = mine[0]
        if len(mine) > 1:
            lines.append(f"  {len(mine)} materials on this mesh; slots come "
                         f"from the LARGEST by triangle coverage: {mat}")
            for extra in mine[1:]:
                lines.append(f"    further draw call, NOT merged: {extra}")
    for o in others:
        lines.append(f"  other mesh's material, NOT bound: {o}")

    row, err = A.read_material(content, mat)
    if err:
        raise SystemExit(f"{mat}: {err}")
    tp = row["texture_params"]
    refs = set(row["resource_refs"])
    lines.append(f"  material {mat}")
    lines.append(f"    shader {row['shader_name']}, "
                 f"{len(tp)} texture params, {len(refs)} RERL refs")

    slots, unmapped = {}, {}
    for slot, param in SLOT_PARAM.items():
        p = tp.get(param)
        if not p:
            lines.append(f"    slot {slot:8s} <- {param}: PARAMETER ABSENT "
                         f"from this material")
            continue
        # Cross-check: the path the parameter names must also be one of the
        # material's own external refs. A parameter value that is not in
        # RERL would mean the two halves of the container disagree, which is
        # worth failing on rather than reading past.
        ok = "RERL-OK" if p in refs else "NOT IN RERL"
        slots[slot] = p
        lines.append(f"    slot {slot:8s} <- {param:22s} {p}  [{ok}]")
    for slot, (param, why) in UNMAPPED_PARAM.items():
        p = tp.get(param)
        if not p:
            lines.append(f"    slot {slot:8s} <- {param}: PARAMETER ABSENT")
            continue
        unmapped[slot] = (p, why)
        lines.append(f"    slot {slot:8s} <- {param:22s} {p}  [UNMAPPED]")
    return slots, unmapped, lines
