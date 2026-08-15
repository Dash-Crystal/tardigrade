

def vm_clips_for(bundle_path):
    """The `<weapon>.clips.pt` beside the mesh bundle, if it exists.

    RIG-LEVEL, NOT ENUM-LEVEL. character-2's rescope found the 21 missing
    sidecars were a CLASS -- 20 knife skin variants plus healthshot, each
    shipping its OWN rig and clips -- while the cs1k enum carries a single
    `knife` id. Resolving by the enum id would silently hand a butterfly
    knife the default knife's clips, so this resolves off the bundle PATH,
    which is the rig.
    """
    if VM_CLIPS[0] is not None:
        return VM_CLIPS[0]
    cand = os.path.splitext(bundle_path)[0] + ".clips.pt"
    if not os.path.exists(cand):
        print(f"animation: no clip sidecar at {cand} -- the viewmodel holds "
              f"its bind pose and the matrix reports decoded=0. Not a "
              f"substitute and none is invented.", flush=True)
        return None
    cs = _anim.ClipSet(cand)
    print(f"animation: {cs.note}", flush=True)
    VM_CLIPS[0] = cs
    ANIM_DECODED["viewmodel"] = sorted(cs.events)
    return cs


# THE CENSUS CONSERVATION LAW (#80). Run totals, so the matrix can state
# whether the attribution BALANCES instead of asserting that it does.
# CLASS_PX_COUNTED is what the counter attributed; CLASS_PX_SHADED is what
# the raster classes said they covered, arrived at INDEPENDENTLY (the `fg`
# coverage mask, written by the opaque and mask classes themselves and
# never derived from the attribution map). CLASS_PX_LEAK is the difference
# that must be zero.
CLASS_PX_COUNTED = [0]
CLASS_PX_SHADED = [0]
CLASS_PX_LEAK = [0]
CLASS_PX_EXTRA = [0]
CLASS_PX_CHUNKS = [0]
# Pixels each class WON, tallied at the coverage site -- the line that
# exists so the IMAGE is right (`fg |= keep`, the depth composite), not at
# the attribution write. That separation is the whole point: the balance
# below compares two quantities recorded by two different lines, so a
# class that draws and forgets to claim shows up as won>0 with counted==0.
# CALIBRATED TWO-SIDED: the first version of this law compared attribution
# against `fg` alone, and a planted defect (mask class skips its fam_pix
# write) went CLEAN -- because a mask pixel over opaque geometry is already
# claimed by the opaque class, so failing to overwrite it leaks nothing and
# only MISATTRIBUTES. That version would have passed the exact bug it was
# written to prevent. This tally is what catches it.
CLASS_PX_WON = {}
# fam_pix's "no class has claimed this pixel" value. NOT a family id, and
# deliberately not 0: 0 is csgo_environment_blend, and an unclaimed pixel
# that reads as a real class is precisely the failure this law exists to
# make impossible.
FAM_UNATTRIBUTED = -1


def class_px_won(fam_ids, mask):
    """Tally the pixels a raster class WON, at its coverage site.

    Deliberately NOT next to the fam_pix write: this is the independent
    operand. It is called from the line that decides the image (coverage
    and depth), so disabling the attribution alone leaves this intact and
    the census reports the discrepancy.
    """
    if fam_ids is None or mask is None or not FAM_NAMES:
        return
    v = fam_ids[mask]
    v = v[v >= 0]
    if v.numel() == 0:
        return
    b = torch.bincount(v.reshape(-1).long(), minlength=len(FAM_NAMES))
    for i, n in enumerate(b.tolist()):
        if n:
            CLASS_PX_WON[i] = CLASS_PX_WON.get(i, 0) + int(n)


def class_px_accumulate(fam_map, fg):
    """Shaded pixels per family id, summed over every frame of the run.

    ONE call site, at the composite (#80 option (c)). `fam_map` is the
    per-pixel attribution map every raster class writes into; pixels still
    carrying FAM_UNATTRIBUTED are not counted, and the caller balances that
    residue against the coverage the classes reported. Returns the number
    of pixels attributed so the caller can state the balance in numbers.
    """
    if fam_map is None or fg is None or not FAM_NAMES:
        return 0
    v = fam_map[fg]
    v = v[v >= 0]                      # drop the unattributed sentinel
    if v.numel() == 0:
        return 0
    b = torch.bincount(v.reshape(-1).long(), minlength=len(FAM_NAMES))
    for i, n in enumerate(b.tolist()):
        if n:
            CLASS_PX[i] = CLASS_PX.get(i, 0) + int(n)
    return int(v.numel())


# Which classes have their OWN transcribed path switched on. A class can be
# visible and shaded by a substitute; that is the defect the pixel count
# alone cannot see, so the flag state is reported beside it.
def _class_path_enabled():
    return {
        "csgo_vertexlitgeneric.vfx": bool(args.sf_vertexlit),
        "csgo_effects.vfx": bool(args.sf_effects),
        "csgo_black_unlit.vfx": bool(args.sf_black_unlit),
        "csgo_water_fancy.vfx": bool(args.water_fancy),
        "csgo_lightmappedgeneric.vfx": args.lmg_family == "on",
        "generic.vfx": args.lmg_family == "on",
    }
