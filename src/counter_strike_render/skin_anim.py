"""ONE animation system: block -> pose -> bone transforms -> LBS over a slice.

WHY THIS IS A MODULE AND NOT A THIRD COPY (#98)
-----------------------------------------------
Three passes need the same five steps -- the viewmodel WEAPON (skinned), the
viewmodel ARMS (the last un-animated surface), and the PLAYERMODELS. Each one
that grows its own copy also grows its own version of the traps below, and
the traps are not hypothetical: every one of them has already cost this
project a defect.

    1  block_for          pick the clip block BY SKELETON NAME
    2  pose_model_space   sample + compose parent-local -> model space
    3  bone_transforms    pose o inverse(bind), joined BY NAME
    4  lbs                linear blend over a vertex SLICE
    5  frame_sequence     one (clip, t) per batch row

FOUR TRAPS THIS MODULE EXISTS TO HOLD CLOSED, each one measured:

  * SELECTING THE BLOCK BY THE BUNDLE'S OWN nm_skeleton. The weapon path
    does that and is right; the ARMS bundle's nm_skeleton names the WEAPON
    rig while its vertices are driven by the VIEWMODEL rig, so the same rule
    silently picks the wrong block. The caller names the skeleton it wants
    and `block_for` refuses rather than guessing.

  * IDENTITY FOR AN UNPOSED BONE. `bind` is the .vmdl_c frame and `pose` is
    the .vnmskel one; every posed bone crosses between them inside
    pose o inverse(bind). A bone left at identity does not stay put, it
    stays in the frame the model left -- m4a4's `sight` detached by 0.46 m
    exactly this way (#90). An orphan takes its NEAREST POSED ANCESTOR,
    which is exact for a rigid attachment because
        pose_a o (bind_a^-1 o bind_o) o bind_o^-1 = pose_a o bind_a^-1.

  * ONE POSE PER BATCH. Posing a whole batch at its first row's choice
    smears up to (B-1) ticks; at B=2 it halved the visible motion of every
    event that landed second. `frame_sequence` resolves one (clip, t) per
    row, and identical rows are computed once so the common case costs
    nothing.

  * `loc` MEANING TWO DIFFERENT THINGS. extract_viewmodel writes UNPOSED
    source-unit positions under that key; extract_arms writes ALREADY-POSED
    ones. Same name, opposite meaning -- and LBS-ing the posed kind double-
    applies the pose (measured: 1.82 m against the arms' own bake). `lbs`
    cannot detect this, so it does not pretend to: the caller states which
    space it is handing over and `assert_bind_space` gives it a cheap way to
    prove the claim before trusting it.

WHAT IS DELIBERATELY NOT HERE: the scripter. Choosing WHICH event a demo row
implies is viewmodel-specific state (event latch, distance, previous row),
and generalising a viewmodel event model onto playermodels would be a fork
wearing a shared name. Callers resolve (clip, t) themselves and hand it in.
"""
import numpy as np

# --------------------------------------------------------------------------
# quaternion helpers -- torch-free so the offline oracles can use them too
# --------------------------------------------------------------------------


def qmul(a, b):
    ax, ay, az, aw = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
    bx, by, bz, bw = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
    return np.stack([aw * bx + ax * bw + ay * bz - az * by,
                     aw * by - ax * bz + ay * bw + az * bx,
                     aw * bz + ax * by - ay * bx + az * bw,
                     aw * bw - ax * bx - ay * by - az * bz], axis=-1)


def qinv(q):
    return np.concatenate([-q[..., :3], q[..., 3:4]], axis=-1)


def qrot(q, v):
    u, w = q[..., :3], q[..., 3:4]
    return (v * (w * w - (u * u).sum(-1, keepdims=True))
            + 2.0 * u * (v * u).sum(-1, keepdims=True)
            + 2.0 * w * np.cross(u, v))


# --------------------------------------------------------------------------
# 1. the block
# --------------------------------------------------------------------------
def block_for(clip, skeleton, require_parents=True):
    """The clip's block for a NAMED skeleton, or None.

    Named, never inferred from the bundle -- see the module docstring's
    first trap. A block without `parents` cannot be composed to model
    space, so it does not count as a match.
    """
    for b in (clip.get("blocks") or []):
        if b.get("skeleton") != skeleton:
            continue
        if require_parents and b.get("parents") is None:
            continue
        return b
    return None


def block_varies(block, tol=1e-6):
    """(varies, max|d| across the block's OWN frames). The held-pose trap.

    A clip can be ABSOLUTE, non-additive, correctly decoded, and STILL be a
    single pose held for its whole declared duration. anim-pm measured 341
    of 1,601 blocks in agents.clips.pt like that, and the first absolute
    `walk` block is one of them -- so a selector keying on index 0 picks a
    frozen stride while the log prints an advancing t.

    THE REASON THIS LIVES HERE rather than in a caller: a clip-on/clip-off
    PIXEL A/B PASSES it. A held pose is still far from bind (27.3 source
    units in that case), so the image moves when the clip is enabled and
    every coverage, silhouette and difference check is satisfied. Only a
    WITHIN-CLIP comparison separates "playing" from "held", which is what
    this is.
    """
    t = block.get("t")
    q = block.get("q")
    d = 0.0
    for a in (t, q):
        if a is None:
            continue
        arr = a.numpy() if hasattr(a, "numpy") else np.asarray(a)
        if arr.shape[0] > 1:
            d = max(d, float(np.abs(arr - arr[0]).max()))
    return d > tol, d


def skeletons_in(clip):
    """Every skeleton the clip carries -- for a refusal that can NAME the
    alternatives instead of saying 'not found'."""
    return [b.get("skeleton") for b in (clip.get("blocks") or [])]


# --------------------------------------------------------------------------
# 2. parent-local -> model space
# --------------------------------------------------------------------------
def compose_model_space(parents, lt, lq):
    """Accumulate parent-local (t, q) down the chain. READ, not re-derived.

    Depth order rather than index order: a child may precede its parent in
    the array, and composing in index order would use a parent that has not
    been composed yet -- silently, since the arithmetic still succeeds.
    """
    parents = [int(p) for p in parents]
    n = len(parents)
    depth = []
    for i in range(n):
        d, j = 0, parents[i]
        seen = set()
        while j >= 0 and j not in seen:
            seen.add(j)
            d += 1
            j = parents[j]
        depth.append(d)
    mt = np.zeros((n, 3), np.float64)
    mq = np.zeros((n, 4), np.float64)
    for i in sorted(range(n), key=lambda k: depth[k]):
        p = parents[i]
        if p < 0:
            mt[i], mq[i] = lt[i], lq[i]
        else:
            mt[i] = mt[p] + qrot(mq[p], lt[i])
            mq[i] = qmul(mq[p], lq[i])
    return mt, mq


# --------------------------------------------------------------------------
# 3. pose o inverse(bind), joined BY NAME
# --------------------------------------------------------------------------
class Substitution(dict):
    """What the caller must PRINT: which bones were substituted and how big
    they are. Carried as data so each caller reports its own magnitudes
    rather than re-deriving them (or, worse, not reporting them)."""


def bone_transforms(palette, bind_t, bind_q, bone_ids, mt, mq,
                    palette_ancestors=None, index=None, weight=None):
    """(xq, xt, report) -- one rigid transform per PALETTE bone.

    `report` has `substituted` [(bone, ancestor, verts, mass%)],
    `unsubstituted` [(bone, verts, mass%, why)] and `n_posed`.
    Vertex counts and weight mass are filled only when index/weight are
    given; they are what makes a substitution priceable rather than a name.
    """
    pal = list(palette)
    ids = list(bone_ids)
    n = len(pal)
    xq = np.tile(np.array([0.0, 0.0, 0.0, 1.0]), (n, 1))
    xt = np.zeros((n, 3), np.float64)
    pidx = {nm: i for i, nm in enumerate(pal)}

    def _size(i):
        if index is None or weight is None:
            return None, None
        on = (index == i)
        return int(on.any(1).sum()), 100.0 * float(
            weight[on].sum()) / max(1e-12, float(weight.sum()))

    # PASS 1: bones the rig names, each with its OWN inverse bind.
    subbed, unsub = [], []
    n_posed = 0
    for i, name in enumerate(pal):
        if name not in ids:
            continue
        j = ids.index(name)
        iq = qinv(bind_q[i])
        it = -qrot(iq, bind_t[i])
        xt[i] = mt[j] + qrot(mq[j], it)
        xq[i] = qmul(mq[j], iq)
        n_posed += 1

    # PASS 2: orphans COPY the ancestor's finished transform. It must be a
    # copy, and the arithmetic says why:
    #
    #     pose_o = pose_a o (bind_a^-1 o bind_o)        rigid attachment
    #     x_o    = pose_o o bind_o^-1 = pose_a o bind_a^-1 = x_a
    #
    # so the term that survives is the ANCESTOR's inverse bind, not the
    # orphan's. Pairing the ancestor's POSE with the orphan's OWN inverse
    # bind -- the obvious-looking one-pass form -- leaves the orphan at the
    # origin of its own bind offset instead of on the parent. The selftest
    # plants exactly that: a child rigidly attached 2 units along +x, parent
    # turned 90 degrees, must land at +2y; the one-pass form puts it at the
    # origin. This is why the ancestor must also be IN THE PALETTE -- that
    # is where its bind, and therefore its finished transform, lives.
    for i, name in enumerate(pal):
        if name in ids:
            continue
        chain = [c for c in ((palette_ancestors or [[]] * n)[i] or [])
                 if c in ids and c in pidx]
        nv, ms = _size(i)
        if not chain:
            unsub.append((name, nv, ms,
                          "no ancestor of it is both posed by this clip and "
                          "in this mesh's palette"
                          if palette_ancestors else
                          "this bundle carries no palette_ancestors -- "
                          "re-run vm_add_parents.py over it"))
            continue
        a = pidx[chain[0]]
        xq[i], xt[i] = xq[a], xt[a]
        subbed.append((name, chain[0], nv, ms))
    return xq, xt, Substitution(substituted=subbed, unsubstituted=unsub,
                                n_posed=n_posed, n_palette=n)


# --------------------------------------------------------------------------
# 4. LBS over a slice
# --------------------------------------------------------------------------
def lbs(loc, index, weight, xq, xt, nrm=None, dirs=None):
    """sum_k w_k * (qrot(xq[b_k], loc) + xt[b_k]), plus rotated DIRECTIONS.

    `dirs` is a list of direction fields -- normals, tangents, anything that
    rotates with the bone but does not translate -- all blended in the SAME
    pass and renormalised. `nrm=` is the one-field spelling of it and is
    kept because it reads better at a single call site.

    WHY dirs IS A LIST (anim-pm's consumer report): the playermodel pass
    needs normals AND tangents, and with a single-field signature it got
    them by calling this twice, the second time passing the tangent as
    `nrm` and throwing the returned positions away. That works and is
    cheap, but it computes the position blend twice to obtain nothing, and
    a workaround in a caller is a signature problem wearing a caller's
    clothes.

    NOT ROTATING DIRECTIONS IS NOT COSMETIC, and this is the sentence to
    read before deciding to skip it: a posed limb whose normals stayed at
    bind LIGHTS exactly as though it had not moved, while its silhouette is
    correct the entire time. No coverage check and no silhouette check can
    see that. extract_arms rotates normals for this reason; the weapon path
    in gpu_render does not, and that is a live gap rather than a choice.

    REFUSES a binding set longer than the geometry rather than truncating:
    a skin and a position array of different lengths came from different
    reads of the mesh, and posing the overlap would move the wrong vertices
    while looking like it worked.
    """
    n = len(loc)
    if len(index) > n:
        raise ValueError(
            f"skin indexes {len(index)} vertices but the target slice has "
            f"{n} -- bindings and positions came from different reads")
    fields = list(dirs) if dirs is not None else []
    if nrm is not None:
        fields = [nrm] + fields
    m = len(index)
    out = np.zeros((m, 3), np.float64)
    douts = [np.zeros((m, 3), np.float64) for _ in fields]
    for k in range(index.shape[1]):
        b = index[:, k]
        w = weight[:, k:k + 1]
        q = xq[b]
        out += w * (qrot(q, loc[:m]) + xt[b])
        for d, f in zip(douts, fields):
            d += w * qrot(q, f[:m])
    for d in douts:
        d /= np.maximum(np.linalg.norm(d, axis=1, keepdims=True), 1e-12)
    if not fields:
        return out
    if dirs is None:
        return out, douts[0]
    return (out, douts) if nrm is None else (out, douts[0], douts[1:])


def assert_bind_space(loc, posed_ref, tol=1e-6):
    """Is `loc` bind-space, or is it already the posed result?

    The cheap guard for the module docstring's fourth trap. If `loc` equals
    the bundle's own baked output, it is NOT bind-space and LBS-ing it will
    double-apply the pose. Returns (ok, max|d|) -- the caller decides, but
    it can no longer do so unknowingly.
    """
    if posed_ref is None or len(posed_ref) != len(loc):
        return True, None
    d = float(np.abs(np.asarray(loc) - np.asarray(posed_ref)).max())
    return d > tol, d


# --------------------------------------------------------------------------
# 5. one (clip, t) per batch row
# --------------------------------------------------------------------------
def frame_sequence(choices, nframes):
    """[(clip, t) or None] per row, with a chose-nothing row CARRIED FORWARD.

    `choices` is the caller's per-row result; None means "no NEW event this
    tick", which is not "no animation" -- the clip already playing
    continues. Rows before the FIRST choice have nothing to carry and stay
    None, which is the bind pose, exactly as a whole batch used to be when
    no row chose.

    Returns (seq, n_carried) so the caller can PRINT the carry count rather
    than have rows silently inherit.
    """
    seq = list(choices)[:nframes] + [None] * max(0, nframes - len(choices))
    carried = 0
    for k in range(len(seq)):
        if seq[k] is None and k and seq[k - 1] is not None:
            seq[k] = seq[k - 1]
            carried += 1
    return seq, carried


def distinct_poses(seq):
    """{key: (clip, t)} and the per-row key list.

    Identical rows are posed ONCE. A 64-tick clip does not change event
    every frame, so the per-frame path costs the same as the per-batch one
    in the common case -- and that is what makes per-frame the default
    rather than a mode someone has to opt into.
    """
    key = [None if s is None else (id(s[0]), float(s[1])) for s in seq]
    uniq = {}
    for k, s in zip(key, seq):
        uniq.setdefault(k, s)
    return uniq, key
