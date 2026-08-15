"""Per-vertex bone bindings, read once for every mesh extractor.

Without these a bundle is a rigid body no matter how good the clip sampler
is: `bolt` and `slide` are named bones in the clip, but if no vertex declares
which bone owns it, nothing on the gun can move and nothing on a player can
walk.

THE TRAP THIS MODULE EXISTS TO CLOSE. BLENDINDICES_0 does NOT index the
skeleton. It indexes a per-mesh slice of DATA.m_remappingTable, whose slice
starts are m_remappingTableStarts and whose slice is selected by the mesh's
own m_nMeshIndex. Treating the blend index as a skeleton index gives a
plausible wrong bone -- vertices land somewhere, they just land on the wrong
part -- and no error is raised. So the remap is READ as a field, applied
here, and the palette this returns is BONE NAMES: a consumer joining names to
a clip's m_boneIDs cannot reintroduce the confusion, because there is no
index left to misinterpret.

Weight sums are a REFUSAL, not a warning. A vertex whose influences sum far
from 1 means the stream was misread (wrong width, wrong normalisation), and
skinning it anyway produces a mesh that is subtly the wrong size.
"""
import numpy as np

# A byte-quantised weight stream reaches us as 0..255; a float one is already
# 0..1. The distinction is READ off the data rather than assumed per-format.
BYTE_WEIGHT_MIN = 1.5
SUM_TOL = 0.02


def remap_slice(model_data, emb):
    """The bone-index remap slice this mesh uses, or None if it ships none."""
    table = list(model_data.get("m_remappingTable") or [])
    starts = list(model_data.get("m_remappingTableStarts") or [])
    if not table or not starts:
        return None, None, None
    mi = int(emb.get("m_nMeshIndex", 0))
    lo = starts[mi]
    hi = starts[mi + 1] if mi + 1 < len(starts) else len(table)
    return [int(x) for x in table[lo:hi]], int(lo), int(hi)


def read_skin(attrs, emb, model_data, bone_names, label=""):
    """(skin dict, printable lines) or (None, lines) when the mesh is rigid.

    `bone_names` is the model skeleton's m_boneName list -- the thing the
    remapped index actually points into.
    """
    lines = []
    if isinstance(attrs, (list, tuple)):
        # A character mesh is several vertex buffers concatenated, and the
        # bundle's positions are concatenated in the same order, so the
        # bindings must be too. A buffer that carries no bindings while its
        # siblings do would silently shift every later vertex's binding by
        # its length, so the mixed case is refused rather than padded.
        have = [("BLENDINDICES_0" in d) for d in attrs]
        if not any(have):
            lines.append("  skin: no buffer carries BLENDINDICES_0 -- rigid; "
                         "no binding is invented")
            return None, lines
        if not all(have):
            raise SystemExit(
                "%s: %d of %d vertex buffers carry BLENDINDICES_0 and the "
                "rest do not. Concatenating them would offset every later "
                "vertex's binding."
                % (label, sum(have), len(have)))
        merged = {"BLENDINDICES_0": np.concatenate(
            [np.atleast_2d(np.asarray(d["BLENDINDICES_0"]).T).T for d in attrs])}
        if all("BLENDWEIGHT_0" in d for d in attrs):
            merged["BLENDWEIGHT_0"] = np.concatenate(
                [np.atleast_2d(np.asarray(d["BLENDWEIGHT_0"]).T).T
                 for d in attrs])
        lines.append("  skin: %d vertex buffers merged, %d vertices total"
                     % (len(attrs), len(merged["BLENDINDICES_0"])))
        sub, more = read_skin(merged, emb, model_data, bone_names, label)
        return sub, lines + more

    if "BLENDINDICES_0" not in attrs:
        lines.append("  skin: NO BLENDINDICES_0 -- this mesh declares no bone "
                     "ownership at all; no binding is invented for it")
        return None, lines

    bi = np.asarray(attrs["BLENDINDICES_0"]).astype(np.int64)
    if bi.ndim == 1:
        bi = bi[:, None]

    if "BLENDWEIGHT_0" in attrs:
        bw = np.asarray(attrs["BLENDWEIGHT_0"], np.float64)
        if bw.ndim == 1:
            bw = bw[:, None]
        byte_scaled = bw.max() > BYTE_WEIGHT_MIN
        if byte_scaled:
            bw = bw / 255.0
    else:
        # NO WEIGHT STREAM IS A STATE, NOT AN ABSENCE -- and it is the state
        # every weapon mesh in this depot ships. The AK's body_hd has all four
        # index slots equal on 21414 of 21414 vertices, i.e. each vertex is
        # owned outright by one bone, which is what a gun is: a body, a bolt,
        # a magazine, each rigid. Collapsing to one influence at weight 1 is
        # then a read of the stream, not a default.
        #
        # It is only a read while the slots AGREE. Differing indices with no
        # weights would mean real blending whose weights we do not have, and
        # inventing 1/k there would silently smear vertices between bones.
        rigid = bool((bi == bi[:, :1]).all())
        if not rigid:
            n_split = int((~(bi == bi[:, :1]).all(1)).sum())
            raise SystemExit(
                "%s: BLENDWEIGHT_0 is absent but %d of %d vertices name "
                "DIFFERENT bones across their index slots. That is real "
                "blending whose weights this container does not carry; "
                "assuming equal weights would smear vertices between bones."
                % (label, n_split, len(bi)))
        lines.append("  skin: no BLENDWEIGHT_0, and all %d index slots agree "
                     "on every one of %d vertices -- RIGID binding, one bone "
                     "per vertex at weight 1 (read, not defaulted)"
                     % (bi.shape[1], len(bi)))
        bi = bi[:, :1]
        bw = np.ones((len(bi), 1))
        byte_scaled = False

    remap, lo, hi = remap_slice(model_data, emb)
    if remap is None:
        lines.append("  skin: m_remappingTable ABSENT -- blend indices taken "
                     "as skeleton indices, which is only right if this model "
                     "ships no remap")
        resolved = bi.copy()
    else:
        if int(bi.max()) >= len(remap):
            raise SystemExit(
                "%s: blend index %d exceeds the %d-entry remap slice [%d:%d] "
                "-- the slice is wrong, and clamping would put vertices on a "
                "plausible wrong bone"
                % (label, int(bi.max()), len(remap), lo, hi))
        resolved = np.asarray(remap, np.int64)[bi]
        lines.append("  skin: m_remappingTable slice [%d:%d] (%d entries) for "
                     "mesh index %d; blend index range %d..%d -> skeleton "
                     "bone range %d..%d"
                     % (lo, hi, len(remap), int(emb.get("m_nMeshIndex", 0)),
                        int(bi.min()), int(bi.max()),
                        int(resolved.min()), int(resolved.max())))
    if int(resolved.max()) >= len(bone_names):
        raise SystemExit(
            "%s: remapped bone %d exceeds the model skeleton's %d bones"
            % (label, int(resolved.max()), len(bone_names)))

    wsum = bw.sum(1)
    bad = int((np.abs(wsum - 1.0) > SUM_TOL).sum())
    lines.append("  skin: %d verts x %d influences, weights %s, sums "
                 "min %.4f max %.4f mean %.4f, %d verts off 1 by more "
                 "than %.2f"
                 % (len(bi), bi.shape[1],
                    "BYTE (/255)" if byte_scaled else "float",
                    wsum.min(), wsum.max(), wsum.mean(), bad, SUM_TOL))
    if bad:
        raise SystemExit(
            "%s: %d of %d vertices have influence weights summing off 1 by "
            "more than %.2f (min %.4f max %.4f). The stream was misread; "
            "skinning it anyway makes a mesh that is subtly the wrong size."
            % (label, bad, len(bi), SUM_TOL, wsum.min(), wsum.max()))
    bw = bw / np.maximum(wsum[:, None], 1e-9)

    # PALETTE OF NAMES, not of indices. Only the bones this mesh actually
    # uses, so a consumer can see at a glance whether the moving part it
    # cares about is even bound.
    used = sorted(set(int(x) for x in np.unique(resolved)))
    order = {b: i for i, b in enumerate(used)}
    palette = [bone_names[b] for b in used]
    packed = np.vectorize(order.get)(resolved).astype(np.int32)

    mass = np.zeros(len(palette))
    for k in range(packed.shape[1]):
        np.add.at(mass, packed[:, k], bw[:, k])
    top = np.argsort(-mass)[:6]
    lines.append("  skin: %d bones bound of %d in the skeleton; heaviest %s"
                 % (len(palette), len(bone_names),
                    ", ".join("%s %.1f%%" % (palette[i], 100.0 * mass[i]
                                             / max(1.0, mass.sum()))
                              for i in top)))

    return {"schema": "iji/mesh-skin/v1",
            "palette": palette,
            "palette_skeleton_index": used,
            "index": packed,
            "weight": bw.astype(np.float32),
            "weight_mass": mass.astype(np.float32),
            "remap_slice": None if remap is None else [lo, hi],
            "weight_stream": "byte/255" if byte_scaled else "float",
            "index_meaning": "index into `palette`, which holds MODEL "
                             "SKELETON BONE NAMES after m_remappingTable was "
                             "applied. Join to a clip by NAME against the "
                             "skeleton's m_boneIDs; the raw BLENDINDICES_0 "
                             "values are gone on purpose.",
            }, lines


def _qmul(a, b):
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return np.array([aw * bx + ax * bw + ay * bz - az * by,
                     aw * by - ax * bz + ay * bw + az * bx,
                     aw * bz + ax * by - ay * bx + az * bw,
                     aw * bw - ax * bx - ay * by - az * bz])


def _qrot(q, v):
    x, y, z, w = q
    u = np.array([x, y, z])
    v = np.asarray(v, np.float64)
    return (v * (w * w - u @ u) + 2.0 * u * (v @ u if v.ndim == 1
                                             else v @ u)[..., None]
            + 2.0 * w * np.cross(u, v)) if v.ndim > 1 else \
        (v * (w * w - u @ u) + 2.0 * u * (v @ u) + 2.0 * w * np.cross(u, v))


def bind_model_space(model_skeleton):
    """Each bone's BIND transform in model space, accumulated down parents.

    Shipped with the bundle because a consumer posing the mesh needs
    pose * inverse(bind) per bone, and re-deriving the bind from a different
    reading of the skeleton is how two halves of one chain drift apart.
    """
    names = list(model_skeleton["m_boneName"])
    parents = list(model_skeleton["m_nParent"])
    lt = [np.asarray(p, np.float64) for p in model_skeleton["m_bonePosParent"]]
    lq = [np.asarray(q, np.float64) for q in model_skeleton["m_boneRotParent"]]
    depth = []
    for i in range(len(names)):
        d, j = 0, parents[i]
        while j >= 0:
            d += 1
            j = parents[j]
        depth.append(d)
    out = {}
    for i in sorted(range(len(names)), key=lambda k: depth[k]):
        p = parents[i]
        if p < 0:
            out[names[i]] = (lt[i], lq[i])
        else:
            pt, pq = out[names[p]]
            out[names[i]] = (pt + _qrot(pq, lt[i]), _qmul(pq, lq[i]))
    return out


def palette_ancestors(model_skeleton, palette):
    """Per palette bone, its ancestor NAMES up m_nParent -- nearest first.

    THE PARENT CHAIN IS A FIELD, so it is read as one. A palette bone the
    animation rig does not name still has a place on the model skeleton, and
    that place is the only thing that says what it should ride. Deriving the
    attachment from bind-transform proximity would put a scope on whichever
    bone happens to be closest, which is a guess that fails silently.

    Names, not indices, for the same reason read_skin returns names: the
    consumer joins against a clip's m_boneIDs, and an index into a skeleton
    it does not hold is an invitation to reindex against the wrong list.
    """
    names = list(model_skeleton["m_boneName"])
    parents = [int(p) for p in model_skeleton["m_nParent"]]
    out = []
    for n in palette:
        chain, j, seen = [], parents[names.index(n)], set()
        while j >= 0 and j not in seen:
            seen.add(j)
            chain.append(names[j])
            j = parents[j]
        out.append(chain)
    return out


def add_skin(bundle, skin, model_skeleton, n_vertices):
    """Attach the skin block to a bundle, or say plainly that there is none.

    THE VERTEX ORDER IS THE CONTRACT. The bundle's baked positions and these
    bindings index the same array, so a length mismatch means they came from
    different reads of the mesh and the bundle would pose the wrong vertices.
    """
    import torch

    if skin is None:
        bundle["skin"] = None
        bundle["skin_absent_reason"] = (
            "the mesh ships no BLENDINDICES_0 / BLENDWEIGHT_0; it is rigid "
            "and no binding is invented for it")
        return bundle
    if len(skin["index"]) != n_vertices:
        raise SystemExit(
            "skin has %d vertices but the bundle bakes %d -- the bindings and "
            "the positions came from different reads of the mesh"
            % (len(skin["index"]), n_vertices))

    bind = bind_model_space(model_skeleton)
    missing = [n for n in skin["palette"] if n not in bind]
    if missing:
        raise SystemExit("palette bones absent from the model skeleton: %s"
                         % missing[:6])
    bt = np.stack([bind[n][0] for n in skin["palette"]])
    bq = np.stack([bind[n][1] for n in skin["palette"]])
    anc = palette_ancestors(model_skeleton, skin["palette"])

    bundle["skin"] = {
        "schema": skin["schema"],
        "palette": skin["palette"],
        "index": torch.tensor(skin["index"], dtype=torch.int32),
        "weight": torch.tensor(skin["weight"], dtype=torch.float32),
        "bind_t": torch.tensor(bt, dtype=torch.float32),
        "bind_q": torch.tensor(bq, dtype=torch.float32),
        "index_meaning": skin["index_meaning"],
        "bind_meaning": "MODEL-SPACE bind (t, q) per palette bone, xyzw. "
                        "Pose a vertex with sum_k w_k * (pose[b_k] o "
                        "inverse(bind[b_k]))(loc), where `loc` is the "
                        "bundle's SOURCE-UNIT positions -- not `pos`, which "
                        "is already view-space and hold-posed.",
        "weight_stream": skin["weight_stream"],
        "remap_slice": skin["remap_slice"],
        "palette_ancestors": anc,
        "palette_ancestors_meaning":
            "Per palette bone, the MODEL SKELETON ancestor names walked up "
            "m_nParent, nearest parent first, root last -- READ from "
            "m_nParent/m_boneName, never inferred from bind proximity. A "
            "consumer whose animation rig does not name a palette bone "
            "substitutes the FIRST ancestor in this list that the rig DOES "
            "name: a bone rigidly attached to its ancestor satisfies "
            "pose o inverse(bind) = pose_anc o inverse(bind_anc), so the "
            "substitution is exact for a rigid attachment and is the only "
            "transform in the .vnmskel frame available for that bone. "
            "IDENTITY is NOT that transform -- it leaves the vertices in "
            "the .vmdl_c frame while every posed bone moved to the "
            ".vnmskel one.",
    }
    bundle["skin_absent_reason"] = None
    return bundle


def moving_part_report(skin, names):
    """How much of the mesh rides on the named bones, e.g. bolt / slide.

    A weapon whose `bolt` carries zero vertices cannot show a cycling bolt no
    matter what the clip does, and that is a fixture fact worth printing
    beside the clip's motion rather than discovering in a render.
    """
    if skin is None:
        return "rigid mesh, no bindings"
    total = float(skin["weight_mass"].sum()) or 1.0
    out = []
    for n in names:
        if n in skin["palette"]:
            i = skin["palette"].index(n)
            out.append("%s %.2f%%" % (n, 100.0 * skin["weight_mass"][i] / total))
        else:
            out.append("%s NOT BOUND" % n)
    return ", ".join(out)
