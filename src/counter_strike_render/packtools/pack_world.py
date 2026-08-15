"""Pack a VRF world glb into flat GPU-renderable arrays.

Output npz:
  vertices  float32 (N, 3)   glTF world space (Y-up, meters)
  uvs       float32 (N, 2)
  faces     int32   (F, 3)
  face_mat  int32   (F,)     index into the texture array
  textures  uint8   (M, T, T, 4)  baseColor RGBA, resized to T=--tex-size
  face_normals float32 (F, 3)

Run on the render node:
  venv/bin/python pack_world.py --glb assets/inferno-gltf-v2.glb \
      --out assets/world_packed.npz --tex-size 1024
"""

import argparse
import numpy as np
import trimesh
from PIL import Image

parser = argparse.ArgumentParser()
parser.add_argument("--glb", required=True)
parser.add_argument("--out", required=True)
parser.add_argument("--tex-size", type=int, default=1024)
args = parser.parse_args()

print("loading", args.glb, flush=True)
scene = trimesh.load(args.glb, process=False)
print("geometries:", len(scene.geometry), flush=True)

T = args.tex_size
tex_layers = []          # np arrays (T,T,4)
tex_index_by_key = {}


def layer_for_material(material):
    image = None
    factor = (1.0, 1.0, 1.0, 1.0)
    if material is not None:
        image = getattr(material, "baseColorTexture", None)
        f = getattr(material, "baseColorFactor", None)
        if f is not None:
            f = np.asarray(f, dtype=np.float64).ravel()
            if f.max() > 1.001:  # stored 0-255
                f = f / 255.0
            factor = tuple(f[:4]) if f.size >= 4 else tuple(f[:3]) + (1.0,)
    key = (id(image) if image is not None else None, factor)
    if key in tex_index_by_key:
        return tex_index_by_key[key]
    if image is not None:
        img = image.convert("RGBA").resize((T, T), Image.BILINEAR)
        arr = np.asarray(img, dtype=np.uint8)
        fa = np.asarray(factor, dtype=np.float32)
        if not np.allclose(fa, 1.0):
            arr = (arr.astype(np.float32) * fa[None, None, :]).clip(0, 255
                                                                    ).astype(np.uint8)
    else:
        arr = np.empty((T, T, 4), dtype=np.uint8)
        arr[:] = (np.asarray(factor) * 255).clip(0, 255).astype(np.uint8)
    tex_index_by_key[key] = len(tex_layers)
    tex_layers.append(arr)
    return tex_index_by_key[key]


all_v, all_uv, all_f, all_fm = [], [], [], []
offset = 0
skipped = 0
for node_name in scene.graph.nodes_geometry:
    transform, geom_name = scene.graph[node_name]
    geom = scene.geometry[geom_name]
    if not isinstance(geom, trimesh.Trimesh) or geom.faces.size == 0:
        skipped += 1
        continue
    # VRF glb primitives reference small index ranges of huge shared
    # vertex buffers; compact to referenced vertices only (else the packed
    # world balloons ~100x).
    used, remapped = np.unique(geom.faces.reshape(-1), return_inverse=True)
    local_faces = remapped.reshape(-1, 3).astype(np.int64)
    v = trimesh.transformations.transform_points(
        geom.vertices[used], transform).astype(np.float32)
    uv = None
    visual = geom.visual
    if hasattr(visual, "uv") and visual.uv is not None and len(visual.uv):
        uv = np.asarray(visual.uv, dtype=np.float32)
        uv = uv[used] if len(uv) > used.max() else None
    if uv is None or len(uv) != len(v):
        uv = np.zeros((len(v), 2), dtype=np.float32)
    material = getattr(visual, "material", None)
    layer = layer_for_material(material)
    faces = local_faces + offset
    all_v.append(v)
    all_uv.append(uv)
    all_f.append(faces)
    all_fm.append(np.full(len(faces), layer, dtype=np.int32))
    offset += len(v)

vertices = np.concatenate(all_v)
uvs = np.concatenate(all_uv)
faces = np.concatenate(all_f).astype(np.int32)
face_mat = np.concatenate(all_fm)
textures = np.stack(tex_layers)

e1 = vertices[faces[:, 1]] - vertices[faces[:, 0]]
e2 = vertices[faces[:, 2]] - vertices[faces[:, 0]]
n = np.cross(e1, e2)
norm = np.linalg.norm(n, axis=1, keepdims=True)
norm[norm == 0] = 1
face_normals = (n / norm).astype(np.float32)

print(f"packed: {len(vertices):,} verts, {len(faces):,} tris, "
      f"{len(textures)} texture layers, {skipped} skipped nodes", flush=True)
lo, hi = vertices.min(axis=0), vertices.max(axis=0)
print("bbox (m):", lo.round(1), hi.round(1), flush=True)
np.savez_compressed(args.out, vertices=vertices, uvs=uvs, faces=faces,
                    face_mat=face_mat, textures=textures,
                    face_normals=face_normals)
print("wrote", args.out, flush=True)
