"""Direct GLB -> GPU-packed world, no trimesh.

Parses the glb container directly (JSON chunk + BIN chunk), builds all
vertex/index arrays as zero-copy buffer views, and does the per-node
transform + vertex compaction in torch on the GPU. Textures decode in a
process pool. Saves a torch .pt bundle for gpu_render.py.

  venv/bin/python fast_pack.py --glb assets/inferno-gltf-v2.glb \
      --out assets/world_packed.pt --tex-size 1024
"""

import argparse
import concurrent.futures as futures
import json
import os
import struct
import time

import numpy as np
import torch

parser = argparse.ArgumentParser()
parser.add_argument("--glb", required=True)
parser.add_argument("--out", required=True)
parser.add_argument("--tex-size", type=int, default=1024)
args = parser.parse_args()
t0 = time.time()
device = torch.device("cuda")

raw = open(args.glb, "rb").read()
magic, version, _length = struct.unpack_from("<III", raw, 0)
assert magic == 0x46546C67, "not a glb"
offset = 12
gltf = None
bin_chunk = None
while offset < len(raw):
    clen, ctype = struct.unpack_from("<II", raw, offset)
    offset += 8
    chunk = raw[offset:offset + clen]
    offset += clen
    if ctype == 0x4E4F534A:
        gltf = json.loads(chunk)
    elif ctype == 0x004E4942:
        bin_chunk = chunk
print(f"container parsed {time.time()-t0:.1f}s: "
      f"{len(gltf['meshes'])} meshes, {len(gltf.get('images', []))} images",
      flush=True)

buffer_views = gltf["bufferViews"]
accessors = gltf["accessors"]
COMPONENT_DTYPE = {5120: np.int8, 5121: np.uint8, 5122: np.int16,
                   5123: np.uint16, 5125: np.uint32, 5126: np.float32}
TYPE_WIDTH = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4, "MAT4": 16}


def accessor_array(index):
    acc = accessors[index]
    view = buffer_views[acc["bufferView"]]
    dtype = COMPONENT_DTYPE[acc["componentType"]]
    width = TYPE_WIDTH[acc["type"]]
    start = view.get("byteOffset", 0) + acc.get("byteOffset", 0)
    stride = view.get("byteStride")
    itemsize = np.dtype(dtype).itemsize * width
    count = acc["count"]
    if stride and stride != itemsize:
        rows = np.lib.stride_tricks.as_strided(
            np.frombuffer(bin_chunk, dtype=np.uint8,
                          offset=start, count=stride * count),
            shape=(count, itemsize), strides=(stride, 1))
        arr = rows.reshape(-1).view(dtype).reshape(count, width)
    else:
        arr = np.frombuffer(bin_chunk, dtype=dtype, offset=start,
                            count=count * width).reshape(count, width)
    return arr


# --- world transforms per node (flat scene graphs from VRF; handle TRS too)
def node_matrix(node):
    if "matrix" in node:
        return np.asarray(node["matrix"], dtype=np.float64).reshape(4, 4).T
    m = np.eye(4)
    if "scale" in node:
        m = m @ np.diag(list(node["scale"]) + [1.0])
    if "rotation" in node:
        x, y, z, w = node["rotation"]
        r = np.array([
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ])
        rm = np.eye(4); rm[:3, :3] = r
        m = rm @ m
    if "translation" in node:
        tm = np.eye(4); tm[:3, 3] = node["translation"]
        m = tm @ m
    return m


nodes = gltf["nodes"]
world_of = {}
def walk(index, parent):
    m = parent @ node_matrix(nodes[index])
    if "mesh" in nodes[index]:
        world_of.setdefault(index, m)
    for child in nodes[index].get("children", ()):
        walk(child, m)
for scene_node in gltf["scenes"][gltf.get("scene", 0)]["nodes"]:
    walk(scene_node, np.eye(4))

# --- flatten on GPU
all_v, all_uv, all_f, all_fm = [], [], [], []
offset_v = 0
for node_index, world in world_of.items():
    mesh = gltf["meshes"][nodes[node_index]["mesh"]]
    wt = torch.from_numpy(np.ascontiguousarray(world, dtype=np.float32)).to(device)
    for prim in mesh["primitives"]:
        attrs = prim["attributes"]
        if "POSITION" not in attrs or "indices" not in prim:
            continue
        idx_np = accessor_array(prim["indices"]).reshape(-1).astype(np.int64)
        idx = torch.from_numpy(idx_np).to(device)
        pos = torch.from_numpy(
            np.ascontiguousarray(accessor_array(attrs["POSITION"]),
                                 dtype=np.float32)).to(device)
        used, remapped = torch.unique(idx, return_inverse=True)
        v = pos[used]
        v = v @ wt[:3, :3].T + wt[:3, 3]
        if "TEXCOORD_0" in attrs:
            uv_np = np.ascontiguousarray(
                accessor_array(attrs["TEXCOORD_0"]), dtype=np.float32)
            uv = torch.from_numpy(uv_np).to(device)[used]
        else:
            uv = torch.zeros((len(v), 2), device=device)
        all_v.append(v)
        all_uv.append(uv)
        all_f.append(remapped.reshape(-1, 3).int() + offset_v)
        all_fm.append(torch.full((len(remapped) // 3,),
                                 prim.get("material", 0),
                                 dtype=torch.int32, device=device))
        offset_v += len(v)

vertices = torch.cat(all_v)
uvs = torch.cat(all_uv)
faces = torch.cat(all_f)
face_prim_mat = torch.cat(all_fm)
print(f"flattened {time.time()-t0:.1f}s: {len(vertices):,} verts, "
      f"{len(faces):,} tris", flush=True)
print("bbox:", vertices.min(0).values.tolist(), vertices.max(0).values.tolist(),
      flush=True)

e1 = vertices[faces[:, 1].long()] - vertices[faces[:, 0].long()]
e2 = vertices[faces[:, 2].long()] - vertices[faces[:, 0].long()]
n = torch.cross(e1, e2, dim=1)
face_normals = n / n.norm(dim=1, keepdim=True).clamp(min=1e-9)

# --- textures: material -> baseColor image layer (parallel decode)
materials = gltf.get("materials", [])
tex_to_image = [t.get("source", -1) for t in gltf.get("textures", [])]
images = gltf.get("images", [])
base = os.path.dirname(os.path.abspath(args.glb))
T = args.tex_size

mat_image = []      # per material: image index or -1
mat_factor = []
mat_mode = []       # 0 = OPAQUE, 1 = MASK (alpha-tested), 2 = BLEND
mat_cutoff = []
_MODE = {"OPAQUE": 0, "MASK": 1, "BLEND": 2}
for material in materials:
    pbr = material.get("pbrMetallicRoughness", {})
    tex = pbr.get("baseColorTexture")
    mat_image.append(tex_to_image[tex["index"]] if tex else -1)
    mat_factor.append(pbr.get("baseColorFactor", [1, 1, 1, 1]))
    # glTF declares this explicitly; guessing it from texture-alpha
    # statistics misclassifies walls whose diffuse carries junk alpha.
    mat_mode.append(_MODE.get(material.get("alphaMode", "OPAQUE"), 0))
    mat_cutoff.append(float(material.get("alphaCutoff", 0.5)))
print("alpha modes: %d opaque, %d mask, %d blend"
      % (mat_mode.count(0), mat_mode.count(1), mat_mode.count(2)), flush=True)

needed_images = sorted({i for i in mat_image if i >= 0})


def decode(image_index):
    from PIL import Image
    uri = images[image_index]["uri"]
    img = Image.open(os.path.join(base, uri)).convert("RGBA")
    return image_index, np.asarray(
        img.resize((T, T), Image.BILINEAR), dtype=np.uint8)


layer_of_image = {}
layers = []
with futures.ProcessPoolExecutor(max_workers=16) as pool:
    for image_index, arr in pool.map(decode, needed_images, chunksize=8):
        layer_of_image[image_index] = len(layers)
        layers.append(arr)
print(f"textures decoded {time.time()-t0:.1f}s: {len(layers)} layers", flush=True)

mat_layer = []
for m_index, image_index in enumerate(mat_image):
    if image_index >= 0:
        mat_layer.append(layer_of_image[image_index])
    else:
        arr = np.empty((T, T, 4), dtype=np.uint8)
        arr[:] = (np.asarray(mat_factor[m_index]) * 255).clip(0, 255
                                                              ).astype(np.uint8)
        mat_layer.append(len(layers))
        layers.append(arr)
mat_layer_t = torch.tensor(mat_layer, dtype=torch.int32, device=device)
face_mat = mat_layer_t[face_prim_mat.long()]
_mode_t = torch.tensor(mat_mode + [0], dtype=torch.int8, device=device)
_cut_t = torch.tensor(mat_cutoff + [0.5], dtype=torch.float32, device=device)
face_alpha_mode = _mode_t[face_prim_mat.long().clamp(max=len(mat_mode))]
face_alpha_cutoff = _cut_t[face_prim_mat.long().clamp(max=len(mat_cutoff))]
textures = torch.from_numpy(np.stack(layers))

torch.save({
    "vertices": vertices.cpu(), "uvs": uvs.cpu(), "faces": faces.cpu(),
    "face_mat": face_mat.cpu(), "face_normals": face_normals.cpu(),
    "face_alpha_mode": face_alpha_mode.cpu(),
    "face_alpha_cutoff": face_alpha_cutoff.cpu(),
    "textures": textures,
}, args.out)
print(f"wrote {args.out} in {time.time()-t0:.1f}s total", flush=True)
