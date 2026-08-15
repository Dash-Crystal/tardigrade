"""Prefiltered radiance chain for --ibl-cube, from the map's own sky cube.

WHY. Every world-shading arm this lane has run printed `GAP spec-cube:
neither --ibl-cube nor --cube-probes given; the environment specular has
no radiance to fetch`. Not "the wrong radiance" -- NONE. Roughness is the
parameter that decides how an environment is SPREAD, so making roughness
correct while the environment is absent is a change whose sign nothing
predicts, which is the leading candidate for why the normal/roughness arm
measured worse on nrmse 5 frames out of 5.

The input is a READ, not a fit: `<map>.sky_cube.npz` is the reference's
own 1024^2 x6 cube, decoded from the capture's Initial Contents, one
beside every world pack. This tool turns it into the format --ibl-cube
already contracts for -- `env_cubemap_mip{0..6}_f16.npy`, each (P,6,N,N,3)
float16 -- so nothing in the renderer changes.

TWO APPROXIMATIONS, BOTH STATED, NEITHER FITTED.

1. P = 1. A sky cube is ONE environment for the whole map; the real
   chain is per-probe and parallax-corrected. Every pixel therefore
   fetches the same radiance regardless of where it stands, which is
   right for sky and wrong for anything under a roof. It is also
   strictly more radiance than the zero that is there today.

2. The chain is NOT a GGX prefilter end to end. Which mip means which
   roughness is READ off the consumer, not chosen here: gpu_render sets
   `lod = scale * sqrt(roughness)` with `scale = nmips - 1`, so mip m is
   `roughness = (m / (nmips-1))^2`. Mips 0..3 are built by box
   downsampling, because at those levels the lobe is narrower than a
   texel and a box average is the better approximation of the two. Mips
   4..6 -- roughness 0.44, 0.69, 1.00, which is the end this scene
   actually samples, mean roughness 0.836 -> lod 5.5 -- are built by
   EXPLICIT cosine-lobe convolution over the whole cube, which is the
   real integral at a resolution where it costs nothing.

   The lobe is Phong with `n = 2/a^2 - 2`, `a = roughness^2`, the
   standard GGX-to-Phong equivalence. That is an approximation of GGX and
   is named as one.

Prior art this does NOT contradict: an earlier arc measured cubemap-IBL
at parity with a 9-coefficient SH representation and dropped the cubemap
on that parity. Parity means either encoding is faithful, so this uses
whichever the renderer can already consume without new code. What is
missing today is ANY radiance, not the fancier encoding.

    python3 sky_cube_to_ibl.py /data/cs2-artifacts/worlds/de_inferno.sky_cube.npz \
        --out /data/cs2-artifacts/worlds/de_inferno.iblcube --mips 7
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import torch

# Cube face axes, the standard D3D/GL order the renderer's own sky_pass
# uses: +X -X +Y -Y +Z -Z, each as (forward, right, up).
FACES = [
    ((1, 0, 0), (0, 0, -1), (0, -1, 0)),
    ((-1, 0, 0), (0, 0, 1), (0, -1, 0)),
    ((0, 1, 0), (1, 0, 0), (0, 0, 1)),
    ((0, -1, 0), (1, 0, 0), (0, 0, -1)),
    ((0, 0, 1), (1, 0, 0), (0, -1, 0)),
    ((0, 0, -1), (-1, 0, 0), (0, -1, 0)),
]


def face_dirs(n, device):
    """(6, n, n, 3) unit directions, and (6, n, n) solid angles."""
    t = (torch.arange(n, device=device, dtype=torch.float32) + 0.5) / n
    uv = t * 2.0 - 1.0
    v, u = torch.meshgrid(uv, uv, indexing="ij")
    out, sa = [], []
    for f, r, up in FACES:
        f = torch.tensor(f, device=device, dtype=torch.float32)
        r = torch.tensor(r, device=device, dtype=torch.float32)
        up = torch.tensor(up, device=device, dtype=torch.float32)
        d = (f.view(1, 1, 3) + u.unsqueeze(-1) * r.view(1, 1, 3)
             + v.unsqueeze(-1) * up.view(1, 1, 3))
        ln = d.norm(dim=-1, keepdim=True)
        out.append(d / ln)
        # Texel solid angle on a cube: (2/n)^2 / |d|^3, with |d| the
        # UNNORMALISED length. Carried because a plain sum over texels
        # over-weights the face centres, and the whole point of the rough
        # mips is that they are an integral.
        sa.append((2.0 / n) ** 2 / (ln[..., 0] ** 3))
    return torch.stack(out), torch.stack(sa)


def box_down(cube):
    """(6,N,N,3) -> (6,N/2,N/2,3), plain 2x2 average."""
    c = cube.permute(0, 3, 1, 2)
    c = torch.nn.functional.avg_pool2d(c, 2)
    return c.permute(0, 2, 3, 1)


def convolve(src, src_dir, src_sa, n_out, power, device, chunk=4096):
    """Cosine-lobe convolution of a cube onto an n_out^2 x6 cube."""
    out_dir, _ = face_dirs(n_out, device)
    od = out_dir.reshape(-1, 3)
    sd = src_dir.reshape(-1, 3)
    sv = src.reshape(-1, 3)
    # THE SOLID ANGLE BELONGS TO THE WEIGHT, NOT TO THE VALUE. Folding it
    # into `sv` and normalising by the bare lobe sum computes
    # sum(w*L*dw)/sum(w) -- a weighted mean multiplied by a texel solid
    # angle of about 4*pi/(6*32^2) = 0.002. The first run printed mip4
    # mean 0.0053 against the source's 2.6993, a factor of 509, which is
    # what that arithmetic predicts and is why the per-mip mean is
    # printed at all: the images looked like plausible blurred cubes.
    sa = src_sa.reshape(1, -1)
    res = torch.empty((od.shape[0], 3), device=device)
    for i in range(0, od.shape[0], chunk):
        c = od[i:i + chunk]
        w = ((c @ sd.T).clamp(min=0.0) ** power) * sa
        res[i:i + chunk] = (w @ sv) / w.sum(dim=1, keepdim=True).clamp(min=1e-12)
    return res.reshape(6, n_out, n_out, 3)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("npz")
    ap.add_argument("--out", required=True)
    ap.add_argument("--mips", type=int, default=7)
    ap.add_argument("--base", type=int, default=256,
                    help="edge of mip0. The source is 1024^2; mip0 is the "
                         "sharpest level any surface can reach and a "
                         "1024^2 x6 f16 mip0 is 37 MB of VRAM for a "
                         "reflection this scene never resolves at mean "
                         "roughness 0.836.")
    ap.add_argument("--exact-from", type=int, default=4,
                    help="first mip built by explicit lobe convolution "
                         "instead of box downsampling")
    ap.add_argument("--src-edge", type=int, default=32,
                    help="edge the convolution integrates over")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available()
                    else "cpu")
    a = ap.parse_args()

    d = np.load(a.npz)
    if "cube" not in d:
        raise SystemExit(f"{a.npz} has no 'cube' array (keys {list(d)})")
    cube = torch.from_numpy(d["cube"].astype(np.float32)).to(a.device)
    if cube.shape[0] != 6 or cube.shape[-1] != 3:
        raise SystemExit(f"expected (6,N,N,3), got {tuple(cube.shape)}")
    print(f"source {tuple(cube.shape)} radiance [{float(cube.min()):.4f}, "
          f"{float(cube.max()):.4f}] mean {float(cube.mean()):.4f}")

    lvl = cube
    while lvl.shape[1] > a.base:
        lvl = box_down(lvl)
    src = lvl
    while src.shape[1] > a.src_edge:
        src = box_down(src)
    sdir, ssa = face_dirs(src.shape[1], a.device)

    os.makedirs(a.out, exist_ok=True)
    scale = a.mips - 1
    for m in range(a.mips):
        rough = (m / scale) ** 2 if scale else 0.0
        n = max(4, a.base >> m)
        if m < a.exact_from:
            while lvl.shape[1] > n:
                lvl = box_down(lvl)
            page, how = lvl, "box"
        else:
            alpha = max(rough * rough, 1e-3)
            power = max(1.0, 2.0 / alpha - 2.0)
            page = convolve(src, sdir, ssa, n, power, a.device)
            how = f"lobe n={power:.1f}"
        arr = page.unsqueeze(0).to(torch.float16).cpu().numpy()   # (1,6,N,N,3)
        p = os.path.join(a.out, f"env_cubemap_mip{m}_f16.npy")
        np.save(p, arr)
        print(f"  mip{m} {arr.shape} roughness {rough:.3f} [{how}] "
              f"mean {float(page.mean()):.4f} max {float(page.max()):.4f} "
              f"-> {p}")
    print(f"wrote {a.mips} mips to {a.out}; pass --ibl-cube {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
