#!/usr/bin/env python
"""Precompute frozen DINOv2 ViT-S/14 patch features for the converted PushT dataset.

The DINO-WM encoder is frozen, so its features are identical every epoch. Compute them
ONCE and store (L, 196, 384) fp16 per episode at {dst}/{split}/feats/episode_{row:05d}.npy.
Training then reads features (page-cache friendly) instead of decoding mp4 + running DINOv2
each step (~30x cheaper) — mathematically identical (model patch D10, dataset D11).

Frames are read from the per-episode mp4 (default; fast, sequential, one open/episode) —
NOT the 46GB h5 (NFS + hdf5 decompression makes 8 concurrent readers ~100x slower). The
mp4 frames are bit-exact vs the h5 (verified), so features are identical.

Feature path EXACTLY mirrors VWorldModel.encode_obs:
  pixels(uint8) /255 -> default_transform(224)[Resize224,CenterCrop224,Norm0.5]
  -> Resize((224//16)*14=196) -> DinoV2Encoder.forward -> x_norm_patchtokens.

Run with the dino_wm conda env. Shard across GPUs via --num-shards/--shard + CUDA_VISIBLE_DEVICES.
"""
import os, sys, time, pickle, argparse
import numpy as np
import torch

sys.path.insert(0, "/home/manu/stable-worldmodel/dino_wm")
from torchvision import transforms as T
from datasets.img_transforms import default_transform
from models.dino import DinoV2Encoder


def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--dst", default="/nas/manu/stable_worldmodel/datasets/pusht_noise")
    p.add_argument("--split", choices=["train", "val"], required=True)
    p.add_argument("--num-shards", type=int, default=1)
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--img-size", type=int, default=224)
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--limit-rows", type=int, default=0)
    return p.parse_args()


def main():
    a = parse()
    import decord
    decord.bridge.set_bridge("native")
    from decord import VideoReader

    split_dir = os.path.join(a.dst, a.split)
    obs_dir = os.path.join(split_dir, "obses")
    out = os.path.join(split_dir, "feats")
    os.makedirs(out, exist_ok=True)
    seq = pickle.load(open(os.path.join(split_dir, "seq_lengths.pkl"), "rb"))
    N = len(seq)

    enc = DinoV2Encoder("dinov2_vits14", "x_norm_patchtokens").cuda().eval()
    dtf = default_transform(a.img_size)                 # Resize224,CenterCrop224,Norm0.5
    etf = T.Resize((a.img_size // 16) * enc.patch_size)  # Resize(196)

    rows = list(range(a.shard, N, a.num_shards))
    if a.limit_rows:
        rows = rows[:a.limit_rows]
    t0 = time.time(); done = 0
    for row in rows:
        outp = os.path.join(out, f"episode_{row:05d}.npy")
        if os.path.exists(outp):
            continue
        vr = VideoReader(os.path.join(obs_dir, f"episode_{row:05d}.mp4"), num_threads=2)
        L = len(vr)
        assert L == seq[row], f"row {row}: mp4 has {L} frames, seq_lengths says {seq[row]}"
        px = vr.get_batch(range(L)).asnumpy()           # (L,224,224,3) uint8
        x = torch.from_numpy(px).permute(0, 3, 1, 2).float() / 255.0
        x = dtf(x); x = etf(x)                           # (L,3,196,196)
        outs = []
        with torch.no_grad():
            for i in range(0, L, a.batch):
                outs.append(enc(x[i:i + a.batch].cuda()).half().cpu())
        feat = torch.cat(outs, 0).numpy()                # (L,196,384) fp16
        np.save(outp, feat)
        done += 1
        if done % 500 == 0:
            print(f"[{a.split} shard{a.shard}] {done}/{len(rows)} {time.time()-t0:.0f}s", flush=True)
    print(f"[{a.split} shard{a.shard}] DONE wrote~{done}/{len(rows)} {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
