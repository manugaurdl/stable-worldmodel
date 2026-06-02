#!/usr/bin/env python
"""Convert our PushT expert HDF5 into DINO-WM's `pusht_noise/{train,val}` layout.

Source: $STABLEWM_HOME/datasets/pusht_expert_train.h5
  columns (verified): state(7D=[ax,ay,bx,by,theta,avx,avy]), action(2D, == DINO-WM's
  post-/action_scale relative action), pixels(uint8 224x224x3, RAW not PNG),
  proprio(4D), episode_idx, step_idx, ep_len(18685), ep_offset(18685).

Target (what dino_wm/datasets/pusht_dset.py reads), per split dir:
  states.pth      (N, T_max, 5)  float32  [ax,ay,bx,by,theta]
  velocities.pth  (N, T_max, 2)  float32  [avx,avy]
  rel_actions.pth (N, T_max, 2)  float32  = h5.action * action_scale(=100)
  abs_actions.pth (N, T_max, 2)  float32  = agent_pos + h5.action*100  (unused when relative=True)
  seq_lengths.pkl  list[int]
  shapes.pkl       ['T']*N
  obses/episode_{i:05d}.mp4   lossless RGB, exactly seq_len frames, fps 10

Also writes stats.json (recomputed STATE/PROPRIO mean/std over the TRAIN split) — these
must be patched into pusht_dset.py:16-19 (hardcoded ones are for the original pusht_noise).

Run with the repo .venv python (has h5py/hdf5plugin/decord/imageio-ffmpeg).
"""
import os, sys, json, pickle, argparse, time
from pathlib import Path
from multiprocessing import Pool

import numpy as np
import h5py
try:
    import hdf5plugin  # noqa: F401  (registers the pixel codec)
except Exception:
    pass

ACTION_SCALE = 100.0


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--src", default="/nas/manu/stable_worldmodel/datasets/pusht_expert_train.h5")
    p.add_argument("--dst", default="/nas/manu/stable_worldmodel/datasets/pusht_noise")
    p.add_argument("--split", type=float, default=0.9)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--limit", type=int, default=0, help="cap total episodes for a smoke test (0=all)")
    p.add_argument("--no-video", action="store_true", help="skip mp4 encode (tensor-only test)")
    p.add_argument("--fps", type=int, default=10)
    return p.parse_args()


# ---- mp4 worker (own h5 handle per process) -------------------------------------------
_W = {}

def _winit(src):
    try:
        import hdf5plugin  # noqa
    except Exception:
        pass
    _W["f"] = h5py.File(src, "r")

def _write_one(job):
    """job = (row, off, L, out_path, fps). Returns (row, ok, msg)."""
    import imageio.v2 as iio
    from decord import VideoReader
    row, off, L, out_path, fps = job
    f = _W["f"]
    px = f["pixels"][off:off + L]            # (L,224,224,3) uint8
    w = iio.get_writer(
        out_path, fps=fps, codec="libx264rgb", pixelformat="rgb24",
        macro_block_size=1, output_params=["-qp", "0", "-preset", "veryfast"],
    )
    for fr in px:
        w.append_data(np.ascontiguousarray(fr))
    w.close()
    n = len(VideoReader(out_path))
    if n != L:
        return (row, False, f"frame count {n} != seq_len {L}")
    return (row, True, "")


def build_split(f, ep_gids, ep_off, ep_len, out_dir, T_max, args):
    """Build tensors in the main proc (cheap sequential reads); encode mp4 in a pool."""
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "obses").mkdir(exist_ok=True)
    N = len(ep_gids)
    states = np.zeros((N, T_max, 5), np.float32)
    vels   = np.zeros((N, T_max, 2), np.float32)
    rel    = np.zeros((N, T_max, 2), np.float32)
    abs_   = np.zeros((N, T_max, 2), np.float32)
    seq_lengths = []
    jobs = []
    for row, gid in enumerate(ep_gids):
        off, L = int(ep_off[gid]), int(ep_len[gid])
        st = f["state"][off:off + L].astype(np.float32)     # (L,7)
        ac = f["action"][off:off + L].astype(np.float32)    # (L,2)
        si = f["step_idx"][off:off + L]
        assert np.array_equal(si, np.arange(L)), f"step_idx not 0..L-1 for gid {gid}"
        states[row, :L] = st[:, 0:5]
        vels[row, :L]   = st[:, 5:7]
        rel[row, :L]    = ac * ACTION_SCALE
        abs_[row, :L]   = st[:, 0:2] + ac * ACTION_SCALE
        seq_lengths.append(L)
        jobs.append((row, off, L, str(out_dir / "obses" / f"episode_{row:05d}.mp4"), args.fps))

    import torch
    torch.save(torch.from_numpy(states), out_dir / "states.pth")
    torch.save(torch.from_numpy(vels),   out_dir / "velocities.pth")
    torch.save(torch.from_numpy(rel),    out_dir / "rel_actions.pth")
    torch.save(torch.from_numpy(abs_),   out_dir / "abs_actions.pth")
    with open(out_dir / "seq_lengths.pkl", "wb") as fh:
        pickle.dump(seq_lengths, fh)
    with open(out_dir / "shapes.pkl", "wb") as fh:
        pickle.dump(["T"] * N, fh)

    # action round-trip sanity (definitional + physical-looseness)
    assert np.allclose(rel[0, :seq_lengths[0]] / ACTION_SCALE,
                       f["action"][int(ep_off[ep_gids[0]]):int(ep_off[ep_gids[0]]) + seq_lengths[0]],
                       atol=1e-4), "rel action round-trip failed"

    if args.no_video:
        print(f"[{out_dir.name}] tensors only (no video): N={N}")
        return seq_lengths

    t0 = time.time()
    fails = []
    with Pool(args.workers, initializer=_winit, initargs=(args.src,)) as pool:
        for i, (row, ok, msg) in enumerate(pool.imap_unordered(_write_one, jobs, chunksize=8)):
            if not ok:
                fails.append((row, msg))
            if (i + 1) % 500 == 0:
                print(f"[{out_dir.name}] {i+1}/{N} mp4 written ({time.time()-t0:.0f}s)", flush=True)
    if fails:
        raise RuntimeError(f"{len(fails)} mp4 frame-count mismatches, e.g. {fails[:3]}")
    print(f"[{out_dir.name}] done: N={N}, {time.time()-t0:.0f}s", flush=True)
    return seq_lengths


def compute_stats(f, train_gids, ep_off, ep_len):
    n = 0
    s = np.zeros(7, np.float64)
    ss = np.zeros(7, np.float64)
    for gid in train_gids:
        off, L = int(ep_off[gid]), int(ep_len[gid])
        st = f["state"][off:off + L].astype(np.float64)
        n += L
        s += st.sum(0)
        ss += (st ** 2).sum(0)
    mean = s / n
    std = np.sqrt(np.maximum(ss / n - mean ** 2, 1e-8))
    return {
        "STATE_MEAN": mean.tolist(),
        "STATE_STD": std.tolist(),
        "PROPRIO_MEAN": mean[[0, 1, 5, 6]].tolist(),
        "PROPRIO_STD": std[[0, 1, 5, 6]].tolist(),
        "ACTION_MEAN": [-0.0087, 0.0068],   # kept hardcoded; matches our data
        "ACTION_STD": [0.2019, 0.2002],
        "n_steps": int(n),
    }


def main():
    args = parse_args()
    src, dst = args.src, Path(args.dst)
    f = h5py.File(src, "r")
    ep_len = f["ep_len"][:]
    ep_off = f["ep_offset"][:]
    n_ep = len(ep_len)
    T_max = int(ep_len.max())

    perm = np.random.default_rng(args.seed).permutation(n_ep)
    if args.limit:
        perm = perm[:args.limit]
    cut = int(args.split * len(perm))
    train_gids, val_gids = perm[:cut], perm[cut:]
    print(f"n_ep={n_ep} (using {len(perm)}) T_max={T_max} -> train={len(train_gids)} val={len(val_gids)}", flush=True)

    print("computing STATE/PROPRIO stats over train split...", flush=True)
    stats = compute_stats(f, train_gids, ep_off, ep_len)
    dst.mkdir(parents=True, exist_ok=True)
    with open(dst / "stats.json", "w") as fh:
        json.dump(stats, fh, indent=2)
    print("STATS:", json.dumps(stats, indent=2), flush=True)

    build_split(f, train_gids, ep_off, ep_len, dst / "train", T_max, args)
    build_split(f, val_gids,   ep_off, ep_len, dst / "val",   T_max, args)
    print("\nCONVERSION DONE.")
    print(">>> PATCH dino_wm/datasets/pusht_dset.py lines 16-19 with STATE/PROPRIO stats above.")


if __name__ == "__main__":
    main()
