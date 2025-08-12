#!/usr/bin/env python3
# train_bc.py — Minimal Behavior Cloning (no imitation lib)
# Uses: numpy, torch
# Expects in --data: rollouts.npz (obs, acts, dones, episode_starts), stats.json (mean/std)

import os, json, argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# ----------------------- Data ---------------------------------

def load_dataset(data_dir: Path):
    npz = np.load(data_dir / "rollouts.npz")
    obs  = npz["obs"].astype(np.float32)     # [N, obs_dim]
    acts = npz["acts"].astype(np.float32)    # [N, act_dim]
    with open(data_dir / "stats.json", "r") as f:
        stats = json.load(f)
    obs_mean = np.array(stats["mean"], dtype=np.float32)
    obs_std  = np.array(stats["std"],  dtype=np.float32)
    obs_std[obs_std < 1e-6] = 1e-6
    return obs, acts, obs_mean, obs_std, stats

def make_splits(N, val_frac=0.1, seed=42):
    rng = np.random.default_rng(seed)
    idx = np.arange(N)
    rng.shuffle(idx)
    n_val = int(N * val_frac)
    return idx[n_val:], idx[:n_val]

class BCDataset(Dataset):
    def __init__(self, obs, acts):
        self.obs = obs
        self.acts = acts
    def __len__(self): return self.obs.shape[0]
    def __getitem__(self, i): return self.obs[i], self.acts[i]

# ----------------------- Model --------------------------------

class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden=(256,256)):
        super().__init__()
        layers = []
        last = in_dim
        for h in hidden:
            layers += [nn.Linear(last, h), nn.ReLU()]
            last = h
        layers += [nn.Linear(last, out_dim), nn.Tanh()]  # predict scaled actions in [-1,1]
        self.net = nn.Sequential(*layers)
    def forward(self, x): return self.net(x)

# ----------------------- Train --------------------------------

def train_one_epoch(model, loader, opt, device):
    model.train()
    loss_fn = nn.MSELoss()
    total = 0.0
    for ob, ac in loader:
        ob = ob.to(device); ac = ac.to(device)
        pred = model(ob)
        loss = loss_fn(pred, ac)
        opt.zero_grad(); loss.backward(); opt.step()
        total += loss.item() * ob.size(0)
    return total / max(1, len(loader.dataset))

@torch.no_grad()
def eval_loss(model, loader, device):
    model.eval()
    loss_fn = nn.MSELoss()
    total = 0.0
    for ob, ac in loader:
        ob = ob.to(device); ac = ac.to(device)
        pred = model(ob)
        loss = loss_fn(pred, ac)
        total += loss.item() * ob.size(0)
    return total / max(1, len(loader.dataset))

# ----------------------- Main ---------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, default="./datasets", help="Folder with rollouts.npz + stats.json")
    ap.add_argument("--out",  type=str, default="./bc_out", help="Output dir")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--bs", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    data_dir = Path(os.path.expanduser(args.data))
    out_dir = Path(os.path.expanduser(args.out)); out_dir.mkdir(parents=True, exist_ok=True)

    # Load + normalize obs
    obs, acts, obs_mean, obs_std, meta = load_dataset(data_dir)
    obs_n = (obs - obs_mean) / obs_std

    # Scale actions to [-1,1] with robust per-dim max (99th percentile)
    a_scale = np.percentile(np.abs(acts), 99.0, axis=0).astype(np.float32)
    a_scale[a_scale < 1e-6] = 1e-6
    acts_n = np.clip(acts / a_scale, -1.0, 1.0)

    # Split
    tr_idx, va_idx = make_splits(obs_n.shape[0], val_frac=0.1, seed=args.seed)
    tr_ds = BCDataset(obs_n[tr_idx], acts_n[tr_idx])
    va_ds = BCDataset(obs_n[va_idx], acts_n[va_idx])
    tr_ld = DataLoader(tr_ds, batch_size=args.bs, shuffle=True, drop_last=False)
    va_ld = DataLoader(va_ds, batch_size=4096, shuffle=False, drop_last=False)

    # Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MLP(in_dim=obs_n.shape[1], out_dim=acts_n.shape[1], hidden=(256,256)).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Print dataset info & baseline MSE before training
    print(f"[info] Dataset size: {obs_n.shape[0]} samples")
    print(f"[info] Obs dim: {obs_n.shape[1]}, Act dim: {acts_n.shape[1]}")
    print(f"[info] Train/Val split: {len(tr_ds)} / {len(va_ds)}")
    base_val_loss = eval_loss(model, va_ld, device)
    print(f"[baseline] Untrained model val MSE: {base_val_loss:.6f}")


    # Train
    best = float("inf")
    for ep in range(1, args.epochs + 1):
        tr_loss = train_one_epoch(model, tr_ld, opt, device)
        va_loss = eval_loss(model, va_ld, device)
        print(f"[{ep:03d}] train {tr_loss:.6f}  val {va_loss:.6f}")
        if va_loss < best:
            best = va_loss
            torch.save(model.state_dict(), out_dir / "bc_policy_best.pt")

    # Save final + scalers + small readme
    torch.save(model.state_dict(), out_dir / "bc_policy_final.pt")
    scalers = {
        "obs_mean": obs_mean.tolist(),
        "obs_std":  obs_std.tolist(),
        "act_scale": a_scale.tolist(),
        "obs_dim": int(obs_n.shape[1]),
        "act_dim": int(acts_n.shape[1]),
        "hidden": [256, 256]
    }
    with open(out_dir / "scalers.json", "w") as f: json.dump(scalers, f, indent=2)
    with open(out_dir / "README_infer.txt", "w") as f:
        f.write(
            "Inference steps:\n"
            "1) obs_n = (obs - obs_mean) / obs_std\n"
            "2) a_scaled = model(obs_n)\n"
            "3) a = a_scaled * act_scale   # Δq in joint space\n"
        )
    print(f"[done] best={best:.6f}  saved to {out_dir}")

if __name__ == "__main__":
    main()
