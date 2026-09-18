
import argparse
import numpy as np
import torch

from model    import HipCatModel, SILENT
from data     import sound_size, WORD
from deviants import make_deviant_stream

A, B, C, D = WORD


def build(cfg, device):
    keys = ["input_size", "n_dg", "n_ca3", "n_ca1", "settle_steps", "initial_cycles",
            "state_carry", "predictive_ca3", "bigloop_from_prev", "theta_gates_ca3",
            "seed", "lr_ecin_dg", "lr_ecin_ca3", "lr_ca3_ca3", "lr_ca3_ca1",
            "eval_readout", "rec_recall_mult", "rec_recall_mult_eval", "mossy_recall_mult",
            "slot_code", "predictive_ca3", "hebb_ca3_ca3"]
    kw = {k: cfg[k] for k in keys if k in cfg}
    kw.setdefault("input_size", sound_size)
    return HipCatModel(device=device, **kw).to(device)


@torch.no_grad()
def run(model, seq, prev, nxt, word_pos, dev_pos, is_dev):
    B, T = seq.shape
    dev = model.device
    seq_d, prev_d, nxt_d = seq.to(dev), prev.to(dev), nxt.to(dev)
    model.reset_state()
    ecin  = torch.zeros(B, T, getattr(model, "n_ecin", model.input_size))
    ecout = torch.zeros(B, T, model.input_size)
    ca3   = torch.zeros(B, T, model.n_ca3)
    pred  = torch.zeros(B, T, dtype=torch.long)
    for t in range(T):
        _, pidx, snap = model.eval_step(seq_d[:, t], prev_d[:, t], nxt_d[:, t],
                                        capture_initial=False)
        s = snap["settled"]
        ecin[:, t]  = s["ECin"].cpu()
        ecout[:, t] = s["ECout"].cpu()
        ca3[:, t]   = s["CA3"].cpu()
        pred[:, t]  = pidx.cpu()

    gap_mask   = (dev_pos == 2)
    afterD     = torch.zeros_like(gap_mask)
    afterD[:, 1:] = gap_mask[:, :-1] & (seq[:, 1:] == D)
    intactC    = (seq == C) & (word_pos == 2) & (~is_dev)

    out = {}
    # current-slot content at the gap = EC_in minus the clamped previous slot
    N = model.input_size
    if getattr(model, "slot_code", "shared") == "separate":
        ecin_cur = ecin[..., :N]
    else:
        ecin_cur = ecin.clone()
        bi, ti = torch.meshgrid(torch.arange(B), torch.arange(T), indexing="ij")
        ecin_cur[bi, ti, prev.clamp(min=0)] = 0.0
    out["recall@gap"] = float((ecin_cur[gap_mask].argmax(1) == C).float().mean())
    out["pred@gap"]   = float((pred[gap_mask] == D).float().mean())
    out["pred@D"]     = float((pred[afterD] == nxt[afterD]).float().mean())

    # overlap of the binarised CA3 pattern at the gap with the standard C code
    # (prototype from the intact words of the same stream), the heatmap metric
    ref = (ca3[intactC].mean(0) > 0.3).float()
    cos = []
    for b, t in zip(*torch.where(gap_mask)):
        x = (ca3[b, t] > 0.01).float()
        cos.append(float((x * ref).sum() / ((x.sum() * ref.sum()).sqrt() + 1e-8)))
    out["ca3_match"] = float(np.mean(cos)) if cos else float("nan")
    out["n_gaps"] = int(gap_mask.sum())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, default=None)
    ap.add_argument("--train", type=int, default=0, help="epochs to train if no ckpt")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--T", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--gap_type", type=str, default="silent", choices=["silent", "gap"],
                    help="silent = omitted C (SILENT pip); gap = C replaced by a background item")
    args = ap.parse_args()

    if args.ckpt:
        ck = torch.load(args.ckpt, map_location=args.device, weights_only=False)
        model = build(ck["config"], args.device)
        model.load_state_dict(ck["state_dict"], strict=False)
    else:
        from training import train_model
        model, _ = train_model(num_epochs=args.train, batch_size=args.batch, T=args.T,
                               ckpt_path="gap_test_tmp.pt", seed=args.seed,
                               device=args.device, verbose=True)

    seq, prev, nxt, wp, dev_pos, onset, is_dev = make_deviant_stream(
        batch_size=args.batch, T=args.T, dev_type=args.gap_type, dev_fraction=0.5,
        seed=args.seed + 999)

    configs = [("intact",             []),
               ("big loop off",       ["bigloop"]),
               ("CA3->CA3 off",       ["ca3_ca3"]),
               ("both off",           ["bigloop", "ca3_ca3"]),
               ("TSP off (MSP only)", ["ecin_dg", "ecin_ca3", "dg_ca3", "ca3_ca3", "ca3_ca1"])]
    print(f"\n{'config':22s} recall@gap  pred@gap  pred@D  ca3_match  (n_gaps)")
    for name, les in configs:
        model.reset_lesions()
        for p in les:
            model.set_pathway(p, False)
        r = run(model, seq, prev, nxt, wp, dev_pos, is_dev)
        print(f"{name:22s} {r['recall@gap']:9.2f} {r['pred@gap']:9.2f} "
              f"{r['pred@D']:7.2f} {r['ca3_match']:9.2f}   ({r['n_gaps']})")
    model.reset_lesions()
    print(f"\nchance for pred@* = {1/sound_size:.3f}")


if __name__ == "__main__":
    main()
