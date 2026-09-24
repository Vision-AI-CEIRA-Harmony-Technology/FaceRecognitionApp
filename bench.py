"""CPU vs GPU latency on this machine, per pipeline step.

Same checkpoint, same PyTorch model, same timing harness - only the device
changes. That single-variable setup is what makes the comparison meaningful.

    python bench.py --weights ../models/Glint360K_R100_TopoFR_9760.pt \
                    --dataset ../data/final_dataset --trials 30

Run it on an idle machine: background load inflates the CPU numbers badly.
"""
import argparse
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config as C
import dataset as D
import pipeline as P
from engine import Embedder, available_devices, device_label

WARMUP = 5


def summarize(times):
    return {k: (float(np.mean(v)), float(np.median(v))) for k, v in times.items()}


def report(title, stats):
    print(f"\n  {title}")
    print(f"    {'step':<22}{'mean':>11}{'median':>11}")
    for step, (mean, med) in stats.items():
        if step == "TOTAL":
            print("  " + "-" * 42)
        print(f"    {step:<22}{mean:>8.2f} ms{med:>8.2f} ms")


def bench_verification(embedder, pairs, trials, threshold):
    sync = embedder.sync
    steps = {k: [] for k in ["preprocess_a", "embed_a", "preprocess_b", "embed_b",
                             "normalize", "compare", "decide"]}
    totals = []
    for ref, match in pairs[:trials]:
        ia, ib = cv2.imread(ref), cv2.imread(match)
        if ia is None or ib is None:
            continue
        a, t_pa = P.timed(lambda: P.to_tensor(ia))
        ea, t_ea = P.timed(lambda: embedder.embed(a), sync)
        b, t_pb = P.timed(lambda: P.to_tensor(ib))
        eb, t_eb = P.timed(lambda: embedder.embed(b), sync)
        (na, nb), t_n = P.timed(lambda: (P.normalize(ea), P.normalize(eb)))
        score, t_c = P.timed(lambda: P.cosine(na, nb))
        _, t_d = P.timed(lambda: bool(score >= threshold))
        vals = [t_pa, t_ea, t_pb, t_eb, t_n, t_c, t_d]
        for k, v in zip(steps, vals):
            steps[k].append(v)
        totals.append(sum(vals))
    stats = summarize(steps)
    stats["TOTAL"] = (float(np.mean(totals)), float(np.median(totals)))
    return stats


def bench_identification(embedder, probes, gallery, trials):
    sync = embedder.sync
    steps = {k: [] for k in ["preprocess_probe", "embed_probe", "normalize",
                             "search", "rank"]}
    totals = []
    for probe in probes[:trials]:
        img = cv2.imread(probe)
        if img is None:
            continue
        a, t_p = P.timed(lambda: P.to_tensor(img))
        e, t_e = P.timed(lambda: embedder.embed(a), sync)
        n, t_n = P.timed(lambda: P.normalize(e))
        search_result, t_s = P.timed(lambda: P.search(gallery, n))
        identity_sims = search_result[1]
        _, t_r = P.timed(lambda: np.argsort(-identity_sims)[:5])
        vals = [t_p, t_e, t_n, t_s, t_r]
        for k, v in zip(steps, vals):
            steps[k].append(v)
        totals.append(sum(vals))
    stats = summarize(steps)
    stats["TOTAL"] = (float(np.mean(totals)), float(np.median(totals)))
    return stats


def main():
    settings = C.load_settings()
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--weights", default=settings.get("weights_path"),
                    help="default: whatever the app was last set up with")
    ap.add_argument("--dataset", default=settings.get("dataset_root"))
    ap.add_argument("--gallery", default=settings.get("gallery_path"))
    ap.add_argument("--backbone", default=settings.get("backbone"))
    ap.add_argument("--trials", type=int, default=30)
    ap.add_argument("--threshold", type=float, default=C.VERIF_THRESHOLD)
    args = ap.parse_args()

    for name, value in (("--weights", args.weights), ("--dataset", args.dataset),
                        ("--gallery", args.gallery)):
        if not value:
            raise SystemExit(f"{name} is required (or run the app once to set it up)")

    gallery = P.load_gallery(args.gallery)
    if gallery is None:
        raise SystemExit(f"no gallery at {args.gallery} - run enroll.py first")
    n_gal = gallery["meta"].get("n_identities", len(set(gallery["names"])))

    # Same seed as the notebook, so the trial set is reproducible across machines.
    rng = np.random.default_rng(42)
    scan = D.scan(args.dataset)
    by_name = {i.name: i for i in scan["identities"]}
    picked = rng.permutation(np.unique(gallery["names"]))

    pairs, probes = [], []
    for name in picked:
        ident = by_name.get(str(name))
        if ident and ident.matches:
            for match in ident.matches:
                pairs.append((ident.reference, match))
                probes.append(match)
                if len(pairs) >= args.trials:
                    break
        if len(pairs) >= args.trials:
            break
    if not pairs:
        raise SystemExit(
            "no reference/match pairs found in the dataset - 1:1 needs reference "
            "and match images.")

    print(f"{os.path.basename(args.weights)} / PyTorch - {len(pairs)} trials, "
          f"gallery {n_gal:,}")

    results = {}
    for dev in available_devices():
        print(f"\n{'=' * 52}\n{device_label(dev)}\n{'=' * 52}")
        embedder = Embedder(args.weights, dev, args.backbone).warmup(WARMUP)

        results[dev] = {
            "1:1": bench_verification(embedder, pairs, args.trials, args.threshold),
            "1:N": bench_identification(embedder, probes, gallery, args.trials),
        }
        report("1:1 verification", results[dev]["1:1"])
        report(f"1:N identification (gallery {n_gal:,}, pre-embedded)",
               results[dev]["1:N"])
        embedder.free()

    if "gpu" in results and "cpu" in results:
        print(f"\n{'=' * 52}\nGPU speed-up (CPU time / GPU time)\n{'=' * 52}")
        for task in ("1:1", "1:N"):
            c = results["cpu"][task]["TOTAL"][0]
            g = results["gpu"][task]["TOTAL"][0]
            print(f"  {task}  {c:8.1f} ms -> {g:7.1f} ms   {c / g:5.2f}x")
    else:
        print("\n  Only one device available - no comparison. "
              "Run `python check_devices.py` to see why.")


if __name__ == "__main__":
    main()
