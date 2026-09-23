"""Build the 1:N gallery: detect, align and embed one reference image per
identity, once.

Everything this costs is enrollment cost. The app never repeats it, which is what
makes the 1:N latency shown in the demo comparable to the 1:1 latency - a search
is only "embed the probe + one matrix product".

Callable from the UI (with a progress callback) or from the command line:

    python enroll.py --weights ../models/Glint360K_R100_TopoFR_9760.pt \
                     --dataset ../data/final_dataset --n 0 --device gpu
"""
import argparse
import os
import sys
from datetime import datetime, timezone
from time import perf_counter

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config as C
import dataset as D
import pipeline as P
from engine import Embedder, available_devices


def build(embedder, identities, root, out_path, batch=32, progress=None):
    """Embed every identity's reference image and write the gallery.

    Each reference is detected + aligned to a 112x112 ArcFace crop before
    embedding, using the same `P.detect_and_align` the app runs on a probe -
    a gallery aligned one way and searched another would silently compare
    embeddings that were never trained to line up. An identity whose image is
    unreadable, or where no face clears the detector's threshold, is skipped
    rather than failing the whole run; both counts land in `meta` so the
    caller can report them.

    `progress(done, total, rate)` is called after each batch so the UI can draw a
    bar; pass None from the CLI.
    """
    ctx_id = 0 if embedder.device == "gpu" else -1
    embs, names, rels = [], [], []
    total = len(identities)
    n_unreadable = 0
    n_no_face = 0
    t0 = perf_counter()

    for start in range(0, total, batch):
        chunk = identities[start:start + batch]
        arrs, keep = [], []
        for ident in chunk:
            img = cv2.imread(ident.reference)
            if img is None:                      # unreadable file - skip, do not fail
                n_unreadable += 1
                continue
            try:
                aligned, _bbox, _kps = P.detect_and_align(img, ctx_id=ctx_id)
            except P.NoFaceDetected:
                n_no_face += 1                   # no face found - skip, do not fail
                continue
            arrs.append(P.to_tensor(aligned))
            keep.append(ident)
        if arrs:
            out = embedder.embed_batch(arrs)
            embs.append(out)
            names += [i.name for i in keep]
            rels += [os.path.relpath(i.reference, root) for i in keep]

        done = min(start + batch, total)
        if progress:
            progress(done, total, done / max(perf_counter() - t0, 1e-6))

    if not embs:
        raise RuntimeError("No image could be read and have a face detected - "
                          "check the dataset folder.")

    emb = P.normalize(np.concatenate(embs, axis=0))   # normalised once, at enrollment
    meta = {
        "weights_id": embedder.id,
        "weights_name": os.path.basename(embedder.weights_path),
        "backbone": embedder.backbone,
        "root": os.path.abspath(root),
        "dataset_id": D.dataset_id(root),
        "n_identities": len(names),
        "n_skipped_unreadable": n_unreadable,
        "n_skipped_no_face": n_no_face,
        "built_on": embedder.device,
        "built_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "elapsed_s": round(perf_counter() - t0, 1),
    }
    P.save_gallery(out_path, emb, names, rels, meta)
    return out_path, meta


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--weights", required=True, help="path to the .pt checkpoint")
    ap.add_argument("--dataset", required=True, help="dataset root folder (or .zip)")
    ap.add_argument("--n", type=int, default=0, help="identities to enroll (0 = all)")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--backbone", default=None, choices=[None, "r50", "r100", "r200"])
    ap.add_argument("--device", default=None, choices=["cpu", "gpu"],
                    help="default: gpu when available (enrollment is much faster there)")
    args = ap.parse_args()

    root = args.dataset
    if root.lower().endswith(".zip"):
        print(f"extracting {root} ...")
        root = D.extract_zip(root, C.EXTRACT_DIR)

    scan = D.scan(root, args.n)
    for w in scan["warnings"]:
        print(f"  ! {w}")
    if not scan["identities"]:
        raise SystemExit("nothing to enroll")
    print(f"{D.summary(scan)}\n  root: {scan['root']}")

    devices = available_devices()
    device = args.device or ("gpu" if "gpu" in devices else "cpu")
    if device not in devices:
        raise SystemExit(f"device '{device}' unavailable - have {devices}. "
                         "Run `python check_devices.py`.")

    print(f"loading {os.path.basename(args.weights)} on {device} ...")
    embedder = Embedder(args.weights, device, args.backbone).warmup()
    print(f"  backbone {embedder.backbone} / {embedder.label}")

    out = C.gallery_path(embedder.id, D.dataset_id(scan["root"]), len(scan["identities"]))

    def show(done, total, rate):
        eta = (total - done) / max(rate, 1e-6)
        print(f"  {done}/{total}  ({rate:.1f} img/s, eta {eta:.0f}s)", flush=True)

    path, meta = build(embedder, scan["identities"], scan["root"], out,
                       batch=args.batch, progress=show)
    if meta["n_skipped_unreadable"] or meta["n_skipped_no_face"]:
        print(f"  skipped {meta['n_skipped_unreadable']} unreadable, "
              f"{meta['n_skipped_no_face']} with no detected face")
    size = os.path.getsize(path) / 1e6
    print(f"\nwrote {path}\n  {meta['n_identities']:,} identities, "
          f"{size:.1f} MB, {meta['elapsed_s']:.0f}s on {device}")


if __name__ == "__main__":
    main()
