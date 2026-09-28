#!/usr/bin/env python3
"""Command-line entry point for the FathomNet-CLEF 2026 PU detection pipeline.

Stages run independently and checkpoint after every epoch, so a long run can be
split across sessions.
"""

import argparse
import json
import sys
from dataclasses import fields
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "src"))

from pu_config import PipelineConfig          # noqa: E402
from pu_ops import *                          # noqa: E402,F401,F403
from pu_data import *                         # noqa: E402,F401,F403
from pu_pipeline import *                     # noqa: E402,F401,F403

STAGES = ("verify", "audit", "train_base", "harvest", "train_pu", "infer")
GPU_STAGES = {"train_base", "harvest", "train_pu", "infer"}

BASE_CKPT = "rtdetr_base.pt"
PU_CKPT = "rtdetr_pu.pt"
COCO_PU_JSON = "coco_with_pseudo.json"
PSEUDO_JSON = "pseudo_labels.json"


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="FathomNet-CLEF 2026 positive-unlabeled detection pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    d = PipelineConfig()

    p.add_argument("--stage", choices=STAGES, default=d.stage)
    p.add_argument("--input-root", action="append", default=None,
                   help="where to look for the annotation JSONs (repeatable)")
    p.add_argument("--work-dir", default=d.work_dir)
    p.add_argument("--max-images", type=int, default=d.max_images,
                   help="cap images used; 0 = full dataset")
    p.add_argument("--epochs", type=int, default=d.epochs)
    p.add_argument("--batch-size", type=int, default=d.batch_size)
    p.add_argument("--img-size", type=int, default=d.img_size)
    p.add_argument("--lr", type=float, default=d.lr)
    p.add_argument("--num-workers", type=int, default=d.num_workers)
    p.add_argument("--download-workers", type=int, default=d.download_workers)
    p.add_argument("--val-fraction", type=float, default=d.val_fraction)

    g = p.add_argument_group("PU loss (stage 4)")
    g.add_argument("--pu-w-min", type=float, default=d.pu_w_min,
                   help="background weight for confident unmatched queries; must stay > 0")
    g.add_argument("--pu-w-max", type=float, default=d.pu_w_max)
    g.add_argument("--pu-tau", type=float, default=d.pu_tau)
    g.add_argument("--pu-temperature", type=float, default=d.pu_temperature)
    g.add_argument("--pu-disabled", action="store_true",
                   help="ablation: ordinary uniform background loss")

    h = p.add_argument_group("pseudo-labels (stage 3)")
    h.add_argument("--pseudo-score-threshold", type=float, default=d.pseudo_score_threshold)
    h.add_argument("--pseudo-min-views", type=int, default=d.pseudo_min_views)
    h.add_argument("--pseudo-max-per-image", type=int, default=d.pseudo_max_per_image)

    p.add_argument("--dry-run", action="store_true",
                   help="print the resolved config and exit")
    return p.parse_args(argv)


def config_from_args(a) -> PipelineConfig:
    cfg = PipelineConfig()
    cfg.stage = a.stage
    cfg.work_dir = a.work_dir
    cfg.max_images = a.max_images
    cfg.epochs = a.epochs
    cfg.batch_size = a.batch_size
    cfg.img_size = a.img_size
    cfg.lr = a.lr
    cfg.num_workers = a.num_workers
    cfg.download_workers = a.download_workers
    cfg.val_fraction = a.val_fraction
    cfg.pu_w_min = a.pu_w_min
    cfg.pu_w_max = a.pu_w_max
    cfg.pu_tau = a.pu_tau
    cfg.pu_temperature = a.pu_temperature
    cfg.pu_disabled = a.pu_disabled
    cfg.pseudo_score_threshold = a.pseudo_score_threshold
    cfg.pseudo_min_views = a.pseudo_min_views
    cfg.pseudo_max_per_image = a.pseudo_max_per_image
    if a.input_root:
        cfg.input_roots = tuple(a.input_root)
    Path(cfg.work_dir).mkdir(parents=True, exist_ok=True)
    return cfg


def banner(text):
    print("\n" + "=" * 66)
    print(text)
    print("=" * 66)


# --------------------------------------------------------------------------- #
# Stages
# --------------------------------------------------------------------------- #


def stage_verify(cfg, env):
    """Run the algorithm checks. No GPU or data required."""
    banner("VERIFY  -- algorithm checks, no GPU or data needed")
    sys.path.insert(0, str(HERE / "tests"))
    import subprocess

    r = subprocess.run(
        [sys.executable, str(HERE / "tests" / "test_pipeline.py")],
        capture_output=True, text=True,
    )
    tail = r.stdout.strip().splitlines()[-6:]
    print("\n".join(tail))
    if r.returncode != 0:
        print(r.stdout[-3000:])
        print(r.stderr[-2000:])
        raise SystemExit("verification FAILED -- do not proceed to GPU stages")

    if not env.get("torch", "MISSING").startswith("MISSING"):
        from pu_criterion import PULossConfig, PUDetectionCriterion  # noqa: F401
        print("\ntorch present; PU criterion imports cleanly")
        _criterion_smoke_test()
    print("\nVerification passed. Next: --stage audit")


def _criterion_smoke_test():
    """Minimal forward/backward through the PU criterion on random tensors."""
    import torch
    from pu_criterion import PULossConfig, PUDetectionCriterion

    torch.manual_seed(0)
    B, Q, C, N = 2, 16, 5, 3
    logits = torch.randn(B, Q, C, requires_grad=True)
    boxes = (torch.rand(B, Q, 4) * 0.5 + 0.25).requires_grad_(True)
    outputs = {"logits": logits, "pred_boxes": boxes}
    targets = [{"labels": torch.randint(0, C, (N,)),
                "boxes": torch.rand(N, 4) * 0.4 + 0.3} for _ in range(B)]

    out = PUDetectionCriterion(PULossConfig(num_classes=C))(outputs, targets)
    assert torch.isfinite(out["loss"]), "non-finite loss"
    out["loss"].backward()
    assert logits.grad is not None and torch.isfinite(logits.grad).all()

    on = PUDetectionCriterion(PULossConfig(num_classes=C, pu_w_min=0.1))(outputs, targets)
    off = PUDetectionCriterion(PULossConfig(num_classes=C, pu_disabled=True))(outputs, targets)
    assert on["loss"].item() < off["loss"].item(), "PU weighting had no effect"
    assert abs(on["loss_bbox"].item() - off["loss_bbox"].item()) < 1e-6, \
        "PU weighting leaked into the box loss"
    print(f"  criterion smoke test OK "
          f"(loss {out['loss'].item():.3f}, PU on {on['loss'].item():.3f} "
          f"vs off {off['loss'].item():.3f})")


def load_data(cfg):
    """Locate, audit, clean and split the training annotations."""
    train_json = find_dataset_json(cfg.input_roots, TRAIN_JSON_NAMES)
    test_json = find_dataset_json(cfg.input_roots, TEST_JSON_NAMES)
    print(f"train json: {train_json}")
    print(f"test  json: {test_json}")
    if train_json is None:
        raise SystemExit(
            "training annotations not found. On Kaggle, attach the competition "
            f"data; searched: {list(cfg.input_roots)}"
        )

    raw = load_coco(train_json)
    print(f"images {len(raw['images'])} | annotations {len(raw['annotations'])} "
          f"| categories {len(raw['categories'])}")

    audit = audit_coco(raw)
    clean, summary = clean_coco(raw, drop_tiny=cfg.drop_tiny_boxes, clip_to_image=True)
    gk = cfg.split_group_key or None
    if gk and raw["images"] and gk not in raw["images"][0]:
        print(f"(no {gk!r} field on image records - per-image split instead)")
        gk = None
    tr, va = image_level_split(clean, val_fraction=cfg.val_fraction,
                              salt=cfg.split_salt, group_key=gk)
    assert not (set(tr) & set(va)), "split leaked"
    return raw, clean, audit, summary, tr, va, test_json


def stage_audit(cfg):
    banner("STAGE 1  -- audit, clean, split")
    raw, clean, audit, summary, tr, va, _ = load_data(cfg)

    for k in ("orphan_image_id", "unknown_category_id", "non_finite_bbox",
              "non_positive_bbox", "out_of_bounds_bbox", "tiny_bbox",
              "exact_duplicates", "near_duplicates", "bad_image_dims"):
        print(f"  {k:24s} {len(audit[k])}")
    print(f"  {'images w/o annotations':24s} {len(audit['images_without_annotations'])}")
    print(f"  {'empty categories':24s} {audit['categories_with_no_instances']}")

    ipi = audit["instances_per_image"]
    print(f"\ninstances per image: {ipi}")
    print(f">>> mean {ipi['mean']:.2f} labelled instances/image" + (
        "  -- consistent with heavily incomplete labelling."
        if ipi["mean"] < 2.0 else
        "  -- higher than expected; check the PU assumption holds."))

    print("\ncleaning:")
    for k in sorted(summary):
        print(f"  {k:26s} {summary[k]}")

    counts = list(audit["category_counts"].values())
    if counts:
        print(f"\ncategories: {len(counts)}  "
              f"imbalance (max/min): {max(counts) / max(min(counts), 1):.1f}x")
        print("top 10:")
        for name, n in list(audit["category_counts"].items())[:10]:
            print(f"  {name:28s} {n}")

    print(f"\nsplit: {len(tr)} train / {len(va)} val")

    out = Path(cfg.work_dir) / "stage1_audit.json"
    out.write_text(json.dumps({
        "audit": {k: (v if isinstance(v, (int, float, str, dict)) else len(v))
                  for k, v in audit.items()},
        "clean_summary": summary, "n_train": len(tr), "n_val": len(va),
    }, indent=2, default=str))
    print(f"\nreport -> {out}")
    print("Next: --stage train_base (needs GPU + Internet)")


def _prepare(ids, coco, cfg):
    """Resolve imagery for `ids`; return those actually available on disk."""
    sub = subset_coco(coco, ids)
    attached = find_attached_image_dir(
        cfg.input_roots, [im.get("file_name", "") for im in sub["images"][:20]])
    cache = Path(attached) if attached else Path(cfg.work_dir) / "images"
    if attached:
        print(f"using attached image directory: {cache}")
    ok, failed = ensure_images(sub, cache, workers=cfg.download_workers)
    resolved = {im["id"]: im.get("_local_path") for im in sub["images"]}
    for im in coco["images"]:
        if im["id"] in resolved:
            im["_local_path"] = resolved[im["id"]]
    if failed:
        print(f">>> {len(failed)} frames unavailable, skipped")
    return [i for i in ids if i in ok]


def criterion_config(cfg, pu_on):
    from pu_criterion import PULossConfig
    return PULossConfig(
        num_classes=cfg.num_classes,
        pu_w_min=cfg.pu_w_min, pu_w_max=cfg.pu_w_max,
        pu_tau=cfg.pu_tau, pu_temperature=cfg.pu_temperature,
        pu_disabled=not pu_on, pseudo_label_weight=cfg.pseudo_label_weight,
    )


def stage_gpu(cfg, stage):
    import torch
    from pu_criterion import PUDetectionCriterion  # noqa: F401

    if not torch.cuda.is_available():
        raise SystemExit(
            f"stage {stage!r} needs a GPU. Kaggle: Settings -> Accelerator -> GPU")

    raw, clean, audit, summary, tr, va, test_json = load_data(cfg)
    c2i, i2c, i2n = build_category_maps(clean)
    if len(c2i) != cfg.num_classes:
        print(f">>> data has {len(c2i)} categories, config says {cfg.num_classes}; "
              "using the data's count")
        cfg.num_classes = len(c2i)

    use_tr = tr if not cfg.max_images else tr[: cfg.max_images]
    use_va = va if not cfg.max_images else va[: max(cfg.max_images // 4, 1)]
    print(f"\nusing {len(use_tr)} train / {len(use_va)} val images"
          + ("" if not cfg.max_images else f" (capped at {cfg.max_images})"))

    work = Path(cfg.work_dir)

    if stage == "train_base":
        banner("STAGE 2  -- baseline detector (PU loss OFF)")
        use_tr = _prepare(use_tr, clean, cfg)
        use_va = _prepare(use_va, clean, cfg)
        _, _, ckpt = train_one_run(clean, use_tr, use_va, cfg,
                                   criterion_config(cfg, pu_on=False), BASE_CKPT)
        print(f"\ncheckpoint: {ckpt}\nNext: --stage harvest")

    elif stage == "harvest":
        banner("STAGE 3  -- conservative pseudo-label recovery")
        ckpt = work / BASE_CKPT
        if not ckpt.is_file():
            raise SystemExit(f"{ckpt} missing -- run --stage train_base first")
        use_tr = _prepare(use_tr, clean, cfg)
        model = build_model(cfg.num_classes, cfg.model_name, torch.device("cuda"))
        model.load_state_dict(load_checkpoint(ckpt, torch.device("cuda"))["model"])
        harvest, totals = harvest_over_dataset(model, clean, use_tr, cfg, i2c)
        merged, added = merge_pseudo_labels(clean, harvest)
        before = audit_coco(clean)["instances_per_image"]["mean"]
        after = audit_coco(merged)["instances_per_image"]["mean"]
        print(f"\nadded {added} pseudo-labels")
        print(f"instances/image {before:.2f} -> {after:.2f}")
        (work / COCO_PU_JSON).write_text(json.dumps(merged))
        (work / PSEUDO_JSON).write_text(json.dumps(
            {str(k): {"boxes": v["boxes"].tolist(), "scores": v["scores"].tolist(),
                      "labels": v["labels"].tolist()} for k, v in harvest.items()}))
        print(f"-> {work / COCO_PU_JSON}\nNext: --stage train_pu")

    elif stage == "train_pu":
        banner("STAGES 3+4  -- retrain with pseudo-labels + PU-aware background loss")
        pu_json = work / COCO_PU_JSON
        if pu_json.is_file():
            data = json.loads(pu_json.read_text())
            n_p = sum(1 for a in data["annotations"] if a.get("is_pseudo"))
            print(f"training on pseudo-augmented data ({n_p} pseudo-labels)")
        else:
            data = clean
            print(">>> no pseudo-labels found; training on cleaned data only")
        use_tr = _prepare(use_tr, data, cfg)
        use_va = _prepare(use_va, data, cfg)
        cc = criterion_config(cfg, pu_on=not cfg.pu_disabled)
        print(f"PU: w_min={cc.pu_w_min} w_max={cc.pu_w_max} tau={cc.pu_tau} "
              f"T={cc.pu_temperature} disabled={cc.pu_disabled}")
        _, _, ckpt = train_one_run(data, use_tr, use_va, cfg, cc, PU_CKPT,
                                   init_from=work / BASE_CKPT)
        print(f"\ncheckpoint: {ckpt}\nNext: --stage infer")

    elif stage == "infer":
        banner("STAGE 5  -- two-scale inference + Soft-NMS")
        ckpt = work / PU_CKPT
        if not ckpt.is_file():
            ckpt = work / BASE_CKPT
            print(f">>> PU checkpoint missing, falling back to {ckpt.name}")
        if not ckpt.is_file():
            raise SystemExit("no checkpoint -- run a training stage first")
        device = torch.device("cuda")
        model = build_model(cfg.num_classes, cfg.model_name, device)
        model.load_state_dict(load_checkpoint(ckpt, device)["model"])
        print(f"scales {cfg.infer_scales}  soft-nms {cfg.softnms_method} "
              f"sigma={cfg.softnms_sigma}")

        use_va = _prepare(use_va, clean, cfg)
        val_preds = predict_two_scale(model, clean, use_va, cfg, i2c)
        print(f"\nvalidation predictions on {len(val_preds)} images")
        evaluate_map(subset_coco(clean, use_va), val_preds, index_to_name=i2n)

        if test_json is not None:
            import csv
            raw_test = load_coco(test_json)
            ids = [im["id"] for im in raw_test["images"]]
            if cfg.max_images:
                ids = ids[: cfg.max_images]
            ids = _prepare(ids, raw_test, cfg)
            preds = predict_two_scale(model, raw_test, ids, cfg, i2c)
            rows = build_submission(preds)
            sub = work / "submission.csv"
            with open(sub, "w", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=SUBMISSION_COLUMNS)
                w.writeheader()
                w.writerows(rows)
            print(f"\nsubmission: {len(rows)} rows -> {sub}")
            print(f"  images with detections: {len(preds)}/{len(ids)}")
        else:
            print("\nno test annotations found - skipping submission")


# --------------------------------------------------------------------------- #


def main(argv=None):
    a = parse_args(argv)
    cfg = config_from_args(a)

    banner(f"FathomNet-CLEF 2026 PU pipeline  |  stage = {cfg.stage}")
    env = probe_environment()
    for k, v in env.items():
        print(f"  {k:14s}: {v}")

    if a.dry_run:
        print("\nresolved config:")
        for f in fields(cfg):
            print(f"  {f.name:26s} {getattr(cfg, f.name)}")
        return 0

    if cfg.stage in GPU_STAGES:
        if not env.get("cuda"):
            print(f"\n>>> stage {cfg.stage!r} needs a GPU. "
                  "Kaggle: Settings -> Accelerator -> GPU")
            return 2
        if not env.get("internet"):
            print("\n>>> no internet: RT-DETR weights and imagery cannot be "
                  "fetched. Kaggle: Settings -> Internet -> On")

    if cfg.stage == "verify":
        stage_verify(cfg, env)
    elif cfg.stage == "audit":
        stage_audit(cfg)
    else:
        stage_gpu(cfg, cfg.stage)
    return 0


if __name__ == "__main__":
    sys.exit(main())
