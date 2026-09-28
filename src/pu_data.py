"""
Stage 1: clean the supervision before touching the network.

COCO annotation auditing, repair, and image-level splitting for
FathomNet-CLEF 2026. Pure stdlib + NumPy so it is testable without torch.
"""

from __future__ import annotations

import collections
import copy
import hashlib
import json
from pathlib import Path

import numpy as np

# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #

#: The official repo ships `dataset_train.json` / `dataset_test.json`, but
#: mirrors and older write-ups use the reversed spelling. Accept both.
TRAIN_JSON_NAMES = ("dataset_train.json", "train_dataset.json")
TEST_JSON_NAMES = ("dataset_test.json", "test_dataset.json")


def find_dataset_json(search_roots, names) -> Path | None:
    """
    Locate the first matching annotation file under any of `search_roots`.

    Kaggle mounts competition data at an unpredictable depth under
    /kaggle/input, so searching beats hardcoding a path.
    """
    for root in search_roots:
        root = Path(root)
        if not root.exists():
            continue
        for name in names:
            direct = root / name
            if direct.is_file():
                return direct
        for name in names:
            hit = next(iter(sorted(root.rglob(name))), None)
            if hit is not None:
                return hit
    return None


def load_coco(path) -> dict:
    with open(path, "r", encoding="utf-8") as fh:
        coco = json.load(fh)
    for key in ("images", "categories"):
        if key not in coco:
            raise ValueError(f"{path}: missing required COCO key {key!r}")
    coco.setdefault("annotations", [])
    return coco


# --------------------------------------------------------------------------- #
# Audit
# --------------------------------------------------------------------------- #


def audit_coco(coco: dict, min_box_side: float = 2.0, tiny_area: float = 32.0) -> dict:
    """
    Report everything wrong (or merely suspicious) about a COCO file.

    This is read-only and never mutates `coco`. It is the first thing to run on
    the challenge data, because in a positive-unlabeled setting an annotation
    bug and a deliberately-missing label look identical downstream, and only one
    of the two is yours to fix.

    Checked:
      - annotations referencing a missing image_id or category_id
      - non-finite, negative-extent, or zero-area boxes
      - boxes extending outside the declared image dimensions
      - exact duplicate annotations (same image, category, box)
      - near-duplicate annotations (same image+category, IoU >= 0.95)
      - degenerately small boxes
      - images carrying no annotations at all
      - per-category instance counts (imbalance)
      - missing/invalid image width/height
    """
    from pu_ops import iou_matrix, xywh_to_xyxy  # local import keeps module standalone

    images = {im["id"]: im for im in coco["images"]}
    cat_ids = {c["id"] for c in coco["categories"]}
    anns = coco["annotations"]

    report: dict = {
        "n_images": len(images),
        "n_annotations": len(anns),
        "n_categories": len(cat_ids),
        "orphan_image_id": [],
        "unknown_category_id": [],
        "non_finite_bbox": [],
        "non_positive_bbox": [],
        "out_of_bounds_bbox": [],
        "tiny_bbox": [],
        "exact_duplicates": [],
        "near_duplicates": [],
        "bad_image_dims": [],
        "images_without_annotations": [],
    }

    for im_id, im in images.items():
        w, h = im.get("width"), im.get("height")
        if not isinstance(w, (int, float)) or not isinstance(h, (int, float)) or w <= 0 or h <= 0:
            report["bad_image_dims"].append(im_id)

    seen: set[tuple] = set()
    by_image_cat: dict[tuple, list] = collections.defaultdict(list)

    for ann in anns:
        aid = ann.get("id")
        im_id = ann.get("image_id")
        cid = ann.get("category_id")

        if im_id not in images:
            report["orphan_image_id"].append(aid)
            continue
        if cid not in cat_ids:
            report["unknown_category_id"].append(aid)

        bbox = ann.get("bbox", None)
        if bbox is None or len(bbox) != 4:
            report["non_finite_bbox"].append(aid)
            continue

        arr = np.asarray(bbox, dtype=np.float64)
        if not np.all(np.isfinite(arr)):
            report["non_finite_bbox"].append(aid)
            continue

        x, y, w, h = arr
        if w <= 0 or h <= 0:
            report["non_positive_bbox"].append(aid)
            continue
        if w < min_box_side or h < min_box_side or (w * h) < tiny_area:
            report["tiny_bbox"].append(aid)

        im = images[im_id]
        iw, ih = im.get("width") or 0, im.get("height") or 0
        if iw > 0 and ih > 0:
            # Allow 1px of float slop before calling it out of bounds.
            if x < -1.0 or y < -1.0 or (x + w) > iw + 1.0 or (y + h) > ih + 1.0:
                report["out_of_bounds_bbox"].append(aid)

        key = (im_id, cid, round(float(x), 3), round(float(y), 3),
               round(float(w), 3), round(float(h), 3))
        if key in seen:
            report["exact_duplicates"].append(aid)
        else:
            seen.add(key)
            by_image_cat[(im_id, cid)].append(ann)

    # Near-duplicates: same image and class, IoU >= 0.95.
    for (im_id, cid), group in by_image_cat.items():
        if len(group) < 2:
            continue
        boxes = xywh_to_xyxy(np.array([g["bbox"] for g in group], dtype=np.float64))
        ious = iou_matrix(boxes, boxes)
        np.fill_diagonal(ious, 0.0)
        ii, jj = np.where(ious >= 0.95)
        for i, j in zip(ii, jj):
            if i < j:
                report["near_duplicates"].append((group[i].get("id"), group[j].get("id")))

    annotated = {a.get("image_id") for a in anns}
    report["images_without_annotations"] = sorted(
        im_id for im_id in images if im_id not in annotated
    )

    counts = collections.Counter(
        a.get("category_id") for a in anns if a.get("image_id") in images
    )
    cat_names = {c["id"]: c.get("name", str(c["id"])) for c in coco["categories"]}
    report["category_counts"] = {
        cat_names.get(cid, str(cid)): n for cid, n in counts.most_common()
    }
    report["categories_with_no_instances"] = sorted(
        cat_names[cid] for cid in cat_ids if counts.get(cid, 0) == 0
    )

    # Instances per image, which is the headline PU statistic: a mean near 1.0
    # on natural underwater scenes is itself evidence of incomplete labelling.
    per_image = collections.Counter(
        a.get("image_id") for a in anns if a.get("image_id") in images
    )
    vals = np.array([per_image.get(i, 0) for i in images], dtype=np.float64)
    report["instances_per_image"] = {
        "mean": float(vals.mean()) if len(vals) else 0.0,
        "median": float(np.median(vals)) if len(vals) else 0.0,
        "max": int(vals.max()) if len(vals) else 0,
        "n_zero": int((vals == 0).sum()),
    }
    return report


def clean_coco(
    coco: dict,
    drop_tiny: bool = False,
    min_box_side: float = 2.0,
    tiny_area: float = 32.0,
    clip_to_image: bool = True,
) -> tuple[dict, dict]:
    """
    Produce a repaired deep copy of `coco` plus a summary of what changed.

    Deliberately conservative. Annotations are dropped only when they cannot
    describe a real object (orphaned, malformed, zero-area, exact duplicate);
    out-of-bounds boxes are clipped rather than discarded, since a box running
    past the frame edge is usually a real organism annotated loosely.

    `drop_tiny` is off by default: in this dataset a 20-pixel box is often a
    genuine small organism, which is exactly what the high-resolution path
    exists to catch. Turn it on only as a deliberate ablation.
    """
    out = copy.deepcopy(coco)
    images = {im["id"]: im for im in out["images"]}
    cat_ids = {c["id"] for c in out["categories"]}

    kept, summary = [], collections.Counter()
    seen: set[tuple] = set()

    for ann in out["annotations"]:
        im_id, cid = ann.get("image_id"), ann.get("category_id")

        if im_id not in images:
            summary["dropped_orphan"] += 1
            continue
        if cid not in cat_ids:
            summary["dropped_unknown_category"] += 1
            continue

        bbox = ann.get("bbox")
        if bbox is None or len(bbox) != 4:
            summary["dropped_malformed"] += 1
            continue
        arr = np.asarray(bbox, dtype=np.float64)
        if not np.all(np.isfinite(arr)):
            summary["dropped_malformed"] += 1
            continue

        x, y, w, h = (float(v) for v in arr)
        if w <= 0 or h <= 0:
            summary["dropped_non_positive"] += 1
            continue

        if clip_to_image:
            iw, ih = images[im_id].get("width") or 0, images[im_id].get("height") or 0
            if iw > 0 and ih > 0:
                x1, y1 = max(0.0, x), max(0.0, y)
                x2, y2 = min(float(iw), x + w), min(float(ih), y + h)
                if (x2 - x1) <= 0 or (y2 - y1) <= 0:
                    summary["dropped_outside_image"] += 1
                    continue
                if (x1, y1, x2, y2) != (x, y, x + w, y + h):
                    summary["clipped"] += 1
                x, y, w, h = x1, y1, x2 - x1, y2 - y1

        if drop_tiny and (w < min_box_side or h < min_box_side or w * h < tiny_area):
            summary["dropped_tiny"] += 1
            continue

        key = (im_id, cid, round(x, 3), round(y, 3), round(w, 3), round(h, 3))
        if key in seen:
            summary["dropped_duplicate"] += 1
            continue
        seen.add(key)

        ann["bbox"] = [x, y, w, h]
        ann["area"] = w * h
        ann.setdefault("iscrowd", 0)
        kept.append(ann)

    out["annotations"] = kept
    summary["kept"] = len(kept)
    summary["input"] = len(coco["annotations"])
    return out, dict(summary)


# --------------------------------------------------------------------------- #
# Splitting
# --------------------------------------------------------------------------- #


def _stable_hash(value, salt: str) -> float:
    """Deterministic hash in [0, 1), stable across processes and Python runs."""
    digest = hashlib.sha256(f"{salt}:{value}".encode("utf-8")).hexdigest()
    return int(digest[:16], 16) / float(1 << 64)


def image_level_split(
    coco: dict,
    val_fraction: float = 0.15,
    salt: str = "fathomnet-pu-v1",
    group_key: str | None = None,
) -> tuple[list, list]:
    """
    Split at the IMAGE level, never the annotation level.

    Splitting annotations puts two boxes from the same frame on both sides of the
    split and leaks. Returns (train_image_ids, val_image_ids).

    `group_key` optionally holds out whole groups instead of individual images --
    pass e.g. "source" or a deployment/dive field if the annotations carry one.
    The competition's train and test imagery come from different institutional
    sources, so a random same-source split flatters the model; grouping is the
    closer proxy for the real evaluation.
    """
    if not (0.0 <= val_fraction < 1.0):
        raise ValueError("val_fraction must be in [0, 1)")

    if group_key:
        groups: dict = collections.defaultdict(list)
        for im in coco["images"]:
            groups[im.get(group_key, "__missing__")].append(im["id"])
        train, val = [], []
        for gname, ids in sorted(groups.items(), key=lambda kv: str(kv[0])):
            (val if _stable_hash(gname, salt) < val_fraction else train).extend(ids)
        return sorted(train), sorted(val)

    train, val = [], []
    for im in coco["images"]:
        (val if _stable_hash(im["id"], salt) < val_fraction else train).append(im["id"])
    return sorted(train), sorted(val)


def subset_coco(coco: dict, image_ids) -> dict:
    """Restrict a COCO dict to `image_ids`, keeping categories intact."""
    wanted = set(image_ids)
    out = {
        "images": [im for im in coco["images"] if im["id"] in wanted],
        "annotations": [a for a in coco["annotations"] if a.get("image_id") in wanted],
        "categories": copy.deepcopy(coco["categories"]),
    }
    for key in ("info", "licenses"):
        if key in coco:
            out[key] = copy.deepcopy(coco[key])
    return out


def merge_pseudo_labels(
    coco: dict,
    pseudo: dict[int, dict],
    mark_key: str = "is_pseudo",
) -> tuple[dict, int]:
    """
    Fold harvested pseudo-labels into a COCO dict as new annotations.

    `pseudo` maps image_id -> {"boxes": (N,4) xyxy, "labels": (N,), "scores": (N,)}.
    Each added annotation is tagged with `mark_key=True` and keeps its originating
    score, so you can down-weight pseudo-labels in the loss, or strip them again
    for an ablation, without re-running the harvest.
    """
    from pu_ops import xyxy_to_xywh

    out = copy.deepcopy(coco)
    next_id = max((a.get("id", 0) for a in out["annotations"]), default=0) + 1
    valid_images = {im["id"] for im in out["images"]}
    added = 0

    for im_id, payload in pseudo.items():
        if im_id not in valid_images:
            continue
        boxes = np.asarray(payload["boxes"], dtype=np.float64).reshape(-1, 4)
        labels = np.asarray(payload["labels"]).reshape(-1)
        scores = np.asarray(payload.get("scores", np.ones(len(boxes)))).reshape(-1)
        if len(boxes) == 0:
            continue
        for bb, lab, sc in zip(xyxy_to_xywh(boxes), labels, scores):
            w, h = float(bb[2]), float(bb[3])
            if w <= 0 or h <= 0:
                continue
            out["annotations"].append(
                {
                    "id": next_id,
                    "image_id": im_id,
                    "category_id": int(lab),
                    "bbox": [float(bb[0]), float(bb[1]), w, h],
                    "area": w * h,
                    "iscrowd": 0,
                    mark_key: True,
                    "score": float(sc),
                }
            )
            next_id += 1
            added += 1
    return out, added


# --------------------------------------------------------------------------- #
# Submission
# --------------------------------------------------------------------------- #

#: Exact column order required by the FathomNet-CLEF 2026 submission format.
SUBMISSION_COLUMNS = [
    "annotation_id",
    "image_id",
    "category_id",
    "bbox_x",
    "bbox_y",
    "bbox_width",
    "bbox_height",
    "score",
]


def build_submission(predictions: dict[int, dict], start_id: int = 1):
    """
    Turn per-image predictions into the competition's submission rows.

    `predictions` maps image_id -> {"boxes": (N,4) xyxy in ORIGINAL frame
    coordinates, "scores": (N,), "labels": (N,)}.

    Returns a list of dicts in SUBMISSION_COLUMNS order. Boxes are emitted as
    COCO xywh because that is what the schema's bbox_x/bbox_y/bbox_width/
    bbox_height fields mean -- emitting xyxy here silently destroys the score.
    """
    from pu_ops import xyxy_to_xywh

    rows = []
    aid = start_id
    for im_id in sorted(predictions):
        p = predictions[im_id]
        boxes = np.asarray(p["boxes"], dtype=np.float64).reshape(-1, 4)
        scores = np.asarray(p["scores"], dtype=np.float64).reshape(-1)
        labels = np.asarray(p["labels"]).reshape(-1)
        if len(boxes) == 0:
            continue
        for bb, sc, lab in zip(xyxy_to_xywh(boxes), scores, labels):
            rows.append(
                {
                    "annotation_id": aid,
                    "image_id": int(im_id),
                    "category_id": int(lab),
                    "bbox_x": round(float(bb[0]), 2),
                    "bbox_y": round(float(bb[1]), 2),
                    "bbox_width": round(float(bb[2]), 2),
                    "bbox_height": round(float(bb[3]), 2),
                    "score": round(float(np.clip(sc, 0.0, 1.0)), 6),
                }
            )
            aid += 1
    return rows
