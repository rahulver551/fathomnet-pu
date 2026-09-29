"""
Verification suite for the FathomNet-CLEF 2026 PU pipeline ops.

Expected values are hand-computed and hardcoded wherever possible (e.g. exp(-2)
for a fully-overlapping Gaussian Soft-NMS decay) so the tests check the algorithm
rather than merely re-deriving it from the implementation.
"""

import json as _json
import math
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pu_data import (  # noqa: E402
    SUBMISSION_COLUMNS,
    audit_coco,
    build_submission,
    clean_coco,
    image_level_split,
    merge_pseudo_labels,
    subset_coco,
)
from pu_ops import (  # noqa: E402
    box_areas,
    clip_boxes,
    consistency_filter,
    dedup_against_gt,
    harvest_pseudo_labels,
    iou_matrix,
    merge_multiscale,
    pu_background_weight,
    scale_boxes,
    soft_nms,
    soft_nms_per_class,
    xywh_to_xyxy,
    xyxy_to_xywh,
)

PASS, FAIL = [], []


def check(name, cond, detail=""):
    if cond:
        PASS.append(name)
        print(f"  PASS  {name}")
    else:
        FAIL.append((name, detail))
        print(f"  FAIL  {name}  {detail}")


def close(a, b, tol=1e-9):
    return abs(float(a) - float(b)) <= tol


# ======================================================================= #
print("\n[1] box conversions and IoU")
# ======================================================================= #

check(
    "xywh->xyxy",
    np.allclose(xywh_to_xyxy([[10, 20, 30, 40]]), [[10, 20, 40, 60]]),
)
check(
    "xyxy->xywh",
    np.allclose(xyxy_to_xywh([[10, 20, 40, 60]]), [[10, 20, 30, 40]]),
)
check(
    "conversion roundtrip",
    np.allclose(xyxy_to_xywh(xywh_to_xyxy([[1.5, 2.5, 3.5, 4.5]])), [[1.5, 2.5, 3.5, 4.5]]),
)
check("areas", np.allclose(box_areas([[0, 0, 10, 10], [0, 0, 3, 4]]), [100.0, 12.0]))

# A=[0,0,10,10] B=[5,5,15,15]: inter 5*5=25, union 100+100-25=175 -> 25/175
iou = iou_matrix([[0, 0, 10, 10]], [[5, 5, 15, 15]])[0, 0]
check("IoU quarter-overlap = 25/175", close(iou, 25.0 / 175.0), f"got {iou}")

# Half overlap: inter 5*10=50, union 150 -> 1/3
iou = iou_matrix([[0, 0, 10, 10]], [[5, 0, 15, 10]])[0, 0]
check("IoU half-overlap = 1/3", close(iou, 1.0 / 3.0), f"got {iou}")

check("IoU identical = 1", close(iou_matrix([[0, 0, 5, 5]], [[0, 0, 5, 5]])[0, 0], 1.0))
check("IoU disjoint = 0", close(iou_matrix([[0, 0, 5, 5]], [[100, 100, 105, 105]])[0, 0], 0.0))
check(
    "IoU contained: 25/100",
    close(iou_matrix([[0, 0, 10, 10]], [[0, 0, 5, 5]])[0, 0], 0.25),
)
check("IoU empty input shape", iou_matrix(np.zeros((0, 4)), [[0, 0, 1, 1]]).shape == (0, 1))
check(
    "IoU zero-area box is 0 not NaN",
    close(iou_matrix([[5, 5, 5, 5]], [[0, 0, 10, 10]])[0, 0], 0.0),
)

# ======================================================================= #
print("\n[2] Soft-NMS")
# ======================================================================= #

# Two identical boxes, Gaussian, sigma=0.5: decay = exp(-1^2/0.5) = exp(-2).
b, s, i = soft_nms(
    [[0, 0, 10, 10], [0, 0, 10, 10]], [1.0, 1.0], method="gaussian", sigma=0.5
)
check("gaussian keeps both identical boxes", len(s) == 2, f"got {len(s)}")
check("gaussian anchor score untouched", close(s[0], 1.0), f"got {s[0]}")
check(
    "gaussian identical decay == exp(-2)",
    close(s[1], math.exp(-2.0)),
    f"got {s[1]} want {math.exp(-2.0)}",
)

# Half overlap IoU=1/3: decay = exp(-(1/3)^2/0.5) = exp(-2/9)
b, s, _ = soft_nms(
    [[0, 0, 10, 10], [5, 0, 15, 10]], [0.9, 0.8], method="gaussian", sigma=0.5
)
want = 0.8 * math.exp(-((1.0 / 3.0) ** 2) / 0.5)
check("gaussian half-overlap decay", close(s[1], want), f"got {s[1]} want {want}")

# Linear, identical boxes: decay = 1-IoU = 0 -> falls below score_threshold.
b, s, _ = soft_nms(
    [[0, 0, 10, 10], [0, 0, 10, 10]], [1.0, 1.0], method="linear", iou_threshold=0.3
)
check("linear drops exact duplicate", len(s) == 1, f"got {len(s)}")

# Linear below threshold leaves score alone: IoU 25/175 = 0.1428 < 0.3
b, s, _ = soft_nms(
    [[0, 0, 10, 10], [5, 5, 15, 15]], [0.9, 0.8], method="linear", iou_threshold=0.3
)
check("linear ignores sub-threshold overlap", close(s[1], 0.8), f"got {s[1]}")

# Hard NMS removes the overlapper entirely.
b, s, _ = soft_nms(
    [[0, 0, 10, 10], [1, 1, 11, 11]], [0.9, 0.8], method="hard", iou_threshold=0.3
)
check("hard NMS deletes overlapper", len(s) == 1, f"got {len(s)}")

# The headline property: Soft-NMS retains a neighbour that hard NMS destroys.
dense = [[0, 0, 10, 10], [2, 2, 12, 12]]
_, s_soft, _ = soft_nms(dense, [0.9, 0.85], method="gaussian", sigma=0.5)
_, s_hard, _ = soft_nms(dense, [0.9, 0.85], method="hard", iou_threshold=0.3)
check(
    "soft retains dense neighbour that hard removes",
    len(s_soft) == 2 and len(s_hard) == 1,
    f"soft={len(s_soft)} hard={len(s_hard)}",
)

# Disjoint boxes must never be touched.
_, s, _ = soft_nms([[0, 0, 5, 5], [50, 50, 55, 55]], [0.7, 0.6], method="gaussian")
check("disjoint boxes unchanged", len(s) == 2 and close(s[0], 0.7) and close(s[1], 0.6))

# Output ordering and index bookkeeping.
_, s, idx = soft_nms(
    [[0, 0, 5, 5], [50, 50, 55, 55], [100, 100, 105, 105]], [0.3, 0.9, 0.6]
)
check("returns descending score order", list(s) == sorted(s, reverse=True), f"{s}")
check("indices map back to input", list(idx) == [1, 2, 0], f"got {list(idx)}")

check("empty input safe", soft_nms(np.zeros((0, 4)), [])[0].shape == (0, 4))
check("single box preserved", close(soft_nms([[0, 0, 1, 1]], [0.5])[1][0], 0.5))

try:
    soft_nms([[0, 0, 1, 1]], [0.5], method="bogus")
    check("rejects unknown method", False, "no raise")
except ValueError:
    check("rejects unknown method", True)

try:
    soft_nms([[0, 0, 1, 1], [0, 0, 2, 2]], [0.5])
    check("rejects length mismatch", False, "no raise")
except ValueError:
    check("rejects length mismatch", True)

# Per-class: overlapping boxes of DIFFERENT classes must both survive fully.
b, s, l = soft_nms_per_class(
    [[0, 0, 10, 10], [0, 0, 10, 10]], [0.9, 0.85], [1, 2], method="gaussian"
)
check(
    "per-class does not suppress across classes",
    len(s) == 2 and close(s[0], 0.9) and close(s[1], 0.85),
    f"got scores {s}",
)
# Same class, identical -> decayed by exp(-2)
b, s, l = soft_nms_per_class(
    [[0, 0, 10, 10], [0, 0, 10, 10]], [0.9, 0.9], [1, 1], method="gaussian", sigma=0.5
)
check(
    "per-class suppresses within class",
    len(s) == 2 and close(s[1], 0.9 * math.exp(-2.0)),
    f"got {s}",
)

# ======================================================================= #
print("\n[3] coordinate mapping and multi-scale merge")
# ======================================================================= #

# 640x360 -> 1920x1080 is exactly 3x in both axes.
sb = scale_boxes([[10, 20, 30, 40]], (640, 360), (1920, 1080))
check("scale_boxes 3x", np.allclose(sb, [[30, 60, 90, 120]]), f"got {sb}")

rt = scale_boxes(scale_boxes([[10, 20, 30, 40]], (640, 360), (1920, 1080)), (1920, 1080), (640, 360))
check("scale_boxes roundtrip", np.allclose(rt, [[10, 20, 30, 40]]), f"got {rt}")

# Anisotropic scaling must scale x and y independently.
sb = scale_boxes([[10, 10, 20, 20]], (100, 200), (300, 400))
check("anisotropic scaling", np.allclose(sb, [[30, 20, 60, 40]]), f"got {sb}")

cb = clip_boxes([[-5, -5, 50, 50]], (40, 30))
check("clip to frame", np.allclose(cb, [[0, 0, 40, 30]]), f"got {cb}")

# Two passes at different resolutions describing the SAME object must collapse
# to one detection once mapped into the original frame.
merged = merge_multiscale(
    [
        {"boxes": [[10, 10, 20, 20]], "scores": [0.9], "labels": [1], "size": (100, 100)},
        {"boxes": [[20, 20, 40, 40]], "scores": [0.8], "labels": [1], "size": (200, 200)},
    ],
    original_size=(100, 100),
    method="gaussian",
    sigma=0.5,
)
check(
    "two passes agreeing map to same box",
    len(merged["scores"]) == 2 and np.allclose(merged["boxes"][0], [10, 10, 20, 20]),
    f"got {merged['boxes']}",
)
check(
    "agreeing duplicate is decayed, not deleted",
    close(merged["scores"][1], 0.8 * math.exp(-2.0)),
    f"got {merged['scores']}",
)

# A high-res-only small detection must survive the merge.
merged = merge_multiscale(
    [
        {"boxes": [[0, 0, 50, 50]], "scores": [0.9], "labels": [1], "size": (100, 100)},
        {"boxes": [[180, 180, 190, 190]], "scores": [0.7], "labels": [2], "size": (200, 200)},
    ],
    original_size=(100, 100),
)
check("small high-res-only detection survives", len(merged["scores"]) == 2)
check(
    "high-res small box mapped correctly",
    any(np.allclose(bb, [90, 90, 95, 95]) for bb in merged["boxes"]),
    f"got {merged['boxes']}",
)

# Pass weighting.
merged = merge_multiscale(
    [{"boxes": [[0, 0, 10, 10]], "scores": [0.8], "labels": [1], "size": (100, 100), "weight": 0.5}],
    original_size=(100, 100),
)
check("pass weight applied", close(merged["scores"][0], 0.4), f"got {merged['scores']}")

merged = merge_multiscale([], original_size=(100, 100))
check("empty merge safe", merged["boxes"].shape == (0, 4))

merged = merge_multiscale(
    [{"boxes": [[200, 200, 300, 300]], "scores": [0.9], "labels": [1], "size": (100, 100)}],
    original_size=(100, 100),
)
check("fully out-of-frame box dropped after clipping", len(merged["scores"]) == 0)

# ======================================================================= #
print("\n[4] PU-aware background weight")
# ======================================================================= #

w = pu_background_weight(np.array([0.5]), w_min=0.25, w_max=1.0, tau=0.5, temperature=0.1)
check("w at tau is midpoint 0.625", close(w[0], 0.625), f"got {w[0]}")

w = pu_background_weight(
    np.array([0.0, 0.25, 0.5, 0.75, 1.0]), w_min=0.25, w_max=1.0, tau=0.5, temperature=0.1
)
check("monotonically decreasing in objectness", bool(np.all(np.diff(w) < 0)), f"got {w}")
check("bounded within [w_min, w_max]", bool(np.all(w >= 0.25) and np.all(w <= 1.0)), f"got {w}")
check("low objectness -> near full penalty", w[0] > 0.99, f"got {w[0]}")
check("high objectness -> near w_min", w[-1] < 0.26, f"got {w[-1]}")
check(
    "never reaches zero (false positives would explode)",
    bool(np.all(w > 0.0)),
    f"got min {w.min()}",
)

# w_min == w_max degenerates to ordinary uniform background supervision.
w = pu_background_weight(np.array([0.0, 0.5, 1.0]), w_min=1.0, w_max=1.0)
check("w_min==w_max reduces to standard loss", bool(np.allclose(w, 1.0)), f"got {w}")

# Temperature controls gate sharpness.
sharp = pu_background_weight(np.array([0.6]), tau=0.5, temperature=0.01)[0]
soft = pu_background_weight(np.array([0.6]), tau=0.5, temperature=1.0)[0]
check("lower temperature gates harder", sharp < soft, f"sharp={sharp} soft={soft}")

try:
    pu_background_weight(np.array([0.5]), w_min=0.9, w_max=0.1)
    check("rejects w_min > w_max", False, "no raise")
except ValueError:
    check("rejects w_min > w_max", True)

try:
    pu_background_weight(np.array([0.5]), temperature=0.0)
    check("rejects zero temperature", False, "no raise")
except ValueError:
    check("rejects zero temperature", True)

# ======================================================================= #
print("\n[5] pseudo-label recovery")
# ======================================================================= #

gt_b = np.array([[0, 0, 10, 10]], dtype=float)
gt_l = np.array([1])

# Same class, identical location -> duplicate, must be rejected.
mask = dedup_against_gt([[0, 0, 10, 10]], [1], gt_b, gt_l, iou_threshold=0.5)
check("dedup rejects same-class duplicate", not bool(mask[0]))

# Different class, identical location -> kept (shrimp on a sponge).
mask = dedup_against_gt([[0, 0, 10, 10]], [2], gt_b, gt_l, iou_threshold=0.5)
check("dedup keeps different class at same place", bool(mask[0]))

# Same class, far away -> kept (this is the missing positive we want).
mask = dedup_against_gt([[90, 90, 100, 100]], [1], gt_b, gt_l, iou_threshold=0.5)
check("dedup keeps distant same-class box", bool(mask[0]))

mask = dedup_against_gt([[0, 0, 10, 10]], [1], np.zeros((0, 4)), np.zeros(0), 0.5)
check("dedup with no GT keeps everything", bool(mask[0]))

mask = dedup_against_gt([[0, 0, 10, 10]], [1], gt_b, gt_l, class_aware=False)
check("class_aware=False suppresses across classes", not bool(mask[0]))

# Consistency: a box present in 2 of 3 views at the same place/class.
views = [
    {"boxes": [[0, 0, 10, 10], [50, 50, 60, 60]], "labels": [1, 1]},
    {"boxes": [[0, 0, 10, 10]], "labels": [1]},          # corroborates only the first
    {"boxes": [[0, 1, 10, 11]], "labels": [1]},          # high IoU with the first
]
mask = consistency_filter(views, iou_threshold=0.6, min_views=2)
check("consistency keeps corroborated box", bool(mask[0]))
check("consistency drops uncorroborated box", not bool(mask[1]))

# Class disagreement is not corroboration.
views = [{"boxes": [[0, 0, 10, 10]], "labels": [1]}, {"boxes": [[0, 0, 10, 10]], "labels": [2]}]
check(
    "consistency requires class agreement",
    not bool(consistency_filter(views, min_views=2)[0]),
)

# min_views=1 means the reference alone suffices.
views = [{"boxes": [[0, 0, 10, 10]], "labels": [1]}, {"boxes": [], "labels": []}]
check("min_views=1 accepts reference alone", bool(consistency_filter(views, min_views=1)[0]))

check("consistency on empty reference", consistency_filter([{"boxes": [], "labels": []}]).shape == (0,))

# Full funnel. Candidates:
#   A [0,0,10,10] s=0.9 c=1  -> duplicates GT, must be removed at dedup
#   B [50,50,60,60] s=0.8 c=1 -> novel and corroborated -> KEEP
#   C [80,80,90,90] s=0.3 c=1 -> below confidence -> removed
#   D [20,20,30,30] s=0.7 c=1 -> novel but uncorroborated -> removed
res = harvest_pseudo_labels(
    [
        {
            "boxes": [[0, 0, 10, 10], [50, 50, 60, 60], [80, 80, 90, 90], [20, 20, 30, 30]],
            "scores": [0.9, 0.8, 0.3, 0.7],
            "labels": [1, 1, 1, 1],
        },
        {"boxes": [[50, 50, 60, 60]], "scores": [0.75], "labels": [1]},
    ],
    gt_boxes=gt_b,
    gt_labels=gt_l,
    score_threshold=0.6,
    gt_iou_threshold=0.5,
    consistency_iou=0.6,
    min_views=2,
)
c = res["counts"]
check("funnel candidates = 4", c["candidates"] == 4, f"{c}")
check("funnel after confidence = 3", c["after_confidence"] == 3, f"{c}")
check("funnel after GT dedup = 2", c["after_gt_dedup"] == 2, f"{c}")
check("funnel after consistency = 1", c["after_consistency"] == 1, f"{c}")
check(
    "funnel kept the correct box",
    len(res["boxes"]) == 1 and np.allclose(res["boxes"][0], [50, 50, 60, 60]),
    f"got {res['boxes']}",
)
check("funnel is monotonically non-increasing",
      c["candidates"] >= c["after_confidence"] >= c["after_gt_dedup"] >= c["after_consistency"] >= c["kept"],
      f"{c}")

# The cap must bite, keeping the most confident.
res = harvest_pseudo_labels(
    [{"boxes": [[i * 20, 0, i * 20 + 10, 10] for i in range(6)],
      "scores": [0.9, 0.95, 0.7, 0.8, 0.85, 0.75],
      "labels": [1] * 6}],
    gt_boxes=np.zeros((0, 4)),
    gt_labels=np.zeros(0),
    score_threshold=0.6,
    min_views=1,
    max_per_image=2,
)
check("cap limits per-image pseudo-labels", res["counts"]["kept"] == 2, f"{res['counts']}")
check(
    "cap keeps the two most confident",
    sorted(np.round(res["scores"], 2).tolist()) == [0.9, 0.95],
    f"got {res['scores']}",
)

# ======================================================================= #
print("\n[6] COCO audit and cleaning")
# ======================================================================= #

coco = {
    "images": [
        {"id": 1, "width": 100, "height": 100, "file_name": "a.png"},
        {"id": 2, "width": 100, "height": 100, "file_name": "b.png"},
        {"id": 3, "width": 0, "height": 100, "file_name": "bad_dims.png"},
        {"id": 4, "width": 100, "height": 100, "file_name": "unannotated.png"},
    ],
    "categories": [{"id": 1, "name": "crab"}, {"id": 2, "name": "urchin"}, {"id": 3, "name": "jelly"}],
    "annotations": [
        {"id": 1, "image_id": 1, "category_id": 1, "bbox": [10, 10, 20, 20]},
        {"id": 2, "image_id": 1, "category_id": 1, "bbox": [10, 10, 20, 20]},   # exact dup
        {"id": 3, "image_id": 1, "category_id": 1, "bbox": [10.1, 10.1, 20, 20]},  # near dup
        {"id": 4, "image_id": 2, "category_id": 2, "bbox": [0, 0, 0, 10]},      # zero width
        {"id": 5, "image_id": 2, "category_id": 2, "bbox": [90, 90, 50, 50]},   # out of bounds
        {"id": 6, "image_id": 2, "category_id": 2, "bbox": [1, 1, 1, 1]},       # tiny
        {"id": 7, "image_id": 999, "category_id": 1, "bbox": [0, 0, 5, 5]},     # orphan image
        {"id": 8, "image_id": 1, "category_id": 77, "bbox": [0, 0, 5, 5]},      # unknown class
        {"id": 9, "image_id": 2, "category_id": 2, "bbox": [float("nan"), 0, 5, 5]},  # NaN
    ],
}

rep = audit_coco(coco)
check("audit counts images", rep["n_images"] == 4, f"{rep['n_images']}")
check("audit finds orphan image_id", 7 in rep["orphan_image_id"], f"{rep['orphan_image_id']}")
check("audit finds unknown category", 8 in rep["unknown_category_id"], f"{rep['unknown_category_id']}")
check("audit finds NaN bbox", 9 in rep["non_finite_bbox"], f"{rep['non_finite_bbox']}")
check("audit finds zero-extent bbox", 4 in rep["non_positive_bbox"], f"{rep['non_positive_bbox']}")
check("audit finds out-of-bounds bbox", 5 in rep["out_of_bounds_bbox"], f"{rep['out_of_bounds_bbox']}")
check("audit finds tiny bbox", 6 in rep["tiny_bbox"], f"{rep['tiny_bbox']}")
check("audit finds exact duplicate", 2 in rep["exact_duplicates"], f"{rep['exact_duplicates']}")
check("audit finds near duplicate", len(rep["near_duplicates"]) >= 1, f"{rep['near_duplicates']}")
check("audit finds bad image dims", 3 in rep["bad_image_dims"], f"{rep['bad_image_dims']}")
check(
    "audit finds unannotated images",
    3 in rep["images_without_annotations"] and 4 in rep["images_without_annotations"],
    f"{rep['images_without_annotations']}",
)
check(
    "audit finds category with no instances",
    "jelly" in rep["categories_with_no_instances"],
    f"{rep['categories_with_no_instances']}",
)
check("audit reports category counts", rep["category_counts"].get("crab", 0) >= 1, f"{rep['category_counts']}")
check("audit reports instances-per-image stats", "mean" in rep["instances_per_image"])

before = len(coco["annotations"])
cleaned, summary = clean_coco(coco)
check("clean does not mutate input", len(coco["annotations"]) == before)
check("clean drops orphan", summary.get("dropped_orphan", 0) == 1, f"{summary}")
check("clean drops unknown category", summary.get("dropped_unknown_category", 0) == 1, f"{summary}")
check("clean drops malformed/NaN", summary.get("dropped_malformed", 0) == 1, f"{summary}")
check("clean drops zero-extent", summary.get("dropped_non_positive", 0) == 1, f"{summary}")
check("clean drops exact duplicate", summary.get("dropped_duplicate", 0) == 1, f"{summary}")
check("clean clips out-of-bounds", summary.get("clipped", 0) == 1, f"{summary}")
check("clean keeps tiny box by default", summary.get("dropped_tiny", 0) == 0, f"{summary}")

for ann in cleaned["annotations"]:
    x, y, w, h = ann["bbox"]
    im = next(i for i in cleaned["images"] if i["id"] == ann["image_id"])
    if im["width"] > 0 and im["height"] > 0:
        assert x >= 0 and y >= 0 and x + w <= im["width"] + 1e-6 and y + h <= im["height"] + 1e-6, ann
check("all cleaned boxes inside frame", True)
check(
    "cleaned areas recomputed",
    all(close(a["area"], a["bbox"][2] * a["bbox"][3]) for a in cleaned["annotations"]),
)
check(
    "clean is idempotent",
    clean_coco(cleaned)[1]["kept"] == summary["kept"],
    f"{clean_coco(cleaned)[1]}",
)

_, summary_tiny = clean_coco(coco, drop_tiny=True)
check("drop_tiny removes tiny box when asked", summary_tiny.get("dropped_tiny", 0) == 1, f"{summary_tiny}")

# ======================================================================= #
print("\n[7] image-level splitting")
# ======================================================================= #

big = {
    "images": [{"id": i, "width": 100, "height": 100} for i in range(1000)],
    "categories": [{"id": 1, "name": "crab"}],
    "annotations": [{"id": i, "image_id": i, "category_id": 1, "bbox": [0, 0, 5, 5]} for i in range(1000)],
}
tr, va = image_level_split(big, val_fraction=0.15)
check("split covers all images", len(tr) + len(va) == 1000, f"{len(tr)}+{len(va)}")
check("split is disjoint", not (set(tr) & set(va)))
check("val fraction approximately honoured", 0.11 < len(va) / 1000 < 0.19, f"{len(va)/1000}")

tr2, va2 = image_level_split(big, val_fraction=0.15)
check("split is deterministic", tr == tr2 and va == va2)

tr3, va3 = image_level_split(big, val_fraction=0.15, salt="different")
check("different salt gives different split", va3 != va)

tr4, va4 = image_level_split(big, val_fraction=0.0)
check("val_fraction=0 gives empty val", len(va4) == 0 and len(tr4) == 1000)

# Grouped split: whole groups must move together.
grouped = {
    "images": [{"id": i, "width": 10, "height": 10, "source": f"dive{i % 20}"} for i in range(400)],
    "categories": [{"id": 1, "name": "crab"}],
    "annotations": [],
}
tr, va = image_level_split(grouped, val_fraction=0.3, group_key="source")
src = {im["id"]: im["source"] for im in grouped["images"]}
tr_src, va_src = {src[i] for i in tr}, {src[i] for i in va}
check("grouped split keeps groups intact", not (tr_src & va_src), f"overlap {tr_src & va_src}")
check("grouped split is non-trivial", len(va) > 0 and len(tr) > 0, f"{len(tr)}/{len(va)}")

sub = subset_coco(cleaned, [1])
check("subset filters images", {im["id"] for im in sub["images"]} == {1})
check("subset filters annotations", all(a["image_id"] == 1 for a in sub["annotations"]))
check("subset preserves categories", len(sub["categories"]) == len(cleaned["categories"]))

# ======================================================================= #
print("\n[8] pseudo-label merge into COCO")
# ======================================================================= #

base = {
    "images": [{"id": 1, "width": 100, "height": 100}],
    "categories": [{"id": 1, "name": "crab"}],
    "annotations": [{"id": 5, "image_id": 1, "category_id": 1, "bbox": [0, 0, 10, 10], "area": 100}],
}
merged_coco, added = merge_pseudo_labels(
    base, {1: {"boxes": [[20, 20, 40, 50]], "labels": [1], "scores": [0.8]}}
)
check("merge added one annotation", added == 1 and len(merged_coco["annotations"]) == 2)
new = [a for a in merged_coco["annotations"] if a.get("is_pseudo")]
check("pseudo annotation is tagged", len(new) == 1)
check("pseudo bbox converted to xywh", np.allclose(new[0]["bbox"], [20, 20, 20, 30]), f"{new[0]['bbox']}")
check("pseudo id does not collide", new[0]["id"] != 5)
check("pseudo keeps originating score", close(new[0]["score"], 0.8))
check("pseudo area computed", close(new[0]["area"], 600.0), f"{new[0]['area']}")
check("merge does not mutate input", len(base["annotations"]) == 1)

_, added = merge_pseudo_labels(base, {999: {"boxes": [[0, 0, 5, 5]], "labels": [1], "scores": [0.9]}})
check("merge ignores unknown image_id", added == 0)

# ======================================================================= #
print("\n[9] submission format")
# ======================================================================= #

rows = build_submission({7: {"boxes": [[10, 20, 40, 60]], "scores": [0.87], "labels": [3]}})
check("submission emits one row", len(rows) == 1)
check("submission column set exact", list(rows[0].keys()) == SUBMISSION_COLUMNS, f"{list(rows[0].keys())}")
check("submission converts xyxy->xywh", (rows[0]["bbox_width"], rows[0]["bbox_height"]) == (30.0, 40.0), f"{rows[0]}")
check("submission x/y are top-left", (rows[0]["bbox_x"], rows[0]["bbox_y"]) == (10.0, 20.0))
check("submission image_id preserved", rows[0]["image_id"] == 7)
check("submission category_id preserved", rows[0]["category_id"] == 3)
check("submission score preserved", close(rows[0]["score"], 0.87))

rows = build_submission(
    {1: {"boxes": [[0, 0, 5, 5], [1, 1, 6, 6]], "scores": [1.5, -0.2], "labels": [1, 2]}}
)
check("submission clamps scores to [0,1]", rows[0]["score"] == 1.0 and rows[1]["score"] == 0.0, f"{[r['score'] for r in rows]}")
check("submission annotation_ids unique", len({r["annotation_id"] for r in rows}) == len(rows))

rows = build_submission({1: {"boxes": np.zeros((0, 4)), "scores": [], "labels": []}})
check("submission handles empty prediction", rows == [])

# ======================================================================= #
print("\n[10] end-to-end: synthetic PU pipeline")
# ======================================================================= #

# Build a synthetic dataset where we KNOW which positives were withheld, then
# confirm the pipeline recovers withheld positives without inventing new ones.
rng = np.random.default_rng(0)
n_images = 40
true_objects, withheld = {}, {}
images, annotations = [], []
ann_id = 1

for im_id in range(1, n_images + 1):
    images.append({"id": im_id, "width": 640, "height": 480, "file_name": f"{im_id}.png"})
    k = int(rng.integers(2, 5))
    boxes = []
    for j in range(k):
        x = float(rng.integers(0, 540))
        y = float(rng.integers(0, 380))
        boxes.append([x, y, x + 80, y + 80])
    true_objects[im_id] = boxes
    # Keep only the first object as a label; the rest are unlabeled positives.
    withheld[im_id] = boxes[1:]
    bb = boxes[0]
    annotations.append({
        "id": ann_id, "image_id": im_id, "category_id": 1,
        "bbox": [bb[0], bb[1], bb[2] - bb[0], bb[3] - bb[1]],
        "area": (bb[2] - bb[0]) * (bb[3] - bb[1]), "iscrowd": 0,
    })
    ann_id += 1

synth = {"images": images, "categories": [{"id": 1, "name": "organism"}], "annotations": annotations}

rep = audit_coco(synth)
check("synthetic data audits clean", rep["non_positive_bbox"] == [] and rep["exact_duplicates"] == [])
check(
    "synthetic mean instances/image is 1.0 (PU signature)",
    close(rep["instances_per_image"]["mean"], 1.0),
    f"{rep['instances_per_image']}",
)

tr_ids, va_ids = image_level_split(synth, val_fraction=0.2)
check("synthetic split disjoint and complete", len(tr_ids) + len(va_ids) == n_images and not (set(tr_ids) & set(va_ids)))

# Simulate a detector: finds every true object with jitter, plus one false
# positive per image from "texture". Second view = a noisier re-detection.
def simulate_detector(im_id, jitter, fp_scale):
    boxes, scores = [], []
    for bb in true_objects[im_id]:
        j = rng.normal(0, jitter, 4)
        boxes.append([bb[0] + j[0], bb[1] + j[1], bb[2] + j[2], bb[3] + j[3]])
        scores.append(float(np.clip(rng.normal(0.85, 0.05), 0, 1)))
    # A texture artifact: high-ish score but its location moves between views.
    fx = float(rng.integers(0, 500))
    boxes.append([fx, 400.0, fx + 60 * fp_scale, 460.0])
    scores.append(float(np.clip(rng.normal(0.7, 0.05), 0, 1)))
    return {"boxes": np.array(boxes, float), "scores": np.array(scores, float),
            "labels": np.ones(len(boxes), int)}

gt_by_image = {im_id: [] for im_id in true_objects}
for a in synth["annotations"]:
    x, y, w, h = a["bbox"]
    gt_by_image[a["image_id"]].append([x, y, x + w, y + h])

harvest, totals = {}, {"kept": 0, "recovered": 0, "spurious": 0}
for im_id in tr_ids:
    v1 = simulate_detector(im_id, jitter=2.0, fp_scale=1.0)
    v2 = simulate_detector(im_id, jitter=3.0, fp_scale=1.0)
    res = harvest_pseudo_labels(
        [v1, v2],
        gt_boxes=np.array(gt_by_image[im_id], float),
        gt_labels=np.ones(len(gt_by_image[im_id]), int),
        score_threshold=0.6, gt_iou_threshold=0.5,
        consistency_iou=0.5, min_views=2, max_per_image=5,
    )
    harvest[im_id] = res
    totals["kept"] += res["counts"]["kept"]
    if len(res["boxes"]) and len(withheld[im_id]):
        m = iou_matrix(res["boxes"], np.array(withheld[im_id], float))
        hit = (m.max(axis=1) >= 0.5)
        totals["recovered"] += int(hit.sum())
        totals["spurious"] += int((~hit).sum())
    else:
        totals["spurious"] += len(res["boxes"])

check("harvest produced pseudo-labels", totals["kept"] > 0, f"{totals}")
check("harvest recovered withheld positives", totals["recovered"] > 0, f"{totals}")
precision = totals["recovered"] / max(totals["kept"], 1)
check("harvest precision > 0.8 (conservative)", precision > 0.8, f"precision={precision:.3f} {totals}")

merged_synth, added = merge_pseudo_labels(synth, {k: v for k, v in harvest.items()})
check("pseudo-labels merged into COCO", added == totals["kept"], f"{added} vs {totals['kept']}")
rep2 = audit_coco(merged_synth)
check(
    "merged dataset still audits clean",
    rep2["non_positive_bbox"] == [] and rep2["orphan_image_id"] == [],
    f"{rep2['non_positive_bbox']} {rep2['orphan_image_id']}",
)
check(
    "instances/image rose after recovery",
    rep2["instances_per_image"]["mean"] > rep["instances_per_image"]["mean"],
    f"{rep['instances_per_image']['mean']} -> {rep2['instances_per_image']['mean']}",
)

# Inference on the val split: two-scale merge then submission.
preds = {}
for im_id in va_ids:
    v_med = simulate_detector(im_id, jitter=3.0, fp_scale=1.0)
    v_hi = simulate_detector(im_id, jitter=1.0, fp_scale=1.0)
    preds[im_id] = merge_multiscale(
        [
            {"boxes": v_med["boxes"], "scores": v_med["scores"], "labels": v_med["labels"], "size": (640, 480)},
            {"boxes": v_hi["boxes"], "scores": v_hi["scores"], "labels": v_hi["labels"], "size": (640, 480), "weight": 1.0},
        ],
        original_size=(640, 480), method="gaussian", sigma=0.5,
    )
check("two-scale inference produced predictions", all(len(p["scores"]) > 0 for p in preds.values()))
for im_id, p in preds.items():
    b = p["boxes"]
    assert np.all(b[:, 0] >= 0) and np.all(b[:, 1] >= 0)
    assert np.all(b[:, 2] <= 640 + 1e-6) and np.all(b[:, 3] <= 480 + 1e-6)
    assert np.all(b[:, 2] > b[:, 0]) and np.all(b[:, 3] > b[:, 1])
check("all merged boxes valid and inside frame", True)

rows = build_submission(preds)
check("submission built from merged predictions", len(rows) > 0)
check("all submission rows well-formed", all(list(r.keys()) == SUBMISSION_COLUMNS for r in rows))
check("all submission widths positive", all(r["bbox_width"] > 0 and r["bbox_height"] > 0 for r in rows))
check("all submission scores in [0,1]", all(0.0 <= r["score"] <= 1.0 for r in rows))
check("submission annotation_ids globally unique", len({r["annotation_id"] for r in rows}) == len(rows))

# PU-aware loss weighting behaves sensibly across a realistic objectness spread.
obj = np.linspace(0, 1, 11)
w = pu_background_weight(obj, w_min=0.25, w_max=1.0, tau=0.6, temperature=0.12)
check("weight curve spans a useful range", (w.max() - w.min()) > 0.5, f"{w.min():.3f}..{w.max():.3f}")
check("weight curve strictly decreasing", bool(np.all(np.diff(w) < 0)))

# ======================================================================= #
print("\n[11] generated modules match the notebook")
# ======================================================================= #

import dataclasses  # noqa: E402
import json as _json  # noqa: E402
import re as _re  # noqa: E402

_root = Path(__file__).resolve().parents[1]
_nb_path = _root / "notebooks" / "fathomnet_clef2026_pu_detection.ipynb"

if not _nb_path.is_file():
    check("notebook present", False, f"{_nb_path} missing")
else:
    _nb = _json.loads(_nb_path.read_text(encoding="utf-8"))
    _cells = ["".join(c["source"]) for c in _nb["cells"] if c["cell_type"] == "code"]

    import ast as _ast
    _bad = []
    for _i, _s in enumerate(_cells):
        try:
            _ast.parse(_s)
        except SyntaxError as _e:
            _bad.append(f"cell {_i}: {_e.msg}")
    check("every notebook code cell parses", not _bad, str(_bad))

    # The config dataclass is generated from the notebook's CFG; if they drift,
    # the CLI runner silently ignores settings the notebook exposes.
    _cfg_cells = [c for c in _cells if "class CFG:" in c]
    check("exactly one CFG cell", len(_cfg_cells) == 1, f"{len(_cfg_cells)}")
    if len(_cfg_cells) == 1:
        _body = _cfg_cells[0].split("class CFG:", 1)[1].split("CFG = CFG()", 1)[0]
        _nb_fields = _re.findall(r"^\s{4}(\w+)\s*:", _body, _re.M)
        try:
            from pu_config import PipelineConfig
            _mod_fields = [f.name for f in dataclasses.fields(PipelineConfig)]
            check("PipelineConfig matches notebook CFG",
                  _mod_fields == _nb_fields,
                  f"module {len(_mod_fields)} vs notebook {len(_nb_fields)}; "
                  f"differ: {set(_mod_fields) ^ set(_nb_fields)}")
        except ImportError as _e:
            check("pu_config importable", False, str(_e))

    # pu_pipeline must be importable without side effects: it is generated from
    # notebook cells that also contain top-level calls, which must not survive.
    _pipe = _root / "src" / "pu_pipeline.py"
    if _pipe.is_file():
        _tree = _ast.parse(_pipe.read_text(encoding="utf-8"))
        _top = [type(n).__name__ for n in _tree.body
                if not isinstance(n, (_ast.Expr, _ast.Import, _ast.ImportFrom,
                                      _ast.FunctionDef, _ast.ClassDef, _ast.Assign))]
        check("pu_pipeline has no top-level statements", not _top, str(_top))
        _fns = {n.name for n in _tree.body if isinstance(n, _ast.FunctionDef)}
        _need = {"probe_environment", "ensure_images", "build_dataset_classes",
                 "build_category_maps", "build_model", "train_one_run",
                 "predict_two_scale", "harvest_over_dataset", "evaluate_map",
                 "load_checkpoint", "make_grad_scaler", "autocast_ctx"}
        check("pu_pipeline exports the full runtime", _need <= _fns,
              f"missing {_need - _fns}")
    else:
        check("pu_pipeline generated", False, f"{_pipe} missing")

# ======================================================================= #
print("\n[12] image acquisition and geometry")
# ======================================================================= #

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pu_pipeline import image_urls, image_url, local_image_path, record_size, save_frame  # noqa: E402

# --- every candidate URL must be offered, not just the first --------------
rec = {"id": 1, "coco_url": "https://a.example/x.png",
       "flickr_url": "https://b.example/x.png", "width": 1920, "height": 1080}
check("image_urls returns every candidate",
      image_urls(rec) == ["https://a.example/x.png", "https://b.example/x.png"],
      str(image_urls(rec)))
check("image_url keeps preference order", image_url(rec) == "https://a.example/x.png")
check("image_urls dedupes identical urls",
      len(image_urls({"coco_url": "https://a/x", "flickr_url": "https://a/x"})) == 1)
check("image_urls ignores non-http values",
      image_urls({"coco_url": "", "flickr_url": None, "url": 17}) == [])
check("image_urls on a record with no urls", image_urls({"id": 5}) == [])

# --- cached filenames are always .jpg -------------------------------------
import tempfile as _tf  # noqa: E402
_cache = Path(_tf.mkdtemp())
check("cache path uses .jpg",
      local_image_path({"id": 1, "file_name": "frame.png"}, _cache).suffix == ".jpg")
check("cache path keeps the stem",
      local_image_path({"id": 1, "file_name": "frame.png"}, _cache).stem == "frame")
check("cache path falls back to a url hash",
      local_image_path({"id": 1, "coco_url": "https://a/x.png"}, _cache).suffix == ".jpg")

# --- record_size is authoritative over the decoded file -------------------
from PIL import Image as _Im  # noqa: E402
_small = _Im.new("RGB", (100, 50), (1, 2, 3))
check("record_size prefers the record", record_size(rec, _small) == (1920, 1080),
      str(record_size(rec, _small)))
check("record_size falls back to the image",
      record_size({"id": 2}, _small) == (100, 50))
try:
    record_size({"id": 3})
    check("record_size raises without dimensions", False, "no raise")
except ValueError:
    check("record_size raises without dimensions", True)
check("record_size rejects zero dimensions",
      record_size({"id": 4, "width": 0, "height": 10}, _small) == (100, 50))

# --- save_frame downscales, keeps aspect, and writes a real JPEG ----------
import io as _io  # noqa: E402
_buf = _io.BytesIO()
_Im.new("RGB", (1920, 1080), (10, 120, 200)).save(_buf, format="PNG")
_png = _buf.getvalue()

_dst = _cache / "scaled.jpg"
save_frame(_png, _dst, max_side=1024)
_out = _Im.open(_dst)
check("save_frame writes JPEG", _out.format == "JPEG", str(_out.format))
check("save_frame honours max_side", max(_out.size) == 1024, str(_out.size))
check("save_frame preserves aspect ratio",
      abs(_out.size[0] / _out.size[1] - 1920 / 1080) < 0.01, str(_out.size))
# The win is decode cost, which scales with pixel count, not with file size --
# a flat test image compresses better as PNG than as JPEG and says nothing.
check("save_frame cuts the pixels to decode",
      (_out.size[0] * _out.size[1]) < 0.4 * (1920 * 1080),
      f"{_out.size} vs (1920, 1080)")
check("save_frame leaves no .part file", not (_cache / "scaled.jpg.part").exists())

_dst2 = _cache / "unscaled.jpg"
save_frame(_png, _dst2, max_side=0)
check("max_side=0 leaves size alone", _Im.open(_dst2).size == (1920, 1080))

_dst3 = _cache / "small.jpg"
_b2 = _io.BytesIO(); _Im.new("RGB", (400, 300)).save(_b2, format="PNG")
save_frame(_b2.getvalue(), _dst3, max_side=1024)
check("save_frame never upscales", _Im.open(_dst3).size == (400, 300))

# --- the geometry contract that downscaling depends on --------------------
# A box normalised against the RECORD size must land back on the same pixels
# after inference denormalises against that same record size, no matter what
# resolution the cached file happens to be stored at.
W, H = 1920, 1080
bbox = [440.0, 384.0, 392.0, 157.0]           # real annotation from image_id 9
cx, cy = (bbox[0] + bbox[2] / 2) / W, (bbox[1] + bbox[3] / 2) / H
bw, bh = bbox[2] / W, bbox[3] / H
back = [(cx - bw / 2) * W, (cy - bh / 2) * H, (cx + bw / 2) * W, (cy + bh / 2) * H]
check("normalise/denormalise round-trips on record dims",
      np.allclose(back, [bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]]),
      str(back))

# --- adaptive harvest ------------------------------------------------------
rng_h = np.random.default_rng(7)
junk = rng_h.uniform(0, 400, (500, 4))
junk[:, 2:] = junk[:, :2] + 20
strong = np.array([[100.0, 100.0, 160.0, 160.0]])
v1 = {"boxes": np.vstack([junk, strong]),
      "scores": np.concatenate([rng_h.uniform(0.01, 0.20, 500), [0.55]]),
      "labels": np.ones(501, dtype=int)}
v2 = {"boxes": np.array([[101.0, 99.0, 161.0, 159.0]]),
      "scores": np.array([0.50]), "labels": np.array([1])}

res_h = harvest_pseudo_labels([v1, v2], np.zeros((0, 4)), np.zeros(0),
                              score_threshold=0.25, min_views=2, pre_top_k=30)
c_h = res_h["counts"]
check("top_k trims the candidate flood", c_h["after_top_k"] == 30, str(c_h))
check("floor removes the noise", c_h["after_confidence"] == 1, str(c_h))
check("the real detection survives", c_h["kept"] == 1, str(c_h))
check("harvest recovers the right box",
      np.allclose(res_h["boxes"][0], [100, 100, 160, 160]), str(res_h["boxes"]))
check("harvest reports a score distribution",
      set(res_h["stats"]) == {"max_score", "p99_score", "p50_score"},
      str(res_h["stats"]))
check("reported max score is correct", close(res_h["stats"]["max_score"], 0.55))
check("funnel stays monotonic with top_k",
      c_h["candidates"] >= c_h["after_top_k"] >= c_h["after_confidence"]
      >= c_h["after_gt_dedup"] >= c_h["after_consistency"] >= c_h["kept"], str(c_h))

# The failure mode from the first full run: a fixed high gate rejecting
# everything. With the old 0.60 threshold this exact input yields nothing.
res_old = harvest_pseudo_labels([v1, v2], np.zeros((0, 4)), np.zeros(0),
                                score_threshold=0.60, min_views=2, pre_top_k=30)
check("a too-high floor still no-ops (regression witness)",
      res_old["counts"]["kept"] == 0, str(res_old["counts"]))
check("and the adaptive floor does not",
      res_h["counts"]["kept"] > res_old["counts"]["kept"])

# --- the progress log ------------------------------------------------------
from pu_pipeline import RunLog  # noqa: E402

_logdir = Path(_tf.mkdtemp())
_rl = RunLog(_logdir, t0=time.time() - 3600.0, budget_s=8 * 3600.0)
_rl.event("run_start", budget_h=8)
_rl.phase(2, "baseline")
_rl.event("epoch", epoch=0, loss=3.1)
_pj = _logdir / "progress.json"
check("progress.json is written", _pj.is_file())
_recs = _json.loads(_pj.read_text())
check("every event is persisted", len(_recs) == 3, str(len(_recs)))
check("events carry elapsed time", all("elapsed_h" in r for r in _recs))
check("elapsed is computed from t0", 0.9 < _recs[0]["elapsed_h"] < 1.1,
      str(_recs[0]["elapsed_h"]))
check("remaining is computed from the budget",
      6.9 < _recs[0]["remaining_h"] < 7.1, str(_recs[0]["remaining_h"]))
check("phase events record their index",
      _recs[1]["kind"] == "phase_start" and _recs[1]["phase"] == 2, str(_recs[1]))
check("event fields are preserved", _recs[2]["loss"] == 3.1, str(_recs[2]))
check("the log survives a killed run (readable mid-flight)",
      _json.loads(_pj.read_text())[-1]["kind"] == "epoch")

# Bookkeeping must never be able to take a run down.
_broken = RunLog(Path("/nonexistent-dir-xyz"), t0=time.time())
try:
    _broken.event("noop")
    check("an unwritable log does not raise", True)
except Exception as e:
    check("an unwritable log does not raise", False, f"{type(e).__name__}: {e}")

check("pre_top_k=None keeps every candidate",
      harvest_pseudo_labels([v1, v2], np.zeros((0, 4)), np.zeros(0),
                            score_threshold=0.25, min_views=2,
                            pre_top_k=None)["counts"]["after_top_k"] == 501)

# ======================================================================= #
print("\n" + "=" * 62)
print(f"PASSED {len(PASS)}   FAILED {len(FAIL)}")
if FAIL:
    print("\nFailures:")
    for name, detail in FAIL:
        print(f"  - {name}: {detail}")
print("=" * 62)
sys.exit(1 if FAIL else 0)
