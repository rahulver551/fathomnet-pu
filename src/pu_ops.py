"""
Core positive-unlabeled detection ops, NumPy reference implementation.

These are the post-processing / data-hygiene algorithms of the pipeline. They are
deliberately NumPy-only so they can be unit-tested on CPU without torch, and so
the notebook's behaviour is reproducible and inspectable.

Box convention throughout:
  - "xywh" = COCO native: [x_min, y_min, width, height]
  - "xyxy" = [x_min, y_min, x_max, y_max]
Unless a function says otherwise it takes and returns xyxy float arrays of shape (N, 4).
"""

from __future__ import annotations

import numpy as np

# --------------------------------------------------------------------------- #
# Box conversions
# --------------------------------------------------------------------------- #


def xywh_to_xyxy(boxes: np.ndarray) -> np.ndarray:
    """COCO [x, y, w, h] -> [x1, y1, x2, y2]."""
    boxes = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
    out = np.empty_like(boxes)
    out[:, 0] = boxes[:, 0]
    out[:, 1] = boxes[:, 1]
    out[:, 2] = boxes[:, 0] + boxes[:, 2]
    out[:, 3] = boxes[:, 1] + boxes[:, 3]
    return out


def xyxy_to_xywh(boxes: np.ndarray) -> np.ndarray:
    """[x1, y1, x2, y2] -> COCO [x, y, w, h]."""
    boxes = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
    out = np.empty_like(boxes)
    out[:, 0] = boxes[:, 0]
    out[:, 1] = boxes[:, 1]
    out[:, 2] = boxes[:, 2] - boxes[:, 0]
    out[:, 3] = boxes[:, 3] - boxes[:, 1]
    return out


def box_areas(boxes: np.ndarray) -> np.ndarray:
    boxes = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
    w = np.clip(boxes[:, 2] - boxes[:, 0], 0.0, None)
    h = np.clip(boxes[:, 3] - boxes[:, 1], 0.0, None)
    return w * h


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Pairwise IoU between two sets of xyxy boxes.

    Returns (len(a), len(b)). Zero-area boxes yield IoU 0 rather than NaN.
    """
    a = np.asarray(a, dtype=np.float64).reshape(-1, 4)
    b = np.asarray(b, dtype=np.float64).reshape(-1, 4)
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), dtype=np.float64)

    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])

    inter = np.clip(x2 - x1, 0.0, None) * np.clip(y2 - y1, 0.0, None)
    union = box_areas(a)[:, None] + box_areas(b)[None, :] - inter
    with np.errstate(divide="ignore", invalid="ignore"):
        iou = np.where(union > 0.0, inter / union, 0.0)
    return iou


# --------------------------------------------------------------------------- #
# Stage 5a: Soft-NMS
# --------------------------------------------------------------------------- #


def soft_nms(
    boxes: np.ndarray,
    scores: np.ndarray,
    method: str = "gaussian",
    sigma: float = 0.5,
    iou_threshold: float = 0.3,
    score_threshold: float = 1e-3,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Soft-NMS (Bodla et al., 2017), single class.

    Rather than deleting an overlapping box outright, its score is *decayed* by
    the amount of overlap. This matters in dense marine scenes where two real
    organisms genuinely overlap: hard NMS would erase the weaker one.

      gaussian: s_i <- s_i * exp(-iou^2 / sigma)          (applied to all overlaps)
      linear:   s_i <- s_i * (1 - iou)   for iou > iou_threshold

    Returns (kept_boxes, kept_scores, kept_indices_into_input).
    Boxes whose decayed score falls below `score_threshold` are dropped.
    """
    if method not in {"gaussian", "linear", "hard"}:
        raise ValueError(f"unknown method {method!r}")

    boxes = np.asarray(boxes, dtype=np.float64).reshape(-1, 4).copy()
    scores = np.asarray(scores, dtype=np.float64).reshape(-1).copy()
    if len(boxes) != len(scores):
        raise ValueError("boxes and scores length mismatch")
    if len(boxes) == 0:
        return boxes, scores, np.zeros(0, dtype=np.int64)

    idx = np.arange(len(boxes))
    keep_b, keep_s, keep_i = [], [], []

    while len(scores) > 0:
        # Take the current maximum as the "anchor" detection.
        m = int(np.argmax(scores))
        b_m, s_m, i_m = boxes[m].copy(), float(scores[m]), int(idx[m])

        if s_m < score_threshold:
            break

        keep_b.append(b_m)
        keep_s.append(s_m)
        keep_i.append(i_m)

        # Remove the anchor, then decay whatever overlaps it.
        boxes = np.delete(boxes, m, axis=0)
        scores = np.delete(scores, m, axis=0)
        idx = np.delete(idx, m, axis=0)
        if len(boxes) == 0:
            break

        ious = iou_matrix(b_m[None, :], boxes)[0]

        if method == "gaussian":
            scores = scores * np.exp(-(ious**2) / sigma)
        elif method == "linear":
            decay = np.where(ious > iou_threshold, 1.0 - ious, 1.0)
            scores = scores * decay
        else:  # hard NMS, for ablation
            scores = np.where(ious > iou_threshold, 0.0, scores)

        surviving = scores >= score_threshold
        boxes, scores, idx = boxes[surviving], scores[surviving], idx[surviving]

    if not keep_b:
        return (
            np.zeros((0, 4), dtype=np.float64),
            np.zeros(0, dtype=np.float64),
            np.zeros(0, dtype=np.int64),
        )
    return (
        np.stack(keep_b),
        np.asarray(keep_s, dtype=np.float64),
        np.asarray(keep_i, dtype=np.int64),
    )


def soft_nms_per_class(
    boxes: np.ndarray,
    scores: np.ndarray,
    labels: np.ndarray,
    **kw,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Apply Soft-NMS independently within each class.

    Cross-class suppression is wrong here: a shrimp sitting on a sponge produces
    two heavily-overlapping boxes of different classes and both are correct.
    """
    boxes = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    labels = np.asarray(labels).reshape(-1)

    out_b, out_s, out_l = [], [], []
    for c in np.unique(labels):
        sel = np.flatnonzero(labels == c)
        kb, ks, _ = soft_nms(boxes[sel], scores[sel], **kw)
        if len(kb):
            out_b.append(kb)
            out_s.append(ks)
            out_l.append(np.full(len(kb), c, dtype=labels.dtype))

    if not out_b:
        return (
            np.zeros((0, 4), dtype=np.float64),
            np.zeros(0, dtype=np.float64),
            np.zeros(0, dtype=labels.dtype),
        )

    b = np.concatenate(out_b)
    s = np.concatenate(out_s)
    l = np.concatenate(out_l)
    order = np.argsort(-s)  # return in descending-score order
    return b[order], s[order], l[order]


# --------------------------------------------------------------------------- #
# Stage 5b: two-scale inference merge
# --------------------------------------------------------------------------- #


def scale_boxes(boxes: np.ndarray, from_size, to_size) -> np.ndarray:
    """
    Map boxes between two image sizes. Sizes are (width, height).

    Used to bring detections produced at an inference resolution back into the
    coordinate frame of the original frame, which is what the metric is computed in.
    """
    boxes = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
    fw, fh = float(from_size[0]), float(from_size[1])
    tw, th = float(to_size[0]), float(to_size[1])
    if fw <= 0 or fh <= 0:
        raise ValueError("from_size must be positive")
    sx, sy = tw / fw, th / fh
    out = boxes.copy()
    out[:, [0, 2]] *= sx
    out[:, [1, 3]] *= sy
    return out


def clip_boxes(boxes: np.ndarray, size) -> np.ndarray:
    """Clip xyxy boxes to an image of size (width, height)."""
    boxes = np.asarray(boxes, dtype=np.float64).reshape(-1, 4).copy()
    w, h = float(size[0]), float(size[1])
    boxes[:, 0] = np.clip(boxes[:, 0], 0.0, w)
    boxes[:, 1] = np.clip(boxes[:, 1], 0.0, h)
    boxes[:, 2] = np.clip(boxes[:, 2], 0.0, w)
    boxes[:, 3] = np.clip(boxes[:, 3], 0.0, h)
    return boxes


def merge_multiscale(
    passes: list[dict],
    original_size,
    method: str = "gaussian",
    sigma: float = 0.5,
    iou_threshold: float = 0.3,
    score_threshold: float = 1e-3,
) -> dict:
    """
    Merge detections from several inference passes into one set.

    Each entry of `passes` is a dict:
        {"boxes": (N,4) xyxy, "scores": (N,), "labels": (N,),
         "size": (w, h) the resolution those boxes are expressed in,
         "weight": optional float score multiplier for this pass}

    Every pass is mapped back into `original_size`, concatenated, clipped, and
    reconciled with per-class Soft-NMS.

    The medium-resolution pass contributes scene context; the high-resolution
    pass recovers small organisms. `weight` lets you trust one pass more than
    the other without retraining anything.
    """
    all_b, all_s, all_l = [], [], []
    for p in passes:
        b = np.asarray(p["boxes"], dtype=np.float64).reshape(-1, 4)
        s = np.asarray(p["scores"], dtype=np.float64).reshape(-1)
        l = np.asarray(p["labels"]).reshape(-1)
        if len(b) == 0:
            continue
        b = scale_boxes(b, p["size"], original_size)
        b = clip_boxes(b, original_size)
        s = s * float(p.get("weight", 1.0))
        all_b.append(b)
        all_s.append(s)
        all_l.append(l)

    if not all_b:
        return {
            "boxes": np.zeros((0, 4)),
            "scores": np.zeros(0),
            "labels": np.zeros(0, dtype=np.int64),
        }

    b = np.concatenate(all_b)
    s = np.concatenate(all_s)
    l = np.concatenate(all_l)

    # Drop degenerate boxes produced by clipping at the frame edge.
    valid = box_areas(b) > 0
    b, s, l = b[valid], s[valid], l[valid]

    b, s, l = soft_nms_per_class(
        b,
        s,
        l,
        method=method,
        sigma=sigma,
        iou_threshold=iou_threshold,
        score_threshold=score_threshold,
    )
    return {"boxes": b, "scores": s, "labels": l}


# --------------------------------------------------------------------------- #
# Stage 4: PU-aware background weight (NumPy reference of the torch version)
# --------------------------------------------------------------------------- #


def pu_background_weight(
    objectness: np.ndarray,
    w_min: float = 0.25,
    w_max: float = 1.0,
    tau: float = 0.5,
    temperature: float = 0.1,
) -> np.ndarray:
    """
    Weight w_bg(q) applied to the background loss of an *unmatched* prediction.

    The blog's Stage 4 in one function. An unmatched query with weak object
    evidence is almost certainly background, so it keeps the full penalty. An
    unmatched query with *strong* object evidence may be a missing annotation,
    so its background penalty is softened -- the model is allowed to say
    "I am less certain that this region is negative."

    Smooth sigmoid interpolation between w_max (probably background) and
    w_min (possibly a missing positive):

        w(q) = w_max - (w_max - w_min) * sigmoid((objectness - tau) / temperature)

    Critically w_min > 0. Setting w_min = 0 removes background supervision for
    confident predictions entirely and false positives explode -- every rock and
    sediment texture becomes an organism. This is the single most important
    hyperparameter in the pipeline.
    """
    if not (0.0 <= w_min <= w_max):
        raise ValueError("require 0 <= w_min <= w_max")
    if temperature <= 0:
        raise ValueError("temperature must be > 0")

    q = np.asarray(objectness, dtype=np.float64)
    gate = 1.0 / (1.0 + np.exp(-(q - tau) / temperature))
    return w_max - (w_max - w_min) * gate


# --------------------------------------------------------------------------- #
# Stage 3: conservative pseudo-label recovery
# --------------------------------------------------------------------------- #


def dedup_against_gt(
    boxes: np.ndarray,
    labels: np.ndarray,
    gt_boxes: np.ndarray,
    gt_labels: np.ndarray,
    iou_threshold: float = 0.5,
    class_aware: bool = True,
) -> np.ndarray:
    """
    Boolean mask of candidates that do NOT already duplicate a ground-truth box.

    With class_aware=True a candidate is only suppressed by a GT box of the same
    class, so a shrimp detected on top of an annotated sponge survives.
    """
    boxes = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
    labels = np.asarray(labels).reshape(-1)
    gt_boxes = np.asarray(gt_boxes, dtype=np.float64).reshape(-1, 4)
    gt_labels = np.asarray(gt_labels).reshape(-1)

    if len(boxes) == 0:
        return np.zeros(0, dtype=bool)
    if len(gt_boxes) == 0:
        return np.ones(len(boxes), dtype=bool)

    ious = iou_matrix(boxes, gt_boxes)
    if class_aware:
        same = labels[:, None] == gt_labels[None, :]
        ious = np.where(same, ious, 0.0)
    return ious.max(axis=1) < iou_threshold


def consistency_filter(
    view_dets: list[dict],
    iou_threshold: float = 0.6,
    min_views: int = 2,
    reference: int = 0,
) -> np.ndarray:
    """
    Keep only reference-view candidates corroborated across TTA views.

    `view_dets` is a list of dicts {"boxes", "labels"}, one per view, ALL already
    mapped back into the original coordinate frame. A reference candidate counts
    a view as agreeing if that view holds a same-class box at IoU >= threshold.

    "One confident prediction = interesting; one confident AND stable prediction
    = much more useful." A texture artifact rarely survives a flip and a resize
    at the same location with the same class; a real organism usually does.

    Returns a boolean mask over the reference view's boxes.
    """
    if not view_dets:
        return np.zeros(0, dtype=bool)
    if not (0 <= reference < len(view_dets)):
        raise IndexError("reference view out of range")

    ref = view_dets[reference]
    ref_b = np.asarray(ref["boxes"], dtype=np.float64).reshape(-1, 4)
    ref_l = np.asarray(ref["labels"]).reshape(-1)
    if len(ref_b) == 0:
        return np.zeros(0, dtype=bool)

    # The reference view always corroborates itself.
    votes = np.ones(len(ref_b), dtype=np.int64)

    for vi, v in enumerate(view_dets):
        if vi == reference:
            continue
        vb = np.asarray(v["boxes"], dtype=np.float64).reshape(-1, 4)
        vl = np.asarray(v["labels"]).reshape(-1)
        if len(vb) == 0:
            continue
        ious = iou_matrix(ref_b, vb)
        same = ref_l[:, None] == vl[None, :]
        agree = (np.where(same, ious, 0.0) >= iou_threshold).any(axis=1)
        votes += agree.astype(np.int64)

    return votes >= min_views


def harvest_pseudo_labels(
    view_dets: list[dict],
    gt_boxes: np.ndarray,
    gt_labels: np.ndarray,
    score_threshold: float = 0.6,
    gt_iou_threshold: float = 0.5,
    consistency_iou: float = 0.6,
    min_views: int = 2,
    max_per_image: int | None = 5,
) -> dict:
    """
    The full Stage 3 funnel for one image, in the blog's order:

        confidence filter -> IoU dedup against GT -> cross-view consistency -> cap

    A pseudo-label is only worth adding if it is more likely to *correct* missing
    supervision than to *inject* a new error, so every stage here removes
    candidates and none adds any. `max_per_image` is a blunt but effective guard
    against one pathological frame dumping dozens of pseudo-labels into training.

    Returns the surviving boxes/scores/labels plus a per-stage survivor count,
    which is what you actually watch to tune the thresholds.
    """
    ref = view_dets[0]
    boxes = np.asarray(ref["boxes"], dtype=np.float64).reshape(-1, 4)
    scores = np.asarray(ref["scores"], dtype=np.float64).reshape(-1)
    labels = np.asarray(ref["labels"]).reshape(-1)

    counts = {"candidates": int(len(boxes))}

    # 1. confidence
    keep = scores >= score_threshold
    boxes, scores, labels = boxes[keep], scores[keep], labels[keep]
    counts["after_confidence"] = int(len(boxes))

    # 2. dedup against existing ground truth
    if len(boxes):
        keep = dedup_against_gt(
            boxes, labels, gt_boxes, gt_labels, iou_threshold=gt_iou_threshold
        )
        boxes, scores, labels = boxes[keep], scores[keep], labels[keep]
    counts["after_gt_dedup"] = int(len(boxes))

    # 3. cross-view consistency, evaluated on the surviving subset
    if len(boxes) and len(view_dets) > 1:
        views = [{"boxes": boxes, "labels": labels}] + [
            {"boxes": v["boxes"], "labels": v["labels"]} for v in view_dets[1:]
        ]
        keep = consistency_filter(
            views,
            iou_threshold=consistency_iou,
            min_views=min_views,
            reference=0,
        )
        boxes, scores, labels = boxes[keep], scores[keep], labels[keep]
    counts["after_consistency"] = int(len(boxes))

    # 4. cap, keeping the most confident
    if max_per_image is not None and len(boxes) > max_per_image:
        order = np.argsort(-scores)[:max_per_image]
        boxes, scores, labels = boxes[order], scores[order], labels[order]
    counts["kept"] = int(len(boxes))

    return {"boxes": boxes, "scores": scores, "labels": labels, "counts": counts}
