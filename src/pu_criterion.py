"""
Stage 4: PU-aware detection criterion (PyTorch).

Implements the blog's objective

    L_total = L_positive + lambda_box * L_box + w_bg(q) * L_background

for an RT-DETR-style head, i.e. sigmoid/focal classification over `num_classes`
with NO explicit background class -- "background" means every class logit is low.

Why a standalone criterion instead of patching the HuggingFace loss:
  * the PU modification is the whole point of the pipeline, so it should be
    readable and testable rather than buried in a monkeypatch;
  * HF's internal loss classes move between versions, and a silent API drift
    here would produce a model that trains without the PU behaviour at all.

It consumes only `logits` and `pred_boxes`, so it works with any DETR-family
model that exposes those two tensors.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment


# --------------------------------------------------------------------------- #
# Box helpers (cxcywh normalised <-> xyxy normalised)
# --------------------------------------------------------------------------- #


def cxcywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    cx, cy, w, h = boxes.unbind(-1)
    return torch.stack(
        [cx - 0.5 * w, cy - 0.5 * h, cx + 0.5 * w, cy + 0.5 * h], dim=-1
    )


def xyxy_to_cxcywh(boxes: torch.Tensor) -> torch.Tensor:
    x1, y1, x2, y2 = boxes.unbind(-1)
    return torch.stack(
        [(x1 + x2) * 0.5, (y1 + y2) * 0.5, (x2 - x1), (y2 - y1)], dim=-1
    )


def box_iou_xyxy(a: torch.Tensor, b: torch.Tensor):
    """Pairwise IoU and union for xyxy boxes. Returns (iou, union)."""
    area_a = (a[:, 2] - a[:, 0]).clamp(min=0) * (a[:, 3] - a[:, 1]).clamp(min=0)
    area_b = (b[:, 2] - b[:, 0]).clamp(min=0) * (b[:, 3] - b[:, 1]).clamp(min=0)

    lt = torch.max(a[:, None, :2], b[None, :, :2])
    rb = torch.min(a[:, None, 2:], b[None, :, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[..., 0] * wh[..., 1]

    union = area_a[:, None] + area_b[None, :] - inter
    iou = inter / union.clamp(min=1e-7)
    return iou, union


def generalized_box_iou(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """GIoU for xyxy boxes. Implemented here to avoid torchvision version drift."""
    iou, union = box_iou_xyxy(a, b)
    lt = torch.min(a[:, None, :2], b[None, :, :2])
    rb = torch.max(a[:, None, 2:], b[None, :, 2:])
    wh = (rb - lt).clamp(min=0)
    enclosing = (wh[..., 0] * wh[..., 1]).clamp(min=1e-7)
    return iou - (enclosing - union) / enclosing


# --------------------------------------------------------------------------- #
# The PU background weight
# --------------------------------------------------------------------------- #


def pu_background_weight(
    objectness: torch.Tensor,
    w_min: float = 0.25,
    w_max: float = 1.0,
    tau: float = 0.5,
    temperature: float = 0.1,
) -> torch.Tensor:
    """
    Torch twin of pu_ops.pu_background_weight -- see that docstring for the idea.

        w(q) = w_max - (w_max - w_min) * sigmoid((q - tau) / temperature)

    An unmatched query with weak object evidence keeps the full background
    penalty. One with strong object evidence has its penalty softened, because it
    may be a missing annotation rather than background.

    Returned detached-safe: the weight is a *supervision* decision, not something
    the model should be able to game by inflating its own confidence, so callers
    pass a detached objectness in.
    """
    if not (0.0 <= w_min <= w_max):
        raise ValueError("require 0 <= w_min <= w_max")
    if temperature <= 0:
        raise ValueError("temperature must be > 0")
    gate = torch.sigmoid((objectness - tau) / temperature)
    return w_max - (w_max - w_min) * gate


def sigmoid_focal_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = 0.25,
    gamma: float = 2.0,
    reduction: str = "none",
) -> torch.Tensor:
    """Sigmoid focal loss (Lin et al. 2017), elementwise by default."""
    prob = logits.sigmoid()
    ce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce * ((1.0 - p_t) ** gamma)
    if alpha >= 0:
        loss = loss * (alpha * targets + (1 - alpha) * (1 - targets))
    if reduction == "sum":
        return loss.sum()
    if reduction == "mean":
        return loss.mean()
    return loss


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #


@dataclass
class PULossConfig:
    num_classes: int = 32

    # Matching costs
    cost_class: float = 2.0
    cost_bbox: float = 5.0
    cost_giou: float = 2.0

    # Loss weights
    w_class: float = 1.0
    w_bbox: float = 5.0
    w_giou: float = 2.0

    # Focal loss
    focal_alpha: float = 0.25
    focal_gamma: float = 2.0

    # --- the PU knobs -----------------------------------------------------
    #: Background weight for an unmatched query with STRONG object evidence.
    #: Must stay > 0. At 0 the model receives no background supervision on its
    #: confident mistakes and false positives explode.
    pu_w_min: float = 0.25
    #: Background weight for an unmatched query with weak object evidence.
    pu_w_max: float = 1.0
    #: Objectness above which a query starts to look like a missing annotation.
    #: Only used when pu_tau_quantile is 0.
    pu_tau: float = 0.5
    #: Derive tau per batch as this quantile of the UNMATCHED queries' objectness
    #: instead of fixing it absolutely. A fixed tau has to be guessed against a
    #: score distribution you do not have yet: at 0.5 on an under-trained
    #: detector the gate never opens and the PU term does nothing. A quantile
    #: adapts as the model sharpens, always softening roughly the same top
    #: fraction of suspicious queries. 0 disables and falls back to pu_tau.
    pu_tau_quantile: float = 0.98
    #: Gate sharpness.
    pu_temperature: float = 0.1
    #: Set True to disable the PU behaviour entirely (ablation baseline).
    pu_disabled: bool = False

    #: Extra multiplier on the classification/box loss of pseudo-labelled GT
    #: boxes, which are less trustworthy than human annotations.
    pseudo_label_weight: float = 0.5

    def background_kwargs(self) -> dict:
        if self.pu_disabled:
            return dict(w_min=self.pu_w_max, w_max=self.pu_w_max,
                        tau=self.pu_tau, temperature=self.pu_temperature)
        return dict(w_min=self.pu_w_min, w_max=self.pu_w_max,
                    tau=self.pu_tau, temperature=self.pu_temperature)


# --------------------------------------------------------------------------- #
# Criterion
# --------------------------------------------------------------------------- #


class PUDetectionCriterion(nn.Module):
    """
    Hungarian-matched detection loss with PU-aware background supervision.

    Targets are a list (length = batch) of dicts:
        {"labels":    LongTensor (n,)            class indices in [0, num_classes)
         "boxes":     FloatTensor (n, 4)         cxcywh, NORMALISED to [0, 1]
         "is_pseudo": BoolTensor  (n,)  optional per-box pseudo-label flag}

    Outputs are a dict:
        {"logits":     FloatTensor (B, Q, num_classes)
         "pred_boxes": FloatTensor (B, Q, 4)     cxcywh, normalised}
    """

    def __init__(self, config: PULossConfig):
        super().__init__()
        self.cfg = config

    # ---------------------------------------------------------------- #
    @torch.no_grad()
    def match(self, outputs: dict, targets: list[dict]) -> list[tuple]:
        """Hungarian assignment between queries and ground-truth boxes."""
        cfg = self.cfg
        logits, pred_boxes = outputs["logits"], outputs["pred_boxes"]
        bs, num_queries = logits.shape[:2]
        indices = []

        for b in range(bs):
            tgt = targets[b]
            n_tgt = int(tgt["labels"].numel())
            if n_tgt == 0:
                indices.append(
                    (
                        torch.zeros(0, dtype=torch.long, device=logits.device),
                        torch.zeros(0, dtype=torch.long, device=logits.device),
                    )
                )
                continue

            prob = logits[b].sigmoid()                      # (Q, C)
            tgt_lab = tgt["labels"].to(logits.device).long()
            tgt_box = tgt["boxes"].to(logits.device).float()

            # Focal-style classification cost, as in Deformable DETR.
            neg = (1 - cfg.focal_alpha) * (prob**cfg.focal_gamma) * (
                -(1 - prob + 1e-8).log()
            )
            pos = cfg.focal_alpha * ((1 - prob) ** cfg.focal_gamma) * (
                -(prob + 1e-8).log()
            )
            cost_class = pos[:, tgt_lab] - neg[:, tgt_lab]   # (Q, n_tgt)

            cost_bbox = torch.cdist(pred_boxes[b], tgt_box, p=1)
            cost_giou = -generalized_box_iou(
                cxcywh_to_xyxy(pred_boxes[b]), cxcywh_to_xyxy(tgt_box)
            )

            C = (
                cfg.cost_class * cost_class
                + cfg.cost_bbox * cost_bbox
                + cfg.cost_giou * cost_giou
            )
            C = torch.nan_to_num(C, nan=1e4, posinf=1e4, neginf=-1e4)

            qi, ti = linear_sum_assignment(C.detach().cpu().numpy())
            indices.append(
                (
                    torch.as_tensor(qi, dtype=torch.long, device=logits.device),
                    torch.as_tensor(ti, dtype=torch.long, device=logits.device),
                )
            )
        return indices

    # ---------------------------------------------------------------- #
    def forward(self, outputs: dict, targets: list[dict]) -> dict:
        cfg = self.cfg
        logits, pred_boxes = outputs["logits"], outputs["pred_boxes"]
        bs, num_queries, num_classes = logits.shape
        device = logits.device

        if num_classes != cfg.num_classes:
            raise ValueError(
                f"model predicts {num_classes} classes but config says {cfg.num_classes}"
            )

        indices = self.match(outputs, targets)

        # Build the dense classification target, plus a per-query mask of which
        # queries were matched and a per-query weight for box/class terms.
        cls_target = torch.zeros_like(logits)
        matched_mask = torch.zeros((bs, num_queries), dtype=torch.bool, device=device)
        pos_weight = torch.ones((bs, num_queries), device=device)

        box_pred_list, box_tgt_list, box_w_list = [], [], []

        for b, (qi, ti) in enumerate(indices):
            if qi.numel() == 0:
                continue
            tgt = targets[b]
            lab = tgt["labels"].to(device).long()[ti]
            box = tgt["boxes"].to(device).float()[ti]

            cls_target[b, qi, lab] = 1.0
            matched_mask[b, qi] = True

            if "is_pseudo" in tgt and tgt["is_pseudo"] is not None:
                is_p = tgt["is_pseudo"].to(device).bool()[ti]
                w = torch.where(
                    is_p,
                    torch.full_like(box[:, 0], cfg.pseudo_label_weight),
                    torch.ones_like(box[:, 0]),
                )
            else:
                w = torch.ones_like(box[:, 0])

            pos_weight[b, qi] = w
            box_pred_list.append(pred_boxes[b, qi])
            box_tgt_list.append(box)
            box_w_list.append(w)

        # ---- classification -------------------------------------------------
        cls_loss_el = sigmoid_focal_loss(
            logits, cls_target, alpha=cfg.focal_alpha, gamma=cfg.focal_gamma
        )                                                     # (B, Q, C)
        cls_loss_q = cls_loss_el.sum(dim=-1)                  # (B, Q)

        # Objectness proxy for an unmatched query: its strongest class score.
        # Detached, so softening the penalty cannot be gamed by the model simply
        # becoming more confident.
        with torch.no_grad():
            objectness = logits.sigmoid().max(dim=-1).values  # (B, Q)

            kw = cfg.background_kwargs()
            if cfg.pu_tau_quantile and not cfg.pu_disabled:
                unmatched_obj = objectness[~matched_mask]
                if unmatched_obj.numel() > 0:
                    q = float(min(max(cfg.pu_tau_quantile, 0.0), 1.0))
                    kw["tau"] = float(
                        torch.quantile(unmatched_obj.float().flatten(), q)
                    )
            tau_used = kw["tau"]

            bg_w = pu_background_weight(objectness, **kw)
            bg_w = torch.where(
                matched_mask, torch.ones_like(bg_w), bg_w
            )                                                 # matched rows unaffected

        pos_term = (cls_loss_q * pos_weight * matched_mask).sum()
        bg_term = (cls_loss_q * bg_w * (~matched_mask)).sum()

        # Normalise by the number of matched boxes, the DETR convention.
        num_boxes = max(int(matched_mask.sum().item()), 1)
        loss_class = cfg.w_class * (pos_term + bg_term) / num_boxes

        # ---- boxes ----------------------------------------------------------
        if box_pred_list:
            bp = torch.cat(box_pred_list, dim=0)
            bt = torch.cat(box_tgt_list, dim=0)
            bw = torch.cat(box_w_list, dim=0)

            l1 = (F.l1_loss(bp, bt, reduction="none").sum(dim=-1) * bw).sum() / num_boxes
            giou_mat = generalized_box_iou(cxcywh_to_xyxy(bp), cxcywh_to_xyxy(bt))
            giou = ((1.0 - giou_mat.diag()) * bw).sum() / num_boxes
        else:
            l1 = logits.sum() * 0.0
            giou = logits.sum() * 0.0

        loss_bbox = cfg.w_bbox * l1
        loss_giou = cfg.w_giou * giou
        total = loss_class + loss_bbox + loss_giou

        return {
            "loss": total,
            "loss_class": loss_class.detach(),
            "loss_bbox": loss_bbox.detach(),
            "loss_giou": loss_giou.detach(),
            # Diagnostics worth logging every epoch:
            "num_matched": torch.tensor(float(num_boxes), device=device),
            "mean_bg_weight": bg_w[~matched_mask].mean().detach()
            if (~matched_mask).any()
            else torch.tensor(0.0, device=device),
            "n_softened": (
                (bg_w < cfg.pu_w_max - 1e-6) & (~matched_mask)
            ).sum().detach(),
            # Logged so a gate that never opens is visible in the training log
            # rather than inferred afterwards from a suspiciously flat bg_w.
            "tau": torch.tensor(float(tau_used), device=device),
            "max_objectness": objectness.max().detach(),
        }
