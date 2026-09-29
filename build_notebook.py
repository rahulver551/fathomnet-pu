#!/usr/bin/env python3
"""
Assemble the Kaggle notebook from the tested source modules plus Kaggle-specific
cells, validating every code cell's syntax before writing the .ipynb.
"""

import ast
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
OUT = ROOT / "notebooks" / "fathomnet_clef2026_pu_detection.ipynb"

CELLS: list[tuple[str, str]] = []


def md(text: str):
    CELLS.append(("markdown", text.strip("\n")))


def code(text: str):
    CELLS.append(("code", text.strip("\n")))


def embed_module(name: str, drop_local_imports=(), renames=()) -> str:
    """Inline a source module as a notebook cell, stripping file-only constructs."""
    text = (SRC / name).read_text(encoding="utf-8")
    text = text.replace("from __future__ import annotations\n", "")
    for line in drop_local_imports:
        text = text.replace(line + "\n", "")
    # In a notebook every cell shares one namespace, so a name defined in two
    # modules silently shadows. Rename on the way in.
    for old, new in renames:
        text = re.sub(rf"\b{re.escape(old)}\b", new, text)
    # Collapse the runs of blank lines left behind by the removals.
    text = re.sub(r"\n{4,}", "\n\n\n", text)
    return text.strip("\n")


# =========================================================================== #
md(r"""
# Teaching an Object Detector Not to Trust the Background

### FathomNet-CLEF 2026 — a positive-unlabeled marine object detection pipeline

This notebook is the executable form of the five-stage pipeline:

| Stage | Idea | Where |
|---|---|---|
| 1 | Clean the supervision before changing the network | §3–§4 |
| 2 | One high-resolution RT-DETR detector | §6–§7 |
| 3 | Conservative pseudo-label recovery | §8 |
| 4 | PU-aware background loss | §5 |
| 5 | Two-scale inference + Soft-NMS | §9 |

**The learning problem.** Training annotations are deliberately incomplete: an
image may contain a crab, two fish and an urchin while only the crab is boxed.
Ordinary detection training reads every unannotated region as background, so the
model is punished for finding organisms that are genuinely there. Evaluation, by
contrast, is fully annotated. The pipeline below is built around one question:

> How do we train a detector when "not labeled" does not mean "negative"?

---

### How to run this on Kaggle

Set `CFG.stage` in the config cell and run top to bottom. Stages are separate so
the work fits inside Kaggle's session limit and GPU quota:

| `CFG.stage` | What it does | Needs GPU | Rough time |
|---|---|---|---|
| `"verify"` | Runs the built-in test suite only | no | seconds |
| `"audit"` | Stage 1: audit + clean + split, no training | no | ~1 min |
| `"train_base"` | Stage 2: baseline detector → checkpoint | **yes** | hours |
| `"harvest"` | Stage 3: pseudo-labels from the baseline | **yes** | ~30 min |
| `"train_pu"` | Stages 3+4: retrain with pseudo-labels + PU loss | **yes** | hours |
| `"infer"` | Stage 5: two-scale inference + submission | **yes** | ~30 min |

**Before a GPU stage:** turn on *Settings → Accelerator → GPU*, and
*Settings → Internet → On* (needed to fetch RT-DETR pretrained weights and, unless
you have attached an images dataset, the imagery itself).

**Start with `"verify"`, then `"audit"`.** Both run on CPU in under a minute and
will catch a wrong dataset path or a broken assumption before you spend GPU quota.

> **Scope note.** The numbers this notebook produces depend entirely on your run.
> Nothing here is a reported leaderboard result, and the defaults below are
> starting points chosen to be safe, not tuned optima.
""")

# --------------------------------------------------------------------------- #
md(r"""
## 1. Configuration

Every knob in one place. The PU-specific ones are grouped at the bottom and are
the only ones worth sweeping first.
""")

code(r'''
import os, sys, json, math, time, random, collections, copy, hashlib, warnings
from dataclasses import dataclass, field, asdict
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore", category=UserWarning)


@dataclass
class CFG:
    # ---- which stage to run -------------------------------------------------
    # "all" | "verify" | "audit" | "train_base" | "harvest" | "train_pu" | "infer"
    # "all" runs the whole pipeline in one session, which is the only way the
    # stages share /kaggle/working -- a fresh commit starts with an empty one.
    # The verification section runs whatever the stage is and halts the notebook
    # if anything fails, so this default cannot skip past a broken build.
    stage: str = "all"

    # ---- paths --------------------------------------------------------------
    # Kaggle mounts competition data under /kaggle/input at an unpredictable
    # depth, so the annotation files are discovered rather than hardcoded.
    input_roots: tuple = ("/kaggle/input", "./data", ".")
    work_dir: str = "/kaggle/working" if Path("/kaggle/working").exists() else "./work"
    image_dir: str = ""        # set automatically; where frames are cached

    # ---- data ---------------------------------------------------------------
    num_classes: int = 32
    val_fraction: float = 0.15
    split_salt: str = "fathomnet-pu-v1"
    split_group_key: str = ""  # e.g. "source" if images carry a provenance field
    drop_tiny_boxes: bool = False

    # Cap the number of training images. 0 = the full set, which is what a real
    # run wants: the wall-clock budget below, not this, is what keeps the run
    # inside the session limit.
    max_images: int = 0
    download_workers: int = 24
    #: Downscale frames to this longest side on download. The sources are
    #: 1920x1080 PNGs; decoding those every epoch made training input-bound.
    #: 0 keeps them untouched.
    download_max_side: int = 1024

    # ---- model / training ---------------------------------------------------
    model_name: str = "PekingU/rtdetr_r50vd_coco_o365"
    img_size: int = 800          # Stage 2: the high-resolution training path
    #: Kept at the profile already proven to fit a T4 at 800px. The throughput
    #: problem was image decode, not GPU occupancy, so raising this trades a
    #: real OOM risk over an unattended 8h run for an uncertain gain.
    batch_size: int = 2
    grad_accum: int = 4
    #: A generous upper bound, not a target. The phase deadline ends training and
    #: the LR schedule is anchored to that deadline, so this only needs to be
    #: larger than the number of epochs that will actually fit.
    epochs: int = 60
    lr: float = 1e-4
    lr_backbone: float = 1e-5
    weight_decay: float = 1e-4
    clip_grad: float = 0.1
    num_workers: int = 4
    #: Steps between training log lines. The first run emitted one every 50 and
    #: Kaggle truncated the log view before the results at the end.
    log_every: int = 200
    amp: bool = True
    seed: int = 1337

    # ---- Stage 3: pseudo-label recovery ------------------------------------
    #: An absolute FLOOR, not the primary filter -- cross-view consistency is.
    #: The previous 0.60 sat above the detector's whole score distribution and
    #: let 15 of 15.9M candidates through, making the stage a no-op.
    pseudo_score_threshold: float = 0.25
    #: Rank candidates and keep only the strongest few per image before
    #: filtering. A DETR head flattened over queries x classes emits thousands
    #: per frame, nearly all noise.
    pseudo_pre_top_k: int = 30
    pseudo_gt_iou_threshold: float = 0.50
    pseudo_consistency_iou: float = 0.60
    pseudo_min_views: int = 2
    pseudo_max_per_image: int = 8
    #: Images to harvest over. Three forward passes each, so the full training
    #: set costs about an hour of GPU. 0 = all.
    harvest_max_images: int = 1500

    # ---- Stage 4: PU-aware background loss ---------------------------------
    # The single most important setting in the notebook. pu_w_min must stay > 0:
    # at 0 the model gets no background supervision on its confident mistakes and
    # every rock and sediment texture becomes an organism.
    pu_w_min: float = 0.25
    pu_w_max: float = 1.00
    pu_tau: float = 0.50
    #: Derive tau per batch as this quantile of unmatched-query objectness.
    #: A fixed tau must be guessed against a score distribution you do not have
    #: yet; at 0.50 the gate never opened and the PU term moved bg_w by ~1%.
    #: 0 disables and falls back to the absolute pu_tau.
    pu_tau_quantile: float = 0.98
    pu_temperature: float = 0.10
    pu_disabled: bool = False     # True = ordinary background loss (ablation)
    pseudo_label_weight: float = 0.50

    # ---- Stage 5: two-scale inference --------------------------------------
    infer_scales: tuple = (640, 960)
    infer_scale_weights: tuple = (1.0, 1.0)
    #: Also run each scale horizontally flipped and merge. Costs one extra
    #: forward pass per scale and reliably helps recall.
    infer_flip_tta: bool = True
    softnms_method: str = "gaussian"
    softnms_sigma: float = 0.50
    softnms_iou_threshold: float = 0.30
    score_threshold: float = 0.01
    max_dets_per_image: int = 100

    # ---- wall-clock budget (stage "all") -----------------------------------
    #: Total budget. Kaggle kills a GPU session at ~9h, so stay under it.
    time_budget_hours: float = 8.0
    #: Held back from training so inference and the submission always happen.
    #: A truncated model that submits beats a trained one killed before writing.
    reserve_hours: float = 1.5
    #: Test images to predict on. 0 = all of them, which is what a real
    #: submission needs; only lower it for a smoke test.
    test_max_images: int = 0


CFG = CFG()

# Allow the stage to be overridden by an env var, handy when scheduling runs.
CFG.stage = os.environ.get("PU_STAGE", CFG.stage)

Path(CFG.work_dir).mkdir(parents=True, exist_ok=True)

random.seed(CFG.seed)
np.random.seed(CFG.seed)

print(f"stage     : {CFG.stage}")
print(f"work_dir  : {CFG.work_dir}")
''')

# --------------------------------------------------------------------------- #
md(r"""
## 2. Environment probe

Establish what is actually available before relying on it. A stage that needs a
GPU stops here with a clear message rather than failing thirty minutes in.
""")

code(r'''
def probe_environment():
    info = {}
    try:
        import torch
        info["torch"] = torch.__version__
        info["cuda"] = torch.cuda.is_available()
        info["n_gpu"] = torch.cuda.device_count() if torch.cuda.is_available() else 0
        if info["cuda"]:
            info["gpu_name"] = torch.cuda.get_device_name(0)
            info["gpu_mem_gb"] = round(
                torch.cuda.get_device_properties(0).total_memory / 1024**3, 1
            )
    except Exception as e:
        info["torch"] = f"MISSING ({e})"
        info["cuda"] = False
        info["n_gpu"] = 0

    for mod in ("torchvision", "transformers", "pycocotools", "scipy", "PIL"):
        try:
            m = __import__(mod)
            info[mod] = getattr(m, "__version__", "present")
        except Exception:
            info[mod] = "MISSING"

    info["on_kaggle"] = Path("/kaggle").exists()
    info["internet"] = False
    try:
        import urllib.request
        urllib.request.urlopen("https://huggingface.co", timeout=8)
        info["internet"] = True
    except Exception:
        pass
    return info


ENV = probe_environment()
for k, v in ENV.items():
    print(f"{k:14s}: {v}")

GPU_STAGES = {"all", "train_base", "harvest", "train_pu", "infer"}
if CFG.stage in GPU_STAGES and not ENV.get("cuda"):
    print(
        "\n>>> STOP: stage "
        f"'{CFG.stage}' needs a GPU but none is visible.\n"
        ">>> Settings -> Accelerator -> GPU, then re-run.\n"
        ">>> ('verify' and 'audit' run fine on CPU.)"
    )
if CFG.stage in GPU_STAGES and not ENV.get("internet"):
    print(
        "\n>>> WARNING: no internet. Pretrained RT-DETR weights cannot be fetched\n"
        ">>> and imagery cannot be downloaded. Settings -> Internet -> On."
    )
''')

# --------------------------------------------------------------------------- #
md(r"""
## 3. Core ops

The post-processing and data-hygiene algorithms. These are deliberately NumPy-only:
they carry the pipeline's actual reasoning, they run identically on CPU, and they
are covered by the test suite in §10 — so a mistake here shows up in seconds
rather than after an epoch of training.

Box convention: `xywh` is COCO-native `[x, y, w, h]`; `xyxy` is `[x1, y1, x2, y2]`.
""")

code(embed_module("pu_ops.py"))

# --------------------------------------------------------------------------- #
md(r"""
## 4. Stage 1 — clean the supervision before changing the network

In a positive-unlabeled setting an annotation *bug* and a deliberately *missing*
label look identical downstream, and only one of the two is yours to fix. So the
audit runs first and is read-only; cleaning is a separate, conservative step that
repairs what it safely can and drops only what cannot describe a real object.

Two details that matter more here than in ordinary detection work:

- **Splits are made at the image level.** Splitting annotations puts two boxes
  from one frame on both sides of the split and leaks. If the images carry a
  provenance field, `split_group_key` holds out whole groups instead — the
  competition's train and test imagery come from different institutional sources,
  so a random same-source split flatters the model.
- **Tiny boxes are kept by default.** A 20-pixel box is often a real small
  organism, which is exactly what the high-resolution path exists to catch.
""")

code(embed_module(
    "pu_data.py",
    drop_local_imports=[
        "    from pu_ops import iou_matrix, xywh_to_xyxy  # local import keeps module standalone",
        "    from pu_ops import xyxy_to_xywh",
    ],
))

# --------------------------------------------------------------------------- #
md(r"""
## 5. Stage 4 — the PU-aware background loss

The heart of the pipeline. A simplified detector objective:

$$L_{\text{total}} = L_{\text{positive}} + \lambda_{\text{box}} L_{\text{box}} + w_{bg}(q)\, L_{\text{background}}$$

Ordinary training sets $w_{bg} \equiv 1$: every unmatched query is background, at
full penalty. The PU version makes that weight depend on how much object evidence
the query carries:

$$w_{bg}(q) = w_{\max} - (w_{\max} - w_{\min})\,\sigma\!\left(\frac{q - \tau}{T}\right)$$

- unmatched **+ weak** object evidence → probably background → full penalty
- unmatched **+ strong** object evidence → possibly a missing annotation → reduced penalty

This is not a claim that confident unmatched predictions are correct. It is the
weaker and more defensible statement: *"I am less certain that this region is negative."*

**`pu_w_min` must stay above zero.** Remove the background penalty entirely and
false positives explode — rocks, coral texture, equipment, sediment and lighting
artifacts all become organisms. The weight buys recall on unlabeled organisms and
pays in precision, and the useful settings are strictly interior:

```
large background weight  ->  fewer false positives, more missed organisms
small background weight  ->  better recall, potentially many false positives
```

Two deliberate implementation choices:

1. **The objectness used for gating is detached.** Otherwise the model could
   lower its own background penalty by becoming more confident — it would learn
   to game the supervision instead of the task.
2. **The criterion is standalone rather than a patch over the HuggingFace loss.**
   HF's internal loss classes move between versions, and a silent API drift there
   would produce a model that trains happily with no PU behaviour at all.
""")

code(embed_module(
    "pu_criterion.py",
    # §3 already defines a NumPy pu_background_weight. Both live in one notebook
    # namespace, so the torch twin is renamed rather than shadowing it -- which
    # would otherwise hand a NumPy array to torch.sigmoid in the §12 tests.
    renames=[("pu_background_weight", "pu_background_weight_torch")],
))

# --------------------------------------------------------------------------- #
md(r"""
### 5b. Self-test for the PU criterion

Runs on CPU in a few seconds. It checks the properties that actually matter — in
particular that `pu_disabled=True` reproduces the standard loss exactly, and that
softening the background weight only ever affects *unmatched* queries.
""")

code(r'''
def selftest_pu_criterion(verbose=True):
    import torch
    torch.manual_seed(0)

    B, Q, C, N = 2, 24, 6, 3
    logits = torch.randn(B, Q, C, requires_grad=True)
    boxes = torch.rand(B, Q, 4) * 0.5 + 0.25
    boxes = boxes.clone().requires_grad_(True)
    outputs = {"logits": logits, "pred_boxes": boxes}
    targets = [
        {"labels": torch.randint(0, C, (N,)), "boxes": torch.rand(N, 4) * 0.4 + 0.3}
        for _ in range(B)
    ]

    results, failures = [], []

    def ck(name, cond, detail=""):
        results.append((name, bool(cond)))
        if not cond:
            failures.append(f"{name} {detail}")
        if verbose:
            print(f"  {'PASS' if cond else 'FAIL'}  {name}  {detail if not cond else ''}")

    base = PULossConfig(num_classes=C)
    crit = PUDetectionCriterion(base)
    out = crit(outputs, targets)

    ck("loss is finite", torch.isfinite(out["loss"]).item(), f"{out['loss']}")
    ck("loss is positive", out["loss"].item() > 0, f"{out['loss'].item()}")

    out["loss"].backward()
    ck("gradient reaches logits", logits.grad is not None and torch.isfinite(logits.grad).all().item())
    ck("gradient reaches boxes", boxes.grad is not None and torch.isfinite(boxes.grad).all().item())
    ck("gradient is non-zero", logits.grad.abs().sum().item() > 0)

    # Matching must be one-to-one and cover every target.
    idx = crit.match(outputs, targets)
    ck("matched count == n targets", all(len(q) == N for q, t in idx))
    ck("query indices unique", all(len(set(q.tolist())) == len(q) for q, t in idx))
    ck("target indices are a permutation", all(sorted(t.tolist()) == list(range(N)) for q, t in idx))

    # pu_disabled must be numerically identical to w_min == w_max.
    a = PUDetectionCriterion(PULossConfig(num_classes=C, pu_disabled=True))(outputs, targets)
    b = PUDetectionCriterion(
        PULossConfig(num_classes=C, pu_w_min=1.0, pu_w_max=1.0)
    )(outputs, targets)
    ck("pu_disabled == uniform background weight",
       abs(a["loss"].item() - b["loss"].item()) < 1e-6,
       f"{a['loss'].item()} vs {b['loss'].item()}")
    ck("pu_disabled softens nothing", a["n_softened"].item() == 0, f"{a['n_softened'].item()}")

    # Lowering w_min must lower total loss (less background penalty) and must
    # leave the box terms untouched, since it only touches unmatched queries.
    hi = PUDetectionCriterion(PULossConfig(num_classes=C, pu_w_min=1.0))(outputs, targets)
    lo = PUDetectionCriterion(PULossConfig(num_classes=C, pu_w_min=0.1))(outputs, targets)
    ck("lower w_min lowers total loss", lo["loss"].item() < hi["loss"].item(),
       f"{lo['loss'].item()} vs {hi['loss'].item()}")
    ck("w_min does not touch box loss",
       abs(lo["loss_bbox"].item() - hi["loss_bbox"].item()) < 1e-6)
    ck("w_min does not touch giou loss",
       abs(lo["loss_giou"].item() - hi["loss_giou"].item()) < 1e-6)
    ck("mean background weight drops with w_min",
       lo["mean_bg_weight"].item() < hi["mean_bg_weight"].item(),
       f"{lo['mean_bg_weight'].item()} vs {hi['mean_bg_weight'].item()}")

    # Weight bounds, and agreement with the NumPy twin from section 3. The two
    # implementations are independent, so this is a real cross-check.
    w = pu_background_weight_torch(torch.linspace(0, 1, 11), w_min=0.25, w_max=1.0,
                                   tau=0.5, temperature=0.1)
    ck("torch weight within bounds", bool((w >= 0.25).all() and (w <= 1.0).all()))
    ck("torch weight strictly decreasing", bool((w.diff() < 0).all()))
    ck("torch weight matches numpy twin",
       np.allclose(w.numpy(), pu_background_weight(np.linspace(0, 1, 11),
                   w_min=0.25, w_max=1.0, tau=0.5, temperature=0.1), atol=1e-6),
       "torch and numpy PU weights disagree")

    # Empty-target images must not crash or produce NaNs.
    empty = [{"labels": torch.zeros(0, dtype=torch.long), "boxes": torch.zeros(0, 4)}
             for _ in range(B)]
    oe = PUDetectionCriterion(base)(outputs, empty)
    ck("empty targets handled", torch.isfinite(oe["loss"]).item(), f"{oe['loss']}")

    # Pseudo-label down-weighting must reduce the positive contribution.
    tp = [dict(t, is_pseudo=torch.ones(N, dtype=torch.bool)) for t in targets]
    p_full = PUDetectionCriterion(PULossConfig(num_classes=C, pseudo_label_weight=1.0))(outputs, tp)
    p_half = PUDetectionCriterion(PULossConfig(num_classes=C, pseudo_label_weight=0.25))(outputs, tp)
    ck("pseudo-label weight reduces box loss",
       p_half["loss_bbox"].item() < p_full["loss_bbox"].item(),
       f"{p_half['loss_bbox'].item()} vs {p_full['loss_bbox'].item()}")

    # Mismatched class count must fail loudly, not silently.
    try:
        PUDetectionCriterion(PULossConfig(num_classes=C + 1))(outputs, targets)
        ck("rejects class-count mismatch", False, "no raise")
    except ValueError:
        ck("rejects class-count mismatch", True)

    n_pass = sum(1 for _, ok in results if ok)
    print(f"\n  criterion self-test: {n_pass}/{len(results)} passed")
    if failures:
        raise AssertionError("PU criterion self-test failures:\n  " + "\n  ".join(failures))
    return True
''')

# --------------------------------------------------------------------------- #
md(r"""
## 6. Loading the challenge data

The official repository ships `dataset_train.json` and `dataset_test.json` in COCO
format, plus a `download.py` — **the imagery is not bundled**, it is fetched from
FathomNet and partner URLs recorded in the annotations. So there are two ways to
get frames, and the resolver below accepts either:

1. an images directory already attached as a Kaggle dataset, or
2. downloading on demand from each image's URL, cached under `work_dir`
   (needs *Settings → Internet → On*).

`CFG.max_images` caps how many frames are used. Leave it small for the first pass.
""")

code(r'''
train_json_path = find_dataset_json(CFG.input_roots, TRAIN_JSON_NAMES)
test_json_path = find_dataset_json(CFG.input_roots, TEST_JSON_NAMES)

print("train json:", train_json_path)
print("test  json:", test_json_path)

if train_json_path is None:
    print(
        "\n>>> Training annotations not found.\n"
        ">>> Attach the FathomNet-CLEF 2026 competition data (Add Input), or drop\n"
        f">>> dataset_train.json under one of: {list(CFG.input_roots)}\n"
        ">>> The 'verify' stage still runs without it."
    )
    raw_train = None
else:
    raw_train = load_coco(train_json_path)
    print(f"\nimages      : {len(raw_train['images'])}")
    print(f"annotations : {len(raw_train['annotations'])}")
    print(f"categories  : {len(raw_train['categories'])}")

    # Which field holds the image URL varies between COCO exports.
    URL_KEYS = ("coco_url", "url", "flickr_url", "image_url", "file_url")
    sample = raw_train["images"][0]
    print("\nimage record keys:", sorted(sample.keys()))
    found = [k for k in URL_KEYS if sample.get(k)]
    print("url field(s)     :", found or "none - imagery must be attached as a dataset")
''')

code(r'''
import urllib.request
from concurrent.futures import ThreadPoolExecutor

URL_KEYS = ("coco_url", "url", "flickr_url", "image_url", "file_url")


def image_urls(im: dict):
    """
    Every candidate URL for an image, in preference order.

    Records carry both coco_url and flickr_url. Returning only the first and
    giving up when it 404s discards images that the other host still serves --
    which cost 27% of the training set on the first full run.
    """
    urls, seen = [], set()
    for k in URL_KEYS:
        v = im.get(k)
        if isinstance(v, str) and v.startswith("http") and v not in seen:
            seen.add(v)
            urls.append(v)
    return urls


def image_url(im: dict):
    u = image_urls(im)
    return u[0] if u else None


def local_image_path(im: dict, cache_dir: Path) -> Path:
    """
    Stable on-disk name for an image record.

    Always .jpg: frames are re-encoded on download (see save_frame), so the
    source extension is irrelevant and a fixed one keeps lookups predictable.
    """
    name = im.get("file_name") or ""
    if name and not name.startswith("http"):
        stem = Path(name).name
        stem = str(Path(stem).with_suffix(".jpg"))
    else:
        url = image_url(im) or str(im["id"])
        stem = hashlib.sha1(url.encode()).hexdigest()[:20] + ".jpg"
    return cache_dir / stem


def save_frame(data: bytes, path: Path, max_side: int = 0):
    """
    Decode, optionally downscale, and store a frame as JPEG.

    The source frames are 1920x1080 PNGs. Decoding and resizing those on every
    epoch made training input-bound rather than GPU-bound; doing it once here
    trades a little download time for a lot of epochs. Box coordinates are
    unaffected because they are normalised against the COCO record's declared
    width/height, never against the stored file's size.
    """
    import io

    from PIL import Image

    img = Image.open(io.BytesIO(data))
    img.load()
    if img.mode != "RGB":
        img = img.convert("RGB")
    if max_side and max(img.size) > max_side:
        w, h = img.size
        s = max_side / float(max(w, h))
        img = img.resize((max(1, round(w * s)), max(1, round(h * s))), Image.BILINEAR)
    tmp = path.with_suffix(path.suffix + ".part")
    img.save(tmp, format="JPEG", quality=92)
    tmp.rename(path)


def find_attached_image_dir(roots, probe_names, max_probe=40):
    """Look for an already-attached directory containing the expected frames."""
    probe = {Path(n).name for n in probe_names if n}
    if not probe:
        return None
    for root in roots:
        root = Path(root)
        if not root.exists():
            continue
        for d in [root] + [p for p in root.rglob("*") if p.is_dir()][:2000]:
            try:
                names = {p.name for p in list(d.iterdir())[:2000] if p.is_file()}
            except Exception:
                continue
            if names and len(probe & names) >= min(3, len(probe)):
                return d
    return None


def image_cache_root(cfg):
    """
    Where downloaded frames live.

    Deliberately NOT work_dir: Kaggle caps a notebook's saved output at 20 GB and
    this dataset's imagery is roughly that on its own, so caching it in the output
    directory makes the commit fail at save time. /kaggle/temp is scratch -- large,
    and discarded with the session.
    """
    for scratch in ("/kaggle/temp", "/tmp"):
        if Path(scratch).is_dir():
            return Path(scratch) / "fathomnet_images"
    return Path(cfg.work_dir) / "images"


def ensure_images(coco, cache_dir, workers=16, verbose=True, deadline=None,
                  max_side=0):
    """
    Make sure every image in `coco` exists on disk; download the missing ones.

    Returns (ok_image_ids, failed_image_ids). Images that cannot be fetched are
    reported rather than silently producing black frames.

    `deadline` (an absolute time.time()) caps how long downloading may take:
    FathomNet's throughput is not ours to control, and an unbounded fetch can eat
    a whole session before training starts. Whatever arrived by then is used.
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    todo = []
    ok, failed = [], []
    for im in coco["images"]:
        p = local_image_path(im, cache_dir)
        im["_local_path"] = str(p)
        if p.is_file() and p.stat().st_size > 0:
            ok.append(im["id"])
        else:
            us = image_urls(im)
            if us:
                todo.append((im["id"], us, p))
            else:
                failed.append(im["id"])

    if todo and verbose:
        n_multi = sum(1 for _, us, _ in todo if len(us) > 1)
        print(f"downloading {len(todo)} frames with {workers} workers "
              f"({n_multi} have a fallback URL"
              + (f", downscaling to {max_side}px" if max_side else "") + ") ...")

    def fetch(job):
        im_id, urls, path = job
        for url in urls:
            for attempt in range(2):
                try:
                    req = urllib.request.Request(
                        url, headers={"User-Agent": "fathomnet-pu-notebook/1.0"}
                    )
                    with urllib.request.urlopen(req, timeout=30) as r:
                        data = r.read()
                    if not data:
                        raise IOError("empty response")
                    save_frame(data, path, max_side)
                    return im_id, True
                except Exception:
                    # Only back off between retries of the same URL; a dead host
                    # should fall through to the next candidate immediately.
                    if attempt == 0:
                        time.sleep(1.0)
        return im_id, False

    if todo:
        from concurrent.futures import as_completed

        t_start = time.time()
        seen_ids = set()
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futures = [ex.submit(fetch, job) for job in todo]
            try:
                for n, fut in enumerate(as_completed(futures), 1):
                    im_id, good = fut.result()
                    seen_ids.add(im_id)
                    (ok if good else failed).append(im_id)
                    if verbose and n % 100 == 0:
                        rate = n / max(time.time() - t_start, 1e-6)
                        eta_min = (len(todo) - n) / max(rate, 1e-6) / 60.0
                        print(f"  {n}/{len(todo)}  {rate:.1f} img/s  "
                              f"eta {eta_min:.0f}m")
                    if deadline is not None and time.time() > deadline:
                        print(f"  >>> download budget spent after "
                              f"{n}/{len(todo)}; using what arrived")
                        break
            finally:
                for f in futures:
                    f.cancel()
        for im_id, _urls, _path in todo:
            if im_id not in seen_ids:
                failed.append(im_id)

    if verbose:
        print(f"available: {len(ok)}   unavailable: {len(failed)}")
    return set(ok), set(failed)
''')

# --------------------------------------------------------------------------- #
md(r"""
## 7. Stage 1 in action — audit, clean, split

The audit is the first real result of the pipeline. The statistic to look at is
**mean instances per image**: on natural underwater scenes a value near 1.0 is
itself evidence of incomplete labelling, and it is the number that should rise
after Stage 3 recovers missing positives.
""")

code(r'''
AUDIT = None
coco_clean = None
train_ids, val_ids = [], []

if raw_train is not None:
    AUDIT = audit_coco(raw_train)

    print("=" * 62)
    print("STAGE 1 AUDIT")
    print("=" * 62)
    for key in (
        "n_images", "n_annotations", "n_categories",
        "orphan_image_id", "unknown_category_id", "non_finite_bbox",
        "non_positive_bbox", "out_of_bounds_bbox", "tiny_bbox",
        "exact_duplicates", "near_duplicates", "bad_image_dims",
    ):
        v = AUDIT[key]
        print(f"{key:24s}: {v if isinstance(v, int) else len(v)}")
    print(f"{'images w/o annotations':24s}: {len(AUDIT['images_without_annotations'])}")
    print(f"{'empty categories':24s}: {AUDIT['categories_with_no_instances']}")
    print(f"\ninstances per image      : {AUDIT['instances_per_image']}")

    ipi = AUDIT["instances_per_image"]["mean"]
    print(
        f"\n>>> mean {ipi:.2f} instances/image."
        + (
            "  Consistent with heavily incomplete labelling."
            if ipi < 2.0
            else "  Higher than expected - check the PU assumption holds."
        )
    )

    print("\ntop 10 categories by instance count:")
    for name, n in list(AUDIT["category_counts"].items())[:10]:
        print(f"  {name:28s} {n:6d}")
    if AUDIT["category_counts"]:
        counts = list(AUDIT["category_counts"].values())
        print(f"\nimbalance ratio (max/min): {max(counts) / max(min(counts), 1):.1f}x")
''')

code(r'''
if raw_train is not None:
    coco_clean, clean_summary = clean_coco(
        raw_train, drop_tiny=CFG.drop_tiny_boxes, clip_to_image=True
    )
    print("cleaning summary:")
    for k in sorted(clean_summary):
        print(f"  {k:26s} {clean_summary[k]}")

    # Image-level split. Group-wise if a provenance field is available.
    gk = CFG.split_group_key or None
    if gk and gk not in coco_clean["images"][0]:
        print(f"\n(no '{gk}' field on image records - falling back to per-image split)")
        gk = None
    train_ids, val_ids = image_level_split(
        coco_clean, val_fraction=CFG.val_fraction, salt=CFG.split_salt, group_key=gk
    )
    assert not (set(train_ids) & set(val_ids)), "split leaked"
    print(f"\nsplit: {len(train_ids)} train / {len(val_ids)} val"
          f"  (group_key={gk!r})")
''')

code(r'''
# Visual check on the two distributions that shape every later decision.
if AUDIT is not None and AUDIT["category_counts"]:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))

    names = list(AUDIT["category_counts"])[:20]
    vals = [AUDIT["category_counts"][n] for n in names]
    axes[0].barh(range(len(names)), vals, color="#0f766e")
    axes[0].set_yticks(range(len(names)))
    axes[0].set_yticklabels(names, fontsize=8)
    axes[0].invert_yaxis()
    axes[0].set_title("Instances per category (top 20)")
    axes[0].set_xlabel("instances")

    per_image = collections.Counter(a["image_id"] for a in coco_clean["annotations"])
    vals = [per_image.get(im["id"], 0) for im in coco_clean["images"]]
    axes[1].hist(vals, bins=range(0, max(vals) + 2), color="#0891b2",
                 edgecolor="white", align="left")
    axes[1].axvline(np.mean(vals), color="#b91c1c", ls="--",
                    label=f"mean {np.mean(vals):.2f}")
    axes[1].set_title("Annotated instances per image")
    axes[1].set_xlabel("boxes in image")
    axes[1].legend()

    plt.tight_layout()
    plt.show()
''')

# --------------------------------------------------------------------------- #
md(r"""
## 8. Stage 2 — a single high-resolution RT-DETR detector

One detector, not an ensemble. The reason for high resolution is concrete: many
marine organisms occupy a small fraction of the frame, and aggressive resizing
erases the shape cues that separate a brittle star from background texture.

Boxes are stored **normalised** `cxcywh`, which makes them invariant to the
resize — the same target tensor is correct at every inference scale, and mapping
detections back to the original frame is a single multiply by `(W, H)`.
""")

code(r'''
class RunLog:
    """
    Structured progress log, printed and persisted as it goes.

    Two problems this solves, both hit on the first full run. Kaggle publishes a
    version's log only when it terminates, and truncates the log view, so
    thousands of per-step lines buried the results at the end. And a run killed
    at the wall left nothing behind at all.

    So: every event is one greppable [PROGRESS] line carrying JSON, and the whole
    history is rewritten to progress.json after each event. Even a run that dies
    mid-phase leaves a readable account of how far it got in the saved output.
    """

    def __init__(self, work_dir, t0=None, budget_s=None):
        self.path = Path(work_dir) / "progress.json"
        self.t0 = t0 if t0 is not None else time.time()
        self.budget_s = budget_s
        self.events = []

    def _clock(self):
        el = time.time() - self.t0
        d = {"elapsed_h": round(el / 3600.0, 3)}
        if self.budget_s:
            d["remaining_h"] = round(max(self.budget_s - el, 0) / 3600.0, 3)
        return d

    def event(self, kind, **fields):
        rec = {"kind": kind, **self._clock(), **fields}
        self.events.append(rec)
        print("[PROGRESS] " + json.dumps(rec), flush=True)
        try:
            self.path.write_text(json.dumps(self.events, indent=1, default=str))
        except Exception:
            pass          # never let bookkeeping take the run down
        return rec

    def phase(self, idx, name):
        c = self._clock()
        print("\n" + "=" * 66, flush=True)
        print(f"[{c['elapsed_h']:5.2f}h elapsed | "
              f"{c.get('remaining_h', 0):5.2f}h left]  PHASE {idx}/5  {name}",
              flush=True)
        print("=" * 66, flush=True)
        return self.event("phase_start", phase=idx, name=name)


def record_size(im: dict, pil=None):
    """
    Original frame size for an image record, as (width, height).

    Prefers the COCO record's declared dimensions -- annotations are expressed
    in that coordinate space, and cached files may have been downscaled. Falls
    back to the decoded image only when the record lacks usable values.
    """
    w, h = im.get("width"), im.get("height")
    if isinstance(w, (int, float)) and isinstance(h, (int, float)) and w > 0 and h > 0:
        return int(w), int(h)
    if pil is not None:
        return pil.size
    raise ValueError(f"image {im.get('id')} has no usable dimensions")


def build_dataset_classes():
    """Defined inside a function so the CPU-only stages never import torch."""
    import torch
    from torch.utils.data import Dataset
    from PIL import Image

    IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    class FathomNetDataset(Dataset):
        """
        COCO-format marine detection dataset.

        Emits normalised cxcywh boxes and contiguous class indices in
        [0, num_classes). `is_pseudo` is carried through so the loss can trust
        human annotations more than harvested ones.
        """

        def __init__(self, coco, image_ids, cat_id_to_index, img_size,
                     train=True, hflip_p=0.5):
            self.img_size = int(img_size)
            self.train = train
            self.hflip_p = hflip_p
            self.cat_id_to_index = dict(cat_id_to_index)

            keep = set(image_ids)
            self.images = [im for im in coco["images"] if im["id"] in keep]
            self.by_image = collections.defaultdict(list)
            for a in coco["annotations"]:
                if a["image_id"] in keep:
                    self.by_image[a["image_id"]].append(a)

        def __len__(self):
            return len(self.images)

        def _load(self, im):
            path = im.get("_local_path")
            if path and Path(path).is_file():
                return Image.open(path).convert("RGB")
            raise FileNotFoundError(f"image {im['id']} not on disk")

        def __getitem__(self, i):
            im = self.images[i]
            # Dimensions come from the COCO record, never from the stored file:
            # cached frames are downscaled on download, while bbox coordinates
            # remain in original-frame pixels. Normalising against the file's
            # size would silently scale every box.
            W, H = record_size(im)
            try:
                pil = self._load(im)
            except Exception:
                pil = Image.new("RGB", (self.img_size, self.img_size), (0, 0, 0))
                return self._pack(pil, [], im["id"], W, H, missing=True)

            anns = self.by_image.get(im["id"], [])
            return self._pack(pil, anns, im["id"], W, H, missing=False)

        def _pack(self, pil, anns, image_id, W, H, missing):
            boxes, labels, is_pseudo = [], [], []
            for a in anns:
                x, y, w, h = a["bbox"]
                if w <= 0 or h <= 0:
                    continue
                idx = self.cat_id_to_index.get(a["category_id"])
                if idx is None:
                    continue
                # -> normalised cxcywh, invariant to the resize below.
                boxes.append([(x + w / 2) / W, (y + h / 2) / H, w / W, h / H])
                labels.append(idx)
                is_pseudo.append(bool(a.get("is_pseudo", False)))

            pil = pil.resize((self.img_size, self.img_size), Image.BILINEAR)

            if self.train and boxes and random.random() < self.hflip_p:
                pil = pil.transpose(Image.FLIP_LEFT_RIGHT)
                boxes = [[1.0 - b[0], b[1], b[2], b[3]] for b in boxes]

            x = torch.from_numpy(np.asarray(pil, dtype=np.float32).copy())
            x = x.permute(2, 0, 1) / 255.0
            x = (x - IMAGENET_MEAN) / IMAGENET_STD

            t = {
                "labels": torch.as_tensor(labels, dtype=torch.long),
                "boxes": torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4),
                "is_pseudo": torch.as_tensor(is_pseudo, dtype=torch.bool),
                "image_id": int(image_id),
                "orig_size": (int(W), int(H)),
                "missing": bool(missing),
            }
            return x, t

    def collate(batch):
        xs = torch.stack([b[0] for b in batch])
        return xs, [b[1] for b in batch]

    return FathomNetDataset, collate


def build_category_maps(coco):
    """Contiguous 0..K-1 indices, with maps back to the competition's ids."""
    cats = sorted(coco["categories"], key=lambda c: c["id"])
    cat_id_to_index = {c["id"]: i for i, c in enumerate(cats)}
    index_to_cat_id = {i: c["id"] for i, c in enumerate(cats)}
    index_to_name = {i: c.get("name", str(c["id"])) for i, c in enumerate(cats)}
    return cat_id_to_index, index_to_cat_id, index_to_name
''')

code(r'''
def build_model(num_classes, model_name, device):
    """
    RT-DETR with a fresh classification head sized to the challenge's classes.

    `ignore_mismatched_sizes=True` is what allows the pretrained COCO head to be
    replaced. Requires internet on first call; the weights then sit in the HF cache.
    """
    import torch
    from transformers import RTDetrForObjectDetection

    model = RTDetrForObjectDetection.from_pretrained(
        model_name, num_labels=num_classes, ignore_mismatched_sizes=True
    )
    model.to(device)
    return model


def build_optimizer(model, cfg):
    """
    AdamW with a lower LR on the pretrained backbone.

    Returns (optimizer, max_lrs) so the scheduler's max_lr always lines up with
    the param groups that actually exist -- an empty backbone group (if the
    parameter naming ever changes) would otherwise silently desynchronise them.
    """
    import torch

    backbone, rest = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (backbone if "backbone" in n else rest).append(p)

    groups, max_lrs = [], []
    if rest:
        groups.append({"params": rest, "lr": cfg.lr})
        max_lrs.append(cfg.lr)
    if backbone:
        groups.append({"params": backbone, "lr": cfg.lr_backbone})
        max_lrs.append(cfg.lr_backbone)
    if not groups:
        raise RuntimeError("no trainable parameters found")

    print(f"param groups: head/neck {len(rest)} tensors, backbone {len(backbone)} tensors")
    return torch.optim.AdamW(groups, weight_decay=cfg.weight_decay), max_lrs


# --- AMP compatibility ---------------------------------------------------- #
# torch.cuda.amp.* is deprecated in favour of torch.amp.* from 2.4 onward.
# Kaggle's torch version moves; these shims work either way.

def make_grad_scaler(enabled):
    import torch
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=enabled)


def autocast_ctx(enabled):
    import torch
    try:
        return torch.amp.autocast("cuda", enabled=enabled)
    except (AttributeError, TypeError):
        return torch.cuda.amp.autocast(enabled=enabled)


def load_checkpoint(path, map_location):
    """torch.load flipped its weights_only default in 2.6; handle both."""
    import torch
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def train_one_run(coco, train_ids, val_ids, cfg, criterion_config,
                  ckpt_name, init_from=None, deadline=None, run_log=None):
    """
    Train the detector, checkpointing every epoch.

    Checkpointing every epoch is not optional on Kaggle: a session that hits the
    wall mid-run otherwise loses the entire GPU allocation it consumed.

    `deadline` is an absolute time.time() value. Training stops at the next step
    boundary once it passes, saves, and returns -- so a long run degrades into a
    shorter one instead of being killed with nothing on disk.
    """
    import torch
    from torch.utils.data import DataLoader

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    FathomNetDataset, collate = build_dataset_classes()
    cat_id_to_index, _, _ = build_category_maps(coco)

    ds_tr = FathomNetDataset(coco, train_ids, cat_id_to_index, cfg.img_size, train=True)
    ds_va = FathomNetDataset(coco, val_ids, cat_id_to_index, cfg.img_size, train=False)
    print(f"train images {len(ds_tr)} | val images {len(ds_va)}")

    dl_tr = DataLoader(ds_tr, batch_size=cfg.batch_size, shuffle=True,
                       num_workers=cfg.num_workers, collate_fn=collate,
                       drop_last=True, pin_memory=True)

    model = build_model(cfg.num_classes, cfg.model_name, device)
    if init_from and Path(init_from).is_file():
        model.load_state_dict(load_checkpoint(init_from, device)["model"])
        print(f"initialised from {init_from}")

    criterion = PUDetectionCriterion(criterion_config).to(device)
    optimizer, max_lrs = build_optimizer(model, cfg)
    steps_per_epoch = max(len(dl_tr) // cfg.grad_accum, 1)
    total_steps = max(steps_per_epoch * cfg.epochs, 1)

    # Warmup + cosine driven by ELAPSED TIME, not step count. A step-based
    # schedule has to be told up front how many steps there will be, and when a
    # wall-clock deadline ends training early the LR is left high -- the model
    # stops mid-anneal and loses most of the benefit of the last hour. Anchoring
    # to the deadline guarantees a full anneal whatever throughput turns out to be.
    t_start = time.time()
    span = max((deadline - t_start), 1.0) if deadline is not None else None
    warm = 0.05

    def lr_scale(_step):
        if span is None:
            frac = min(_step / float(total_steps), 1.0)
        else:
            frac = min(max((time.time() - t_start) / span, 0.0), 1.0)
        if frac < warm:
            return max(frac / warm, 1e-3)
        return 0.5 * (1.0 + math.cos(math.pi * (frac - warm) / (1.0 - warm)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_scale)
    use_amp = bool(cfg.amp and device.type == "cuda")
    scaler = make_grad_scaler(use_amp)

    ckpt_path = Path(cfg.work_dir) / ckpt_name
    history = []

    for epoch in range(cfg.epochs):
        model.train()
        running = collections.defaultdict(float)
        n_batches = 0
        t0 = time.time()
        optimizer.zero_grad(set_to_none=True)

        stopped_early = False
        for step, (pixel_values, targets) in enumerate(dl_tr):
            if deadline is not None and time.time() > deadline:
                print(f"  deadline reached at epoch {epoch} step {step}; "
                      "stopping cleanly")
                stopped_early = True
                break
            pixel_values = pixel_values.to(device, non_blocking=True)

            with autocast_ctx(use_amp):
                out = model(pixel_values=pixel_values)

            # The loss runs outside autocast: the Hungarian matching cost and the
            # focal/GIoU terms are numerically touchy in fp16, and this costs
            # nothing since the backbone forward is where the time goes.
            losses = criterion(
                {"logits": out.logits.float(), "pred_boxes": out.pred_boxes.float()},
                targets,
            )
            loss = losses["loss"] / cfg.grad_accum

            if not torch.isfinite(loss):
                print(f"  !! non-finite loss at step {step}, skipping batch")
                optimizer.zero_grad(set_to_none=True)
                continue

            try:
                scaler.scale(loss).backward()
            except torch.cuda.OutOfMemoryError:
                # One oversized frame should cost a batch, not the whole run.
                print(f"  !! CUDA OOM at step {step}, skipping batch")
                optimizer.zero_grad(set_to_none=True)
                del out, losses, loss
                torch.cuda.empty_cache()
                continue

            if (step + 1) % cfg.grad_accum == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.clip_grad)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()

            for k, v in losses.items():
                running[k] += float(v.detach().item() if hasattr(v, "item") else v)
            n_batches += 1

            if step % max(int(getattr(cfg, "log_every", 200)), 1) == 0:
                print(
                    f"  e{epoch} s{step}/{len(dl_tr)} "
                    f"loss {losses['loss'].item():.3f} "
                    f"cls {losses['loss_class'].item():.3f} "
                    f"bbox {losses['loss_bbox'].item():.3f} "
                    f"matched {int(losses['num_matched'].item())} "
                    f"bg_w {losses['mean_bg_weight'].item():.3f} "
                    f"softened {int(losses['n_softened'].item())} "
                    f"tau {losses['tau'].item():.3f} "
                    f"maxobj {losses['max_objectness'].item():.3f} "
                    f"lr {optimizer.param_groups[0]['lr']:.2e}"
                )

        epoch_stats = {k: v / max(n_batches, 1) for k, v in running.items()}
        epoch_stats["epoch"] = epoch
        epoch_stats["seconds"] = round(time.time() - t0, 1)
        # Throughput is the number that says whether the decode fix worked.
        seen = n_batches * cfg.batch_size
        epoch_stats["imgs_per_s"] = round(seen / max(epoch_stats["seconds"], 1e-6), 2)
        history.append(epoch_stats)

        msg = (f"epoch {epoch} done in {epoch_stats['seconds']}s  "
               f"mean loss {epoch_stats['loss']:.4f}  "
               f"{epoch_stats['imgs_per_s']:.1f} img/s")
        if deadline is not None:
            left = deadline - time.time()
            eta = left / max(epoch_stats["seconds"], 1e-6)
            msg += f"  ~{eta:.1f} epochs left in this phase"
        print(msg, flush=True)
        if run_log is not None:
            run_log.event("epoch", ckpt=ckpt_name, epoch=epoch,
                          loss=round(epoch_stats["loss"], 4),
                          imgs_per_s=epoch_stats["imgs_per_s"],
                          seconds=epoch_stats["seconds"],
                          lr=optimizer.param_groups[0]["lr"])

        torch.save(
            {
                "model": model.state_dict(),
                "epoch": epoch,
                "cfg": asdict(cfg),
                "criterion_config": asdict(criterion_config),
                "history": history,
            },
            ckpt_path,
        )
        print(f"  checkpoint -> {ckpt_path}")

        if stopped_early:
            break

    return model, history, ckpt_path
''')

# --------------------------------------------------------------------------- #
md(r"""
## 9. Stage 5 — two-scale inference and Soft-NMS

Two passes instead of a large test-time-augmentation ensemble: the
medium-resolution pass supplies scene context, the high-resolution pass recovers
small organisms. Because the model emits normalised boxes, both passes denormalise
into the original frame directly, and the merge happens in original-frame
coordinates — which is also the frame the metric is computed in.

Merging uses **per-class** Soft-NMS. Cross-class suppression would be wrong here:
a shrimp on a sponge produces two heavily overlapping boxes of different classes
and both are correct. And Soft-NMS rather than hard NMS because dense marine
scenes contain genuinely overlapping organisms — decaying a neighbour's score
degrades gracefully where deleting it loses a real detection.
""")

code(r'''
def predict_two_scale(model, coco, image_ids, cfg, index_to_cat_id,
                      views_for_consistency=False):
    """
    Run the detector at each scale in cfg.infer_scales and merge.

    Returns image_id -> {"boxes" (xyxy, ORIGINAL frame), "scores", "labels"},
    where labels are competition category_ids. With
    views_for_consistency=True it also returns the per-view detections, which is
    what Stage 3 needs to test whether a candidate survives a transformation.
    """
    import torch
    from PIL import Image

    device = next(model.parameters()).device
    model.eval()

    IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    by_id = {im["id"]: im for im in coco["images"]}
    out_final, out_views = {}, {}

    weights = list(cfg.infer_scale_weights)
    if len(weights) < len(cfg.infer_scales):
        weights += [1.0] * (len(cfg.infer_scales) - len(weights))

    def prep(pil, size):
        x = pil.resize((size, size), Image.BILINEAR)
        x = torch.from_numpy(np.asarray(x, dtype=np.float32).copy())
        x = x.permute(2, 0, 1) / 255.0
        return ((x - IMAGENET_MEAN) / IMAGENET_STD).unsqueeze(0)

    @torch.no_grad()
    def run(pil, size, W, H, flip=False):
        if flip:
            pil = pil.transpose(Image.FLIP_LEFT_RIGHT)
        pv = prep(pil, size).to(device)
        with autocast_ctx(bool(cfg.amp and device.type == "cuda")):
            o = model(pixel_values=pv)
        prob = o.logits[0].sigmoid().float().cpu().numpy()        # (Q, C)
        boxes = o.pred_boxes[0].float().cpu().numpy()             # (Q, 4) cxcywh norm

        # Flatten query x class into independent candidates, as DETR-family
        # post-processing does: one query may legitimately support two classes.
        q_idx, c_idx = np.where(prob >= cfg.score_threshold)
        if len(q_idx) == 0:
            return (np.zeros((0, 4)), np.zeros(0), np.zeros(0, dtype=int))

        scores = prob[q_idx, c_idx]
        b = boxes[q_idx]
        cx, cy, bw, bh = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
        if flip:
            cx = 1.0 - cx
        xyxy = np.stack(
            [(cx - bw / 2) * W, (cy - bh / 2) * H,
             (cx + bw / 2) * W, (cy + bh / 2) * H], axis=1
        )
        return xyxy, scores, c_idx

    for n, im_id in enumerate(image_ids, 1):
        im = by_id[im_id]
        path = im.get("_local_path")
        if not path or not Path(path).is_file():
            continue
        try:
            pil = Image.open(path).convert("RGB")
        except Exception:
            continue
        # Predictions must land in the original coordinate frame, which is what
        # the metric and the submission are expressed in -- not the frame of a
        # downscaled cache file.
        W, H = record_size(im, pil)

        passes, views = [], []
        for size, w in zip(cfg.infer_scales, weights):
            xyxy, scores, labels = run(pil, size, W, H)
            passes.append({"boxes": xyxy, "scores": scores, "labels": labels,
                           "size": (W, H), "weight": float(w)})
            views.append({"boxes": xyxy, "scores": scores, "labels": labels})

            # Flip TTA on the final predictions only. During harvesting the
            # flipped pass is the consistency evidence and must stay a separate
            # view, not be merged into the same detection set.
            if getattr(cfg, "infer_flip_tta", False) and not views_for_consistency:
                fb, fs, fl = run(pil, size, W, H, flip=True)
                passes.append({"boxes": fb, "scores": fs, "labels": fl,
                               "size": (W, H), "weight": float(w)})

        if views_for_consistency:
            # One extra flipped view at the largest scale: a texture artifact
            # rarely reappears at the same place and class under a flip.
            xyxy, scores, labels = run(pil, max(cfg.infer_scales), W, H, flip=True)
            views.append({"boxes": xyxy, "scores": scores, "labels": labels})

        merged = merge_multiscale(
            passes,
            original_size=(W, H),
            method=cfg.softnms_method,
            sigma=cfg.softnms_sigma,
            iou_threshold=cfg.softnms_iou_threshold,
            score_threshold=cfg.score_threshold,
        )
        if len(merged["scores"]) > cfg.max_dets_per_image:
            keep = np.argsort(-merged["scores"])[: cfg.max_dets_per_image]
            merged = {k: merged[k][keep] for k in ("boxes", "scores", "labels")}

        # Map contiguous indices back to competition category_ids.
        merged["labels"] = np.array(
            [index_to_cat_id[int(i)] for i in merged["labels"]], dtype=np.int64
        )
        out_final[im_id] = merged
        if views_for_consistency:
            out_views[im_id] = views

        if n % 100 == 0:
            print(f"  inferred {n}/{len(image_ids)}")

    return (out_final, out_views) if views_for_consistency else out_final
''')

# --------------------------------------------------------------------------- #
md(r"""
## 10. Stage 3 — conservative pseudo-label recovery

Self-training, kept deliberately timid. A pseudo-label earns its place only if it
is more likely to *correct* missing supervision than to *inject* a new error, so
every step in the funnel removes candidates and none adds any:

```
confidence filter  ->  IoU dedup vs GT  ->  cross-view consistency  ->  per-image cap
```

The per-stage survivor counts are the thing to watch. If `after_consistency` is
close to `after_gt_dedup`, the consistency check is not discriminating and the
thresholds need tightening; if it collapses to near zero, nothing is being
recovered and the whole stage is a no-op.
""")

code(r'''
def harvest_over_dataset(model, coco, image_ids, cfg, index_to_cat_id):
    """Run the Stage 3 funnel across a set of images and aggregate the counts."""
    cat_index_of = {v: k for k, v in index_to_cat_id.items()}

    _, views_by_image = predict_two_scale(
        model, coco, image_ids, cfg, index_to_cat_id, views_for_consistency=True
    )

    gt_by_image = collections.defaultdict(list)
    for a in coco["annotations"]:
        x, y, w, h = a["bbox"]
        gt_by_image[a["image_id"]].append(
            ([x, y, x + w, y + h], cat_index_of.get(a["category_id"], -1))
        )

    harvest, totals = {}, collections.Counter()
    score_maxima = []
    for im_id, views in views_by_image.items():
        gt = gt_by_image.get(im_id, [])
        gt_boxes = np.array([g[0] for g in gt], dtype=float).reshape(-1, 4)
        gt_labels = np.array([g[1] for g in gt], dtype=int)

        res = harvest_pseudo_labels(
            views,
            gt_boxes=gt_boxes,
            gt_labels=gt_labels,
            score_threshold=cfg.pseudo_score_threshold,
            gt_iou_threshold=cfg.pseudo_gt_iou_threshold,
            consistency_iou=cfg.pseudo_consistency_iou,
            min_views=cfg.pseudo_min_views,
            max_per_image=cfg.pseudo_max_per_image,
            pre_top_k=cfg.pseudo_pre_top_k,
        )
        for k, v in res["counts"].items():
            totals[k] += v
        score_maxima.append(res["stats"]["max_score"])
        if len(res["boxes"]):
            # Back to competition category_ids for merging into the COCO file.
            res["labels"] = np.array(
                [index_to_cat_id[int(i)] for i in res["labels"]], dtype=np.int64
            )
            harvest[im_id] = res

    print("\nStage 3 funnel across", len(views_by_image), "images:")
    for k in ("candidates", "after_top_k", "after_confidence", "after_gt_dedup",
              "after_consistency", "kept"):
        print(f"  {k:20s} {totals[k]}")

    # The score distribution is the evidence the thresholds should be set from.
    # Without it, a stage that rejects everything looks identical to one with
    # nothing to find.
    if score_maxima:
        sm = np.array(score_maxima, dtype=float)
        print(f"\n  per-image best detection score: "
              f"median {np.median(sm):.3f}  p90 {np.percentile(sm, 90):.3f}  "
              f"max {sm.max():.3f}")
        print(f"  (confidence floor is {cfg.pseudo_score_threshold:.2f}; "
              f"{100.0 * (sm >= cfg.pseudo_score_threshold).mean():.0f}% of images "
              "have at least one candidate above it)")
    if totals["after_gt_dedup"]:
        survival = totals["after_consistency"] / totals["after_gt_dedup"]
        print(f"\n  consistency survival rate: {survival:.1%}")
        if survival > 0.9:
            print("  >>> consistency check is barely filtering - raise consistency_iou")
        elif survival < 0.05:
            print("  >>> almost nothing survives - lower the thresholds or this stage is a no-op")
    return harvest, dict(totals)
''')

# --------------------------------------------------------------------------- #
md(r"""
## 11. Evaluation — and why one mAP number is not enough

PU learning breaks validation too. If a validation frame contains an unlabeled
fish and the model finds it, the evaluator scores that as a false positive. So a
model that gets **better** at discovering unlabeled organisms can look **worse**
on an incompletely annotated split.

The consequence is practical: mAP computed on the incomplete training split is a
*relative* signal at best, and the diagnostics below matter as much as the number.
A rising count of high-confidence "false positives" alongside stable recall on
confidently-labeled classes is the signature of the PU loss working, not failing.
""")

code(r'''
def evaluate_map(coco_gt_dict, predictions, index_to_name=None, verbose=True):
    """COCO mAP via pycocotools, plus the PU diagnostics worth reading beside it."""
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    import tempfile

    gt = {
        "images": coco_gt_dict["images"],
        "annotations": [
            dict(a, iscrowd=a.get("iscrowd", 0), area=a.get("area", a["bbox"][2] * a["bbox"][3]))
            for a in coco_gt_dict["annotations"]
        ],
        "categories": coco_gt_dict["categories"],
    }
    for i, a in enumerate(gt["annotations"], 1):
        a.setdefault("id", i)

    dets = []
    for im_id, p in predictions.items():
        for bb, sc, lab in zip(xyxy_to_xywh(p["boxes"]), p["scores"], p["labels"]):
            dets.append({
                "image_id": int(im_id), "category_id": int(lab),
                "bbox": [float(v) for v in bb], "score": float(sc),
            })

    if not dets:
        print("no detections - nothing to evaluate")
        return {"mAP": 0.0, "mAP50": 0.0, "n_dets": 0}

    with tempfile.TemporaryDirectory() as td:
        gt_path = Path(td) / "gt.json"
        dt_path = Path(td) / "dt.json"
        gt_path.write_text(json.dumps(gt))
        dt_path.write_text(json.dumps(dets))

        coco_gt = COCO(str(gt_path))
        coco_dt = coco_gt.loadRes(str(dt_path))
        e = COCOeval(coco_gt, coco_dt, "bbox")
        e.params.imgIds = sorted(predictions)
        e.evaluate(); e.accumulate()
        if verbose:
            e.summarize()
        else:
            import io, contextlib
            with contextlib.redirect_stdout(io.StringIO()):
                e.summarize()

    stats = {
        "mAP": float(e.stats[0]), "mAP50": float(e.stats[1]), "mAP75": float(e.stats[2]),
        "mAP_small": float(e.stats[3]), "mAP_medium": float(e.stats[4]),
        "mAP_large": float(e.stats[5]), "n_dets": len(dets),
    }

    # --- PU diagnostics ---------------------------------------------------
    per_class = collections.Counter(d["category_id"] for d in dets)
    scores = np.array([d["score"] for d in dets])
    stats["dets_per_image"] = len(dets) / max(len(predictions), 1)
    stats["n_high_conf"] = int((scores >= 0.5).sum())
    stats["mean_score"] = float(scores.mean())

    if verbose:
        print("\n--- PU diagnostics ---")
        print(f"detections/image      : {stats['dets_per_image']:.1f}")
        print(f"high-confidence (>=.5): {stats['n_high_conf']}")
        print(f"mean score            : {stats['mean_score']:.3f}")
        print(f"distinct classes used : {len(per_class)}")
        if index_to_name:
            name_of = {}
            for c in coco_gt_dict["categories"]:
                name_of[c["id"]] = c.get("name", str(c["id"]))
            print("top predicted classes :")
            for cid, n in per_class.most_common(8):
                print(f"   {name_of.get(cid, cid):26s} {n}")
        print(
            "\nReminder: on an incompletely annotated split a rise in "
            "'false positives'\nmay be recovered unlabeled organisms. Compare "
            "against the fully\nannotated evaluation set before concluding "
            "precision got worse."
        )
    return stats
''')

# --------------------------------------------------------------------------- #
md(r"""
## 12. Verification

Everything in §3–§4 is checked here against hand-computed expected values, and the
PU criterion against its own properties. This is the `"verify"` stage and it runs
on CPU in seconds. Run it before any GPU stage.
""")

code(r'''
def run_numpy_tests(verbose=True):
    """Self-contained checks on the NumPy ops. Returns (n_pass, failures)."""
    results, failures = [], []

    def ck(name, cond, detail=""):
        results.append(bool(cond))
        if not cond:
            failures.append(f"{name} {detail}")
        if verbose and not cond:
            print(f"  FAIL  {name}  {detail}")

    def close(a, b, tol=1e-9):
        return abs(float(a) - float(b)) <= tol

    # --- boxes and IoU ---------------------------------------------------
    ck("xywh->xyxy", np.allclose(xywh_to_xyxy([[10, 20, 30, 40]]), [[10, 20, 40, 60]]))
    ck("xyxy->xywh", np.allclose(xyxy_to_xywh([[10, 20, 40, 60]]), [[10, 20, 30, 40]]))
    ck("IoU quarter overlap = 25/175",
       close(iou_matrix([[0, 0, 10, 10]], [[5, 5, 15, 15]])[0, 0], 25 / 175))
    ck("IoU half overlap = 1/3",
       close(iou_matrix([[0, 0, 10, 10]], [[5, 0, 15, 10]])[0, 0], 1 / 3))
    ck("IoU identical = 1", close(iou_matrix([[0, 0, 5, 5]], [[0, 0, 5, 5]])[0, 0], 1.0))
    ck("IoU disjoint = 0", close(iou_matrix([[0, 0, 5, 5]], [[9, 9, 14, 14]])[0, 0], 0.0))
    ck("IoU zero-area is 0 not NaN",
       close(iou_matrix([[5, 5, 5, 5]], [[0, 0, 10, 10]])[0, 0], 0.0))

    # --- Soft-NMS: exact expected decays --------------------------------
    _, s, _ = soft_nms([[0, 0, 10, 10], [0, 0, 10, 10]], [1.0, 1.0],
                       method="gaussian", sigma=0.5)
    ck("gaussian keeps both identical boxes", len(s) == 2, f"{len(s)}")
    ck("gaussian identical decay == exp(-2)", close(s[1], math.exp(-2.0)), f"{s[1]}")

    _, s, _ = soft_nms([[0, 0, 10, 10], [5, 0, 15, 10]], [0.9, 0.8],
                       method="gaussian", sigma=0.5)
    ck("gaussian half-overlap decay",
       close(s[1], 0.8 * math.exp(-((1 / 3) ** 2) / 0.5)), f"{s[1]}")

    _, s, _ = soft_nms([[0, 0, 10, 10], [0, 0, 10, 10]], [1.0, 1.0],
                       method="linear", iou_threshold=0.3)
    ck("linear drops exact duplicate", len(s) == 1, f"{len(s)}")

    _, ss, _ = soft_nms([[0, 0, 10, 10], [2, 2, 12, 12]], [0.9, 0.85], method="gaussian")
    _, sh, _ = soft_nms([[0, 0, 10, 10], [2, 2, 12, 12]], [0.9, 0.85],
                        method="hard", iou_threshold=0.3)
    ck("soft keeps dense neighbour that hard deletes",
       len(ss) == 2 and len(sh) == 1, f"soft={len(ss)} hard={len(sh)}")

    _, s, _ = soft_nms([[0, 0, 5, 5], [50, 50, 55, 55]], [0.7, 0.6])
    ck("disjoint boxes untouched", len(s) == 2 and close(s[0], 0.7) and close(s[1], 0.6))

    _, s, l = soft_nms_per_class([[0, 0, 10, 10], [0, 0, 10, 10]], [0.9, 0.85], [1, 2])
    ck("per-class does not suppress across classes",
       len(s) == 2 and close(s[0], 0.9) and close(s[1], 0.85), f"{s}")

    # --- coordinate mapping ---------------------------------------------
    ck("scale_boxes 3x",
       np.allclose(scale_boxes([[10, 20, 30, 40]], (640, 360), (1920, 1080)),
                   [[30, 60, 90, 120]]))
    ck("scale_boxes anisotropic",
       np.allclose(scale_boxes([[10, 10, 20, 20]], (100, 200), (300, 400)),
                   [[30, 20, 60, 40]]))
    ck("clip to frame", np.allclose(clip_boxes([[-5, -5, 50, 50]], (40, 30)),
                                    [[0, 0, 40, 30]]))

    m = merge_multiscale(
        [{"boxes": [[10, 10, 20, 20]], "scores": [0.9], "labels": [1], "size": (100, 100)},
         {"boxes": [[20, 20, 40, 40]], "scores": [0.8], "labels": [1], "size": (200, 200)}],
        original_size=(100, 100), method="gaussian", sigma=0.5)
    ck("two passes agreeing map onto the same box",
       np.allclose(m["boxes"][0], [10, 10, 20, 20]), f"{m['boxes']}")
    ck("agreeing duplicate decayed not deleted",
       len(m["scores"]) == 2 and close(m["scores"][1], 0.8 * math.exp(-2.0)), f"{m['scores']}")
    ck("out-of-frame box dropped",
       len(merge_multiscale(
           [{"boxes": [[200, 200, 300, 300]], "scores": [0.9], "labels": [1],
             "size": (100, 100)}], original_size=(100, 100))["scores"]) == 0)

    # --- PU weight -------------------------------------------------------
    w = pu_background_weight(np.array([0.5]), w_min=0.25, w_max=1.0,
                             tau=0.5, temperature=0.1)
    ck("w at tau is midpoint 0.625", close(w[0], 0.625), f"{w[0]}")
    w = pu_background_weight(np.linspace(0, 1, 11), w_min=0.25, w_max=1.0,
                             tau=0.5, temperature=0.1)
    ck("w strictly decreasing", bool(np.all(np.diff(w) < 0)))
    ck("w bounded", bool(np.all(w >= 0.25) and np.all(w <= 1.0)))
    ck("w never zero", bool(np.all(w > 0)))
    ck("w_min==w_max is uniform",
       bool(np.allclose(pu_background_weight(np.array([0.0, 0.5, 1.0]),
                                             w_min=1.0, w_max=1.0), 1.0)))

    # --- pseudo-label funnel --------------------------------------------
    gt_b, gt_l = np.array([[0, 0, 10, 10]], float), np.array([1])
    ck("dedup rejects same-class duplicate",
       not bool(dedup_against_gt([[0, 0, 10, 10]], [1], gt_b, gt_l)[0]))
    ck("dedup keeps other class at same place",
       bool(dedup_against_gt([[0, 0, 10, 10]], [2], gt_b, gt_l)[0]))
    ck("dedup keeps distant same-class box",
       bool(dedup_against_gt([[90, 90, 100, 100]], [1], gt_b, gt_l)[0]))
    ck("consistency needs class agreement",
       not bool(consistency_filter(
           [{"boxes": [[0, 0, 10, 10]], "labels": [1]},
            {"boxes": [[0, 0, 10, 10]], "labels": [2]}], min_views=2)[0]))

    res = harvest_pseudo_labels(
        [{"boxes": [[0, 0, 10, 10], [50, 50, 60, 60], [80, 80, 90, 90], [20, 20, 30, 30]],
          "scores": [0.9, 0.8, 0.3, 0.7], "labels": [1, 1, 1, 1]},
         {"boxes": [[50, 50, 60, 60]], "scores": [0.75], "labels": [1]}],
        gt_boxes=gt_b, gt_labels=gt_l, score_threshold=0.6,
        gt_iou_threshold=0.5, consistency_iou=0.6, min_views=2)
    c = res["counts"]
    ck("funnel counts 4/3/2/1",
       (c["candidates"], c["after_confidence"], c["after_gt_dedup"],
        c["after_consistency"]) == (4, 3, 2, 1), f"{c}")
    ck("funnel kept the right box",
       len(res["boxes"]) == 1 and np.allclose(res["boxes"][0], [50, 50, 60, 60]),
       f"{res['boxes']}")
    ck("funnel monotonically non-increasing",
       c["candidates"] >= c["after_confidence"] >= c["after_gt_dedup"]
       >= c["after_consistency"] >= c["kept"], f"{c}")

    # --- audit / clean / split ------------------------------------------
    toy = {
        "images": [{"id": 1, "width": 100, "height": 100},
                   {"id": 2, "width": 100, "height": 100}],
        "categories": [{"id": 1, "name": "crab"}, {"id": 2, "name": "urchin"}],
        "annotations": [
            {"id": 1, "image_id": 1, "category_id": 1, "bbox": [10, 10, 20, 20]},
            {"id": 2, "image_id": 1, "category_id": 1, "bbox": [10, 10, 20, 20]},
            {"id": 3, "image_id": 2, "category_id": 2, "bbox": [0, 0, 0, 10]},
            {"id": 4, "image_id": 2, "category_id": 2, "bbox": [90, 90, 50, 50]},
            {"id": 5, "image_id": 999, "category_id": 1, "bbox": [0, 0, 5, 5]},
            {"id": 6, "image_id": 1, "category_id": 77, "bbox": [0, 0, 5, 5]},
        ],
    }
    r = audit_coco(toy)
    ck("audit finds orphan", 5 in r["orphan_image_id"], f"{r['orphan_image_id']}")
    ck("audit finds unknown class", 6 in r["unknown_category_id"])
    ck("audit finds zero-extent", 3 in r["non_positive_bbox"])
    ck("audit finds out-of-bounds", 4 in r["out_of_bounds_bbox"])
    ck("audit finds exact duplicate", 2 in r["exact_duplicates"])

    n_before = len(toy["annotations"])
    cleaned, summ = clean_coco(toy)
    ck("clean does not mutate input", len(toy["annotations"]) == n_before)
    ck("clean drops orphan", summ.get("dropped_orphan") == 1, f"{summ}")
    ck("clean drops duplicate", summ.get("dropped_duplicate") == 1, f"{summ}")
    ck("clean clips out-of-bounds", summ.get("clipped") == 1, f"{summ}")
    ck("clean is idempotent", clean_coco(cleaned)[1]["kept"] == summ["kept"])

    big = {"images": [{"id": i, "width": 10, "height": 10} for i in range(500)],
           "categories": [{"id": 1, "name": "c"}], "annotations": []}
    tr, va = image_level_split(big, val_fraction=0.2)
    ck("split complete", len(tr) + len(va) == 500)
    ck("split disjoint", not (set(tr) & set(va)))
    ck("split fraction sane", 0.15 < len(va) / 500 < 0.26, f"{len(va)/500}")
    ck("split deterministic", image_level_split(big, val_fraction=0.2) == (tr, va))

    # --- submission ------------------------------------------------------
    rows = build_submission({7: {"boxes": [[10, 20, 40, 60]], "scores": [0.87], "labels": [3]}})
    ck("submission column order exact",
       list(rows[0].keys()) == SUBMISSION_COLUMNS, f"{list(rows[0].keys())}")
    ck("submission emits xywh",
       (rows[0]["bbox_width"], rows[0]["bbox_height"]) == (30.0, 40.0), f"{rows[0]}")
    ck("submission x/y top-left", (rows[0]["bbox_x"], rows[0]["bbox_y"]) == (10.0, 20.0))
    rows = build_submission({1: {"boxes": [[0, 0, 5, 5], [1, 1, 6, 6]],
                                 "scores": [1.5, -0.2], "labels": [1, 2]}})
    ck("submission clamps scores", rows[0]["score"] == 1.0 and rows[1]["score"] == 0.0)
    ck("submission ids unique", len({r["annotation_id"] for r in rows}) == len(rows))

    n_pass = sum(results)
    print(f"  numpy ops: {n_pass}/{len(results)} passed")
    return n_pass, failures


print("=" * 62)
print("VERIFICATION")
print("=" * 62)
n_pass, fails = run_numpy_tests()
if fails:
    print("\nFAILURES:")
    for f in fails:
        print("  -", f)
    raise AssertionError(f"{len(fails)} numpy op checks failed")

if ENV.get("torch", "MISSING").startswith("MISSING"):
    print("\n  (torch missing - skipping PU criterion self-test)")
else:
    print()
    selftest_pu_criterion(verbose=False)

print("\nAll verification passed.")
''')

# --------------------------------------------------------------------------- #
md(r"""
### 12b. End-to-end rehearsal on synthetic data

A synthetic dataset where we **know** which positives were withheld, so the claim
"Stage 3 recovers missing positives without inventing new ones" becomes a
measurement rather than an assertion. This is the only place in the notebook where
pseudo-label precision can actually be computed, because it is the only place the
withheld labels are known.
""")

code(r'''
def synthetic_rehearsal(n_images=60, seed=0, verbose=True):
    """
    Build a PU dataset by construction: generate k true objects per frame, then
    annotate only the first. Everything else is a deliberately withheld positive.
    """
    rng = np.random.default_rng(seed)
    images, annotations = [], []
    true_objects, withheld = {}, {}
    ann_id = 1

    for im_id in range(1, n_images + 1):
        images.append({"id": im_id, "width": 640, "height": 480,
                       "file_name": f"{im_id}.png"})
        boxes = []
        for _ in range(int(rng.integers(2, 5))):
            x, y = float(rng.integers(0, 540)), float(rng.integers(0, 380))
            boxes.append([x, y, x + 80, y + 80])
        true_objects[im_id] = boxes
        withheld[im_id] = boxes[1:]
        bb = boxes[0]
        annotations.append({"id": ann_id, "image_id": im_id, "category_id": 1,
                            "bbox": [bb[0], bb[1], bb[2] - bb[0], bb[3] - bb[1]],
                            "area": 6400.0, "iscrowd": 0})
        ann_id += 1

    synth = {"images": images, "categories": [{"id": 1, "name": "organism"}],
             "annotations": annotations}

    rep = audit_coco(synth)
    assert abs(rep["instances_per_image"]["mean"] - 1.0) < 1e-9
    if verbose:
        print(f"synthetic: {n_images} images, "
              f"{rep['instances_per_image']['mean']:.2f} labelled instances/image")
        print(f"withheld positives: {sum(len(v) for v in withheld.values())}")

    tr_ids, va_ids = image_level_split(synth, val_fraction=0.2)

    def detector(im_id, jitter):
        """Finds every true object with jitter, plus one wandering texture artifact."""
        boxes, scores = [], []
        for bb in true_objects[im_id]:
            j = rng.normal(0, jitter, 4)
            boxes.append([bb[0] + j[0], bb[1] + j[1], bb[2] + j[2], bb[3] + j[3]])
            scores.append(float(np.clip(rng.normal(0.85, 0.05), 0, 1)))
        fx = float(rng.integers(0, 500))
        boxes.append([fx, 400.0, fx + 60.0, 460.0])
        scores.append(float(np.clip(rng.normal(0.70, 0.05), 0, 1)))
        return {"boxes": np.array(boxes, float), "scores": np.array(scores, float),
                "labels": np.ones(len(boxes), int)}

    gt_by_image = collections.defaultdict(list)
    for a in synth["annotations"]:
        x, y, w, h = a["bbox"]
        gt_by_image[a["image_id"]].append([x, y, x + w, y + h])

    harvest, kept, recovered, spurious = {}, 0, 0, 0
    for im_id in tr_ids:
        res = harvest_pseudo_labels(
            [detector(im_id, 2.0), detector(im_id, 3.0)],
            gt_boxes=np.array(gt_by_image[im_id], float),
            gt_labels=np.ones(len(gt_by_image[im_id]), int),
            score_threshold=0.6, gt_iou_threshold=0.5,
            consistency_iou=0.5, min_views=2, max_per_image=5)
        harvest[im_id] = res
        kept += res["counts"]["kept"]
        if len(res["boxes"]) and withheld[im_id]:
            hit = iou_matrix(res["boxes"], np.array(withheld[im_id], float)).max(axis=1) >= 0.5
            recovered += int(hit.sum()); spurious += int((~hit).sum())
        else:
            spurious += len(res["boxes"])

    precision = recovered / max(kept, 1)
    total_withheld = sum(len(withheld[i]) for i in tr_ids)
    recall = recovered / max(total_withheld, 1)

    merged, added = merge_pseudo_labels(synth, harvest)
    rep2 = audit_coco(merged)

    if verbose:
        print(f"\npseudo-labels kept    : {kept}")
        print(f"  recovered withheld  : {recovered}")
        print(f"  spurious            : {spurious}")
        print(f"  precision           : {precision:.1%}")
        print(f"  recall of withheld  : {recall:.1%}")
        print(f"\ninstances/image {rep['instances_per_image']['mean']:.2f}"
              f" -> {rep2['instances_per_image']['mean']:.2f} after recovery")

    assert precision > 0.8, f"pseudo-label precision too low: {precision:.3f}"
    assert recovered > 0, "recovered nothing"
    assert rep2["instances_per_image"]["mean"] > rep["instances_per_image"]["mean"]
    assert rep2["non_positive_bbox"] == [] and rep2["orphan_image_id"] == []

    # Two-scale inference + submission on the held-out split.
    preds = {}
    for im_id in va_ids:
        med, hi = detector(im_id, 3.0), detector(im_id, 1.0)
        preds[im_id] = merge_multiscale(
            [{"boxes": med["boxes"], "scores": med["scores"], "labels": med["labels"],
              "size": (640, 480)},
             {"boxes": hi["boxes"], "scores": hi["scores"], "labels": hi["labels"],
              "size": (640, 480)}],
            original_size=(640, 480), method="gaussian", sigma=0.5)

    for p in preds.values():
        b = p["boxes"]
        assert np.all(b[:, 0] >= 0) and np.all(b[:, 1] >= 0)
        assert np.all(b[:, 2] <= 640 + 1e-6) and np.all(b[:, 3] <= 480 + 1e-6)
        assert np.all(b[:, 2] > b[:, 0]) and np.all(b[:, 3] > b[:, 1])

    rows = build_submission(preds)
    assert rows and all(list(r.keys()) == SUBMISSION_COLUMNS for r in rows)
    assert all(0.0 <= r["score"] <= 1.0 for r in rows)
    assert len({r["annotation_id"] for r in rows}) == len(rows)
    if verbose:
        print(f"\nsubmission rows from held-out split: {len(rows)}  (schema OK)")
        print("\nrehearsal passed.")
    return {"precision": precision, "recall": recall, "kept": kept, "rows": len(rows)}


REHEARSAL = synthetic_rehearsal()
''')

# --------------------------------------------------------------------------- #
md(r"""
## 12c. Running everything in one session

Kaggle gives a generous weekly GPU quota but caps a single run at roughly nine
hours, and — more importantly — **every commit starts a fresh container**. Nothing
written to `work_dir` by one version is visible to the next, so chaining the
stages across commits means re-wiring each version's output back in as an input.

Running all five stages inside one session avoids that entirely: the checkpoints
and the pseudo-labelled COCO file stay on disk between phases.

The risk is the opposite one — being killed at the wall with nothing to show. So
the run is budgeted. Time is split across the training phases, `reserve_hours` is
held back, and **inference and the submission always run**, on whatever checkpoint
exists. A truncated model that submits beats a fully-trained one that never wrote
its predictions. Harvest and the PU retrain are individually fault-tolerant for
the same reason: if either fails, the run falls back to the baseline checkpoint
and still produces a submission.
""")

code(r'''
def run_full_pipeline(coco_clean, test_json_path, train_ids, val_ids, cfg,
                      index_to_cat_id, index_to_name, prepare_images):
    """
    Execute every stage in one session under a wall-clock budget.

    `prepare_images(ids, coco, cfg)` resolves imagery and returns the ids that
    are actually on disk. Returns a dict summarising what each phase produced.
    """
    import csv
    import torch

    t0 = time.time()
    budget = float(cfg.time_budget_hours) * 3600.0
    reserve = min(float(cfg.reserve_hours) * 3600.0, budget * 0.5)
    train_budget = max(budget - reserve, 60.0)
    work = Path(cfg.work_dir)
    summary = {"phases": {}, "submission": None}

    run_log = RunLog(work, t0=t0, budget_s=budget)
    run_log.event("run_start", budget_h=cfg.time_budget_hours,
                  reserve_h=cfg.reserve_hours, img_size=cfg.img_size,
                  batch_size=cfg.batch_size, max_images=cfg.max_images,
                  pu_tau_quantile=cfg.pu_tau_quantile,
                  pseudo_score_threshold=cfg.pseudo_score_threshold)

    def left():
        return budget - (time.time() - t0)

    # Training time is split across the two fits and the harvest between them.
    # The harvest is inference-only, so it gets the smallest share.
    d_base = t0 + train_budget * 0.45
    d_harvest = t0 + train_budget * 0.55
    d_pu = t0 + train_budget

    # ---- phase 1: imagery ------------------------------------------------
    run_log.phase(1, "resolving training imagery")
    # Cap the fetch at a quarter of the training budget: imagery that has not
    # arrived by then costs more than it is worth.
    d_download = time.time() + train_budget * 0.25
    use_train = prepare_images(train_ids, coco_clean, cfg, deadline=d_download)
    use_val = prepare_images(val_ids, coco_clean, cfg, deadline=d_download)
    summary["phases"]["images"] = {"train": len(use_train), "val": len(use_val)}
    run_log.event("images_resolved", train=len(use_train), val=len(use_val),
                  requested_train=len(train_ids), requested_val=len(val_ids))
    print(f"train {len(use_train)} | val {len(use_val)}")
    if not use_train:
        raise SystemExit("no training imagery could be resolved")

    # ---- phase 2: baseline ------------------------------------------------
    run_log.phase(2, "baseline detector (PU loss OFF)")
    cc_base = PULossConfig(
        num_classes=cfg.num_classes, pu_disabled=True,
        pseudo_label_weight=cfg.pseudo_label_weight,
    )
    model, hist, base_ckpt = train_one_run(
        coco_clean, use_train, use_val, cfg, cc_base, "rtdetr_base.pt",
        deadline=d_base, run_log=run_log,
    )
    summary["phases"]["train_base"] = {
        "epochs": len(hist), "final_loss": hist[-1]["loss"] if hist else None,
    }

    # ---- phase 3: pseudo-label harvest ------------------------------------
    coco_for_pu, added = coco_clean, 0
    if left() > reserve:
        run_log.phase(3, "conservative pseudo-label recovery")
        try:
            # Harvesting runs three forward passes per image; on the full
            # training set that is an hour of GPU for a step whose output is
            # then filtered down to a handful of labels. Sample instead.
            cap = getattr(cfg, "harvest_max_images", 0)
            budget_ids = use_train[:cap] if cap else use_train
            print(f"harvesting over {len(budget_ids)} of {len(use_train)} images")
            harvest, totals = harvest_over_dataset(
                model, coco_clean, budget_ids, cfg, index_to_cat_id)
            coco_for_pu, added = merge_pseudo_labels(coco_clean, harvest)
            before = audit_coco(coco_clean)["instances_per_image"]["mean"]
            after = audit_coco(coco_for_pu)["instances_per_image"]["mean"]
            print(f"\nadded {added} pseudo-labels; "
                  f"instances/image {before:.2f} -> {after:.2f}")
            (work / "coco_with_pseudo.json").write_text(json.dumps(coco_for_pu))
            summary["phases"]["harvest"] = {"added": added, "funnel": totals,
                                            "instances_before": before,
                                            "instances_after": after}
            run_log.event("harvest", added=added, funnel=dict(totals),
                          instances_before=round(before, 3),
                          instances_after=round(after, 3))
        except Exception as e:
            print(f">>> harvest failed ({type(e).__name__}: {e}); "
                  "continuing with the cleaned data")
            summary["phases"]["harvest"] = {"error": f"{type(e).__name__}: {e}"}
    else:
        print(">>> skipping harvest: not enough time left")
        summary["phases"]["harvest"] = {"skipped": "out of time"}

    # ---- phase 4: PU retrain ----------------------------------------------
    final_ckpt = base_ckpt
    if left() > reserve:
        run_log.phase(4, "retrain with pseudo-labels + PU-aware background loss")
        try:
            cc_pu = PULossConfig(
                num_classes=cfg.num_classes,
                pu_w_min=cfg.pu_w_min, pu_w_max=cfg.pu_w_max,
                pu_tau=cfg.pu_tau, pu_temperature=cfg.pu_temperature,
                pu_disabled=cfg.pu_disabled,
                pseudo_label_weight=cfg.pseudo_label_weight,
            )
            print(f"PU: w_min={cc_pu.pu_w_min} w_max={cc_pu.pu_w_max} "
                  f"tau={cc_pu.pu_tau} T={cc_pu.pu_temperature}")
            model, hist_pu, pu_ckpt = train_one_run(
                coco_for_pu, use_train, use_val, cfg, cc_pu, "rtdetr_pu.pt",
                init_from=base_ckpt, deadline=d_pu, run_log=run_log,
            )
            final_ckpt = pu_ckpt
            summary["phases"]["train_pu"] = {
                "epochs": len(hist_pu),
                "final_loss": hist_pu[-1]["loss"] if hist_pu else None,
                "pseudo_labels_used": added,
            }
        except Exception as e:
            print(f">>> PU retrain failed ({type(e).__name__}: {e}); "
                  "falling back to the baseline checkpoint")
            summary["phases"]["train_pu"] = {"error": f"{type(e).__name__}: {e}"}
    else:
        print(">>> skipping PU retrain: not enough time left")
        summary["phases"]["train_pu"] = {"skipped": "out of time"}

    print(f"\nusing checkpoint: {final_ckpt}")

    # ---- phase 5: inference + submission (always) -------------------------
    run_log.phase(5, "two-scale inference + submission")
    if test_json_path is None:
        print(">>> no test annotations; cannot build a submission")
        return summary

    raw_test = load_coco(test_json_path)
    test_ids = [im["id"] for im in raw_test["images"]]
    if cfg.test_max_images:
        test_ids = test_ids[: cfg.test_max_images]
    print(f"test images declared: {len(test_ids)}")

    # The submission needs these, so give the test fetch most of the reserve.
    test_ids = prepare_images(test_ids, raw_test, cfg,
                              deadline=time.time() + reserve * 0.5)
    print(f"test images resolved: {len(test_ids)}")

    preds = predict_two_scale(model, raw_test, test_ids, cfg, index_to_cat_id)
    rows = build_submission(preds)

    sub = work / "submission.csv"
    with open(sub, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=SUBMISSION_COLUMNS)
        w.writeheader()
        w.writerows(rows)

    n_declared = len(raw_test["images"])
    covered = len(preds)
    print(f"\nsubmission: {len(rows)} rows -> {sub}")
    print(f"  images with >=1 detection: {covered}/{n_declared}")
    if covered < n_declared:
        print(f"  >>> {n_declared - covered} test images have no predictions "
              "(undownloadable or no detection above threshold); they score as "
              "missed")

    summary["submission"] = {
        "rows": len(rows), "images_covered": covered,
        "images_declared": n_declared, "path": str(sub),
        "checkpoint": str(final_ckpt),
    }
    run_log.event("submission", rows=len(rows), images_covered=covered,
                  images_declared=n_declared, checkpoint=Path(final_ckpt).name)

    # Validation diagnostics, only if there is time left over.
    if left() > 300 and use_val:
        try:
            run_log.event("validation_start")
            vp = predict_two_scale(model, coco_clean, use_val, cfg, index_to_cat_id)
            summary["val_stats"] = evaluate_map(
                subset_coco(coco_clean, use_val), vp, index_to_name=index_to_name)
        except Exception as e:
            print(f"(validation diagnostics skipped: {type(e).__name__}: {e})")

    (work / "run_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    run_log.event("done", hours=round((time.time() - t0) / 3600.0, 3),
                  submission_rows=summary["submission"]["rows"]
                  if summary.get("submission") else 0)
    print(f"\nDONE in {(time.time() - t0) / 3600:.2f}h", flush=True)
    return summary
''')

md(r"""
## 13. Stage runner

Dispatches on `CFG.stage`. Use `"all"` for a real run: it executes every stage in
one session, which is the only way they share `work_dir`, and it ends by writing
`submission.csv`.

The individual stage names remain for development and ablations. Note that
resuming one in a *fresh* Kaggle commit will not see the previous stage's
checkpoint unless you add this notebook's own output as an input dataset.
""")

code(r'''
BASE_CKPT = "rtdetr_base.pt"
PU_CKPT = "rtdetr_pu.pt"
PSEUDO_JSON = "pseudo_labels.json"
COCO_PU_JSON = "coco_with_pseudo.json"

def criterion_config_from(cfg, pu_on):
    return PULossConfig(
        num_classes=cfg.num_classes,
        pu_w_min=cfg.pu_w_min, pu_w_max=cfg.pu_w_max,
        pu_tau=cfg.pu_tau, pu_temperature=cfg.pu_temperature,
        pu_disabled=not pu_on,
        pseudo_label_weight=cfg.pseudo_label_weight,
    )


def prepare_images_for(ids, coco, cfg, deadline=None):
    """Resolve imagery for `ids`, then return only the ids actually on disk."""
    subset = subset_coco(coco, ids)
    attached = find_attached_image_dir(
        cfg.input_roots, [im.get("file_name", "") for im in subset["images"][:20]]
    )
    cache = Path(attached) if attached else image_cache_root(cfg)
    print(f"image cache: {cache}" + ("  (attached dataset)" if attached else ""))
    ok, failed = ensure_images(subset, cache, workers=cfg.download_workers,
                               deadline=deadline,
                               max_side=getattr(cfg, "download_max_side", 0))
    # Propagate the resolved paths back onto the parent COCO dict.
    resolved = {im["id"]: im.get("_local_path") for im in subset["images"]}
    for im in coco["images"]:
        if im["id"] in resolved:
            im["_local_path"] = resolved[im["id"]]
    if failed:
        print(f">>> {len(failed)} frames unavailable and will be skipped")
    return [i for i in ids if i in ok]


if CFG.stage == "verify":
    print("\nVerification stage complete. "
          "Set CFG.stage='audit' next, then the GPU stages.")

elif CFG.stage == "audit":
    if raw_train is None:
        print("no training annotations found - nothing to audit")
    else:
        out = Path(CFG.work_dir) / "stage1_audit.json"
        out.write_text(json.dumps(
            {"audit": {k: (v if isinstance(v, (int, float, dict, str)) else len(v))
                       for k, v in AUDIT.items()},
             "clean_summary": clean_summary,
             "n_train": len(train_ids), "n_val": len(val_ids)}, indent=2, default=str))
        print(f"\nStage 1 report -> {out}")
        print("Set CFG.stage='train_base' and enable GPU + Internet next.")

elif CFG.stage in {"all", "train_base", "harvest", "train_pu", "infer"}:
    import torch
    assert torch.cuda.is_available(), "this stage needs a GPU"
    assert raw_train is not None, "training annotations not found"

    cat_id_to_index, index_to_cat_id, index_to_name = build_category_maps(coco_clean)
    assert len(cat_id_to_index) == CFG.num_classes, (
        f"config says {CFG.num_classes} classes but the data has "
        f"{len(cat_id_to_index)} - fix CFG.num_classes"
    )

    use_train = train_ids if not CFG.max_images else train_ids[: CFG.max_images]
    use_val = val_ids if not CFG.max_images else val_ids[: max(CFG.max_images // 4, 1)]
    print(f"using {len(use_train)} train / {len(use_val)} val images"
          + ("" if not CFG.max_images else f"  (capped by max_images={CFG.max_images})"))

    if CFG.stage == "all":
        RUN = run_full_pipeline(
            coco_clean, test_json_path, use_train, use_val, CFG,
            index_to_cat_id, index_to_name, prepare_images_for,
        )
        print("\n" + json.dumps(RUN.get("submission") or {}, indent=2))

    elif CFG.stage == "train_base":
        use_train = prepare_images_for(use_train, coco_clean, CFG)
        use_val = prepare_images_for(use_val, coco_clean, CFG)
        print("\n--- Stage 2: baseline detector (PU loss OFF) ---")
        model, hist, ckpt = train_one_run(
            coco_clean, use_train, use_val, CFG,
            criterion_config_from(CFG, pu_on=False), BASE_CKPT)
        print(f"\nbaseline checkpoint: {ckpt}")
        print("Next: CFG.stage='harvest'")

    elif CFG.stage == "harvest":
        use_train = prepare_images_for(use_train, coco_clean, CFG)
        ckpt = Path(CFG.work_dir) / BASE_CKPT
        assert ckpt.is_file(), f"run stage 'train_base' first ({ckpt} missing)"
        device = torch.device("cuda")
        model = build_model(CFG.num_classes, CFG.model_name, device)
        model.load_state_dict(load_checkpoint(ckpt, device)["model"])
        print("\n--- Stage 3: conservative pseudo-label recovery ---")
        harvest, totals = harvest_over_dataset(
            model, coco_clean, use_train, CFG, index_to_cat_id)
        coco_pu, added = merge_pseudo_labels(coco_clean, harvest)
        print(f"\nadded {added} pseudo-labels")
        rep_before = audit_coco(coco_clean)["instances_per_image"]["mean"]
        rep_after = audit_coco(coco_pu)["instances_per_image"]["mean"]
        print(f"instances/image {rep_before:.2f} -> {rep_after:.2f}")
        (Path(CFG.work_dir) / COCO_PU_JSON).write_text(json.dumps(coco_pu))
        (Path(CFG.work_dir) / PSEUDO_JSON).write_text(json.dumps(
            {str(k): {"boxes": v["boxes"].tolist(), "scores": v["scores"].tolist(),
                      "labels": v["labels"].tolist()} for k, v in harvest.items()}))
        print(f"-> {Path(CFG.work_dir) / COCO_PU_JSON}")
        print("Next: CFG.stage='train_pu'")

    elif CFG.stage == "train_pu":
        pu_json = Path(CFG.work_dir) / COCO_PU_JSON
        if pu_json.is_file():
            coco_for_training = json.loads(pu_json.read_text())
            n_pseudo = sum(1 for a in coco_for_training["annotations"] if a.get("is_pseudo"))
            print(f"training on pseudo-augmented data ({n_pseudo} pseudo-labels)")
        else:
            coco_for_training = coco_clean
            print(">>> no pseudo-labels found; training on cleaned data only")
        use_train = prepare_images_for(use_train, coco_for_training, CFG)
        use_val = prepare_images_for(use_val, coco_for_training, CFG)
        print("\n--- Stages 3+4: retrain with PU-aware background loss ---")
        cc = criterion_config_from(CFG, pu_on=not CFG.pu_disabled)
        print(f"PU: w_min={cc.pu_w_min} w_max={cc.pu_w_max} "
              f"tau={cc.pu_tau} T={cc.pu_temperature} disabled={cc.pu_disabled}")
        model, hist, ckpt = train_one_run(
            coco_for_training, use_train, use_val, CFG, cc, PU_CKPT,
            init_from=Path(CFG.work_dir) / BASE_CKPT)
        print(f"\nPU checkpoint: {ckpt}")
        print("Next: CFG.stage='infer'")

    elif CFG.stage == "infer":
        ckpt = Path(CFG.work_dir) / PU_CKPT
        if not ckpt.is_file():
            ckpt = Path(CFG.work_dir) / BASE_CKPT
            print(f">>> PU checkpoint missing, falling back to {ckpt}")
        assert ckpt.is_file(), "no checkpoint - run a training stage first"
        device = torch.device("cuda")
        model = build_model(CFG.num_classes, CFG.model_name, device)
        model.load_state_dict(load_checkpoint(ckpt, device)["model"])

        print("\n--- Stage 5: two-scale inference + Soft-NMS ---")
        print(f"scales {CFG.infer_scales}  soft-nms {CFG.softnms_method}"
              f" sigma={CFG.softnms_sigma}")

        # Validation pass, for the diagnostics (mAP here is a relative signal only).
        use_val = prepare_images_for(use_val, coco_clean, CFG)
        val_preds = predict_two_scale(model, coco_clean, use_val, CFG, index_to_cat_id)
        print(f"\nvalidation predictions on {len(val_preds)} images")
        val_stats = evaluate_map(subset_coco(coco_clean, use_val), val_preds,
                                 index_to_name=index_to_name)

        # Test pass -> submission.
        if test_json_path is not None:
            raw_test = load_coco(test_json_path)
            test_ids = [im["id"] for im in raw_test["images"]]
            if CFG.max_images:
                test_ids = test_ids[: CFG.max_images]
            test_ids = prepare_images_for(test_ids, raw_test, CFG)
            test_preds = predict_two_scale(model, raw_test, test_ids, CFG, index_to_cat_id)
            rows = build_submission(test_preds)
            import csv
            sub = Path(CFG.work_dir) / "submission.csv"
            with open(sub, "w", newline="") as fh:
                wr = csv.DictWriter(fh, fieldnames=SUBMISSION_COLUMNS)
                wr.writeheader()
                wr.writerows(rows)
            print(f"\nsubmission: {len(rows)} rows -> {sub}")
            print(f"  images with detections: {len(test_preds)}/{len(test_ids)}")
        else:
            print("\nno test annotations found - skipping submission")

else:
    raise ValueError(f"unknown stage {CFG.stage!r}")
''')

# --------------------------------------------------------------------------- #
md(r"""
## 14. Ablations worth running

The pipeline was kept small on purpose: if a number moves, you want to know
*which idea* moved it. Four runs answer that, and they are all the same code with
a different config:

| Run | Change | Question it answers |
|---|---|---|
| A | `pu_disabled=True`, no pseudo-labels | Baseline |
| B | `pu_disabled=True` + pseudo-labels | Did Stage 3 alone help? |
| C | `pu_disabled=False`, no pseudo-labels | Did the PU loss alone help? |
| D | Both | Do they compose, or overlap? |

Then sweep `pu_w_min` over `{0.1, 0.25, 0.5, 0.75}` at fixed everything else.
Expect a precision–recall trade-off rather than a free win — and expect
`pu_w_min=0` to fail loudly, which is the point.

Two cautions when reading the results:

- **Validation mAP on the training split is partially wrong**, because that split
  is also incompletely annotated. Treat it as a relative signal between runs.
- **A rise in high-confidence "false positives" is ambiguous.** It is the
  expected signature of recovering unlabeled organisms *and* of the background
  weight being too low. Only the fully annotated evaluation set separates the two.
""")

# =========================================================================== #


def main():
    nb = {
        "cells": [],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3.11"},
            "accelerator": "GPU",
            "kaggle": {"isGpuEnabled": True, "isInternetEnabled": True,
                       "language": "python", "sourceType": "notebook"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }

    errors = []
    n_code = 0
    for i, (kind, src) in enumerate(CELLS):
        lines = src.splitlines(keepends=True)
        if kind == "code":
            n_code += 1
            try:
                ast.parse(src)
            except SyntaxError as e:
                errors.append(f"cell {i} (code #{n_code}) line {e.lineno}: {e.msg}")
            nb["cells"].append({
                "cell_type": "code", "execution_count": None,
                "metadata": {}, "outputs": [], "source": lines,
            })
        else:
            nb["cells"].append({
                "cell_type": "markdown", "metadata": {}, "source": lines,
            })

    if errors:
        print("SYNTAX ERRORS:")
        for e in errors:
            print("  -", e)
        return 1

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(nb, indent=1), encoding="utf-8")

    # Re-read and re-validate what actually landed on disk.
    check = json.loads(OUT.read_text(encoding="utf-8"))
    bad = []
    for i, c in enumerate(check["cells"]):
        if c["cell_type"] == "code":
            try:
                ast.parse("".join(c["source"]))
            except SyntaxError as e:
                bad.append(f"cell {i}: {e.msg}")
    if bad:
        print("POST-WRITE VALIDATION FAILED:", bad)
        return 1

    md_cells = sum(1 for c in check["cells"] if c["cell_type"] == "markdown")
    code_cells = sum(1 for c in check["cells"] if c["cell_type"] == "code")
    print(f"wrote {OUT}")
    print(f"  cells: {code_cells} code + {md_cells} markdown = {len(check['cells'])}")
    print(f"  size : {OUT.stat().st_size / 1024:.0f} KB")
    print("  all code cells parse cleanly")
    return 0


if __name__ == "__main__":
    sys.exit(main())
