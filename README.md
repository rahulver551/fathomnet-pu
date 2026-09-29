# Teaching an Object Detector Not to Trust the Background

A positive-unlabeled (PU) object detection pipeline for **FathomNet-CLEF 2026**.

The training annotations are deliberately incomplete: an image may contain a crab,
two fish and an urchin while only the crab is boxed. Ordinary detection training
reads every unannotated region as background, so the model is punished for finding
organisms that are genuinely there. Evaluation, by contrast, is fully annotated.
Everything here follows from one question:

> How do we train a detector when "not labeled" does not mean "negative"?

## The five stages

| Stage | Idea | Implementation |
|---|---|---|
| 1 | Clean the supervision before changing the network | `src/pu_data.py` |
| 2 | One high-resolution RT-DETR detector | notebook §8 |
| 3 | Conservative pseudo-label recovery | `src/pu_ops.py` |
| 4 | PU-aware background loss | `src/pu_criterion.py` |
| 5 | Two-scale inference + Soft-NMS | `src/pu_ops.py` |

### Stage 4 is the core idea

```
L_total = L_positive + λ_box · L_box + w_bg(q) · L_background

w_bg(q) = w_max − (w_max − w_min) · σ((q − τ) / T)
```

- unmatched **+ weak** object evidence → probably background → full penalty
- unmatched **+ strong** object evidence → possibly a missing annotation → reduced penalty

This is not a claim that confident unmatched predictions are correct. It is the
weaker, defensible statement: *"I am less certain that this region is negative."*

**`pu_w_min` must stay above zero.** Remove the background penalty entirely and
false positives explode — rocks, coral texture, equipment and sediment all become
organisms. The weight buys recall on unlabeled organisms and pays in precision;
the useful settings are strictly interior.

## Layout

```
kaggle_run.py            command-line entry point, one stage per invocation
notebooks/…pu_detection.ipynb   self-contained notebook (inlines everything)
src/pu_ops.py            Soft-NMS, IoU, two-scale merge, pseudo-label funnel
src/pu_data.py           COCO audit/clean/split, pseudo-label merge, submission
src/pu_criterion.py      PU-aware Hungarian detection loss (PyTorch)
src/pu_pipeline.py       detector training, inference, harvesting  (generated)
src/pu_config.py         pipeline configuration                    (generated)
tests/test_pipeline.py   145 checks against hand-computed values
build_notebook.py        assembles the notebook from src/
build_pipeline_module.py extracts the generated modules from the notebook
```

Two files under `src/` are generated from the notebook and should not be edited
by hand; the test suite asserts they have not drifted. After changing anything:

```bash
python3 build_notebook.py         # rebuild and syntax-validate the notebook
python3 build_pipeline_module.py  # regenerate pu_pipeline.py and pu_config.py
python3 tests/test_pipeline.py    # 145 checks, including drift guards
```

## Running the tests

```bash
python3 tests/test_pipeline.py     # 145 checks, NumPy only, no GPU
```

Expected values are hardcoded where possible (e.g. `exp(-2)` for a
fully-overlapping Gaussian Soft-NMS decay) so the tests check the algorithm rather
than re-deriving it from the implementation.

The suite includes an end-to-end rehearsal on synthetic data built so the withheld
positives are *known*, which makes Stage 3's behaviour measurable rather than
asserted. Current numbers on that fixture:

```
pseudo-label precision      95.9%
recall of withheld positives 98.9%
instances/image              1.00 -> 2.63 after recovery
```

## Running on Kaggle

```python
!git clone -q https://github.com/rahulver551/fathomnet-pu.git /kaggle/working/repo
%run /kaggle/working/repo/kaggle_run.py --stage verify   # seconds, CPU
%run /kaggle/working/repo/kaggle_run.py --stage all      # the real run
```

`--stage all` executes every stage in one session and ends by writing
`submission.csv`. **Run it as one commit.** Kaggle's weekly GPU quota is generous
but a single run is capped near nine hours, and every commit starts a fresh
container — nothing written to `work_dir` by one version is visible to the next.
Chaining the stages across commits means re-wiring each version's output back in
as an input dataset; running them together avoids the problem outright.

The budget is wall-clock, not epoch-count:

```
--time-budget-hours 8.0    total; stay under Kaggle's ~9h session cap
--reserve-hours     1.5    held back so inference always runs
```

Training splits the remaining time across the baseline fit, the harvest and the
PU retrain, and each stops cleanly at its deadline with a checkpoint saved. The
reserve guarantees the run reaches inference and writes a submission on whatever
checkpoint exists — a truncated model that submits beats a trained one killed
before writing. Harvest and the PU retrain are individually fault-tolerant: if
either fails the run falls back to the baseline checkpoint and still submits.

Imagery is cached in `/kaggle/temp`, not `work_dir`, because Kaggle caps saved
output at 20 GB and this dataset's frames are about that on their own.

The individual stage names (`train_base`, `harvest`, `train_pu`, `infer`) remain
for development and ablations. `--dry-run` prints the resolved config without
running anything. The notebook in `notebooks/` is an equivalent self-contained
alternative driven by its `CFG.stage` field.

| Stage | What it does | GPU | Rough time |
|---|---|---|---|
| `verify` | Algorithm checks, no data needed | no | seconds |
| `audit` | Stage 1: audit + clean + split | no | ~1 min |
| `all` | Every stage, then `submission.csv` | yes | budgeted |
| `train_base` | Stage 2: baseline detector → checkpoint | yes | hours |
| `harvest` | Stage 3: pseudo-labels from the baseline | yes | ~30 min |
| `train_pu` | Stages 3+4: retrain with pseudo-labels + PU loss | yes | hours |
| `infer` | Stage 5: two-scale inference + submission | yes | ~30 min |

Before a GPU stage, enable *Settings → Accelerator → GPU* and
*Settings → Internet → On*. Internet is needed for the pretrained RT-DETR weights
and — unless an images dataset is attached — for the imagery itself.

**Start with `verify`, then `audit`.** Both run on CPU in under a minute and will
catch a wrong dataset path or a broken assumption before any GPU quota is spent.
Each GPU stage checkpoints every epoch, so a session that hits the wall is resumed
by running the next stage in a fresh session.

### Monitoring a run

Every phase, epoch and result is emitted as a single greppable line:

```
[PROGRESS] {"kind": "epoch", "elapsed_h": 1.4, "remaining_h": 6.6, "epoch": 3,
            "loss": 2.71, "imgs_per_s": 14.2, "lr": 6.1e-05}
```

and the full history is rewritten to `work_dir/progress.json` after each event,
so a run killed at the wall still leaves a readable account in the saved output.
Kaggle publishes a version's log only when it terminates and truncates the log
view, which is why per-step lines are throttled (`log_every`) and the things
worth reading are few and structured.

`imgs_per_s` on each epoch line is the number to watch: it is what says whether
the input pipeline or the GPU is the constraint.

### Cost

The competition download is 9.15 MB of annotations only. All 6,463 training frames
and 1,425 test frames stream from FathomNet at runtime — roughly 19 GB and 4 GB of
1920×1080 PNGs respectively. A full run is several GPU-hours across 2–3 sessions
against Kaggle's 30 GPU-hours/week. Start with `--max-images 300`.

### Data

The official repository ships `dataset_train.json` and `dataset_test.json` in COCO
format plus a `download.py` — **the imagery is not bundled**, it is fetched from
FathomNet and partner URLs recorded in the annotations. The notebook resolves
frames from either an attached Kaggle images dataset or by downloading on demand
with a local cache.

Submission is a CSV with exactly these columns:

```
annotation_id, image_id, category_id, bbox_x, bbox_y, bbox_width, bbox_height, score
```

`bbox_*` are COCO `xywh` in original-frame pixels. `build_submission()` handles the
conversion; emitting `xyxy` here silently destroys the score.

## Reading the results honestly

PU learning breaks validation too. If a validation frame contains an unlabeled
fish and the model finds it, the evaluator scores that as a false positive. So a
model that gets **better** at discovering unlabeled organisms can look **worse** on
an incompletely annotated split.

Consequences for interpretation:

- Validation mAP on the training split is a **relative** signal between runs, not
  an absolute measure.
- A rise in high-confidence "false positives" is **ambiguous**: it is the expected
  signature of recovering unlabeled organisms *and* of the background weight being
  too low. Only the fully annotated evaluation set separates the two.
- Watch the Stage 3 funnel counts. If `after_consistency ≈ after_gt_dedup` the
  consistency check is not discriminating; if it collapses to near zero, the stage
  is a no-op.

## Ablations

The pipeline is deliberately small so that if a number moves, you know which idea
moved it.

| Run | Change | Question |
|---|---|---|
| A | `--pu-disabled`, skip `harvest` | Baseline |
| B | `--pu-disabled` + pseudo-labels | Did Stage 3 alone help? |
| C | PU loss on, skip `harvest` | Did the PU loss alone help? |
| D | Both | Do they compose, or overlap? |

Then sweep `--pu-w-min` over `{0.1, 0.25, 0.5, 0.75}`. Expect a precision–recall
trade-off rather than a free win — and expect `--pu-w-min 0` to fail loudly.

## Scope

This is a deliberately simplified pipeline: one detector rather than the
multi-architecture ensembles that appear in published competition solutions. The
point is interpretability, not maximum score.

No leaderboard results are reported here. The defaults are starting points chosen
to be safe, not tuned optima, and any numbers you see come from your own run.

## Data use

FathomNet imagery carries a range of Creative Commons licenses (CC0, CC BY,
CC BY-NC, CC BY-NC-ND). Verify the individual image license and required
attribution before publishing any frame, and do not publish a modified or
annotated version of an image under a NoDerivatives license. See the
[FathomNet data-use policy](https://www.fathomnet.org/datause).

## Sources

- [FathomNet-CLEF 2026 Kaggle competition](https://www.kaggle.com/competitions/fathomnet-2026)
- [Official challenge repository](https://github.com/fathomnet/fgvc-comp-2026)
- Bodla et al., *Soft-NMS — Improving Object Detection With One Line of Code* (2017)
- Zhao et al., *DETRs Beat YOLOs on Real-time Object Detection* (RT-DETR, 2023)
