# Certified Decision Transfer under Unlabeled Shift

Review artifact for Rate Matching (RM) and Certified Rate Matching (C-RM).
RM transfers the reference policy's action count to a candidate ranking.
C-RM uses historical positive examples and independent, unlabeled target
decisions to decide whether to approve a replacement.

## Start here: CPU-only reproduction

Python 3.10 or newer is recommended. From the repository root:

```bash
python -m pip install -r requirements.txt
python experiments/example.py
python experiments/audit_release.py
```

The first command after installation is a synthetic implementation smoke test,
not a reported experiment. The audit reads the included compressed JSON records;
it requires no GPU, model weights, dataset download, token, or pickle loading.
It checks archive hashes, theory identities, approval counts, and both population
and finite-batch losses. All comparisons in each released archive are included,
including rejected proposals and observed losses.

| Audit | Comparisons | Expected result |
| --- | ---: | --- |
| GoEmotions, TF-IDF to Qwen, three runs | 26,250 | 3,578 harmful proposals; fully paired gate approves 834; 0 observed population and batch losses |
| GoEmotions, TF-IDF to ModernBERT | 26,250 | 256 fully paired approvals; 0 observed population and batch losses |
| GoEmotions, RoBERTa to Qwen | 26,250 | 0 approvals |
| GoEmotions, RoBERTa to ModernBERT | 26,250 | 0 approvals |
| COCO source splits 1 / 2 / 3 | 31,500 each | Direct-cell: 153 / 190 / 95; factorized: 616 / 846 / 439; 0 observed population losses |
| Full COCO primary archive, 100 target resamples | 63,000 | Factorized: 1,334 approvals, 0 population losses, 6 finite-batch losses |

The three COCO source splits use 50 target resamples; the first is the first
50 resamples of the 100-resample primary archive. These are related audits,
not independent sets to pool. An empirical population endpoint is computed
against the known benchmark mixture; it is distinct from a finite evaluation batch.

## Code map

| File in `experiments/` | Purpose |
| --- | --- |
| `example.py` | Exact-count RM, deterministic tie handling, and a small C-RM example |
| `direct_cell_certificate.py` | CPU-only extraction of the original fully paired certificate functions |
| `analyze_factorized_certificate.py` | Disagreement counts, six one-sided CP bounds, factorized gate, risk-budget curves |
| `audit_release.py` | Standalone audit of the included records |
| `bench_goemo_model_upgrade_crm.py` | Strict model-selection / certification split and model-upgrade evaluation |
| `bench_coco_crm_certificate.py` | COCO representation transfer and certificate evaluation |
| `bench_coco_crm_selected_candidates.py` | Multiple-candidate experiment |
| `simulate_certificate_calibration.py` | Synthetic null / harmful / beneficial calibration study |
| `verify_theory_strengthening.py` | Numerical checks of decision-regret identities |
| `risk_controlled_postprocessing.py` | Risk-controlled post-processing comparison |
| `bench_independent_factorized_replication.py` | Independent replication protocol |
| `bench_wilds_crm_subgroups.py` | Subgroup shift evaluation |
| `finetune_*`, `extract_*`, `build_goemo_tfidf_scores.py` | Model training and feature / score preparation |

Original experiment modules are copied byte-for-byte from the research workspace;
their SHA-256 hashes are in `MANIFEST.json`. The CPU-only direct-cell module is
extracted verbatim from the original certificate implementation, with only its
imports reduced. The example and release audit are new entry points, not new
experiments. Original command-line defaults are historical defaults: use the
explicit settings below for the headline protocol.

## What the audit reconstructs

- **GoEmotions:** decision margins from saved confidence-bound endpoints;
  all per-comparison approvals, harmful proposals, exact calibration counts,
  F1 means and both loss counts. Recomputing the confidence bounds from scores
  requires the score preparation and benchmark scripts below.
- **COCO:** integer disagreement cells reconstructed from the saved counts and
  rates, the original direct-cell CP margin, and new factorized CP margins.
  It then recomputes the approval and loss curves without fitting a model.
- **Integrity:** complete record exports are JSON + gzip rather than object-pickled
  NPY files. Numerical fields are unchanged. Only machine-specific paths in the
  summary metadata are normalized. Source-array hashes are recorded for provenance.

## Reproduce from data and models

The original training and feature extraction paths require CUDA. Install a
PyTorch / torchvision pair appropriate for the CUDA driver, then install
`requirements-training.txt`. `environment-versions.json` records the restored
server's installed versions as a reference, not a portable lockfile. The
CPU-only requirements were tested independently for the included audit.

Use a data disk for caches and outputs. For example, on Linux:

```bash
export DATA_ROOT=/path/to/data-disk
export HF_HOME="$DATA_ROOT/cache/huggingface"
export TORCH_HOME="$DATA_ROOT/cache/torch"
mkdir -p "$DATA_ROOT/scores" "$DATA_ROOT/checkpoints" "$DATA_ROOT/results"
```

### GoEmotions

Data: the `simplified` configuration of
`google-research-datasets/go_emotions`. The scripts preserve the official
train / validation / test order and check label alignment. The strict protocol
uses 20% of validation examples for checkpoint selection and the remaining 80%
for certification, with split seed 20260820. Do not use the legacy default
checkpoint fraction of 1.0 for this protocol.

```bash
python experiments/build_goemo_tfidf_scores.py \
  --output "$DATA_ROOT/scores/tfidf.npz"
python experiments/finetune_goemo_roberta.py \
  --model answerdotai/ModernBERT-base --seed 42 \
  --checkpoint-fraction 0.20 --checkpoint-split-seed 20260820 \
  --output "$DATA_ROOT/scores/modernbert_s42.npz" \
  --checkpoint-dir "$DATA_ROOT/checkpoints/modernbert_s42"
python experiments/bench_goemo_model_upgrade_crm.py \
  --reference-scores "$DATA_ROOT/scores/tfidf.npz" \
  --receiver-scores "$DATA_ROOT/scores/modernbert_s42.npz" \
  --checkpoint-fraction 0.20 --checkpoint-split-seed 20260820 \
  --delta 0.05 --target-size 5000 --target-seeds 50 \
  --output-prefix "$DATA_ROOT/results/tfidf_to_modernbert_s42"
```

Repeat with training seeds 43 and 44. `FacebookAI/roberta-base` supplies the
RoBERTa reference; `Qwen/Qwen3-0.6B` supplies the fine-tuned Qwen candidate.
The released archives, rather than a fresh stochastic
training run, are the exact reference for the numerical audit.

### COCO

The feature extractor accepts either COCO images plus the official instance
annotation JSON or the archived parquet layout. The core comparison uses
ResNet-50 reference features and DINOv2-base receiver features.

```bash
python experiments/extract_coco_deep_features.py \
  --image-root "$DATA_ROOT/coco/train2017" \
  --annotation "$DATA_ROOT/coco/annotations/instances_train2017.json" \
  --receiver facebook/dinov2-base \
  --output "$DATA_ROOT/coco_features.npz"
python experiments/bench_coco_crm_certificate.py \
  --features "$DATA_ROOT/coco_features.npz" \
  --output-prefix "$DATA_ROOT/results/coco_split0" \
  --classes 30 --head-seeds 3 --target-seeds 50 --target-size 5000 \
  --delta 0.05 --split-seed 0 \
  --train-fraction 0.50 --threshold-fraction 0.05 \
  --certificate-fraction 0.25 --target-fraction 0.20
python experiments/analyze_factorized_certificate.py \
  --input "$DATA_ROOT/results/coco_split0.npy" \
  --protocol "$DATA_ROOT/results/coco_split0.json" \
  --json "$DATA_ROOT/results/factorized.json" \
  --csv "$DATA_ROOT/results/factorized.csv"
```

For the other source splits, use split seeds 1 and 2. The headline archives use
the restored 117,266-image feature archive; extracting a fresh full official
COCO set is a new run and need not reproduce those numbers bit-for-bit.

## Interpretation and scope

The nominal error budget controls an erroneous approval for a pre-specified
comparison under the stated sampling and positive-conditional stability
assumptions. It is not a bound on the fraction of approved upgrades that are
harmful. The guarantee concerns population F-beta, not every finite evaluation
batch. Policies and tuning decisions must be fixed before using independent
certification data. Repeated comparisons share trained models and source evidence.

The post-processing baseline includes source-calibrated and target-labeled
settings. Keep these supervision settings separate when reporting results.
The release includes core scripts for additional domains, but the bundled
record replay covers the GoEmotions and COCO audits listed above, not every
experiment in the paper.

## Large-asset backup and anonymity

The dataset, feature, model and environment snapshot is also archived on
Hugging Face (about 190 GB). The author-account address is intentionally not
embedded in this double-blind review artifact. The small record archives and
environment-version inventory needed for the CPU audit are included here.
They do not require the large backup. External datasets and model weights
remain subject to their original licenses and access conditions.
