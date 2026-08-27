# DevelopersHub Phase 2 — ML/AI Tasks

Three ML/AI tasks: a BERT news classifier, a scikit-learn churn-prediction pipeline, and a
multimodal (image + tabular) housing price model.

## Setup

```bash
pip install -r requirements.txt
```

## Tasks

### Task 1 — News Topic Classifier (BERT)

Fine-tunes BERT on the AG News dataset to classify headlines into 4 categories.

```bash
python task1_bert_classifier.py
```

See `README_TASK1.md` for details. Reported accuracy (~84%) is from the original task
documentation and has not been independently re-run in this pass — BERT fine-tuning on CPU
takes long enough that it wasn't re-verified alongside Tasks 2 and 3.

### Task 2 — Customer Churn Prediction (ML Pipeline)

Builds a scikit-learn pipeline (preprocessing + model), compares Logistic Regression and Random
Forest, and tunes both with `GridSearchCV`.

```bash
python task2_ml_pipeline.py
```

**Verified results** (re-run in this pass, on real data):

| Model | Test Accuracy | Test F1 | ROC-AUC |
| --- | --- | --- | --- |
| Logistic Regression (baseline) | 79.35% | 0.581 | 0.840 |
| Random Forest (baseline) | 79.21% | 0.549 | 0.828 |
| Logistic Regression (tuned) | 79.21% | 0.581 | 0.836 |
| Random Forest (tuned) | 73.46% | 0.000 | 0.775 |

Logistic Regression was the strongest model here. The tuned Random Forest's grid search
converged on a degenerate configuration (F1 = 0 — it never predicts the minority class), which
is a legitimate finding worth noting rather than a fabricated number: worth revisiting with
`class_weight='balanced'` or a wider hyperparameter grid.

See `README_TASK2.md` for details.

### Task 3 — Multimodal Housing Price Prediction

Combines CNN-derived image features (ResNet50 transfer learning) with tabular features to
predict housing prices.

```bash
python task3_multimodal_ml.py
```

See `README_TASK3.md` for details. Not re-run in this pass — this environment's TensorFlow
install is broken (a protobuf version conflict unrelated to this project's code), so Task 3's
reported numbers are from the original task documentation, not independently re-verified here.

## What was fixed in this pass

- All three scripts crashed on Windows: they print Unicode checkmarks (✓) without setting
  stdout's encoding, which fails under the default `cp1252` console encoding. Fixed by forcing
  UTF-8 stdout at the top of each script.
- All three scripts hardcoded absolute output paths from wherever they were originally written
  (`/mnt/user-data/outputs/...`, `/home/claude/house_images`) — completely non-portable. Fixed
  to use relative `./outputs/` and `./house_images` paths, with `./outputs` created automatically.
- Task 2 crashed before finishing: `comparison_df['F1-Score'].idxmax()` referenced a column that
  doesn't exist — the actual column is `'Test F1-Score'`. Fixed the key.
