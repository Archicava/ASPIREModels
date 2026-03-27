# ASPIREModels - ASD Detection Model Training Pipeline

PyTorch-based neural network training pipeline for Autism Spectrum Disorder
(ASD) detection using clinical features. Part of the
[ASPIRE LAB](https://github.com/Archicava/aspire) ecosystem -- trains the
binary classifier that powers the Aspire ASD Screening API consumed by the
ASPIRE frontend.

## Clinical features

The model uses the **top 8 clinical features** (representing ~84 % of
predictive power):

| # | Feature                    | Values / Range                                  |
|---|----------------------------|-------------------------------------------------|
| 1 | Developmental milestones   | Global (G), Motor (M), Cognitive (C) delay      |
| 2 | IQ / DQ                   | Numeric score 0 -- 150 (default 70)             |
| 3 | Intellectual disability    | ICD codes (N, F70.0, F71, F72)                  |
| 4 | Language disorder          | Y / N                                           |
| 5 | Language development       | Normal, Delayed, Absent                         |
| 6 | Dysmorphism                | Yes / No                                        |
| 7 | Behaviour disorder         | Aggressivity, agitation, irascibility            |
| 8 | Neurological examination   | Normal or abnormal with description             |

## Architecture

`BaseASDDetector` is a configurable feed-forward network:

```
Input (8 features)
  -> Hidden layers (configurable, default [64, 32])
  -> BatchNorm + Dropout
  -> Sigmoid output (binary probability)
```

Supports ReLU, LeakyReLU, ELU, GELU, SiLU, and Tanh activations.

## Training pipeline

```
Load Data (ASD patients + healthy controls)
  -> Stratified Train / Val / Test split (68 % / 15 % / 20 %)
  -> ClinicalPreprocessor (StandardScaler + LabelEncoder)
  -> Grid search over hyperparameters
       - Learning rates: 1e-3, 5e-4, 1e-4
       - Hidden sizes: [64,32], [32,16], [128,64,32]
       - Dropout: 0.2, 0.3, 0.4
  -> Adam optimizer with weight decay (1e-4)
  -> ReduceLROnPlateau scheduler (optimises recall)
  -> Early stopping (patience 15)
  -> Best model selection (highest recall with precision >= 0.80)
```

Metrics tracked per epoch: accuracy, precision, recall, F1, AUC-ROC, and
confusion matrix.

## Quick start

```bash
# 1. Place the data files in ../aspire/data/ (relative to this repo):
#      patients_db.xlsx         (sheet: 'asd_patients')
#      healthy_patients_gpt2.xlsx (sheet: 'healthy_patients')

# 2. Install dependencies
pip install torch numpy pandas scikit-learn joblib openpyxl

# 3. Run training (GPU used automatically when available)
python train_base_model.py

# 4. Inspect results
cat _output/grid_search_*/summary.json
```

## Output structure

Each grid search run produces a timestamped directory:

```
_output/
  grid_search_YYYYMMDD_HHMMSS/
    grid_search_config.json        # full configuration
    preprocessor.joblib            # fitted scaler / encoders
    all_results.json               # per-run metrics
    summary.json                   # best model summary
    logs/
      master_training.log
      run_XXX_*.log
    run_001_H64_32_lr1e-03_d0.2_ReLU/
      checkpoints/
        best_model.pth             # best weights
        latest.pth
        epoch_010.pth              # periodic saves
      logs/
        run_config.json
        training_history.json
```

## Dependencies

| Package       | Purpose                          |
|---------------|----------------------------------|
| PyTorch       | Neural network training          |
| NumPy         | Numerical operations             |
| pandas        | Data loading and manipulation    |
| scikit-learn  | Preprocessing, metrics, splits   |
| joblib        | Preprocessor serialisation       |
| openpyxl      | Excel workbook reading           |

## Integration with ASPIRE LAB

```
ASPIREModels (this repo)          ASPIRE (frontend)
  train_base_model.py               lib/aspire-api.ts
         |                                |
    trains model                   calls /predict
         |                                |
         v                                v
  best_model.pth  -->  Aspire ASD Screening API  <--  case submissions
                       (serves predictions at
                        ASPIRE_API_URL, default
                        http://localhost:5083)
```

The trained `.pth` checkpoint and `preprocessor.joblib` are deployed behind a
REST API. The ASPIRE frontend maps `CaseSubmission` fields to the 8-feature
`struct_data` payload and receives back a prediction (Healthy / ASD),
probability, confidence, and risk level.

## Configuration

Key constants in `train_base_model.py`:

| Parameter                  | Default                              |
|----------------------------|--------------------------------------|
| `data_dir`                 | `../aspire/data/`                    |
| `output_dir`               | `./_output/`                         |
| `device`                   | `cuda` (if available) or `cpu`       |
| `seed`                     | 42                                   |
| `num_epochs`               | 100                                  |
| `early_stopping_patience`  | 15                                   |
| `test_size`                | 0.20                                 |
| `val_size`                 | 0.15                                 |
| `min_precision`            | 0.80                                 |

