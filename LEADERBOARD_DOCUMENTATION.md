# Leaderboard methodology (Grand Banquet)

This document describes how **composite scoring** and **overfitting assessment** work in Grand Banquet’s Results Analysis outputs (CSV/HTML leaderboard). For installation, the Windows executable, and how to **cite** the software and ChemRxiv preprint, see [README.md](README.md).

---

## 1. Methodology

### 1.1 Composite Scoring Framework

The composite score integrates six components, each weighted to reflect its importance in chemometric model evaluation:

```text
Composite_Score = Σ (w_i × Metric_i)
```

Where the weights are:

| Metric | Weight | Rationale |
| --- | --- | --- |
| Validation Kappa (Val_Kappa) | 0.20 | Agreement metric, measures classification agreement beyond chance |
| Validation MCC (Val_MCC) | 0.18 | Discrimination metric, balanced measure for imbalanced datasets |
| Validation Accuracy (Val_Accuracy) | 0.12 | Overall correctness, fundamental performance indicator |
| Validation Precision (Val_Precision) | 0.10 | Positive predictive value, important for classification tasks |
| Validation F1 Score (Val_F1) | 0.10 | Harmonic mean of precision and recall |
| Overfitting Score | 0.30 | **Model stability and generalisation capability** |

**Total Weight**: 1.00

The high weight (30%) assigned to the overfitting score reflects the critical importance of model stability in chemometric applications, where models must generalise to new samples collected under potentially different conditions.

### 1.2 Overfitting Assessment: Ratio-Based Approach

#### 1.2.1 Theoretical Foundation

Overfitting occurs when a model learns dataset-specific patterns that do not generalise. In the context of kappa values across three dataset splits:

- **Training Kappa (Train_Kappa)**: Performance on data used for model training
- **Cross-Validation Kappa (CV_Kappa)**: Performance on data used for hyperparameter tuning
- **Validation Kappa (Val_Kappa)**: Performance on held-out data, representing true generalisation

A well-generalised model should show consistent kappa values across all three splits. Significant discrepancies indicate overfitting:

- **Train_Kappa >> CV_Kappa**: Model overfits to training data
- **CV_Kappa >> Val_Kappa**: Model overfits to cross-validation data
- **Val_Kappa > Train_Kappa**: Unusual pattern, may indicate data leakage or sampling issues

#### 1.2.2 Ratio Calculation

Three ratios are computed to assess consistency:

1. **Train/CV Ratio**: `R_TC = CV_Kappa / Train_Kappa`
   - Ideal value: 1.0 (perfect consistency)
   - Values < 1.0 indicate training set overfitting
   - Values > 1.0 are capped at 1.0 (CV should not exceed training performance)

2. **CV/Val Ratio**: `R_CV = Val_Kappa / CV_Kappa`
   - Ideal value: 1.0 (perfect consistency)
   - Values < 1.0 indicate cross-validation overfitting
   - Values > 1.0 are capped at 1.0 (validation should not exceed CV performance)

3. **Train/Val Ratio**: `R_TV = Val_Kappa / Train_Kappa`
   - Ideal value: 1.0 (perfect consistency)
   - Values < 1.0 indicate overall overfitting
   - Values > 1.0 are capped at 1.0 (validation should not exceed training performance)

#### 1.2.3 Penalty Mechanism

To penalize severe overfitting, additional penalties are applied:

- **Severe Overfitting (Ratio < 0.8)**: Apply 50% penalty
  - Indicates >20% performance drop between splits
  - Suggests significant overfitting

- **Moderate Overfitting (0.8 ≤ Ratio < 0.9)**: Apply 20% penalty
  - Indicates 10-20% performance drop
  - Suggests mild overfitting

- **Good Consistency (Ratio ≥ 0.9)**: No penalty
  - Indicates <10% performance variation
  - Suggests good generalisation

The penalized ratios are calculated as:

```text
R'_TC = R_TC × penalty_factor
R'_CV = R_CV × penalty_factor
R'_TV = R_TV × penalty_factor
```

#### 1.2.4 Overfitting Score Calculation

The final overfitting score uses the **geometric mean** of the three penalized ratios:

```text
Overfitting_Score = (R'_TC × R'_CV × R'_TV)^(1/3)
```

The geometric mean is preferred over the arithmetic mean because:

1. **Multiplicative nature**: Overfitting is multiplicative—if any ratio is poor, the overall score should reflect this
2. **Stricter assessment**: Geometric mean is more sensitive to outliers, ensuring that poor consistency in any dimension significantly reduces the score
3. **Mathematical properties**: Geometric mean of ratios is scale-invariant and appropriate for ratio-based metrics

**Score Interpretation**:

- **0.8 - 1.0**: Excellent consistency, minimal overfitting
- **0.6 - 0.8**: Good consistency, acceptable overfitting
- **0.4 - 0.6**: Moderate overfitting, caution advised
- **< 0.4**: Severe overfitting, model likely unreliable

### 1.3 Example Calculations

#### Example 1: Ideal Model (No Overfitting)

```text
Train_Kappa = 0.95
CV_Kappa = 0.93
Val_Kappa = 0.92

Ratios:
R_TC = 0.93 / 0.95 = 0.979
R_CV = 0.92 / 0.93 = 0.989
R_TV = 0.92 / 0.95 = 0.968

Penalties: None (all ratios ≥ 0.9)

Overfitting_Score = (0.979 × 0.989 × 0.968)^(1/3) = 0.978
```

#### Example 2: Moderate Overfitting

```text
Train_Kappa = 1.0
CV_Kappa = 0.85
Val_Kappa = 0.75

Ratios:
R_TC = 0.85 / 1.0 = 0.85 (20% penalty → 0.85 × 0.8 = 0.68)
R_CV = 0.75 / 0.85 = 0.882 (20% penalty → 0.882 × 0.8 = 0.706)
R_TV = 0.75 / 1.0 = 0.75 (20% penalty → 0.75 × 0.8 = 0.60)

Overfitting_Score = (0.68 × 0.706 × 0.60)^(1/3) = 0.66
```

#### Example 3: Severe Overfitting

```text
Train_Kappa = 1.0
CV_Kappa = 0.70
Val_Kappa = 0.50

Ratios:
R_TC = 0.70 / 1.0 = 0.70 (50% penalty → 0.70 × 0.5 = 0.35)
R_CV = 0.50 / 0.70 = 0.714 (50% penalty → 0.714 × 0.5 = 0.357)
R_TV = 0.50 / 1.0 = 0.50 (50% penalty → 0.50 × 0.5 = 0.25)

Overfitting_Score = (0.35 × 0.357 × 0.25)^(1/3) = 0.31
```

### 1.4 Composite Score Integration

The composite score combines performance metrics with overfitting assessment:

```text
Composite_Score = 
    0.20 × Val_Kappa +
    0.18 × Val_MCC +
    0.12 × Val_Accuracy +
    0.10 × Val_Precision +
    0.10 × Val_F1 +
    0.30 × Overfitting_Score
```

This ensures that:

1. **High-performing models** with good generalisation receive the highest scores
2. **High-performing models** with overfitting receive reduced scores
3. **Moderate-performing models** with excellent generalisation can rank higher than overfitted high-performers
4. **Model selection** prioritises both performance and stability

## 2. Implementation

### 2.1 Data Collection

For each model combination (preprocessing × feature selection × classifier), the following metrics are collected:

**Validation Set Metrics** (Primary evaluation):

- Kappa
- MCC (Matthews Correlation Coefficient)
- Accuracy
- Precision
- Recall
- F1 Score
- Sensitivity
- Specificity
- Confusion Matrix

**Cross-Validation Metrics**:

- Kappa
- MCC
- Accuracy

**Training Set Metrics**:

- Kappa
- MCC
- Accuracy

**Performance Timing**:

- Training Time (seconds)
- Prediction Time (seconds)

### 2.2 Ranking Algorithm

1. **Calculate Ratios**: Compute Train/CV, CV/Val, and Train/Val ratios for each model
2. **Apply Penalties**: Apply overfitting penalties based on ratio thresholds
3. **Compute Overfitting Score**: Calculate geometric mean of penalized ratios
4. **Compute Composite Score**: Weighted sum of validation metrics and overfitting score
5. **Sort**: Rank models by composite score (descending)
6. **Assign Ranks**: Assign rank 1 to highest composite score

## 3. Advantages Over Traditional Approaches

### 3.1 Multi-Metric Integration

Unlike single-metric ranking (e.g., accuracy-only), GB's composite score integrates:

- **Agreement metrics** (Kappa): Measures classification quality beyond chance
- **Discrimination metrics** (MCC): Balanced measure for imbalanced datasets
- **Precision metrics** (Precision, F1): Important for classification tasks
- **Stability metrics** (Overfitting Score): Ensures generalisation

### 3.2 Overfitting Awareness

Traditional approaches often ignore overfitting, leading to selection of models that fail in production. GB's 30% weight on overfitting ensures that:

- Models with inconsistent performance are penalized
- Models with good generalisation are rewarded
- Selected models are more likely to perform well on new data

### 3.3 Standardised Evaluation

GB provides:

- **Consistent methodology** across all model combinations
- **Reproducible outputs** (CSV and HTML with timestamps)
- **Transparent scoring** (all weights and calculations documented)

### 3.4 Interactive Exploration

The HTML interface enables:

- **Rapid exploration** of different ranking criteria
- **Visual identification** of patterns and outliers
- **Custom analysis** based on specific requirements

## 4. Limitations and Future Work

### 4.1 Current Limitations

1. **Fixed Weights**: Current weights are based on general chemometric best practices but may need adjustment for specific applications
2. **Kappa-Based Overfitting**: Overfitting assessment relies solely on Kappa values; other metrics could be incorporated
3. **Binary/Multi-class Assumption**: The approach works for both binary and multi-class classification but may benefit from class-specific considerations

### 4.2 Future Enhancements

1. **Adaptive Weighting**: Allow users to adjust weights based on application requirements
2. **Multi-Metric Overfitting**: Extend overfitting assessment to include MCC, Accuracy, and F1 consistency
3. **Statistical Significance**: Incorporate statistical tests to assess whether performance differences are significant
4. **Ensemble Recommendations**: Suggest ensemble combinations of top-ranked models
