# Grand Banquet (GB)

<div align="center">

**A next-generation chemometrics framework for classification analysis on data matrix**

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.8+-green.svg)](https://www.python.org/)
[![Version](https://img.shields.io/badge/Version-v0.0-orange.svg)](README.md)

*Replacing ad-hoc "try-and-trust" workflows with near-exhaustive pipeline search (Glass box) and standardised, multi-metric ranking*

</div>

---

## Overview

Grand Banquet (GB) is a next-generation chemometrics framework for classification analysis on data matrix that systematically explores up to **990 unique combinations** (15 preprocessing × 6 feature selection × 11 classifiers) and provides comprehensive performance evaluation with multi-metric ranking.

### Key Features

- **Exhaustive Pipeline Enumeration**: Systematically explores 990 method combinations
- **Multi-Metric Evaluation**: Kappa, MCC, Accuracy, Precision, Sensitivity, Specificity
- **Overfitting Assessment**: Ratio-based evaluation penalises inconsistent models
- **Composite Leaderboard**: Balances predictive performance with generalisation
- **Parallel Processing**: Model-level parallelism for faster training
- **Interactive Visualisations**: HTML leaderboards with sorting and filtering
- **AI-Powered Reports**: LLM-generated narrative summaries

## Quick Start

### Installation

1. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the application** （"grand_banquet.py" will be published soon）
   ```bash
   python grand_banquet.py
   ```

   > **Tip**: Ensure that `123.png` and `grand_banquet.py` are placed in the same folder. The image `123.png` is inspired by David Hockney's masterpiece *Portrait of an Artist (Pool with Two Figures)*, paying homage to this iconic work. Much like one standing by a pool, gazing at one's own reflection in the water, we too observe data. However sophisticated our methods may be, it remains akin to viewing the data through rippling waters: a pursuit of truth through the wave of uncertainty.

### Quick Start: Generate 990 Models with One Click

**The fastest way to explore all 990 combinations (15 preprocessing × 6 feature selection × 11 classifiers):**

1. **Load Your Data** (Tab 1: Data Loading)
   - Select your CSV/XLSX file
   - Click "Load and Split Data"

2. **Select All Options** (Tab 5: Results Analysis)
   - Navigate to **"5. Results Analysis"** tab
   - Click the **"Select All Options"** button
   - This automatically selects all 15 preprocessing methods, 6 feature selection algorithms, and 11 classification models
   - Console will show: "All options selected successfully!"

3. **Configure & Run** (Tabs 2-4)
   - **Tab 2**: Click "Run Preprocessing"
   - **Tab 3**: Click "Run Variable Selection"
   - **Tab 4**: Set CV folds and Click "Train Models" (trains all 990 combinations in parallel!)

4. **View Results** (Tab 5)
   - Generate leaderboard and performance reports
   - Review interactive HTML leaderboard with all 990 model rankings

For detailed step-by-step guide, see [USER_GUIDE.md](USER_GUIDE.md#25-quick-demo-efficient-workflow-method) (Section 2.5)

### Basic Usage (Manual Selection)

If you prefer to select methods manually:

1. **Load Data**: Navigate to "1. Data Loading" tab and select your CSV/XLSX file
2. **Preprocess**: Choose preprocessing methods in "2. Preprocessing" tab
3. **Select Features**: Choose feature selection algorithms in "3. Variable Selection" tab
4. **Train Models**: Select classifiers in "4. Model Training" tab
5. **Analyse Results**: View performance metrics and generate leaderboard in "5. Results Analysis" tab

## Documentation

- **[User Guide](USER_GUIDE.md)** - Step-by-step user guide (Start here)
- **[Preprint]([https://doi.org/10.26434/chemrxiv.15000013/v1])** - Preprint paper (ChemRxiv) *(https://doi.org/10.26434/chemrxiv.15000013/v1)*
- **[Leaderboard System](LEADERBOARD_DOCUMENTATION.md)** - Performance evaluation details

## Supported Methods

### Preprocessing (15 methods)
Raw, Baseline Correction, Normalise, Centre and Scale, Log Transform, Centring, Pareto Scaling, SNV, MSC, First/Second Derivative, Area/Range Normalisation, SG Smooth, SG Derivative

### Feature Selection (6 algorithms)
VIP, Genetic Algorithm (GA), iPLS, CARS, SFLA, None

### Classification Models (11 models)
PLS-DA, LDA, SVM (Linear/RBF), Random Forest, XGBoost, Gradient Boosting, KNN, Logistic Regression, Naive Bayes, Neural Network

## Licence

This project is licensed under the **Apache Licence 2.0**.
See [LICENSE](LICENSE) for full terms.

## Contact

- **Developer**: COFCO Nutrition & Health Research Institute Co., Ltd.
- **Year**: 2025
- **Email**: jiawenyang@cofco.com / wjia02@qub.ac.uk

## Acknowledgements

Grand Banquet integrates established chemometric methods and machine learning algorithms from the scientific community. Special thanks to the open-source community for the excellent libraries that make this project possible.

## Software copyright in China
Registration Number: 2025SR1655639

## Citation

If you use Grand Banquet in your research, please cite:

```bibtex
@software{grand_banquet_2025,
  title = {Grand Banquet: A Comprehensive Chemometric Framework for Sensor Data Analysis},
  author = {COFCO Nutrition \& Health Research Institute Co., Ltd.},
  year = {2025},
  version = {v0.0},
  url = {https://github.com/your-username/grand-banquet},
  licence = {Apache-2.0}
}
```

---

<div align="center">

**Inspired by the concept: "There is no one silver bullet"**

*Grand Banquet offers a low-bias baseline for modern data analysis and a practical step towards method governance and reproducibility.*

</div>



