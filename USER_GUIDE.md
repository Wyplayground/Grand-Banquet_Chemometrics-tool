# How to Use Grand Banquet (GB)

A comprehensive step-by-step guide to using Grand Banquet (GB) for chemometric data analysis.

**Distribution:** The public release is a **Windows executable** (see [README.md](README.md)). Follow the steps below after launching that build—you do **not** need to install Python or run `pip install`.

---

## Table of Contents

1. [Data Preparation](#1-data-preparation)
2. [Complete Workflow](#2-complete-workflow)
3. [Troubleshooting](#3-troubleshooting)
4. [FAQ](#4-faq)

---

## 1. Data Preparation

### 1.1 Data Format Requirements

Your data must be in **CSV or Excel format** (.csv, .xlsx, .xls) with the following structure:

```csv
Sample name,Class,Feature1,Feature2,Feature3,Feature4,...
Sample_001,Class_A,0.123,0.456,0.789,0.234,...
Sample_002,Class_B,0.234,0.567,0.890,0.345,...
Sample_003,Class_A,0.345,0.678,0.901,0.456,...
```

**Required Columns** (in this exact order):

- **Column 0**: `Sample name` - Unique identifier for each sample (string)
- **Column 1**: `Class` - Class label (string or numeric)
- **Column 2 onwards**: Feature columns - Numeric values (wavelengths, m/z values, chemical shifts, etc.)

**Important**: You only need **one data file**. The system will automatically split it into training (calibration) and validation sets using stratified sampling.

### 1.2 Data Quality Checklist

Before loading your data, ensure:

- No missing values in feature columns (NaN values will be automatically filled with column means)
- All feature values are numeric
- Sample names are unique
- Class labels are consistent (rows with NaN labels will be automatically removed)
- Sufficient samples per class (minimum 3-5 per class for stratified split)
- No special characters in column names (use underscores instead of spaces)
- File encoding is UTF-8 or compatible

---

## 1.5 Quick Demo: Efficient Workflow Method

This section demonstrates a recommended workflow method for efficient analysis:

### Recommended Workflow: "Select All Options" Method

This method allows you to quickly select all options and then run each step individually, giving you better control over the process.

#### Step-by-Step Demo

1. **Load Your Data** (Tab 1: Data Loading)
   - Select your data file
   - Set test data ratio (default: 0.2)
   - Click "Load and Split Data"
   - Wait for data to be loaded and split

2. **Select All Options** (Tab 5: Results Analysis)
   - Navigate to **"5. Results Analysis"** tab
   - Click the **"Select All Options"** button
   - This automatically selects:
     - All 15 preprocessing methods (Individual mode)
     - All 6 feature selection algorithms (VIP, GA, iPLS, CARS, SFLA, None)
     - All 11 classification models (PLS-DA, LDA, SVM, RF, XGBoost, etc.)
   - Console will show: "✓ All options selected successfully!"

3. **Configure Cross-Validation** (Tab 4: Model Training)
   - Navigate to **"4. Model Training"** tab
   - Set **"Number of folds"** to **3** (for faster processing)
   - This reduces computation time while maintaining reasonable validation

4. **Run Preprocessing** (Tab 2: Preprocessing)
   - Navigate to **"2. Preprocessing"** tab
   - Verify all preprocessing methods are selected (should be checked after Step 2)
   - Click **"Run Preprocessing"**
   - Wait for all preprocessing methods to complete
   - Results saved to `02_preprocessed_data/`

5. **Run Variable Selection** (Tab 3: Variable Selection)
   - Navigate to **"3. Variable Selection"** tab
   - Verify all variable selection methods are selected (should be checked after Step 2)
   - Click **"Run Variable Selection"**
   - Wait for all feature selection algorithms to complete
   - Results saved to `03_variable_selection/`

6. **Train Models** (Tab 4: Model Training)
   - Navigate to **"4. Model Training"** tab
   - Verify all models are selected (should be checked after Step 2)
   - Verify CV folds is set to **3** (from Step 3)
   - Click **"Train Models"**
   - Models train in parallel (faster with multiple models)
   - Results saved to `04_model_training/`

7. **Analyze Results** (Tab 5: Results Analysis)
   - Navigate back to **"5. Results Analysis"** tab
   - View performance summary
   - Click **"Generate Performance Report"** to save CSV
   - Click **"Generate Leaderboard (CSV/HTML)"** for ranking
   - Review interactive HTML leaderboard

### Advantages of This Method

- **Quick Setup**: "Select All Options" button saves time selecting individual checkboxes
- **Better Control**: Run each step individually, allowing you to monitor progress
- **Faster Processing**: CV folds = 3 reduces computation time significantly
- **Comprehensive Analysis**: Still covers all 990 combinations (15 × 6 × 11)
- **Parallel Training**: Model training uses parallel processing for speed

**Note**: This method is recommended for users who want comprehensive analysis with better control over each step.

---

## 2. Complete Workflow

### Step 1: Launch the Application

1. Download the release package from the repository **Releases** page (see [README.md](README.md)).
2. Extract all files to one folder if the download is an archive.
3. If your package includes `123.png`, keep it in the **same folder** as the `.exe`.
4. Double-click the Grand Banquet `.exe` to start the application.

The GUI window will open with the following tabs:

1. Data Loading
2. Preprocessing
3. Variable Selection
4. Model Training
5. Results Analysis
6. AI Report Generation

### Step 2: Load Your Data

1. **Navigate to "1. Data Loading" tab**
2. **Select Base Directory**: Choose where results will be saved
   - Click "Browse" next to "Base Directory"
   - Select or create a folder for output files
3. **Load Data File**:
   - Click "Browse" next to "Raw data file"
   - Select your CSV file (must contain Sample name, Class, and feature columns)
   - The system will automatically split your data into training and validation sets
4. **Click "Load and Split Data"**:
   - The system automatically splits your data using stratified sampling
   - Training set (calibration set) and validation set are created automatically
   - Both sets are saved to `01_raw_data/` folder for reference

**Expected Output**:

- Console shows: "Data loaded successfully"
- Console shows: "Data split completed: Training set: X samples, Test set: Y samples"
- Calibration and validation sets automatically saved to `01_raw_data/calibration_set.csv` and `01_raw_data/validation_set.csv`
- Status message: "Data loaded and split successfully!"

### Step 3: Preprocess Your Data

1. **Navigate to "2. Preprocessing" tab**
2. **Select Mode**:
   - **Combination Mode**: Apply multiple methods sequentially (on progress)
   - **Individual Mode**: Evaluate each method separately
3. **Select Preprocessing Methods**:
   - Check boxes for desired methods
4. **Click "Run Preprocessing"**
5. **Wait for completion**: Progress bar shows status

**Expected Output**:

- Preprocessed data saved to `02_preprocessed_data/`
- Status message: "Preprocessing completed successfully"

### Step 4: Select Features (Variable Selection)

1. **Navigate to "3. Variable Selection" tab**
2. **Select Feature Selection Algorithms**:
   - **VIP**: Fast, PLS-based
   - **GA**: Genetic Algorithm
   - **iPLS**: Interval-based
   - **CARS**: Monte Carlo-based
   - **SFLA**: Metaheuristic
   - **None**: Use all features
3. **Configure Parameters** (if needed):
   - VIP: Number of top features
   - GA: Population size, generations
   - iPLS: Number of intervals
4. **Click "Run Variable Selection"**
5. **Review Results**: Selected features saved to `03_variable_selection/`

**Expected Output**:

- Selected features CSV files
- Feature selection statistics
- Status message: "Variable selection completed successfully"

### Step 5: Train Models

1. **Navigate to "4. Model Training" tab**
2. **Select Classification Models**:
   - Check boxes for desired models
   - Recommended selection:
     - PLS-DA
     - Random Forest
     - SVM
     - XGBoost
3. **Configure Cross-Validation**:
   - Number of folds: 5
   - Minimum: 3 folds
4. **Click "Train Models"**
5. **Monitor Progress**: Console output shows training status

**Expected Output**:

- Trained models saved to `04_model_training/`
- Performance metrics displayed
- Status message: "Model training completed"

**Performance Note**: With parallel processing enabled, multiple models train simultaneously, significantly reducing total time.

### Step 6: Analyze Results

1. **Navigate to "5. Results Analysis" tab**
2. **View Performance Summary**:
   - Table shows all models with key metrics
   - Kappa, MCC, Accuracy, Precision, etc.
3. **Generate Performance Report**:
   - Click "Generate Performance Report"
   - CSV file saved to `05_results/pipeline_results.csv`
4. **Compare Models**:
   - Click "Compare Models" for visualizations
   - Plots saved to `05_results/comparison_plots/`
5. **Generate Leaderboard**:
   - Click "Generate Leaderboard (CSV/HTML)"
   - Interactive HTML leaderboard opens in browser
   - CSV file saved for further analysis

**Expected Output**:

- Performance report CSV
- Comparison visualizations
- Interactive HTML leaderboard
- Model ranking by composite score

### Step 7: Generate AI Reports (Optional)

1. **Navigate to "6. AI Report Generation" tab**
2. **Obtain API Key**:
   - Aliyun DashScope: [DashScope console](https://dashscope.console.aliyun.com/)
3. **Enter API Key** in the text field
4. **Select Model**: Choose LLM (e.g., "qwen-plus", "QWQ")
5. **Analyze Performance**: Click "Analyze Model Performance" (required first step)
6. **Generate Reports**: Click "Generate AI Report"
7. **Review Reports**: Word and PDF files saved to `06_ai_reports/`

**Expected Output**:

- Narrative-style analysis reports
- Word (.docx) and PDF formats
- Reports for different model groups (Perfect, Top 10%, etc.)

---

## 3. Troubleshooting

### Problem 1: "FileNotFoundError" when loading data

**Solution**:

- Check file path is correct
- Ensure file is CSV format
- Check file is not open in another program

### Problem 2: "Memory error" during training

**Solution**:

- Reduce number of preprocessing combinations
- Use feature selection to reduce dimensionality
- Close other applications to free memory
- Process in smaller batches

### Problem 3: Models fail to train

**Solution**:

- Check data has no missing values
- Verify class labels are valid
- Ensure sufficient samples per class (minimum 3-5)
- Check console output for specific error messages

### Problem 4: XGBoost import error

**Solution**:

- In the **released executable**, libraries are bundled with the app. Grand Banquet **automatically falls back to LightGBM** if XGBoost is not available—no separate install step is required.
- If the console still reports an XGBoost-related failure after updating to the latest release, contact the maintainers (see FAQ).

### Problem 5: Slow performance

**Solution**:

- Enable parallel processing (default: enabled)
- Reduce number of models to train
- Use faster feature selection (VIP instead of GA)
- Reduce cross-validation folds (minimum: 3)

### Problem 6: API key error for AI reports

**Solution**:

- Verify API key is correct
- Check API quota/balance
- Ensure internet connection

---

## 4. FAQ

### Q1: Can I use my own preprocessing methods?

**A**: Currently, you can use the 15 built-in methods. Custom preprocessing is **not** available in the distributed executable; if you need a specialised workflow, contact the developers.

### Q2: Do I need to prepare separate calibration and validation files?

**A**: No, Grand Banquet automatically splits your data. Just load one CSV file, and the system will:

- Automatically split it into training (calibration) and validation sets
- Use stratified sampling to maintain class distribution
- Save both sets to `01_raw_data/` folder for your reference
- You can adjust the split ratio (default: 20% for validation)

### Q3: How do I choose the best model?

**A**: Use the leaderboard composite score as primary criterion, then check:

- Overfitting Score > 0.75
- Application-specific metrics near 1.00 (Kappa, MCC, etc.)
- Model interpretability (if needed)

### Q4: Can I use Grand Banquet for regression tasks?

**A**: Currently, Grand Banquet is designed for classification tasks. Regression support may be added in future versions.

### Q5: How do I cite Grand Banquet?

**A**: See [README.md](README.md) for citation information.

### Q6: Is commercial use allowed?

**A**: Yes, Grand Banquet uses Apache License 2.0, which allows both commercial and non-commercial use. See [LICENSE](LICENSE) for details.

### Q7: Can I modify the code?

**A**: This public release is a **compiled Windows application**, not source code—you cannot change the built-in algorithms yourself. The project is licensed under **Apache License 2.0**; you may use and redistribute the executable according to [LICENSE](LICENSE). For feature requests or collaboration, use the contact details in **Q9**.

### Q8: What data types are supported?

**A**: Grand Banquet works with any high-dimensional sensor data:

- Spectroscopy (IR, FTIR, NIR, UV-Vis, Raman)
- Mass spectrometry (LC-MS, GC-MS, MALDI-TOF)
- NMR (1H-NMR, 13C-NMR)
- Chromatography (HPLC, GC, LC)
- Electrochemistry
- Generic sensor data

### Q9: How do I report bugs or request features?

**Email**: [wjia02@qub.ac.uk](mailto:wjia02@qub.ac.uk) / [jiawenyang@cofco.com](mailto:jiawenyang@cofco.com)

---
