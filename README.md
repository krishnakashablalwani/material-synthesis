# Material Synthesis - ML-Based Photocatalysis Analysis

[![ML Training and Parity Plots](https://github.com/krishnakashablalwani/material-synthesis/actions/workflows/ml-pipeline.yml/badge.svg)](https://github.com/krishnakashablalwani/material-synthesis/actions/workflows/ml-pipeline.yml)

Compare experimental photocatalysis data with AI-generated theoretical predictions using machine learning.

## Overview

This tool trains Random Forest and Gradient Boosting models on literature-based ZnFe₂O₄ photocatalysis data, generates theoretical predictions for your experimental samples, and creates publication-ready parity plots with statistical analysis.

## Quick Start

```bash
git clone https://github.com/krishnakashablalwani/material-synthesis.git
cd material-synthesis
pip install -r requirements.txt
```

## Usage

### 1. Prepare Your Data

Edit `data/experimental_photocatalysis.csv`:

```csv
Sample,Crystallite size (nm),Surface area (m2 g^-1),Band gap direct (eV),Band gap indirect (eV),Rate constant (k x 10^-1 / h),Rate of hydrogen evolved (umol (h g.cat)^-1),Quantum yield (%)
μS 10 min,47,2.3,1.93,1.83,1.390,99.2,0.14
μS 30 min,35,4.6,1.93,1.83,2.320,133.5,0.19
μS 150 min,29,5.6,1.93,1.83,0.005,92.2,0.13
CS,53,2.2,1.90,1.81,2.150,31.7,0.05
```

### 2. Train Models & Generate Theoretical Values

```bash
python train_ml_models.py
```

Trains ML models on literature data and generates `data/theoretical_photocatalysis.xlsx`.

### 3. Create Parity Plots

```bash
python generate_parity_plots.py
```

Generates 7 individual parity plots in `output/` directory plus comparison table and summary report.

### 4. Run Statistical Validation (Optional)

```bash
python run_n_iterations.py 1000
```

Runs 1000 training iterations with different random seeds and generates:
- `ml_iterations_all_results_1000.xlsx` - All iteration metrics
- `ml_iterations_summary_1000.xlsx` - Statistical summary (mean, std, min, max)

## Output Files

### Single Run
- `parity_[Property].png` - Individual parity plots (7 files)
- `comparison_table.csv` - Experimental vs theoretical comparison
- `summary_report.txt` - R², MAE, Mean Error statistics

### Multiple Iterations
- `ml_iterations_all_results_N.xlsx` - All N runs with complete metrics
- `ml_iterations_summary_N.xlsx` - Summary statistics across all runs

## Properties Analyzed

1. Crystallite size (nm)
2. Surface area (m² g⁻¹)
3. Band gap direct (eV)
4. Band gap indirect (eV)
5. Rate constant (k × 10⁻¹ / h)
6. Rate of hydrogen evolved (μmol (h g.cat)⁻¹)
7. Quantum yield (%)

## Machine Learning Details

**Training Data**: 18 synthetic samples from ZnFe₂O₄ literature (microwave vs conventional synthesis)

**Features**: 
- Synthesis time (minutes)
- Synthesis method (microwave=1, conventional=0)

**Models**:
- Random Forest (100 trees, max_depth=5)
- Gradient Boosting (100 trees, max_depth=3)
- Best model selected via 3-fold cross-validation

**Randomization**: Each training run uses different random seeds for model variability

## Time Complexity

- Single run: ~5 seconds
- 1000 iterations: ~50-70 minutes
- Complexity: O(iterations × 7 properties × 100 trees × 18 samples × log(100))

## GitHub Actions

Automated workflows run on every push:

**Standard Job**:
- Trains models
- Generates parity plots
- Uploads artifacts

**1000 Iterations Job**:
- Runs complete statistical validation
- Generates iteration results
- Uploads Excel summaries

Manually trigger workflows: Actions tab → Run workflow

## Project Structure

```
material-synthesis/
├── .github/workflows/ml-pipeline.yml  # CI/CD automation
├── data/
│   ├── experimental_photocatalysis.csv  # Your data
│   └── theoretical_photocatalysis.xlsx  # AI predictions
├── output/                              # Generated plots & reports
├── train_ml_models.py                   # ML training
├── generate_parity_plots.py             # Plot generation
├── run_n_iterations.py                  # Statistical validation
└── requirements.txt                     # Dependencies
```

## Requirements

- Python 3.8+
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- openpyxl

## Understanding the Plots

**Perfect Agreement Line (red dashed)**: Ideal experimental = theoretical

**±10% Error Band (gray)**: Acceptable measurement range

### Statistical Metrics

**R² (R-squared / Coefficient of Determination)**:
- Measures how well the model predictions match the experimental data
- Range: -∞ to 1.0
- R² = 1.0: Perfect predictions (all points on the line)
- R² = 0.0: Model performs no better than using the mean value
- R² < 0.0: Model performs worse than a horizontal line
- **Interpretation**: Higher R² indicates better model fit

**MAE (Mean Absolute Error)**:
- Average of the absolute differences between predicted and experimental values
- Same units as the measured property
- MAE = 0: Perfect predictions
- Lower MAE indicates more accurate predictions
- **Example**: MAE = 0.534 nm for crystallite size means predictions are off by ~0.5 nm on average

**Mean Error (%)**:
- Average percentage deviation from experimental values
- Normalized metric for comparing different properties
- **Example**: Mean Error = 1.3% means predictions deviate by 1.3% from experimental values on average
- Lower percentage indicates better relative accuracy