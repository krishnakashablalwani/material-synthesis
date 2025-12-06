# Material Synthesis - ML-Based Photocatalysis Analysis

[![ML Training and Parity Plots](https://github.com/krishnakashablalwani/material-synthesis/actions/workflows/ml-pipeline.yml/badge.svg)](https://github.com/krishnakashablalwani/material-synthesis/actions/workflows/ml-pipeline.yml)

AI/ML-powered tool for comparing experimental photocatalysis data with theoretical predictions using machine learning models.

## Features

- **Machine Learning Models**: Random Forest & Gradient Boosting algorithms trained on literature data
- **Automated Parity Plots**: Individual plots for 7 photocatalysis properties
- **Statistical Analysis**: R², MAE, and Mean Error calculations
- **100 Iteration Analysis**: Comprehensive statistical validation across multiple runs
- **GitHub Actions**: Automated CI/CD pipeline for continuous testing

## Quick Start

### 1. Clone the Repository
```bash
git clone https://github.com/krishnakashablalwani/material-synthesis.git
cd material-synthesis
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Add Your Data
Place your experimental data in `data/experimental_photocatalysis.csv`:
```csv
Sample,Crystallite size (nm),Surface area (m2 g^-1),Band gap direct (eV),Band gap indirect (eV),Rate constant (k x 10^-1 / h),Rate of hydrogen evolved (umol (h g.cat)^-1),Quantum yield (%)
μS 10 min,47,2.3,1.93,1.83,1.390,99.2,0.14
...
```

### 4. Generate Predictions and Plots

#### Train ML models and generate theoretical values:
```bash
python train_ml_models.py
```

#### Create parity plots:
```bash
python generate_parity_plots.py
```

#### Run 100 iterations for statistical validation:
```bash
python run_n_iterations.py
```

## Output Files

All outputs are saved to the `output/` directory:

### Single Run
- `parity_[PropertyName].png` - Individual parity plots (7 files)
- `comparison_table.csv` - Side-by-side experimental vs theoretical comparison
- `summary_report.txt` - Statistical summary (R², MAE, Mean Error)

### 100 Iterations
- `ml_iterations_all_results_[timestamp].xlsx` - All 100 runs with metrics for each property
- `ml_iterations_summary_[timestamp].xlsx` - Summary statistics (mean, std, min, max, median)

## Properties Analyzed

1. Crystallite size (nm)
2. Surface area (m² g⁻¹)
3. Band gap direct (eV)
4. Band gap indirect (eV)
5. Rate constant (k × 10⁻¹ / h)
6. Rate of hydrogen evolved (μmol (h g.cat)⁻¹)
7. Quantum yield (%)

## Machine Learning Approach

The models are trained on synthetic data derived from literature values for ZnFe₂O₄ and similar spinel ferrites:
- **Training samples**: 18 datapoints (microwave vs conventional synthesis)
- **Features**: Synthesis time and method
- **Algorithms**: Random Forest & Gradient Boosting (best performer selected via cross-validation)

## GitHub Actions Workflow

The repository includes automated CI/CD:

### On Every Push/PR
- Trains ML models
- Generates parity plots
- Uploads artifacts (plots, tables, reports)

### Weekly or Manual Trigger
- Runs 100 iterations for comprehensive validation
- Generates statistical summary
- Uploads iteration results

**Manually trigger 100 iterations**: Go to Actions → ML Training and Parity Plots → Run workflow

## Project Structure

```
material-synthesis/
├── .github/workflows/
│   └── ml-pipeline.yml          # GitHub Actions workflow
├── data/
│   └── experimental_photocatalysis.csv  # Your experimental data
├── output/                      # Generated plots and reports
├── train_ml_models.py          # ML model training
├── generate_parity_plots.py    # Parity plot generation
├── run_n_iterations.py         # 100 iteration runner
├── requirements.txt            # Python dependencies
└── README.md
```

## Requirements

- Python 3.8+
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- openpyxl

## For Your Teacher

This project demonstrates:
- **Machine Learning**: Ensemble methods (Random Forest, Gradient Boosting) for materials property prediction
- **Statistical Validation**: R² scores, Mean Absolute Error, percentage error analysis
- **Reproducibility**: Automated workflows, version control, comprehensive documentation
- **Data Visualization**: Publication-ready parity plots with error bands

## License

MIT License - Feel free to use for research and education

## Citation

If you use this tool in your research, please cite:
```
[Your Paper Title]
[Authors]
[Journal/Conference, Year]
```

## Contact

For questions or issues, please open an issue on GitHub.
