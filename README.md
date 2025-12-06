# Material Synthesis Parity Plot Project

## What This Does
This project compares your experimental data with AI-generated theoretical values and creates parity plots.

## Files You Need
1. `experimental_photocatalysis.csv` - Your experimental data (already created)
2. `theoretical_photocatalysis.xlsx` - AI baseline values (already created)

## How to Run

### Simple Way (Just double-click!)
1. Open `generate_parity_plots.py` in VS Code
2. Press F5 (or click Run)

### Command Line Way
```bash
python generate_parity_plots.py
```

## What You'll Get
After running, you'll get 3 files:

1. **parity_plots_all.png** - Beautiful parity plots for all properties
   - Shows experimental vs theoretical values
   - Includes R² score and error statistics
   
2. **comparison_table.csv** - Detailed comparison table
   - Shows every measurement side-by-side
   - Calculates percent error for each

3. **summary_report.txt** - Overall statistics
   - R² scores for each property
   - Mean absolute errors
   - Average percent errors

## Understanding the Plots

**Perfect Agreement Line (red dashed)**: If all points fall on this line, experimental = theoretical exactly

**±10% Error Band (gray)**: Acceptable range for most measurements

**R² Score**: How well the data matches (1.0 = perfect match)

**Mean % Error**: Average difference between experimental and theoretical

## Need Help?

If something doesn't work:
1. Make sure the two data files are in the same folder as the script
2. Check that you have installed the required packages (pandas, openpyxl, matplotlib, seaborn, scikit-learn)
3. Python version should be 3.8 or newer

## For Your Teacher

This script generates publication-ready parity plots comparing experimental photocatalysis data with theoretical predictions. All statistics are automatically calculated and displayed.
