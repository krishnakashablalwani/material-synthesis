import pandas as pd
import numpy as np
import sys
import os
import subprocess
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

n_iterations = 10

from train_ml_models import PhotocatalysisPredictor
from sklearn.metrics import r2_score, mean_absolute_error


def run_single_iteration(exp_df):
    import io
    import contextlib
    
    f = io.StringIO()
    with contextlib.redirect_stdout(f):
        predictor = PhotocatalysisPredictor()
        predictor.train_models()
        predictions_df = predictor.predict_properties(exp_df)
    
    stats = {}
    
    properties = [col for col in exp_df.columns if col != 'Sample']
    
    for prop in properties:
        exp_vals = exp_df[prop].values
        theo_vals = predictions_df[prop].values
        
        r2 = r2_score(theo_vals, exp_vals)
        mae = mean_absolute_error(theo_vals, exp_vals)
        percent_error = np.abs((exp_vals - theo_vals) / theo_vals * 100)
        mean_error = percent_error.mean()
        
        stats[prop] = {
            'R2': r2,
            'MAE': mae,
            'Mean_Error': mean_error
        }
    
    return stats


def main():
    print("\n" + "="*70)
    print("RUNNING 100 ML TRAINING & EVALUATION ITERATIONS")
    print("="*70)
    print("\nThis will take a few minutes...")
    print("Collecting R2, MAE, and Mean Error for each property across 100 runs")
    
    exp_df = pd.read_csv('data/experimental_photocatalysis.csv')
    print(f"\nLoaded {len(exp_df)} samples: {', '.join(exp_df['Sample'].tolist())}")
    
    all_results = []
    
    for i in range(n_iterations):
        print(f"\r[{i+1}/{n_iterations}] Running iteration {i+1}...", end='', flush=True)
        
        try:
            stats = run_single_iteration(exp_df)
            
            result_row = {'Iteration': i + 1}
            
            for prop_name, metrics in stats.items():
                result_row[f'{prop_name}_R2'] = metrics['R2']
                result_row[f'{prop_name}_MAE'] = metrics['MAE']
                result_row[f'{prop_name}_Mean_Error'] = metrics['Mean_Error']
            
            all_results.append(result_row)
            
        except Exception as e:
            print(f"\nError in iteration {i+1}: {e}")
            continue
    
    print(f"\n\nCompleted {len(all_results)} successful iterations!")
    
    df_results = pd.DataFrame(all_results)
    
    print("\n" + "="*70)
    print("CALCULATING SUMMARY STATISTICS")
    print("="*70)
    
    summary_data = []
    
    properties = set()
    for col in df_results.columns:
        if col != 'Iteration':
            prop = col.rsplit('_', 1)[0]
            properties.add(prop)
    
    for prop in sorted(properties):
        r2_col = f'{prop}_R2'
        mae_col = f'{prop}_MAE'
        error_col = f'{prop}_Mean_Error'
        
        if r2_col in df_results.columns:
            summary_data.append({
                'Property': prop,
                'Metric': 'R2',
                'Mean': df_results[r2_col].mean(),
                'Std': df_results[r2_col].std(),
                'Min': df_results[r2_col].min(),
                'Max': df_results[r2_col].max(),
                'Median': df_results[r2_col].median()
            })
        
        if mae_col in df_results.columns:
            summary_data.append({
                'Property': prop,
                'Metric': 'MAE',
                'Mean': df_results[mae_col].mean(),
                'Std': df_results[mae_col].std(),
                'Min': df_results[mae_col].min(),
                'Max': df_results[mae_col].max(),
                'Median': df_results[mae_col].median()
            })
        
        if error_col in df_results.columns:
            summary_data.append({
                'Property': prop,
                'Metric': 'Mean Error (%)',
                'Mean': df_results[error_col].mean(),
                'Std': df_results[error_col].std(),
                'Min': df_results[error_col].min(),
                'Max': df_results[error_col].max(),
                'Median': df_results[error_col].median()
            })
    
    df_summary = pd.DataFrame(summary_data)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    output_dir = Path('output')
    output_dir.mkdir(exist_ok=True)
    
    all_results_file = output_dir / f'ml_iterations_all_results_{n_iterations}.xlsx'
    df_results.to_excel(all_results_file, index=False)
    print(f"\nSaved all iteration results: {all_results_file}")
    
    summary_file = output_dir / f'ml_iterations_summary_{n_iterations}.xlsx'
    
    with pd.ExcelWriter(summary_file, engine='openpyxl') as writer:
        df_summary.to_excel(writer, sheet_name='Summary', index=False)
        df_results.to_excel(writer, sheet_name='All_Iterations', index=False)
    
    print(f"Saved summary statistics: {summary_file}")
    
    print("\n" + "="*70)
    print("SUMMARY STATISTICS (Mean +/- Std)")
    print("="*70)
    
    for prop in sorted(properties):
        print(f"\n{prop}:")
        prop_summary = df_summary[df_summary['Property'] == prop]
        
        for _, row in prop_summary.iterrows():
            metric = row['Metric']
            mean = row['Mean']
            std = row['Std']
            print(f"  {metric}: {mean:.4f} +/- {std:.4f}")
    
    print("\n" + "="*70)
    print(f"COMPLETE! Ran {len(all_results)} successful iterations out of {n_iterations}")
    print(f"Results saved to: {output_dir}/")
    print("="*70)
    
    print("\nPushing results to GitHub...")
    try:
        subprocess.run(['git', 'add', 'output/ml_iterations_*.xlsx'], check=True, cwd=os.getcwd())
        subprocess.run(['git', 'commit', '-m', f'Add ML iteration results: {n_iterations} iterations completed'], 
                      check=True, cwd=os.getcwd())
        subprocess.run(['git', 'push'], check=True, cwd=os.getcwd())
        print("✓ Results pushed to GitHub successfully!")
    except subprocess.CalledProcessError as e:
        print(f"⚠ Warning: Could not push to GitHub: {e}")
    
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
