import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import r2_score, mean_absolute_error

sns.set_style("whitegrid")
plt.rcParams['font.size'] = 10


def load_data():
    exp = pd.read_csv('data/experimental_photocatalysis.csv')
    theo = pd.read_excel('data/theoretical_photocatalysis.xlsx')
    return exp, theo


def create_parity_plots(exp, theo, output_dir='.'):
    properties = [col for col in exp.columns if col != 'Sample']
    
    output_path_dir = Path(output_dir) / 'output'
    output_path_dir.mkdir(parents=True, exist_ok=True)
    
    print("\nGenerating parity plots...")
    print("="*60)
    
    saved_files = []
    
    for prop in properties:
        fig, ax = plt.subplots(figsize=(8, 7))
        
        exp_vals = exp[prop].values
        theo_vals = theo[prop].values
        
        r2 = r2_score(theo_vals, exp_vals)
        mae = mean_absolute_error(theo_vals, exp_vals)
        percent_error = np.abs((exp_vals - theo_vals) / theo_vals * 100)
        mean_error = percent_error.mean()
        
        ax.scatter(theo_vals, exp_vals, s=150, alpha=0.7, 
                  edgecolors='black', linewidth=1.5, c='steelblue')
        
        min_val = min(exp_vals.min(), theo_vals.min())
        max_val = max(exp_vals.max(), theo_vals.max())
        padding = (max_val - min_val) * 0.1
        
        ax.plot([min_val-padding, max_val+padding], 
               [min_val-padding, max_val+padding], 
               'r--', lw=2, alpha=0.7, label='Perfect Agreement')
        
        x_range = np.linspace(min_val-padding, max_val+padding, 100)
        ax.fill_between(x_range, x_range*0.9, x_range*1.1, 
                       alpha=0.15, color='gray', label='±10% Error')
        
        ax.set_xlabel('Theoretical', fontsize=13, fontweight='bold')
        ax.set_ylabel('Experimental', fontsize=13, fontweight='bold')
        ax.set_title(prop, fontsize=14, fontweight='bold', pad=15)
        ax.legend(loc='upper left', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        stats_text = f'R² = {r2:.3f}\nMAE = {mae:.3f}\nError = {mean_error:.1f}%'
        ax.text(0.95, 0.05, stats_text, transform=ax.transAxes,
               fontsize=11, verticalalignment='bottom', horizontalalignment='right',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        safe_name = prop.replace('/', '_').replace('(', '').replace(')', '').replace(' ', '_').replace('^', '')
        output_path = Path(output_dir) / f'output/parity_{safe_name}.png'
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        saved_files.append(output_path)
        
        print(f"\n{prop}:")
        print(f"  R² Score: {r2:.3f}")
        print(f"  Mean Absolute Error: {mae:.3f}")
        print(f"  Mean % Error: {mean_error:.1f}%")
        print(f"  ✓ Saved: {output_path}")
    
    print("\n" + "="*60)
    return saved_files


def create_comparison_table(exp, theo, output_dir='.'):
    comparison = []
    for idx, sample in enumerate(exp['Sample']):
        for col in exp.columns:
            if col == 'Sample':
                continue
            exp_val = exp.loc[idx, col]
            theo_val = theo.loc[idx, col]
            error = abs(exp_val - theo_val) / theo_val * 100
            
            comparison.append({
                'Sample': sample,
                'Property': col,
                'Experimental': exp_val,
                'Theoretical': theo_val,
                'Error (%)': round(error, 2)
            })
    
    df_comp = pd.DataFrame(comparison)
    output_path = Path(output_dir) / 'output/comparison_table.csv'
    df_comp.to_csv(output_path, index=False)
    print(f"✓ Saved: {output_path}")
    
    return df_comp


def create_summary_report(exp, theo, output_dir='.'):
    report = []
    report.append("SUMMARY REPORT")
    report.append("="*60)
    report.append(f"\nNumber of samples: {len(exp)}")
    report.append(f"Number of properties: {len(exp.columns) - 1}")
    report.append(f"\nSamples: {', '.join(exp['Sample'].tolist())}")
    
    report.append("\n\nOVERALL STATISTICS:")
    report.append("-"*60)
    
    for col in exp.columns:
        if col == 'Sample':
            continue
        exp_vals = exp[col].values
        theo_vals = theo[col].values
        
        r2 = r2_score(theo_vals, exp_vals)
        mae = mean_absolute_error(theo_vals, exp_vals)
        mean_error = np.abs((exp_vals - theo_vals) / theo_vals * 100).mean()
        
        report.append(f"\n{col}:")
        report.append(f"  R² = {r2:.3f}")
        report.append(f"  MAE = {mae:.3f}")
        report.append(f"  Mean Error = {mean_error:.1f}%")
    
    report_text = "\n".join(report)
    
    output_path = Path(output_dir) / 'output/summary_report.txt'
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(report_text)
    
    print(f"✓ Saved: {output_path}")
    print("\n" + report_text)


def main():
    print("\n" + "="*60)
    print("PARITY PLOT GENERATOR")
    print("="*60)
    
    try:
        print("\nLoading data...")
        exp, theo = load_data()
        print(f"✓ Loaded {len(exp)} samples")
        
        saved_files = create_parity_plots(exp, theo)
        
        print("\nCreating comparison table...")
        create_comparison_table(exp, theo)
        
        print("\nCreating summary report...")
        create_summary_report(exp, theo)
        
        print("\n" + "="*60)
        print(f"DONE! Created {len(saved_files)} individual parity plots:")
        for f in saved_files:
            print(f"  - {f.name}")
        print("\nAlso created:")
        print("  - comparison_table.csv")
        print("  - summary_report.txt")
        print("="*60 + "\n")
        
    except FileNotFoundError as e:
        print(f"\n❌ Error: Could not find required files")
        print(f"   Details: {e}")
        print("   Make sure these files exist:")
        print("   - data/experimental_photocatalysis.csv")
        print("   - data/theoretical_photocatalysis.xlsx")
    except Exception as e:
        import traceback
        print(f"\n❌ Error: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    main()
