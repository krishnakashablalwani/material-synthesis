import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import cross_val_score
import warnings
warnings.filterwarnings('ignore')


class PhotocatalysisPredictor:
    
    def __init__(self):
        self.model = None
        self.scaler = None
        self.property_names = [
            'Surface area (m2 g^-1)',
            'Band gap direct (eV)',
            'Rate constant (k x 10^-1 / h)',
            'Quantum yield (%)',
            'Crystallite size (nm)'
        ]
        
    def create_training_data(self):
        training_data = {
            'time_min': [5, 10, 10, 15, 20, 30, 30, 45, 60, 90, 120, 150, 180],
            'surface_area': [1.8, 2.3, 2.4, 2.6, 3.2, 4.6, 5.0, 5.4, 5.6, 5.5, 5.2, 5.6, 4.8],
            'bandgap_direct': [1.92, 1.93, 1.94, 1.94, 1.95, 1.93, 1.94, 1.95, 1.93, 1.94, 1.95, 1.93, 1.92],
            'rate_constant': [0.8, 1.4, 1.5, 1.8, 2.0, 2.3, 2.5, 2.4, 2.2, 1.8, 1.2, 0.8, 0.3],
            'quantum_yield': [0.08, 0.14, 0.15, 0.17, 0.18, 0.19, 0.20, 0.19, 0.18, 0.16, 0.14, 0.13, 0.12],
            'crystallite_size': [52, 47, 45, 42, 38, 35, 33, 31, 29, 28, 27, 29, 32]
        }
        
        return pd.DataFrame(training_data)
    
    def train_models(self):
        print("\nTraining multi-output AI/ML model (microwave synthesis only)...")
        print("="*60)
        
        df_train = self.create_training_data()
        X = df_train[['time_min']].values
        y = df_train[['surface_area', 'bandgap_direct', 'rate_constant', 'quantum_yield', 'crystallite_size']].values
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        rf_base = RandomForestRegressor(n_estimators=100, max_depth=5)
        gb_base = GradientBoostingRegressor(n_estimators=100, max_depth=3)
        
        rf_model = MultiOutputRegressor(rf_base)
        gb_model = MultiOutputRegressor(gb_base)
        
        rf_model.fit(X_scaled, y)
        gb_model.fit(X_scaled, y)
        
        rf_scores = []
        gb_scores = []
        for i in range(y.shape[1]):
            rf_score = cross_val_score(RandomForestRegressor(n_estimators=100, max_depth=5), 
                                      X_scaled, y[:, i], cv=3, scoring='r2').mean()
            gb_score = cross_val_score(GradientBoostingRegressor(n_estimators=100, max_depth=3), 
                                      X_scaled, y[:, i], cv=3, scoring='r2').mean()
            rf_scores.append(rf_score)
            gb_scores.append(gb_score)
        
        rf_avg = np.mean(rf_scores)
        gb_avg = np.mean(gb_scores)
        
        if rf_avg >= gb_avg:
            self.model = rf_model
            model_type = "Random Forest"
            scores = rf_scores
        else:
            self.model = gb_model
            model_type = "Gradient Boosting"
            scores = gb_scores
        
        print(f"\nModel Type: {model_type}")
        print(f"Training on {len(df_train)} microwave synthesis samples")
        print(f"\nCross-validation R² Scores:")
        for prop_name, score in zip(self.property_names, scores):
            print(f"  {prop_name}: {score:.3f}")
        print(f"  Average R²: {np.mean(scores):.3f}")
        
        print("\n" + "="*60)
        print("✓ Multi-output model trained successfully!")
        
    def predict_properties(self, samples_df):
        predictions = []
        
        for idx, row in samples_df.iterrows():
            sample_name = row['Sample']
            
            import re
            time_match = re.search(r'(\d+)\s*min', sample_name)
            if time_match:
                time_min = float(time_match.group(1))
            else:
                time_min = 10
            
            X_sample = np.array([[time_min]])
            X_scaled = self.scaler.transform(X_sample)
            pred_values = self.model.predict(X_scaled)[0]
            
            pred_row = {'Sample': sample_name}
            for prop_name, pred_value in zip(self.property_names, pred_values):
                pred_row[prop_name] = pred_value
            
            predictions.append(pred_row)
        
        return pd.DataFrame(predictions)


def main():
    print("\n" + "="*60)
    print("AI/ML THEORETICAL VALUE GENERATION")
    print("="*60)
    
    print("\nLoading experimental data...")
    exp_df = pd.read_csv('data/experimental_photocatalysis.csv')
    print(f"✓ Loaded {len(exp_df)} samples")
    
    predictor = PhotocatalysisPredictor()
    predictor.train_models()
    
    print("\nGenerating AI predictions...")
    predictions_df = predictor.predict_properties(exp_df)
    
    output_path = 'data/theoretical_photocatalysis.xlsx'
    predictions_df.to_excel(output_path, index=False)
    print(f"\n✓ Saved AI-generated theoretical values to: {output_path}")
    
    print("\nAI-Generated Theoretical Values:")
    print("-"*60)
    print(predictions_df.to_string(index=False))
    
    print("\n" + "="*60)
    print("DONE! Now run generate_parity_plots.py to create parity plots")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()

