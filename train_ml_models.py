import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
import warnings
warnings.filterwarnings('ignore')


class PhotocatalysisPredictor:
    
    def __init__(self):
        self.models = {}
        self.scalers = {}
        
    def create_training_data(self):
        training_data = {
            'time_min': [5, 10, 10, 15, 20, 30, 30, 45, 60, 90, 120, 150, 180,
                        180, 240, 300, 360, 480, 600],
            
            'synthesis_method': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                0, 0, 0, 0, 0, 0],
            
            'crystallite_size': [52, 47, 45, 42, 38, 35, 33, 31, 29, 28, 27, 29, 32,
                                53, 55, 58, 62, 65, 68],
            
            'surface_area': [1.8, 2.3, 2.4, 2.6, 3.2, 4.6, 5.0, 5.4, 5.6, 5.5, 5.2, 5.6, 4.8,
                           2.2, 2.0, 1.8, 1.6, 1.5, 1.4],
            
            'bandgap_direct': [1.92, 1.93, 1.94, 1.94, 1.95, 1.93, 1.94, 1.95, 1.93, 1.94, 1.95, 1.93, 1.92,
                             1.90, 1.90, 1.91, 1.91, 1.92, 1.92],
            
            'bandgap_indirect': [1.82, 1.83, 1.84, 1.84, 1.85, 1.83, 1.84, 1.85, 1.83, 1.84, 1.85, 1.83, 1.82,
                               1.81, 1.81, 1.82, 1.82, 1.83, 1.83],
            
            'rate_constant': [0.8, 1.4, 1.5, 1.8, 2.0, 2.3, 2.5, 2.4, 2.2, 1.8, 1.2, 0.8, 0.3,
                            2.1, 1.9, 1.7, 1.5, 1.3, 1.1],
            
            'h2_evolved': [55, 100, 105, 115, 125, 134, 140, 135, 125, 110, 95, 92, 85,
                         32, 35, 38, 40, 42, 45],
            
            'quantum_yield': [0.08, 0.14, 0.15, 0.17, 0.18, 0.19, 0.20, 0.19, 0.18, 0.16, 0.14, 0.13, 0.12,
                            0.05, 0.05, 0.06, 0.06, 0.07, 0.07]
        }
        
        return pd.DataFrame(training_data)
    
    def train_models(self):
        print("\nTraining AI/ML models...")
        print("="*60)
        
        df_train = self.create_training_data()
        X = df_train[['time_min', 'synthesis_method']].values
        
        properties = {
            'crystallite_size': 'Crystallite size (nm)',
            'surface_area': 'Surface area (m2 g^-1)',
            'bandgap_direct': 'Band gap direct (eV)',
            'bandgap_indirect': 'Band gap indirect (eV)',
            'rate_constant': 'Rate constant (k x 10^-1 / h)',
            'h2_evolved': 'Rate of hydrogen evolved (umol (h g.cat)^-1)',
            'quantum_yield': 'Quantum yield (%)'
        }
        
        for prop_key, prop_name in properties.items():
            y = df_train[prop_key].values
            
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            rf_model = RandomForestRegressor(n_estimators=100, max_depth=5)
            gb_model = GradientBoostingRegressor(n_estimators=100, max_depth=3)
            
            rf_model.fit(X_scaled, y)
            gb_model.fit(X_scaled, y)
            
            rf_score = cross_val_score(rf_model, X_scaled, y, cv=3, scoring='r2').mean()
            gb_score = cross_val_score(gb_model, X_scaled, y, cv=3, scoring='r2').mean()
            
            if rf_score >= gb_score:
                self.models[prop_key] = rf_model
                best_type = "Random Forest"
                best_score = rf_score
            else:
                self.models[prop_key] = gb_model
                best_type = "Gradient Boosting"
                best_score = gb_score
            
            self.scalers[prop_key] = scaler
            
            print(f"\n{prop_name}:")
            print(f"  Model: {best_type}")
            print(f"  Cross-val R² Score: {best_score:.3f}")
        
        print("\n" + "="*60)
        print("✓ All models trained successfully!")
        
    def predict_properties(self, samples_df):
        predictions = []
        
        for idx, row in samples_df.iterrows():
            sample_name = row['Sample']
            
            if 'μS' in sample_name or 'microwave' in sample_name.lower():
                synthesis_method = 1
            else:
                synthesis_method = 0
            
            import re
            time_match = re.search(r'(\d+)\s*min', sample_name)
            if time_match:
                time_min = float(time_match.group(1))
            else:
                time_min = 10
            
            X_sample = np.array([[time_min, synthesis_method]])
            pred_row = {'Sample': sample_name}
            
            for prop_key in self.models.keys():
                scaler = self.scalers[prop_key]
                model = self.models[prop_key]
                
                X_scaled = scaler.transform(X_sample)
                pred_value = model.predict(X_scaled)[0]
                
                if prop_key == 'crystallite_size':
                    pred_row['Crystallite size (nm)'] = pred_value
                elif prop_key == 'surface_area':
                    pred_row['Surface area (m2 g^-1)'] = pred_value
                elif prop_key == 'bandgap_direct':
                    pred_row['Band gap direct (eV)'] = pred_value
                elif prop_key == 'bandgap_indirect':
                    pred_row['Band gap indirect (eV)'] = pred_value
                elif prop_key == 'rate_constant':
                    pred_row['Rate constant (k x 10^-1 / h)'] = pred_value
                elif prop_key == 'h2_evolved':
                    pred_row['Rate of hydrogen evolved (umol (h g.cat)^-1)'] = pred_value
                elif prop_key == 'quantum_yield':
                    pred_row['Quantum yield (%)'] = pred_value
            
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

