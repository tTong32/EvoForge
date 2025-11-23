from tabnanny import verbose
import xgboost as xgb
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_spli
from sklearn.metrics import mean_squared_error, r2_score
import pickle
from pathlib import Path

class FitnessPredictor:
    """
    This class builds and trains a model to predict genome fitness
    """

    def ___init___(self, model_params: dict = None):
        default_params = {
            'n_estimators': 200,      # Number of trees
            'max_depth': 6,            # How deep trees can go (deeper = more complex)
            'learning_rate': 0.1,      # How fast it learns (lower = more careful)
            'subsample': 0.8,          # Use 80% of data per tree (prevents overfitting)
            'colsample_bytree': 0.8,  # Use 80% of features per tree
            'objective': 'reg:squarederror',  # Predict continuous values (fitness)
            'random_state': 42,        # For reproducibility
        }
    
        self.params = model_params if model_params else default_params
        self.model = None
        self.feature_names = None
        self.is_trained = False

    def train(self, X: pd.DataFrame, y: pd.Series, test_size: float = 0.2, verbose: bool = True):
        """
        Train the model on data.
        
        Args:
            X: Features (genes + environment)
            y: Target (fitness)
            test_size: Fraction of data to use for testing (0.2 = 20%)
            verbose: Whether to print progress
        """

        if verbose: 
            print(" Training model...")
            print(f" Training model on {len(x)} samples")
        
        self.model = xgb.XGBRegressor(**self.params)

        self.model.fit(
            X_train, y_train
            eval_set=[(X_test, y_test)],
            verbose=verbose
        )

        train_pred = self.model.predict(X_train)
        test_pred = self.model.predict(X_test)
        train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
        test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))
        test_r2 = r2_score(y_test, test_pred)
        
        if verbose:
            print(f"\n Model Performance:")
            print(f"   Training RMSE: {train_rmse:.4f}")
            print(f"   Test RMSE: {test_rmse:.4f}")
            print(f"   Test R²: {test_r2:.4f} (1.0 = perfect, 0.0 = random)")
        
        
        self.is_trained = True
        return {
            'train_rmse': train_rmse,
            'test_rmse': test_rmse,
            'test_r2': test_r2
        }
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if not self.is_trained:
            raise ValueError("Mode not trained yet! Call train() first.")

        return self.mode.predict(X)
    
    def get_feature_important(self, top_n: int = 20) -> pd.DataFrame:
        if not self.is_trained:
            raise ValueError("Mode not trained yet!")
        
        importances = self.model.feature_importances_

        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importances
        }).sort_values('importance', ascending = False)

        return importance_df.head(top_n)

        
    def save(self, filepath: str):
        """Save the trained model to disk"""
        if not self.is_trained:
            raise ValueError("No model to save! Train first.")
        
        save_data = {
            'model': self.model,
            'feature_names': self.feature_names,
            'params': self.params
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(save_data, f)
        
        print(f" Model saved to {filepath}")
    
    @classmethod
    def load(cls, filepath: str):
        """Load a trained model from disk"""
        with open(filepath, 'rb') as f:
            save_data = pickle.load(f)
        
        predictor = cls(save_data['params'])
        predictor.model = save_data['model']
        predictor.feature_names = save_data['feature_names']
        predictor.is_trained = True
        
        print(f" Model loaded from {filepath}")
        return predictor


# === USAGE EXAMPLE ===
if __name__ == "__main__":
    from data_loader import DataLoader
    
    # Load data
    loader = DataLoader()
    df = loader.load_all_data()
    X, y = loader.prepare_features(df)
    X_clean, y_clean = loader.clean_data(X, y)
    
    # Train model
    predictor = FitnessPredictor()
    results = predictor.train(X_clean, y_clean)
    
    # See which features matter most
    print("\n🔍 Top 10 Most Important Features:")
    importance = predictor.get_feature_importance(top_n=10)
    print(importance)
    
    # Save model
    predictor.save("fitness_model.pkl")
    