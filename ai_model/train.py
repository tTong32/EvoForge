"""
Step 5: Complete Training Pipeline
This is the main script that runs everything.
"""

from data_loader import DataLoader
from model import FitnessPredictor
from explainer import FitnessExplainer
from analyzer import EcosystemAnalyzer
import matplotlib.pyplot as plt
from pathlib import Path


def main():
    print("=" * 60)
    print("AI Model Training")
    print("=" * 60)
    
    # Create outputs directory
    outputs_dir = Path("../data/outputs")
    outputs_dir.mkdir(parents=True, exist_ok=True)
    
    # === STEP 1: Load Data ===
    print(" STEP 1: Loading Data")
    loader = DataLoader()
    df = loader.load_all_data()
    X, y = loader.prepare_features(df)
    X_clean, y_clean = loader.clean_data(X, y)
    
    # === STEP 2: Train Model ===
    print("STEP 2: Training Model")
    predictor = FitnessPredictor()
    results = predictor.train(X_clean, y_clean, verbose=True)
    
    # Show feature importance
    print(" Top 15 Most Important Features:")
    importance = predictor.get_feature_importance(top_n=15)
    print(importance.to_string(index=False))
    
    # Save model to outputs directory
    model_path = outputs_dir / "fitness_model.pkl"
    predictor.save(str(model_path))
    
    # === STEP 3: Create Explainer ===
    print(" STEP 3: Creating Explainer")
    explainer = FitnessExplainer(
        predictor.model,
        X_clean.columns.tolist(),
        loader.gene_names
    )
    
    # Explain a few examples
    print(" Example Explanations:")
    for i in [0, 10, 50]:  # Explain first, 11th, 51st samples
        if i < len(X_clean):
            print(f"\n--- Example {i+1} ---")
            sample = X_clean.iloc[[i]]
            explanation = explainer.explain_prediction(sample, threshold=0.03)
            print(explanation)
    
    # === STEP 4: Analyze Patterns ===
    print(" STEP 4: Analyzing Patterns")
    analyzer = EcosystemAnalyzer(X_clean, y_clean, loader.gene_names, outputs_dir=str(outputs_dir))
    
    # Find niches
    niches = analyzer.find_niches(n_clusters=5)
    print(" Ecological Niches Found:")
    print(niches.to_string(index=False))
    
    # Analyze trade-offs
    print(" Analyzing Trade-offs:")
    analyzer.analyze_tradeoffs('speed', 'size')
    analyzer.analyze_tradeoffs('thermal_tolerance', 'metabolism_rate')
    
    print("\n" + "=" * 60)
    print(" Training Complete!")
    print("=" * 60)
    print(" Outputs:")
    print(f"   - {model_path} (trained model)")
    print(f"   - {outputs_dir}/tradeoff_*.png (trade-off visualizations)")


if __name__ == "__main__":
    main()