import shap
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple

class FitnessExplainer:

    def __init__(self, model, feature_names: List[str], gene_names: Dict[int, str]):
        """
        Initialize the explainer.
        
        Args:
            model: Trained XGBoost model
            feature_names: List of feature names (from X.columns)
            gene_names: Mapping of gene index -> gene name
        """
        self.model = model
        self.feature_names = feature_names
        self.gene_names = gene_names
        
        # Create SHAP explainer (this is the magic!)
        # TreeExplainer is optimized for tree-based models like XGBoost
        self.explainer = shap.TreeExplainer(model)
        
        print(" SHAP explainer initialized")
    
    def explain_prediction(self, X_sample: pd.DataFrame, threshold: float = 0.05) -> str:
        # Get SHAP values (contribution of each feature)
        shap_values = self.explainer.shap_values(X_sample)
        
        # Get the actual feature values
        feature_values = X_sample.iloc[0].values
        
        # Get the base prediction (what model predicts on average)
        base_value = self.explainer.expected_value
        
        # Get the actual prediction
        prediction = self.model.predict(X_sample)[0]
        
        # Build explanation
        explanation_parts = []
        explanation_parts.append(
            f" Predicted Fitness: {prediction:.4f} "
            f"(baseline: {base_value:.4f})"
        )
        explanation_parts.append("")
        explanation_parts.append(" Key Factors:")

        feature_contributions = []
        for i, (feature_name, shap_val, feature_val) in enumerate(
            zip(self.feature_names, shap_values[0], feature_values)
        ):
            if abs(shap_val) >= threshold:
                feature_contributions.append((feature_name, shap_val, feature_val))

        feature_contributions.sort(key=lambda x: abs(x[1]), reverse=True)

        for feature_name, contribution, value in feature_contributions[:10]:  # Top 10
            sign = "+" if contribution > 0 else ""
            
            # Try to make it more readable
            if feature_name in self.gene_names.values():
                # It's a gene
                explanation_parts.append(
                    f"   {sign}{contribution:+.4f} from {feature_name} = {value:.3f}"
                )
            elif "x" in feature_name:
                # It's an interaction
                parts = feature_name.split("_x_")
                explanation_parts.append(
                    f"   {sign}{contribution:+.4f} from interaction: "
                    f"{parts[0]} × {parts[1]} = {value:.3f}"
                )
            else:
                # It's an environment feature
                explanation_parts.append(
                    f"   {sign}{contribution:+.4f} from {feature_name} = {value:.3f}"
                )
        
        return "\n".join(explanation_parts)

    def explain_genome_environment_math(self, genome_genes: List[float], environment: Dict[str, float]) -> str:
        explanations = []
        explanations.append(" Genome-Environment Analysis")
        explanations.append("=" * 50)
        
        # Check thermal tolerance match
        if len(genome_genes) > 17 and 'env_temp_avg' in environment:
            thermal_tol = genome_genes[17]
            temp = environment['env_temp_avg']
            match_score = 1.0 - abs(thermal_tol - temp)  # Closer = better match
            
            explanations.append(
                f"\n Thermal Adaptation:"
            )
            explanations.append(
                f"   Gene 17 (thermal_tolerance) = {thermal_tol:.2f}"
            )
            explanations.append(
                f"   Environment temperature = {temp:.2f}"
            )
            explanations.append(
                f"   Match score = {match_score:.2f} "
                f"({'Good match' if match_score > 0.8 else 'Mismatch'})"
            )
        
        # Check foraging vs resources
        if len(genome_genes) > 20 and 'resource_plant_avg' in environment:
            foraging = genome_genes[20]
            plants = environment['resource_plant_avg']
            
            explanations.append(
                f"\n Foraging Strategy:"
            )
            explanations.append(
                f"   Gene 20 (foraging_bias) = {foraging:.2f}"
            )
            explanations.append(
                f"   Plant resources = {plants:.2f}"
            )
            
            if foraging > 0.7 and plants > 0.5:
                explanations.append(
                    "   High foraging drive matches abundant resources"
                )
            elif foraging > 0.7 and plants < 0.3:
                explanations.append(
                    "   High foraging drive but low resources - may struggle"
                )
            elif foraging < 0.3 and plants > 0.5:
                explanations.append(
                    "   Low foraging drive despite abundant resources - inefficient"
                )
        
        # Check movement cost vs terrain
        if len(genome_genes) > 3 and 'env_elevation_avg' in environment:
            movement_cost = genome_genes[3]
            elevation = environment['env_elevation_avg']
            
            explanations.append(
                f"\n Movement Efficiency:"
            )
            explanations.append(
                f"   Gene 3 (movement_cost) = {movement_cost:.2f} "
                f"({'Low' if movement_cost < 0.5 else 'High'})"
            )
            explanations.append(
                f"   Elevation = {elevation:.0f} "
                f"({'Mountainous' if elevation > 500 else 'Lowland'})"
            )
            
            if movement_cost < 0.3 and elevation > 500:
                explanations.append(
                    "   Low movement cost helps in difficult terrain"
                )
        
        return "\n".join(explanations)

    def get_feature_attribution(self, X_sample: pd.DataFrame) -> pd.DataFrame:

        """ 
        Get detailed feature attribution for a prediction.

        Returns a DataFrame with each feature's contribution
        """

        shap_values = self.explainer.shap_values(X_sample)
            
        attribution_df = pd.DataFrame({
            'feature': self.feature_names,
            'value': X_sample.iloc[0].values,
            'shap_value': shap_values[0],
            'abs_shap_value': np.abs(shap_values[0])
        }).sort_values('abs_shap_value', ascending=False)
        
        return attribution_df

# === USAGE EXAMPLE ===
if __name__ == "__main__":
    from data_loader import DataLoader
    from model import FitnessPredictor
    
    # Load and prepare data
    loader = DataLoader()
    df = loader.load_all_data()
    X, y = loader.prepare_features(df)
    X_clean, y_clean = loader.clean_data(X, y)
    
    # Train model
    predictor = FitnessPredictor()
    predictor.train(X_clean, y_clean)
    
    # Create explainer
    explainer = FitnessExplainer(
        predictor.model,
        X_clean.columns.tolist(),
        loader.gene_names
    )
    
    # Explain a prediction
    sample = X_clean.iloc[[0]]  # First sample
    explanation = explainer.explain_prediction(sample)
    print(explanation)