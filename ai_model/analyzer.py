
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict

class EcosystemAnalyzer:
    """
    Analyzes patterns in genome-environment relationships.
    
    Discovers things like:
    - What strategies work in different environments?
    - Are there distinct niches?
    - What are the trade-offs?
    """
    
    def __init__(self, X: pd.DataFrame, y: pd.Series, gene_names: Dict[int, str]):
        """
        Initialize analyzer.
        
        Args:
            X: Features dataframe
            y: Fitness values
            gene_names: Gene index -> name mapping
        """
        self.X = X
        self.y = y
        self.gene_names = gene_names
    
    def find_niches(self, n_clusters: int = 5) -> pd.DataFrame:
        """
        Find distinct ecological niches using clustering.
        
        A niche is a combination of genome traits + environment
        where organisms with similar traits cluster together.
        
        Args:
            n_clusters: Number of niches to find
            
        Returns:
            DataFrame with niche assignments
        """
        print(f" Finding {n_clusters} ecological niches...")
        
        # Use PCA to reduce dimensions (makes clustering easier)
        pca = PCA(n_components=10)  # Reduce to 10 dimensions
        X_reduced = pca.fit_transform(self.X)
        
        # Cluster the data
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        niches = kmeans.fit_predict(X_reduced)
        
        # Analyze each niche
        results = []
        for niche_id in range(n_clusters):
            niche_mask = niches == niche_id
            niche_X = self.X[niche_mask]
            niche_y = self.y[niche_mask]
            
            # Average traits in this niche
            avg_fitness = niche_y.mean()
            avg_traits = niche_X.mean()
            
            # Find most distinctive traits
            global_avg = self.X.mean()
            trait_diffs = (avg_traits - global_avg).abs().sort_values(ascending=False)
            
            results.append({
                'niche_id': niche_id,
                'size': niche_mask.sum(),
                'avg_fitness': avg_fitness,
                'top_trait': trait_diffs.index[0],
                'top_trait_value': avg_traits[trait_diffs.index[0]],
            })
        
        return pd.DataFrame(results)
    
    def analyze_tradeoffs(self, trait1: str, trait2: str):
        """
        Analyze trade-offs between two traits.
        
        Example: Speed vs Size
        - Fast organisms might be smaller
        - Large organisms might be slower
        
        Args:
            trait1: Name of first trait
            trait2: Name of second trait
        """
        if trait1 not in self.X.columns or trait2 not in self.X.columns:
            print(f"⚠️  Traits not found: {trait1}, {trait2}")
            return
        
        # Calculate correlation
        correlation = self.X[trait1].corr(self.X[trait2])
        
        print(f"\n📊 Trade-off Analysis: {trait1} vs {trait2}")
        print(f"   Correlation: {correlation:.3f}")
        
        if correlation < -0.3:
            print("   ✅ Strong trade-off (negative correlation)")
        elif correlation > 0.3:
            print("   ✅ Positive relationship (no trade-off)")
        else:
            print("   ➖ Weak relationship")
        
        # Plot
        plt.figure(figsize=(10, 6))
        plt.scatter(self.X[trait1], self.X[trait2], 
                   c=self.y, cmap='viridis', alpha=0.6)
        plt.colorbar(label='Fitness')
        plt.xlabel(trait1)
        plt.ylabel(trait2)
        plt.title(f'Trade-off: {trait1} vs {trait2}')
        plt.tight_layout()
        plt.savefig(f'tradeoff_{trait1}_vs_{trait2}.png', dpi=150)
        print(f"   💾 Saved plot to tradeoff_{trait1}_vs_{trait2}.png")
    
    def find_optimal_genomes(self, environment: Dict[str, float], 
                            top_n: int = 10) -> pd.DataFrame:
        """
        Find genomes that would thrive in a given environment.
        
        Args:
            environment: Dictionary of environment features
            top_n: How many top genomes to return
            
        Returns:
            DataFrame of optimal genomes
        """
        print(f"🎯 Finding optimal genomes for environment...")
        
        # Filter data to similar environments
        # (This is simplified - you'd want more sophisticated matching)
        env_cols = [col for col in self.X.columns if col.startswith('env_') or 
                   col.startswith('terrain_') or col.startswith('resource_')]
        
        # Find samples with similar environment
        # (Using simple distance metric)
        distances = []
        for idx, row in self.X.iterrows():
            dist = 0.0
            for key, target_val in environment.items():
                if key in self.X.columns:
                    dist += abs(row[key] - target_val) ** 2
            distances.append(np.sqrt(dist))
        
        # Get top N by fitness
        top_indices = self.y.nlargest(top_n).index
        
        results = []
        for idx in top_indices:
            results.append({
                'genome_index': idx,
                'fitness': self.y[idx],
                'genes': self.X.loc[idx, [col for col in self.X.columns 
                                         if col in self.gene_names.values()]].to_dict()
            })
        
        return pd.DataFrame(results)


# === USAGE EXAMPLE ===
if __name__ == "__main__":
    from data_loader import DataLoader
    
    loader = DataLoader()
    df = loader.load_all_data()
    X, y = loader.prepare_features(df)
    X_clean, y_clean = loader.clean_data(X, y)
    
    analyzer = EcosystemAnalyzer(X_clean, y_clean, loader.gene_names)
    
    # Find niches
    niches = analyzer.find_niches(n_clusters=5)
    print("\n", niches)
    
    # Analyze trade-offs
    analyzer.analyze_tradeoffs('speed', 'size')
    analyzer.analyze_tradeoffs('thermal_tolerance', 'metabolism_rate')