import pandas as pd
import numpy as np
from pathlib import Path
from typing import Any, Tuple, List

class DataLoader:

    def __init__(self, logs_dir: str = "../data/logs"):
        self.logs_dir = Path(logs_dir)

        # Map genes from genetics.rs file
        self.gene_names = {
            0: "speed",
            1: "size", 
            2: "metabolism_rate",
            3: "movement_cost",
            4: "max_energy",
            5: "reproduction_cooldown",
            6: "reproduction_threshold",
            7: "sensory_range",
            8: "aggression",
            9: "boldness",
            10: "speed_fast_twitch",
            11: "speed_endurance",
            12: "structural_density",
            13: "metabolic_flexibility",
            14: "reproductive_investment",
            15: "sensory_focus",
            16: "social_sensitivity",
            17: "thermal_tolerance",  # Important for temperature matching!
            18: "mutation_control",
            19: "developmental_plasticity",
            20: "foraging_bias",
            21: "risk_tolerance",
            22: "exploration_drive",
            23: "clutch_size",
            24: "offspring_energy_share",
            25: "hunger_memory",
            26: "threat_decay",
            27: "resource_selectivity",
            28: "migration_drive",
            29: "plant_consumption_rate",
            30: "meat_consumption_rate",
            31: "attack_strength",
        }
    
    def load_all_data(self) -> pd.DataFrame:

        print("Searching for CSV files...")

        csv_files = list(self.logs_dir.glob("species_fitness_*.csv"))

        if not csv_files:
            raise FileNotFoundError(f"No CSV files found in {self.logs_dir}. Please run the simulator to generate logs.")

        print(f"Found {len(csv_files)} CSV files.")

        dataframes = []
        for file in csv_files:
            print(f"Loading {file}...")
            df = pd.read_csv(file)
            dataframes.append(df)
        
        combined_df = pd.concat(dataframes, ignore_index=True)

        print(f"Loaded {len(combined_df)} total rows")
        return combined_df

    def prepare_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        print("Preparing features...")

        gene_cols = [f"gene_{i}" for i in range(32)]
        gene_features = df[gene_cols].copy()

        gene_features.columns = [self.gene_names.get(i, f"gene_{i}") for i in range(32)]
        print(f"Extracted {len(gene_cols)} gene features")

        env_cols = [
            # Climate
            "env_temp_avg", "env_temp_min", "env_temp_max", "env_temp_std",
            "env_humidity_avg", "env_humidity_min", "env_humidity_max", "env_humidity_std",
            "env_elevation_avg", "env_elevation_min", "env_elevation_max",
            
            # Terrain (percentages)
            "terrain_ocean_pct", "terrain_plains_pct", "terrain_forest_pct",
            "terrain_desert_pct", "terrain_tundra_pct", "terrain_mountain_pct",
            "terrain_swamp_pct", "terrain_volcanic_pct",
            
            # Resources
            "resource_plant_avg", "resource_mineral_avg", "resource_sunlight_avg",
            "resource_water_avg", "resource_detritus_avg", "resource_prey_avg",
            
            # Ecosystem context
            "ecosystem_total_population", "ecosystem_species_count",
        ]

        available_env_cols = [col for col in env_cols if col in df.columns]
        env_features = df[available_env_cols].copy()

        print(f"Extracted {len(available_env_cols)} environment features!")

        interactions = []
        interaction_names = []

        if "gene_17" in df.columns and "env_temp_avg" in df.columns:
            # Thermal tolerance × Temperature
            interactions.append(df["gene_17"] * df["env_temp_avg"])
            interaction_names.append("thermal_tolerance_x_temp")
        
        if "gene_20" in df.columns and "resource_plant_avg" in df.columns:
            # Foraging bias × Plant resources
            interactions.append(df["gene_20"] * df["resource_plant_avg"])
            interaction_names.append("foraging_bias_x_plants")
        
        if "gene_3" in df.columns and "env_elevation_avg" in df.columns:
            # Movement cost × Elevation (harder to move in mountains)
            interactions.append(df["gene_3"] * df["env_elevation_avg"])
            interaction_names.append("movement_cost_x_elevation")
        
        if interactions:
            interaction_features = pd.DataFrame(
                dict(zip(interaction_names, interactions))
            )
            print(f"Created {len(interaction_names)} interaction features")
        else:
            interaction_features = pd.DataFrame()
            print("No interaction features created (missing columns)")

        x = pd.concat([gene_features, env_features, interaction_features], axis=1)

        if "population_growth_rate" in df.columns:
            y = df["population_growth_rate"].copy()
        elif "current_population" in df.columns:
            # Alternative: use population as proxy for fitness
            y = df["current_population"].copy()
            print(" CAUTION: Using 'current_population' as fitness proxy")
        else:
            raise ValueError("No fitness column found! Need 'population_growth_rate' or 'current_population'")
        
        print(f" Target variable: {y.name}")
        print(f" Final feature matrix: {x.shape[0]} samples × {x.shape[1]} features")
        
        return x, y

    def clean_data(self, x: pd.DataFrame, y: pd.Series) -> Tuple[pd.DataFrame, pd.Series]:
        print("Cleaning Data...")

        initial_count = len(x)

        valid_mask = ~(x.isna().any(axis = 1) | np.isinf(x).any(axis=1) | y.isna() | np.isinf(y))
        x_clean = x[valid_mask].copy()
        y_clean = y[valid_mask].copy()

        removed = initial_count - len(x_clean)
        print(f" Removed {removed} invalid rows ({removed/initial_count*100:.1f}%)")
        print(f" Clean data: {len(x_clean)} samples")

        return x_clean, y_clean