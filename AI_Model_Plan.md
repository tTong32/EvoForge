# AI Model Analysis Plan: Genome-Environment Fitness Prediction

## Goal

Build an interpretable AI model that analyzes and explains **why certain genomes thrive in specific environments** within the evolution simulation. The model should:

1. **Predict** genome fitness in given environments
2. **Understand** which genes matter and why
3. **Explain** genome-environment interactions
4. **Analyze** ecosystem dynamics and niche specialization

## Research Question

**"Why does this genome thrive in this environment?"**

The model should provide interpretable explanations like:
- "Gene 17 (thermal_tolerance) = 0.85 matches desert temperature (0.8), contributing +0.15 to fitness"
- "High foraging_drive (0.92) compensates for low resource density in this environment"
- "This genome's low movement_cost (0.23) allows efficient resource searching in sparse areas"

## Current State Assessment

### ✅ What Works
- Time-series logging of organisms (every 50 ticks)
- Entity tracking (can follow organisms over time)
- Rich trait data (35+ expressed traits logged)
- Behavior state tracking
- Species ID tracking

### ❌ Critical Gaps

#### Priority 1: Must Fix
1. **Genome genes not logged** - Only expressed traits logged, not raw 32 gene values
2. **Cell/environmental data not logged** - No temperature, humidity, resources, terrain
3. **Parent-child relationships not logged** - Can't track lineage or reproductive success

#### Priority 2: Strongly Recommended
4. **Offspring success not tracked** - Can't measure if offspring survived/reproduced
5. **Energy flows not tracked** - Can't compute efficiency (gained/consumed)
6. **Spatial context not logged** - No local competition, predator/prey density

#### Priority 3: Nice to Have
7. **Birth/death events not explicitly logged** - Hard to compute lifetimes
8. **No cumulative metrics** - Everything is snapshot, requires post-processing

**Current Readiness: ~30%** - Can do basic trait-survival correlation, but cannot explain genome-environment interactions.

## Required Data Schema

### Enhanced Organism Logging

Each organism log entry should include:

#### Genome Encoding (32 genes)
```
genome_gene_0, genome_gene_1, ..., genome_gene_31  // Raw gene values [0.0-1.0]
```

#### Environmental Context (from current cell)
```
cell_temperature, cell_humidity, cell_elevation, cell_terrain_type,
cell_resource_plant, cell_resource_mineral, cell_resource_sunlight,
cell_resource_water, cell_resource_detritus, cell_resource_prey,
cell_resource_pressure_plant, cell_resource_pressure_mineral,
cell_resource_pressure_sunlight, cell_resource_pressure_water,
cell_resource_pressure_detritus, cell_resource_pressure_prey,
cell_plant_community_diversity,
cell_dead_organic_matter, cell_animal_nutrients
```

#### Spatial Context (derived features)
```
local_organism_density,        // Organisms in sensory_range
local_predator_count, local_prey_count,
local_conspecific_count,        // Same species
local_competitor_count,         // Different species, same type
distance_to_nearest_resource,   // Distance to richest nearby cell
resource_gradient_x, resource_gradient_y,  // Direction to more resources
competition_intensity           // Local density / local resources
```

#### Lineage Tracking
```
parent_entity,                  // Entity ID of parent (if spawned)
birth_tick,                     // When organism was created
```

#### Success Metrics (computed over time)
```
lifetime_ticks,                 // How long organism lived
total_offspring_spawned,
offspring_survived_to_reproduce,
total_energy_consumed,
total_energy_gained,
net_energy_efficiency,          // gained / consumed
reproduction_success_rate,      // offspring_survived / offspring_spawned
survived_100_ticks, survived_500_ticks  // Binary survival markers
```

## Implementation Steps

### Phase 1: Enhanced Logging (Critical)

#### Step 1.1: Add Genome Genes to Logging
**File:** `src/organisms/systems.rs`
**Function:** `log_all_organisms()`

Add genome query parameter and log all 32 genes:
```rust
query: Query<
    (
        Entity,
        &Position,
        &Velocity,
        &Energy,
        &Age,
        &Size,
        &OrganismType,
        &Behavior,
        &CachedTraits,
        &Genome,  // ADD THIS
    ),
    With<Alive>,
>
```

In the logging loop, add:
```rust
// Log genome genes
for i in 0..32 {
    write!(writer, ",genome_gene_{}", i)?;
}
// Then in writeln! macro:
for i in 0..32 {
    write!(writer, ",{:.6}", genome.genes[i])?;
}
```

#### Step 1.2: Add Cell Data to Logging
**File:** `src/organisms/systems.rs`
**Function:** `log_all_organisms()`

Add `world_grid: Res<WorldGrid>` parameter and query cell at organism position:
```rust
if let Some(cell) = world_grid.get_cell(position.x(), position.y()) {
    // Log cell data
    write!(writer, 
        ",{:.6},{:.6},{},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6}",
        cell.temperature,
        cell.humidity,
        cell.elevation,
        cell.get_resource(ResourceType::Plant),
        // ... etc for all resources and pressures
    )?;
}
```

#### Step 1.3: Add Parent Entity Tracking
**File:** `src/organisms/components.rs` (if not exists)
**File:** `src/organisms/systems.rs`

Add optional `ParentEntity` component or track in reproduction system:
```rust
#[derive(Component)]
pub struct ParentEntity(pub Option<Entity>);
```

Log `parent_entity` in CSV (None/0 if spawned initially).

#### Step 1.4: Add Spatial Context
**File:** `src/organisms/systems.rs`
**Function:** `log_all_organisms()`

Use `spatial_hash` to compute:
- Local organism density (query radius around position)
- Predator/prey counts (filter by species/type)
- Distance to nearest resource (query world_grid)

### Phase 2: Cumulative Metrics Tracking

#### Step 2.1: Add Cumulative Energy Components
**File:** `src/organisms/components.rs`

```rust
#[derive(Component)]
pub struct EnergyStats {
    pub total_consumed: f32,
    pub total_gained: f32,
}
```

Update in `update_metabolism()` and `handle_eating()`.

#### Step 2.2: Track Offspring Success
**Option A:** Post-process logs to link parent → offspring → outcomes
**Option B:** Add component tracking:
```rust
#[derive(Component)]
pub struct ReproductiveStats {
    pub offspring_spawned: u32,
    pub offspring_survived: u32,
    pub offspring_reproduced: u32,
}
```

Update when offspring spawn/die/reproduce.

### Phase 3: Post-Processing Pipeline

#### Step 3.1: Create Labeled Dataset
**Language:** Python
**File:** `scripts/create_labeled_dataset.py`

```python
import pandas as pd
import numpy as np

# Load all organism tracking CSVs
dfs = []
for csv_file in glob.glob('data/logs/organisms_snapshot_*.csv'):
    df = pd.read_csv(csv_file)
    dfs.append(df)

all_data = pd.concat(dfs)

# Group by entity to compute lifetime metrics
organism_lifetimes = all_data.groupby('entity').agg({
    'tick': ['min', 'max', 'count'],
    'energy_current': 'mean',
    'genome_gene_0': 'first',  # Genome doesn't change
    # ... aggregate all features
})

# Compute success metrics
organism_lifetimes['lifetime_ticks'] = (
    organism_lifetimes[('tick', 'max')] - organism_lifetimes[('tick', 'min')]
)

# Create composite fitness score
def calculate_fitness(row):
    survival_bonus = min(row['lifetime_ticks'] / 1000.0, 1.0) * 0.4
    reproduction_bonus = min(row['offspring_survived'] / 5.0, 1.0) * 0.4
    efficiency_bonus = min(row['net_energy_efficiency'], 2.0) / 2.0 * 0.2
    return survival_bonus + reproduction_bonus + efficiency_bonus

organism_lifetimes['fitness_score'] = organism_lifetimes.apply(calculate_fitness, axis=1)
organism_lifetimes['thrives'] = (
    organism_lifetimes['fitness_score'] > organism_lifetimes['fitness_score'].quantile(0.8)
).astype(int)

# Save labeled dataset
organism_lifetimes.to_csv('data/outputs/labeled_genomes.csv')
```

#### Step 3.2: Feature Engineering
**File:** `scripts/feature_engineering.py`

```python
# Gene-environment interactions
for gene_idx in range(32):
    for env_feature in ['cell_temperature', 'cell_humidity', 'cell_resource_plant']:
        df[f'gene_{gene_idx}_x_{env_feature}'] = (
            df[f'genome_gene_{gene_idx}'] * df[env_feature]
        )

# Trait-environment matching
df['metabolism_food_match'] = (
    df['expressed_metabolism_rate'] * df['cell_resource_plant']
)

# Niche specialization score
df['niche_fit_score'] = (
    (1.0 - abs(df['expressed_thermal_tolerance'] - df['cell_temperature'])) +
    (df['expressed_foraging_drive'] * df['local_resource_density']) +
    (1.0 / (1.0 + df['competition_intensity']))
)
```

### Phase 4: Model Training

#### Step 4.1: Model Architecture
**Recommended:** XGBoost + SHAP for interpretability

```python
import xgboost as xgb
import shap

# Features: genome + environment
X = df[[f'genome_gene_{i}' for i in range(32)] + 
       ['cell_temperature', 'cell_humidity', 'local_organism_density', ...]]
y = df['thrives']  # Binary: top 20% fitness

# Train
model = xgb.XGBClassifier(
    max_depth=6,
    learning_rate=0.1,
    n_estimators=200,
    objective='binary:logistic'
)
model.fit(X, y)

# Explain
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X)

# Visualize
shap.summary_plot(shap_values, X)
shap.waterfall_plot(explainer.expected_value, shap_values[0], X.iloc[0])
```

#### Step 4.2: Explanation Generation
**File:** `scripts/generate_explanations.py`

```python
def explain_genome(genome_vector, environment_vector, model, explainer):
    """Generate natural language explanation"""
    X = combine(genome_vector, environment_vector)
    shap_values = explainer.shap_values(X)
    
    # Get top contributing factors
    top_contributors = get_top_shap_features(shap_values, n=5)
    
    explanation = f"""
    Genome Analysis Report:
    =======================
    
    Fitness Score: {model.predict_proba(X)[0][1]:.2f} ({'THRIVES' if model.predict(X)[0] else 'STRUGGLES'})
    
    Why this genome thrives:
    1. Gene {top_contributors[0].gene} ({top_contributors[0].value:.2f}) 
       interacts with {top_contributors[0].env_feature} 
       ({top_contributors[0].env_value:.2f})
       Impact: {top_contributors[0].shap_value:+.2f}
    
    2. Expressed trait {top_contributors[1].trait} 
       matches environment's {top_contributors[1].env_feature}
       Impact: {top_contributors[1].shap_value:+.2f}
    ...
    """
    return explanation
```

## Success Metrics Definition

### Composite Fitness Score
```python
fitness = (
    0.4 * survival_score +      # Lifetime / max_lifetime
    0.4 * reproduction_score +  # Offspring survived / expected
    0.2 * efficiency_score       # Energy gained / consumed
)
```

### Binary "Thrives" Label
```python
thrives = fitness_score > 80th_percentile  # Top 20% are "thriving"
```

## Model Output Format

The model should produce explanations like:

```
Genome Analysis Report:
=======================

Genome ID: 12345
Environment: Desert (temp=0.8, humidity=0.2, resources=low)

Fitness Score: 0.72 (THRIVES)

Why this genome thrives:
1. Gene 17 (thermal_tolerance) = 0.85 → High heat tolerance
   matches desert temperature (0.8)
   Impact: +0.15 fitness

2. Gene 20 (foraging_drive) = 0.92 → High foraging efficiency
   compensates for low resource density
   Impact: +0.12 fitness

3. Gene 3 (movement_cost) = 0.23 → Low movement cost
   allows efficient resource searching
   Impact: +0.08 fitness

Environment-Genome Mismatch:
- Gene 7 (sensory_range) = 0.45 → Low sensory range
  limits ability to find sparse resources
  Impact: -0.05 fitness

Recommendation: This genome is well-adapted to desert environments
with sparse resources. Consider increasing sensory_range for even
better performance.
```

## Validation Strategy

1. **Temporal Validation:** Train on early ticks, test on later ticks
2. **Environmental Validation:** Train on one terrain type, test on others
3. **Intervention Testing:** Change genes in simulation, verify predictions
4. **Causal Inference:** Use DoWhy/causal forests to validate relationships

## Potential Issues & Mitigations

### Issue 1: Overfitting to Simulation Quirks
**Mitigation:** Test on varied conditions, use regularization, validate against known biology

### Issue 2: Missing Confounders
**Mitigation:** Log spatial context, disaster events, other potential confounders

### Issue 3: Non-Stationarity
**Mitigation:** Track time/epoch, use online learning, retrain periodically

### Issue 4: Explanation Quality
**Mitigation:** Combine multiple explanation methods, validate with interventions

## Timeline Estimate

- **Phase 1 (Enhanced Logging):** 2-3 days
- **Phase 2 (Cumulative Metrics):** 1-2 days
- **Phase 3 (Post-Processing):** 2-3 days
- **Phase 4 (Model Training):** 3-5 days
- **Total:** ~2 weeks for full implementation

## Next Steps

1. ✅ Review and approve this plan
2. Implement Phase 1 (enhanced logging) - **CRITICAL**
3. Run simulation to generate enhanced data
4. Implement Phase 2 (cumulative metrics)
5. Create post-processing pipeline (Phase 3)
6. Train initial model (Phase 4)
7. Iterate and refine based on results

## Files to Modify

### Rust Code
- `src/organisms/systems.rs` - Add logging fields
- `src/organisms/components.rs` - Add tracking components (if needed)
- `src/organisms/mod.rs` - Register new components (if needed)

### Python Scripts (to create)
- `scripts/create_labeled_dataset.py` - Post-process logs
- `scripts/feature_engineering.py` - Create interaction features
- `scripts/train_model.py` - Train XGBoost model
- `scripts/generate_explanations.py` - Generate explanations
- `scripts/visualize_results.py` - SHAP plots, feature importance

## Dependencies

### Rust (already in Cargo.toml)
- No new dependencies needed

### Python (new)
```bash
pip install pandas numpy xgboost shap matplotlib seaborn scikit-learn
```

## Success Criteria

The project is successful if the model can:
1. ✅ Predict genome fitness with >70% accuracy
2. ✅ Identify top 5 genes that matter most
3. ✅ Explain genome-environment interactions
4. ✅ Generate human-readable explanations
5. ✅ Validate predictions through simulation interventions

---

**Last Updated:** [Current Date]
**Status:** Planning Phase - Ready for Implementation

