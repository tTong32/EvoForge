use bevy::prelude::*;

/// Ecosystem tuning parameters for Step 8 - Easy balance adjustment
#[derive(Resource)]
pub struct EcosystemTuning {
    // Resource regeneration rates
    pub plant_regeneration_rate: f32,
    pub water_regeneration_rate: f32,
    pub sunlight_regeneration_rate: f32,
    pub mineral_regeneration_rate: f32,
    pub detritus_regeneration_rate: f32,
    pub prey_regeneration_rate: f32,

    // Resource decay rates
    pub plant_decay_rate: f32,
    pub water_decay_rate: f32,
    pub sunlight_decay_rate: f32,
    pub mineral_decay_rate: f32,
    pub detritus_decay_rate: f32,
    pub prey_decay_rate: f32,

    // Consumption rates
    pub consumption_rate_base: f32,
    pub energy_conversion_efficiency: f32,
    pub decomposer_efficiency_multiplier: f32,

    // Metabolism tuning
    pub base_metabolism_multiplier: f32,
    pub movement_cost_multiplier: f32,

    // Reproduction tuning
    pub reproduction_chance_multiplier: f32,
    pub min_reproduction_cooldown: f32,
    pub max_reproduction_cooldown: f32,

    // Spawn parameters
    pub initial_spawn_count: usize,
    
    // Speciation
    pub speciation_threshold: f32,
}

impl Default for EcosystemTuning {
    fn default() -> Self {
        Self {
            // Balanced regeneration rates (tuned for Step 8 - balanced with consumption)
            // These rates ensure resources regenerate faster than they're consumed
            plant_regeneration_rate: 0.10,      // Increased from 0.08 for better balance
            water_regeneration_rate: 0.15,      // Increased from 0.12
            sunlight_regeneration_rate: 0.20,   // Increased from 0.15 (sunlight should be abundant)
            mineral_regeneration_rate: 0.06,    // Increased from 0.05
            detritus_regeneration_rate: 0.04,   // Increased from 0.03 (more detritus = more decomposers)
            prey_regeneration_rate: 0.03,       // Increased from 0.02 (prey should regenerate from death)

            // Decay rates (resources naturally decay over time)
            // Balanced to prevent resource accumulation while allowing regeneration
            plant_decay_rate: 0.008,           // Slightly reduced from 0.01
            water_decay_rate: 0.004,           // Slightly reduced from 0.005
            sunlight_decay_rate: 0.025,        // Increased from 0.02 (sunlight cycles quickly)
            mineral_decay_rate: 0.0005,        // Reduced from 0.001 (minerals persist longer)
            detritus_decay_rate: 0.012,         // Reduced from 0.015 (detritus persists for decomposers)
            prey_decay_rate: 0.025,           // Increased from 0.02 (prey moves/dies quickly)

            // Consumption (balanced with regeneration rates)
            // Lower consumption ensures resources can regenerate
            consumption_rate_base: 4.0,         // Reduced from 5.0 to balance with regeneration
            energy_conversion_efficiency: 0.35, // Increased from 0.3 (organisms get more energy)
            decomposer_efficiency_multiplier: 0.6, // Increased from 0.5 (decomposers are more efficient)

            // Metabolism (balanced to prevent energy drain)
            base_metabolism_multiplier: 0.9,    // Reduced from 1.0 (organisms use less energy)
            movement_cost_multiplier: 0.85,      // Reduced from 1.0 (movement costs less)

            // Reproduction (tuned for stability - prevents instant spawning)
            reproduction_chance_multiplier: 0.03, // 3% chance per frame when conditions met (reduced from 10%)
            min_reproduction_cooldown: 600.0,    // Minimum 600 ticks (~10 seconds at 60 FPS)
            max_reproduction_cooldown: 3600.0,  // Maximum 3600 ticks (~60 seconds at 60 FPS)

            // Spawn
            initial_spawn_count: 100,

            // Speciation
            speciation_threshold: 0.15,
        }
    }
}

impl EcosystemTuning {
    /// Create balanced preset for stable ecosystem
    pub fn balanced() -> Self {
        Self::default()
    }

    /// Create preset for fast evolution (higher mutation, faster reproduction)
    pub fn fast_evolution() -> Self {
        let mut tuning = Self::default();
        tuning.reproduction_chance_multiplier = 0.08; // 8% chance (reduced from 15% for balance)
        tuning.min_reproduction_cooldown = 300.0;     // Faster reproduction
        tuning.max_reproduction_cooldown = 1800.0;
        tuning.plant_regeneration_rate = 0.15;        // More resources
        tuning.water_regeneration_rate = 0.20;
        tuning
    }

    /// Create preset for slow, stable ecosystem (lower reproduction, higher resources)
    pub fn stable() -> Self {
        let mut tuning = Self::default();
        tuning.reproduction_chance_multiplier = 0.02; // 2% chance (reduced from 5%)
        tuning.min_reproduction_cooldown = 800.0;     // Slower reproduction
        tuning.max_reproduction_cooldown = 4800.0;
        tuning.plant_regeneration_rate = 0.18;       // More resources for stability
        tuning.water_regeneration_rate = 0.22;
        tuning.consumption_rate_base = 3.5;           // Lower consumption
        tuning
    }

    /// Create preset for competitive ecosystem (scarce resources, faster decay)
    pub fn competitive() -> Self {
        let mut tuning = Self::default();
        tuning.plant_regeneration_rate = 0.06;        // Scarce resources
        tuning.water_regeneration_rate = 0.10;
        tuning.plant_decay_rate = 0.015;              // Faster decay
        tuning.consumption_rate_base = 5.5;           // Higher consumption
        tuning.base_metabolism_multiplier = 1.1;     // Higher metabolism
        tuning
    }
}

