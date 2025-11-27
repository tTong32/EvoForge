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
    pub plant_consumption_rate_base: f32,
    pub meat_consumption_rate_base: f32,
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
            // Balanced regeneration rates (scaled for 600-tick lifetime: 22.34x faster)
            // These rates ensure resources regenerate faster than they're consumed
            plant_regeneration_rate: 2.23,      // 0.10 × 22.34
            water_regeneration_rate: 3.35,      // 0.15 × 22.34
            sunlight_regeneration_rate: 4.47,   // 0.20 × 22.34
            mineral_regeneration_rate: 1.34,    // 0.06 × 22.34
            detritus_regeneration_rate: 0.89,   // 0.04 × 22.34
            prey_regeneration_rate: 0.67,       // 0.03 × 22.34

            // Decay rates (resources naturally decay over time, scaled 22.34x faster)
            // Balanced to prevent resource accumulation while allowing regeneration
            plant_decay_rate: 0.179,           // 0.008 × 22.34
            water_decay_rate: 0.089,           // 0.004 × 22.34
            sunlight_decay_rate: 0.559,        // 0.025 × 22.34
            mineral_decay_rate: 0.011,         // 0.0005 × 22.34
            detritus_decay_rate: 0.268,        // 0.012 × 22.34
            prey_decay_rate: 0.559,           // 0.025 × 22.34

            // Consumption (balanced with 20.1x metabolism for 600-tick lifetime)
            // Increased to match high metabolism rate
            plant_consumption_rate_base: 28.0,  // Increased to balance high metabolism
            meat_consumption_rate_base: 20.0,   // Increased to balance high metabolism
            energy_conversion_efficiency: 0.65, // Increased for better energy balance
            decomposer_efficiency_multiplier: 0.6, // Increased from 0.5 (decomposers are more efficient)

            // Metabolism (scaled for 600-tick lifetime: 22.34x faster)
            base_metabolism_multiplier: 20.1,   // 0.9 × 22.34 (organisms die in ~600 ticks)
            movement_cost_multiplier: 0.85,      // Reduced from 1.0 (movement costs less)

            // Reproduction (scaled for 600-tick lifetime: 22.34x faster)
            reproduction_chance_multiplier: 0.03, // 3% chance per frame when conditions met
            min_reproduction_cooldown: 27.0,     // 600 / 22.34 ≈ 27 ticks
            max_reproduction_cooldown: 161.0,   // 3600 / 22.34 ≈ 161 ticks

            // Spawn
            initial_spawn_count: 99,

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
        tuning.min_reproduction_cooldown = 13.4;      // 300 / 22.34 ≈ 13 ticks
        tuning.max_reproduction_cooldown = 80.6;       // 1800 / 22.34 ≈ 81 ticks
        tuning.plant_regeneration_rate = 3.35;        // 0.15 × 22.34
        tuning.water_regeneration_rate = 4.47;        // 0.20 × 22.34
        tuning
    }

    /// Create preset for slow, stable ecosystem (lower reproduction, higher resources)
    pub fn stable() -> Self {
        let mut tuning = Self::default();
        tuning.reproduction_chance_multiplier = 0.02; // 2% chance (reduced from 5%)
        tuning.min_reproduction_cooldown = 35.8;     // 800 / 22.34 ≈ 36 ticks
        tuning.max_reproduction_cooldown = 215.0;     // 4800 / 22.34 ≈ 215 ticks
        tuning.plant_regeneration_rate = 4.02;        // 0.18 × 22.34
        tuning.water_regeneration_rate = 4.91;      // 0.22 × 22.34
        tuning.plant_consumption_rate_base = 24.5;   // Scaled proportionally
        tuning.meat_consumption_rate_base = 17.5;   // Scaled proportionally
        tuning
    }

    /// Create preset for competitive ecosystem (scarce resources, faster decay)
    pub fn competitive() -> Self {
        let mut tuning = Self::default();
        tuning.plant_regeneration_rate = 1.34;        // 0.06 × 22.34
        tuning.water_regeneration_rate = 2.23;      // 0.10 × 22.34
        tuning.plant_decay_rate = 0.335;            // 0.015 × 22.34
        tuning.plant_consumption_rate_base = 35.0;  // Scaled proportionally
        tuning.meat_consumption_rate_base = 28.0;   // Scaled proportionally
        tuning.base_metabolism_multiplier = 24.6;   // 1.1 × 22.34
        tuning
    }
}

