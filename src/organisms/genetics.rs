use bevy::prelude::*;
use smallvec::SmallVec;

/// Size of the genome (number of genes)
pub const GENOME_SIZE: usize = 32;

/// Genome representation - array of floating-point genes (0.0 to 1.0)
/// Each gene encodes a trait that affects organism behavior/characteristics
#[derive(Component, Debug, Clone)]
pub struct Genome {
    /// Genes stored as SmallVec for small genomes (avoids heap allocation)
    pub genes: SmallVec<[f32; GENOME_SIZE]>,
}

impl Genome {
    /// Create a new random genome
    /// Optimized: Uses fastrand for better performance
    pub fn random() -> Self {
        let mut rng = fastrand::Rng::new();
        let mut genes = SmallVec::new();
        for _ in 0..GENOME_SIZE {
            genes.push(rng.f32());
        }
        Self { genes }
    }

    /// Create a genome with specific genes
    pub fn new(genes: Vec<f32>) -> Self {
        let mut genome = SmallVec::new();
        for gene in genes.iter().take(GENOME_SIZE) {
            genome.push(gene.clamp(0.0, 1.0));
        }
        // Fill remaining slots if needed
        while genome.len() < GENOME_SIZE {
            genome.push(0.5);
        }
        Self { genes: genome }
    }

    /// Get a gene value (clamped to valid range)
    pub fn get_gene(&self, index: usize) -> f32 {
        if index < self.genes.len() {
            self.genes[index].clamp(0.0, 1.0)
        } else {
            0.5 // Default value
        }
    }

    /// Set a gene value (clamped to valid range)
    pub fn set_gene(&mut self, index: usize, value: f32) {
        if index < self.genes.len() {
            self.genes[index] = value.clamp(0.0, 1.0);
        }
    }

    /// Clone genome with optional mutations
    /// Optimized: Uses faster uniform mutation instead of expensive Box-Muller transform
    pub fn clone_with_mutation(&self, mutation_rate: f32) -> Self {
        let mut new_genes = SmallVec::new();
        let mut rng = fastrand::Rng::new();

        for &gene in self.genes.iter() {
            let mut new_gene = gene;

            // Apply mutation with probability
            if rng.f32() < mutation_rate {
                // Uniform mutation: add random value in range [-0.1, 0.1]
                // This is faster than Box-Muller and produces similar results for small mutations
                let mutation = (rng.f32() - 0.5) * 0.2;
                new_gene = (new_gene + mutation).clamp(0.0, 1.0);
            }

            new_genes.push(new_gene);
        }

        Self { genes: new_genes }
    }

    /// Crossover two genomes (sexual reproduction)
    /// Optimized: Uses faster uniform mutation instead of expensive Box-Muller transform
    pub fn crossover(parent_a: &Genome, parent_b: &Genome, mutation_rate: f32) -> Self {
        let mut rng = fastrand::Rng::new();
        let mut new_genes = SmallVec::new();

        // Uniform crossover: for each gene, randomly choose from parent A or B
        for i in 0..GENOME_SIZE {
            let gene_a = parent_a.get_gene(i);
            let gene_b = parent_b.get_gene(i);

            // 50/50 chance to choose from each parent
            let mut new_gene = if rng.bool() { gene_a } else { gene_b };

            // Apply mutation with probability
            if rng.f32() < mutation_rate {
                // Uniform mutation: add random value in range [-0.1, 0.1]
                let mutation = (rng.f32() - 0.5) * 0.2;
                new_gene = (new_gene + mutation).clamp(0.0, 1.0);
            }

            new_genes.push(new_gene);
        }

        Self { genes: new_genes }
    }

    /// Calculate genetic distance between two genomes (for speciation)
    pub fn distance(&self, other: &Genome) -> f32 {
        let mut sum = 0.0;
        let min_len = self.genes.len().min(other.genes.len());

        for i in 0..min_len {
            let diff = (self.genes[i] - other.genes[i]).abs();
            sum += diff * diff; // Squared difference
        }

        (sum / min_len as f32).sqrt() // Root mean squared difference
    }
}

/// Trait indices in the genome
/// Each trait is encoded by one or more genes
pub mod traits {
    use super::*;
    use crate::organisms::components::{IncubationType, SpawnType};

    /// Helper: convert a [0,1] gene value into [-1,1]
    fn gene_to_signed(value: f32) -> f32 {
        (value * 2.0) - 1.0
    }

    /// Helper: sigmoid activation for smoother response curves
    fn sigmoid(x: f32) -> f32 {
        1.0 / (1.0 + (-x).exp())
    }

    /// Maps a weighted sum of genes into the desired output range.
    fn express_with_weights(
        genome: &Genome,
        weights: &[(usize, f32)],
        bias: f32,
        min: f32,
        max: f32,
    ) -> f32 {
        let mut sum = bias;
        for (index, weight) in weights {
            let gene_value = genome.get_gene(*index);
            sum += gene_to_signed(gene_value) * *weight;
        }

        let normalized = sigmoid(sum.clamp(-6.0, 6.0));
        min + normalized * (max - min)
    }

    /// Base trait indices (primary drivers)
    pub const SPEED: usize = 0;
    pub const SIZE: usize = 1;
    pub const METABOLISM_RATE: usize = 2;
    pub const MOVEMENT_COST: usize = 3;
    pub const MAX_ENERGY: usize = 4;
    pub const REPRODUCTION_COOLDOWN: usize = 5;
    pub const REPRODUCTION_THRESHOLD: usize = 6;
    pub const SENSORY_RANGE: usize = 7;
    pub const AGGRESSION: usize = 8;
    pub const BOLDNESS: usize = 9;

    /// Modifier genes enabling richer expression
    pub const SPEED_FAST_TWITCH: usize = 10;
    pub const SPEED_ENDURANCE: usize = 11;
    pub const STRUCTURAL_DENSITY: usize = 12;
    pub const METABOLIC_FLEXIBILITY: usize = 13;
    pub const REPRODUCTIVE_INVESTMENT: usize = 14;
    pub const SENSORY_FOCUS: usize = 15;
    pub const SOCIAL_SENSITIVITY: usize = 16;
    pub const THERMAL_TOLERANCE: usize = 17;
    pub const MUTATION_CONTROL: usize = 18;
    pub const DEVELOPMENTAL_PLASTICITY: usize = 19;
    pub const FORAGING_BIAS: usize = 20;
    pub const RISK_TOLERANCE: usize = 21;
    pub const EXPLORATION_DRIVE: usize = 22;
    pub const CLUTCH_SIZE: usize = 23;
    pub const OFFSPRING_ENERGY_SHARE: usize = 24;
    pub const HUNGER_MEMORY: usize = 25;
    pub const THREAT_DECAY: usize = 26;
    pub const RESOURCE_SELECTIVITY: usize = 27;
    pub const MIGRATION_DRIVE: usize = 28;
    // Phase 2+: additional predation/learning/offspring/defense genes (reuse indices where possible)
    pub const PLANT_CONSUMPTION_RATE: usize = 29;
    pub const MEAT_CONSUMPTION_RATE: usize = 30;
    pub const ATTACK_STRENGTH: usize = 31;
    pub const ARMOR_GENE: usize = 32;
    pub const POISON_GENE: usize = 33;
    pub const FLEE_SPEED_GENE: usize = 34;
    pub const ENDURANCE_GENE: usize = 35;
    pub const HUNTING_STRATEGY_GENE: usize = 36;

    /// Express speed trait (0.5 to 20.0 units/sec) using multiple genes.
    pub fn express_speed(genome: &Genome) -> f32 {
        express_with_weights(
            genome,
            &[
                (SPEED, 1.4),
                (SPEED_FAST_TWITCH, 0.9),
                (SPEED_ENDURANCE, 0.6),
                (METABOLISM_RATE, 0.3),
                (STRUCTURAL_DENSITY, -0.6),
            ],
            0.1,
            0.5,
            20.0,
        )
    }

    /// Express size trait (0.3 to 3.0 units) with structural modifiers.
    pub fn express_size(genome: &Genome) -> f32 {
        express_with_weights(
            genome,
            &[
                (SIZE, 1.2),
                (STRUCTURAL_DENSITY, 0.8),
                (DEVELOPMENTAL_PLASTICITY, 0.4),
                (METABOLISM_RATE, -0.4),
            ],
            0.0,
            0.3,
            3.0,
        )
    }

    /// Express metabolism rate trait (0.003 to 0.03 per second).
    pub fn express_metabolism_rate(genome: &Genome) -> f32 {
        express_with_weights(
            genome,
            &[
                (METABOLISM_RATE, 1.1),
                (METABOLIC_FLEXIBILITY, 0.7),
                (SPEED_ENDURANCE, 0.4),
                (STRUCTURAL_DENSITY, -0.3),
            ],
            0.0,
            0.003,
            0.03,
        )
    }

    /// Express movement cost trait (0.008 to 0.12).
    pub fn express_movement_cost(genome: &Genome) -> f32 {
        express_with_weights(
            genome,
            &[
                (MOVEMENT_COST, 1.0),
                (SIZE, 0.6),
                (STRUCTURAL_DENSITY, 0.5),
                (METABOLIC_FLEXIBILITY, -0.5),
            ],
            0.2,
            0.008,
            0.12,
        )
    }

    /// Express max energy trait (40.0 to 220.0).
    pub fn express_max_energy(genome: &Genome) -> f32 {
        express_with_weights(
            genome,
            &[
                (MAX_ENERGY, 1.2),
                (SIZE, 0.7),
                (METABOLISM_RATE, -0.5),
                (THERMAL_TOLERANCE, 0.3),
            ],
            0.0,
            40.0,
            220.0,
        )
    }

    /// Express reproduction cooldown trait (600 to 3600 ticks - tuned for stability).
    pub fn express_reproduction_cooldown(genome: &Genome) -> f32 {
        express_with_weights(
            genome,
            &[
                (REPRODUCTION_COOLDOWN, 1.0),
                (REPRODUCTIVE_INVESTMENT, 0.9),
                (METABOLISM_RATE, -0.4),
                (DEVELOPMENTAL_PLASTICITY, 0.5),
            ],
            0.0,
            600.0, // Minimum 600 ticks (~10 seconds at 60 FPS)
            3600.0, // Maximum 3600 ticks (~60 seconds at 60 FPS)
        )
    }

    /// Express reproduction threshold trait (0.45 to 0.95 energy ratio).
    pub fn express_reproduction_threshold(genome: &Genome) -> f32 {
        express_with_weights(
            genome,
            &[
                (REPRODUCTION_THRESHOLD, 1.0),
                (REPRODUCTIVE_INVESTMENT, 0.8),
                (MAX_ENERGY, 0.3),
                (METABOLIC_FLEXIBILITY, -0.5),
            ],
            0.2,
            0.45,
            0.95,
        )
    }

    /// Express sensory range trait (6.0 to 65.0 units).
    pub fn express_sensory_range(genome: &Genome) -> f32 {
        express_with_weights(
            genome,
            &[
                (SENSORY_RANGE, 1.0),
                (SENSORY_FOCUS, 0.8),
                (SOCIAL_SENSITIVITY, 0.6),
                (THERMAL_TOLERANCE, -0.3),
            ],
            0.1,
            6.0,
            65.0,
        )
    }

    /// Express aggression trait (0.0 to 1.0).
    pub fn express_aggression(genome: &Genome) -> f32 {
        express_with_weights(
            genome,
            &[
                (AGGRESSION, 1.0),
                (SPEED_FAST_TWITCH, 0.4),
                (SENSORY_FOCUS, 0.2),
                (SOCIAL_SENSITIVITY, -0.6),
                (PLANT_CONSUMPTION_RATE, -0.1),
                (MEAT_CONSUMPTION_RATE, 0.25),
            ],
            0.0,
            0.0,
            1.0,
        )
    }

    /// Express boldness trait (0.0 to 1.0).
    pub fn express_boldness(genome: &Genome) -> f32 {
        express_with_weights(
            genome,
            &[
                (BOLDNESS, 1.0),
                (REPRODUCTIVE_INVESTMENT, 0.5),
                (THERMAL_TOLERANCE, 0.3),
                (SOCIAL_SENSITIVITY, -0.4),
            ],
            0.0,
            0.0,
            1.0,
        )
    }

    /// Express mutation rate trait (0.002 to 0.06 probability per gene).
    pub fn express_mutation_rate(genome: &Genome) -> f32 {
        express_with_weights(
            genome,
            &[
                (MUTATION_CONTROL, 1.2),
                (DEVELOPMENTAL_PLASTICITY, 0.6),
                (METABOLIC_FLEXIBILITY, 0.3),
            ],
            -0.2,
            0.002,
            0.06,
        )
    }

    pub fn express_foraging_drive(genome: &Genome) -> f32 {
        express_with_weights(
            genome,
            &[
                (FORAGING_BIAS, 1.1),
                (METABOLISM_RATE, 0.4),
                (RESOURCE_SELECTIVITY, -0.3),
                (PLANT_CONSUMPTION_RATE, 0.3),
                (MEAT_CONSUMPTION_RATE, -0.1),
            ],
            0.0,
            0.0,
            1.0,
        )
    }

    pub fn express_risk_tolerance(genome: &Genome) -> f32 {
        express_with_weights(
            genome,
            &[(RISK_TOLERANCE, 1.0), (BOLDNESS, 0.7), (AGGRESSION, 0.3)],
            0.0,
            0.05,
            0.95,
        )
    }

    pub fn express_exploration_drive(genome: &Genome) -> f32 {
        express_with_weights(
            genome,
            &[
                (EXPLORATION_DRIVE, 1.0),
                (SENSORY_RANGE, 0.4),
                (MIGRATION_DRIVE, 0.5),
            ],
            -0.2,
            0.0,
            1.0,
        )
    }

    pub fn express_clutch_size(genome: &Genome) -> f32 {
        express_with_weights(
            genome,
            &[
                (CLUTCH_SIZE, 1.0),
                (REPRODUCTIVE_INVESTMENT, -0.4),
                (SIZE, -0.2),
            ],
            0.3,
            1.0,
            6.0,
        )
    }

    pub fn express_offspring_energy_share(genome: &Genome) -> f32 {
        express_with_weights(
            genome,
            &[
                (OFFSPRING_ENERGY_SHARE, 1.0),
                (REPRODUCTIVE_INVESTMENT, 0.7),
                (METABOLISM_RATE, -0.4),
            ],
            0.0,
            0.05,
            0.45,
        )
    }

    pub fn express_hunger_memory_rate(genome: &Genome) -> f32 {
        express_with_weights(
            genome,
            &[
                (HUNGER_MEMORY, 1.0),
                (FORAGING_BIAS, 0.4),
                (METABOLIC_FLEXIBILITY, 0.3),
            ],
            0.0,
            0.5,
            3.0,
        )
    }

    pub fn express_threat_decay_rate(genome: &Genome) -> f32 {
        express_with_weights(
            genome,
            &[
                (THREAT_DECAY, 1.0),
                (RISK_TOLERANCE, -0.6),
                (SOCIAL_SENSITIVITY, -0.3),
            ],
            0.2,
            0.2,
            2.5,
        )
    }

    pub fn express_resource_selectivity(genome: &Genome) -> f32 {
        express_with_weights(
            genome,
            &[
                (RESOURCE_SELECTIVITY, 1.0),
                (FORAGING_BIAS, -0.5),
                (SENSORY_FOCUS, 0.4),
            ],
            0.0,
            0.0,
            1.0,
        )
    }

    /// Express consumption rate trait (fraction of plant/prey that can be eaten per second).
    pub fn express_plant_consumption_rate(genome: &Genome) -> f32 {
        express_with_weights(
            genome,
            &[
                (PLANT_CONSUMPTION_RATE, 1.0),
                (SIZE, 0.5),
                (METABOLISM_RATE, -0.3),
                (AGGRESSION, -0.4),
            ],
            0.0,
            0.02,
            0.4,
        )
    }

    pub fn express_meat_consumption_rate(genome: &Genome) -> f32 {
        express_with_weights(
            genome,
            &[
                (MEAT_CONSUMPTION_RATE, 1.0),
                (ATTACK_STRENGTH, 0.4),
                (METABOLISM_RATE, -0.3),
                (AGGRESSION, 0.5),
            ],
            0.0,
            0.02,
            0.4,
        )
    }

    /// Express attack strength trait (damage per second applied to prey energy).
    pub fn express_attack_strength(genome: &Genome) -> f32 {
        express_with_weights(
            genome,
            &[
                (ATTACK_STRENGTH, 1.0),
                (SIZE, 0.6),
                (AGGRESSION, 0.5),
            ],
            0.0,
            1.0,
            12.0,
        )
    }

    /// Defensive armor (0–1, reduces physical damage).
    pub fn express_armor(genome: &Genome) -> f32 {
        express_with_weights(
            genome,
            &[
                (ARMOR_GENE, 1.0),
                (STRUCTURAL_DENSITY, 0.8),
            ],
            0.0,
            0.0,
            1.0,
        )
    }

    /// Poison strength (0–1, discourages predators; can reduce their success).
    pub fn express_poison_strength(genome: &Genome) -> f32 {
        express_with_weights(
            genome,
            &[
                (POISON_GENE, 1.0),
                (THERMAL_TOLERANCE, 0.3),
            ],
            0.0,
            0.0,
            1.0,
        )
    }

    /// Flee speed bonus (0.5–2.0 multiplier to base speed when fleeing).
    pub fn express_flee_speed(genome: &Genome) -> f32 {
        express_with_weights(
            genome,
            &[
                (SPEED, 0.8),
                (FLEE_SPEED_GENE, 1.0),
                (SPEED_ENDURANCE, 0.5),
            ],
            0.0,
            0.5,
            2.0,
        )
    }

    /// Endurance (0–1, makes prey harder to finish off).
    pub fn express_endurance(genome: &Genome) -> f32 {
        express_with_weights(
            genome,
            &[
                (ENDURANCE_GENE, 1.0),
                (SPEED_ENDURANCE, 0.7),
                (MAX_ENERGY, 0.3),
            ],
            0.0,
            0.0,
            1.0,
        )
    }

    /// Express coordination trait (0.0 to 1.0) used for pack hunting.
    pub fn express_coordination(genome: &Genome) -> f32 {
        express_with_weights(
            genome,
            &[
                (SOCIAL_SENSITIVITY, 1.0),
                (SENSORY_FOCUS, 0.5),
                (MIGRATION_DRIVE, 0.3),
            ],
            0.0,
            0.0,
            1.0,
        )
    }

    /// Whether this species tends to form packs.
    pub fn express_forms_packs(genome: &Genome) -> bool {
        let v = express_with_weights(
            genome,
            &[
                (SOCIAL_SENSITIVITY, 1.0),
                (EXPLORATION_DRIVE, -0.5),
                (RISK_TOLERANCE, 0.3),
            ],
            0.0,
            0.0,
            1.0,
        );
        v > 0.5
    }

    /// Preferred pack lifetime in seconds.
    pub fn express_pack_lifetime(genome: &Genome) -> f32 {
        express_with_weights(
            genome,
            &[
                (SOCIAL_SENSITIVITY, 0.8),
                (MIGRATION_DRIVE, 0.4),
            ],
            0.0,
            50.0,
            600.0,
        )
    }

    /// Preferred pack size (1–8).
    pub fn express_pack_size_preference(genome: &Genome) -> f32 {
        express_with_weights(
            genome,
            &[
                (SOCIAL_SENSITIVITY, 1.0),
                (REPRODUCTIVE_INVESTMENT, -0.3),
            ],
            0.0,
            1.0,
            8.0,
        )
    }

    /// Learning rate for prey knowledge (0.05–0.4).
    pub fn express_learning_rate(genome: &Genome) -> f32 {
        express_with_weights(
            genome,
            &[
                (MUTATION_CONTROL, 0.8),
                (DEVELOPMENTAL_PLASTICITY, 0.7),
            ],
            -0.1,
            0.05,
            0.4,
        )
    }

    /// Teaching ability (0.1–1.0) for knowledge transfer to offspring.
    pub fn express_teaching_ability(genome: &Genome) -> f32 {
        express_with_weights(
            genome,
            &[
                (SOCIAL_SENSITIVITY, 1.0),
                (SENSORY_FOCUS, 0.4),
            ],
            0.0,
            0.1,
            1.0,
        )
    }

    /// Hunting strategy: 0–0.33 -> Ambush, 0.33–0.66 -> Pursuit, >0.66 -> Pack.
    pub fn express_hunting_strategy(genome: &Genome) -> crate::organisms::components::HuntingStrategy {
        let v = express_with_weights(
            genome,
            &[
                (HUNTING_STRATEGY_GENE, 1.0),
                (EXPLORATION_DRIVE, -0.3),
                (SENSORY_FOCUS, 0.3),
            ],
            0.0,
            0.0,
            1.0,
        );
        if v < 0.33 {
            crate::organisms::components::HuntingStrategy::Ambush
        } else if v < 0.66 {
            crate::organisms::components::HuntingStrategy::Pursuit
        } else {
            crate::organisms::components::HuntingStrategy::Pack
        }
    }

    // === Phase 3: Offspring & care traits expressed from genome ===

    pub fn express_spawn_type(genome: &Genome) -> SpawnType {
        let v = express_with_weights(
            genome,
            &[(REPRODUCTIVE_INVESTMENT, 1.0), (SIZE, 0.4)],
            0.0,
            0.0,
            1.0,
        );
        if v > 0.5 {
            SpawnType::Egg
        } else {
            SpawnType::Baby
        }
    }

    pub fn express_incubation_type(genome: &Genome) -> IncubationType {
        let v = express_with_weights(
            genome,
            &[(RISK_TOLERANCE, 1.0), (EXPLORATION_DRIVE, 0.5)],
            0.0,
            0.0,
            1.0,
        );
        if v > 0.5 {
            IncubationType::Guarded
        } else {
            IncubationType::Abandoned
        }
    }

    pub fn express_incubation_duration(genome: &Genome) -> f32 {
        express_with_weights(
            genome,
            &[(REPRODUCTIVE_INVESTMENT, 1.0), (DEVELOPMENTAL_PLASTICITY, 0.6)],
            0.0,
            1.34,  // 30.0 / 22.34 (scaled for 600-tick lifetime)
            13.4,  // 300.0 / 22.34
        )
    }

    pub fn express_parental_care_age(genome: &Genome) -> f32 {
        express_with_weights(
            genome,
            &[(DEVELOPMENTAL_PLASTICITY, 1.0), (SIZE, 0.5)],
            0.0,
            2.24,  // 50.0 / 22.34 (scaled for 600-tick lifetime)
            17.9,  // 400.0 / 22.34
        )
    }

    pub fn express_meal_share_percentage(genome: &Genome) -> f32 {
        express_with_weights(
            genome,
            &[(OFFSPRING_ENERGY_SHARE, 1.0), (REPRODUCTIVE_INVESTMENT, 0.6)],
            0.0,
            0.05,
            0.6,
        )
    }

    pub fn express_child_growth_rate(genome: &Genome) -> f32 {
        express_with_weights(
            genome,
            &[(DEVELOPMENTAL_PLASTICITY, 1.0), (METABOLISM_RATE, 0.5)],
            0.0,
            0.223, // 0.01 × 22.34 (scaled for 600-tick lifetime - faster growth)
            2.23,  // 0.1 × 22.34
        )
    }

    pub fn express_child_max_growth_rate(genome: &Genome) -> f32 {
        express_with_weights(
            genome,
            &[(DEVELOPMENTAL_PLASTICITY, 1.0), (METABOLIC_FLEXIBILITY, 0.5)],
            0.0,
            0.447, // 0.02 × 22.34 (scaled for 600-tick lifetime - faster growth)
            4.47,  // 0.2 × 22.34
        )
    }

    pub fn express_father_provides_care(genome: &Genome) -> bool {
        let v = express_with_weights(
            genome,
            &[(SOCIAL_SENSITIVITY, 1.0), (RISK_TOLERANCE, -0.4)],
            0.0,
            0.0,
            1.0,
        );
        v > 0.5
    }

    pub fn express_can_produce_milk(genome: &Genome) -> bool {
        let v = express_with_weights(
            genome,
            &[(REPRODUCTIVE_INVESTMENT, 1.0), (METABOLISM_RATE, 0.4)],
            0.0,
            0.0,
            1.0,
        );
        v > 0.5
    }

    pub fn express_milk_amount(genome: &Genome) -> f32 {
        express_with_weights(
            genome,
            &[(REPRODUCTIVE_INVESTMENT, 1.0), (MAX_ENERGY, 0.4)],
            0.0,
            1.0,
            8.0,
        )
    }

    pub fn express_knowledge_transfer_rate(genome: &Genome) -> f32 {
        express_with_weights(
            genome,
            &[(SOCIAL_SENSITIVITY, 1.0), (SENSORY_FOCUS, 0.4)],
            0.0,
            0.1,
            1.0,
        )
    }
}

/// Default mutation rate (probability of mutation per gene)
pub const DEFAULT_MUTATION_RATE: f32 = 0.01; // Backwards-compatible baseline
