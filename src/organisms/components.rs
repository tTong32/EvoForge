use bevy::prelude::*;
use glam::Vec2;
use std::collections::HashMap;

/// Position in world coordinates
#[derive(Component, Debug, Clone, Copy)]
pub struct Position(pub Vec2);

impl Position {
    pub fn new(x: f32, y: f32) -> Self {
        Self(Vec2::new(x, y))
    }

    pub fn x(&self) -> f32 {
        self.0.x
    }

    pub fn y(&self) -> f32 {
        self.0.y
    }

    pub fn as_vec2(&self) -> Vec2 {
        self.0
    }
}

/// Velocity in world units per second
#[derive(Component, Debug, Clone, Copy)]
pub struct Velocity(pub Vec2);

impl Velocity {
    pub fn new(x: f32, y: f32) -> Self {
        Self(Vec2::new(x, y))
    }

    pub fn zero() -> Self {
        Self(Vec2::ZERO)
    }
}

/// Current energy level (0.0 = dead, 1.0 = full energy)
#[derive(Component, Debug, Clone, Copy)]
pub struct Energy {
    pub current: f32,
    pub max: f32,
}

impl Energy {
    pub fn new(max: f32) -> Self {
        Self { current: max, max }
    }

    pub fn with_energy(max: f32, current: f32) -> Self {
        Self {
            current: current.min(max),
            max,
        }
    }

    pub fn ratio(&self) -> f32 {
        if self.max > 0.0 {
            self.current / self.max
        } else {
            0.0
        }
    }

    pub fn is_dead(&self) -> bool {
        self.current <= 0.0
    }
}

/// Age in simulation ticks
#[derive(Component, Debug, Clone, Copy)]
pub struct Age(pub u32);

impl Age {
    pub fn new() -> Self {
        Self(0)
    }

    pub fn increment(&mut self) {
        self.0 += 1;
    }

    pub fn ticks(&self) -> u32 {
        self.0
    }
}

/// Size of the organism (affects collision, metabolism, etc.)
#[derive(Component, Debug, Clone, Copy)]
pub struct Size(pub f32);

impl Size {
    pub fn new(size: f32) -> Self {
        Self(size)
    }

    pub fn value(&self) -> f32 {
        self.0
    }
}

/// Metabolism parameters (affects energy consumption)
#[derive(Component, Debug, Clone, Copy)]
pub struct Metabolism {
    /// Base metabolic rate (energy consumed per second)
    pub base_rate: f32,
    /// Movement cost multiplier (multiplies velocity magnitude)
    pub movement_cost: f32,
}

impl Metabolism {
    pub fn new(base_rate: f32, movement_cost: f32) -> Self {
        Self {
            base_rate,
            movement_cost,
        }
    }

    /// Default metabolism for a basic organism
    pub fn default() -> Self {
        Self {
            base_rate: 0.01,     // 1% max energy per second
            movement_cost: 0.05, // Additional cost for movement
        }
    }
}

/// Species ID for tracking and speciation (Stage 4+)
#[derive(Component, Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SpeciesId(pub u32);

impl SpeciesId {
    pub fn new(id: u32) -> Self {
        Self(id)
    }

    pub fn value(&self) -> u32 {
        self.0
    }
}

/// Marker component for organisms that are alive
#[derive(Component, Debug)]
pub struct Alive;

/// Marker indicating this organism was killed by predation.
#[derive(Component, Debug)]
pub struct KilledByPredation;

/// Organism type (for future behavior differentiation)
#[derive(Component, Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum OrganismType {
    Producer,   // Plants, algae - generate energy from resources
    Consumer,   // Animals - consume other organisms/resources
    Decomposer, // Fungi, bacteria - consume detritus
}

/// High-level hunting strategy for consumers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum HuntingStrategy {
    Ambush,
    Pursuit,
    Pack,
}

/// How an organism spawns its offspring.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SpawnType {
    Egg,
    Baby,
}

/// How an egg is incubated.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IncubationType {
    Guarded,
    Abandoned,
}

/// Reproduction cooldown (ticks remaining until organism can reproduce again)
#[derive(Component, Debug, Clone, Copy)]
pub struct ReproductionCooldown(pub u32);

/// Cached trait values derived from genome (updated when genome changes)
/// This avoids recalculating traits every frame
#[derive(Component, Debug, Clone)]
pub struct CachedTraits {
    pub speed: f32,
    pub size: f32,
    pub metabolism_rate: f32,
    pub movement_cost: f32,
    pub max_energy: f32,
    pub reproduction_cooldown: f32,
    pub reproduction_threshold: f32,
    pub sensory_range: f32,
    pub aggression: f32,
    pub boldness: f32,
    pub mutation_rate: f32,
    pub foraging_drive: f32,
    pub risk_tolerance: f32,
    pub exploration_drive: f32,
    pub clutch_size: f32,
    pub offspring_energy_share: f32,
    pub hunger_memory_rate: f32,
    pub threat_decay_rate: f32,
    pub resource_selectivity: f32,
    // Predation & learning related traits (Phase 2)
    pub consumption_rate: f32,
    pub attack_strength: f32,
    pub coordination: f32,
    pub forms_packs: bool,
    pub pack_lifetime: f32,
    pub pack_size_preference: f32,
    pub learning_rate: f32,
    pub teaching_ability: f32,
    // Offspring & care traits (Phase 3)
    pub spawn_type: SpawnType,
    pub incubation_type: IncubationType,
    pub incubation_duration: f32,
    pub parental_care_age: f32,
    pub meal_share_percentage: f32,
    pub growth_rate: f32,
    pub max_growth_rate: f32,
    pub father_provides_care: bool,
    pub can_produce_milk: bool,
    pub milk_amount: f32,
    pub knowledge_transfer_rate: f32,
    // Defense & predation modifiers (Phase 2 multi-factor)
    pub armor: f32,
    pub poison_strength: f32,
    pub flee_speed: f32,
    pub endurance: f32,
    pub hunting_strategy: HuntingStrategy,
}

/// Predator is currently feeding on a carcass (multi-tick consumption).
#[derive(Component, Debug, Clone, Copy)]
pub struct PredatorFeeding {
    /// Remaining energy that can be extracted from the carcass.
    pub remaining_energy: f32,
}

/// Per-organism learning about prey species (huntability knowledge).
#[derive(Component, Debug, Clone)]
pub struct IndividualLearning {
    /// prey_species_id -> knowledge score (0.0–1.0)
    pub prey_knowledge: HashMap<u32, f32>,
    /// How quickly this organism updates its knowledge.
    pub learning_rate: f32,
}

impl IndividualLearning {
    pub fn new(learning_rate: f32) -> Self {
        Self {
            prey_knowledge: HashMap::new(),
            learning_rate,
        }
    }

    /// Get current knowledge score for a prey species (default ~0.4).
    pub fn get_score(&self, prey_species_id: u32) -> f32 {
        *self.prey_knowledge.get(&prey_species_id).unwrap_or(&0.4)
    }

    /// Update knowledge after a successful hunt (moves score toward 1.0).
    pub fn update_on_success(&mut self, prey_species_id: u32) {
        let current = self.get_score(prey_species_id);
        let lr = self.learning_rate.clamp(0.05, 0.5);
        let updated = current + (1.0 - current) * lr;
        self.prey_knowledge.insert(prey_species_id, updated.clamp(0.0, 1.0));
    }
}

/// Relationship used for parent->child knowledge transfer.
#[derive(Component, Debug, Clone, Copy)]
pub struct ParentChildRelationship {
    pub parent: Entity,
    pub child: Entity,
    pub time_together: f32,
}

/// Egg component – spawned instead of a full organism for egg-laying species.
#[derive(Component, Debug, Clone, Copy)]
pub struct Egg {
    pub parent: Entity,
    pub incubation_time_remaining: f32,
    pub incubation_type: IncubationType,
}

/// Attachment of a child to a caring parent – child follows parent.
#[derive(Component, Debug, Clone, Copy)]
pub struct ParentalAttachment {
    pub parent: Entity,
    pub care_until_age: f32,
}

/// Child growth and starvation tracking.
#[derive(Component, Debug, Clone, Copy)]
pub struct ChildGrowth {
    pub growth: f32,
    pub base_rate: f32,
    pub max_rate: f32,
    pub food_deficit: f32,
    pub independence_age: f32,
}


impl CachedTraits {
    pub fn from_genome(genome: &crate::organisms::genetics::Genome) -> Self {
        use crate::organisms::genetics::traits;
        Self {
            speed: traits::express_speed(genome),
            size: traits::express_size(genome),
            metabolism_rate: traits::express_metabolism_rate(genome),
            movement_cost: traits::express_movement_cost(genome),
            max_energy: traits::express_max_energy(genome),
            reproduction_cooldown: traits::express_reproduction_cooldown(genome),
            reproduction_threshold: traits::express_reproduction_threshold(genome),
            sensory_range: traits::express_sensory_range(genome),
            aggression: traits::express_aggression(genome),
            boldness: traits::express_boldness(genome),
            mutation_rate: traits::express_mutation_rate(genome),
            foraging_drive: traits::express_foraging_drive(genome),
            risk_tolerance: traits::express_risk_tolerance(genome),
            exploration_drive: traits::express_exploration_drive(genome),
            clutch_size: traits::express_clutch_size(genome),
            offspring_energy_share: traits::express_offspring_energy_share(genome),
            hunger_memory_rate: traits::express_hunger_memory_rate(genome),
            threat_decay_rate: traits::express_threat_decay_rate(genome),
            resource_selectivity: traits::express_resource_selectivity(genome),
            consumption_rate: traits::express_consumption_rate(genome),
            attack_strength: traits::express_attack_strength(genome),
            coordination: traits::express_coordination(genome),
            forms_packs: traits::express_forms_packs(genome),
            pack_lifetime: traits::express_pack_lifetime(genome),
            pack_size_preference: traits::express_pack_size_preference(genome),
            learning_rate: traits::express_learning_rate(genome),
            teaching_ability: traits::express_teaching_ability(genome),
            spawn_type: traits::express_spawn_type(genome),
            incubation_type: traits::express_incubation_type(genome),
            incubation_duration: traits::express_incubation_duration(genome),
            parental_care_age: traits::express_parental_care_age(genome),
            meal_share_percentage: traits::express_meal_share_percentage(genome),
            growth_rate: traits::express_child_growth_rate(genome),
            max_growth_rate: traits::express_child_max_growth_rate(genome),
            father_provides_care: traits::express_father_provides_care(genome),
            can_produce_milk: traits::express_can_produce_milk(genome),
            milk_amount: traits::express_milk_amount(genome),
            knowledge_transfer_rate: traits::express_knowledge_transfer_rate(genome),
            armor: traits::express_armor(genome),
            poison_strength: traits::express_poison_strength(genome),
            flee_speed: traits::express_flee_speed(genome),
            endurance: traits::express_endurance(genome),
            hunting_strategy: traits::express_hunting_strategy(genome),
        }
    }
}

impl ReproductionCooldown {
    pub fn new(ticks: u32) -> Self {
        Self(ticks)
    }

    pub fn is_ready(&self) -> bool {
        self.0 == 0
    }

    pub fn decrement(&mut self) {
        if self.0 > 0 {
            self.0 -= 1;
        }
    }

    pub fn reset(&mut self, ticks: u32) {
        self.0 = ticks;
    }
}
