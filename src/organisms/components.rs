use bevy::prelude::*;
use glam::Vec2;

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
        Self {
            current: max,
            max,
        }
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
            base_rate: 0.01,  // 1% max energy per second
            movement_cost: 0.05,  // Additional cost for movement
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

/// Organism type (for future behavior differentiation)
#[derive(Component, Debug, Clone, Copy, PartialEq, Eq)]
pub enum OrganismType {
    Producer,   // Plants, algae - generate energy from resources
    Consumer,   // Animals - consume other organisms/resources
    Decomposer, // Fungi, bacteria - consume detritus
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

