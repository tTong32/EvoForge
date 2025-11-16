use smallvec::SmallVec;

use crate::organisms::Genome;

/// Max plant “slots” per cell; beyond this we’ll merge tiny species.
pub const MAX_PLANT_SPECIES_PER_CELL: usize = 4;

/// Percentage-based representation of a plant species within a cell.
/// All species percentages in a cell should sum to <= 1.0.
#[derive(Debug, Clone)]
pub struct PlantSpecies {
    pub species_id: u32,
    pub genome: Genome,
    /// Fraction of the cell’s plant biomass (0.0–1.0).
    pub percentage: f32,
    /// Age in simulation seconds.
    pub age: f32,
}

/// Container type for the plant community in a single cell.
pub type PlantCommunity = SmallVec<[PlantSpecies; MAX_PLANT_SPECIES_PER_CELL]>;


