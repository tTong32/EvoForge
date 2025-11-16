use crate::organisms::genetics::Genome;

/// Bundle of expressed plant traits used by the cell-based plant system.
#[derive(Debug, Clone, Copy)]
pub struct PlantTraits {
    pub growth_rate: f32,
    pub spread_rate: f32,
    pub sunlight_efficiency: f32,
    pub water_efficiency: f32,
    pub mineral_efficiency: f32,
    pub competitive_ability: f32,
    pub defense_poison: f32,
    pub defense_thorns: f32,
    pub height: f32,
    pub root_depth: f32,
    pub lifespan: f32,
}

impl PlantTraits {
    /// Express plant traits from the shared Genome.
    /// For now this reuses existing animal trait functions but maps them differently.
    pub fn from_genome(genome: &Genome) -> Self {
        use crate::organisms::genetics::traits as animal_traits;

        Self {
            growth_rate: animal_traits::express_metabolism_rate(genome).clamp(0.01, 0.2),
            spread_rate: animal_traits::express_exploration_drive(genome).clamp(0.01, 0.3),
            sunlight_efficiency: animal_traits::express_resource_selectivity(genome)
                .clamp(0.2, 1.2),
            water_efficiency: animal_traits::express_foraging_drive(genome).clamp(0.2, 1.2),
            mineral_efficiency: animal_traits::express_mutation_rate(genome)
                .mul_add(10.0, 0.5)
                .clamp(0.1, 1.5),
            competitive_ability: animal_traits::express_aggression(genome).clamp(0.1, 1.5),
            defense_poison: animal_traits::express_risk_tolerance(genome).clamp(0.0, 1.0),
            defense_thorns: animal_traits::express_boldness(genome).clamp(0.0, 1.0),
            height: animal_traits::express_size(genome).clamp(0.1, 5.0),
            root_depth: animal_traits::express_size(genome).sqrt().clamp(0.1, 3.0),
            lifespan: animal_traits::express_max_energy(genome)
                .mul_add(0.002, 5.0)
                .clamp(10.0, 500.0),
        }
    }
}


