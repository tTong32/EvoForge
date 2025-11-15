mod behavior;
mod components;
mod genetics;
mod speciation;
mod systems;
mod tuning;
mod ecosystem_stats;
mod disease;
mod coevolution;

pub use behavior::*;
use bevy::prelude::*;
pub use components::*;
pub use genetics::*;
pub use speciation::*;
pub use tuning::*;
pub use ecosystem_stats::*;
pub use disease::*;
pub use coevolution::*;

// Re-export specific types for visualization
pub use disease::Infected;

pub struct OrganismPlugin;

impl Plugin for OrganismPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<systems::TrackedOrganism>()
            .init_resource::<systems::AllOrganismsLogger>()
            .init_resource::<systems::SpatialHashTracker>()
            .init_resource::<crate::utils::SpatialHashGrid>()
            .init_resource::<behavior::SensoryDataCache>() // Add sensory cache (optimization 3)
            .init_resource::<speciation::SpeciesTracker>() // Step 8: Speciation system
            .init_resource::<tuning::EcosystemTuning>() // Step 8: Tuning parameters
            .init_resource::<ecosystem_stats::EcosystemStats>() // Step 8: Ecosystem statistics
            .init_resource::<disease::DiseaseSystem>() // Step 9: Disease system
            .init_resource::<coevolution::CoEvolutionSystem>() // Step 9: Co-evolution system
            .add_systems(Startup, systems::spawn_initial_organisms)
            .add_systems(
                Update,
                (
                    systems::update_spatial_hash,
                    systems::update_metabolism,
                    systems::update_behavior,
                    systems::update_movement,
                    systems::handle_eating,
                    systems::update_age,
                    systems::handle_reproduction,
                    systems::handle_death,
                    update_speciation, // Step 8: Update species assignments
                    disease::update_disease_system, // Step 9: Update diseases (spawn and spread)
                    disease::update_infected_organisms_system, // Step 9: Update infected organisms (damage)
                    coevolution::update_coevolution_system, // Step 9: Update co-evolution
                )
                    .chain(),
            )
            .add_systems(
                Update,
                (
                    ecosystem_stats::collect_ecosystem_stats, // Step 8: Ecosystem statistics
                    systems::log_all_organisms,
                    systems::log_tracked_organism,
                ).chain(),
            );
    }
}
