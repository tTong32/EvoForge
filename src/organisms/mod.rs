mod behavior;
mod components;
mod genetics;
mod speciation;
mod systems;
mod tuning;
mod ecosystem_stats;
mod disease;
mod coevolution;
mod plant_traits;
mod offspring;

pub use behavior::*;
use bevy::prelude::*;
pub use components::*;
pub use genetics::*;
pub use speciation::*;
pub use tuning::*;
pub use ecosystem_stats::*;
pub use disease::*;
pub use coevolution::*;
pub use plant_traits::*;

// Re-export specific types for visualization
pub use disease::Infected;

pub struct OrganismPlugin;

impl Plugin for OrganismPlugin {
    fn build(&self, app: &mut App) {
        app
            .init_resource::<systems::SpatialHashTracker>()
            .init_resource::<systems::PredationEventGrouping>() // Optimization: reuse HashMap
            .init_resource::<systems::SpatialQueryBuffer>() // Optimization: reuse Vec buffer for spatial queries
            .init_resource::<systems::ReproductionEventBuffer>() // Optimization: reuse Vec buffer for reproduction events
            .init_resource::<systems::GlobalRng>() // Optimization: resource-backed RNG
            .init_resource::<systems::EligibleChildrenSet>() // Optimization: reuse HashSet for eligible children
            .init_resource::<systems::CellModificationsBuffer>() // Optimization: batch cell modifications
            .init_resource::<crate::utils::SpatialHashGrid>()
            .init_resource::<behavior::SensoryDataCache>() // Add sensory cache (optimization 3)
            .init_resource::<speciation::SpeciesTracker>() // Step 8: Speciation system
            .init_resource::<speciation::SpeciationBuffer>() // Optimization: reuse HashMap for speciation
            .init_resource::<tuning::EcosystemTuning>() // Step 8: Tuning parameters
            .init_resource::<ecosystem_stats::EcosystemStats>() // Step 8: Ecosystem statistics
            .init_resource::<ecosystem_stats::SpeciesFitnessLogger>() // AI model training data logger
            .init_resource::<disease::DiseaseSystem>() // Step 9: Disease system
            .init_resource::<disease::DiseaseSystemBuffers>() // Optimization: reuse disease system buffers
            .init_resource::<coevolution::CoEvolutionSystem>() // Step 9: Co-evolution system
            .init_resource::<coevolution::CoEvolutionSystemBuffers>() // Optimization: reuse coevolution system buffers
            .add_event::<systems::PredationDamageEvent>()
            .add_systems(Startup, systems::spawn_initial_organisms)
            .add_systems(
                Update,
                (
                    systems::update_spatial_hash,
                    systems::update_metabolism,
                    systems::update_behavior,
                    systems::update_movement,
                    systems::handle_eating,
                    systems::apply_predation_damage,
                    systems::update_age,
                    systems::handle_reproduction,
                )
            )
            .add_systems(
                Update,
                (
                    systems::handle_death,
                    offspring::update_egg_incubation,
                    systems::update_parent_child_learning,
                    update_speciation,
                    disease::update_disease_system,
                    disease::update_infected_organisms_system,
                    coevolution::update_coevolution_system,
                )
            )
            .add_systems(
                Update,
                (
                    ecosystem_stats::collect_ecosystem_stats, // Step 8: Ecosystem statistics
                    ecosystem_stats::log_species_fitness, // AI model training data logging
                )
            );
    }
}

