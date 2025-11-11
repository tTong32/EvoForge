mod behavior;
mod components;
mod genetics;
mod systems;

pub use behavior::*;
use bevy::prelude::*;
pub use components::*;
pub use genetics::*;

pub struct OrganismPlugin;

impl Plugin for OrganismPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<systems::TrackedOrganism>()
            .init_resource::<systems::AllOrganismsLogger>()
            .init_resource::<crate::utils::SpatialHashGrid>()
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
                )
                    .chain(),
            )
            .add_systems(
                Update,
                (systems::log_all_organisms, systems::log_tracked_organism).chain(),
            );
    }
}
