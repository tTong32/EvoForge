mod chunk;
mod cell;
mod grid;

use bevy::prelude::*;

pub use chunk::Chunk;
pub use cell::Cell;
pub use cell::TerrainType;
pub use grid::WorldGrid;

pub struct WorldPlugin;

impl Plugin for WorldPlugin {
    fn build(&self, app: &mut App) {
        app
            .init_resource::<WorldGrid>()
            .add_systems(Startup, initialize_world)
            .add_systems(Update, update_world);
    }
}

fn initialize_world(_world_grid: ResMut<WorldGrid>) {
    info!("Initializing world grid...");
    // World will be initialized with chunks as needed (sparse loading)
    info!("World grid initialized");
}

fn update_world(_world_grid: Res<WorldGrid>) {
    // Placeholder for world update logic
    // Will handle chunk updates, climate simulation, etc.
}

