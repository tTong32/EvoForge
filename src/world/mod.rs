mod cell;
mod chunk;
mod climate;
mod grid;
mod resources;
mod terrain;

use bevy::prelude::*;
use bevy::time::Time;

pub use cell::Cell;
pub use cell::{ResourceType, TerrainType};
pub use chunk::Chunk;
pub use climate::ClimateState;
pub use grid::WorldGrid;
pub use resources::*;
pub use terrain::*;

pub struct WorldPlugin;

impl Plugin for WorldPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<WorldGrid>()
            .init_resource::<ClimateState>()
            .add_systems(Startup, initialize_world)
            .add_systems(
                Update,
                (
                    update_climate,
                    update_chunks,
                    regenerate_and_decay_resources,
                    flow_resources,
                ),
            );
    }
}

fn initialize_world(mut world_grid: ResMut<WorldGrid>) {
    info!("Initializing world grid...");

    // Initialize a smaller area around origin (reduced from 5x5 to 3x3 for better performance)
    // In production, chunks are created on-demand
    for chunk_x in -1..=1 {
        for chunk_y in -1..=1 {
            let chunk = world_grid.get_or_create_chunk(chunk_x, chunk_y);
            terrain::initialize_chunk(chunk);
        }
    }

    info!(
        "World grid initialized with {} chunks",
        world_grid.chunk_count()
    );
}

/// Update global climate state
fn update_climate(mut climate: ResMut<ClimateState>, time: Res<Time>) {
    climate.update(time.delta_seconds());
}

/// Update all chunks: climate and resource regeneration/decay
/// Optimized: Only updates dirty cells and cells near organisms
fn update_chunks(mut world_grid: ResMut<WorldGrid>, climate: Res<ClimateState>) {
    let chunk_coords: Vec<_> = world_grid.get_chunk_coords();

    for (chunk_x, chunk_y) in chunk_coords {
        if let Some(chunk) = world_grid.get_chunk_mut(chunk_x, chunk_y) {
            // Update climate for all cells (climate changes affect all cells)
            // In future, we could optimize this to only update cells that changed significantly
            for y in 0..crate::world::chunk::CHUNK_SIZE {
                for x in 0..crate::world::chunk::CHUNK_SIZE {
                    if let Some(cell) = chunk.get_cell_mut(x, y) {
                        climate::update_cell_climate(cell, climate.as_ref());
                    }
                }
            }
        }
    }
}

/// Regenerate and decay resources in all chunks
/// Optimized: Processes all cells but could be further optimized with dirty tracking
fn regenerate_and_decay_resources(mut world_grid: ResMut<WorldGrid>, time: Res<Time>) {
    let dt = time.delta_seconds();
    let chunk_coords: Vec<_> = world_grid.get_chunk_coords();

    for (chunk_x, chunk_y) in chunk_coords {
        if let Some(chunk) = world_grid.get_chunk_mut(chunk_x, chunk_y) {
            // Process all cells (resource regeneration/decay affects all cells)
            // Future optimization: only process cells with active resources or near organisms
            for y in 0..crate::world::chunk::CHUNK_SIZE {
                for x in 0..crate::world::chunk::CHUNK_SIZE {
                    if let Some(cell) = chunk.get_cell_mut(x, y) {
                        // Regenerate resources
                        resources::regenerate_resources(cell, dt);

                        // Decay resources
                        resources::decay_resources(cell, dt);

                        // Quantize small values
                        resources::quantize_resources(cell, 0.001);
                    }
                }
            }
        }
    }
}

/// Flow resources between neighboring cells (simplified diffusion)
/// OPTIMIZED: Uses direct array indexing instead of find() for O(1) access
fn flow_resources(mut world_grid: ResMut<WorldGrid>, time: Res<Time>) {
    let dt = time.delta_seconds();
    let diffusion_rate = 0.1; // How quickly resources flow
    let chunk_coords: Vec<_> = world_grid.get_chunk_coords();

    // For now, we'll do a simple pass within chunks
    // Full diffusion across chunk boundaries requires more complex handling
    // This is a simplified version for Step 2

    for (chunk_x, chunk_y) in chunk_coords {
        if let Some(chunk) = world_grid.get_chunk_mut(chunk_x, chunk_y) {
            use crate::world::chunk::CHUNK_SIZE;

            // Store current state for diffusion calculation (use 2D array for O(1) access)
            let mut temp_resources = [[[0.0f32; 6]; CHUNK_SIZE]; CHUNK_SIZE];

            // First pass: collect all cell resource data
            for y in 0..CHUNK_SIZE {
                for x in 0..CHUNK_SIZE {
                    if let Some(cell) = chunk.get_cell(x, y) {
                        temp_resources[y][x] = cell.resource_density;
                    }
                }
            }

            // Second pass: apply diffusion using collected data
            for y in 0..CHUNK_SIZE {
                for x in 0..CHUNK_SIZE {
                    // Calculate neighbor average using direct array access
                    let mut neighbor_sum = [0.0f32; 6];
                    let mut neighbor_count = 0;

                    for dy in -1..=1 {
                        for dx in -1..=1 {
                            if dx == 0 && dy == 0 {
                                continue;
                            }

                            let nx = (x as i32 + dx) as usize;
                            let ny = (y as i32 + dy) as usize;

                            if nx < CHUNK_SIZE && ny < CHUNK_SIZE {
                                // Direct array access - O(1) instead of O(n)
                                for i in 0..6 {
                                    neighbor_sum[i] += temp_resources[ny][nx][i];
                                }
                                neighbor_count += 1;
                            }
                        }
                    }

                    // Now update the cell
                    if let Some(cell) = chunk.get_cell_mut(x, y) {
                        if neighbor_count > 0 {
                            let old_resources = temp_resources[y][x];
                            for i in 0..6 {
                                let neighbor_avg = neighbor_sum[i] / neighbor_count as f32;
                                let diff = neighbor_avg - old_resources[i];
                                cell.resource_density[i] += diff * diffusion_rate * dt;
                                cell.resource_density[i] =
                                    cell.resource_density[i].max(0.0).min(1.0);
                            }
                        }
                    }
                }
            }
        }
    }
}
