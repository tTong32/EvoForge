mod cell;
mod chunk;
mod climate;
mod grid;
mod resources;
mod terrain;

use bevy::prelude::*;
use bevy::time::Time;
use glam::Vec2;
use std::collections::HashSet;

pub use cell::Cell;
pub use cell::{ResourceType, TerrainType};
pub use chunk::Chunk;
pub use climate::ClimateState;
pub use grid::WorldGrid;
pub use resources::*;
pub use terrain::*;

/// Track which chunks/cells need updates (optimization 2)
#[derive(Resource, Default)]
pub struct DirtyChunks {
    /// Chunks that are dirty and need full updates
    dirty_chunks: HashSet<(i32, i32)>,
    /// Cells with organisms nearby (update these more frequently)
    active_cells: HashSet<((i32, i32), (usize, usize))>, // ((chunk_x, chunk_y), (cell_x, cell_y))
    /// Frame counter for cache decay
    frame_counter: u32,
}

impl DirtyChunks {
    pub fn mark_chunk_dirty(&mut self, chunk_x: i32, chunk_y: i32) {
        self.dirty_chunks.insert((chunk_x, chunk_y));
    }
    
    pub fn mark_cell_active(&mut self, chunk_x: i32, chunk_y: i32, cell_x: usize, cell_y: usize) {
        self.active_cells.insert(((chunk_x, chunk_y), (cell_x, cell_y)));
    }
    
    pub fn should_update_cell(&self, chunk_x: i32, chunk_y: i32, cell_x: usize, cell_y: usize) -> bool {
        // Update if chunk is dirty OR cell is active
        self.dirty_chunks.contains(&(chunk_x, chunk_y)) 
            || self.active_cells.contains(&((chunk_x, chunk_y), (cell_x, cell_y)))
    }
    
    pub fn clear_dirty_chunks(&mut self) {
        self.dirty_chunks.clear();
    }
    
    pub fn decay_active_cells(&mut self) {
        // Every 10 frames, reduce active cells to only those near organisms
        self.frame_counter += 1;
        if self.frame_counter % 10 == 0 {
            // Keep active cells for tracking, but this could be further optimized
            // For now, we'll keep them and let mark_active_chunks refresh them
        }
    }
}

pub struct WorldPlugin;

impl Plugin for WorldPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<WorldGrid>()
            .init_resource::<ClimateState>()
            .init_resource::<DirtyChunks>()
            .add_systems(Startup, initialize_world)
            .add_systems(
                Update,
                (
                    update_climate,
                    mark_active_chunks,
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

/// Mark chunks/cells as active based on organism positions
fn mark_active_chunks(
    mut dirty_chunks: ResMut<DirtyChunks>,
    organism_query: Query<&crate::organisms::Position, With<crate::organisms::Alive>>,
) {
    const ACTIVE_RANGE: f32 = 10.0; // Cells within this range of organisms are "active"
    dirty_chunks.active_cells.clear(); // Refresh active cells each frame
    
    for position in organism_query.iter() {
        let world_x = position.x();
        let world_y = position.y();
        
        // Find all cells within active range
        let cell_size = 1.0;
        let range_cells = (ACTIVE_RANGE / cell_size).ceil() as i32;
        
        for dy in -range_cells..=range_cells {
            for dx in -range_cells..=range_cells {
                let check_x = world_x + (dx as f32 * cell_size);
                let check_y = world_y + (dy as f32 * cell_size);
                let distance = Vec2::new(dx as f32, dy as f32).length() * cell_size;
                
                if distance <= ACTIVE_RANGE {
                    let (chunk_x, chunk_y) = crate::world::chunk::Chunk::world_to_chunk(check_x, check_y);
                    let (cell_x, cell_y) = crate::world::chunk::Chunk::world_to_local(check_x, check_y);
                    dirty_chunks.mark_cell_active(chunk_x, chunk_y, cell_x, cell_y);
                }
            }
        }
    }
    
    dirty_chunks.decay_active_cells();
}

/// Update all chunks: climate and resource regeneration/decay
/// OPTIMIZED: Only updates dirty cells and cells near organisms
fn update_chunks(
    mut world_grid: ResMut<WorldGrid>, 
    climate: Res<ClimateState>,
    dirty_chunks: Res<DirtyChunks>,
) {
    let chunk_coords: Vec<_> = world_grid.get_chunk_coords();

    for (chunk_x, chunk_y) in chunk_coords {
        if let Some(chunk) = world_grid.get_chunk_mut(chunk_x, chunk_y) {
            // Only update climate for active cells or if chunk is dirty
            for y in 0..crate::world::chunk::CHUNK_SIZE {
                for x in 0..crate::world::chunk::CHUNK_SIZE {
                    if dirty_chunks.should_update_cell(chunk_x, chunk_y, x, y) {
                        if let Some(cell) = chunk.get_cell_mut(x, y) {
                            let world_pos = Vec2::new(
                                chunk_x as f32 * crate::world::chunk::CHUNK_SIZE as f32 + x as f32,
                                chunk_y as f32 * crate::world::chunk::CHUNK_SIZE as f32 + y as f32,
                            );
                            climate::update_cell_climate(cell, climate.as_ref(), world_pos);
                        }
                    }
                }
            }
        }
    }
}

/// Regenerate and decay resources in all chunks
/// OPTIMIZED: Sparse updates - only process cells with resources or near organisms
fn regenerate_and_decay_resources(
    mut world_grid: ResMut<WorldGrid>, 
    time: Res<Time>,
    dirty_chunks: Res<DirtyChunks>,
) {
    let dt = time.delta_seconds();
    let chunk_coords: Vec<_> = world_grid.get_chunk_coords();

    for (chunk_x, chunk_y) in chunk_coords {
        if let Some(chunk) = world_grid.get_chunk_mut(chunk_x, chunk_y) {
            for y in 0..crate::world::chunk::CHUNK_SIZE {
                for x in 0..crate::world::chunk::CHUNK_SIZE {
                    if dirty_chunks.should_update_cell(chunk_x, chunk_y, x, y) {
                        if let Some(cell) = chunk.get_cell_mut(x, y) {
                            // Check if cell has any meaningful resources first
                            let has_resources = (0..crate::world::cell::RESOURCE_TYPE_COUNT)
                                .any(|i| cell.resource_density[i] > 0.001);
                            
                            // Only update if cell has resources OR is active (near organisms)
                            if has_resources || dirty_chunks.active_cells.contains(&((chunk_x, chunk_y), (x, y))) {
                                resources::regenerate_resources(cell, dt);
                                resources::decay_resources(cell, dt);
                                resources::quantize_resources(cell, 0.001);
                            }
                        }
                    }
                }
            }
        }
    }
}

/// Flow resources between neighboring cells (simplified diffusion)
/// OPTIMIZED: Uses direct array indexing instead of find() for O(1) access
/// Flow resources between neighboring cells (simplified diffusion)
/// OPTIMIZED: Uses flat Vec to avoid any stack allocations
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
            const RESOURCE_COUNT: usize = crate::world::cell::RESOURCE_TYPE_COUNT;
            
            // Use flat Vec to avoid any stack allocation issues
            // Layout: [cell0_r0, cell0_r1, ..., cell0_r5, cell1_r0, ...]
            let total_size = CHUNK_SIZE * CHUNK_SIZE * RESOURCE_COUNT;
            let mut temp_resources = Vec::with_capacity(total_size);
            temp_resources.resize(total_size, 0.0f32);

            // First pass: copy current resource densities into flat buffer
            for y in 0..CHUNK_SIZE {
                for x in 0..CHUNK_SIZE {
                    if let Some(cell) = chunk.get_cell(x, y) {
                        let base_idx = (y * CHUNK_SIZE + x) * RESOURCE_COUNT;
                        for i in 0..RESOURCE_COUNT {
                            temp_resources[base_idx + i] = cell.resource_density[i];
                        }
                    }
                }
            }

            // Second pass: apply diffusion using the buffered data
            for y in 0..CHUNK_SIZE {
                for x in 0..CHUNK_SIZE {
                    let index = y * CHUNK_SIZE + x;
                    let base_idx = index * RESOURCE_COUNT;
                    let mut neighbor_sum = [0.0f32; RESOURCE_COUNT];
                    let mut neighbor_count = 0;

                    for dy in -1..=1 {
                        for dx in -1..=1 {
                            if dx == 0 && dy == 0 {
                                continue;
                            }

                            let nx = x as isize + dx as isize;
                            let ny = y as isize + dy as isize;

                            if nx >= 0
                                && nx < CHUNK_SIZE as isize
                                && ny >= 0
                                && ny < CHUNK_SIZE as isize
                            {
                                let n_index = (ny as usize * CHUNK_SIZE + nx as usize) * RESOURCE_COUNT;
                                for i in 0..RESOURCE_COUNT {
                                    neighbor_sum[i] += temp_resources[n_index + i];
                                }
                                neighbor_count += 1;
                            }
                        }
                    }

                    if neighbor_count > 0 {
                        if let Some(cell) = chunk.get_cell_mut(x, y) {
                            for i in 0..RESOURCE_COUNT {
                                let old_value = temp_resources[base_idx + i];
                                let neighbor_avg = neighbor_sum[i] / neighbor_count as f32;
                                let diff = neighbor_avg - old_value;
                                cell.resource_density[i] =
                                    (old_value + diff * diffusion_rate * dt).clamp(0.0, 1.0);
                            }
                        }
                    }
                }
            }
        }
    }
}
