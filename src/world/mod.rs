mod cell;
mod chunk;
mod climate;
mod grid;
mod resources;
mod terrain;
mod events;
mod plants;
mod plant_systems;

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
pub use events::*;
pub use plants::*;

// Re-export specific types for visualization
pub use events::{DisasterEvents, Disaster, DisasterType};

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
            .init_resource::<events::DisasterEvents>() // Step 9: Major disasters
            .add_systems(Startup, initialize_world)
            .add_systems(
                Update,
                (
                    update_climate,
                    mark_active_chunks,
                    update_chunks,
                    regenerate_and_decay_resources,
                    flow_resources,
                    plant_systems::update_plants_system,
                    plant_systems::plant_spread_system,
                    events::update_disaster_events, // Step 9: Update disasters
                ),
            )
            .add_systems(
                Update,
                events::apply_disaster_damage_system, // Step 9: Apply disaster damage to organisms
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

    // Seed founder plant species in the initialized chunks.
    plant_systems::initialize_founder_plants(&mut world_grid);

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
/// Step 10: PARALLELIZED - Processes chunks in parallel using rayon
/// OPTIMIZED: Only updates dirty cells and cells near organisms
fn update_chunks(
    mut world_grid: ResMut<WorldGrid>, 
    climate: Res<ClimateState>,
    dirty_chunks: Res<DirtyChunks>,
) {
    use rayon::prelude::*;
    
    let chunk_coords: Vec<_> = world_grid.get_chunk_coords();
    let climate_ref = climate.as_ref();
    
    // Collect cells that need updating (read-only phase)
    let cells_to_update: Vec<_> = chunk_coords
        .par_iter()
        .flat_map(|&(chunk_x, chunk_y)| {
            world_grid
                .get_chunk(chunk_x, chunk_y)
                .map(|chunk| {
                    let mut updates = Vec::new();
                    for y in 0..crate::world::chunk::CHUNK_SIZE {
                        for x in 0..crate::world::chunk::CHUNK_SIZE {
                            if dirty_chunks.should_update_cell(chunk_x, chunk_y, x, y) {
                                if let Some(cell) = chunk.get_cell(x, y) {
                                    let world_pos = Vec2::new(
                                        chunk_x as f32 * crate::world::chunk::CHUNK_SIZE as f32 + x as f32,
                                        chunk_y as f32 * crate::world::chunk::CHUNK_SIZE as f32 + y as f32,
                                    );
                                    // Clone cell data for parallel processing
                                    let cell_data = (chunk_x, chunk_y, x, y, world_pos, cell.clone());
                                    updates.push(cell_data);
                                }
                            }
                        }
                    }
                    updates
                })
                .unwrap_or_default()
        })
        .collect();
    
    // Process updates in parallel (compute new climate values)
    let updated_cells: Vec<_> = cells_to_update
        .par_iter()
        .map(|(chunk_x, chunk_y, x, y, world_pos, cell)| {
            let mut new_cell = cell.clone();
            climate::update_cell_climate(&mut new_cell, climate_ref, *world_pos);
            (*chunk_x, *chunk_y, *x, *y, new_cell)
        })
        .collect();
    
    // Write back results (sequential, but fast)
    for (chunk_x, chunk_y, x, y, new_cell) in updated_cells {
        if let Some(cell) = world_grid.get_chunk_mut(chunk_x, chunk_y)
            .and_then(|chunk| chunk.get_cell_mut(x, y)) {
            *cell = new_cell;
        }
    }
}

/// Regenerate and decay resources in all chunks
/// Step 10: PARALLELIZED - Processes chunks in parallel using rayon
/// OPTIMIZED: Sparse updates - only process cells with resources or near organisms
/// Step 8: Uses tuning parameters for ecosystem balance
fn regenerate_and_decay_resources(
    mut world_grid: ResMut<WorldGrid>, 
    time: Res<Time>,
    dirty_chunks: Res<DirtyChunks>,
    tuning: Option<Res<crate::organisms::EcosystemTuning>>, // Step 8: Tuning parameters
) {
    use rayon::prelude::*;
    
    let dt = time.delta_seconds();
    let chunk_coords: Vec<_> = world_grid.get_chunk_coords();
    let tuning_ref = tuning.as_deref();

    // Collect cells that need updating (read-only phase)
    let cells_to_update: Vec<_> = chunk_coords
        .par_iter()
        .flat_map(|&(chunk_x, chunk_y)| {
            world_grid
                .get_chunk(chunk_x, chunk_y)
                .map(|chunk| {
                    let mut updates = Vec::new();
                    for y in 0..crate::world::chunk::CHUNK_SIZE {
                        for x in 0..crate::world::chunk::CHUNK_SIZE {
                            if dirty_chunks.should_update_cell(chunk_x, chunk_y, x, y) {
                                if let Some(cell) = chunk.get_cell(x, y) {
                                    // Check if cell has any meaningful resources first
                                    let has_resources = (0..crate::world::cell::RESOURCE_TYPE_COUNT)
                                        .any(|i| cell.resource_density[i] > 0.001);
                                    
                                    // Only update if cell has resources OR is active (near organisms)
                                    if has_resources || dirty_chunks.active_cells.contains(&((chunk_x, chunk_y), (x, y))) {
                                        updates.push((chunk_x, chunk_y, x, y, cell.clone()));
                                    }
                                }
                            }
                        }
                    }
                    updates
                })
                .unwrap_or_default()
        })
        .collect();
    
    // Process updates in parallel
    let updated_cells: Vec<_> = cells_to_update
        .par_iter()
        .map(|(chunk_x, chunk_y, x, y, cell)| {
            let mut new_cell = cell.clone();
            resources::regenerate_resources(&mut new_cell, dt, tuning_ref);
            resources::decay_resources(&mut new_cell, dt, tuning_ref);
            resources::quantize_resources(&mut new_cell, 0.001);
            (*chunk_x, *chunk_y, *x, *y, new_cell)
        })
        .collect();
    
    // Write back results (sequential, but fast)
    for (chunk_x, chunk_y, x, y, new_cell) in updated_cells {
        if let Some(cell) = world_grid.get_chunk_mut(chunk_x, chunk_y)
            .and_then(|chunk| chunk.get_cell_mut(x, y)) {
            *cell = new_cell;
        }
    }
}

/// Flow resources between neighboring cells (simplified diffusion)
/// Step 10: PARALLELIZED - Processes chunks in parallel using rayon
/// OPTIMIZED: Uses direct array indexing instead of find() for O(1) access
/// OPTIMIZED: Uses flat Vec to avoid any stack allocations
fn flow_resources(mut world_grid: ResMut<WorldGrid>, time: Res<Time>) {
    use rayon::prelude::*;
    
    let dt = time.delta_seconds();
    let diffusion_rate = 0.1; // How quickly resources flow
    let chunk_coords: Vec<_> = world_grid.get_chunk_coords();

    // Step 10: Process chunks in parallel
    // For now, we'll do a simple pass within chunks
    // Full diffusion across chunk boundaries requires more complex handling
    // This is a simplified version for Step 2

    // Collect chunk data for parallel processing
    let chunk_data: Vec<_> = chunk_coords
        .par_iter()
        .filter_map(|&(chunk_x, chunk_y)| {
            world_grid.get_chunk(chunk_x, chunk_y).map(|chunk| {
                use crate::world::chunk::CHUNK_SIZE;
                const RESOURCE_COUNT: usize = crate::world::cell::RESOURCE_TYPE_COUNT;
                
                // Copy chunk data for parallel processing
                let mut temp_resources = Vec::with_capacity(CHUNK_SIZE * CHUNK_SIZE * RESOURCE_COUNT);
                temp_resources.resize(CHUNK_SIZE * CHUNK_SIZE * RESOURCE_COUNT, 0.0f32);
                
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
                
                (chunk_x, chunk_y, temp_resources)
            })
        })
        .collect();
    
    // Process diffusion in parallel
    let updated_chunks: Vec<_> = chunk_data
        .par_iter()
        .map(|(chunk_x, chunk_y, temp_resources)| {
            use crate::world::chunk::CHUNK_SIZE;
            const RESOURCE_COUNT: usize = crate::world::cell::RESOURCE_TYPE_COUNT;
            
            let mut new_resources = temp_resources.clone();
            
            // Apply diffusion
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
                        for i in 0..RESOURCE_COUNT {
                            let old_value = temp_resources[base_idx + i];
                            let neighbor_avg = neighbor_sum[i] / neighbor_count as f32;
                            let diff = neighbor_avg - old_value;
                            new_resources[base_idx + i] =
                                (old_value + diff * diffusion_rate * dt).clamp(0.0, 1.0);
                        }
                    }
                }
            }
            
            (*chunk_x, *chunk_y, new_resources)
        })
        .collect();
    
    // Write back results
    for (chunk_x, chunk_y, new_resources) in updated_chunks {
        if let Some(chunk) = world_grid.get_chunk_mut(chunk_x, chunk_y) {
            use crate::world::chunk::CHUNK_SIZE;
            const RESOURCE_COUNT: usize = crate::world::cell::RESOURCE_TYPE_COUNT;
            
            for y in 0..CHUNK_SIZE {
                for x in 0..CHUNK_SIZE {
                    if let Some(cell) = chunk.get_cell_mut(x, y) {
                        let base_idx = (y * CHUNK_SIZE + x) * RESOURCE_COUNT;
                        for i in 0..RESOURCE_COUNT {
                            cell.resource_density[i] = new_resources[base_idx + i];
                        }
                    }
                }
            }
        }
    }
}
