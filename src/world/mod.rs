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
pub use cell::{ResourceType, TerrainType, RESOURCE_TYPE_COUNT};
pub use chunk::Chunk;
pub use chunk::CHUNK_SIZE;
pub use chunk::CHUNK_WORLD_SIZE;
pub use chunk::CELL_SIZE;
pub use climate::ClimateState;
pub use grid::WorldGrid;
pub use resources::*;
pub use terrain::*;
pub use events::*;
pub use plants::*;

// Re-export specific types for visualization
pub use events::{DisasterEvents, Disaster, DisasterType};

/// World size configuration
/// Maximum radius in chunks from origin (0,0)
/// Default: 100 chunks = ~6,400 world units radius
pub const MAX_WORLD_RADIUS_CHUNKS: i32 = 100;

/// Distance in chunks beyond organism activity before unloading chunks
/// Chunks further than this from any organism will be unloaded
/// Default: 50 chunks = ~3,200 world units
pub const CHUNK_UNLOAD_DISTANCE: i32 = 50;

/// Frequency of chunk unloading (every N frames)
/// Higher = less frequent but more efficient
pub const CHUNK_UNLOAD_FREQUENCY: u32 = 60; // Every 60 frames (~1 second at 60 FPS)

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

/// Resource to track world bounds and chunk unloading state
#[derive(Resource, Default)]
pub struct WorldBounds {
    /// Frame counter for periodic chunk unloading
    unload_counter: u32,
    /// Last known bounds of organism activity
    activity_bounds: Option<(i32, i32, i32, i32)>, // (min_x, max_x, min_y, max_y)
}

/// Reusable buffer for cell updates (avoids repeated allocations)
#[derive(Resource, Default)]
pub struct CellUpdateBuffer {
    /// Buffer for cell updates: (chunk_x, chunk_y, cell_x, cell_y) -> updated Cell
    update_buffer: std::collections::HashMap<(i32, i32, usize, usize), Cell>,
}

impl WorldBounds {
    pub fn update_activity_bounds(&mut self, min_x: i32, max_x: i32, min_y: i32, max_y: i32) {
        self.activity_bounds = Some((min_x, max_x, min_y, max_y));
    }
    
    pub fn should_unload_chunks(&mut self) -> bool {
        self.unload_counter += 1;
        if self.unload_counter >= CHUNK_UNLOAD_FREQUENCY {
            self.unload_counter = 0;
            true
        } else {
            false
        }
    }
    
    pub fn activity_bounds(&self) -> Option<(i32, i32, i32, i32)> {
        self.activity_bounds
    }
}

pub struct WorldPlugin;

impl Plugin for WorldPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<WorldGrid>()
            .init_resource::<ClimateState>()
            .init_resource::<DirtyChunks>()
            .init_resource::<WorldBounds>()
            .init_resource::<CellUpdateBuffer>() // Reusable buffer for cell updates
            .init_resource::<events::DisasterEvents>() // Step 9: Major disasters
            .add_systems(Startup, initialize_world)
            .add_systems(
                Update,
                (
                    update_climate,
                    mark_active_chunks,
                    unload_distant_chunks, // Unload chunks far from organisms
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
/// OPTIMIZED: Uses spatial hash - O(buckets_with_organisms × range²) instead of O(organisms × range²)
fn mark_active_chunks(
    mut dirty_chunks: ResMut<DirtyChunks>,
    spatial_hash: Res<crate::utils::SpatialHashGrid>,
) {
    const ACTIVE_RANGE: f32 = 10.0;
    dirty_chunks.active_cells.clear();
    
    // Get all buckets that contain organisms
    let organism_buckets = spatial_hash.organisms.get_active_buckets();
    
    let cell_size = spatial_hash.organisms.cell_size();
    let bucket_radius = (ACTIVE_RANGE / cell_size).ceil() as i32;
    
    // For each bucket with organisms, mark all cells within ACTIVE_RANGE
    for (bucket_x, bucket_y) in organism_buckets {
        // Convert bucket coordinates to world position (center of bucket)
        let bucket_world_x = (bucket_x as f32 + 0.5) * cell_size;
        let bucket_world_y = (bucket_y as f32 + 0.5) * cell_size;
        
        // Mark all cells within ACTIVE_RANGE of this bucket
        for dy in -bucket_radius..=bucket_radius {
            for dx in -bucket_radius..=bucket_radius {
                let check_x = bucket_world_x + (dx as f32 * cell_size);
                let check_y = bucket_world_y + (dy as f32 * cell_size);
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

/// Unload chunks that are far from organism activity
/// Runs periodically to prevent memory bloat from organisms exploring
fn unload_distant_chunks(
    mut world_grid: ResMut<WorldGrid>,
    mut world_bounds: ResMut<WorldBounds>,
    organism_query: Query<&crate::organisms::Position, With<crate::organisms::Alive>>,
) {
    // Only run periodically
    if !world_bounds.should_unload_chunks() {
        return;
    }
    
    // Find bounds of organism activity
    let mut min_chunk_x = i32::MAX;
    let mut max_chunk_x = i32::MIN;
    let mut min_chunk_y = i32::MAX;
    let mut max_chunk_y = i32::MIN;
    let mut has_organisms = false;
    
    for position in organism_query.iter() {
        has_organisms = true;
        let (chunk_x, chunk_y) = Chunk::world_to_chunk(position.x(), position.y());
        min_chunk_x = min_chunk_x.min(chunk_x);
        max_chunk_x = max_chunk_x.max(chunk_x);
        min_chunk_y = min_chunk_y.min(chunk_y);
        max_chunk_y = max_chunk_y.max(chunk_y);
    }
    
    // If no organisms, don't unload anything (might be startup)
    if !has_organisms {
        return;
    }
    
    // Update activity bounds
    world_bounds.update_activity_bounds(min_chunk_x, max_chunk_x, min_chunk_y, max_chunk_y);
    
    // Expand bounds by unload distance
    let unload_min_x = min_chunk_x - CHUNK_UNLOAD_DISTANCE;
    let unload_max_x = max_chunk_x + CHUNK_UNLOAD_DISTANCE;
    let unload_min_y = min_chunk_y - CHUNK_UNLOAD_DISTANCE;
    let unload_max_y = max_chunk_y + CHUNK_UNLOAD_DISTANCE;
    
    // Find chunks to unload
    let chunks_to_remove: Vec<_> = world_grid
        .get_chunk_coords()
        .into_iter()
        .filter(|(chunk_x, chunk_y)| {
            *chunk_x < unload_min_x
                || *chunk_x > unload_max_x
                || *chunk_y < unload_min_y
                || *chunk_y > unload_max_y
        })
        .collect();
    
    let removed_count = chunks_to_remove.len();
    
    // Remove distant chunks
    for (chunk_x, chunk_y) in chunks_to_remove {
        world_grid.remove_chunk(chunk_x, chunk_y);
    }
    
    if removed_count > 0 {
        info!("Unloaded {} distant chunks. Active chunks: {}", removed_count, world_grid.chunk_count());
    }
}

/// Update all chunks: climate and resource regeneration/decay
/// Step 10: PARALLELIZED - Processes chunks in parallel using rayon
/// OPTIMIZED: Only updates dirty cells and cells near organisms
/// OPTIMIZED: Uses buffer to minimize cell cloning
fn update_chunks(
    mut world_grid: ResMut<WorldGrid>, 
    climate: Res<ClimateState>,
    dirty_chunks: Res<DirtyChunks>,
    mut update_buffer: ResMut<CellUpdateBuffer>,
) {
    use rayon::prelude::*;
    
    let chunk_coords: Vec<_> = world_grid.get_chunk_coords();
    let climate_ref = climate.as_ref();
    
    // Clear buffer but keep capacity
    update_buffer.update_buffer.clear();
    
    // Phase 1: Collect cell coordinates that need updating (read-only, parallel)
    // Store coordinates instead of cloning cells
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
                                if chunk.get_cell(x, y).is_some() {
                                    let world_pos = Vec2::new(
                                        chunk_x as f32 * crate::world::chunk::CHUNK_WORLD_SIZE + x as f32 * crate::world::chunk::CELL_SIZE,
                                        chunk_y as f32 * crate::world::chunk::CHUNK_WORLD_SIZE + y as f32 * crate::world::chunk::CELL_SIZE,
                                    );
                                    // Store coordinates only, not cell data
                                    updates.push((chunk_x, chunk_y, x, y, world_pos));
                                }
                            }
                        }
                    }
                    updates
                })
                .unwrap_or_default()
        })
        .collect();
    
    // Phase 2: Process updates in parallel - read from world_grid, compute, store in buffer
    let cell_updates: Vec<_> = cells_to_update
        .par_iter()
        .filter_map(|(chunk_x, chunk_y, x, y, world_pos)| {
            // Read cell (immutable borrow) - clone only once for computation
            let cell = world_grid
                .get_chunk(*chunk_x, *chunk_y)
                .and_then(|chunk| chunk.get_cell(*x, *y))?;
            
            // Clone only for computation (small, temporary)
            let mut new_cell = cell.clone();
            climate::update_cell_climate(&mut new_cell, climate_ref, *world_pos);
            
            Some(((*chunk_x, *chunk_y, *x, *y), new_cell))
        })
        .collect();
    
    // Phase 3: Store updates in buffer
    for ((chunk_x, chunk_y, x, y), new_cell) in cell_updates {
        update_buffer.update_buffer.insert((chunk_x, chunk_y, x, y), new_cell);
    }
    
    // Phase 4: Apply all updates to world_grid (sequential, but fast)
    for ((chunk_x, chunk_y, x, y), new_cell) in &update_buffer.update_buffer {
        if let Some(cell) = world_grid
            .get_chunk_mut(*chunk_x, *chunk_y)
            .and_then(|chunk| chunk.get_cell_mut(*x, *y)) {
            *cell = new_cell.clone(); // Final clone during write
        }
    }
}

/// Regenerate and decay resources in all chunks
/// Step 10: PARALLELIZED - Processes chunks in parallel using rayon
/// OPTIMIZED: Sparse updates - only process cells with resources or near organisms
/// Step 8: Uses tuning parameters for ecosystem balance
/// OPTIMIZED: Uses buffer to minimize cell cloning
fn regenerate_and_decay_resources(
    mut world_grid: ResMut<WorldGrid>, 
    time: Res<Time>,
    dirty_chunks: Res<DirtyChunks>,
    tuning: Option<Res<crate::organisms::EcosystemTuning>>, // Step 8: Tuning parameters
    mut update_buffer: ResMut<CellUpdateBuffer>, // Reuse same buffer
) {
    use rayon::prelude::*;
    
    let dt = time.delta_seconds();
    let chunk_coords: Vec<_> = world_grid.get_chunk_coords();
    let tuning_ref = tuning.as_deref();
    
    // Clear buffer but keep capacity
    update_buffer.update_buffer.clear();

    // Phase 1: Collect coordinates (read-only, parallel) - no cloning yet
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
                                        // Store coordinates only, not cell data
                                        updates.push((chunk_x, chunk_y, x, y));
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
    
    // Phase 2: Process in parallel - read, compute, store in buffer
    let cell_updates: Vec<_> = cells_to_update
        .par_iter()
        .filter_map(|(chunk_x, chunk_y, x, y)| {
            // Read cell (immutable borrow) - clone only once for computation
            let cell = world_grid
                .get_chunk(*chunk_x, *chunk_y)
                .and_then(|chunk| chunk.get_cell(*x, *y))?;
            
            // Clone only for computation
            let mut new_cell = cell.clone();
            resources::regenerate_resources(&mut new_cell, dt, tuning_ref);
            resources::decay_resources(&mut new_cell, dt, tuning_ref);
            resources::quantize_resources(&mut new_cell, 0.001);
            
            Some(((*chunk_x, *chunk_y, *x, *y), new_cell))
        })
        .collect();
    
    // Phase 3: Store in buffer
    for ((chunk_x, chunk_y, x, y), new_cell) in cell_updates {
        update_buffer.update_buffer.insert((chunk_x, chunk_y, x, y), new_cell);
    }
    
    // Phase 4: Apply to world_grid
    for ((chunk_x, chunk_y, x, y), new_cell) in &update_buffer.update_buffer {
        if let Some(cell) = world_grid
            .get_chunk_mut(*chunk_x, *chunk_y)
            .and_then(|chunk| chunk.get_cell_mut(*x, *y)) {
            *cell = new_cell.clone();
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
