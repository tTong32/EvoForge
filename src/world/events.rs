use bevy::prelude::*;
use bevy::time::Time;
use glam::Vec2;
use crate::world::cell::{Cell, TerrainType, ResourceType};
use crate::world::grid::WorldGrid;
use crate::organisms::{Position, Energy, Alive};
use crate::world::climate::ClimateState;

/// Major disaster events that affect organisms and terrain
#[derive(Resource, Debug)]
pub struct DisasterEvents {
    /// Queue of active disasters
    pub active_disasters: Vec<Disaster>,
    /// Cooldown before next disaster can spawn
    pub spawn_cooldown: f32,
    /// Total disasters spawned (for tracking)
    pub total_disasters: u32,
}

impl Default for DisasterEvents {
    fn default() -> Self {
        Self {
            active_disasters: Vec::new(),
            spawn_cooldown: 500.0, // Initial cooldown (longer than climate events)
            total_disasters: 0,
        }
    }
}

/// Types of major disasters
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DisasterType {
    /// Volcanic eruption - high heat, ash, blocks sunlight
    Volcano,
    /// Meteor impact - destroys organisms, creates crater, adds minerals
    Meteor,
    /// Flood - increases water, kills low-lying organisms
    Flood,
    /// Drought - reduces water resources, increases mortality
    Drought,
}

/// A major disaster event
#[derive(Debug, Clone)]
pub struct Disaster {
    /// Unique ID for this disaster
    pub id: u32,
    /// Type of disaster
    pub disaster_type: DisasterType,
    /// Center position of the disaster
    pub center: Vec2,
    /// Radius of effect
    pub radius: f32,
    /// Intensity (0.0 to 1.0)
    pub intensity: f32,
    /// Time remaining (in seconds)
    pub time_remaining: f32,
    /// Total duration
    pub duration: f32,
    /// Whether this disaster has been processed (for one-time effects)
    pub processed: bool,
}

impl Disaster {
    /// Create a new disaster
    pub fn new(id: u32, disaster_type: DisasterType, center: Vec2, radius: f32, intensity: f32, duration: f32) -> Self {
        Self {
            id,
            disaster_type,
            center,
            radius,
            intensity,
            time_remaining: duration,
            duration,
            processed: false,
        }
    }

    /// Check if a position is within the disaster's effect radius
    pub fn contains(&self, pos: Vec2) -> bool {
        pos.distance(self.center) <= self.radius
    }

    /// Get influence factor at a position (1.0 at center, 0.0 at edge)
    pub fn influence_at(&self, pos: Vec2) -> f32 {
        let distance = pos.distance(self.center);
        if distance > self.radius {
            return 0.0;
        }
        let normalized_dist = distance / self.radius;
        // Smooth falloff
        (1.0 - normalized_dist).powf(2.0) * self.intensity
    }
}

/// Update disaster events system
pub fn update_disaster_events(
    mut disaster_events: ResMut<DisasterEvents>,
    time: Res<Time>,
    mut world_grid: ResMut<WorldGrid>,
    climate: Res<ClimateState>,
) {
    let dt = time.delta_seconds();

    // Update existing disasters
    for disaster in &mut disaster_events.active_disasters {
        disaster.time_remaining -= dt;

        // Apply disaster effects (one-time or continuous)
        apply_disaster_effects(
            &disaster,
            &mut world_grid,
            climate.as_ref(),
        );
    }

    // Remove expired disasters
    disaster_events.active_disasters.retain(|d| d.time_remaining > 0.0);

    // Spawn new disasters
    disaster_events.spawn_cooldown -= dt;
    if disaster_events.spawn_cooldown <= 0.0 {
        // Lower probability than climate events (major disasters are rarer)
        if fastrand::f32() < 0.001 {
            spawn_random_disaster(&mut disaster_events, &world_grid);
        }
        // Reset cooldown (300-1000 seconds)
        disaster_events.spawn_cooldown = fastrand::f32() * 700.0 + 300.0;
    }
}

/// Apply disaster effects to the world
fn apply_disaster_effects(
    disaster: &Disaster,
    world_grid: &mut WorldGrid,
    climate: &crate::world::ClimateState,
) {
    match disaster.disaster_type {
        DisasterType::Volcano => apply_volcano_effects(disaster, world_grid, climate),
        DisasterType::Meteor => {
            if !disaster.processed {
                apply_meteor_impact(disaster, world_grid);
            }
        },
        DisasterType::Flood => apply_flood_effects(disaster, world_grid),
        DisasterType::Drought => apply_drought_effects(disaster, world_grid),
    }
}

/// Apply volcano effects (continuous heat, ash, blocked sunlight)
fn apply_volcano_effects(
    disaster: &Disaster,
    world_grid: &mut WorldGrid,
    _climate: &crate::world::ClimateState,
) {
    let (min_chunk_x, min_chunk_y) = crate::world::chunk::Chunk::world_to_chunk(
        disaster.center.x - disaster.radius,
        disaster.center.y - disaster.radius,
    );
    let (max_chunk_x, max_chunk_y) = crate::world::chunk::Chunk::world_to_chunk(
        disaster.center.x + disaster.radius,
        disaster.center.y + disaster.radius,
    );

    for chunk_x in min_chunk_x..=max_chunk_x {
        for chunk_y in min_chunk_y..=max_chunk_y {
            if let Some(chunk) = world_grid.get_chunk_mut(chunk_x, chunk_y) {
                use crate::world::chunk::CHUNK_SIZE;
                
                for y in 0..CHUNK_SIZE {
                    for x in 0..CHUNK_SIZE {
                        let world_pos = Vec2::new(
                            chunk_x as f32 * CHUNK_SIZE as f32 + x as f32,
                            chunk_y as f32 * CHUNK_SIZE as f32 + y as f32,
                        );

                        let influence = disaster.influence_at(world_pos);
                        if influence > 0.01 {
                            if let Some(cell) = chunk.get_cell_mut(x, y) {
                                // Increase temperature significantly
                                cell.temperature = (cell.temperature + influence * 0.4).min(1.0);
                                
                                // Block sunlight near center (ash cloud)
                                if influence > 0.5 {
                                    cell.resource_density[ResourceType::Sunlight as usize] *= 0.3;
                                }
                                
                                // Add ash (detritus) resources
                                cell.add_resource(ResourceType::Detritus, influence * 0.1 * 0.016); // Per frame
                                
                                // Change terrain to volcanic if very close
                                if influence > 0.8 && cell.terrain != TerrainType::Volcanic {
                                    cell.terrain = TerrainType::Volcanic;
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

/// Apply meteor impact (one-time: destroys organisms, creates crater, adds minerals)
fn apply_meteor_impact(
    disaster: &Disaster,
    world_grid: &mut WorldGrid,
) {
    // Mark as processed so it only happens once
    // Note: We can't mark it here since we're borrowing, so we'll do it in the caller
    
    let (min_chunk_x, min_chunk_y) = crate::world::chunk::Chunk::world_to_chunk(
        disaster.center.x - disaster.radius,
        disaster.center.y - disaster.radius,
    );
    let (max_chunk_x, max_chunk_y) = crate::world::chunk::Chunk::world_to_chunk(
        disaster.center.x + disaster.radius,
        disaster.center.y + disaster.radius,
    );

    for chunk_x in min_chunk_x..=max_chunk_x {
        for chunk_y in min_chunk_y..=max_chunk_y {
            if let Some(chunk) = world_grid.get_chunk_mut(chunk_x, chunk_y) {
                use crate::world::chunk::CHUNK_SIZE;
                
                for y in 0..CHUNK_SIZE {
                    for x in 0..CHUNK_SIZE {
                        let world_pos = Vec2::new(
                            chunk_x as f32 * CHUNK_SIZE as f32 + x as f32,
                            chunk_y as f32 * CHUNK_SIZE as f32 + y as f32,
                        );

                        let influence = disaster.influence_at(world_pos);
                        if influence > 0.01 {
                            if let Some(cell) = chunk.get_cell_mut(x, y) {
                                // Add minerals (meteorite fragments)
                                cell.add_resource(ResourceType::Mineral, influence * 0.5);
                                
                                // Create crater (lower elevation)
                                cell.elevation = (cell.elevation as f32 * (1.0 - influence * 0.3)).max(0.0) as u16;
                                
                                // Destroy plants near impact
                                if influence > 0.7 {
                                    cell.resource_density[ResourceType::Plant as usize] *= 0.1;
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    info!("[DISASTER] Meteor impact at ({:.1}, {:.1}) with radius {:.1}", 
        disaster.center.x, disaster.center.y, disaster.radius);
}

/// Apply flood effects (increases water, damages low-lying areas)
fn apply_flood_effects(
    disaster: &Disaster,
    world_grid: &mut WorldGrid,
) {
    let (min_chunk_x, min_chunk_y) = crate::world::chunk::Chunk::world_to_chunk(
        disaster.center.x - disaster.radius,
        disaster.center.y - disaster.radius,
    );
    let (max_chunk_x, max_chunk_y) = crate::world::chunk::Chunk::world_to_chunk(
        disaster.center.x + disaster.radius,
        disaster.center.y + disaster.radius,
    );

    for chunk_x in min_chunk_x..=max_chunk_x {
        for chunk_y in min_chunk_y..=max_chunk_y {
            if let Some(chunk) = world_grid.get_chunk_mut(chunk_x, chunk_y) {
                use crate::world::chunk::CHUNK_SIZE;
                
                for y in 0..CHUNK_SIZE {
                    for x in 0..CHUNK_SIZE {
                        let world_pos = Vec2::new(
                            chunk_x as f32 * CHUNK_SIZE as f32 + x as f32,
                            chunk_y as f32 * CHUNK_SIZE as f32 + y as f32,
                        );

                        let influence = disaster.influence_at(world_pos);
                        if influence > 0.01 {
                            if let Some(cell) = chunk.get_cell_mut(x, y) {
                                // Increase water resources
                                cell.add_resource(ResourceType::Water, influence * 0.2 * 0.016); // Per frame
                                
                                // Increase humidity
                                cell.humidity = (cell.humidity + influence * 0.3 * 0.016).min(1.0);
                                
                                // Damage low-lying areas (reduce elevation-based resources)
                                if cell.elevation < 10000 {
                                    let low_lying_factor = 1.0 - (cell.elevation as f32 / 10000.0);
                                    cell.resource_density[ResourceType::Plant as usize] *= 
                                        (1.0 - influence * low_lying_factor * 0.1 * 0.016).max(0.0);
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

/// Spawn a random disaster (called from update_disaster_events)
fn spawn_random_disaster(
    disaster_events: &mut DisasterEvents,
    world_grid: &WorldGrid,
) {
    // Get a random position in the world (within loaded chunks)
    let chunk_coords = world_grid.get_chunk_coords();
    if chunk_coords.is_empty() {
        return;
    }

    let (chunk_x, chunk_y) = chunk_coords[fastrand::usize(..chunk_coords.len())];
    let center = Vec2::new(
        (chunk_x as f32 + fastrand::f32()) * crate::world::chunk::CHUNK_SIZE as f32,
        (chunk_y as f32 + fastrand::f32()) * crate::world::chunk::CHUNK_SIZE as f32,
    );

    // Choose disaster type
    let disaster_type = match fastrand::u8(..4) {
        0 => DisasterType::Volcano,
        1 => DisasterType::Meteor,
        2 => DisasterType::Flood,
        _ => DisasterType::Drought,
    };

    // Set parameters based on type
    let (radius, intensity, duration) = match disaster_type {
        DisasterType::Volcano => (80.0 + fastrand::f32() * 40.0, 0.7 + fastrand::f32() * 0.3, 300.0),
        DisasterType::Meteor => (30.0 + fastrand::f32() * 20.0, 0.8 + fastrand::f32() * 0.2, 1.0), // Instant
        DisasterType::Flood => (60.0 + fastrand::f32() * 40.0, 0.6 + fastrand::f32() * 0.4, 200.0),
        DisasterType::Drought => (100.0 + fastrand::f32() * 50.0, 0.5 + fastrand::f32() * 0.5, 400.0),
    };

    let disaster_id = disaster_events.total_disasters;
    let disaster = Disaster::new(disaster_id, disaster_type, center, radius, intensity, duration);
    disaster_events.active_disasters.push(disaster);
    disaster_events.total_disasters += 1;

    info!("[DISASTER] {:?} spawned at ({:.1}, {:.1}) with radius {:.1}", 
        disaster_type, center.x, center.y, radius);
}

/// Apply drought effects (reduces water, increases mortality pressure)
fn apply_drought_effects(
    disaster: &Disaster,
    world_grid: &mut WorldGrid,
) {
    let (min_chunk_x, min_chunk_y) = crate::world::chunk::Chunk::world_to_chunk(
        disaster.center.x - disaster.radius,
        disaster.center.y - disaster.radius,
    );
    let (max_chunk_x, max_chunk_y) = crate::world::chunk::Chunk::world_to_chunk(
        disaster.center.x + disaster.radius,
        disaster.center.y + disaster.radius,
    );

    for chunk_x in min_chunk_x..=max_chunk_x {
        for chunk_y in min_chunk_y..=max_chunk_y {
            if let Some(chunk) = world_grid.get_chunk_mut(chunk_x, chunk_y) {
                use crate::world::chunk::CHUNK_SIZE;
                
                for y in 0..CHUNK_SIZE {
                    for x in 0..CHUNK_SIZE {
                        let world_pos = Vec2::new(
                            chunk_x as f32 * CHUNK_SIZE as f32 + x as f32,
                            chunk_y as f32 * CHUNK_SIZE as f32 + y as f32,
                        );

                        let influence = disaster.influence_at(world_pos);
                        if influence > 0.01 {
                            if let Some(cell) = chunk.get_cell_mut(x, y) {
                                // Reduce water resources
                                cell.resource_density[ResourceType::Water as usize] *= 
                                    (1.0 - influence * 0.15 * 0.016).max(0.0);
                                
                                // Reduce humidity
                                cell.humidity = (cell.humidity - influence * 0.2 * 0.016).max(0.0);
                                
                                // Increase temperature
                                cell.temperature = (cell.temperature + influence * 0.1 * 0.016).min(1.0);
                                
                                // Reduce plant growth
                                cell.resource_density[ResourceType::Plant as usize] *= 
                                    (1.0 - influence * 0.1 * 0.016).max(0.0);
                            }
                        }
                    }
                }
            }
        }
    }
}

/// System to apply disaster damage to organisms
pub fn apply_disaster_damage_system(
    mut disaster_events: ResMut<DisasterEvents>,
    mut organism_query: Query<(&crate::organisms::Position, &mut crate::organisms::Energy), With<crate::organisms::Alive>>,
    time: Res<Time>,
) {
    let dt = time.delta_seconds();

    // Process each disaster
    for disaster_idx in 0..disaster_events.active_disasters.len() {
        let disaster = disaster_events.active_disasters[disaster_idx].clone();
        let pos = disaster.center;
        let radius = disaster.radius;
        let disaster_type = disaster.disaster_type;
        let processed = disaster.processed;

        // Apply damage to organisms within range
        for (position, mut energy) in organism_query.iter_mut() {
            let organism_pos = Vec2::new(position.x(), position.y());
            let distance = organism_pos.distance(pos);

            if distance <= radius {
                let influence = disaster.influence_at(organism_pos);
                
                match disaster_type {
                    DisasterType::Volcano => {
                        // High heat damage
                        let damage = influence * 0.3 * dt;
                        energy.current = (energy.current - damage).max(0.0);
                    },
                    DisasterType::Meteor => {
                        // Instant massive damage near impact (only once)
                        if !processed && influence > 0.8 {
                            energy.current *= 0.1; // 90% damage
                        }
                    },
                    DisasterType::Flood => {
                        // Drowning damage for low-lying organisms
                        if influence > 0.7 {
                            let damage = influence * 0.15 * dt;
                            energy.current = (energy.current - damage).max(0.0);
                        }
                    },
                    DisasterType::Drought => {
                        // Starvation damage (reduced resources)
                        let damage = influence * 0.1 * dt;
                        energy.current = (energy.current - damage).max(0.0);
                    },
                }
            }
        }

        // Mark meteor as processed after applying damage (only mark once)
        if matches!(disaster_type, DisasterType::Meteor) && !processed {
            disaster_events.active_disasters[disaster_idx].processed = true;
        }
    }
}

/// Mark disaster as processed (for one-time effects like meteor)
pub fn mark_disaster_processed(
    mut disaster_events: ResMut<DisasterEvents>,
    disaster_idx: usize,
) {
    if let Some(disaster) = disaster_events.active_disasters.get_mut(disaster_idx) {
        disaster.processed = true;
    }
}
