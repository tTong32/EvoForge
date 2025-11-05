use bevy::prelude::*;
use crate::organisms::components::*;
use crate::world::WorldGrid;
use rand::Rng;

/// Resource to track which organism we're logging
#[derive(Resource, Default)]
pub struct TrackedOrganism {
    entity: Option<Entity>,
    log_counter: u32,
}

/// Spawn initial organisms in the world
pub fn spawn_initial_organisms(
    mut commands: Commands,
    mut tracked: ResMut<TrackedOrganism>,
    _world_grid: Res<WorldGrid>,
) {
    info!("Spawning initial organisms...");
    
    let mut rng = rand::thread_rng();
    let spawn_count = 100; // Start with 100 organisms
    
    // Spawn organisms randomly within initialized chunks
    // Chunks are from -1 to 1, each chunk is 64x64 cells
    let world_size = 3 * 64; // 3 chunks * 64 cells
    let spawn_range = world_size as f32 / 2.0; // -range to +range
    
    let mut first_entity = None;
    
    for i in 0..spawn_count {
        let x = rng.gen_range(-spawn_range..spawn_range);
        let y = rng.gen_range(-spawn_range..spawn_range);
        
        // Random organism properties
        let max_energy = rng.gen_range(50.0..100.0);
        let size = rng.gen_range(0.5..2.0);
        let organism_type = match rng.gen_range(0..3) {
            0 => OrganismType::Producer,
            1 => OrganismType::Consumer,
            _ => OrganismType::Decomposer,
        };
        
        // Random initial velocity
        let vel_x = rng.gen_range(-10.0..10.0);
        let vel_y = rng.gen_range(-10.0..10.0);
        
        let entity = commands.spawn((
            Position::new(x, y),
            Velocity::new(vel_x, vel_y),
            Energy::new(max_energy),
            Age::new(),
            Size::new(size),
            Metabolism::default(),
            SpeciesId::new(0), // All start as same species for now
            organism_type,
            Alive,
        )).id();
        
        // Track the first organism spawned
        if i == 0 {
            first_entity = Some(entity);
        }
    }
    
    // Set the first organism as the tracked one
    if let Some(entity) = first_entity {
        tracked.entity = Some(entity);
        info!("[TRACKED] Started tracking organism entity: {:?}", entity);
        info!("[TRACKED] Logging will begin after 10 ticks...");
    }
    
    info!("Spawned {} organisms", spawn_count);
}

/// Update metabolism - organisms consume energy over time
pub fn update_metabolism(
    mut query: Query<(&mut Energy, &Velocity, &Metabolism, &Size)>,
    time: Res<Time>,
) {
    let dt = time.delta_seconds();
    
    for (mut energy, velocity, metabolism, size) in query.iter_mut() {
        // Base metabolic cost (proportional to size)
        let base_cost = metabolism.base_rate * size.value() * dt;
        
        // Movement cost (proportional to speed)
        let speed = velocity.0.length();
        let movement_cost = speed * metabolism.movement_cost * dt;
        
        // Total energy consumed
        let total_cost = base_cost + movement_cost;
        
        // Deduct energy
        energy.current -= total_cost;
        energy.current = energy.current.max(0.0);
    }
}

/// Update organism movement (simple wandering behavior)
pub fn update_movement(
    mut query: Query<(&mut Position, &mut Velocity, &Energy, &Size, Entity)>,
    time: Res<Time>,
    tracked: ResMut<TrackedOrganism>,
) {
    let dt = time.delta_seconds();
    let mut rng = rand::thread_rng();
    let mut direction_changed = false;
    
    for (mut position, mut velocity, energy, size, entity) in query.iter_mut() {
        // Skip if dead or very low energy
        if energy.is_dead() || energy.ratio() < 0.1 {
            velocity.0 = Vec2::ZERO;
            if tracked.entity == Some(entity) {
                info!("[TRACKED] Organism stopped moving (low energy: {:.2}%)", energy.ratio() * 100.0);
            }
            continue;
        }
        
        // Simple wandering behavior: random walk with momentum
        // Occasionally add random velocity changes
        if rng.gen_bool(0.05) { // 5% chance per frame to change direction
            let angle = rng.gen_range(0.0..std::f32::consts::TAU);
            let speed = rng.gen_range(5.0..15.0);
            velocity.0 = Vec2::from_angle(angle) * speed;
            
            if tracked.entity == Some(entity) {
                direction_changed = true;
            }
        }
        
        // Apply velocity damping (friction)
        velocity.0 *= 0.95;
        
        // Clamp velocity to max speed (based on size - larger organisms move slower)
        let max_speed = 20.0 / size.value().max(0.5);
        if velocity.0.length() > max_speed {
            velocity.0 = velocity.0.normalize() * max_speed;
        }
        
        // Update position
        let old_position = position.0;
        position.0 += velocity.0 * dt;
        
        // Simple boundary checking (keep organisms within reasonable bounds)
        // In future, this will wrap or use proper world boundaries
        let max_pos = 200.0;
        let hit_boundary = position.0.x != old_position.x + velocity.0.x * dt || 
                          position.0.y != old_position.y + velocity.0.y * dt;
        position.0.x = position.0.x.clamp(-max_pos, max_pos);
        position.0.y = position.0.y.clamp(-max_pos, max_pos);
        
        if tracked.entity == Some(entity) && hit_boundary {
            info!("[TRACKED] Organism hit world boundary at ({:.2}, {:.2})", position.0.x, position.0.y);
        }
    }
    
    // Log direction change if it happened
    if direction_changed {
        if let Some(entity) = tracked.entity {
            if let Ok((_position, velocity, _energy, _size, _entity)) = query.get(entity) {
                info!("[TRACKED] Direction changed - New velocity: ({:.2}, {:.2}), Speed: {:.2}", 
                      velocity.0.x, velocity.0.y, velocity.0.length());
            }
        }
    }
}

/// Update organism age
pub fn update_age(
    mut query: Query<&mut Age>,
) {
    for mut age in query.iter_mut() {
        age.increment();
    }
}

/// Handle organism death (remove entities with zero energy)
pub fn handle_death(
    mut commands: Commands,
    mut tracked: ResMut<TrackedOrganism>,
    query: Query<(Entity, &Energy), With<Alive>>,
) {
    for (entity, energy) in query.iter() {
        if energy.is_dead() {
            if tracked.entity == Some(entity) {
                info!("[TRACKED] Organism died! Final energy: {:.2}", energy.current);
                tracked.entity = None; // Clear tracking
            }
            info!("Organism died at energy level: {:.2}", energy.current);
            commands.entity(entity).despawn();
        }
    }
}

/// Log tracked organism information periodically
pub fn log_tracked_organism(
    tracked: ResMut<TrackedOrganism>,
    query: Query<(Entity, &Position, &Velocity, &Energy, &Age, &Size, &OrganismType), With<Alive>>,
) {
    let mut tracked_mut = tracked;
    tracked_mut.log_counter += 1;
    
    // Log every 10 ticks for more frequent output (change to 60 for less frequent)
    if tracked_mut.log_counter % 10 != 0 {
        return;
    }
    
    if let Some(entity) = tracked_mut.entity {
        if let Ok((_entity, position, velocity, energy, age, size, org_type)) = query.get(entity) {
            let action = if velocity.0.length() < 0.1 {
                "Resting"
            } else if velocity.0.length() > 10.0 {
                "Moving Fast"
            } else {
                "Wandering"
            };
            
            info!(
                "[TRACKED ORGANISM] Tick: {} | Pos: ({:.2}, {:.2}) | Vel: ({:.2}, {:.2}) | Speed: {:.2} | Energy: {:.2}/{:.2} ({:.1}%) | Age: {} | Size: {:.2} | Type: {:?} | Action: {}",
                tracked_mut.log_counter,
                position.0.x,
                position.0.y,
                velocity.0.x,
                velocity.0.y,
                velocity.0.length(),
                energy.current,
                energy.max,
                energy.ratio() * 100.0,
                age.0,
                size.value(),
                org_type,
                action
            );
        } else {
            // Entity no longer exists (probably died)
            info!("[TRACKED] Organism entity {:?} no longer exists", entity);
            tracked_mut.entity = None;
        }
    }
}

