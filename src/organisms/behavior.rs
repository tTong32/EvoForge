use crate::organisms::components::*;
use crate::world::{ResourceType, WorldGrid};
use bevy::prelude::*;
use glam::Vec2;
use std::collections::HashMap;

/// Behavior state machine - organisms can be in one of these states
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BehaviorState {
    /// Random wandering (default state)
    Wandering,
    /// Chasing prey or moving toward food
    Chasing,
    /// Consuming resources or prey
    Eating,
    /// Fleeing from predators
    Fleeing,
    /// Attempting to mate
    Mating,
    /// Resting (low energy, not moving much)
    Resting,
    /// Long-range movement toward richer territory
    Migrating,
}

/// Component tracking organism's current behavior state
#[derive(Component, Debug)]
pub struct Behavior {
    pub state: BehaviorState,
    /// Target entity (for chasing, fleeing, mating)
    pub target_entity: Option<Entity>,
    /// Target position (for chasing food, fleeing direction)
    pub target_position: Option<Vec2>,
    /// Time in current state (for state transitions)
    pub state_time: f32,
    /// Rolling memory of hunger pressure (0-1)
    pub hunger_memory: f32,
    /// Timer tracking recent threats (seconds remaining)
    pub threat_timer: f32,
    /// Location of the last perceived threat
    pub recent_threat: Option<Vec2>,
    /// Long-range migration target (if any)
    pub migration_target: Option<Vec2>,
}

impl Default for Behavior {
    fn default() -> Self {
        Self {
            state: BehaviorState::Wandering,
            target_entity: None,
            target_position: None,
            state_time: 0.0,
            hunger_memory: 0.0,
            threat_timer: 0.0,
            recent_threat: None,
            migration_target: None,
        }
    }
}

impl Behavior {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn set_state(&mut self, new_state: BehaviorState) {
        if self.state != new_state {
            self.state = new_state;
            self.state_time = 0.0;
            // Clear targets when changing states
            self.target_entity = None;
            self.target_position = None;
            if !matches!(self.state, BehaviorState::Migrating) {
                self.migration_target = None;
            }
        }
    }
}

/// Sensory information about nearby entities
#[derive(Debug, Clone)]
pub struct SensoryData {
    /// Nearby organisms (entity, position, distance, is_predator, is_prey, is_mate)
    pub nearby_organisms: Vec<(Entity, Vec2, f32, bool, bool, bool)>,
    /// Nearby resources (position, resource_type, distance, value)
    pub nearby_resources: Vec<(Vec2, ResourceType, f32, f32)>,
    /// Current cell resource values
    pub current_cell_resources: [f32; 6],
    /// Closest predator information
    pub nearest_predator: Option<(Entity, Vec2, f32)>,
    /// Highest value resource in range
    pub richest_resource: Option<(Vec2, ResourceType, f32, f32)>,
}

impl SensoryData {
    pub fn new() -> Self {
        Self {
            nearby_organisms: Vec::new(),
            nearby_resources: Vec::new(),
            current_cell_resources: [0.0; 6],
            nearest_predator: None,
            richest_resource: None,
        }
    }
}

/// Cache sensory data for organisms that haven't moved much (optimization 3)
#[derive(Resource, Default)]
pub struct SensoryDataCache {
    cache: HashMap<Entity, (Vec2, SensoryData, u32)>, // (position, data, age_in_frames)
    max_cache_age: u32,
}

impl SensoryDataCache {
    pub fn new(max_age: u32) -> Self {
        Self {
            cache: HashMap::new(),
            max_cache_age: max_age,
        }
    }
    
    pub fn get_or_compute<F>(&mut self, 
        entity: Entity, 
        position: Vec2, 
        sensory_range: f32,
        compute_fn: F) -> SensoryData
    where
        F: FnOnce() -> SensoryData,
    {
        let cache_threshold = sensory_range * 0.3; // Cache if moved less than 30% of sensory range
        
        if let Some((cached_pos, cached_data, age)) = self.cache.get_mut(&entity) {
            *age += 1;
            
            // Use cache if position hasn't changed much and cache isn't too old
            if (position - *cached_pos).length_squared() < cache_threshold * cache_threshold && *age < self.max_cache_age {
                return cached_data.clone();
            }
        }
        
        // Compute new sensory data
        let data = compute_fn();
        self.cache.insert(entity, (position, data.clone(), 0));
        data
    }
    
    pub fn invalidate(&mut self, entity: Entity) {
        self.cache.remove(&entity);
    }
    
    pub fn cleanup(&mut self) {
        // Remove old entries periodically
        self.cache.retain(|_, (_, _, age)| *age < self.max_cache_age);
    }
}

/// Collect sensory information for an organism (OPTIMIZED - optimization 3)
pub fn collect_sensory_data(
    entity: Entity,
    position: Vec2,
    sensory_range: f32,
    species_id: SpeciesId,
    organism_type: OrganismType,
    size: f32,
    world_grid: &WorldGrid,
    spatial_hash: &crate::utils::SpatialHash,
    organism_query: &Query<
        (Entity, &Position, &SpeciesId, &OrganismType, &Size, &Energy),
        With<Alive>,
    >,
) -> SensoryData {
    let mut sensory = SensoryData::new();

    // Get current cell resources
    if let Some(cell) = world_grid.get_cell(position.x, position.y) {
        sensory.current_cell_resources = cell.resource_density;
    }

    // Query nearby organisms using spatial hash (much faster than iterating all)
    let nearby_entities = spatial_hash.query_radius(position, sensory_range);
    let sensory_range_sq = sensory_range * sensory_range; // Use squared distance to avoid sqrt

    for other_entity in nearby_entities {
        if other_entity == entity {
            continue; // Skip self
        }

        if let Ok((_, other_pos, other_species, other_type, other_size, other_energy)) =
            organism_query.get(other_entity)
        {
            // Use squared distance to avoid sqrt
            let distance_sq = (position - other_pos.0).length_squared();
            if distance_sq <= sensory_range_sq {
                let distance = distance_sq.sqrt(); // Only compute sqrt when needed
                let is_predator =
                    is_predator_of(organism_type, *other_type, other_size.value(), size);
                let is_prey = is_prey_of(organism_type, *other_type, size, other_size.value());
                let is_mate = *other_species == species_id
                    && *other_type == organism_type
                    && !other_energy.is_dead()
                    && distance_sq <= (sensory_range * 0.5).powi(2); // Use squared for mate check

                if is_predator {
                    match &mut sensory.nearest_predator {
                        Some((_, _, current_distance)) if *current_distance <= distance => {}
                        _ => sensory.nearest_predator = Some((other_entity, other_pos.0, distance)),
                    }
                }

                sensory.nearby_organisms.push((
                    other_entity,
                    other_pos.0,
                    distance,
                    is_predator,
                    is_prey,
                    is_mate,
                ));
            }
        }
    }

    // OPTIMIZED: Find nearby resource-rich cells with early termination (optimization 3)
    let cell_size = 1.0;
    let search_radius = (sensory_range / cell_size).ceil() as i32;
    let sensory_range_sq = sensory_range * sensory_range;
    
    // Pre-compute bounds to avoid redundant checks
    let min_x = (position.x - sensory_range) as i32;
    let max_x = (position.x + sensory_range) as i32;
    let min_y = (position.y - sensory_range) as i32;
    let max_y = (position.y + sensory_range) as i32;
    
    let mut best_resource_value = 0.0f32;
    const MAX_RESOURCES_TO_CHECK: usize = 20; // Early termination limit
    let mut resources_found = 0;

    // Only check cells within sensory range bounds
    for dy in -search_radius..=search_radius {
        for dx in -search_radius..=search_radius {
            let check_x = position.x + (dx as f32 * cell_size);
            let check_y = position.y + (dy as f32 * cell_size);
            
            // Bounds check before distance calculation
            if check_x < min_x as f32 || check_x > max_x as f32 
                || check_y < min_y as f32 || check_y > max_y as f32 {
                continue;
            }
            
            let distance_sq = (dx as f32 * dx as f32 + dy as f32 * dy as f32) * (cell_size * cell_size);
            if distance_sq > sensory_range_sq {
                continue;
            }

            if let Some(cell) = world_grid.get_cell(check_x, check_y) {
                // Early termination if we've found enough resources
                if resources_found >= MAX_RESOURCES_TO_CHECK && best_resource_value > 0.5 {
                    break;
                }
                
                // Only check relevant resource types for this organism
                let resource_types: Vec<ResourceType> = match organism_type {
                    OrganismType::Producer => vec![
                        ResourceType::Sunlight,
                        ResourceType::Water,
                        ResourceType::Mineral,
                    ],
                    OrganismType::Consumer => vec![
                        ResourceType::Plant,
                        ResourceType::Prey,
                        ResourceType::Water, // Consumers also need water
                    ],
                    OrganismType::Decomposer => vec![
                        ResourceType::Detritus,
                    ],
                };

                for resource_type in resource_types.iter() {
                    let value = cell.get_resource(*resource_type);
                    if value > 0.1 {
                        let distance = distance_sq.sqrt();
                        let entry = (Vec2::new(check_x, check_y), *resource_type, distance, value);
                        
                        if value > best_resource_value {
                            best_resource_value = value;
                            sensory.richest_resource = Some(entry.clone());
                        }

                        sensory.nearby_resources.push(entry);
                        resources_found += 1;
                    }
                }
            }
        }
        
        // Early termination for outer loop
        if resources_found >= MAX_RESOURCES_TO_CHECK && best_resource_value > 0.5 {
            break;
        }
    }

    // Only sort if we have resources (avoid unnecessary work)
    if !sensory.nearby_resources.is_empty() {
        sensory
            .nearby_resources
            .sort_by(|a, b| a.2.partial_cmp(&b.2).unwrap_or(std::cmp::Ordering::Equal));
    }

    sensory
}

/// Determine if one organism is a predator of another
fn is_predator_of(
    predator_type: OrganismType,
    prey_type: OrganismType,
    predator_size: f32,
    prey_size: f32,
) -> bool {
    match (predator_type, prey_type) {
        (OrganismType::Consumer, OrganismType::Consumer) => {
            // Larger consumers can be predators of smaller ones
            predator_size > prey_size * 1.5
        }
        (OrganismType::Consumer, OrganismType::Producer) => {
            // Consumers can eat producers
            true
        }
        (OrganismType::Consumer, OrganismType::Decomposer) => {
            // Consumers can eat decomposers
            true
        }
        _ => false,
    }
}

/// Determine if one organism is prey for another
fn is_prey_of(
    predator_type: OrganismType,
    prey_type: OrganismType,
    predator_size: f32,
    prey_size: f32,
) -> bool {
    is_predator_of(predator_type, prey_type, predator_size, prey_size)
}

pub struct BehaviorDecision {
    pub state: BehaviorState,
    pub target_entity: Option<Entity>,
    pub target_position: Option<Vec2>,
    pub migration_target: Option<Vec2>,
}

pub fn decide_behavior_with_memory(
    energy: &Energy,
    cached_traits: &crate::organisms::components::CachedTraits,
    organism_type: OrganismType,
    sensory: &SensoryData,
    current_state: BehaviorState,
    state_time: f32,
    hunger_memory: f32,
    threat_timer: f32,
    recent_threat: Option<Vec2>,
    has_migration_target: bool,
) -> BehaviorDecision {
    // Step 8: Improved behavior differentiation between organism types
    // Priority system: Survival > Reproduction > Exploration
    
    let aggression = cached_traits.aggression;
    let boldness = cached_traits.boldness;
    let risk_tolerance = cached_traits.risk_tolerance;

    // PRODUCERS: Stationary, focus on growth, minimal movement
    if organism_type == OrganismType::Producer {
        // Producers don't flee (they're stationary)
        // They focus on eating (photosynthesis) and staying in place
        
        let hunger_pressure = ((1.0 - energy.ratio()).max(0.0) * 0.8) + (hunger_memory * 0.2);
        let hunger_barrier = 0.4; // Producers are less sensitive to hunger
        
        if hunger_pressure > hunger_barrier {
            if is_at_food_source(organism_type, sensory) {
                return BehaviorDecision {
                    state: BehaviorState::Eating,
                    target_entity: None,
                    target_position: None,
                    migration_target: None,
                };
            }
            // Producers can slowly move toward better resource areas
            if let Some(best_food) = find_best_food_source_weighted(
                organism_type,
                sensory,
                cached_traits.resource_selectivity,
            ) {
                if matches!(current_state, BehaviorState::Eating) && state_time < 5.0 {
                    return BehaviorDecision {
                        state: BehaviorState::Eating,
                        target_entity: None,
                        target_position: Some(best_food),
                        migration_target: None,
                    };
                }
                // Only move if resources are very low
                if energy.ratio() < 0.3 {
                    return BehaviorDecision {
                        state: BehaviorState::Chasing,
                        target_entity: None,
                        target_position: Some(best_food),
                        migration_target: None,
                    };
                }
            }
        }
        
        // Producers rest when low energy (conserving resources)
        if energy.ratio() < 0.2 {
            return BehaviorDecision {
                state: BehaviorState::Resting,
                target_entity: None,
                target_position: None,
                migration_target: None,
            };
        }
        
        // Producers mostly stay in place (wandering is minimal)
        return BehaviorDecision {
            state: BehaviorState::Wandering,
            target_entity: None,
            target_position: None,
            migration_target: None,
        };
    }
    
    // DECOMPOSERS: Slow movement, stay near detritus, less aggressive
    if organism_type == OrganismType::Decomposer {
        // Decomposers don't flee (they're small and not typically prey)
        // They focus on finding detritus and staying near it
        
        let hunger_pressure = ((1.0 - energy.ratio()).max(0.0) * 0.6) + (hunger_memory * 0.4);
        let hunger_barrier = 0.35; // Decomposers are moderately sensitive
        
        if hunger_pressure > hunger_barrier {
            if is_at_food_source(organism_type, sensory) {
                return BehaviorDecision {
                    state: BehaviorState::Eating,
                    target_entity: None,
                    target_position: None,
                    migration_target: None,
                };
            }
            
            // Decomposers slowly move toward detritus
            if let Some(best_food) = find_best_food_source_weighted(
                organism_type,
                sensory,
                cached_traits.resource_selectivity,
            ) {
                if matches!(current_state, BehaviorState::Eating) && state_time < 3.0 {
                    return BehaviorDecision {
                        state: BehaviorState::Eating,
                        target_entity: None,
                        target_position: Some(best_food),
                        migration_target: None,
                    };
                }
                return BehaviorDecision {
                    state: BehaviorState::Chasing,
                    target_entity: None,
                    target_position: Some(best_food),
                    migration_target: None,
                };
            }
        }
        
        // Decomposers rest when low energy
        if energy.ratio() < 0.2 {
            return BehaviorDecision {
                state: BehaviorState::Resting,
                target_entity: None,
                target_position: None,
                migration_target: None,
            };
        }
        
        // Decomposers wander slowly looking for detritus
        return BehaviorDecision {
            state: BehaviorState::Wandering,
            target_entity: None,
            target_position: None,
            migration_target: None,
        };
    }
    
    // CONSUMERS: Active hunting, more movement, aggressive behaviors
    // (Original behavior logic for consumers)
    if let Some((entity, pred_pos, distance)) = sensory.nearest_predator {
        let flee_threshold = 8.0 + (boldness * 14.0) + (risk_tolerance * 6.0);
        let memory_bonus = if threat_timer > 0.0 { 5.0 } else { 0.0 };
        if distance < flee_threshold + memory_bonus {
            return BehaviorDecision {
                state: BehaviorState::Fleeing,
                target_entity: Some(entity),
                target_position: Some(pred_pos),
                migration_target: None,
            };
        }
    } else if threat_timer > 0.0 {
        // Keep fleeing briefly even when predator left
        if let Some(threat_pos) = recent_threat {
            return BehaviorDecision {
                state: BehaviorState::Fleeing,
                target_entity: None,
                target_position: Some(threat_pos),
                migration_target: None,
            };
        }
    }

    let hunger_pressure = ((1.0 - energy.ratio()).max(0.0) * 0.7) + (hunger_memory * 0.3);
    let hunger_barrier = (0.3 - cached_traits.foraging_drive * 0.15).clamp(0.1, 0.5);

    if hunger_pressure > hunger_barrier {
        // Consumers actively hunt prey
        if energy.ratio() > 0.4 && aggression > 0.4 {
            if let Some((entity, prey_pos, distance, _, _is_prey, _)) = sensory
                .nearby_organisms
                .iter()
                .filter(|(_, _, _, _, is_prey, _)| *is_prey)
                .min_by(|a, b| a.2.partial_cmp(&b.2).unwrap_or(std::cmp::Ordering::Equal))
            {
                if *distance < 5.0 {
                    return BehaviorDecision {
                        state: BehaviorState::Eating,
                        target_entity: Some(*entity),
                        target_position: Some(*prey_pos),
                        migration_target: None,
                    };
                } else if *distance < 30.0 {
                    return BehaviorDecision {
                        state: BehaviorState::Chasing,
                        target_entity: Some(*entity),
                        target_position: Some(*prey_pos),
                        migration_target: None,
                    };
                }
            }
        }
        
        // Also eat plant resources
        if let Some(best_food) = find_best_food_source_weighted(
            organism_type,
            sensory,
            cached_traits.resource_selectivity,
        ) {
            if matches!(current_state, BehaviorState::Eating) && state_time < 2.0 {
                return BehaviorDecision {
                    state: BehaviorState::Eating,
                    target_entity: None,
                    target_position: Some(best_food),
                    migration_target: None,
                };
            }
            return BehaviorDecision {
                state: BehaviorState::Chasing,
                target_entity: None,
                target_position: Some(best_food),
                migration_target: None,
            };
        }

        if is_at_food_source(organism_type, sensory) {
            return BehaviorDecision {
                state: BehaviorState::Eating,
                target_entity: None,
                target_position: None,
                migration_target: None,
            };
        }
    }

    let reproduction_threshold = cached_traits.reproduction_threshold;
    if energy.ratio() >= reproduction_threshold {
        if let Some((entity, mate_pos, distance, _, _, _is_mate)) = sensory
            .nearby_organisms
            .iter()
            .filter(|(_, _, _, _, _, is_mate)| *is_mate)
            .min_by(|a, b| a.2.partial_cmp(&b.2).unwrap_or(std::cmp::Ordering::Equal))
        {
            if *distance < 15.0 {
                return BehaviorDecision {
                    state: BehaviorState::Mating,
                    target_entity: Some(*entity),
                    target_position: Some(*mate_pos),
                    migration_target: None,
                };
            }
        }
    }

    if energy.ratio() < 0.15 {
        return BehaviorDecision {
            state: BehaviorState::Resting,
            target_entity: None,
            target_position: None,
            migration_target: None,
        };
    }

    if !has_migration_target
        && cached_traits.exploration_drive > 0.4
        && sensory.nearby_resources.is_empty()
    {
        if let Some((target_pos, _, _, _)) = sensory.richest_resource {
            return BehaviorDecision {
                state: BehaviorState::Migrating,
                target_entity: None,
                target_position: None,
                migration_target: Some(target_pos),
            };
        }
    }

    BehaviorDecision {
        state: BehaviorState::Wandering,
        target_entity: None,
        target_position: None,
        migration_target: None,
    }
}

pub fn decide_behavior(
    energy: &Energy,
    cached_traits: &crate::organisms::components::CachedTraits,
    organism_type: OrganismType,
    sensory: &SensoryData,
    current_state: BehaviorState,
    state_time: f32,
) -> (BehaviorState, Option<Entity>, Option<Vec2>) {
    let decision = decide_behavior_with_memory(
        energy,
        cached_traits,
        organism_type,
        sensory,
        current_state,
        state_time,
        0.0,
        0.0,
        None,
        false,
    );
    (
        decision.state,
        decision.target_entity,
        decision.target_position,
    )
}

/// Find the best food source for an organism type
fn find_best_food_source(organism_type: OrganismType, sensory: &SensoryData) -> Option<Vec2> {
    find_best_food_source_weighted(organism_type, sensory, 0.0)
}

fn find_best_food_source_weighted(
    organism_type: OrganismType,
    sensory: &SensoryData,
    selectivity: f32,
) -> Option<Vec2> {
    let preferred_resources = match organism_type {
        OrganismType::Producer => vec![
            ResourceType::Sunlight,
            ResourceType::Water,
            ResourceType::Mineral,
        ],
        OrganismType::Consumer => vec![ResourceType::Prey, ResourceType::Plant],
        OrganismType::Decomposer => vec![ResourceType::Detritus],
    };

    let mut best: Option<(Vec2, f32)> = None;
    for (pos, resource_type, distance, value) in &sensory.nearby_resources {
        if !preferred_resources.contains(resource_type) {
            continue;
        }

        if *value <= 0.2 {
            continue;
        }

        let score = value * (1.0 + selectivity) - distance * (0.1 + (1.0 - selectivity) * 0.05);
        match &best {
            Some((_, best_score)) if score <= *best_score => {}
            _ => best = Some((*pos, score)),
        }
    }

    best.map(|(pos, _)| pos)
}

/// Check if organism is at a food source
fn is_at_food_source(organism_type: OrganismType, sensory: &SensoryData) -> bool {
    let preferred_resources = match organism_type {
        OrganismType::Producer => vec![ResourceType::Sunlight, ResourceType::Water],
        OrganismType::Consumer => vec![ResourceType::Plant, ResourceType::Prey],
        OrganismType::Decomposer => vec![ResourceType::Detritus],
    };

    for resource_type in preferred_resources {
        let idx = resource_type as usize;
        if sensory.current_cell_resources[idx] > 0.2 {
            return true;
        }
    }

    false
}

/// Calculate velocity for a behavior state
pub fn calculate_behavior_velocity(
    behavior: &Behavior,
    position: Vec2,
    cached_traits: &crate::organisms::components::CachedTraits,
    _organism_type: OrganismType,
    energy: &Energy,
    time: f32,
) -> Vec2 {
    let max_speed = cached_traits.speed;
    let speed_factor = energy.ratio().max(0.3); // Minimum 30% speed even when low energy
    let current_speed = max_speed * speed_factor;

    match behavior.state {
        BehaviorState::Fleeing => {
            let source = behavior.target_position.or(behavior.recent_threat);
            if let Some(flee_from) = source {
                // Move away from threat
                let direction = (position - flee_from).normalize_or_zero();
                direction * current_speed // Flee at max speed
            } else {
                // Random direction if no target
                let angle = (time * 2.0).sin() * std::f32::consts::PI;
                Vec2::from_angle(angle) * current_speed
            }
        }
        BehaviorState::Chasing => {
            if let Some(target) = behavior.target_position {
                // Move toward target
                let direction = (target - position).normalize_or_zero();
                direction * current_speed
            } else {
                Vec2::ZERO
            }
        }
        BehaviorState::Eating => {
            // Slow down or stop while eating
            Vec2::ZERO
        }
        BehaviorState::Mating => {
            if let Some(target) = behavior.target_position {
                // Move toward mate slowly
                let direction = (target - position).normalize_or_zero();
                direction * current_speed * 0.5
            } else {
                Vec2::ZERO
            }
        }
        BehaviorState::Resting => {
            // Minimal movement
            Vec2::ZERO
        }
        BehaviorState::Migrating => {
            if let Some(target) = behavior.migration_target.or(behavior.target_position) {
                let direction = (target - position).normalize_or_zero();
                direction * current_speed * 0.8
            } else {
                let angle = (time * 0.4 + (position.x * 0.3) + (position.y * 0.17)).cos()
                    * std::f32::consts::TAU;
                Vec2::from_angle(angle) * current_speed * 0.5
            }
        }
        BehaviorState::Wandering => {
            // Step 8: Different wandering speeds based on organism type
            let wander_speed_mult = match _organism_type {
                OrganismType::Producer => 0.1, // Producers barely move
                OrganismType::Decomposer => 0.4, // Decomposers move slowly
                OrganismType::Consumer => 0.7, // Consumers move more actively
            };
            // Random walk with occasional direction changes
            let angle =
                (time * 0.5 + (position.x + position.y) * 0.1).sin() * std::f32::consts::TAU;
            Vec2::from_angle(angle) * current_speed * wander_speed_mult
        }
    }
}
