use crate::organisms::components::*;
use crate::world::{ResourceType, WorldGrid};
use bevy::prelude::*;
use glam::Vec2;

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
}

impl Default for Behavior {
    fn default() -> Self {
        Self {
            state: BehaviorState::Wandering,
            target_entity: None,
            target_position: None,
            state_time: 0.0,
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
}

impl SensoryData {
    pub fn new() -> Self {
        Self {
            nearby_organisms: Vec::new(),
            nearby_resources: Vec::new(),
            current_cell_resources: [0.0; 6],
        }
    }
}

/// Collect sensory information for an organism
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

    for other_entity in nearby_entities {
        if other_entity == entity {
            continue; // Skip self
        }

        if let Ok((_, other_pos, other_species, other_type, other_size, other_energy)) =
            organism_query.get(other_entity)
        {
            let distance = (position - other_pos.0).length();
            if distance <= sensory_range {
                let is_predator =
                    is_predator_of(organism_type, *other_type, other_size.value(), size);
                let is_prey = is_prey_of(organism_type, *other_type, size, other_size.value());
                let is_mate = *other_species == species_id
                    && *other_type == organism_type
                    && !other_energy.is_dead()
                    && distance <= sensory_range * 0.5; // Mates need to be closer

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

    // Find nearby resource-rich cells (simplified - check nearby cells in grid)
    let cell_size = 1.0; // Assume 1 unit per cell for now
    let search_radius = (sensory_range / cell_size).ceil() as i32;

    for dy in -search_radius..=search_radius {
        for dx in -search_radius..=search_radius {
            let check_x = position.x + (dx as f32 * cell_size);
            let check_y = position.y + (dy as f32 * cell_size);
            let distance = Vec2::new(dx as f32, dy as f32).length() * cell_size;

            if distance <= sensory_range {
                if let Some(cell) = world_grid.get_cell(check_x, check_y) {
                    // Check each resource type
                    for (_idx, resource_type) in [
                        ResourceType::Plant,
                        ResourceType::Water,
                        ResourceType::Detritus,
                        ResourceType::Prey,
                    ]
                    .iter()
                    .enumerate()
                    {
                        let value = cell.get_resource(*resource_type);
                        if value > 0.1 {
                            // Only consider cells with meaningful resources
                            sensory.nearby_resources.push((
                                Vec2::new(check_x, check_y),
                                *resource_type,
                                distance,
                                value,
                            ));
                        }
                    }
                }
            }
        }
    }

    // Sort resources by distance (closest first)
    sensory
        .nearby_resources
        .sort_by(|a, b| a.2.partial_cmp(&b.2).unwrap_or(std::cmp::Ordering::Equal));

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

/// Make behavior decision based on sensory data and organism state
/// Returns the new behavior state and optional target
pub fn decide_behavior(
    energy: &Energy,
    cached_traits: &crate::organisms::components::CachedTraits,
    organism_type: OrganismType,
    sensory: &SensoryData,
    current_state: BehaviorState,
    state_time: f32,
) -> (BehaviorState, Option<Entity>, Option<Vec2>) {
    // Priority system: Survival > Reproduction > Exploration

    // 1. HIGHEST PRIORITY: Flee from predators (survival)
    let aggression = cached_traits.aggression;
    let boldness = cached_traits.boldness;

    // Find nearest predator
    let nearest_predator = sensory
        .nearby_organisms
        .iter()
        .filter(|(_, _, _, is_pred, _, _)| *is_pred)
        .min_by(|a, b| a.2.partial_cmp(&b.2).unwrap_or(std::cmp::Ordering::Equal));

    if let Some((entity, pred_pos, distance, _, _, _)) = nearest_predator {
        // Flee if predator is close or if boldness is low
        let flee_threshold = 10.0 + boldness * 20.0; // Bolder organisms flee later
        if *distance < flee_threshold {
            // Calculate flee direction (away from predator)
            return (BehaviorState::Fleeing, Some(*entity), Some(*pred_pos));
        }
    }

    // 2. SECOND PRIORITY: Eat if energy is low (survival)
    let energy_threshold = 0.3; // Start seeking food below 30% energy
    if energy.ratio() < energy_threshold {
        // Find best food source
        if let Some(best_food) = find_best_food_source(organism_type, sensory) {
            if current_state == BehaviorState::Eating && state_time < 2.0 {
                // Continue eating if we just started
                return (BehaviorState::Eating, None, Some(best_food));
            }
            return (BehaviorState::Chasing, None, Some(best_food));
        }

        // Check if we're at a resource-rich cell
        if is_at_food_source(organism_type, sensory) {
            return (BehaviorState::Eating, None, None);
        }
    }

    // 3. THIRD PRIORITY: Chase prey (if aggressive and energy is decent)
    if organism_type == OrganismType::Consumer && energy.ratio() > 0.4 && aggression > 0.5 {
        if let Some((entity, prey_pos, distance, _, _is_prey, _)) = sensory
            .nearby_organisms
            .iter()
            .filter(|(_, _, _, _, is_prey, _)| *is_prey)
            .min_by(|a, b| a.2.partial_cmp(&b.2).unwrap_or(std::cmp::Ordering::Equal))
        {
            if *distance < 5.0 {
                // Very close - start eating
                return (BehaviorState::Eating, Some(*entity), Some(*prey_pos));
            } else if *distance < 30.0 {
                // Within chase range
                return (BehaviorState::Chasing, Some(*entity), Some(*prey_pos));
            }
        }
    }

    // 4. FOURTH PRIORITY: Mate if energy is high enough
    let reproduction_threshold = cached_traits.reproduction_threshold;
    if energy.ratio() >= reproduction_threshold {
        if let Some((entity, mate_pos, distance, _, _, _is_mate)) = sensory
            .nearby_organisms
            .iter()
            .filter(|(_, _, _, _, _, is_mate)| *is_mate)
            .min_by(|a, b| a.2.partial_cmp(&b.2).unwrap_or(std::cmp::Ordering::Equal))
        {
            if *distance < 15.0 {
                // Within mating range
                return (BehaviorState::Mating, Some(*entity), Some(*mate_pos));
            }
        }
    }

    // 5. FIFTH PRIORITY: Rest if energy is very low
    if energy.ratio() < 0.15 {
        return (BehaviorState::Resting, None, None);
    }

    // 6. DEFAULT: Wander
    (BehaviorState::Wandering, None, None)
}

/// Find the best food source for an organism type
fn find_best_food_source(organism_type: OrganismType, sensory: &SensoryData) -> Option<Vec2> {
    let preferred_resources = match organism_type {
        OrganismType::Producer => vec![
            ResourceType::Sunlight,
            ResourceType::Water,
            ResourceType::Mineral,
        ],
        OrganismType::Consumer => vec![ResourceType::Prey, ResourceType::Plant],
        OrganismType::Decomposer => vec![ResourceType::Detritus],
    };

    // Find closest resource of preferred type
    for resource_type in preferred_resources {
        if let Some((pos, _, _, value)) = sensory
            .nearby_resources
            .iter()
            .find(|(_, rt, _, _)| *rt == resource_type)
        {
            if *value > 0.2 {
                // Must have meaningful amount
                return Some(*pos);
            }
        }
    }

    None
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
            if let Some(flee_from) = behavior.target_position {
                // Move away from threat
                let direction = (position - flee_from).normalize_or_zero();
                direction * current_speed * 1.5 // Flee faster
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
        BehaviorState::Wandering => {
            // Random walk with occasional direction changes
            let angle =
                (time * 0.5 + (position.x + position.y) * 0.1).sin() * std::f32::consts::TAU;
            Vec2::from_angle(angle) * current_speed * 0.7
        }
    }
}
