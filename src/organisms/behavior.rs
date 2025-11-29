use crate::organisms::components::*;
use crate::world::{ResourceType, WorldGrid};
use bevy::prelude::*;
use glam::Vec2;
use std::collections::HashMap;
use std::sync::Arc;
use smallvec::{SmallVec, smallvec};

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

/// Information about a nearby organism detected by senses.
#[derive(Debug, Clone)]
pub struct NearbyOrganism {
    pub entity: Entity,
    pub position: Vec2,
    pub distance: f32,
    pub species_id: SpeciesId,
    pub organism_type: OrganismType,
    pub is_predator: bool,
    pub is_prey: bool,
    pub is_mate: bool,
    pub size: f32,
    pub speed: f32,
    pub armor: f32,
    pub poison_strength: f32,
}

/// Sensory information about nearby entities and resources.
#[derive(Debug, Clone)]
pub struct SensoryData {
    /// Nearby organisms detected this frame.
    pub nearby_organisms: Vec<NearbyOrganism>,
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
            nearby_organisms: Vec::with_capacity(16), // Pre-allocate for typical neighbor count
            nearby_resources: Vec::with_capacity(20), // Pre-allocate for MAX_RESOURCES_TO_CHECK
            current_cell_resources: [0.0; 6],
            nearest_predator: None,
            richest_resource: None,
        }
    }
}

/// Cache sensory data for organisms that haven't moved much (optimization 3)
/// Uses Arc<SensoryData> to avoid expensive cloning - Arc cloning is just a reference count increment
#[derive(Resource)]
pub struct SensoryDataCache {
    cache: HashMap<Entity, (Vec2, Arc<SensoryData>, u32)>, // (position, data, age_in_frames)
    max_cache_age: u32,
    cleanup_counter: u32,
    cleanup_interval: u32,
}

impl Default for SensoryDataCache {
    fn default() -> Self {
        Self {
            cache: HashMap::new(),
            max_cache_age: 15, // Increased from 10 for better cache hit rate
            cleanup_counter: 0,
            cleanup_interval: 60, // Cleanup every 60 frames
        }
    }
}

impl SensoryDataCache {
    pub fn new(max_age: u32) -> Self {
        Self {
            cache: HashMap::new(),
            max_cache_age: max_age,
            cleanup_counter: 0,
            cleanup_interval: 60,
        }
    }
    
    pub fn get_or_compute<F>(&mut self, 
        entity: Entity, 
        position: Vec2, 
        sensory_range: f32,
        compute_fn: F) -> Arc<SensoryData>
    where
        F: FnOnce() -> SensoryData,
    {
        // Adaptive cache threshold: larger sensory range = larger movement tolerance
        // Cache if moved less than 40% of sensory range (increased from 30%)
        let cache_threshold_sq = (sensory_range * 0.4) * (sensory_range * 0.4);
        
        if let Some((cached_pos, cached_data, age)) = self.cache.get_mut(&entity) {
            *age += 1;
            
            // Use cache if position hasn't changed much and cache isn't too old
            // Use squared distance to avoid sqrt
            if (position - *cached_pos).length_squared() < cache_threshold_sq && *age < self.max_cache_age {
                // Clone Arc (cheap - just increments reference count)
                return Arc::clone(cached_data);
            }
        }
        
        // Compute new sensory data
        let data = Arc::new(compute_fn());
        self.cache.insert(entity, (position, Arc::clone(&data), 0));
        data
    }
    
    pub fn invalidate(&mut self, entity: Entity) {
        self.cache.remove(&entity);
    }
    
    pub fn cleanup(&mut self) {
        // Remove old entries periodically
        self.cache.retain(|_, (_, _, age)| *age < self.max_cache_age);
    }
    
    /// Periodic cleanup - call this from update system
    pub fn maybe_cleanup(&mut self) {
        self.cleanup_counter += 1;
        if self.cleanup_counter >= self.cleanup_interval {
            self.cleanup();
            self.cleanup_counter = 0;
        }
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
    query_buffer: &mut Vec<Entity>, // Reusable buffer to avoid allocations
    organism_query: &Query<
        (
            Entity,
            &Position,
            &SpeciesId,
            &OrganismType,
            &Size,
            &Energy,
            &crate::organisms::components::CachedTraits,
        ),
        With<Alive>,
    >,
) -> SensoryData {
    let mut sensory = SensoryData::new();

    // Get current cell resources
    if let Some(cell) = world_grid.get_cell(position.x, position.y) {
        sensory.current_cell_resources = cell.resource_density;
    }

    // Query nearby organisms using spatial hash with reusable buffer (avoids allocation)
    spatial_hash.query_radius_into(position, sensory_range, query_buffer);
    let nearby_entities = query_buffer;
    let sensory_range_sq = sensory_range * sensory_range; // Use squared distance to avoid sqrt
    
    // Pre-allocate based on expected neighbor count (most organisms have < 20 neighbors)
    let expected_neighbors = nearby_entities.len().min(32);
    sensory.nearby_organisms.reserve(expected_neighbors);

    for other_entity in nearby_entities.iter().copied() {
        if other_entity == entity {
            continue; // Skip self
        }

        if let Ok((_, other_pos, other_species, other_type, other_size, other_energy, other_traits)) =
            organism_query.get(other_entity)
        {
            // Use squared distance to avoid sqrt
            let distance_sq = (position - other_pos.0).length_squared();
            if distance_sq <= sensory_range_sq {
                let is_predator =
                    is_predator_of(organism_type, *other_type, other_size.value(), size);
                let is_prey = is_prey_of(organism_type, *other_type, size, other_size.value());
                let is_mate = *other_species == species_id
                    && *other_type == organism_type
                    && !other_energy.is_dead()
                    && distance_sq <= (sensory_range * 0.5).powi(2);

                // Only compute sqrt and add to nearby_organisms if it's relevant (predator, prey, or mate)
                if is_predator || is_prey || is_mate {
                    let distance = distance_sq.sqrt(); // Only compute sqrt when needed
                    
                    if is_predator {
                        match &mut sensory.nearest_predator {
                            Some((_, _, current_distance)) if *current_distance <= distance => {}
                            _ => sensory.nearest_predator = Some((other_entity, other_pos.0, distance)),
                        }
                    }

                    sensory.nearby_organisms.push(NearbyOrganism {
                        entity: other_entity,
                        position: other_pos.0,
                        distance,
                        species_id: *other_species,
                        organism_type: *other_type,
                        is_predator,
                        is_prey,
                        is_mate,
                        size: other_size.value(),
                        speed: other_traits.speed,
                        armor: other_traits.armor,
                        poison_strength: other_traits.poison_strength,
                    });
                } else if is_predator {
                    // Handle predator case even if not adding to nearby_organisms
                    let distance = distance_sq.sqrt();
                    match &mut sensory.nearest_predator {
                        Some((_, _, current_distance)) if *current_distance <= distance => {}
                        _ => sensory.nearest_predator = Some((other_entity, other_pos.0, distance)),
                    }
                }
            }
        }
    }

    // OPTIMIZED: Find nearby resource-rich cells with early termination (optimization 3)
    let cell_size = 1.0;
    let search_radius = (sensory_range / cell_size).ceil() as i32;
    // Reuse sensory_range_sq from above (already computed on line 234)
    
    // Pre-compute bounds to avoid redundant checks
    let min_x = (position.x - sensory_range) as i32;
    let max_x = (position.x + sensory_range) as i32;
    let min_y = (position.y - sensory_range) as i32;
    let max_y = (position.y + sensory_range) as i32;
    
    let mut best_resource_value = 0.0f32;
    const MAX_RESOURCES_TO_CHECK: usize = 20; // Early termination limit
    let mut resources_found = 0;
    
    // Collect all candidate resources with equal values to break ties randomly
    // This prevents directional bias from iteration order
    let mut best_resource_candidates: SmallVec<[(Vec2, ResourceType, f32, f32); 8]> = SmallVec::new();

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
                
                // Only check relevant resource types for this organism (use SmallVec to avoid allocation)
                let resource_types: SmallVec<[ResourceType; 3]> = match organism_type {
                    OrganismType::Consumer => smallvec![
                        ResourceType::Plant,
                        ResourceType::Prey,
                        ResourceType::Water, // Consumers also need water
                    ],
                };

                for resource_type in resource_types.iter() {
                    let value = cell.get_resource(*resource_type);
                    if value > 0.1 {
                        let distance = distance_sq.sqrt();
                        
                        // Collect all resources with the best value (or better) to break ties randomly
                        if value > best_resource_value {
                            // New best value - clear old candidates
                            best_resource_value = value;
                            best_resource_candidates.clear();
                            best_resource_candidates.push((
                                Vec2::new(check_x, check_y),
                                *resource_type,
                                distance,
                                value,
                            ));
                        } else if (value - best_resource_value).abs() < 0.001 {
                            // Equal value - add to candidates for random selection
                            best_resource_candidates.push((
                                Vec2::new(check_x, check_y),
                                *resource_type,
                                distance,
                                value,
                            ));
                        }

                        sensory.nearby_resources.push((
                            Vec2::new(check_x, check_y),
                            *resource_type,
                            distance,
                            value,
                        ));
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
    
    // Select best resource from candidates using entity-based hash for deterministic but varied selection
    if !best_resource_candidates.is_empty() {
        // Use entity ID and position to create a pseudo-random but deterministic selection
        // This ensures each organism picks differently but consistently
        let hash = ((entity.index() as u64)
            .wrapping_mul(2654435761)
            .wrapping_add(position.x.to_bits() as u64)
            .wrapping_add(position.y.to_bits() as u64)) as usize;
        let selected_idx = hash % best_resource_candidates.len();
        let selected = best_resource_candidates[selected_idx];
        sensory.richest_resource = Some(selected);
    }

    // Optimization: Only sort if resources will be used for decision-making
    // Many organisms don't need sorted resources, so we defer sorting
    // Sorting will happen lazily when find_best_food_source_weighted is called if needed

    sensory
}

/// Determine if one organism is a predator of another
#[inline]
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
        _ => false,
    }
}

/// Determine if one organism is prey for another
#[inline]
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
    species_id: SpeciesId,
    organism_type: OrganismType,
    sensory: &SensoryData,
    current_state: BehaviorState,
    state_time: f32,
    hunger_memory: f32,
    threat_timer: f32,
    recent_threat: Option<Vec2>,
    has_migration_target: bool,
    learning: Option<&crate::organisms::components::IndividualLearning>,
    ) -> BehaviorDecision {
    // Step 8: Improved behavior differentiation between organism types
    // Priority system: Survival > Reproduction > Exploration
    
    let aggression = cached_traits.aggression;
    let boldness = cached_traits.boldness;
    let risk_tolerance = cached_traits.risk_tolerance;

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
            // Estimate local pack size (self + nearby same-species consumers).
            let pack_size_estimate = 1.0
                + sensory
                    .nearby_organisms
                    .iter()
                    .filter(|o| o.organism_type == OrganismType::Consumer && o.species_id == species_id)
                    .count() as f32;

            if let Some(best_prey) = select_best_prey(
                sensory,
                cached_traits,
                learning,
                cached_traits.size,
                hunger_pressure,
                pack_size_estimate,
            ) {
                if best_prey.distance < 5.0 {
                    return BehaviorDecision {
                        state: BehaviorState::Eating,
                        target_entity: Some(best_prey.entity),
                        target_position: Some(best_prey.position),
                        migration_target: None,
                    };
                } else if best_prey.distance < 30.0 {
                    return BehaviorDecision {
                        state: BehaviorState::Chasing,
                        target_entity: Some(best_prey.entity),
                        target_position: Some(best_prey.position),
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
        if let Some(best_mate) = sensory
            .nearby_organisms
            .iter()
            .filter(|o| o.is_mate)
            .min_by(|a, b| a.distance.partial_cmp(&b.distance).unwrap_or(std::cmp::Ordering::Equal))
        {
            if best_mate.distance < 15.0 {
                return BehaviorDecision {
                    state: BehaviorState::Mating,
                    target_entity: Some(best_mate.entity),
                    target_position: Some(best_mate.position),
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

/// Select the best prey target based on physical factors and learned knowledge.
fn select_best_prey(
    sensory: &SensoryData,
    cached_traits: &crate::organisms::components::CachedTraits,
    learning: Option<&crate::organisms::components::IndividualLearning>,
    predator_size: f32,
    hunger_pressure: f32,
    pack_size_estimate: f32,
) -> Option<NearbyOrganism> {
    let mut best: Option<(NearbyOrganism, f32)> = None;

    let base_speed = cached_traits.speed;
    let desperation = hunger_pressure.clamp(0.0, 1.5); // higher when very hungry

    for candidate in sensory.nearby_organisms.iter().filter(|o| o.is_prey) {
        // --- Size factor ---
        let size_ratio = (predator_size / (candidate.size.max(0.1))).clamp(0.1, 4.0);
        let size_score = if size_ratio >= 1.0 {
            0.6 + (size_ratio - 1.0).min(1.5) * 0.3
        } else {
            (0.25 * size_ratio) * (0.5 + desperation)
        };

        // --- Speed factor ---
        let prey_speed = candidate.speed.max(0.1);
        let relative_speed = (base_speed / prey_speed).clamp(0.2, 3.0);
        let speed_score = if relative_speed >= 1.0 {
            0.5 + (relative_speed - 1.0) * 0.3
        } else {
            (0.3 * relative_speed) * (0.5 + desperation * 0.5)
        };

        // --- Defense factor (armor + poison) ---
        let defense_score = candidate.armor * 0.7 + candidate.poison_strength * 0.8;
        let defense_penalty = defense_score * (1.0 - cached_traits.risk_tolerance).max(0.2);
        let defense_factor = (1.0 - defense_penalty).clamp(0.2, 1.0);

        // --- Pack factor ---
        let pack_factor = if pack_size_estimate > 1.0 {
            let pack_bonus =
                (pack_size_estimate.sqrt() - 1.0).max(0.0) * cached_traits.coordination;
            (1.0 + pack_bonus).clamp(1.0, 2.0)
        } else {
            1.0
        };

        // --- Strategy factor ---
        let strategy_factor = match cached_traits.hunting_strategy {
            crate::organisms::components::HuntingStrategy::Ambush => {
                // Ambush: likes close, slower prey.
                let dist_pref = (20.0 / (candidate.distance + 5.0)).clamp(0.5, 2.0);
                let speed_pref = (prey_speed / (base_speed + 0.1)).clamp(0.3, 1.2);
                dist_pref * (1.3 - 0.3 * speed_pref)
            }
            crate::organisms::components::HuntingStrategy::Pursuit => {
                // Pursuit: likes long chases, gains when faster than prey.
                (relative_speed * 0.7 + 0.6).clamp(0.5, 2.0)
            }
            crate::organisms::components::HuntingStrategy::Pack => {
                // Pack: strongly prefers when others present.
                (1.0 + (pack_size_estimate - 1.0) * 0.5).clamp(1.0, 3.0)
            }
        };

        // --- Distance cost: closer prey is better ---
        let distance_cost = (candidate.distance / (base_speed + 1.0)).clamp(0.0, 5.0);

        // --- Learned knowledge about this prey species ---
        let knowledge = learning
            .map(|l| l.get_score(candidate.species_id.value()))
            .unwrap_or(0.4);
        let knowledge_bonus = 0.5 + knowledge * 0.5; // 0.5–1.0

        // --- Desperation mechanics ---
        let desperation_bonus = 0.7 + desperation * 0.6;

        let score = size_score
            * speed_score
            * defense_factor
            * pack_factor
            * strategy_factor
            * knowledge_bonus
            * desperation_bonus
            - distance_cost * 0.25;

        match &best {
            Some((_, best_score)) if score <= *best_score => {}
            _ => best = Some((candidate.clone(), score)),
        }
    }

    best.map(|(o, _)| o)
}

pub fn decide_behavior(
    energy: &Energy,
    cached_traits: &crate::organisms::components::CachedTraits,
    organism_type: OrganismType,
    species_id: SpeciesId,
    sensory: &SensoryData,
    current_state: BehaviorState,
    state_time: f32,
) -> (BehaviorState, Option<Entity>, Option<Vec2>) {
    let decision = decide_behavior_with_memory(
        energy,
        cached_traits,
        species_id,
        organism_type,
        sensory,
        current_state,
        state_time,
        0.0,
        0.0,
        None,
        false,
        None,
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

#[inline]
fn find_best_food_source_weighted(
    organism_type: OrganismType,
    sensory: &SensoryData,
    selectivity: f32,
) -> Option<Vec2> {
    let preferred_resources = match organism_type {
        OrganismType::Consumer => vec![ResourceType::Prey, ResourceType::Plant],
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
#[inline]
fn is_at_food_source(organism_type: OrganismType, sensory: &SensoryData) -> bool {
    let preferred_resources = match organism_type {
        OrganismType::Consumer => vec![ResourceType::Plant, ResourceType::Prey],
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
#[inline]
pub fn calculate_behavior_velocity(
    behavior: &Behavior,
    position: Vec2,
    cached_traits: &crate::organisms::components::CachedTraits,
    _organism_type: OrganismType,
    energy: &Energy,
    time: f32,
    entity_id: u64, // Entity ID for pseudo-random but deterministic variation
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
                // Random direction if no target - use entity-based pseudo-random
                let seed = entity_id
                    .wrapping_mul(2654435761)
                    .wrapping_add((time * 1000.0) as u64);
                let hash = seed.wrapping_mul(0x9e3779b9).wrapping_add(seed >> 32);
                let angle = ((hash as f32) / (u32::MAX as f32 + 1.0)) * std::f32::consts::TAU;
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
                // Add small random offset to migration target to prevent all organisms converging
                let seed = entity_id
                    .wrapping_mul(2654435761)
                    .wrapping_add((time * 1000.0) as u64);
                let hash1 = seed.wrapping_mul(0x9e3779b9).wrapping_add(seed >> 32);
                let hash2 = seed.wrapping_mul(1103515245).wrapping_add(seed >> 16);
                let offset_magnitude = ((hash1 as f32) / (u32::MAX as f32 + 1.0)) * 5.0 - 2.5; // -2.5 to 2.5
                let offset_angle = ((hash2 as f32) / (u32::MAX as f32 + 1.0)) * std::f32::consts::TAU;
                let offset = Vec2::from_angle(offset_angle) * offset_magnitude;
                let adjusted_target = target + offset;
                let direction = (adjusted_target - position).normalize_or_zero();
                direction * current_speed * 0.8
            } else {
                // Use entity-based pseudo-random angle instead of deterministic sin/cos
                let seed = entity_id
                    .wrapping_mul(2654435761)
                    .wrapping_add(position.x.to_bits() as u64)
                    .wrapping_add(position.y.to_bits() as u64)
                    .wrapping_add((time * 1000.0) as u64);
                let hash = seed.wrapping_mul(0x9e3779b9).wrapping_add(seed >> 32);
                let angle = ((hash as f32) / (u32::MAX as f32 + 1.0)) * std::f32::consts::TAU;
                Vec2::from_angle(angle) * current_speed * 0.5
            }
        }
        BehaviorState::Wandering => {
            // Step 8: Different wandering speeds based on organism type
            let wander_speed_mult = match _organism_type {
                OrganismType::Consumer => 0.7, // Consumers move more actively
            };
            // Random walk with pseudo-random but deterministic angle based on entity and position
            // This ensures each organism has a unique wandering pattern
            let seed = entity_id
                .wrapping_mul(2654435761)
                .wrapping_add(position.x.to_bits() as u64)
                .wrapping_add(position.y.to_bits() as u64)
                .wrapping_add((time * 1000.0) as u64);
            // Use a hash-based pseudo-random angle (0 to 2π)
            // Mix bits for better distribution
            let hash = seed
                .wrapping_mul(0x9e3779b9)
                .wrapping_add(seed >> 32);
            let angle = ((hash as f32) / (u32::MAX as f32 + 1.0)) * std::f32::consts::TAU;
            Vec2::from_angle(angle) * current_speed * wander_speed_mult
        }
    }
}
