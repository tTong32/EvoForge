use crate::organisms::behavior::*;
use crate::organisms::components::*;
use crate::organisms::genetics::{traits, Genome};
use crate::utils::SpatialHashGrid;
use crate::world::{ResourceType, WorldGrid};
use bevy::prelude::*;
use bevy::ecs::system::ParamSet;
use glam::Vec2;

use std::collections::{HashMap, HashSet};
use std::fs::{File, OpenOptions};
use std::io::{BufWriter, Write};
use std::path::PathBuf;
use smallvec::SmallVec;

const ALL_ORGANISMS_HEADER: &str = "tick,entity,position_x,position_y,velocity_x,velocity_y,speed,energy_current,energy_max,energy_ratio,age,size,organism_type,behavior_state,state_time,target_x,target_y,target_entity,sensory_range,aggression,boldness,mutation_rate,reproduction_threshold,reproduction_cooldown,foraging_drive,risk_tolerance,exploration_drive,clutch_size,offspring_energy_share,hunger_memory,threat_timer,resource_selectivity,migration_target_x,migration_target_y,migration_active";

/// Event representing damage dealt during a predation attack.
#[derive(Event, Debug, Clone, Copy)]
pub struct PredationDamageEvent {
    pub predator: Entity,
    pub prey: Entity,
    pub damage: f32,
}

/// Helper to find the nearest potential caregiver for an orphaned child.
fn find_nearest_caregiver(
    former_parent: Entity,
    child_pos: &Position,
    child_traits: &CachedTraits,
) -> Option<(Entity, f32)> {
    // This is a placeholder; a real implementation would query the world for
    // nearby adults of the same species with appropriate traits.
    // For now, we return None and rely on independence fallback.
    let _ = (former_parent, child_pos, child_traits);
    None
}

fn ensure_logs_directory() -> PathBuf {
    let logs_dir = PathBuf::from("data/logs");
    if !logs_dir.exists() {
        std::fs::create_dir_all(&logs_dir).expect("Failed to create logs directory");
    }
    logs_dir
}


/// Resource for bulk organism logging (UNUSED)
#[derive(Resource)]
pub struct AllOrganismsLogger {
    csv_writer: Option<BufWriter<File>>,
    csv_path: PathBuf,
    header_written: bool,
    tick_counter: u64,
    sample_interval: u64,
    flush_interval: u64,
}

impl Default for AllOrganismsLogger {
    fn default() -> Self {
        let logs_dir = ensure_logs_directory();
        let timestamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs();
        let csv_path = logs_dir.join(format!("organisms_snapshot_{}.csv", timestamp));

        Self {
            csv_writer: None,
            csv_path,
            header_written: false,
            tick_counter: 0,
            sample_interval: 50, // snapshot every 50 ticks by default
            flush_interval: 500, // flush every ~500 logged ticks
        }
    }
}

impl AllOrganismsLogger {
    fn ensure_writer(&mut self) -> Option<&mut BufWriter<File>> {
        if self.csv_writer.is_none() {
            let file = match OpenOptions::new()
                .create(true)
                .append(true)
                .open(&self.csv_path)
            {
                Ok(file) => file,
                Err(err) => {
                    error!("Failed to open all-organism CSV file: {err}");
                    return None;
                }
            };
            self.csv_writer = Some(BufWriter::new(file));
            info!(
                "[LOGGER] Streaming all-organism snapshots to {}",
                self.csv_path.display()
            );
        }
        self.csv_writer.as_mut()
    }
}

/// Spawn initial organisms in the world (Step 8: Uses tuning parameters)
pub fn spawn_initial_organisms(
    mut commands: Commands,
    mut species_tracker: ResMut<crate::organisms::speciation::SpeciesTracker>, // Step 8: Speciation
    tuning: Res<crate::organisms::EcosystemTuning>, // Step 8: Tuning parameters
    _world_grid: Res<WorldGrid>,
) {
    info!("Spawning initial organisms...");

    let mut rng = fastrand::Rng::new();
    let spawn_count = tuning.initial_spawn_count;
    // Note: initial_species_count removed - using fixed value for now
    let species_count = tuning.initial_species_count;

    // Spawn organisms randomly within initialized chunks
    // Chunks are from -1 to 1
    let world_size = 3 * crate::world::CHUNK_SIZE as i32; // 3 chunks
    let spawn_range = world_size as f32 / 2.0; // -range to +range

    let base_genomes: Vec<Genome> = (0..species_count)
        .map(|_| Genome::random())
        .collect();

    for (species_idx, base_genome) in base_genomes.iter().enumerate() {
        for org_idx in 0..(spawn_count/species_count) {
            let x = rng.f32() * spawn_range * 2.0 - spawn_range;
            let y = rng.f32() * spawn_range * 2.0 - spawn_range;

            // Create random genome for this organism
            let mut genome = base_genome.clone();

            for i in 0..crate::organisms::genetics::GENOME_SIZE {
                if rng.f32() < 0.1 {
                    let mutation = (rng.f32() - 0.5) * 0.1;
                    let new_value = (genome.get_gene(i) + mutation).clamp(0.0, 1.0);
                    genome.set_gene(i, new_value);
                }
            }

            // Express traits from genome
            let size = traits::express_size(&genome);
            let max_energy = traits::express_max_energy(&genome);
            let metabolism_rate = traits::express_metabolism_rate(&genome);
            let movement_cost = traits::express_movement_cost(&genome);
            let reproduction_cooldown = traits::express_reproduction_cooldown(&genome) as u32;

            // Phase 1+: only spawn consumer organisms; plants are cell-based now.
            let organism_type = OrganismType::Consumer;

            // Random initial velocity
            let vel_x = rng.f32() * 20.0 - 10.0;
            let vel_y = rng.f32() * 20.0 - 10.0;

            let cached_traits = CachedTraits::from_genome(&genome);
            
            // Step 8: Assign species ID using speciation system
            let species_id = species_tracker.find_or_create_species(&genome);

            let learning_rate = traits::express_learning_rate(&genome);

            let entity = commands
                .spawn((
                    Position::new(x, y),
                    Velocity::new(vel_x, vel_y),
                    Energy::new(max_energy),
                    Age::new(),
                    Size::new(size),
                    Metabolism::new(metabolism_rate, movement_cost),
                    ReproductionCooldown::new(reproduction_cooldown),
                    genome,
                    cached_traits,
                    species_id, // Step 8: Use speciation-assigned species ID
                    organism_type,
                    // Phase 2: individual learning about prey.
                    IndividualLearning::new(learning_rate),
                    Behavior::new(),
                    Alive,
                ))
                .id();

        }
    }

    info!("Spawned {} organisms", spawn_count);
}

#[derive(Resource)]
pub struct SpatialHashTracker {
    previous_positions: HashMap<Entity, Vec2>,
}
 impl Default for SpatialHashTracker {
    fn default() -> Self {
        Self {
            previous_positions: HashMap::new(),
        }
    }
 }

/// Resource to reuse HashMap for predation damage event grouping (optimization)
#[derive(Resource, Default)]
pub struct PredationEventGrouping {
    attacks_by_prey: HashMap<Entity, Vec<PredationDamageEvent>>,
}

/// Resource to reuse Vec buffer for spatial hash queries (optimization)
#[derive(Resource, Default)]
pub struct SpatialQueryBuffer {
    buffer: Vec<bevy::prelude::Entity>,
}

/// Resource to reuse Vec buffer for reproduction events (optimization)
#[derive(Resource, Default)]
pub struct ReproductionEventBuffer {
    events: Vec<PendingReproductionSpawn>,
}

/// Resource-backed RNG for thread-safe random number generation (optimization)
#[derive(Resource)]
pub struct GlobalRng {
    rng: fastrand::Rng,
}

impl Default for GlobalRng {
    fn default() -> Self {
        Self {
            rng: fastrand::Rng::new(),
        }
    }
}

impl GlobalRng {
    pub fn f32(&mut self) -> f32 {
        self.rng.f32()
    }
    
    pub fn u32(&mut self, bound: u32) -> u32 {
        self.rng.u32(..bound)
    }
}

/// Reusable HashSet for eligible children lookup (optimization)
#[derive(Resource, Default)]
pub struct EligibleChildrenSet {
    set: HashSet<Entity>,
}

impl EligibleChildrenSet {
    pub fn clear(&mut self) {
        self.set.clear();
    }
    
    pub fn insert(&mut self, entity: Entity) {
        self.set.insert(entity);
    }
    
    pub fn contains(&self, entity: &Entity) -> bool {
        self.set.contains(entity)
    }
    
    pub fn len(&self) -> usize {
        self.set.len()
    }
    
    pub fn is_empty(&self) -> bool {
        self.set.is_empty()
    }
}

struct PendingReproductionSpawn {
    parent: Entity,
    position: Vec2,
    genomes: SmallVec<[Genome; 6]>, // Optimization: Use SmallVec for small clutch sizes
    species_id: SpeciesId,
    organism_type: OrganismType,
    energy_share: f32,
}

/// Update spatial hash grid with current organism positions
pub fn update_spatial_hash(
    mut spatial_hash: ResMut<SpatialHashGrid>,
    mut tracker: ResMut<SpatialHashTracker>,
    query: Query<(Entity, &Position), With<Alive>>,
    mut removed: RemovedComponents<Alive>, // Entites that lost alive component
) {
    for entity in removed.read() {
        spatial_hash.organisms.remove(entity);
        tracker.previous_positions.remove(&entity);
    }
    // Update only the entities that have moved or are new
    for (entity, position) in query.iter(){
        let current_pos = position.0;

        if let Some(old_pos) =tracker.previous_positions.get(&entity) {
            // Only update if position changed significant (so avoid micro-updates)
            if (current_pos - *old_pos).length_squared() > 0.25 {
                spatial_hash.organisms.insert(entity, current_pos);
                tracker.previous_positions.insert(entity, current_pos);
            }
        } else {
            // New entity - insert
            spatial_hash.organisms.insert(entity, current_pos);
            tracker.previous_positions.insert(entity, current_pos);
        }
    }
}

/// Update metabolism - organisms consume energy over time
/// Step 10: PARALLELIZED - Uses Bevy's parallel query iterator
/// Step 8: Uses tuning parameters for ecosystem balance
/// Uses cached traits if available, otherwise falls back to Metabolism component
pub fn update_metabolism(
    mut query: Query<(
        &mut Energy,
        &Velocity,
        &Metabolism,
        &Size,
        Option<&CachedTraits>,
    )>,
    time: Res<Time>,
    tuning: Res<crate::organisms::EcosystemTuning>, // Step 8: Tuning parameters
) {
    let dt = time.delta_seconds();
    let base_metabolism_mult = tuning.base_metabolism_multiplier;
    let movement_cost_mult = tuning.movement_cost_multiplier;

    // Step 10: Use parallel iteration for better performance on multi-core systems
    query.par_iter_mut().for_each(|(mut energy, velocity, metabolism, size, traits_opt)| {
        // Use cached traits if available, otherwise use Metabolism component
        let (base_rate, organism_movement_cost) = if let Some(traits) = traits_opt {
            (traits.metabolism_rate, traits.movement_cost)
        } else {
            (metabolism.base_rate, metabolism.movement_cost)
        };

        // Step 8: Apply tuning multipliers
        let effective_base_rate = base_rate * base_metabolism_mult;
        let effective_movement_cost = organism_movement_cost * movement_cost_mult;

        // Base metabolic cost (proportional to size)
        let base_cost = effective_base_rate * size.value() * dt;

        // Movement cost (proportional to speed)
        let speed = velocity.0.length();
        let movement_cost = speed * effective_movement_cost * dt;

        // Total energy consumed
        let total_cost = base_cost + movement_cost;

        // Deduct energy
        energy.current -= total_cost;
        energy.current = energy.current.max(0.0);
    });
}

#[derive(Resource, Default)]
pub struct BehvaiourUpdateBuffer {
    entity_data: Vec<(Entity, Position, Energy, CachedTraits, SpeciesId, OrganismType, Size)>,
}

/// Cell coordinate for identifying a specific cell in the world grid
/// Uses chunk coordinates and local cell coordinates for efficient hashing
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct CellCoordinate {
    chunk_x: i32,
    chunk_y: i32,
    local_x: usize,
    local_y: usize,
}

impl CellCoordinate {
    fn from_world_pos(world_x: f32, world_y: f32) -> Self {
        use crate::world::Chunk;
        
        let (chunk_x, chunk_y) = Chunk::world_to_chunk(world_x, world_y);
        let (local_x, local_y) = Chunk::world_to_local(world_x, world_y);
        
        Self {
            chunk_x,
            chunk_y,
            local_x,
            local_y,
        }
    }
    
    fn world_x(&self) -> f32 {
        use crate::world::{CHUNK_WORLD_SIZE, CELL_SIZE};
        (self.chunk_x as f32 * CHUNK_WORLD_SIZE) + (self.local_x as f32 * CELL_SIZE) + (CELL_SIZE / 2.0)
    }
    
    fn world_y(&self) -> f32 {
        use crate::world::{CHUNK_WORLD_SIZE, CELL_SIZE};
        (self.chunk_y as f32 * CHUNK_WORLD_SIZE) + (self.local_y as f32 * CELL_SIZE) + (CELL_SIZE / 2.0)
    }
}

/// Plant consumption data for a single organism
#[derive(Debug, Clone)]
struct PlantConsumption {
    /// Total fraction of plants eaten (0.0-1.0)
    total_fraction_eaten: f32,
    /// Per-species consumption (species_id -> fraction_eaten)
    species_consumption: HashMap<u32, f32>,
}

/// Cell modifications collected from organisms during eating
#[derive(Debug, Default, Clone)]
struct CellModification {
    /// Energy that should be gained by each organism consuming from this cell
    /// Maps entity -> energy gained
    organism_energy_gains: HashMap<Entity, f32>,
    
    /// Plant consumption aggregations
    plant_consumption: Option<PlantConsumption>,
    
    /// Resource deltas per resource type [Plant, Mineral, Sunlight, Water, Detritus, Prey]
    resource_deltas: [f32; 6],
    
    /// Pressure deltas per resource type (additive)
    pressure_deltas: [f32; 6],
    
    /// Animal nutrient delta (from dead organisms)
    animal_nutrient_delta: f32,
}

impl CellModification {
    fn new() -> Self {
        Self {
            organism_energy_gains: HashMap::new(),
            plant_consumption: None,
            resource_deltas: [0.0; 6],
            pressure_deltas: [0.0; 6],
            animal_nutrient_delta: 0.0,
        }
    }
    
    /// Merge another modification into this one (for aggregating multiple organisms)
    fn merge(&mut self, other: CellModification) {
        // Merge energy gains
        for (entity, energy) in other.organism_energy_gains {
            *self.organism_energy_gains.entry(entity).or_insert(0.0) += energy;
        }
        
        // Merge plant consumption
        if let Some(other_plant) = other.plant_consumption {
            if let Some(ref mut plant) = self.plant_consumption {
                plant.total_fraction_eaten += other_plant.total_fraction_eaten;
                for (species_id, fraction) in other_plant.species_consumption {
                    *plant.species_consumption.entry(species_id).or_insert(0.0) += fraction;
                }
            } else {
                self.plant_consumption = Some(other_plant);
            }
        }
        
        // Merge resource deltas (additive for most, but need to be careful about bounds)
        for i in 0..6 {
            self.resource_deltas[i] += other.resource_deltas[i];
        }
        
        // Merge pressure deltas (additive)
        for i in 0..6 {
            self.pressure_deltas[i] += other.pressure_deltas[i];
        }
        
        // Merge animal nutrients
        self.animal_nutrient_delta += other.animal_nutrient_delta;
    }
}

/// Buffer for collecting cell modifications before applying them
#[derive(Resource, Default)]
pub struct CellModificationsBuffer {
    /// Maps cell coordinates to their modifications
    modifications: HashMap<CellCoordinate, CellModification>,
    /// Energy gains per organism (extracted for easy access)
    organism_energy_gains: HashMap<Entity, f32>,
}

impl CellModificationsBuffer {
    fn clear(&mut self) {
        // Clear modifications but keep HashMap capacity
        self.modifications.clear();
        self.organism_energy_gains.clear();
    }
    
    fn add_modification(&mut self, coord: CellCoordinate, modification: CellModification) {
        // Extract energy gains before merging
        for (entity, energy) in modification.organism_energy_gains.iter() {
            *self.organism_energy_gains.entry(*entity).or_insert(0.0) += energy;
        }
        
        // Create modification without energy gains for cell updates
        let mut cell_mod = modification.clone();
        cell_mod.organism_energy_gains.clear();
        
        self.modifications
            .entry(coord)
            .or_insert_with(CellModification::new)
            .merge(cell_mod);
    }
    
    fn take_modifications(&mut self) -> HashMap<CellCoordinate, CellModification> {
        std::mem::take(&mut self.modifications)
    }
    
    fn take_energy_gains(&mut self) -> HashMap<Entity, f32> {
        std::mem::take(&mut self.organism_energy_gains)
    }
}

/// Update behavior decisions based on sensory input and organism state
pub fn update_behavior(
    mut query: Query<
        (
            Entity,
            &Position,
            &mut Behavior,
            &Energy,
            &CachedTraits,
            &SpeciesId,
            &OrganismType,
            &Size,
            Option<&IndividualLearning>,
        ),
        With<Alive>,
    >,
    world_grid: Res<WorldGrid>,
    spatial_hash: Res<SpatialHashGrid>,
    organism_query: Query<
        (
            Entity,
            &Position,
            &SpeciesId,
            &OrganismType,
            &Size,
            &Energy,
            &CachedTraits,
        ),
        With<Alive>,
    >,
    mut sensory_cache: ResMut<crate::organisms::behavior::SensoryDataCache>, // Add cache
    mut spatial_query_buffer: ResMut<SpatialQueryBuffer>, // Reusable buffer for spatial queries
    time: Res<Time>,
) {
    // Periodic cleanup of sensory cache
    sensory_cache.maybe_cleanup();
    let dt = time.delta_seconds();
    

    // Pre-compute constant values outside loop
    let hunger_decay_factor = (1.0 - dt * 0.25).max(0.65);
    let migration_reached_threshold_sq = 16.0;

    for (entity, position, mut behavior, energy, cached_traits, species_id, organism_type, size, learning_opt) in
        query.iter_mut()
    {
        // Update state time
        behavior.state_time += dt;

        // Settle migration target if already reached (use squared distance to avoid sqrt)
        if let Some(target) = behavior.migration_target {
            if (position.0 - target).length_squared() < migration_reached_threshold_sq {
                behavior.migration_target = None;
            }
        }

        // Update hunger & threat memories - cache energy ratio
        let energy_ratio = energy.ratio();
        let hunger_input = (1.0 - energy_ratio).max(0.0);
        behavior.hunger_memory = (behavior.hunger_memory
            + hunger_input * cached_traits.hunger_memory_rate * dt)
            .min(2.0);
        behavior.hunger_memory *= hunger_decay_factor;

        // Get sensory range from cached traits
        let sensory_range = cached_traits.sensory_range;

        // Collect sensory data using cache (optimization 3)
        let sensory = sensory_cache.get_or_compute(
            entity,
            position.0,
            sensory_range,
            || {
                // Use reusable buffer to avoid allocations
                spatial_query_buffer.buffer.clear();
                collect_sensory_data(
                    entity,
                    position.0,
                    sensory_range,
                    *species_id,
                    *organism_type,
                    size.value(),
                    &world_grid,
                    &spatial_hash.organisms,
                    &mut spatial_query_buffer.buffer,
                    &organism_query,
                )
            }
        );

        if let Some((_, threat_pos, _)) = sensory.nearest_predator {
            behavior.threat_timer =
                (behavior.threat_timer + cached_traits.threat_decay_rate).min(10.0);
            behavior.recent_threat = Some(threat_pos);
        } else {
            behavior.threat_timer =
                (behavior.threat_timer - dt * cached_traits.threat_decay_rate).max(0.0);
            if behavior.threat_timer <= 0.0 {
                behavior.recent_threat = None;
            }
        }

        // Make behavior decision using cached traits
        let decision = decide_behavior_with_memory(
            energy,
            cached_traits,
            *species_id,
            *organism_type,
            &sensory,
            behavior.state,
            behavior.state_time,
            behavior.hunger_memory,
            behavior.threat_timer,
            behavior.recent_threat,
            behavior.migration_target.is_some(),
            learning_opt,
        );

        // Update behavior state and targets
        behavior.set_state(decision.state);
        behavior.target_entity = decision.target_entity;
        behavior.target_position = decision.target_position;

        if matches!(behavior.state, BehaviorState::Migrating) {
            if let Some(target) = decision
                .migration_target
                .or(behavior.migration_target)
                .or_else(|| sensory.richest_resource.map(|(pos, _, _, _)| pos))
            {
                behavior.migration_target = Some(target);
            }
        }
    }
}

/// Update organism movement based on behavior state
pub fn update_movement(
    mut query: Query<
        (
            &mut Position,
            &mut Velocity,
            &Behavior,
            &Energy,
            &CachedTraits,
            &OrganismType,
            Entity,
        ),
        With<Alive>,
    >,
    time: Res<Time>,
) {
    let dt = time.delta_seconds();
    let time_elapsed = time.elapsed_seconds();

    // Optimization: Parallel iteration for independent position/velocity updates
    query.par_iter_mut().for_each(|(mut position, mut velocity, behavior, energy, cached_traits, organism_type, entity)| {
        // Skip if dead
        if energy.is_dead() {
            velocity.0 = Vec2::ZERO;
            return;
        }

        // Calculate velocity based on behavior state using cached traits
        let desired_velocity = calculate_behavior_velocity(
            behavior,
            position.0,
            cached_traits,
            *organism_type,
            energy,
            time_elapsed,
            entity.index() as u64, // Entity ID for pseudo-random variation
        );

        // Smooth velocity transitions (lerp for smoother movement)
        let lerp_factor = 0.3; // How quickly velocity changes
        velocity.0 = velocity.0.lerp(desired_velocity, lerp_factor);

        // Apply velocity damping (friction) for wandering/resting
        if behavior.state == BehaviorState::Wandering || behavior.state == BehaviorState::Resting {
            velocity.0 *= 0.98;
        }

        // Update position
        position.0 += velocity.0 * dt;

        // Simple boundary checking (keep organisms within reasonable bounds)
        let max_pos = 200.0;
        position.0.x = position.0.x.clamp(-max_pos, max_pos);
        position.0.y = position.0.y.clamp(-max_pos, max_pos);
    });
}

/// Calculate what an organism would consume from a cell (read-only operation)
/// Returns a CellModification that can be applied later
fn calculate_consumption_modification(
    cell: &crate::world::Cell,
    entity: Entity,
    organism_type: &OrganismType,
    size: &Size,
    traits_opt: Option<&CachedTraits>,
    plant_consumption_rate: f32,
    meat_consumption_rate: f32,
    dt: f32,
    energy_conversion_efficiency: f32,
) -> Option<CellModification> {
    let mut modification = CellModification::new();
    
    match organism_type {
        OrganismType::Consumer => {
            // Calculate plant consumption
            let max_fraction = (plant_consumption_rate * dt).min(1.0);
            let mut remaining_fraction = max_fraction;
            let mut total_eaten = 0.0;
            
            if !cell.plant_community.is_empty() && remaining_fraction > 0.0 {
                let mut species_consumption = HashMap::new();
                
                // Calculate what would be eaten (without modifying)
                for species in cell.plant_community.iter() {
                    if remaining_fraction <= 0.0 {
                        break;
                    }
                    let bite = species.percentage.min(remaining_fraction);
                    remaining_fraction -= bite;
                    total_eaten += bite;
                    
                    if bite > 0.0 {
                        species_consumption.insert(species.species_id, bite);
                    }
                }
                
                if total_eaten > 0.0 {
                    modification.plant_consumption = Some(PlantConsumption {
                        total_fraction_eaten: total_eaten,
                        species_consumption,
                    });
                    
                    // Calculate energy from plants
                    let plant_energy = total_eaten * size.value() * energy_conversion_efficiency;
                    modification.organism_energy_gains.insert(entity, plant_energy);
                }
            }
            
            // Calculate prey resource consumption
            let current_prey = cell.get_resource(ResourceType::Prey);
            let prey_resource = current_prey.min(meat_consumption_rate * dt);
            
            if prey_resource > 0.0 {
                modification.resource_deltas[ResourceType::Prey as usize] = -prey_resource;
                modification.pressure_deltas[ResourceType::Prey as usize] = prey_resource;
                
                let prey_energy = prey_resource * 2.0 * energy_conversion_efficiency;
                *modification.organism_energy_gains.entry(entity).or_insert(0.0) += prey_energy;
            }
            
            Some(modification)
        }
        _ => None, // Other organism types don't consume from cells in this system
    }
}

/// Apply cell modifications to the actual cell
fn apply_cell_modifications(
    cell: &mut crate::world::Cell,
    modification: &CellModification,
) {
    // Apply plant consumption
    if let Some(ref plant_consumption) = modification.plant_consumption {
        if !cell.plant_community.is_empty() && plant_consumption.total_fraction_eaten > 0.0 {
            // Apply consumption to each species
            for species in cell.plant_community.iter_mut() {
                if let Some(&fraction_eaten) = plant_consumption.species_consumption.get(&species.species_id) {
                    species.percentage = (species.percentage - fraction_eaten).max(0.0);
                }
            }
            
            // Remove species with zero or negative percentages
            cell.plant_community.retain(|s| s.percentage > 0.0);
            
            // Normalize plant community percentages
            let sum: f32 = cell.plant_community.iter().map(|s| s.percentage).sum();
            if sum > 0.0 {
                let inv = 1.0 / sum;
                cell.plant_community.iter_mut().for_each(|s| s.percentage *= inv);
            } else {
                cell.plant_community.clear();
            }
        }
    }
    
    // Apply resource deltas
    for i in 0..6 {
        if modification.resource_deltas[i] != 0.0 {
            let current = cell.resource_density[i];
            cell.resource_density[i] = (current + modification.resource_deltas[i]).max(0.0);
        }
    }
    
    // Apply pressure deltas
    for i in 0..6 {
        if modification.pressure_deltas[i] > 0.0 {
            let resource_type = match i {
                0 => ResourceType::Plant,
                1 => ResourceType::Mineral,
                2 => ResourceType::Sunlight,
                3 => ResourceType::Water,
                4 => ResourceType::Detritus,
                5 => ResourceType::Prey,
                _ => continue,
            };
            cell.add_pressure(resource_type, modification.pressure_deltas[i]);
        }
    }
    
    // Apply animal nutrient delta
    if modification.animal_nutrient_delta > 0.0 {
        cell.animal_nutrients += modification.animal_nutrient_delta;
    }
}

/// Handle eating behavior - consume resources or prey (Step 8: Uses tuning parameters)
/// OPTIMIZED: Split into collection and application phases for better performance
pub fn handle_eating(
    mut query: Query<
        (
            Entity,
            &Position,
            &mut Energy,
            &Behavior,
            &OrganismType,
            &Size,
            &SpeciesId,
            Option<&CachedTraits>,
            Option<&mut PredatorFeeding>,
        ),
        (With<Alive>, Without<crate::organisms::components::ParentalAttachment>),
    >,
    mut world_grid: ResMut<WorldGrid>,
    tuning: Res<crate::organisms::EcosystemTuning>, // Step 8: Tuning parameters
    mut damage_writer: EventWriter<PredationDamageEvent>,
    time: Res<Time>,
    mut eligible_children: ResMut<EligibleChildrenSet>, // Optimization: reuse HashSet
    mut cell_mod_buffer: ResMut<CellModificationsBuffer>, // Optimization: batch cell modifications
    mut param_set: ParamSet<(
        Query<
            (
                Entity,
                &crate::organisms::components::ParentalAttachment,
                &mut Energy,
                &crate::organisms::components::ChildGrowth,
                &Age,
            ),
            With<Alive>,
        >,
    )>,
) {
    let dt = time.delta_seconds();
    let energy_conversion_efficiency = tuning.energy_conversion_efficiency;

    // Clear modification buffer from previous frame
    cell_mod_buffer.clear();

    // PHASE 1: COLLECTION - Collect all cell modifications without mutating cells
    // Also handle non-cell modifications (carcass feeding, predation)
    for (entity, position, mut energy, behavior, organism_type, size, species_id, traits_opt, feeding_opt) in
        query.iter_mut()
    {
        if behavior.state != BehaviorState::Eating {
            continue;
        }

        // Hard capped per-organism consumption rate
        let plant_consumption_rate = traits_opt
            .map(|t| t.plant_consumption_rate)
            .unwrap_or(tuning.plant_consumption_rate_base);
        let meat_consumption_rate = traits_opt
            .map(|t| t.meat_consumption_rate)
            .unwrap_or(tuning.meat_consumption_rate_base);

        match organism_type {
            OrganismType::Consumer => {
                // 1) Carcass feeding - doesn't modify cells, handle immediately
                if let Some(mut feeding) = feeding_opt {
                    let max_intake = (meat_consumption_rate * dt).min(feeding.remaining_energy);
                    if max_intake > 0.0 {
                        feeding.remaining_energy -= max_intake;
                        let gained = max_intake * energy_conversion_efficiency;
                        energy.current = (energy.current + gained).min(energy.max);
                        
                        // Log eating from carcass
                        info!(
                            "[EATING] Entity {} (Species {}) ate {:.2} energy from carcass | Energy: {:.2}/{:.2}",
                            entity.index(),
                            species_id.value(),
                            gained,
                            energy.current,
                            energy.max
                        );
                    }
                    // While feeding from carcass, skip other food sources
                    continue;
                }

                // 2) Predation attacks - send events, don't modify cells
                if let Some(prey_entity) = behavior.target_entity {
                    if let Some(traits) = traits_opt {
                        let damage = traits.attack_strength * dt;
                        if damage > 0.0 {
                            damage_writer.send(PredationDamageEvent {
                                predator: entity,
                                prey: prey_entity,
                                damage,
                            });
                            
                            // Log predation attack
                            info!(
                                "[EATING] Entity {} (Species {}) attacked prey {} | Damage: {:.2}",
                                entity.index(),
                                species_id.value(),
                                prey_entity.index(),
                                damage
                            );
                        }
                    }
                }

                // 3) Cell resource consumption - collect modifications for batching
                if let Some(cell) = world_grid.get_cell(position.x(), position.y()) {
                    let cell_coord = CellCoordinate::from_world_pos(position.x(), position.y());
                    
                    if let Some(modification) = calculate_consumption_modification(
                        cell,
                        entity,
                        organism_type,
                        size,
                        traits_opt,
                        plant_consumption_rate,
                        meat_consumption_rate,
                        dt,
                        energy_conversion_efficiency,
                    ) {
                        cell_mod_buffer.add_modification(cell_coord, modification);
                    }
                }
            }
        }
    }

    // PHASE 2: APPLICATION - Apply all collected modifications to cells
    let modifications = cell_mod_buffer.take_modifications();
    
    for (cell_coord, modification) in modifications {
        if let Some(cell) = world_grid.get_cell_mut(cell_coord.world_x(), cell_coord.world_y()) {
            apply_cell_modifications(cell, &modification);
        }
    }

    // PHASE 3: ENERGY APPLICATION - Apply energy gains to organisms
    let energy_gains = cell_mod_buffer.take_energy_gains();
    
    for (entity, _position, mut energy, behavior, organism_type, _size, species_id, traits_opt, _feeding_opt) in
        query.iter_mut()
    {
        if behavior.state != BehaviorState::Eating {
            continue;
        }

        if let Some(&energy_gain) = energy_gains.get(&entity) {
            // Log eating from cell resources
            if energy_gain > 0.0 {
                info!(
                    "[EATING] Entity {} (Species {}) consumed {:.2} energy from resources | Energy: {:.2}/{:.2}",
                    entity.index(),
                    species_id.value(),
                    energy_gain,
                    energy.current,
                    energy.max
                );
            }
            
            let mut parent_energy_gain = energy_gain;

            // Handle meal sharing with attached children
            if let (OrganismType::Consumer, Some(traits)) = (organism_type, traits_opt) {
                if traits.meal_share_percentage > 0.0 {
                    eligible_children.clear();
                    param_set.p0().for_each(|(child_entity, attachment, _child_energy, _child_growth, child_age)| {
                        if attachment.parent == entity 
                            && (child_age.0 as f32) <= attachment.care_until_age {
                            eligible_children.insert(child_entity);
                        }
                    });

                    if !eligible_children.is_empty() {
                        let child_count = eligible_children.len() as f32;
                        let share_total = energy_gain * traits.meal_share_percentage;
                        let per_child = share_total / child_count;

                        // Apply energy sharing to collected children
                        param_set.p0().for_each_mut(|(child_entity, _attachment, mut child_energy, _child_growth, _child_age)| {
                            if eligible_children.contains(&child_entity) {
                                child_energy.current =
                                    (child_energy.current + per_child).min(child_energy.max);
                                parent_energy_gain -= per_child;
                            }
                        });
                    }
                }
            }

            energy.current = (energy.current + parent_energy_gain).min(energy.max);
        }
    }
}

/// Update organism age and reproduction cooldown
/// Step 10: Use parallel iteration for better performance on multi-core systems
pub fn update_age(mut query: Query<(&mut Age, &mut ReproductionCooldown)>) {
    // Step 10: Parallel iteration for simple independent operations
    query.par_iter_mut().for_each(|(mut age, mut cooldown)| {
        age.increment();
        cooldown.decrement();
    });
}

/// Handle reproduction - both asexual and sexual (Step 8: Uses speciation system)
pub fn handle_reproduction(
    mut commands: Commands,
    mut query: Query<
        (
            Entity,
            &Position,
            &mut Energy,
            &mut ReproductionCooldown,
            &Genome,
            &CachedTraits,
            &SpeciesId,
            &OrganismType,
        ),
        With<Alive>,
    >,
    mut species_tracker: ResMut<crate::organisms::speciation::SpeciesTracker>, // Step 8: Speciation
    tuning: Res<crate::organisms::EcosystemTuning>, // Step 8: Tuning parameters
    spatial_hash: Res<SpatialHashGrid>,
    mut spatial_query_buffer: ResMut<SpatialQueryBuffer>,
    mut reproduction_buffer: ResMut<ReproductionEventBuffer>, // Optimization: reuse buffer
    mut rng: ResMut<GlobalRng>, // Optimization: resource-backed RNG
    organism_query: Query<(Entity, &Position, &Genome, &SpeciesId, &CachedTraits), With<Alive>>,
) {
    // Clear and reuse buffer from previous frame (optimization)
    reproduction_buffer.events.clear();

    for (entity, position, energy, cooldown, genome, cached_traits, species_id, org_type) in
        query.iter()
    {
        if !cooldown.is_ready() {
            continue;
        }

        // Cache energy ratio to avoid multiple calculations
        let energy_ratio = energy.ratio();
        if energy_ratio < cached_traits.reproduction_threshold {
            continue;
        }

        // Use tuning parameter for reproduction chance
        if rng.f32() >= tuning.reproduction_chance_multiplier {
            continue;
        }

        let clutch_size = cached_traits.clutch_size.max(1.0).round().clamp(1.0, 6.0) as usize;
        if clutch_size == 0 {
            continue;
        }

        let parent_mutation_rate = cached_traits.mutation_rate.clamp(0.001, 0.08);
        let use_sexual = rng.f32() < 0.35;

        // Optimized: Store entity and mutation rate instead of cloning genome
        let mut mate_data: Option<(Entity, f32)> = None;

        if use_sexual {
            let sensory_range = cached_traits.sensory_range;
            let sensory_range_sq = sensory_range * sensory_range; // Avoid repeated sqrt - moved outside loop
            // Use reusable buffer to avoid allocation
            spatial_hash
                .organisms
                .query_radius_into(position.0, sensory_range, &mut spatial_query_buffer.buffer);
            let nearby_entities = &spatial_query_buffer.buffer;

            for other_entity in nearby_entities.iter().copied() {
                if other_entity == entity {
                    continue;
                }

                if let Ok((_, other_pos, _other_genome, other_species, other_traits)) =
                    organism_query.get(other_entity)
                {
                    if *other_species != *species_id {
                        continue;
                    }

                    // Use squared distance to avoid sqrt
                    let distance_sq = (position.0 - other_pos.0).length_squared();
                    if distance_sq <= sensory_range_sq {
                        mate_data = Some((
                            other_entity,
                            other_traits.mutation_rate.clamp(0.001, 0.08),
                        ));
                        break;
                    }
                }
            }
        }

        // Optimization: Use SmallVec to avoid heap allocation for small clutch sizes (≤6)
        let mut offspring_genomes: SmallVec<[Genome; 6]> = SmallVec::with_capacity(clutch_size);
        if let Some((mate_entity, mate_mut_rate)) = mate_data {
            // Query mate genome only when needed (avoids cloning until necessary)
            if let Ok((_, _, mate_genome, _, _)) = organism_query.get(mate_entity) {
                let crossover_rate = ((parent_mutation_rate + mate_mut_rate) * 0.5).clamp(0.001, 0.08);
                for _ in 0..clutch_size {
                    offspring_genomes.push(Genome::crossover(genome, mate_genome, crossover_rate));
                }
            } else {
                // Fallback to asexual if mate no longer exists
                for _ in 0..clutch_size {
                    offspring_genomes.push(genome.clone_with_mutation(parent_mutation_rate));
                }
            }
        } else {
            for _ in 0..clutch_size {
                offspring_genomes.push(genome.clone_with_mutation(parent_mutation_rate));
            }
        }

        reproduction_buffer.events.push(PendingReproductionSpawn {
            parent: entity,
            position: position.0,
            genomes: offspring_genomes,
            species_id: *species_id,
            organism_type: *org_type,
            energy_share: cached_traits.offspring_energy_share,
        });
    }

    // Process events (need to move out of buffer, so we collect into a temporary Vec)
    let events = std::mem::take(&mut reproduction_buffer.events);
    for event in events {
        if let Ok((_, _, mut parent_energy, mut parent_cooldown, _, parent_traits, _, _)) =
            query.get_mut(event.parent)
        {
            let count = event.genomes.len() as f32;
            if count == 0.0 {
                continue;
            }

            let available_energy = parent_energy.current.max(0.0);
            let total_energy_to_share = available_energy * event.energy_share;
            let per_child_energy = (total_energy_to_share / count)
                .min(available_energy / count)  // Can't give more than parent has per child
                .max(0.0);
            let total_energy_cost = per_child_energy * count;
            parent_energy.current = (available_energy - total_energy_cost).max(0.0);

            // Capture offspring count before moving event.genomes
            let offspring_count = event.genomes.len();
            let mut spawned_species = None;
            for offspring_genome in event.genomes {
                let cached = CachedTraits::from_genome(&offspring_genome);
                let adult_size = cached.size;
                let adult_max_energy = cached.max_energy;
                let metabolism_rate = cached.metabolism_rate;
                let movement_cost = cached.movement_cost;
                let reproduction_cooldown = cached.reproduction_cooldown.max(1.0) as u32;

                let offset = Vec2::new(rng.f32() * 10.0 - 5.0, rng.f32() * 10.0 - 5.0);

                // Step 8: Assign species ID using speciation system
                let offspring_species = species_tracker.find_or_create_species(&offspring_genome);
                if spawned_species.is_none() {
                    spawned_species = Some(offspring_species);
                }

                let spawn_position = Vec2::new(event.position.x + offset.x, event.position.y + offset.y);

                match cached.spawn_type {
                    crate::organisms::components::SpawnType::Egg => {
                        // Spawn an egg entity – no Alive marker, so it won't behave/move.
                        commands.spawn((
                            Position::new(spawn_position.x, spawn_position.y),
                            crate::organisms::components::Egg {
                                parent: event.parent,
                                incubation_time_remaining: cached.incubation_duration,
                                incubation_type: cached.incubation_type,
                            },
                            offspring_genome,
                            cached,
                            offspring_species,
                            event.organism_type,
                        ));
                    }
                    crate::organisms::components::SpawnType::Baby => {
                        // Spawn a baby organism directly with reduced stats (10% adult size/energy).
                        let size_factor = 0.1;
                        let child_size = adult_size * size_factor;
                        let child_max_energy = adult_max_energy * 0.3;
                        let initial_energy = (per_child_energy * 0.9)
                            .min(child_max_energy)
                            .max(child_max_energy * 0.2);

                        let child_entity = commands
                            .spawn((
                                Position::new(spawn_position.x, spawn_position.y),
                                Velocity::new(0.0, 0.0),
                                Energy::with_energy(child_max_energy, initial_energy),
                                Age::new(),
                                Size::new(child_size),
                                Metabolism::new(metabolism_rate, movement_cost),
                                ReproductionCooldown::new(reproduction_cooldown),
                                offspring_genome,
                                cached,
                                offspring_species,
                                event.organism_type,
                                IndividualLearning::new(parent_traits.learning_rate),
                                Behavior::new(),
                                Alive,
                            ))
                            .id();

                        // Track parent-child relationship and attachment for care & learning.
                        commands.entity(child_entity).insert(crate::organisms::components::ParentChildRelationship {
                            parent: event.parent,
                            child: child_entity,
                            time_together: 0.0,
                        });
                        commands.entity(child_entity).insert(crate::organisms::components::ParentalAttachment {
                            parent: event.parent,
                            care_until_age: parent_traits.parental_care_age,
                        });
                        commands.entity(child_entity).insert(crate::organisms::components::ChildGrowth {
                            growth: size_factor,
                            base_rate: parent_traits.growth_rate,
                            max_rate: parent_traits.max_growth_rate,
                            food_deficit: 0.0,
                            independence_age: parent_traits.parental_care_age,
                        });
                    }
                }
            }

            parent_cooldown.reset(parent_traits.reproduction_cooldown.max(1.0) as u32);
            
            // Log reproduction
            if let Some(species) = spawned_species {
                info!(
                    "[REPRODUCTION] Entity {} (Species {}) reproduced | Offspring: {} | Type: {:?} | Position: ({:.1}, {:.1})",
                    event.parent.index(),
                    species.value(),
                    offspring_count,
                    event.organism_type,
                    event.position.x,
                    event.position.y
                );
            }
        }
    }
}

/// Apply predation damage to prey organisms and generate carcasses for predators.
pub fn apply_predation_damage(
    mut commands: Commands,
    mut events: EventReader<PredationDamageEvent>,
    mut prey_query: Query<(Entity, &mut Energy, &Size, &SpeciesId, &OrganismType), With<Alive>>,
    mut predator_feeding_query: Query<&mut PredatorFeeding>,
    mut predator_learning_query: Query<&mut IndividualLearning>,
    predator_traits_query: Query<&CachedTraits, With<Alive>>,
    mut grouping: ResMut<PredationEventGrouping>,
) {
    // Reuse HashMap from previous frame (optimization - avoids allocation)
    // Clear vectors but keep capacity for reuse
    for attacks in grouping.attacks_by_prey.values_mut() {
        attacks.clear();
    }
    
    // Group attacks by prey to compute pack-based bonuses.
    for ev in events.read() {
        grouping.attacks_by_prey.entry(ev.prey).or_default().push(*ev);
    }

    // Process attacks - iterate over mutable entries to avoid taking ownership
    // This preserves HashMap capacity and avoids reallocation
    for (prey_entity, attacks) in grouping.attacks_by_prey.iter_mut() {
        if let Ok((_, mut energy, size, species_id, organism_type)) = prey_query.get_mut(*prey_entity) {
            if energy.current <= 0.0 {
                continue;
            }

            let mut total_damage = 0.0_f32;
            let pack_size = attacks.len() as f32;

            // Apply each predator's damage with a pack coordination bonus.
            for ev in attacks.iter() {
                let base_damage = ev.damage;
                if base_damage <= 0.0 {
                    continue;
                }

                let coord = predator_traits_query
                    .get(ev.predator)
                    .map(|t| t.coordination)
                    .unwrap_or(0.0);

                // Pack bonus: up to ~2x damage when many well-coordinated predators attack.
                let pack_factor = if pack_size > 1.0 {
                    let mult = 1.0 + coord * (pack_size.sqrt() - 1.0).max(0.0);
                    mult.clamp(1.0, 2.0)
                } else {
                    1.0
                };

                total_damage += base_damage * pack_factor;
            }

            if total_damage <= 0.0 {
                continue;
            }

            let before = energy.current;
            energy.current = (energy.current - total_damage).max(0.0);

            if before > 0.0 && energy.current <= 0.0 {
                // Log when prey gets eaten
                info!(
                    "[EATEN] Entity {} (Species {}, Type: {:?}) was killed by predation | Size: {:.2}",
                    prey_entity.index(),
                    species_id.value(),
                    organism_type,
                    size.value()
                );
                
                // Prey died from this coordinated attack: create a shared carcass.
                let carcass_energy_total = size.value().max(0.1) * 8.0;
                let share_per_pred = carcass_energy_total / pack_size.max(1.0);

                for ev in attacks.iter() {
                    if let Ok(mut feeding) = predator_feeding_query.get_mut(ev.predator) {
                        feeding.remaining_energy += share_per_pred;
                    } else {
                        commands
                            .entity(ev.predator)
                            .insert(PredatorFeeding { remaining_energy: share_per_pred });
                    }

                    // Update each predator's learning about this prey species.
                    if let Ok(mut learning) = predator_learning_query.get_mut(ev.predator) {
                        learning.update_on_success(species_id.value());
                    }
                }

                // Mark prey as killed by predation for death handling.
                // Verify entity still exists before inserting (it might have been despawned)
                if prey_query.get(*prey_entity).is_ok() {
                    commands
                        .entity(*prey_entity)
                        .insert(crate::organisms::components::KilledByPredation);
                }
            }
        }
    }
}

/// Parent-child knowledge transfer system.
/// Optimized: Only update when parent is nearby, cache distance check
pub fn update_parent_child_learning(
    mut commands: Commands,
    time: Res<Time>,
    mut children_query: Query<
        (
            Entity,
            &mut crate::organisms::components::ParentChildRelationship,
            &mut IndividualLearning,
            &Position,
            &CachedTraits,
        ),
        With<Alive>,
    >,
    parents_query: Query<(&Position, &IndividualLearning, &CachedTraits), (With<Alive>, Without<crate::organisms::components::ParentChildRelationship>)>,
) {
    let dt = time.delta_seconds();
    const LEARNING_DISTANCE: f32 = 20.0;
    const LEARNING_DISTANCE_SQ: f32 = LEARNING_DISTANCE * LEARNING_DISTANCE;

    for (child_entity, mut rel, mut child_learning, child_pos, child_traits) in
        children_query.iter_mut()
    {
        if let Ok((parent_pos, parent_learning, parent_traits)) = parents_query.get(rel.parent) {
            // Use squared distance to avoid sqrt (optimization)
            let distance_sq = (child_pos.as_vec2() - parent_pos.as_vec2()).length_squared();
            if distance_sq < LEARNING_DISTANCE_SQ {
                rel.time_together += dt;
                // Continuous knowledge transfer: blend child toward parent using genome-driven rate.
                let base_rate = parent_traits.knowledge_transfer_rate
                    .max(child_traits.knowledge_transfer_rate)
                    .max(0.05);
                let transfer_rate = base_rate * dt;
                // Only iterate over parent's knowledge (typically smaller than child's)
                for (prey_id, parent_score) in parent_learning.prey_knowledge.iter() {
                    let child_score = child_learning.get_score(*prey_id);
                    let updated =
                        child_score + (*parent_score - child_score) * transfer_rate;
                    child_learning
                        .prey_knowledge
                        .insert(*prey_id, updated.clamp(0.0, 1.0));
                }
            }
        } else {
            // Parent no longer exists – remove relationship.
            commands
                .entity(child_entity)
                .remove::<crate::organisms::components::ParentChildRelationship>();
        }
    }
}

/// Handle organism death (remove entities with zero energy)
pub fn handle_death(
    mut commands: Commands,
    mut spatial_hash: ResMut<SpatialHashGrid>,
    mut world_grid: ResMut<WorldGrid>,
    query: Query<(Entity, &Energy, &Position, &Size, &SpeciesId, &OrganismType), (With<Alive>, Without<crate::organisms::components::ParentalAttachment>)>,
    mut children_query: Query<
        (
            Entity,
            &mut crate::organisms::components::ParentalAttachment,
            &mut Energy,
            &CachedTraits,
            &mut crate::organisms::components::ParentChildRelationship,
            &Position,
        ),
        With<Alive>,
    >,
    mut predation_flags: Query<&mut crate::organisms::components::KilledByPredation>,
) {
    for (entity, energy, position, size, species_id, organism_type) in query.iter() {
        if energy.is_dead() {
            // Add nutrients to the cell from animal biomass.
            if let Some(cell) = world_grid.get_cell_mut(position.x(), position.y()) {
                let biomass = size.value().max(0.1);
                cell.animal_nutrients += biomass * 0.05;
            }

            // Determine cause of death
            let killed_by_predation = predation_flags.get_mut(entity).is_ok();
            let cause = if killed_by_predation {
                "predation"
            } else {
                "starvation/exhaustion"
            };
            
            // Log death
            info!(
                "[DEATH] Entity {} (Species {}, Type: {:?}) died from {} | Position: ({:.1}, {:.1}), Size: {:.2}, Final Energy: {:.2}",
                entity.index(),
                species_id.value(),
                organism_type,
                cause,
                position.x(),
                position.y(),
                size.value(),
                energy.current
            );

            // Handle attached children depending on cause of death.
            for (child_entity, mut attachment, mut child_energy, child_traits, mut rel, child_pos) in
                children_query.iter_mut()
            {
                if attachment.parent == entity {
                    if killed_by_predation {
                        // If parent was hunted, child dies with parent.
                        child_energy.current = 0.0;
                    } else {
                        // Other causes: try to reattach to a new caregiver if allowed.
                        if child_traits.father_provides_care {
                            // Find nearest alive conspecific as a surrogate "father".
                            if let Some((new_parent, _dist)) = find_nearest_caregiver(
                                entity,
                                child_pos,
                                &child_traits,
                            ) {
                                attachment.parent = new_parent;
                                rel.parent = new_parent;
                                rel.time_together = 0.0;
                                continue;
                            }
                        }
                        // No suitable caregiver: child becomes independent.
                        commands
                            .entity(child_entity)
                            .remove::<crate::organisms::components::ParentalAttachment>();
                    }
                }
            }

            // Remove from spatial hash before despawning
            spatial_hash.organisms.remove(entity);
            commands.entity(entity).despawn();
        }
    }
}

pub fn log_all_organisms(
    mut state: ResMut<AllOrganismsLogger>,
    query: Query<
        (
            Entity,
            &Position,
            &Velocity,
            &Energy,
            &Age,
            &Size,
            &OrganismType,
            &Behavior,
            &CachedTraits,
        ),
        With<Alive>,
    >,
) {
    state.tick_counter += 1;

    if state.sample_interval > 1 && state.tick_counter % state.sample_interval != 0 {
        return;
    }

    let tick = state.tick_counter;
    let header_needed = !state.header_written;
    let flush_interval = state.flush_interval;

    {
        let writer = match state.ensure_writer() {
            Some(writer) => writer,
            None => return,
        };

        if header_needed {
            writeln!(writer, "{}", ALL_ORGANISMS_HEADER)
                .expect("Failed to write all-organisms header");
        }

        for (entity, position, velocity, energy, age, size, org_type, behavior, cached_traits) in
            query.iter()
        {
            let speed = velocity.0.length();

            let energy_ratio = energy.ratio();
            // Optimized: Use match instead of format! to avoid string allocation
            let behavior_state = match behavior.state {
                crate::organisms::behavior::BehaviorState::Wandering => "Wandering",
                crate::organisms::behavior::BehaviorState::Chasing => "Chasing",
                crate::organisms::behavior::BehaviorState::Eating => "Eating",
                crate::organisms::behavior::BehaviorState::Fleeing => "Fleeing",
                crate::organisms::behavior::BehaviorState::Mating => "Mating",
                crate::organisms::behavior::BehaviorState::Resting => "Resting",
                crate::organisms::behavior::BehaviorState::Migrating => "Migrating",
            };
            let organism_type = match org_type {
                OrganismType::Consumer => "Consumer",
            };
            let (target_x, target_y) = behavior
                .target_position
                .map(|pos| (pos.x, pos.y))
                .unwrap_or((f32::NAN, f32::NAN));
            // Optimized: Format target_entity directly in write! to avoid string allocation
            let target_entity_index = behavior.target_entity.map(|e| e.index());
            let migration = behavior.migration_target.or(behavior.target_position);
            let (migration_x, migration_y) = migration
                .map(|pos| (pos.x, pos.y))
                .unwrap_or((f32::NAN, f32::NAN));
            let migration_active = if behavior.state == BehaviorState::Migrating
                || behavior.migration_target.is_some()
            {
                1u8
            } else {
                0u8
            };

            // Optimized: Single write! block instead of duplicate code, format entity directly
            match target_entity_index {
                Some(idx) => {
                    write!(
                        writer,
                        "{tick},{entity},{pos_x:.6},{pos_y:.6},{vel_x:.6},{vel_y:.6},{speed:.6},{energy_current:.6},{energy_max:.6},{energy_ratio:.6},{age},{size:.6},{organism_type},{behavior_state},{state_time:.6},{target_x:.6},{target_y:.6},{target_entity},{sensory_range:.6},{aggression:.6},{boldness:.6},{mutation_rate:.6},{reproduction_threshold:.6},{reproduction_cooldown:.6},{foraging_drive:.6},{risk_tolerance:.6},{exploration_drive:.6},{clutch_size:.6},{offspring_share:.6},{hunger_memory:.6},{threat_timer:.6},{resource_selectivity:.6},{migration_x:.6},{migration_y:.6},{migration_active}\n",
                        tick = tick,
                        entity = entity.index(),
                        pos_x = position.0.x,
                        pos_y = position.0.y,
                        vel_x = velocity.0.x,
                        vel_y = velocity.0.y,
                        speed = speed,
                        energy_current = energy.current,
                        energy_max = energy.max,
                        energy_ratio = energy_ratio,
                        age = age.0,
                        size = size.value(),
                        organism_type = organism_type,
                        behavior_state = behavior_state,
                        state_time = behavior.state_time,
                        target_x = target_x,
                        target_y = target_y,
                        target_entity = idx,
                        sensory_range = cached_traits.sensory_range,
                        aggression = cached_traits.aggression,
                        boldness = cached_traits.boldness,
                        mutation_rate = cached_traits.mutation_rate,
                        reproduction_threshold = cached_traits.reproduction_threshold,
                        reproduction_cooldown = cached_traits.reproduction_cooldown,
                        foraging_drive = cached_traits.foraging_drive,
                        risk_tolerance = cached_traits.risk_tolerance,
                        exploration_drive = cached_traits.exploration_drive,
                        clutch_size = cached_traits.clutch_size,
                        offspring_share = cached_traits.offspring_energy_share,
                        hunger_memory = behavior.hunger_memory,
                        threat_timer = behavior.threat_timer,
                        resource_selectivity = cached_traits.resource_selectivity,
                        migration_x = migration_x,
                        migration_y = migration_y,
                        migration_active = migration_active
                    )
                    .expect("Failed to write all-organism CSV row");
                }
                None => {
                    write!(
                        writer,
                        "{tick},{entity},{pos_x:.6},{pos_y:.6},{vel_x:.6},{vel_y:.6},{speed:.6},{energy_current:.6},{energy_max:.6},{energy_ratio:.6},{age},{size:.6},{organism_type},{behavior_state},{state_time:.6},{target_x:.6},{target_y:.6},None,{sensory_range:.6},{aggression:.6},{boldness:.6},{mutation_rate:.6},{reproduction_threshold:.6},{reproduction_cooldown:.6},{foraging_drive:.6},{risk_tolerance:.6},{exploration_drive:.6},{clutch_size:.6},{offspring_share:.6},{hunger_memory:.6},{threat_timer:.6},{resource_selectivity:.6},{migration_x:.6},{migration_y:.6},{migration_active}\n",
                        tick = tick,
                        entity = entity.index(),
                        pos_x = position.0.x,
                        pos_y = position.0.y,
                        vel_x = velocity.0.x,
                        vel_y = velocity.0.y,
                        speed = speed,
                        energy_current = energy.current,
                        energy_max = energy.max,
                        energy_ratio = energy_ratio,
                        age = age.0,
                        size = size.value(),
                        organism_type = organism_type,
                        behavior_state = behavior_state,
                        state_time = behavior.state_time,
                        target_x = target_x,
                        target_y = target_y,
                        sensory_range = cached_traits.sensory_range,
                        aggression = cached_traits.aggression,
                        boldness = cached_traits.boldness,
                        mutation_rate = cached_traits.mutation_rate,
                        reproduction_threshold = cached_traits.reproduction_threshold,
                        reproduction_cooldown = cached_traits.reproduction_cooldown,
                        foraging_drive = cached_traits.foraging_drive,
                        risk_tolerance = cached_traits.risk_tolerance,
                        exploration_drive = cached_traits.exploration_drive,
                        clutch_size = cached_traits.clutch_size,
                        offspring_share = cached_traits.offspring_energy_share,
                        hunger_memory = behavior.hunger_memory,
                        threat_timer = behavior.threat_timer,
                        resource_selectivity = cached_traits.resource_selectivity,
                        migration_x = migration_x,
                        migration_y = migration_y,
                        migration_active = migration_active
                    )
                    .expect("Failed to write all-organism CSV row");
                }
            }
        }

        if flush_interval > 0 && tick % flush_interval == 0 {
            writer
                .flush()
                .expect("Failed to flush all-organism CSV writer");
        }
    }

    if header_needed {
        state.header_written = true;
    }
}

