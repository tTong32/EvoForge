use crate::organisms::behavior::*;
use crate::organisms::components::*;
use crate::organisms::genetics::{traits, Genome};
use crate::utils::SpatialHashGrid;
use crate::world::{ResourceType, WorldGrid};
use bevy::prelude::*;
use glam::Vec2;

use std::collections::HashMap;
use std::fs::{File, OpenOptions};
use std::io::{BufWriter, Write};
use std::path::PathBuf;

const ALL_ORGANISMS_HEADER: &str = "tick,entity,position_x,position_y,velocity_x,velocity_y,speed,energy_current,energy_max,energy_ratio,age,size,organism_type,behavior_state,state_time,target_x,target_y,target_entity,sensory_range,aggression,boldness,mutation_rate,reproduction_threshold,reproduction_cooldown,foraging_drive,risk_tolerance,exploration_drive,clutch_size,offspring_energy_share,hunger_memory,threat_timer,resource_selectivity,migration_target_x,migration_target_y,migration_active";

fn ensure_logs_directory() -> PathBuf {
    let logs_dir = PathBuf::from("data/logs");
    if !logs_dir.exists() {
        std::fs::create_dir_all(&logs_dir).expect("Failed to create logs directory");
    }
    logs_dir
}

/// Resource to track which organism we're logging
#[derive(Resource)]
pub struct TrackedOrganism {
    entity: Option<Entity>,
    log_counter: u32,
    csv_writer: Option<BufWriter<File>>,
    csv_path: PathBuf,
    header_written: bool,
}

// TRACKED ORGANISM LOGGING
impl Default for TrackedOrganism {
    fn default() -> Self {
        let logs_dir = ensure_logs_directory();

        // Create CSV file with timestamp
        let timestamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs();
        let csv_path = logs_dir.join(format!("organism_tracking_{}.csv", timestamp));

        Self {
            entity: None,
            log_counter: 0,
            csv_writer: None,
            csv_path,
            header_written: false,
        }
    }
}

/// Resource for bulk organism logging
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
    mut tracked: ResMut<TrackedOrganism>,
    mut species_tracker: ResMut<crate::organisms::speciation::SpeciesTracker>, // Step 8: Speciation
    tuning: Res<crate::organisms::EcosystemTuning>, // Step 8: Tuning parameters
    _world_grid: Res<WorldGrid>,
) {
    info!("Spawning initial organisms...");

    let mut rng = fastrand::Rng::new();
    let spawn_count = tuning.initial_spawn_count;

    // Spawn organisms randomly within initialized chunks
    // Chunks are from -1 to 1, each chunk is 64x64 cells
    let world_size = 3 * 64; // 3 chunks * 64 cells
    let spawn_range = world_size as f32 / 2.0; // -range to +range

    let mut first_entity = None;

    for i in 0..spawn_count {
        let x = rng.f32() * spawn_range * 2.0 - spawn_range;
        let y = rng.f32() * spawn_range * 2.0 - spawn_range;

        // Create random genome for this organism
        let genome = Genome::random();

        // Express traits from genome
        let size = traits::express_size(&genome);
        let max_energy = traits::express_max_energy(&genome);
        let metabolism_rate = traits::express_metabolism_rate(&genome);
        let movement_cost = traits::express_movement_cost(&genome);
        let reproduction_cooldown = traits::express_reproduction_cooldown(&genome) as u32;

        let organism_type = match rng.usize(0..3) {
            0 => OrganismType::Producer,
            1 => OrganismType::Consumer,
            _ => OrganismType::Decomposer,
        };

        // Random initial velocity
        let vel_x = rng.f32() * 20.0 - 10.0;
        let vel_y = rng.f32() * 20.0 - 10.0;

        let cached_traits = CachedTraits::from_genome(&genome);
        
        // Step 8: Assign species ID using speciation system
        let species_id = species_tracker.find_or_create_species(&genome);

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
                Behavior::new(),
                Alive,
            ))
            .id();

        // Track the first organism spawned
        if i == 0 {
            first_entity = Some(entity);
        }
    }

    // TRACKED ORGANISM LOGGING
    // Set the first organism as the tracked one
    if let Some(entity) = first_entity {
        tracked.entity = Some(entity);

        // Initialize CSV writer
        let file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&tracked.csv_path)
            .expect("Failed to open CSV file for writing");
        tracked.csv_writer = Some(BufWriter::new(file));

        info!("[TRACKED] Started tracking organism entity: {:?}", entity);
        info!("[TRACKED] CSV logging to: {}", tracked.csv_path.display());
        info!("[TRACKED] Logging will begin after 10 ticks...");
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
            if (current_pos - *old_pos).length_squared() > 0.01 {
                spatial_hash.organisms.insert(entity, current_pos);
                tracker.previous_positions.insert(entity, current_pos);
            }
        } else {
            // New entity - insert
            spatial_hash.organisms.insert(entity, current_pos);
            tracker.previous_positions.insert(entity, current_pos);
        }
    }

    // Finally, clean up entries that no longer exist (safety check)
    tracker.previous_positions.retain(|entity, _| {
        query.get(*entity).is_ok()
    });
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

    // Step 10: Bevy automatically parallelizes systems, so regular iteration is fine
    // Chunk processing is parallelized separately for better performance
    for (mut energy, velocity, metabolism, size, traits_opt) in query.iter_mut() {
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
        ),
        With<Alive>,
    >,
    world_grid: Res<WorldGrid>,
    spatial_hash: Res<SpatialHashGrid>,
    organism_query: Query<
        (Entity, &Position, &SpeciesId, &OrganismType, &Size, &Energy),
        With<Alive>,
    >,
    mut sensory_cache: ResMut<crate::organisms::behavior::SensoryDataCache>, // Add cache
    time: Res<Time>,
) {
    let dt = time.delta_seconds();

    for (entity, position, mut behavior, energy, cached_traits, species_id, organism_type, size) in
        query.iter_mut()
    {
        // Update state time
        behavior.state_time += dt;

        // Settle migration target if already reached
        if let Some(target) = behavior.migration_target {
            if (position.0 - target).length() < 4.0 {
                behavior.migration_target = None;
            }
        }

        // Update hunger & threat memories
        let hunger_input = (1.0 - energy.ratio()).max(0.0);
        behavior.hunger_memory = (behavior.hunger_memory
            + hunger_input * cached_traits.hunger_memory_rate * dt)
            .min(2.0);
        behavior.hunger_memory *= (1.0 - dt * 0.25).max(0.65);

        // Get sensory range from cached traits
        let sensory_range = cached_traits.sensory_range;

        // Collect sensory data using cache (optimization 3)
        let sensory = sensory_cache.get_or_compute(
            entity,
            position.0,
            sensory_range,
            || collect_sensory_data(
                entity,
                position.0,
                sensory_range,
                *species_id,
                *organism_type,
                size.value(),
                &world_grid,
                &spatial_hash.organisms,
                &organism_query,
            )
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
            *organism_type,
            &sensory,
            behavior.state,
            behavior.state_time,
            behavior.hunger_memory,
            behavior.threat_timer,
            behavior.recent_threat,
            behavior.migration_target.is_some(),
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
    tracked: ResMut<TrackedOrganism>,
) {
    let dt = time.delta_seconds();
    let time_elapsed = time.elapsed_seconds();

    for (mut position, mut velocity, behavior, energy, cached_traits, organism_type, entity) in
        query.iter_mut()
    {
        // Skip if dead
        if energy.is_dead() {
            velocity.0 = Vec2::ZERO;
            continue;
        }

        // Calculate velocity based on behavior state using cached traits
        let desired_velocity = calculate_behavior_velocity(
            behavior,
            position.0,
            cached_traits,
            *organism_type,
            energy,
            time_elapsed,
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

        if tracked.entity == Some(entity) && behavior.state_time < dt * 2.0 {
            // Log behavior changes
            info!(
                "[TRACKED] Behavior: {:?}, Velocity: ({:.2}, {:.2}), Speed: {:.2}",
                behavior.state,
                velocity.0.x,
                velocity.0.y,
                velocity.0.length()
            );
        }
    }
}

/// Handle eating behavior - consume resources or prey (Step 8: Uses tuning parameters)
pub fn handle_eating(
    mut query: Query<
        (
            Entity,
            &Position,
            &mut Energy,
            &Behavior,
            &OrganismType,
            &Size,
        ),
        With<Alive>,
    >,
    mut world_grid: ResMut<WorldGrid>,
    tuning: Res<crate::organisms::EcosystemTuning>, // Step 8: Tuning parameters
    _organism_query: Query<(&Position, &mut Energy, &Size), (With<Alive>, Without<Behavior>)>,
    time: Res<Time>,
) {
    let dt = time.delta_seconds();
    let consumption_rate = tuning.consumption_rate_base;
    let energy_conversion_efficiency = tuning.energy_conversion_efficiency;

    for (_entity, position, mut energy, behavior, organism_type, _size) in query.iter_mut() {
        if behavior.state != BehaviorState::Eating {
            continue;
        }

        // Get current cell
        if let Some(cell) = world_grid.get_cell_mut(position.x(), position.y()) {
            let consumed = match organism_type {
                OrganismType::Producer => {
                    // Producers consume sunlight, water, minerals
                    let sunlight = cell
                        .get_resource(ResourceType::Sunlight)
                        .min(consumption_rate * dt);
                    let water = cell
                        .get_resource(ResourceType::Water)
                        .min(consumption_rate * dt * 0.5);
                    let mineral = cell
                        .get_resource(ResourceType::Mineral)
                        .min(consumption_rate * dt * 0.2);

                    cell.set_resource(
                        ResourceType::Sunlight,
                        cell.get_resource(ResourceType::Sunlight) - sunlight,
                    );
                    cell.set_resource(
                        ResourceType::Water,
                        cell.get_resource(ResourceType::Water) - water,
                    );
                    cell.set_resource(
                        ResourceType::Mineral,
                        cell.get_resource(ResourceType::Mineral) - mineral,
                    );
                    cell.add_pressure(ResourceType::Sunlight, sunlight);
                    cell.add_pressure(ResourceType::Water, water);
                    cell.add_pressure(ResourceType::Mineral, mineral);

                    (sunlight + water + mineral) * energy_conversion_efficiency
                }
                OrganismType::Consumer => {
                    // Consumers consume plants or prey resources
                    let plant = cell
                        .get_resource(ResourceType::Plant)
                        .min(consumption_rate * dt);
                    let prey_resource = cell
                        .get_resource(ResourceType::Prey)
                        .min(consumption_rate * dt);

                    cell.set_resource(
                        ResourceType::Plant,
                        cell.get_resource(ResourceType::Plant) - plant,
                    );
                    cell.set_resource(
                        ResourceType::Prey,
                        cell.get_resource(ResourceType::Prey) - prey_resource,
                    );
                    cell.add_pressure(ResourceType::Plant, plant);
                    cell.add_pressure(ResourceType::Prey, prey_resource);

                    (plant + prey_resource * 2.0) * energy_conversion_efficiency
                    // Prey is more nutritious
                }
                OrganismType::Decomposer => {
                    // Decomposers consume detritus
                    let detritus = cell
                        .get_resource(ResourceType::Detritus)
                        .min(consumption_rate * dt);

                    cell.set_resource(
                        ResourceType::Detritus,
                        cell.get_resource(ResourceType::Detritus) - detritus,
                    );
                    cell.add_pressure(ResourceType::Detritus, detritus);

                    // Step 8: Use tuning parameter for decomposer efficiency
                    detritus * energy_conversion_efficiency * tuning.decomposer_efficiency_multiplier
                }
            };

            // Add energy (clamped to max)
            energy.current = (energy.current + consumed).min(energy.max);
        }
    }
}

/// Update organism age and reproduction cooldown
/// Step 10: Bevy automatically parallelizes systems at the scheduler level
pub fn update_age(mut query: Query<(&mut Age, &mut ReproductionCooldown)>) {
    // Step 10: Bevy's scheduler handles parallelization automatically
    for (mut age, mut cooldown) in query.iter_mut() {
        age.increment();
        cooldown.decrement();
    }
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
    organism_query: Query<(Entity, &Position, &Genome, &SpeciesId, &CachedTraits), With<Alive>>,
) {
    struct PendingSpawn {
        parent: Entity,
        position: Vec2,
        genomes: Vec<Genome>,
        species_id: SpeciesId,
        organism_type: OrganismType,
        energy_share: f32,
    }

    let mut rng = fastrand::Rng::new();
    let mut reproduction_events: Vec<PendingSpawn> = Vec::new();

    for (entity, position, energy, cooldown, genome, cached_traits, species_id, org_type) in
        query.iter()
    {
        if !cooldown.is_ready() {
            continue;
        }

        if energy.ratio() < cached_traits.reproduction_threshold {
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

        let mut mate_data: Option<(Genome, f32)> = None;

        if use_sexual {
            let sensory_range = cached_traits.sensory_range;
            let nearby_entities = spatial_hash
                .organisms
                .query_radius(position.0, sensory_range);

            for other_entity in nearby_entities {
                if other_entity == entity {
                    continue;
                }

                if let Ok((_, other_pos, other_genome, other_species, other_traits)) =
                    organism_query.get(other_entity)
                {
                    if *other_species != *species_id {
                        continue;
                    }

                    let distance = (position.0 - other_pos.0).length();
                    if distance <= sensory_range {
                        mate_data = Some((
                            other_genome.clone(),
                            other_traits.mutation_rate.clamp(0.001, 0.08),
                        ));
                        break;
                    }
                }
            }
        }

        let mut offspring_genomes = Vec::with_capacity(clutch_size);
        if let Some((mate_genome, mate_mut_rate)) = mate_data.as_ref() {
            let crossover_rate = ((parent_mutation_rate + mate_mut_rate) * 0.5).clamp(0.001, 0.08);
            for _ in 0..clutch_size {
                offspring_genomes.push(Genome::crossover(genome, mate_genome, crossover_rate));
            }
        } else {
            for _ in 0..clutch_size {
                offspring_genomes.push(genome.clone_with_mutation(parent_mutation_rate));
            }
        }

        reproduction_events.push(PendingSpawn {
            parent: entity,
            position: position.0,
            genomes: offspring_genomes,
            species_id: *species_id,
            organism_type: *org_type,
            energy_share: cached_traits.offspring_energy_share,
        });
    }

    for event in reproduction_events {
        if let Ok((_, _, mut parent_energy, mut parent_cooldown, _, parent_traits, _, _)) =
            query.get_mut(event.parent)
        {
            let count = event.genomes.len() as f32;
            if count == 0.0 {
                continue;
            }

            let available_energy = parent_energy.current.max(0.0);
            let per_child_energy = (available_energy * event.energy_share)
                .min(available_energy / count)
                .max(0.0);
            let total_energy_cost = per_child_energy * count;
            parent_energy.current = (available_energy - total_energy_cost).max(0.0);

            let mut spawned_species = None;
            for offspring_genome in event.genomes {
                let cached = CachedTraits::from_genome(&offspring_genome);
                let size = cached.size;
                let max_energy = cached.max_energy;
                let metabolism_rate = cached.metabolism_rate;
                let movement_cost = cached.movement_cost;
                let reproduction_cooldown = cached.reproduction_cooldown.max(1.0) as u32;

                let offset = Vec2::new(rng.f32() * 10.0 - 5.0, rng.f32() * 10.0 - 5.0);
                let initial_energy = (per_child_energy * 0.9)
                    .min(max_energy)
                    .max(max_energy * 0.15);

                // Step 8: Assign species ID using speciation system
                let offspring_species = species_tracker.find_or_create_species(&offspring_genome);
                if spawned_species.is_none() {
                    spawned_species = Some(offspring_species);
                }
                
                commands.spawn((
                    Position::new(event.position.x + offset.x, event.position.y + offset.y),
                    Velocity::new(0.0, 0.0),
                    Energy::with_energy(max_energy, initial_energy),
                    Age::new(),
                    Size::new(size),
                    Metabolism::new(metabolism_rate, movement_cost),
                    ReproductionCooldown::new(reproduction_cooldown),
                    offspring_genome,
                    cached,
                    offspring_species, // Step 8: Use speciation-assigned species ID
                    event.organism_type,
                    Behavior::new(),
                    Alive,
                ));
            }

            parent_cooldown.reset(parent_traits.reproduction_cooldown.max(1.0) as u32);
            
            // Step 8: Log species information on reproduction
            if let Some(species) = spawned_species {
                let species_count = species_tracker.species_count();
                if count as u32 % 10 == 0 || species_count <= 5 {
                    // Log every 10th reproduction or when few species exist
                    info!(
                        "[REPRODUCTION] Spawned {} offspring | Species: {} (parent: {})",
                        count as u32,
                        species_count,
                        species.value()
                    );
                }
            }
        }
    }
}

/// Handle organism death (remove entities with zero energy)
pub fn handle_death(
    mut commands: Commands,
    mut tracked: ResMut<TrackedOrganism>,
    mut spatial_hash: ResMut<SpatialHashGrid>,
    query: Query<(Entity, &Energy), With<Alive>>,
) {
    for (entity, energy) in query.iter() {
        if energy.is_dead() {
            if tracked.entity == Some(entity) {
                info!(
                    "[TRACKED] Organism died! Final energy: {:.2}",
                    energy.current
                );
                tracked.entity = None; // Clear tracking
            }
            info!("Organism died at energy level: {:.2}", energy.current);
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
            let behavior_state = format!("{:?}", behavior.state);
            let organism_type = format!("{:?}", org_type);
            let (target_x, target_y) = behavior
                .target_position
                .map(|pos| (pos.x, pos.y))
                .unwrap_or((f32::NAN, f32::NAN));
            let target_entity = behavior
                .target_entity
                .map(|entity| entity.index().to_string())
                .unwrap_or_else(|| "None".to_string());
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

            writeln!(
                writer,
                "{tick},{entity},{pos_x:.6},{pos_y:.6},{vel_x:.6},{vel_y:.6},{speed:.6},{energy_current:.6},{energy_max:.6},{energy_ratio:.6},{age},{size:.6},{organism_type},{behavior_state},{state_time:.6},{target_x:.6},{target_y:.6},{target_entity},{sensory_range:.6},{aggression:.6},{boldness:.6},{mutation_rate:.6},{reproduction_threshold:.6},{reproduction_cooldown:.6},{foraging_drive:.6},{risk_tolerance:.6},{exploration_drive:.6},{clutch_size:.6},{offspring_share:.6},{hunger_memory:.6},{threat_timer:.6},{resource_selectivity:.6},{migration_x:.6},{migration_y:.6},{migration_active}",
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
                target_entity = target_entity,
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

/// Log tracked organism information periodically
pub fn log_tracked_organism(
    tracked: ResMut<TrackedOrganism>,
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
    let mut tracked_mut = tracked;
    tracked_mut.log_counter += 1;

    // default cadence: every 10 ticks
    if tracked_mut.log_counter % 10 != 0 {
        return;
    }

    if let Some(entity) = tracked_mut.entity {
        if let Ok((
            _entity,
            position,
            velocity,
            energy,
            age,
            size,
            org_type,
            behavior,
            cached_traits,
        )) = query.get(entity)
        {
            let speed = velocity.0.length();
            let behavior_state = format!("{:?}", behavior.state);
            let sensory_range = cached_traits.sensory_range;
            let aggression = cached_traits.aggression;
            let boldness = cached_traits.boldness;
            let mutation_rate = cached_traits.mutation_rate;

            let target_info = if let Some(target_pos) = behavior.target_position {
                format!("({:.1}, {:.1})", target_pos.x, target_pos.y)
            } else {
                "None".to_string()
            };

            info!(
                "[TRACKED ORGANISM] Tick: {} | Pos: ({:.2}, {:.2}) | Vel: ({:.2}, {:.2}) | Speed: {:.2} | Energy: {:.2}/{:.2} ({:.1}%) | Age: {} | Size: {:.2} | Type: {:?} | Behavior: {} | StateTime: {:.1}s | Target: {} | SensoryRange: {:.1} | Aggression: {:.2} | Boldness: {:.2} | MutationRate: {:.4}",
                tracked_mut.log_counter,
                position.0.x,
                position.0.y,
                velocity.0.x,
                velocity.0.y,
                speed,
                energy.current,
                energy.max,
                energy.ratio() * 100.0,
                age.0,
                size.value(),
                org_type,
                behavior_state,
                behavior.state_time,
                target_info,
                sensory_range,
                aggression,
                boldness,
                mutation_rate,
            );

            let needs_header = !tracked_mut.header_written;
            let tick = tracked_mut.log_counter;

            if let Some(ref mut writer) = tracked_mut.csv_writer {
                if needs_header {
                    writeln!(
                        writer,
                        "tick,position_x,position_y,velocity_x,velocity_y,speed,energy_current,energy_max,energy_ratio,age,size,organism_type,behavior_state,state_time,target_x,target_y,target_entity,sensory_range,aggression,boldness,mutation_rate,foraging_drive,risk_tolerance,exploration_drive,clutch_size,offspring_energy_share,hunger_memory,threat_timer,resource_selectivity,migration_target_x,migration_target_y,migration_active"
                    )
                    .expect("Failed to write CSV header");
                }

                let (target_x, target_y) = if let Some(target_pos) = behavior.target_position {
                    (target_pos.x, target_pos.y)
                } else {
                    (f32::NAN, f32::NAN)
                };
                let target_entity = behavior
                    .target_entity
                    .map(|entity| entity.index().to_string())
                    .unwrap_or_else(|| "None".to_string());
                let (migration_x, migration_y) = behavior
                    .migration_target
                    .or(behavior.target_position)
                    .map(|pos| (pos.x, pos.y))
                    .unwrap_or((f32::NAN, f32::NAN));
                let migration_active = if behavior.state == BehaviorState::Migrating
                    || behavior.migration_target.is_some()
                {
                    1u8
                } else {
                    0u8
                };

                writeln!(
                    writer,
                    "{tick},{pos_x:.6},{pos_y:.6},{vel_x:.6},{vel_y:.6},{speed:.6},{energy_current:.6},{energy_max:.6},{energy_ratio:.6},{age},{size:.6},{organism_type:?},{behavior_state},{state_time:.6},{target_x:.6},{target_y:.6},{target_entity},{sensory_range:.6},{aggression:.6},{boldness:.6},{mutation_rate:.6},{foraging_drive:.6},{risk_tolerance:.6},{exploration_drive:.6},{clutch_size:.6},{offspring_share:.6},{hunger_memory:.6},{threat_timer:.6},{resource_selectivity:.6},{migration_x:.6},{migration_y:.6},{migration_active}",
                    tick = tick,
                    pos_x = position.0.x,
                    pos_y = position.0.y,
                    vel_x = velocity.0.x,
                    vel_y = velocity.0.y,
                    speed = speed,
                    energy_current = energy.current,
                    energy_max = energy.max,
                    energy_ratio = energy.ratio(),
                    age = age.0,
                    size = size.value(),
                    organism_type = org_type,
                    behavior_state = behavior_state,
                    state_time = behavior.state_time,
                    target_x = target_x,
                    target_y = target_y,
                    target_entity = target_entity,
                    sensory_range = sensory_range,
                    aggression = aggression,
                    boldness = boldness,
                    mutation_rate = mutation_rate,
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
                .expect("Failed to write CSV row");

                if tick % 100 == 0 {
                    writer.flush().expect("Failed to flush CSV writer");
                }
            }

            if needs_header {
                tracked_mut.header_written = true;
            }
        } else {
            info!("[TRACKED] Organism entity {:?} no longer exists", entity);
            tracked_mut.entity = None;

            if let Some(mut writer) = tracked_mut.csv_writer.take() {
                writer.flush().expect("Failed to flush CSV writer on close");
            }
        }
    }
}
