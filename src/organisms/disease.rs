use bevy::prelude::*;
use glam::Vec2;
use crate::organisms::components::{Position, Energy, SpeciesId, Alive, CachedTraits};
use std::collections::{HashMap, HashSet};

/// Disease system resource
#[derive(Resource, Debug)]
pub struct DiseaseSystem {
    /// Active diseases in the simulation
    pub active_diseases: Vec<Disease>,
    /// Disease resistance traits per species (evolved)
    pub species_resistance: HashMap<u32, f32>, // SpeciesId -> resistance (0.0 to 1.0)
    /// Total diseases spawned
    pub total_diseases: u32,
    /// Cooldown before next disease can spawn
    pub spawn_cooldown: f32,
}

impl Default for DiseaseSystem {
    fn default() -> Self {
        Self {
            active_diseases: Vec::new(),
            species_resistance: HashMap::new(),
            total_diseases: 0,
            spawn_cooldown: 800.0, // Initial cooldown
        }
    }
}

/// Resource-backed buffers for disease system to avoid allocations (optimization)
#[derive(Resource, Default)]
pub struct DiseaseSystemBuffers {
    /// Reusable buffer for new infections
    pub new_infections: Vec<(Entity, u32)>,
    /// Reusable HashMap for infected organisms by disease ID
    pub infected_by_disease: HashMap<u32, Vec<(Entity, Vec2)>>,
    /// Reusable HashSet for infected entities
    pub infected_entities: HashSet<Entity>,
    /// Reusable HashSet for tracking infections this tick
    pub infected_this_tick: HashSet<Entity>,
    /// Reusable Vec for entities to remove
    pub to_remove: Vec<Entity>,
    /// Reusable HashMap for species traits
    pub species_traits: HashMap<u32, Vec<f32>>,
}

impl DiseaseSystemBuffers {
    pub fn clear(&mut self) {
        self.new_infections.clear();
        for vec in self.infected_by_disease.values_mut() {
            vec.clear();
        }
        self.infected_entities.clear();
        self.infected_this_tick.clear();
        self.to_remove.clear();
        for vec in self.species_traits.values_mut() {
            vec.clear();
        }
    }
}

/// A disease that can spread between organisms
#[derive(Debug, Clone)]
pub struct Disease {
    /// Unique disease ID
    pub id: u32,
    /// Disease type
    pub disease_type: DiseaseType,
    /// Virulence (how quickly it spreads, 0.0 to 1.0)
    pub virulence: f32,
    /// Lethality (how much damage it does, 0.0 to 1.0)
    pub lethality: f32,
    /// Contagion radius (how far it can spread)
    pub contagion_radius: f32,
    /// Species specificity (if Some, only affects that species)
    pub target_species: Option<u32>,
    /// Time remaining before disease dies out
    pub time_remaining: f32,
    /// Duration of disease
    pub duration: f32,
}

/// Types of diseases
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DiseaseType {
    /// Viral disease - spreads quickly, moderate lethality
    Viral,
    /// Bacterial disease - spreads moderately, variable lethality
    Bacterial,
    /// Parasitic disease - spreads slowly, high lethality
    Parasitic,
    /// Fungal disease - spreads slowly, affects plants more
    Fungal,
}

/// Component indicating an organism is infected
#[derive(Component, Debug, Clone)]
pub struct Infected {
    /// Disease ID
    pub disease_id: u32,
    /// Time infected
    pub infection_time: f32,
    /// Damage accumulated
    pub damage_accumulated: f32,
}

/// Update disease system (spawn and spread diseases)
pub fn update_disease_system(
    mut commands: Commands,
    mut disease_system: ResMut<DiseaseSystem>,
    time: Res<Time>,
    organism_query: Query<(Entity, &Position, &SpeciesId), With<Alive>>,
    infected_query: Query<(Entity, &Position, &Infected), With<Alive>>,
    spatial_hash: Res<crate::utils::SpatialHashGrid>,
) {
    let dt = time.delta_seconds();

    // Update existing diseases
    for disease in &mut disease_system.active_diseases {
        disease.time_remaining -= dt;
    }

    // Remove expired diseases
    disease_system.active_diseases.retain(|d| d.time_remaining > 0.0);

    // Spawn new diseases
    disease_system.spawn_cooldown -= dt;
    if disease_system.spawn_cooldown <= 0.0 {
        if fastrand::f32() < 0.0005 { // Very rare
            spawn_random_disease(&mut disease_system, &organism_query);
        }
        disease_system.spawn_cooldown = fastrand::f32() * 1200.0 + 600.0;
    }

    // Spread diseases (read-only access to Infected)
    spread_diseases(
        &mut commands,
        &disease_system,
        &organism_query,
        &infected_query,
        &spatial_hash,
        &mut buffers,
        dt,
    );
}

/// Update infected organisms (apply damage) - separate system to avoid query conflicts
pub fn update_infected_organisms_system(
    mut commands: Commands,
    disease_system: Res<DiseaseSystem>,
    mut buffers: ResMut<DiseaseSystemBuffers>, // Optimization: reuse buffers
    mut infected_query: Query<(Entity, &mut Infected, &mut Energy, &SpeciesId), With<Alive>>,
    time: Res<Time>,
) {
    let dt = time.delta_seconds();
    update_infected_organisms(
        &mut commands,
        &disease_system,
        &mut infected_query,
        &mut buffers,
        dt,
    );
}

/// Spread diseases between organisms
fn spread_diseases(
    commands: &mut Commands,
    disease_system: &DiseaseSystem,
    organism_query: &Query<(Entity, &Position, &SpeciesId), With<Alive>>,
    infected_query: &Query<(Entity, &Position, &Infected), With<Alive>>,
    spatial_hash: &Res<crate::utils::SpatialHashGrid>,
    buffers: &mut DiseaseSystemBuffers, // Optimization: reuse buffers
    dt: f32,
) {
    // Clear and reuse buffers from previous frame
    buffers.clear();
    buffers.new_infections.clear();

    // Create a set of infected entities by disease ID for quick lookup
    for (entity, pos, infected) in infected_query.iter() {
        buffers.infected_by_disease
            .entry(infected.disease_id)
            .or_insert_with(Vec::new)
            .push((entity, Vec2::new(pos.x(), pos.y())));
    }

    // Create a set of all infected entities for quick lookup
    buffers.infected_entities.extend(
        infected_query.iter().map(|(entity, _, _)| entity)
    );

    for disease in &disease_system.active_diseases {
        // Get all organisms infected with this disease
        if let Some(infected_organisms) = buffers.infected_by_disease.get(&disease.id) {
            // For each infected organism, try to spread to nearby uninfected organisms
            for (infected_entity, infected_pos) in infected_organisms {
                let nearby_entities = spatial_hash.organisms.query_radius(*infected_pos, disease.contagion_radius);

                for nearby_entity in nearby_entities {
                    // Skip if it's the same entity
                    if *infected_entity == nearby_entity {
                        continue;
                    }

                    // Skip if already infected (any disease)
                    if buffers.infected_entities.contains(&nearby_entity) {
                        continue;
                    }

                    // Check if nearby organism exists and is alive
                    if let Ok((entity, position, species_id)) = organism_query.get(nearby_entity) {
                        // Check if disease targets this species (or no target)
                        if let Some(target_species) = disease.target_species {
                            if species_id.value() != target_species {
                                continue;
                            }
                        }

                        // Calculate infection chance
                        let distance = infected_pos.distance(Vec2::new(position.x(), position.y()));
                        if distance > disease.contagion_radius {
                            continue;
                        }

                        let distance_factor = 1.0 - (distance / disease.contagion_radius).min(1.0);
                        
                        // Get species resistance
                        let resistance = disease_system.species_resistance
                            .get(&species_id.value())
                            .copied()
                            .unwrap_or(0.5); // Default resistance
                        
                        // Infection probability
                        let infection_chance = disease.virulence * distance_factor * (1.0 - resistance) * dt * 0.1;
                        
                        if fastrand::f32() < infection_chance {
                            buffers.new_infections.push((entity, disease.id));
                            break; // Only one infection per disease per tick per organism
                        }
                    }
                }
            }
        }
    }

    // Apply new infections using commands (avoid duplicates)
    for (entity, disease_id) in &buffers.new_infections {
        if !buffers.infected_this_tick.contains(entity) {
            commands.entity(*entity).insert(Infected {
                disease_id: *disease_id,
                infection_time: 0.0,
                damage_accumulated: 0.0,
            });
            buffers.infected_this_tick.insert(*entity);
            info!("[DISEASE] Organism {:?} infected with disease {}", entity, disease_id);
        }
    }
}

/// Update infected organisms (apply damage)
fn update_infected_organisms(
    commands: &mut Commands,
    disease_system: &DiseaseSystem,
    infected_query: &mut Query<(Entity, &mut Infected, &mut Energy, &SpeciesId), With<Alive>>,
    buffers: &mut DiseaseSystemBuffers, // Optimization: reuse buffers
    dt: f32,
) {
    buffers.to_remove.clear();

    for (entity, mut infected, mut energy, species_id) in infected_query.iter_mut() {
        // Find disease
        if let Some(disease) = disease_system.active_diseases.iter()
            .find(|d| d.id == infected.disease_id) {
            
            // Get species resistance
            let resistance = disease_system.species_resistance
                .get(&species_id.value())
                .copied()
                .unwrap_or(0.5);

            // Apply damage
            let damage = disease.lethality * (1.0 - resistance) * dt * 0.05;
            energy.current = (energy.current - damage).max(0.0);
            infected.damage_accumulated += damage;
            infected.infection_time += dt;

            // Remove infection if organism dies, disease expires, or organism recovered
            if energy.current <= 0.0 {
                // Organism died - will be handled by death system
                buffers.to_remove.push(entity);
            } else if infected.infection_time > disease.duration {
                // Disease expired - organism recovered
                buffers.to_remove.push(entity);
                info!("[DISEASE] Organism {:?} recovered from disease {}", entity, disease.id);
            }
        } else {
            // Disease no longer exists - remove infection
            buffers.to_remove.push(entity);
        }
    }

    // Remove infection components
    for entity in &buffers.to_remove {
        commands.entity(*entity).remove::<Infected>();
    }
}

/// Spawn a random disease
fn spawn_random_disease(
    disease_system: &mut DiseaseSystem,
    organism_query: &Query<(Entity, &Position, &SpeciesId), With<Alive>>,
) {
    // Choose random disease type
    let disease_type = match fastrand::u8(..4) {
        0 => DiseaseType::Viral,
        1 => DiseaseType::Bacterial,
        2 => DiseaseType::Parasitic,
        _ => DiseaseType::Fungal,
    };

    // Set parameters based on type
    let (virulence, lethality, contagion_radius, duration) = match disease_type {
        DiseaseType::Viral => (0.8 + fastrand::f32() * 0.2, 0.4 + fastrand::f32() * 0.3, 15.0, 200.0),
        DiseaseType::Bacterial => (0.5 + fastrand::f32() * 0.3, 0.3 + fastrand::f32() * 0.5, 10.0, 300.0),
        DiseaseType::Parasitic => (0.2 + fastrand::f32() * 0.3, 0.7 + fastrand::f32() * 0.3, 8.0, 500.0),
        DiseaseType::Fungal => (0.3 + fastrand::f32() * 0.2, 0.5 + fastrand::f32() * 0.3, 12.0, 400.0),
    };

    // Optionally target a random species (50% chance)
    let target_species = if fastrand::f32() < 0.5 {
        // Get a random species from organisms
        let species_ids: Vec<u32> = organism_query.iter()
            .map(|(_, _, species_id)| species_id.value())
            .collect();
        if !species_ids.is_empty() {
            Some(species_ids[fastrand::usize(..species_ids.len())])
        } else {
            None
        }
    } else {
        None
    };

    let disease_id = disease_system.total_diseases;
    let disease = Disease {
        id: disease_id,
        disease_type,
        virulence,
        lethality,
        contagion_radius,
        target_species,
        time_remaining: duration,
        duration,
    };

    disease_system.active_diseases.push(disease);
    disease_system.total_diseases += 1;

    info!("[DISEASE] {:?} disease spawned (virulence: {:.2}, lethality: {:.2})", 
        disease_type, virulence, lethality);
}

/// Update species resistance based on evolution (called from genetics system)
pub fn update_species_resistance(
    mut disease_system: ResMut<DiseaseSystem>,
    mut buffers: ResMut<DiseaseSystemBuffers>, // Optimization: reuse buffers
    _species_tracker: Res<crate::organisms::speciation::SpeciesTracker>,
    organism_query: Query<(&SpeciesId, &CachedTraits)>,
) {
    // Update resistance for each species based on average disease resistance trait
    // This would be calculated from genome traits in a real implementation
    // For now, we'll use a placeholder that gets resistance from cached traits
    
    // Clear and reuse buffer
    for vec in buffers.species_traits.values_mut() {
        vec.clear();
    }
    
    // Group organisms by species
    for (species_id, traits) in organism_query.iter() {
        let species_id_val = species_id.value();
        // Use a trait index for disease resistance (would need to add to CachedTraits)
        // For now, use risk_tolerance as a proxy (higher risk tolerance = lower resistance)
        let resistance_proxy = 1.0 - traits.risk_tolerance;
        buffers.species_traits
            .entry(species_id_val)
            .or_insert_with(Vec::new)
            .push(resistance_proxy);
    }

    // Calculate average resistance per species
    for (species_id, resistances) in &buffers.species_traits {
        if !resistances.is_empty() {
            let avg_resistance = resistances.iter().sum::<f32>() / resistances.len() as f32;
            disease_system.species_resistance.insert(*species_id, avg_resistance);
        }
    }
}
