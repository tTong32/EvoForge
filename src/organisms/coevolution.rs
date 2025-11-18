use bevy::prelude::*;
use crate::organisms::components::{SpeciesId, OrganismType, CachedTraits};
use std::collections::HashMap;

/// Co-evolution system resource tracking species interactions
#[derive(Resource, Debug)]
pub struct CoEvolutionSystem {
    /// Predator-prey relationships (predator_species -> prey_species -> interaction strength)
    pub predator_prey: HashMap<(u32, u32), InteractionStrength>,
    /// Parasite-host relationships (parasite_species -> host_species -> interaction strength)
    pub parasite_host: HashMap<(u32, u32), InteractionStrength>,
    /// Mutualistic relationships (species_a -> species_b -> interaction strength)
    pub mutualistic: HashMap<(u32, u32), InteractionStrength>,
    /// Competitive relationships (species_a -> species_b -> competition strength)
    pub competitive: HashMap<(u32, u32), InteractionStrength>,
    /// Evolved defenses per species (species_id -> defense traits)
    pub species_defenses: HashMap<u32, DefenseTraits>,
    /// Co-evolution pressure tracking (for logging/analysis)
    pub evolution_pressure: HashMap<u32, EvolutionPressure>,
}

impl Default for CoEvolutionSystem {
    fn default() -> Self {
        Self {
            predator_prey: HashMap::new(),
            parasite_host: HashMap::new(),
            mutualistic: HashMap::new(),
            competitive: HashMap::new(),
            species_defenses: HashMap::new(),
            evolution_pressure: HashMap::new(),
        }
    }
}

/// Resource-backed buffers for coevolution system to avoid allocations (optimization)
#[derive(Resource, Default)]
pub struct CoEvolutionSystemBuffers {
    /// Reusable HashMap for grouping organisms by species
    pub species_groups: HashMap<u32, Vec<(OrganismType, CachedTraits)>>,
    /// Reusable Vec for species IDs
    pub species_ids: Vec<u32>,
    /// Reusable HashMap for species defense needs
    pub species_defense_needs: HashMap<u32, DefenseTraits>,
    /// Reusable HashMap for species counts
    pub species_counts: HashMap<u32, u32>,
}

impl CoEvolutionSystemBuffers {
    pub fn clear(&mut self) {
        for vec in self.species_groups.values_mut() {
            vec.clear();
        }
        self.species_ids.clear();
        self.species_defense_needs.clear();
        self.species_counts.clear();
    }
}

/// Strength of an interaction between species
#[derive(Debug, Clone, Copy)]
pub struct InteractionStrength {
    /// Current strength (0.0 to 1.0)
    pub strength: f32,
    /// How much it has changed over time (for tracking evolution)
    pub change_rate: f32,
    /// Number of interactions observed
    pub interaction_count: u32,
}

impl Default for InteractionStrength {
    fn default() -> Self {
        Self {
            strength: 0.5,
            change_rate: 0.0,
            interaction_count: 0,
        }
    }
}

/// Defense traits that have evolved in response to predators/parasites
#[derive(Debug, Clone)]
pub struct DefenseTraits {
    /// Physical defense (armor, spines, etc.)
    pub physical_defense: f32,
    /// Chemical defense (toxins, poisons)
    pub chemical_defense: f32,
    /// Behavioral defense (camouflage, mimicry)
    pub behavioral_defense: f32,
    /// Speed/escape capability
    pub escape_capability: f32,
    /// Immunity/resistance to diseases
    pub disease_resistance: f32,
}

impl Default for DefenseTraits {
    fn default() -> Self {
        Self {
            physical_defense: 0.5,
            chemical_defense: 0.5,
            behavioral_defense: 0.5,
            escape_capability: 0.5,
            disease_resistance: 0.5,
        }
    }
}

/// Evolution pressure on a species
#[derive(Debug, Clone)]
pub struct EvolutionPressure {
    /// Predation pressure (how much predators affect this species)
    pub predation_pressure: f32,
    /// Competition pressure (how much competition affects this species)
    pub competition_pressure: f32,
    /// Disease pressure (how much diseases affect this species)
    pub disease_pressure: f32,
    /// Resource pressure (scarcity of resources)
    pub resource_pressure: f32,
}

impl Default for EvolutionPressure {
    fn default() -> Self {
        Self {
            predation_pressure: 0.0,
            competition_pressure: 0.0,
            disease_pressure: 0.0,
            resource_pressure: 0.0,
        }
    }
}

/// Update co-evolution system
pub fn update_coevolution_system(
    mut coevolution: ResMut<CoEvolutionSystem>,
    mut buffers: ResMut<CoEvolutionSystemBuffers>, // Optimization: reuse buffers
    organism_query: Query<(&SpeciesId, &OrganismType, &CachedTraits), With<crate::organisms::components::Alive>>,
    time: Res<Time>,
) {
    let dt = time.delta_seconds();
    
    // Clear buffers for reuse
    buffers.clear();

    // Detect and update species interactions
    detect_species_interactions(&mut coevolution, &mut buffers, &organism_query, dt);

    // Update evolved defenses based on interactions
    update_evolved_defenses(&mut coevolution, &mut buffers, &organism_query, dt);

    // Update evolution pressure
    update_evolution_pressure(&mut coevolution, &mut buffers, &organism_query, dt);
}

/// Detect species interactions based on organism traits and proximity
fn detect_species_interactions(
    coevolution: &mut CoEvolutionSystem,
    buffers: &mut CoEvolutionSystemBuffers, // Optimization: reuse buffers
    organism_query: &Query<(&SpeciesId, &OrganismType, &CachedTraits), With<crate::organisms::components::Alive>>,
    dt: f32,
) {
    // Group organisms by species (reuse buffer)
    for (species_id, org_type, traits) in organism_query.iter() {
        buffers.species_groups
            .entry(species_id.value())
            .or_insert_with(Vec::new)
            .push((*org_type, traits.clone()));
    }

    buffers.species_ids.clear();
    buffers.species_ids.extend(buffers.species_groups.keys().copied());

    // Check for predator-prey relationships
    for &predator_id in &buffers.species_ids {
        for &prey_id in &buffers.species_ids {
            if predator_id == prey_id {
                continue;
            }

            let predator_group = &buffers.species_groups[&predator_id];
            let prey_group = &buffers.species_groups[&prey_id];

            // Check if predator-prey relationship is likely
            if is_predator_prey_relationship(predator_group, prey_group) {
                let key = (predator_id, prey_id);
                let interaction = coevolution.predator_prey
                    .entry(key)
                    .or_insert_with(InteractionStrength::default);
                
                // Increase interaction strength based on population sizes and traits
                let predator_count = predator_group.len() as f32;
                let prey_count = prey_group.len() as f32;
                let interaction_rate = (predator_count * prey_count / 1000.0).min(1.0) * dt * 0.01;
                interaction.strength = (interaction.strength + interaction_rate).min(1.0);
                interaction.interaction_count += 1;
            }
        }
    }

    // Check for competitive relationships (same trophic level)
    for i in 0..buffers.species_ids.len() {
        for j in (i + 1)..buffers.species_ids.len() {
            let species_a = buffers.species_ids[i];
            let species_b = buffers.species_ids[j];

            let group_a = &buffers.species_groups[&species_a];
            let group_b = &buffers.species_groups[&species_b];

            // Check if competitive relationship is likely
            if is_competitive_relationship(group_a, group_b) {
                let key = (species_a, species_b);
                let interaction = coevolution.competitive
                    .entry(key)
                    .or_insert_with(InteractionStrength::default);
                
                // Increase competition strength based on overlap
                let competition_rate = (group_a.len().min(group_b.len()) as f32 / 500.0).min(1.0) * dt * 0.01;
                interaction.strength = (interaction.strength + competition_rate).min(1.0);
                interaction.interaction_count += 1;
            }
        }
    }

    // Check for mutualistic relationships (different trophic levels that benefit each other)
    for &species_a in &species_ids {
        for &species_b in &species_ids {
            if species_a >= species_b {
                continue;
            }

            let group_a = &species_groups[&species_a];
            let group_b = &species_groups[&species_b];

            // Check if mutualistic relationship is likely
            if is_mutualistic_relationship(group_a, group_b) {
                let key = (species_a, species_b);
                let interaction = coevolution.mutualistic
                    .entry(key)
                    .or_insert_with(InteractionStrength::default);
                
                // Increase mutualistic strength
                let mutualism_rate = ((group_a.len() + group_b.len()) as f32 / 1000.0).min(1.0) * dt * 0.005;
                interaction.strength = (interaction.strength + mutualism_rate).min(1.0);
                interaction.interaction_count += 1;
            }
        }
    }
}

/// Check if a predator-prey relationship is likely
fn is_predator_prey_relationship(
    predator_group: &[(OrganismType, CachedTraits)],
    prey_group: &[(OrganismType, CachedTraits)],
) -> bool {
    // Check if predator is consumer and prey is producer or smaller consumer
    let predator_is_consumer = predator_group.iter()
        .any(|(org_type, _)| matches!(org_type, OrganismType::Consumer));
    
    let prey_is_producer = prey_group.iter()
        .any(|(org_type, _)| matches!(org_type, OrganismType::Producer));
    
    let prey_is_smaller = predator_group.iter()
        .any(|(_, traits)| {
            prey_group.iter()
                .any(|(_, prey_traits)| traits.size > prey_traits.size * 1.5)
        });

    predator_is_consumer && (prey_is_producer || prey_is_smaller)
}

/// Check if a competitive relationship is likely
fn is_competitive_relationship(
    group_a: &[(OrganismType, CachedTraits)],
    group_b: &[(OrganismType, CachedTraits)],
) -> bool {
    // Same trophic level and similar size
    let same_trophic = group_a.iter().any(|(org_type_a, _)| {
        group_b.iter().any(|(org_type_b, _)| org_type_a == org_type_b)
    });

    let similar_size = group_a.iter().any(|(_, traits_a)| {
        group_b.iter().any(|(_, traits_b)| {
            (traits_a.size - traits_b.size).abs() < traits_a.size * 0.5
        })
    });

    same_trophic && similar_size
}

/// Check if a mutualistic relationship is likely
fn is_mutualistic_relationship(
    group_a: &[(OrganismType, CachedTraits)],
    group_b: &[(OrganismType, CachedTraits)],
) -> bool {
    // Different trophic levels (e.g., producer-consumer mutualism)
    let different_trophic = group_a.iter().any(|(org_type_a, _)| {
        group_b.iter().any(|(org_type_b, _)| org_type_a != org_type_b)
    });

    // Complementary traits (e.g., one provides resources, other provides protection)
    different_trophic && group_a.len() > 0 && group_b.len() > 0
}

/// Update evolved defenses based on interactions
fn update_evolved_defenses(
    coevolution: &mut CoEvolutionSystem,
    buffers: &mut CoEvolutionSystemBuffers, // Optimization: reuse buffers
    _organism_query: &Query<(&SpeciesId, &OrganismType, &CachedTraits), With<crate::organisms::components::Alive>>,
    dt: f32,
) {
    // For each species, calculate defense needs based on interactions (reuse buffer)
    buffers.species_defense_needs.clear();

    // Calculate predation pressure per species
    for (prey_species, _) in &coevolution.predator_prey {
        let (_, prey_id) = *prey_species;
        let defenses = buffers.species_defense_needs
            .entry(prey_id)
            .or_insert_with(DefenseTraits::default);

        // Increase escape capability and physical defense in response to predation
        let predation_strength = coevolution.predator_prey
            .get(prey_species)
            .map(|i| i.strength)
            .unwrap_or(0.0);

        defenses.escape_capability += predation_strength * dt * 0.01;
        defenses.physical_defense += predation_strength * dt * 0.005;
        defenses.escape_capability = defenses.escape_capability.min(1.0);
        defenses.physical_defense = defenses.physical_defense.min(1.0);
    }

    // Calculate parasite pressure per species
    for (parasite_species, _) in &coevolution.parasite_host {
        let (_, host_id) = *parasite_species;
        let defenses = buffers.species_defense_needs
            .entry(host_id)
            .or_insert_with(DefenseTraits::default);

        // Increase disease resistance and chemical defense in response to parasites
        let parasite_strength = coevolution.parasite_host
            .get(parasite_species)
            .map(|i| i.strength)
            .unwrap_or(0.0);

        defenses.disease_resistance += parasite_strength * dt * 0.01;
        defenses.chemical_defense += parasite_strength * dt * 0.005;
        defenses.disease_resistance = defenses.disease_resistance.min(1.0);
        defenses.chemical_defense = defenses.chemical_defense.min(1.0);
    }

    // Update species defenses (evolution happens gradually)
    for (species_id, defenses) in &buffers.species_defense_needs {
        let current_defenses = coevolution.species_defenses
            .entry(*species_id)
            .or_insert_with(DefenseTraits::default);

        // Gradually evolve defenses toward needed levels
        let evolution_rate = 0.001 * dt; // Slow evolution
        current_defenses.physical_defense = lerp(
            current_defenses.physical_defense,
            defenses.physical_defense,
            evolution_rate,
        );
        current_defenses.chemical_defense = lerp(
            current_defenses.chemical_defense,
            defenses.chemical_defense,
            evolution_rate,
        );
        current_defenses.behavioral_defense = lerp(
            current_defenses.behavioral_defense,
            defenses.behavioral_defense,
            evolution_rate,
        );
        current_defenses.escape_capability = lerp(
            current_defenses.escape_capability,
            defenses.escape_capability,
            evolution_rate,
        );
        current_defenses.disease_resistance = lerp(
            current_defenses.disease_resistance,
            defenses.disease_resistance,
            evolution_rate,
        );
    }
}

/// Update evolution pressure tracking
fn update_evolution_pressure(
    coevolution: &mut CoEvolutionSystem,
    buffers: &mut CoEvolutionSystemBuffers, // Optimization: reuse buffers
    organism_query: &Query<(&SpeciesId, &OrganismType, &CachedTraits), With<crate::organisms::components::Alive>>,
    _dt: f32,
) {
    // Group organisms by species (reuse buffer)
    buffers.species_counts.clear();
    
    for (species_id, _, _) in organism_query.iter() {
        *buffers.species_counts.entry(species_id.value()).or_insert(0) += 1;
    }

    // Calculate pressure for each species
    for species_id in buffers.species_counts.keys() {
        let pressure = coevolution.evolution_pressure
            .entry(*species_id)
            .or_insert_with(EvolutionPressure::default);

        // Predation pressure
        let predation_count: u32 = coevolution.predator_prey
            .iter()
            .filter(|((_, prey_id), interaction)| {
                *prey_id == *species_id && interaction.strength > 0.1
            })
            .map(|(_, interaction)| interaction.interaction_count)
            .sum();
        pressure.predation_pressure = (predation_count as f32 / 1000.0).min(1.0);

        // Competition pressure
        let competition_count: u32 = coevolution.competitive
            .iter()
            .filter(|((species_a, species_b), interaction)| {
                (*species_a == *species_id || *species_b == *species_id) && interaction.strength > 0.1
            })
            .map(|(_, interaction)| interaction.interaction_count)
            .sum();
        pressure.competition_pressure = (competition_count as f32 / 1000.0).min(1.0);
    }
}

/// Linear interpolation helper
fn lerp(a: f32, b: f32, t: f32) -> f32 {
    a + (b - a) * t
}

/// Get defense traits for a species (for use in other systems)
pub fn get_species_defenses(
    coevolution: &CoEvolutionSystem,
    species_id: u32,
) -> DefenseTraits {
    coevolution.species_defenses
        .get(&species_id)
        .cloned()
        .unwrap_or_default()
}

/// Get interaction strength between two species
pub fn get_interaction_strength(
    coevolution: &CoEvolutionSystem,
    species_a: u32,
    species_b: u32,
) -> f32 {
    // Check predator-prey
    if let Some(interaction) = coevolution.predator_prey.get(&(species_a, species_b)) {
        return interaction.strength;
    }
    if let Some(interaction) = coevolution.predator_prey.get(&(species_b, species_a)) {
        return -interaction.strength; // Negative for reverse direction
    }

    // Check mutualistic
    if let Some(interaction) = coevolution.mutualistic.get(&(species_a.min(species_b), species_a.max(species_b))) {
        return interaction.strength;
    }

    // Check competitive
    if let Some(interaction) = coevolution.competitive.get(&(species_a.min(species_b), species_a.max(species_b))) {
        return -interaction.strength; // Negative for competition
    }

    0.0 // No interaction
}
