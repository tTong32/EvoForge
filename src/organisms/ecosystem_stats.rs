use crate::organisms::components::*;
use bevy::prelude::*;
use std::collections::HashMap;

/// Ecosystem statistics for Step 8 - Tuning and analysis
#[derive(Resource, Default)]
pub struct EcosystemStats {
    /// Total population count
    pub total_population: u32,
    /// Population by organism type
    pub population_by_type: HashMap<OrganismType, u32>,
    /// Population by species
    pub population_by_species: HashMap<u32, u32>,
    /// Average traits per species
    pub species_traits: HashMap<u32, SpeciesTraits>,
    /// Tick counter for logging
    pub tick_counter: u64,
}

#[derive(Default)]
pub struct SpeciesTraits {
    pub avg_size: f32,
    pub avg_energy: f32,
    pub avg_speed: f32,
    pub avg_sensory_range: f32,
    pub count: u32,
}

impl EcosystemStats {
    pub fn reset(&mut self) {
        self.total_population = 0;
        self.population_by_type.clear();
        self.population_by_species.clear();
        self.species_traits.clear();
    }
}

/// Collect ecosystem statistics periodically (Step 8 - Ecosystem tuning)
pub fn collect_ecosystem_stats(
    mut stats: ResMut<EcosystemStats>,
    query: Query<
        (
            &SpeciesId,
            &OrganismType,
            &Size,
            &Energy,
            &CachedTraits,
        ),
        With<Alive>,
    >,
    species_tracker: Option<Res<crate::organisms::speciation::SpeciesTracker>>,
) {
    stats.tick_counter += 1;
    
    // Collect stats every 100 ticks (not every tick for performance)
    if stats.tick_counter % 100 != 0 {
        return;
    }

    stats.reset();

    let mut species_trait_data: HashMap<u32, (f32, f32, f32, f32, u32)> = HashMap::new();

    for (species_id, org_type, size, energy, traits) in query.iter() {
        stats.total_population += 1;
        
        // Count by type
        *stats.population_by_type.entry(*org_type).or_insert(0) += 1;
        
        // Count by species
        let species_id_val = species_id.value();
        *stats.population_by_species.entry(species_id_val).or_insert(0) += 1;
        
        // Accumulate trait data per species
        let entry = species_trait_data.entry(species_id_val).or_insert((0.0, 0.0, 0.0, 0.0, 0));
        entry.0 += size.value();
        entry.1 += energy.current;
        entry.2 += traits.speed;
        entry.3 += traits.sensory_range;
        entry.4 += 1;
    }

    // Calculate averages
    for (species_id, (size_sum, energy_sum, speed_sum, sensory_sum, count)) in species_trait_data {
        if count > 0 {
            stats.species_traits.insert(
                species_id,
                SpeciesTraits {
                    avg_size: size_sum / count as f32,
                    avg_energy: energy_sum / count as f32,
                    avg_speed: speed_sum / count as f32,
                    avg_sensory_range: sensory_sum / count as f32,
                    count,
                },
            );
        }
    }

    // Log ecosystem summary every 500 ticks
    if stats.tick_counter % 500 == 0 {
        let species_count = species_tracker
            .map(|t| t.species_count())
            .unwrap_or(0);
        
        let consumers = stats.population_by_type.get(&OrganismType::Consumer).copied().unwrap_or(0);

        info!(
            "[ECOSYSTEM] Tick {} | Population: {} | Species: {} | Consumers: {}",
            stats.tick_counter,
            stats.total_population,
            species_count,
            consumers,
        );
    }
}

