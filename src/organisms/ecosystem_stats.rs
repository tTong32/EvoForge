use crate::organisms::components::*;
use crate::organisms::genetics::{Genome, GENOME_SIZE};
use crate::organisms::speciation::SpeciesTracker;
use crate::world::cell::{Cell, TerrainType, RESOURCE_TYPE_COUNT};
use crate::world::{WorldGrid, ClimateState};
use bevy::prelude::*;
use std::collections::{HashMap, VecDeque};
use std::fs::{File, OpenOptions};
use std::io::{BufWriter, Write};
use std::path::PathBuf;

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

#[derive(Default)]
struct EnvironmentAggregate {
    temperature_samples: Vec<f32>,
    humidity_samples: Vec<f32>,
    elevation_samples: Vec<f32>,
    terrain_counts: HashMap<TerrainType, u32>,
    resource_samples: [Vec<f32>; RESOURCE_TYPE_COUNT],
    sample_count: u32,
}

impl EnvironmentAggregate {
    fn new() -> Self {
        Self {
            temperature_samples: Vec::new(),
            humidity_samples: Vec::new(),
            elevation_samples: Vec::new(),
            terrain_counts: HashMap::new(),
            resource_samples: [
                Vec::new(), Vec::new(), Vec::new(),
                Vec::new(), Vec::new(), Vec::new(),
            ],
            sample_count: 0,
        }
    }

    fn add_sample(
        &mut self,
        temperature: f32,
        humidity: f32,
        elevation: f32,
        terrain: TerrainType,
        resource_density: &[f32; RESOURCE_TYPE_COUNT],
    ) {
        self.temperature_samples.push(temperature);
        self.humidity_samples.push(humidity);
        self.elevation_samples.push(elevation);
        *self.terrain_counts.entry(terrain).or_insert(0) += 1;
        
        for (i, &density) in resource_density.iter().enumerate() {
            self.resource_samples[i].push(density);
        }
        
        self.sample_count += 1;
    }

    /// Calculate statistics from collected samples
    fn calculate_stats(&self) -> EnvironmentStats {
        let count = self.sample_count as f32;
        if count == 0.0 {
            return EnvironmentStats::default();
        }

        // Helper to calculate standard deviation
        let calc_std = |samples: &[f32], mean: f32| -> f32 {
            if samples.len() < 2 {
                return 0.0;
            }
            let variance = samples.iter()
                .map(|&x| (x - mean).powi(2))
                .sum::<f32>() / (samples.len() as f32 - 1.0);
            variance.sqrt()
        };

        // Temperature stats
        let temp_avg = self.temperature_samples.iter().sum::<f32>() / count;
        let temp_min = self.temperature_samples.iter().copied()
            .fold(f32::INFINITY, f32::min);
        let temp_max = self.temperature_samples.iter().copied()
            .fold(f32::NEG_INFINITY, f32::max);
        let temp_std = calc_std(&self.temperature_samples, temp_avg);

        // Humidity stats
        let humidity_avg = self.humidity_samples.iter().sum::<f32>() / count;
        let humidity_min = self.humidity_samples.iter().copied()
            .fold(f32::INFINITY, f32::min);
        let humidity_max = self.humidity_samples.iter().copied()
            .fold(f32::NEG_INFINITY, f32::max);
        let humidity_std = calc_std(&self.humidity_samples, humidity_avg);

        // Elevation stats
        let elevation_avg = self.elevation_samples.iter().sum::<f32>() / count;
        let elevation_min = self.elevation_samples.iter().copied()
            .fold(f32::INFINITY, f32::min);
        let elevation_max = self.elevation_samples.iter().copied()
            .fold(f32::NEG_INFINITY, f32::max);

        // Terrain percentages
        let total_terrain = self.terrain_counts.values().sum::<u32>() as f32;
        let terrain_pct = |terrain: TerrainType| -> f32 {
            if total_terrain > 0.0 {
                *self.terrain_counts.get(&terrain).unwrap_or(&0) as f32 / total_terrain
            } else {
                0.0
            }
        };

        // Resource averages
        let resource_avg = |samples: &[f32]| -> f32 {
            if samples.is_empty() {
                0.0
            } else {
                samples.iter().sum::<f32>() / samples.len() as f32
            }
        };

        EnvironmentStats {
            temp_avg,
            temp_min,
            temp_max,
            temp_std,
            humidity_avg,
            humidity_min,
            humidity_max,
            humidity_std,
            elevation_avg,
            elevation_min,
            elevation_max,
            terrain_ocean_pct: terrain_pct(TerrainType::Ocean),
            terrain_plains_pct: terrain_pct(TerrainType::Plains),
            terrain_forest_pct: terrain_pct(TerrainType::Forest),
            terrain_desert_pct: terrain_pct(TerrainType::Desert),
            terrain_tundra_pct: terrain_pct(TerrainType::Tundra),
            terrain_mountain_pct: terrain_pct(TerrainType::Mountain),
            terrain_swamp_pct: terrain_pct(TerrainType::Swamp),
            terrain_volcanic_pct: terrain_pct(TerrainType::Volcanic),
            resource_plant_avg: resource_avg(&self.resource_samples[0]),
            resource_mineral_avg: resource_avg(&self.resource_samples[1]),
            resource_sunlight_avg: resource_avg(&self.resource_samples[2]),
            resource_water_avg: resource_avg(&self.resource_samples[3]),
            resource_detritus_avg: resource_avg(&self.resource_samples[4]),
            resource_prey_avg: resource_avg(&self.resource_samples[5]),
        }
    }
}

/// Calculated environment statistics for a species
#[derive(Default, Clone, Copy)]
struct EnvironmentStats {
    temp_avg: f32,
    temp_min: f32,
    temp_max: f32,
    temp_std: f32,
    humidity_avg: f32,
    humidity_min: f32,
    humidity_max: f32,
    humidity_std: f32,
    elevation_avg: f32,
    elevation_min: f32,
    elevation_max: f32,
    terrain_ocean_pct: f32,
    terrain_plains_pct: f32,
    terrain_forest_pct: f32,
    terrain_desert_pct: f32,
    terrain_tundra_pct: f32,
    terrain_mountain_pct: f32,
    terrain_swamp_pct: f32,
    terrain_volcanic_pct: f32,
    resource_plant_avg: f32,
    resource_mineral_avg: f32,
    resource_sunlight_avg: f32,
    resource_water_avg: f32,
    resource_detritus_avg: f32,
    resource_prey_avg: f32,
}

/// Logger for species-level fitness data (for AI model training)
#[derive(Resource)]
pub struct SpeciesFitnessLogger {
    csv_writer: Option<BufWriter<File>>,
    csv_path: PathBuf,
    tick_counter: u64,
    header_written: bool,
    log_interval: u64,
    population_history: HashMap<u32, VecDeque<(u64, u32)>>, // species_id -> (tick, population)
}

impl Default for SpeciesFitnessLogger {
    fn default() -> Self {
        let logs_dir = PathBuf::from("data/logs");
        if !logs_dir.exists() {
            std::fs::create_dir_all(&logs_dir).expect("Failed to create logs directory");
        }
        
        let timestamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs();
        let csv_path = logs_dir.join(format!("species_fitness_{}.csv", timestamp));

        Self {
            csv_writer: None,
            csv_path,
            tick_counter: 0,
            header_written: false,
            log_interval: 50, // Log every 50 ticks
            population_history: HashMap::new(),
        }
    }
}

impl SpeciesFitnessLogger {
    fn ensure_writer(&mut self) -> Option<&mut BufWriter<File>> {
        if self.csv_writer.is_none() {
            let file = OpenOptions::new()
                .create(true)
                .append(true)
                .open(&self.csv_path)
                .ok()?;
            self.csv_writer = Some(BufWriter::new(file));
        }
        self.csv_writer.as_mut()
    }
}

/// Log species-level fitness data for AI model training
pub fn log_species_fitness(
    mut logger: ResMut<SpeciesFitnessLogger>,
    stats: Res<EcosystemStats>,
    species_tracker: Res<SpeciesTracker>,
    world_grid: Res<WorldGrid>,
    organism_query: Query<(&Position, &SpeciesId), With<Alive>>,
    _climate: Res<ClimateState>,
) {
    logger.tick_counter += 1;
    
    // Only log at specified interval
    if logger.tick_counter % logger.log_interval != 0 {
        return;
    }

    // Aggregate environment data per species
    let mut species_env_data: HashMap<u32, EnvironmentAggregate> = HashMap::new();

    for (position, species_id) in organism_query.iter() {
        if let Some(cell) = world_grid.get_cell(position.x(), position.y()) {
            let env = species_env_data
                .entry(species_id.value())
                .or_insert_with(EnvironmentAggregate::new);
            
            env.add_sample(
                cell.temperature,
                cell.humidity,
                cell.elevation as f32,
                cell.terrain,
                &cell.resource_density,
            );
        }
    }

    // Step 2: Calculate environment statistics for each species
    let mut species_env_stats: HashMap<u32, EnvironmentStats> = HashMap::new();
    for (species_id, aggregate) in species_env_data {
        species_env_stats.insert(species_id, aggregate.calculate_stats());
    }

    // Step 3: Update population history and calculate growth rates
    let mut population_growth_rates: HashMap<u32, f32> = HashMap::new();
    let current_tick = logger.tick_counter;

    for (species_id, &current_pop) in stats.population_by_species.iter() {
        // Update history
        let history = logger.population_history
            .entry(*species_id)
            .or_insert_with(VecDeque::new);
        
        history.push_back((current_tick, current_pop));
        
        // Keep only last 100 samples (adjust as needed)
        while history.len() > 100 {
            history.pop_front();
        }

        // Calculate growth rate
        let growth_rate = if history.len() >= 2 {
            let (old_tick, old_pop) = history.front().unwrap();
            let (new_tick, new_pop) = history.back().unwrap();
            
            if *old_pop > 0 && *new_tick > *old_tick {
                let time_diff = (*new_tick - *old_tick) as f32;
                let pop_diff = *new_pop as f32 - *old_pop as f32;
                // Growth rate per tick
                pop_diff / (*old_pop as f32) / time_diff
            } else {
                0.0
            }
        } else {
            0.0
        };
        
        population_growth_rates.insert(*species_id, growth_rate);
    }

    // Step 4: Get writer and write CSV header if needed
    let writer = match logger.ensure_writer() {
        Some(w) => w,
        None => {
            eprintln!("Failed to create CSV writer for species fitness logging");
            return;
        }
    };

    if !logger.header_written {
        // Write CSV header
        write!(writer, "tick,species_id,").unwrap();
        
        // Gene columns
        for i in 0..GENOME_SIZE {
            write!(writer, "gene_{},", i).unwrap();
        }
        
        // Environment columns
        write!(writer, "env_temp_avg,env_temp_min,env_temp_max,env_temp_std,").unwrap();
        write!(writer, "env_humidity_avg,env_humidity_min,env_humidity_max,env_humidity_std,").unwrap();
        write!(writer, "env_elevation_avg,env_elevation_min,env_elevation_max,").unwrap();
        write!(writer, "terrain_ocean_pct,terrain_plains_pct,terrain_forest_pct,terrain_desert_pct,").unwrap();
        write!(writer, "terrain_tundra_pct,terrain_mountain_pct,terrain_swamp_pct,terrain_volcanic_pct,").unwrap();
        write!(writer, "resource_plant_avg,resource_mineral_avg,resource_sunlight_avg,").unwrap();
        write!(writer, "resource_water_avg,resource_detritus_avg,resource_prey_avg,").unwrap();
        
        // Ecosystem columns
        write!(writer, "ecosystem_total_population,ecosystem_species_count,").unwrap();
        
        // Fitness/label columns
        write!(writer, "current_population,population_growth_rate\n").unwrap();
        
        logger.header_written = true;
    }

    // Step 5: Write data for each species
    let ecosystem_total_pop = stats.total_population;
    let ecosystem_species_count = species_tracker.species_count();

    // Get all species that have data
    let all_species: Vec<u32> = stats.population_by_species.keys().copied().collect();

    for species_id in all_species {
        // Get genome from tracker
        let genome = match species_tracker.get_genome(species_id) {
            Some(g) => g,
            None => continue, // Skip species without genome data
        };

        // Get environment stats (default if species has no organisms in this tick)
        let env_stats = species_env_stats.get(&species_id)
            .copied()
            .unwrap_or_default();

        // Get population data
        let current_pop = stats.population_by_species.get(&species_id).copied().unwrap_or(0);
        let growth_rate = population_growth_rates.get(&species_id).copied().unwrap_or(0.0);

        // Write CSV row
        write!(writer, "{},{}", current_tick, species_id).unwrap();
        
        // Write genes
        for i in 0..GENOME_SIZE {
            write!(writer, ",{:.6}", genome.get_gene(i)).unwrap();
        }
        
        // Write environment stats
        write!(writer, ",{:.6},{:.6},{:.6},{:.6}", 
            env_stats.temp_avg, env_stats.temp_min, env_stats.temp_max, env_stats.temp_std).unwrap();
        write!(writer, ",{:.6},{:.6},{:.6},{:.6}",
            env_stats.humidity_avg, env_stats.humidity_min, env_stats.humidity_max, env_stats.humidity_std).unwrap();
        write!(writer, ",{:.6},{:.6},{:.6}",
            env_stats.elevation_avg, env_stats.elevation_min, env_stats.elevation_max).unwrap();
        write!(writer, ",{:.6},{:.6},{:.6},{:.6}",
            env_stats.terrain_ocean_pct, env_stats.terrain_plains_pct, 
            env_stats.terrain_forest_pct, env_stats.terrain_desert_pct).unwrap();
        write!(writer, ",{:.6},{:.6},{:.6},{:.6}",
            env_stats.terrain_tundra_pct, env_stats.terrain_mountain_pct,
            env_stats.terrain_swamp_pct, env_stats.terrain_volcanic_pct).unwrap();
        write!(writer, ",{:.6},{:.6},{:.6}",
            env_stats.resource_plant_avg, env_stats.resource_mineral_avg, env_stats.resource_sunlight_avg).unwrap();
        write!(writer, ",{:.6},{:.6},{:.6}",
            env_stats.resource_water_avg, env_stats.resource_detritus_avg, env_stats.resource_prey_avg).unwrap();
        
        // Write ecosystem stats
        write!(writer, ",{},{}", ecosystem_total_pop, ecosystem_species_count).unwrap();
        
        // Write fitness labels
        write!(writer, ",{},{:.6}\n", current_pop, growth_rate).unwrap();
    }

    // Flush writer periodically
    if current_tick % 500 == 0 {
        writer.flush().unwrap();
    }
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

