use bevy::prelude::*;

use crate::organisms::Genome;
use crate::organisms::PlantTraits;
use crate::world::cell::ResourceType;
use crate::world::{Cell, WorldGrid};
use crate::world::plants::{PlantCommunity, PlantSpecies};

/// Update plant growth, competition, and nutrient cycling in all cells.
pub fn update_plants_system(
    mut world_grid: ResMut<WorldGrid>, 
    timer: Res<crate::world::PlantUpdateTimer>,
    time: Res<Time>
) {
    let dt = match timer.should_update() {
        Some(accumulated_dt) => accumulated_dt,
        None => return,
    };

    let coords = world_grid.get_chunk_coords();
    for (cx, cy) in coords {
        if let Some(chunk) = world_grid.get_chunk_mut(cx, cy) {
            for y in 0..crate::world::chunk::CHUNK_SIZE {
                for x in 0..crate::world::chunk::CHUNK_SIZE {
                    if let Some(cell) = chunk.get_cell_mut(x, y) {
                        update_cell_plants(cell, dt);
                    }
                }
            }
        }
    }
}

fn update_cell_plants(cell: &mut Cell, dt: f32) {
    if cell.plant_community.is_empty() {
        // Still decay nutrients over time even if no plants.
        decay_nutrients(cell, dt);
        return;
    }

    let mut heights = Vec::with_capacity(cell.plant_community.len());
    let mut growth_deltas = Vec::with_capacity(cell.plant_community.len());

    // Resource availability from legacy scalar fields for now.
    let sunlight = cell.get_resource(ResourceType::Sunlight);
    let water = cell.get_resource(ResourceType::Water);
    let mineral = cell.get_resource(ResourceType::Mineral);

    // Nutrient boost from dead matter and animal nutrients.
    let nutrient_boost = (cell.dead_organic_matter + cell.animal_nutrients).clamp(0.0, 1.0);

    let base_env_match = {
        let temp = 1.0 - (cell.temperature - 0.5).abs() * 2.0;
        let humidity = 1.0 - (cell.humidity - 0.6).abs() * 2.0;
        (0.4 * temp + 0.6 * humidity).clamp(0.0, 1.0)
    };

    for species in cell.plant_community.iter_mut() {
        let traits = PlantTraits::from_genome(&species.genome);

        let env_match = base_env_match * (1.0 + 0.3 * nutrient_boost);
        let resource_avail = resource_availability(sunlight, water, mineral, &traits);

        let growth = species.percentage
            * traits.growth_rate
            * resource_avail
            * env_match
            * dt;

        growth_deltas.push(growth);
        heights.push(traits.height);
        species.age += dt;

        // Simple lifespan-based mortality: very old plants shed biomass into dead matter.
        if species.age > traits.lifespan {
            let dying = (species.percentage * 0.2 * dt).min(species.percentage);
            species.percentage -= dying;
            cell.dead_organic_matter += dying;
        }
    }

    // Apply growth.
    for (species, delta) in cell
        .plant_community
        .iter_mut()
        .zip(growth_deltas.iter())
    {
        species.percentage += *delta;
    }

    // Competition: bias toward taller species.
    let mut weighted_sum = 0.0;
    for (species, height) in cell.plant_community.iter().zip(heights.iter()) {
        let comp_weight = *height;
        weighted_sum += species.percentage * comp_weight;
    }

    if weighted_sum > 0.0 {
        for (species, height) in cell.plant_community.iter_mut().zip(heights.iter()) {
            let comp_weight = *height;
            let target_share = (species.percentage * comp_weight) / weighted_sum;
            species.percentage = species.percentage + (target_share - species.percentage) * 0.4;
        }
    }

    // Clamp and renormalize to <= 1.0.
    let sum: f32 = cell.plant_community.iter().map(|s| s.percentage).sum();
    if sum > 1.0 {
        let inv = 1.0 / sum;
        for species in cell.plant_community.iter_mut() {
            species.percentage *= inv;
        }
    }

    // Decay nutrients a bit each tick.
    decay_nutrients(cell, dt);
}

fn decay_nutrients(cell: &mut Cell, dt: f32) {
    cell.dead_organic_matter *= (1.0 - 0.1 * dt).max(0.0);
    cell.animal_nutrients *= (1.0 - 0.05 * dt).max(0.0);
}

fn environment_match(cell: &Cell, _traits: &PlantTraits) -> f32 {
    let temp = 1.0 - (cell.temperature - 0.5).abs() * 2.0;
    let humidity = 1.0 - (cell.humidity - 0.6).abs() * 2.0;
    (0.4 * temp + 0.6 * humidity).clamp(0.0, 1.0)
}

fn resource_availability(
    sunlight: f32,
    water: f32,
    mineral: f32,
    traits: &PlantTraits,
) -> f32 {
    let s = sunlight * traits.sunlight_efficiency;
    let w = water * traits.water_efficiency;
    let m = mineral * traits.mineral_efficiency;
    (0.6 * s + 0.3 * w + 0.1 * m).clamp(0.0, 1.0)
}

/// Simple plant spreading between neighboring cells.
pub fn plant_spread_system(
    mut world_grid: ResMut<WorldGrid>, 
    timer: Res<crate::world::PlantUpdateTimer>,
    time: Res<Time>
) {
    let dt = match timer.should_update() {
        Some(accumulated_dt) => accumulated_dt,
        None => return,
    }

    #[derive(Clone)]
    struct SpreadEvent {
        world_x: f32,
        world_y: f32,
        species_id: u32,
        genome: Genome,
        delta_percentage: f32,
    }

    let mut events: Vec<SpreadEvent> = Vec::new();

    let coords = world_grid.get_chunk_coords();
    for (chunk_x, chunk_y) in coords {
        if let Some(chunk) = world_grid.get_chunk(chunk_x, chunk_y) {
            for y in 0..crate::world::chunk::CHUNK_SIZE {
                for x in 0..crate::world::chunk::CHUNK_SIZE {
                    if let Some(cell) = chunk.get_cell(x, y) {
                        if cell.plant_community.is_empty() {
                            continue;
                        }

                        let world_x = chunk_x as f32 * crate::world::chunk::CHUNK_WORLD_SIZE
                            + x as f32 * crate::world::chunk::CELL_SIZE;
                        let world_y = chunk_y as f32 * crate::world::chunk::CHUNK_WORLD_SIZE
                            + y as f32 * crate::world::chunk::CELL_SIZE;

                        for species in cell.plant_community.iter() {
                            let traits = PlantTraits::from_genome(&species.genome);
                            let potential = species.percentage * traits.spread_rate * dt;
                            if potential <= 0.0001 {
                                continue;
                            }

                            // Four neighbors (von Neumann neighborhood).
                            let neighbors = [(1.0f32, 0.0f32), (-1.0, 0.0), (0.0, 1.0), (0.0, -1.0)];
                            let per_neighbor = potential / neighbors.len() as f32;

                            for (dx, dy) in neighbors.iter() {
                                events.push(SpreadEvent {
                                    world_x: world_x + dx,
                                    world_y: world_y + dy,
                                    species_id: species.species_id,
                                    genome: species.genome.clone(),
                                    delta_percentage: per_neighbor,
                                });
                            }
                        }
                    }
                }
            }
        }
    }

    // Apply spreading events.
    for ev in events {
        if let Some(cell) = world_grid.get_cell_mut(ev.world_x, ev.world_y) {
            colonize_cell(cell, ev.species_id, &ev.genome, ev.delta_percentage);
        }
    }
}

fn colonize_cell(cell: &mut Cell, species_id: u32, genome: &Genome, delta_percentage: f32) {
    if delta_percentage <= 0.0 {
        return;
    }

    if let Some(existing) = cell.plant_community.iter_mut().find(|s| s.species_id == species_id) {
        existing.percentage += delta_percentage;
    } else {
        if cell.plant_community.len() < crate::world::plants::MAX_PLANT_SPECIES_PER_CELL {
            cell.plant_community.push(PlantSpecies {
                species_id,
                genome: genome.clone(),
                percentage: delta_percentage,
                age: 0.0,
            });
        } else if let Some((idx, _)) = cell
            .plant_community
            .iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| a.percentage.partial_cmp(&b.percentage).unwrap())
        {
            cell.plant_community[idx].percentage += delta_percentage;
        }
    }

    // Normalize to keep total <= 1.0.
    let sum: f32 = cell.plant_community.iter().map(|s| s.percentage).sum();
    if sum > 1.0 {
        let inv = 1.0 / sum;
        for s in cell.plant_community.iter_mut() {
            s.percentage *= inv;
        }
    }
}

/// Seed founder plant species in initialized chunks.
pub fn initialize_founder_plants(world_grid: &mut WorldGrid) {
    let mut rng = fastrand::Rng::new();

    for (cx, cy) in world_grid.get_chunk_coords() {
        if let Some(chunk) = world_grid.get_chunk_mut(cx, cy) {
            for y in 0..crate::world::chunk::CHUNK_SIZE {
                for x in 0..crate::world::chunk::CHUNK_SIZE {
                    if let Some(cell) = chunk.get_cell_mut(x, y) {
                        // Skip cells that are clearly unsuitable (e.g. ocean/volcanic).
                        if matches!(
                            cell.terrain,
                            crate::world::TerrainType::Ocean | crate::world::TerrainType::Volcanic
                        ) {
                            continue;
                        }

                        // 30% chance to seed a cell.
                        if rng.f32() < 0.3 {
                            let genome = Genome::random();
                            let species_id = rng.u32(..1_000_000);

                            let mut community = PlantCommunity::new();
                            community.push(PlantSpecies {
                                species_id,
                                genome,
                                percentage: rng.f32() * 0.4 + 0.2, // 20–60%
                                age: 0.0,
                            });
                            cell.plant_community = community;
                        }
                    }
                }
            }
        }
    }
}


