use crate::world::cell::{Cell, ResourceType, RESOURCE_TYPE_COUNT};

/// Resource regeneration rates per terrain type (base rates, multiplied by tuning)
/// [Plant, Mineral, Sunlight, Water, Detritus, Prey]
pub const BASE_REGENERATION_RATES: [[f32; RESOURCE_TYPE_COUNT]; 8] = [
    // Ocean
    [0.0, 0.1, 0.3, 1.0, 0.2, 0.5],
    // Plains
    [0.3, 0.1, 0.8, 0.4, 0.2, 0.3],
    // Forest
    [0.8, 0.05, 0.5, 0.6, 0.4, 0.2],
    // Desert
    [0.05, 0.2, 1.0, 0.1, 0.05, 0.1],
    // Tundra
    [0.1, 0.1, 0.6, 0.3, 0.1, 0.1],
    // Mountain
    [0.05, 0.5, 0.7, 0.2, 0.05, 0.05],
    // Swamp
    [0.4, 0.05, 0.4, 1.0, 0.6, 0.3],
    // Volcanic
    [0.0, 0.8, 0.9, 0.1, 0.1, 0.0],
];

/// Base resource decay rates (multiplied by tuning parameters)
pub const BASE_DECAY_RATES: [f32; RESOURCE_TYPE_COUNT] = [
    0.01, // Plant - slow decay
    0.0,  // Mineral - doesn't decay
    0.1,  // Sunlight - very fast decay (needs constant regeneration)
    0.02, // Water - slow decay (evaporation)
    0.05, // Detritus - medium decay (decomposition)
    0.03, // Prey - medium decay (moves away or dies)
];

/// Maximum resource capacity per cell
pub const MAX_RESOURCE_DENSITY: f32 = 1.0;

/// Resource regeneration rate multiplier based on temperature
pub fn temperature_regeneration_multiplier(temperature: f32) -> f32 {
    // Optimal temperature around 0.5, drops off at extremes
    let optimal_temp = 0.5;
    let deviation = (temperature - optimal_temp).abs();
    1.0 - (deviation * 2.0).min(1.0)
}

/// Resource regeneration rate multiplier based on humidity
pub fn humidity_regeneration_multiplier(humidity: f32, resource_type: ResourceType) -> f32 {
    match resource_type {
        ResourceType::Plant => 0.5 + humidity * 0.5, // Plants like humidity
        ResourceType::Water => humidity,             // Water depends on humidity
        ResourceType::Sunlight => 1.0,               // Sunlight independent
        ResourceType::Mineral => 1.0,                // Mineral independent
        ResourceType::Detritus => 0.5 + humidity * 0.5, // Detritus decomposes faster with moisture
        ResourceType::Prey => 0.3 + humidity * 0.7,  // Prey prefers moderate humidity
    }
}

/// Update resource regeneration for a single cell
/// Step 8: Now uses tuning parameters for ecosystem balance
pub fn regenerate_resources(cell: &mut Cell, dt: f32, tuning: Option<&crate::organisms::EcosystemTuning>) {
    let terrain_idx = cell.terrain as usize;
    let temp_mult = temperature_regeneration_multiplier(cell.temperature);

    // Get tuning multipliers (default to 1.0 if no tuning provided)
    let plant_mult = tuning.map(|t| t.plant_regeneration_rate / 0.08).unwrap_or(1.0);
    let mineral_mult = tuning.map(|t| t.mineral_regeneration_rate / 0.05).unwrap_or(1.0);
    let sunlight_mult = tuning.map(|t| t.sunlight_regeneration_rate / 0.15).unwrap_or(1.0);
    let water_mult = tuning.map(|t| t.water_regeneration_rate / 0.12).unwrap_or(1.0);
    let detritus_mult = tuning.map(|t| t.detritus_regeneration_rate / 0.03).unwrap_or(1.0);
    let prey_mult = tuning.map(|t| t.prey_regeneration_rate / 0.02).unwrap_or(1.0);

    let multipliers = [plant_mult, mineral_mult, sunlight_mult, water_mult, detritus_mult, prey_mult];

    for (resource_idx, &base_regeneration_rate) in BASE_REGENERATION_RATES[terrain_idx].iter().enumerate() {
        let resource_type = match resource_idx {
            0 => ResourceType::Plant,
            1 => ResourceType::Mineral,
            2 => ResourceType::Sunlight,
            3 => ResourceType::Water,
            4 => ResourceType::Detritus,
            5 => ResourceType::Prey,
            _ => continue,
        };

        let humidity_mult = humidity_regeneration_multiplier(cell.humidity, resource_type);
        let adaptation = 1.0 + cell.resource_adaptation[resource_idx].clamp(-0.5, 1.5);
        let tuning_mult = multipliers[resource_idx];
        let effective_rate = base_regeneration_rate * temp_mult * humidity_mult * adaptation * tuning_mult;

        let current = cell.resource_density[resource_idx];
        let new_value = (current + effective_rate * dt).min(MAX_RESOURCE_DENSITY);
        cell.resource_density[resource_idx] = new_value;

        // Gradually relax pressure memory
        let pressure = cell.resource_pressure[resource_idx];
        if pressure > 0.0 {
            cell.resource_pressure[resource_idx] =
                (pressure - dt * 0.1 * (1.0 + pressure * 0.2)).max(0.0);
        }
    }

    update_resource_adaptation(cell, dt);
}

/// Apply decay to resources in a cell
/// Step 8: Now uses tuning parameters for ecosystem balance
pub fn decay_resources(cell: &mut Cell, dt: f32, tuning: Option<&crate::organisms::EcosystemTuning>) {
    // Get tuning multipliers (default to 1.0 if no tuning provided)
    let plant_decay_mult = tuning.map(|t| t.plant_decay_rate / 0.01).unwrap_or(1.0);
    let mineral_decay_mult = tuning.map(|t| t.mineral_decay_rate / 0.001).unwrap_or(1.0);
    let sunlight_decay_mult = tuning.map(|t| t.sunlight_decay_rate / 0.02).unwrap_or(1.0);
    let water_decay_mult = tuning.map(|t| t.water_decay_rate / 0.005).unwrap_or(1.0);
    let detritus_decay_mult = tuning.map(|t| t.detritus_decay_rate / 0.015).unwrap_or(1.0);
    let prey_decay_mult = tuning.map(|t| t.prey_decay_rate / 0.02).unwrap_or(1.0);

    let multipliers = [plant_decay_mult, mineral_decay_mult, sunlight_decay_mult, water_decay_mult, detritus_decay_mult, prey_decay_mult];

    for (idx, &base_decay_rate) in BASE_DECAY_RATES.iter().enumerate() {
        if base_decay_rate > 0.0 {
            let effective_decay = base_decay_rate * multipliers[idx];
            let current = cell.resource_density[idx];
            cell.resource_density[idx] = (current * (1.0 - effective_decay * dt)).max(0.0);
        }
    }
}

/// Quantize small resource values to zero (performance optimization)
pub fn quantize_resources(cell: &mut Cell, threshold: f32) {
    for resource in &mut cell.resource_density {
        if *resource < threshold {
            *resource = 0.0;
        }
    }
}

/// Adjust resource adaptation based on sustained pressure and climate
fn update_resource_adaptation(cell: &mut Cell, dt: f32) {
    for idx in 0..RESOURCE_TYPE_COUNT {
        let pressure = cell.resource_pressure[idx];
        let target = (pressure * 0.08) - 0.05; // mild boost under pressure
        let current = cell.resource_adaptation[idx];

        // Climate stress pushes adaptation negative for plants & water
        let climate_stress = match idx {
            x if x == ResourceType::Plant as usize || x == ResourceType::Water as usize => {
                ((0.5 - cell.humidity).abs() + (0.5 - cell.temperature).abs()) * 0.15
            }
            _ => 0.0,
        };

        let delta = (target - current) * dt * 0.5 - climate_stress * dt;
        cell.resource_adaptation[idx] = (current + delta).clamp(-0.5, 1.5);
    }
}
