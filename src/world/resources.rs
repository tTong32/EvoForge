use crate::world::cell::{Cell, ResourceType, RESOURCE_TYPE_COUNT};

/// Resource regeneration rates per terrain type
/// [Plant, Mineral, Sunlight, Water, Detritus, Prey]
pub const REGENERATION_RATES: [[f32; RESOURCE_TYPE_COUNT]; 8] = [
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

/// Resource decay rates (how quickly resources disappear)
pub const DECAY_RATES: [f32; RESOURCE_TYPE_COUNT] = [
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
pub fn regenerate_resources(cell: &mut Cell, dt: f32) {
    let terrain_idx = cell.terrain as usize;
    let temp_mult = temperature_regeneration_multiplier(cell.temperature);

    for (resource_idx, &regeneration_rate) in REGENERATION_RATES[terrain_idx].iter().enumerate() {
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
        let effective_rate = regeneration_rate * temp_mult * humidity_mult;

        let current = cell.resource_density[resource_idx];
        let new_value = (current + effective_rate * dt).min(MAX_RESOURCE_DENSITY);
        cell.resource_density[resource_idx] = new_value;
    }
}

/// Apply decay to resources in a cell
pub fn decay_resources(cell: &mut Cell, dt: f32) {
    for (idx, &decay_rate) in DECAY_RATES.iter().enumerate() {
        if decay_rate > 0.0 {
            let current = cell.resource_density[idx];
            cell.resource_density[idx] = (current * (1.0 - decay_rate * dt)).max(0.0);
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
