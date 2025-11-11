use crate::world::cell::{Cell, TerrainType};
use bevy::prelude::*;

/// Global climate state
#[derive(Resource, Clone, Debug)]
pub struct ClimateState {
    /// Global base temperature (0.0 = freezing, 1.0 = boiling)
    pub base_temperature: f32,
    /// Global base humidity (0.0 = dry, 1.0 = saturated)
    pub base_humidity: f32,
    /// Current season (0.0 to 1.0, cycles annually)
    pub season: f32,
    /// Time in simulation ticks
    pub time: u64,
}

impl Default for ClimateState {
    fn default() -> Self {
        Self {
            base_temperature: 0.5,
            base_humidity: 0.5,
            season: 0.0,
            time: 0,
        }
    }
}

impl ClimateState {
    /// Update climate state (called each tick)
    pub fn update(&mut self, _dt: f32) {
        self.time += 1;

        // Seasonal cycle (1000 ticks = 1 year)
        let season_period = 1000.0;
        self.season = ((self.time as f32) / season_period) % 1.0;

        // Seasonal temperature variation
        let season_amplitude = 0.2;
        let seasonal_temp = (self.season * 2.0 * std::f32::consts::PI).sin() * season_amplitude;
        self.base_temperature = 0.5 + seasonal_temp;

        // Seasonal humidity variation (opposite phase to temperature)
        let seasonal_humidity =
            ((self.season * 2.0 * std::f32::consts::PI) + std::f32::consts::PI).sin() * 0.15;
        self.base_humidity = 0.5 + seasonal_humidity;

        // Long-term climate drift (slow random walk for ice ages/global warming)
        // This is a simplified version - can be enhanced later
        // OPTIMIZED: Use fastrand instead of thread_rng() for better performance
        let drift_rate = 0.0001;
        self.base_temperature += (fastrand::f32() - 0.5) * drift_rate;
        self.base_temperature = self.base_temperature.clamp(0.2, 0.8);
    }

    /// Get temperature for a cell based on elevation and terrain
    pub fn get_cell_temperature(&self, elevation: u16, terrain: TerrainType) -> f32 {
        let base = self.base_temperature;

        // Elevation effect (higher = colder)
        let elevation_factor = (elevation as f32 / 65535.0) * 0.3; // Max 0.3 colder at max elevation
        let elevation_effect = -elevation_factor;

        // Terrain modifiers
        let terrain_modifier = match terrain {
            TerrainType::Ocean => 0.0,      // Ocean moderates temperature
            TerrainType::Plains => 0.0,     // Neutral
            TerrainType::Forest => -0.05,   // Slightly cooler
            TerrainType::Desert => 0.15,    // Hotter
            TerrainType::Tundra => -0.2,    // Much colder
            TerrainType::Mountain => -0.25, // Very cold
            TerrainType::Swamp => 0.05,     // Slightly warmer
            TerrainType::Volcanic => 0.3,   // Very hot
        };

        (base + elevation_effect + terrain_modifier).clamp(0.0, 1.0)
    }

    /// Get humidity for a cell based on terrain and temperature
    pub fn get_cell_humidity(&self, terrain: TerrainType, temperature: f32) -> f32 {
        let base = self.base_humidity;

        // Terrain modifiers
        let terrain_modifier = match terrain {
            TerrainType::Ocean => 0.3,     // Very humid
            TerrainType::Plains => 0.0,    // Neutral
            TerrainType::Forest => 0.2,    // Humid
            TerrainType::Desert => -0.3,   // Very dry
            TerrainType::Tundra => 0.1,    // Slightly humid
            TerrainType::Mountain => -0.1, // Dry (high altitude)
            TerrainType::Swamp => 0.4,     // Very humid
            TerrainType::Volcanic => -0.2, // Dry
        };

        // Temperature effect (hotter = can hold more water, but evaporation increases)
        let temp_effect = (temperature - 0.5) * 0.2;

        (base + terrain_modifier + temp_effect).clamp(0.0, 1.0)
    }
}

/// Update climate for a single cell
pub fn update_cell_climate(cell: &mut Cell, climate: &ClimateState) {
    cell.temperature = climate.get_cell_temperature(cell.elevation, cell.terrain);
    cell.humidity = climate.get_cell_humidity(cell.terrain, cell.temperature);
}
