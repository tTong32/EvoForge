use crate::world::cell::{Cell, TerrainType};
use bevy::prelude::*;
use glam::Vec2;

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
    /// Phase offset for spatial variation
    pub noise_phase: f32,
    /// Cooldown before spawning next stochastic event
    pub event_cooldown: f32,
    /// Active transient climate events
    pub events: Vec<ClimateEvent>,
    /// Seed driving deterministic regional offsets
    pub regional_seed: u64,
}

impl Default for ClimateState {
    fn default() -> Self {
        Self {
            base_temperature: 0.5,
            base_humidity: 0.5,
            season: 0.0,
            time: 0,
            noise_phase: 0.0,
            event_cooldown: 5.37, // 120.0 / 22.34 (scaled for 600-tick lifetime)
            events: Vec::new(),
            regional_seed: fastrand::u64(..),
        }
    }
}

impl ClimateState {
    /// Update climate state (called each tick)
    pub fn update(&mut self, _dt: f32) {
        self.time += 1;

        // Seasonal cycle (44.8 ticks = 1 year, scaled for 600-tick lifetime)
        let season_period = 44.8; // 1000.0 / 22.34
        self.season = ((self.time as f32) / season_period) % 1.0;

        // Seasonal temperature variation
        let season_amplitude = 0.2;
        let seasonal_temp = (self.season * 2.0 * std::f32::consts::PI).sin() * season_amplitude;
        self.base_temperature = 0.5 + seasonal_temp;

        // Seasonal humidity variation (opposite phase to temperature)
        let seasonal_humidity =
            ((self.season * 2.0 * std::f32::consts::PI) + std::f32::consts::PI).sin() * 0.15;
        self.base_humidity = 0.5 + seasonal_humidity;

        // Long-term climate drift (scaled 22.34x faster for 600-tick lifetime)
        let drift_rate = 0.00223; // 0.0001 × 22.34
        self.base_temperature += (fastrand::f32() - 0.5) * drift_rate;
        self.base_temperature = self.base_temperature.clamp(0.2, 0.8);

        let dt = 1.0f32;
        self.noise_phase += 0.335 * dt; // 0.015 × 22.34 (scaled for 600-tick lifetime)

        // Decay stochastic events
        for event in &mut self.events {
            event.time_remaining -= dt;
        }
        self.events.retain(|event| event.time_remaining > 0.0);

        // Randomly spawn new event
        self.event_cooldown -= dt;
        if self.event_cooldown <= 0.0 {
            if fastrand::f32() < 0.02 {
                self.spawn_event();
            }
            self.event_cooldown = fastrand::f32() * 13.4 + 5.37; // Scaled for 600-tick lifetime
        }
    }

    /// Get temperature for a cell based on elevation and terrain
    pub fn get_cell_temperature(&self, elevation: u16, terrain: TerrainType) -> f32 {
        let base = self.base_temperature;

        // Elevation effect (higher = colder)
        let elevation_factor = (elevation as f32 / 65535.0) * 0.3;
        let elevation_effect = -elevation_factor;

        // Terrain modifiers
        let terrain_modifier = match terrain {
            TerrainType::Ocean => 0.0,
            TerrainType::Plains => 0.0,
            TerrainType::Forest => -0.05,
            TerrainType::Desert => 0.15,
            TerrainType::Tundra => -0.2,
            TerrainType::Mountain => -0.25,
            TerrainType::Swamp => 0.05,
            TerrainType::Volcanic => 0.3,
        };

        (base + elevation_effect + terrain_modifier).clamp(0.0, 1.0)
    }

    /// Get humidity for a cell based on terrain and temperature
    pub fn get_cell_humidity(&self, terrain: TerrainType, temperature: f32) -> f32 {
        let base = self.base_humidity;

        let terrain_modifier = match terrain {
            TerrainType::Ocean => 0.3,
            TerrainType::Plains => 0.0,
            TerrainType::Forest => 0.2,
            TerrainType::Desert => -0.3,
            TerrainType::Tundra => 0.1,
            TerrainType::Mountain => -0.1,
            TerrainType::Swamp => 0.4,
            TerrainType::Volcanic => -0.2,
        };

        let temp_effect = (temperature - 0.5) * 0.2;

        (base + terrain_modifier + temp_effect).clamp(0.0, 1.0)
    }

    fn regional_offsets(&self, world_pos: Vec2) -> (f32, f32) {
        let scale = 0.005;
        let angle_x = world_pos.x * scale + self.noise_phase;
        let angle_y = world_pos.y * scale * 1.3 - self.noise_phase * 0.6;
        let temp = (angle_x.sin() * angle_y.cos()) * 0.08;
        let humidity = (angle_x.cos() * 0.06) + (angle_y.sin() * 0.05);
        (temp, humidity)
    }

    fn event_offsets(&self, world_pos: Vec2) -> (f32, f32) {
        let mut temp = 0.0;
        let mut humidity = 0.0;
        for event in &self.events {
            let distance = world_pos.distance(event.center);
            if distance <= event.radius {
                let influence = 1.0 - (distance / event.radius).powf(1.5);
                temp += event.temperature_delta * influence;
                humidity += event.humidity_delta * influence;
            }
        }
        (temp, humidity)
    }

    fn spawn_event(&mut self) {
        let mut rng = fastrand::Rng::with_seed(self.regional_seed ^ self.time);
        let center = Vec2::new(rng.f32() * 400.0 - 200.0, rng.f32() * 400.0 - 200.0);
        let radius = rng.f32() * 120.0 + 60.0;
        let (temperature_delta, humidity_delta, duration) = match rng.u8(..4) {
            0 => (0.08, -0.12, 8.06),  // heatwave (180.0 / 22.34)
            1 => (-0.1, 0.15, 8.95),   // cold rainstorm (200.0 / 22.34)
            2 => (0.0, -0.2, 9.85),    // drought (220.0 / 22.34)
            _ => (0.05, 0.18, 7.16),   // tropical storm (160.0 / 22.34)
        };

        self.events.push(ClimateEvent {
            center,
            radius,
            temperature_delta,
            humidity_delta,
            time_remaining: duration,
        });
    }
}

/// Update climate for a single cell
pub fn update_cell_climate(cell: &mut Cell, climate: &ClimateState, world_pos: Vec2) {
    let mut temperature = climate.get_cell_temperature(cell.elevation, cell.terrain);
    let mut humidity = climate.get_cell_humidity(cell.terrain, temperature);

    let (regional_temp, regional_humidity) = climate.regional_offsets(world_pos);
    temperature += regional_temp;
    humidity += regional_humidity;

    let (event_temp, event_humidity) = climate.event_offsets(world_pos);
    temperature += event_temp;
    humidity += event_humidity;

    cell.temperature = temperature.clamp(0.0, 1.0);
    cell.humidity = humidity.clamp(0.0, 1.0);
}

#[derive(Clone, Debug)]
pub struct ClimateEvent {
    pub center: Vec2,
    pub radius: f32,
    pub temperature_delta: f32,
    pub humidity_delta: f32,
    pub time_remaining: f32,
}
