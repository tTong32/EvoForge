/// Represents a single cell in the world grid
/// Each cell contains environmental data and resource information
#[derive(Debug, Clone, Copy)]
pub struct Cell {
    /// Temperature in arbitrary units (0.0 = freezing, 1.0 = boiling)
    pub temperature: f32,
    /// Humidity level (0.0 = dry, 1.0 = saturated)
    pub humidity: f32,
    /// Elevation relative to sea level (0-65535, stored as u16)
    pub elevation: u16,
    /// Terrain type (affects resource generation and movement)
    pub terrain: TerrainType,
    /// Resource densities for each resource type
    /// [Plant, Mineral, Sunlight, Water, Detritus, Prey]
    pub resource_density: [f32; 6],
}

impl Default for Cell {
    fn default() -> Self {
        Self {
            temperature: 0.5,
            humidity: 0.5,
            elevation: 0,
            terrain: TerrainType::Plains,
            resource_density: [0.0; 6],
        }
    }
}

impl Cell {
    /// Create a new cell with default values
    pub fn new() -> Self {
        Self::default()
    }

    /// Create a cell with specific terrain type
    pub fn with_terrain(terrain: TerrainType) -> Self {
        Self {
            terrain,
            ..Default::default()
        }
    }

    /// Get resource density for a specific resource type
    pub fn get_resource(&self, resource_type: ResourceType) -> f32 {
        self.resource_density[resource_type as usize]
    }

    /// Set resource density for a specific resource type
    pub fn set_resource(&mut self, resource_type: ResourceType, value: f32) {
        self.resource_density[resource_type as usize] = value.max(0.0);
    }

    /// Add resource density (with clamping to prevent negative)
    pub fn add_resource(&mut self, resource_type: ResourceType, amount: f32) {
        let idx = resource_type as usize;
        self.resource_density[idx] = (self.resource_density[idx] + amount).max(0.0);
    }
}

/// Terrain types that affect environmental properties and movement
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum TerrainType {
    Ocean = 0,
    Plains = 1,
    Forest = 2,
    Desert = 3,
    Tundra = 4,
    Mountain = 5,
    Swamp = 6,
    Volcanic = 7,
}

impl Default for TerrainType {
    fn default() -> Self {
        TerrainType::Plains
    }
}

/// Resource types in the ecosystem
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(usize)]
pub enum ResourceType {
    Plant = 0,
    Mineral = 1,
    Sunlight = 2,
    Water = 3,
    Detritus = 4,
    Prey = 5,
}

pub const RESOURCE_TYPE_COUNT: usize = 6;
