use crate::world::cell::TerrainType;
use crate::world::chunk::{Chunk, CHUNK_SIZE};
use rand::{Rng, SeedableRng};

/// Generate terrain for a chunk based on chunk coordinates
/// This creates simple procedural terrain - can be enhanced with noise later
pub fn generate_chunk_terrain(chunk: &mut Chunk) {
    // Use chunk coordinates as seed for deterministic generation
    let seed = (chunk.chunk_x as u64).wrapping_mul(31) ^ (chunk.chunk_y as u64);
    let mut local_rng = rand::rngs::StdRng::seed_from_u64(seed);

    for y in 0..CHUNK_SIZE {
        for x in 0..CHUNK_SIZE {
            if let Some(cell) = chunk.get_cell_mut(x, y) {
                // Simple terrain generation based on distance from center and elevation
                let center_x = CHUNK_SIZE as f32 / 2.0;
                let center_y = CHUNK_SIZE as f32 / 2.0;
                let dist_from_center =
                    ((x as f32 - center_x).powi(2) + (y as f32 - center_y).powi(2)).sqrt()
                        / (CHUNK_SIZE as f32 / 2.0);

                // Generate elevation (0-65535)
                let base_elevation = (dist_from_center * 10000.0) as u16;
                let elevation_noise = local_rng.gen_range(0..5000);
                cell.elevation = (base_elevation + elevation_noise).min(65535);

                // Determine terrain type based on elevation and position
                let elevation_normalized = cell.elevation as f32 / 65535.0;

                cell.terrain = if elevation_normalized < 0.2 {
                    // Low elevation - water/swamp
                    if local_rng.gen_bool(0.7) {
                        TerrainType::Ocean
                    } else {
                        TerrainType::Swamp
                    }
                } else if elevation_normalized < 0.3 {
                    // Low land - plains/forest
                    if local_rng.gen_bool(0.6) {
                        TerrainType::Plains
                    } else {
                        TerrainType::Forest
                    }
                } else if elevation_normalized < 0.5 {
                    // Mid elevation - varied
                    match local_rng.gen_range(0..4) {
                        0 => TerrainType::Plains,
                        1 => TerrainType::Forest,
                        2 => TerrainType::Desert,
                        _ => TerrainType::Tundra,
                    }
                } else if elevation_normalized < 0.8 {
                    // High elevation - tundra/mountain
                    if local_rng.gen_bool(0.7) {
                        TerrainType::Tundra
                    } else {
                        TerrainType::Mountain
                    }
                } else {
                    // Very high - mountain/volcanic
                    if local_rng.gen_bool(0.9) {
                        TerrainType::Mountain
                    } else {
                        TerrainType::Volcanic
                    }
                };
            }
        }
    }
}

/// Initialize a chunk with generated terrain
pub fn initialize_chunk(chunk: &mut Chunk) {
    generate_chunk_terrain(chunk);
}
