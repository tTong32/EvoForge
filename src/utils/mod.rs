/// Utility functions and helpers for the simulation

pub mod spatial_hash;
pub use spatial_hash::*;

/// Convert between different coordinate systems
pub mod coordinates {
    /// Convert world coordinates to chunk coordinates
    pub fn world_to_chunk(world_x: f32, world_y: f32, chunk_size: usize) -> (i32, i32) {
        (
            (world_x / chunk_size as f32).floor() as i32,
            (world_y / chunk_size as f32).floor() as i32,
        )
    }

    /// Convert chunk coordinates to world coordinates (center of chunk)
    pub fn chunk_to_world_center(chunk_x: i32, chunk_y: i32, chunk_size: usize) -> (f32, f32) {
        (
            (chunk_x as f32 + 0.5) * chunk_size as f32,
            (chunk_y as f32 + 0.5) * chunk_size as f32,
        )
    }
}

/// Mathematical utilities
pub mod math {
    /// Clamp a value between min and max
    pub fn clamp<T: PartialOrd>(value: T, min: T, max: T) -> T {
        if value < min {
            min
        } else if value > max {
            max
        } else {
            value
        }
    }

    /// Linear interpolation
    pub fn lerp(a: f32, b: f32, t: f32) -> f32 {
        a + (b - a) * t.clamp(0.0, 1.0)
    }

    /// Smoothstep interpolation
    pub fn smoothstep(edge0: f32, edge1: f32, x: f32) -> f32 {
        let t = ((x - edge0) / (edge1 - edge0)).clamp(0.0, 1.0);
        t * t * (3.0 - 2.0 * t)
    }
}

