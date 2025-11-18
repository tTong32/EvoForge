use bevy::prelude::*;
use glam::Vec2;
use std::collections::HashMap;

/// Spatial hash grid for efficient neighbor queries
/// Divides space into buckets for O(1) average-case neighbor lookups
pub struct SpatialHash {
    /// Cell size for spatial partitioning (larger = fewer buckets, faster but less precise)
    cell_size: f32,
    /// Map from bucket coordinates to list of entities in that bucket
    buckets: HashMap<(i32, i32), Vec<Entity>>,
    /// Map from entity to its current bucket (for fast removal)
    entity_buckets: HashMap<Entity, (i32, i32)>,
}

impl SpatialHash {
    /// Create a new spatial hash with the given cell size
    /// Smaller cell_size = more precise but more buckets
    pub fn new(cell_size: f32) -> Self {
        Self {
            cell_size,
            buckets: HashMap::new(),
            entity_buckets: HashMap::new(),
        }
    }

    /// Get bucket coordinates for a world position
    fn world_to_bucket(&self, pos: Vec2) -> (i32, i32) {
        (
            (pos.x / self.cell_size).floor() as i32,
            (pos.y / self.cell_size).floor() as i32,
        )
    }

    /// Clear all entries (call at start of each frame before rebuilding)
    pub fn clear(&mut self) {
        self.buckets.clear();
        self.entity_buckets.clear();
    }

    /// Insert an entity at a position
    pub fn insert(&mut self, entity: Entity, position: Vec2) {
        let bucket = self.world_to_bucket(position);

        // Remove from old bucket if it exists
        if let Some(old_bucket) = self.entity_buckets.remove(&entity) {
            if let Some(bucket_vec) = self.buckets.get_mut(&old_bucket) {
                // Optimized: Use position + swap_remove instead of retain
                if let Some(pos) = bucket_vec.iter().position(|&e| e == entity) {
                    bucket_vec.swap_remove(pos);
                }
                if bucket_vec.is_empty() {
                    self.buckets.remove(&old_bucket);
                }
            }
        }

        // Add to new bucket (pre-allocate capacity for better performance)
        self.buckets
            .entry(bucket)
            .or_insert_with(|| Vec::with_capacity(8)) // Pre-allocate for typical bucket size
            .push(entity);
        self.entity_buckets.insert(entity, bucket);
    }

    /// Remove an entity from the spatial hash
    pub fn remove(&mut self, entity: Entity) {
        if let Some(bucket) = self.entity_buckets.remove(&entity) {
            if let Some(bucket_vec) = self.buckets.get_mut(&bucket) {
                // Optimized: Use position + swap_remove instead of retain
                if let Some(pos) = bucket_vec.iter().position(|&e| e == entity) {
                    bucket_vec.swap_remove(pos);
                }
                if bucket_vec.is_empty() {
                    self.buckets.remove(&bucket);
                }
            }
        }
    }

    /// Get all entities within a radius of a position
    /// Returns entities in nearby buckets (may include some outside radius)
    pub fn query_radius(&self, position: Vec2, radius: f32) -> Vec<Entity> {
        let center_bucket = self.world_to_bucket(position);
        let radius_buckets = (radius / self.cell_size).ceil() as i32;

        let mut results = Vec::new();

        // Check all buckets within radius
        for dy in -radius_buckets..=radius_buckets {
            for dx in -radius_buckets..=radius_buckets {
                let bucket = (center_bucket.0 + dx, center_bucket.1 + dy);
                if let Some(entities) = self.buckets.get(&bucket) {
                    results.extend(entities.iter().copied());
                }
            }
        }

        results
    }

    /// Get all entities within a radius, filling into a reusable buffer
    /// More efficient when called multiple times per frame (avoids allocations)
    pub fn query_radius_into(&self, position: Vec2, radius: f32, results: &mut Vec<Entity>) {
        results.clear();
        let center_bucket = self.world_to_bucket(position);
        let radius_buckets = (radius / self.cell_size).ceil() as i32;

        // Pre-allocate based on expected bucket count (most queries hit 4-9 buckets)
        let expected_buckets = ((radius_buckets * 2 + 1) * (radius_buckets * 2 + 1)).min(16) as usize;
        results.reserve(expected_buckets * 8); // Assume ~8 entities per bucket on average (increased from 4)

        // Check all buckets within radius
        for dy in -radius_buckets..=radius_buckets {
            for dx in -radius_buckets..=radius_buckets {
                let bucket = (center_bucket.0 + dx, center_bucket.1 + dy);
                if let Some(entities) = self.buckets.get(&bucket) {
                    results.extend(entities.iter().copied());
                }
            }
        }
    }

    /// Get entities in a specific bucket
    pub fn get_bucket(&self, bucket: (i32, i32)) -> Option<&Vec<Entity>> {
        self.buckets.get(&bucket)
    }

    /// Get the number of buckets currently in use
    pub fn bucket_count(&self) -> usize {
        self.buckets.len()
    }
}

/// Resource for the spatial hash grid
#[derive(Resource)]
pub struct SpatialHashGrid {
    /// Spatial hash for organisms
    pub organisms: SpatialHash,
}

impl Default for SpatialHashGrid {
    fn default() -> Self {
        Self {
            // Use cell size of 16 units - balances precision vs performance
            // Organisms with sensory range up to 50 will check ~9 buckets
            organisms: SpatialHash::new(16.0),
        }
    }
}
