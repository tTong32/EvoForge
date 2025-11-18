use crate::world::cell::Cell;
use crate::world::chunk::Chunk;
use bevy::prelude::*;
use std::collections::HashMap;

/// The world grid manages chunks in a sparse storage system
/// Only active chunks are kept in memory for efficiency
#[derive(Resource, Default)]
pub struct WorldGrid {
    /// Sparse storage: HashMap keyed by (chunk_x, chunk_y)
    chunks: HashMap<(i32, i32), Chunk>,
    /// Set of dirty chunks that need updates this tick
    dirty_chunks: Vec<(i32, i32)>,
}

impl WorldGrid {
    /// Get or create a chunk at the specified chunk coordinates
    pub fn get_or_create_chunk(&mut self, chunk_x: i32, chunk_y: i32) -> &mut Chunk {
        let key = (chunk_x, chunk_y);
        if !self.chunks.contains_key(&key) {
            let chunk = Chunk::new(chunk_x, chunk_y);
            self.chunks.insert(key, chunk);
        }
        self.chunks.get_mut(&key).unwrap()
    }

    /// Get a chunk without creating it if it doesn't exist
    pub fn get_chunk(&self, chunk_x: i32, chunk_y: i32) -> Option<&Chunk> {
        self.chunks.get(&(chunk_x, chunk_y))
    }

    /// Get a mutable chunk without creating it if it doesn't exist
    pub fn get_chunk_mut(&mut self, chunk_x: i32, chunk_y: i32) -> Option<&mut Chunk> {
        self.chunks.get_mut(&(chunk_x, chunk_y))
    }

    /// Get a cell at world coordinates
    pub fn get_cell(&self, world_x: f32, world_y: f32) -> Option<&Cell> {
        let (chunk_x, chunk_y) = Chunk::world_to_chunk(world_x, world_y);
        let (local_x, local_y) = Chunk::world_to_local(world_x, world_y);

        self.get_chunk(chunk_x, chunk_y)
            .and_then(|chunk| chunk.get_cell(local_x, local_y))
    }

    /// Get a mutable cell at world coordinates (creates chunk if needed)
    pub fn get_cell_mut(&mut self, world_x: f32, world_y: f32) -> Option<&mut Cell> {
        let (chunk_x, chunk_y) = Chunk::world_to_chunk(world_x, world_y);
        let (local_x, local_y) = Chunk::world_to_local(world_x, world_y);

        let chunk = self.get_or_create_chunk(chunk_x, chunk_y);
        chunk.get_cell_mut(local_x, local_y)
    }

    /// Get all dirty chunks (chunks that have been modified)
    pub fn get_dirty_chunks(&self) -> Vec<(i32, i32)> {
        self.chunks
            .iter()
            .filter(|(_, chunk)| chunk.dirty)
            .map(|(key, _)| *key)
            .collect()
    }

    /// Clear dirty flags for all chunks
    pub fn clear_dirty_flags(&mut self) {
        for chunk in self.chunks.values_mut() {
            chunk.mark_clean();
        }
        self.dirty_chunks.clear();
    }

    /// Get the number of active chunks
    pub fn chunk_count(&self) -> usize {
        self.chunks.len()
    }

    /// Get all chunk coordinates
    pub fn get_chunk_coords(&self) -> Vec<(i32, i32)> {
        self.chunks.keys().copied().collect()
    }

    /// Remove a chunk (useful for cleanup of distant chunks)
    pub fn remove_chunk(&mut self, chunk_x: i32, chunk_y: i32) {
        self.chunks.remove(&(chunk_x, chunk_y));
    }
    
    /// Check if chunk coordinates are within world bounds
    /// Returns true if chunk is within MAX_WORLD_RADIUS_CHUNKS of origin
    pub fn is_within_bounds(&self, chunk_x: i32, chunk_y: i32) -> bool {
        use crate::world::MAX_WORLD_RADIUS_CHUNKS;
        chunk_x.abs() <= MAX_WORLD_RADIUS_CHUNKS && chunk_y.abs() <= MAX_WORLD_RADIUS_CHUNKS
    }
}
