use crate::world::cell::Cell;

/// Size of a chunk in cells (64x64 = 4096 cells per chunk)
pub const CHUNK_SIZE: usize = 64;

/// A chunk represents a fixed-size region of the world
/// Chunks are stored sparsely (only active chunks exist in memory)
#[derive(Debug)]
pub struct Chunk {
    /// The cells in this chunk, stored as a Vec to avoid stack overflow
    /// Access: cells[y * CHUNK_SIZE + x]
    cells: Vec<Cell>,
    /// Chunk coordinates in chunk-space (not world-space)
    pub chunk_x: i32,
    pub chunk_y: i32,
    /// Dirty flag - indicates if this chunk has been modified this tick
    pub dirty: bool,
    /// Set of cell coordinates that have been modified (for efficient updates)
    /// Stored as (x, y) tuples in local chunk coordinates
    pub dirty_cells: std::collections::HashSet<(usize, usize)>,
}

impl Chunk {
    /// Create a new chunk at the specified chunk coordinates
    pub fn new(chunk_x: i32, chunk_y: i32) -> Self {
        // Use Vec to allocate on heap directly, avoiding stack overflow
        let mut cells = Vec::with_capacity(CHUNK_SIZE * CHUNK_SIZE);
        cells.resize_with(CHUNK_SIZE * CHUNK_SIZE, Cell::new);
        Self {
            cells,
            chunk_x,
            chunk_y,
            dirty: false,
            dirty_cells: std::collections::HashSet::new(),
        }
    }

    /// Get a cell at local coordinates (0..CHUNK_SIZE)
    pub fn get_cell(&self, x: usize, y: usize) -> Option<&Cell> {
        if x < CHUNK_SIZE && y < CHUNK_SIZE {
            Some(&self.cells[y * CHUNK_SIZE + x])
        } else {
            None
        }
    }

    /// Get a mutable cell at local coordinates
    pub fn get_cell_mut(&mut self, x: usize, y: usize) -> Option<&mut Cell> {
        if x < CHUNK_SIZE && y < CHUNK_SIZE {
            self.dirty = true;
            self.dirty_cells.insert((x, y));
            Some(&mut self.cells[y * CHUNK_SIZE + x])
        } else {
            None
        }
    }

    /// Convert world coordinates to chunk coordinates
    pub fn world_to_chunk(world_x: f32, world_y: f32) -> (i32, i32) {
        (
            (world_x / CHUNK_SIZE as f32).floor() as i32,
            (world_y / CHUNK_SIZE as f32).floor() as i32,
        )
    }

    /// Convert world coordinates to local cell coordinates within a chunk
    pub fn world_to_local(world_x: f32, world_y: f32) -> (usize, usize) {
        (
            (world_x.rem_euclid(CHUNK_SIZE as f32)) as usize,
            (world_y.rem_euclid(CHUNK_SIZE as f32)) as usize,
        )
    }

    /// Mark chunk as clean (not dirty)
    pub fn mark_clean(&mut self) {
        self.dirty = false;
        self.dirty_cells.clear();
    }

    /// Get dirty cells in this chunk
    pub fn get_dirty_cells(&self) -> &std::collections::HashSet<(usize, usize)> {
        &self.dirty_cells
    }

    /// Get all cells in this chunk (for iteration)
    pub fn cells(&self) -> &[Cell] {
        &self.cells
    }

    /// Get mutable access to all cells (marks chunk as dirty)
    pub fn cells_mut(&mut self) -> &mut [Cell] {
        self.dirty = true;
        &mut self.cells
    }
}
