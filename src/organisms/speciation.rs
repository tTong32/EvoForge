use crate::organisms::components::SpeciesId;
use crate::organisms::genetics::Genome;
use bevy::prelude::*;
use std::collections::HashMap;

/// Speciation threshold - genetic distance below which organisms are same species
/// Step 8: Now configurable via EcosystemTuning
pub const DEFAULT_SPECIATION_THRESHOLD: f32 = 0.15;

/// Tracks species information for speciation system
#[derive(Resource)]
pub struct SpeciesTracker {
    /// Map from SpeciesId to representative genome (centroid)
    species_centroids: HashMap<u32, Genome>,
    /// Next available species ID
    next_species_id: u32,
    /// Counter for speciation updates (update periodically)
    update_counter: u32,
    /// Speciation threshold (configurable via tuning)
    threshold: f32,
}

impl Default for SpeciesTracker {
    fn default() -> Self {
        Self {
            species_centroids: HashMap::new(),
            next_species_id: 0,
            update_counter: 0,
            threshold: DEFAULT_SPECIATION_THRESHOLD,
        }
    }
}

impl SpeciesTracker {
    /// Create with custom threshold
    pub fn with_threshold(threshold: f32) -> Self {
        Self {
            threshold,
            ..Default::default()
        }
    }

    /// Find or assign species ID for a genome
    pub fn find_or_create_species(&mut self, genome: &Genome) -> SpeciesId {
        // Check if genome matches any existing species (within threshold)
        for (species_id, centroid) in &self.species_centroids {
            let distance = genome.distance(centroid);
            if distance < self.threshold {
                return SpeciesId::new(*species_id);
            }
        }

        // No match found - create new species
        let new_id = self.next_species_id;
        self.next_species_id += 1;
        self.species_centroids.insert(new_id, genome.clone());
        SpeciesId::new(new_id)
    }

    /// Update species centroids periodically based on average genomes
    pub fn update_centroids(
        &mut self,
        organisms: &[(Entity, &Genome, &SpeciesId)],
    ) {
        // Group organisms by species
        let mut species_genomes: HashMap<u32, Vec<&Genome>> = HashMap::new();
        
        for (_entity, genome, species_id) in organisms {
            species_genomes
                .entry(species_id.value())
                .or_insert_with(Vec::new)
                .push(genome);
        }

        // Update centroids with average genome per species
        for (species_id, genomes) in species_genomes {
            if genomes.is_empty() {
                continue;
            }

            // Calculate average genome
            let genome_size = genomes[0].genes.len();
            let mut avg_genes = Vec::with_capacity(genome_size);
            avg_genes.resize(genome_size, 0.5);
            let mut avg_genome = Genome::new(avg_genes);
            
            for genome in &genomes {
                for i in 0..avg_genome.genes.len().min(genome.genes.len()) {
                    avg_genome.genes[i] += genome.genes[i];
                }
            }

            // Average the values
            let count = genomes.len() as f32;
            for i in 0..avg_genome.genes.len() {
                avg_genome.genes[i] /= count;
                avg_genome.genes[i] = avg_genome.genes[i].clamp(0.0, 1.0);
            }

            self.species_centroids.insert(species_id, avg_genome);
        }
    }

    /// Get number of species
    pub fn species_count(&self) -> usize {
        self.species_centroids.len()
    }

    /// Clean up extinct species (remove species with no organisms)
    pub fn cleanup_extinct(&mut self, active_species: &std::collections::HashSet<u32>) {
        self.species_centroids.retain(|id, _| active_species.contains(id));
    }

    /// Get all species IDs
    pub fn get_all_species(&self) -> Vec<u32> {
        self.species_centroids.keys().copied().collect()
    }

    /// Get the representative genome for a species
    pub fn get_genome(&self, species_id: u32) -> Option<&Genome> {
        self.species_centroids.get(&species_id)
    }
}

/// Update species assignments periodically (Step 8 - Speciation)
pub fn update_speciation(
    mut tracker: ResMut<SpeciesTracker>,
    tuning: Option<Res<crate::organisms::EcosystemTuning>>, // Step 8: Optional tuning
    mut query: Query<(Entity, &Genome, &mut SpeciesId), With<crate::organisms::components::Alive>>,
) {
    // Update threshold from tuning if available
    if let Some(tuning) = tuning {
        tracker.threshold = tuning.speciation_threshold;
    }
    tracker.update_counter += 1;
    
    // Update centroids every 100 ticks (not every tick for performance)
    if tracker.update_counter % 100 == 0 {
        let organisms: Vec<_> = query.iter().collect();
        let previous_count = tracker.species_count();
        tracker.update_centroids(&organisms);
        let new_count = tracker.species_count();
        
        if new_count != previous_count {
            info!("[SPECIATION] Species count changed: {} -> {}", previous_count, new_count);
        }
    }

    // Reassign species IDs based on current centroids (every 500 ticks for performance)
    if tracker.update_counter % 500 == 0 {
        let mut updated_count = 0;
        for (_entity, genome, mut species_id) in query.iter_mut() {
            let new_species = tracker.find_or_create_species(genome);
            if new_species != *species_id {
                *species_id = new_species;
                updated_count += 1;
            }
        }
        
        let species_count = tracker.species_count();
        if updated_count > 0 || tracker.update_counter % 5000 == 0 {
            info!(
                "[SPECIATION] Updated {} organism species assignments | Total species: {}",
                updated_count,
                species_count
            );
        }
    }
}

