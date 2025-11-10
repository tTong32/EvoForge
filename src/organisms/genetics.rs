use bevy::prelude::*;
use smallvec::SmallVec;

/// Size of the genome (number of genes)
pub const GENOME_SIZE: usize = 32;

/// Genome representation - array of floating-point genes (0.0 to 1.0)
/// Each gene encodes a trait that affects organism behavior/characteristics
#[derive(Component, Debug, Clone)]
pub struct Genome {
    /// Genes stored as SmallVec for small genomes (avoids heap allocation)
    pub genes: SmallVec<[f32; GENOME_SIZE]>,
}

impl Genome {
    /// Create a new random genome
    /// Optimized: Uses fastrand for better performance
    pub fn random() -> Self {
        let mut rng = fastrand::Rng::new();
        let mut genes = SmallVec::new();
        for _ in 0..GENOME_SIZE {
            genes.push(rng.f32());
        }
        Self { genes }
    }
    
    /// Create a genome with specific genes
    pub fn new(genes: Vec<f32>) -> Self {
        let mut genome = SmallVec::new();
        for gene in genes.iter().take(GENOME_SIZE) {
            genome.push(gene.clamp(0.0, 1.0));
        }
        // Fill remaining slots if needed
        while genome.len() < GENOME_SIZE {
            genome.push(0.5);
        }
        Self { genes: genome }
    }
    
    /// Get a gene value (clamped to valid range)
    pub fn get_gene(&self, index: usize) -> f32 {
        if index < self.genes.len() {
            self.genes[index].clamp(0.0, 1.0)
        } else {
            0.5 // Default value
        }
    }
    
    /// Set a gene value (clamped to valid range)
    pub fn set_gene(&mut self, index: usize, value: f32) {
        if index < self.genes.len() {
            self.genes[index] = value.clamp(0.0, 1.0);
        }
    }
    
    /// Clone genome with optional mutations
    /// Optimized: Uses faster uniform mutation instead of expensive Box-Muller transform
    pub fn clone_with_mutation(&self, mutation_rate: f32) -> Self {
        let mut new_genes = SmallVec::new();
        let mut rng = fastrand::Rng::new();
        
        for &gene in self.genes.iter() {
            let mut new_gene = gene;
            
            // Apply mutation with probability
            if rng.f32() < mutation_rate {
                // Uniform mutation: add random value in range [-0.1, 0.1]
                // This is faster than Box-Muller and produces similar results for small mutations
                let mutation = (rng.f32() - 0.5) * 0.2;
                new_gene = (new_gene + mutation).clamp(0.0, 1.0);
            }
            
            new_genes.push(new_gene);
        }
        
        Self { genes: new_genes }
    }
    
    /// Crossover two genomes (sexual reproduction)
    /// Optimized: Uses faster uniform mutation instead of expensive Box-Muller transform
    pub fn crossover(parent_a: &Genome, parent_b: &Genome, mutation_rate: f32) -> Self {
        let mut rng = fastrand::Rng::new();
        let mut new_genes = SmallVec::new();
        
        // Uniform crossover: for each gene, randomly choose from parent A or B
        for i in 0..GENOME_SIZE {
            let gene_a = parent_a.get_gene(i);
            let gene_b = parent_b.get_gene(i);
            
            // 50/50 chance to choose from each parent
            let mut new_gene = if rng.bool() {
                gene_a
            } else {
                gene_b
            };
            
            // Apply mutation with probability
            if rng.f32() < mutation_rate {
                // Uniform mutation: add random value in range [-0.1, 0.1]
                let mutation = (rng.f32() - 0.5) * 0.2;
                new_gene = (new_gene + mutation).clamp(0.0, 1.0);
            }
            
            new_genes.push(new_gene);
        }
        
        Self { genes: new_genes }
    }
    
    /// Calculate genetic distance between two genomes (for speciation)
    pub fn distance(&self, other: &Genome) -> f32 {
        let mut sum = 0.0;
        let min_len = self.genes.len().min(other.genes.len());
        
        for i in 0..min_len {
            let diff = (self.genes[i] - other.genes[i]).abs();
            sum += diff * diff; // Squared difference
        }
        
        (sum / min_len as f32).sqrt() // Root mean squared difference
    }
}

/// Trait indices in the genome
/// Each trait is encoded by one or more genes
pub mod traits {
    use super::*;
    
    /// Speed trait (how fast the organism can move)
    pub const SPEED: usize = 0;
    
    /// Size trait (affects metabolism, collision, etc.)
    pub const SIZE: usize = 1;
    
    /// Base metabolic rate trait
    pub const METABOLISM_RATE: usize = 2;
    
    /// Movement cost trait
    pub const MOVEMENT_COST: usize = 3;
    
    /// Maximum energy trait
    pub const MAX_ENERGY: usize = 4;
    
    /// Reproduction cooldown trait
    pub const REPRODUCTION_COOLDOWN: usize = 5;
    
    /// Reproduction threshold trait (energy level needed to reproduce)
    pub const REPRODUCTION_THRESHOLD: usize = 6;
    
    /// Sensory range trait (how far organism can sense)
    pub const SENSORY_RANGE: usize = 7;
    
    /// Aggression trait (for future behavior)
    pub const AGGRESSION: usize = 8;
    
    /// Boldness trait (for future behavior)
    pub const BOLDNESS: usize = 9;
    
    /// Remaining genes reserved for future traits
    /// (genes 10-31 available for future expansion)
    
    /// Express a trait from genome using sigmoid activation
    /// Maps gene value [0.0, 1.0] to trait range [min, max]
    pub fn express_trait(genome: &Genome, gene_index: usize, min: f32, max: f32) -> f32 {
        let gene_value = genome.get_gene(gene_index);
        // Linear interpolation (can be changed to sigmoid for non-linear)
        min + gene_value * (max - min)
    }
    
    /// Express speed trait (0.5 to 20.0 units/sec)
    pub fn express_speed(genome: &Genome) -> f32 {
        express_trait(genome, SPEED, 0.5, 20.0)
    }
    
    /// Express size trait (0.3 to 3.0 units)
    pub fn express_size(genome: &Genome) -> f32 {
        express_trait(genome, SIZE, 0.3, 3.0)
    }
    
    /// Express metabolism rate trait (0.005 to 0.02 per second)
    pub fn express_metabolism_rate(genome: &Genome) -> f32 {
        express_trait(genome, METABOLISM_RATE, 0.005, 0.02)
    }
    
    /// Express movement cost trait (0.01 to 0.1)
    pub fn express_movement_cost(genome: &Genome) -> f32 {
        express_trait(genome, MOVEMENT_COST, 0.01, 0.1)
    }
    
    /// Express max energy trait (30.0 to 150.0)
    pub fn express_max_energy(genome: &Genome) -> f32 {
        express_trait(genome, MAX_ENERGY, 30.0, 150.0)
    }
    
    /// Express reproduction cooldown trait (500 to 2000 ticks)
    /// Higher values = longer between reproductions
    pub fn express_reproduction_cooldown(genome: &Genome) -> f32 {
        express_trait(genome, REPRODUCTION_COOLDOWN, 500.0, 2000.0)
    }
    
    /// Express reproduction threshold trait (0.5 to 0.9 energy ratio)
    /// Higher values = need more energy to reproduce
    pub fn express_reproduction_threshold(genome: &Genome) -> f32 {
        express_trait(genome, REPRODUCTION_THRESHOLD, 0.5, 0.9)
    }
    
    /// Express sensory range trait (5.0 to 50.0 units)
    pub fn express_sensory_range(genome: &Genome) -> f32 {
        express_trait(genome, SENSORY_RANGE, 5.0, 50.0)
    }
    
    /// Express aggression trait (for behavior decisions)
    pub fn express_aggression(genome: &Genome) -> f32 {
        express_trait(genome, AGGRESSION, 0.0, 1.0)
    }
    
    /// Express boldness trait (for behavior decisions)
    pub fn express_boldness(genome: &Genome) -> f32 {
        express_trait(genome, BOLDNESS, 0.0, 1.0)
    }
}

/// Default mutation rate (probability of mutation per gene)
pub const DEFAULT_MUTATION_RATE: f32 = 0.01; // 1% chance per gene

