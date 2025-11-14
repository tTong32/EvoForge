# 🧬 Evolutionary Ecosystem Simulator — Project Overview

A modular, open-ended simulation of biological evolution and ecosystems featuring dynamic environments, emergent behaviors, and millions of autonomous agents.  
This document summarizes all major systems, dependencies, and development milestones.

---

## 🧱 Core Vision

- **Goal:** Simulate natural selection and ecological interactions in a scalable virtual world.  
- **Scale:** 1200-1500 agents across a 2D world divided into evolving biomes.  
- **Language Stack:**  
  - **Backend:** Rust (performance, memory safety, parallelism)  
  - **Frontend / Visualization:** Rust+Bevy (for interactive visualization)  
  - **Data:** CSV / JSON logging for analytics  

---

## ⚙️ System Overview

### 1. **World System**
- **Structure:**  
  - 2D grid divided into *cells* (chunks), each tracking temperature, humidity, terrain type, elevation, resource density, etc.  
- **Features:**  
  - Dynamic climate simulation (temperature gradients, rainfall, weather cycles).  
  - Events: volcanic eruptions, droughts, floods, meteor strikes.  
  - Seasonal and long-term change (ice ages, global warming).  
  - Regional variation mapped with low-frequency climate noise and transient weather events (storms, heatwaves, droughts).  
- **Performance:** Use sparse chunks & parallel updates (Rust multithreading).

---

### 2. **Resource System**
- **Types:** Plant matter, minerals, sunlight, water, detritus, prey biomass.  
- **Properties:**  
  - Regeneration rate, decay rate, nutrient value.  
  - Resources evolve and mutate (e.g. plants adapt leaf shape or toxin production).  
  - Cells track local consumption pressure and adapt regeneration accordingly (resource overuse suppresses regrowth; scarcity rebounds).  
- **Flow:**  
  - Resources flow between neighboring cells (e.g. nutrient runoff, seed dispersal).  

---

### 3. **Organism System**
- **Entities:** Every living thing (animal, plant, fungus, microbe).  
- **Attributes:** Position, velocity, energy, age, size, color, sensory organs, reproduction cooldown, genome.  
- **Lifecycle:** Spawn → Grow → Reproduce → Die → Decompose.  
- **Interactions:** Predation, competition, symbiosis, mating, territorial behavior.  
- **Scaling:** Efficient entity-component-system (ECS) design (Bevy ECS or custom Rust ECS).

---

### 4. **Genetic System**
- **Representation:**  
  - Each organism has a *genome* (array of floating-point genes).  
  - Each gene encodes physical and behavioral traits.  
- **Mutation & Crossover:**  
  - Random mutations with Gaussian noise.  
  - Sexual reproduction mixes parental genes.  
- **Expression:**  
  - Traits are derived from weighted blends of multiple genes (primary loci + modifier loci controlling metabolism, risk tolerance, exploratory drive, etc.).  
  - Enables emergent traits like camouflage, speed, metabolism without hardcoding behaviors.

---

### 5. **Behavior System**
- **Goal:** Simulate adaptive, non-AI decision-making.  
- **Mechanism:**  
  - Each tick, organisms use local sensory data (sight, smell, energy) to pick an action (move, eat, mate, flee).  
  - Rule-based or small state-machine approach for scalability.  
  - Behavior affected by genetic parameters (e.g. “boldness”, “curiosity”, “aggression”).  
  - Memory-driven modifiers for hunger and threat accumulate over time, unlocking behaviors such as adaptive fleeing, sustained foraging, and long-range migration.  
- **Actions:** Wander, chase, forage, rest, reproduce, migrate.

---

### 6. **Metabolism & Energy System**
- **Energy Flow:**  
  - Organisms consume food/resources → convert to energy → expend on movement, growth, reproduction.  
  - Starvation or inefficiency → death → nutrients recycled into environment.  
- **Evolvable metabolism:** Genes can change digestion rate, energy efficiency, toxin resistance.

---

### 7. **Reproduction & Heredity**
- **Types:** Asexual (clone + mutation), sexual (gene crossover + mutation).  
- **Parent Selection:** Fitness-weighted random choice.  
- **Offspring Generation:**  
  - Offspring inherits genome from parent(s).  
  - Mutation rate and genome length are evolvable.
  - Clutch size, energy allocation per offspring, and mutation rate blending are trait-driven, enabling diverse reproduction strategies.

---

### 8. **Ecosystem & Emergent Dynamics**
- Natural trophic layers (producers → consumers → decomposers).  
- Biomes emerge (forests, deserts, tundra) via resource distribution + climate.  
- Population dynamics: extinction, radiation, co-evolution, speciation.  
- Environmental feedback loops (e.g., more plants → more herbivores → more predators).  

---

### 9. **Statistics, Logging & Visualization**
- Real-time data collection (population counts, mean traits, mutation frequency).  
- Heatmaps for resource density and organism distribution.  
- Time-series graphs for ecosystem health and diversity.  
- Visualization modes:  
  - Overview map (biomes, resources, organisms).  
  - Trait evolution charts over time.  

---

### 10. **Event & Climate System**
- **Events:** Volcanoes, floods, meteor strikes, disease outbreaks.  
- **Climate:** Dynamic temperature + humidity cycles (seasons, long-term drift).  
- **Impact:** Alters resource regrowth, mortality, migration, adaptation pressure.  

---

## 🔧 Implementation Strategies

### 1. **World System Implementation**

**Data Structures:**
- **Sparse Chunk Storage:** Use `HashMap<(i32, i32), Chunk>` for active chunks only (memory efficient for large worlds)
- **Chunk Structure:** Fixed-size array (e.g., 64×64 cells per chunk) stored in contiguous memory for cache efficiency
- **Cell Data:** Packed struct with bitfields where possible (temperature: f32, humidity: f32, elevation: u16, terrain: u8 enum, resource_density: [f32; 6])
- **Dirty Chunk Tracking:** Maintain a set of modified chunks each tick to minimize updates

**Parallelization:**
- **Chunk-Level Parallelism:** Use `rayon` or `std::thread` to process chunks in parallel (each chunk independent)
- **Work Stealing:** Distribute chunk updates across thread pool for load balancing
- **Spatial Partitioning:** Assign chunks to threads based on spatial locality to improve cache hits

**Climate Updates:**
- **Hierarchical Updates:** Global climate → regional climate → chunk-level → cell-level
- **Lazy Evaluation:** Only update cells when accessed or when climate changes significantly
- **Event System:** Use an event queue for discrete events (volcanoes, meteors) to avoid polling
- **Regional Noise & Events:** Blend low-frequency noise for spatial variation and spawn stochastic weather events that temporarily perturb temperature/humidity fields with smooth falloff.

**Performance Optimizations:**
- **SIMD Operations:** Use `packed_simd` or `std::arch` for vectorized temperature/humidity calculations
- **Reduced Precision:** Use `f32` instead of `f64` where acceptable (reduces memory by 50%)
- **Spatial Hashing:** Use spatial hash grid for fast neighbor lookups (O(1) average case)

---

### 2. **Resource System Implementation**

**Data Structures:**
- **Resource Map:** Store per-cell resource arrays as `[f32; RESOURCE_TYPE_COUNT]` (stack-allocated, cache-friendly)
- **Resource Types:** Enum or const array for type safety (PLANT, MINERAL, SUNLIGHT, WATER, DETRITUS, PREY)
- **Regeneration Cache:** Pre-compute regeneration rates per biome type to avoid per-tick calculations

**Flow Simulation:**
- **Diffusion Algorithm:** Use finite difference method with time-stepping (e.g., `new = old + dt * diffusion_rate * laplacian`)
- **Batch Processing:** Process all resource flows in a separate parallel pass after world updates
- **Boundary Conditions:** Handle chunk boundaries with ghost cells or explicit boundary checks

**Evolution/Mutation:**
- **Plant Traits:** Store evolvable traits (toxin level, growth rate) as metadata attached to resource producers
- **Lazy Mutation:** Only mutate plant types when reproduction occurs (not every tick)
- **Trait Storage:** Use small hashmap or sparse array for plant trait variations per chunk
- **Consumption Feedback:** Persist per-cell consumption pressure/adaptation factors so over-harvested resources recover slowly and climates stress-sensitive resources differently.

**Performance Optimizations:**
- **Separate Passes:** Resource regeneration → Flow → Consumption (clear separation enables parallelization)
- **Sparse Updates:** Only update cells with active resources or neighboring active cells
- **Quantization:** Round very small resource values to zero to avoid floating-point drift

---

### 3. **Organism System Implementation**

**ECS Architecture:**
- **Use Bevy ECS:** Leverage `bevy_ecs` for efficient component storage and querying
- **Component Design:**
  - `Position(f32, f32)` - separate from transform for spatial queries
  - `Velocity(f32, f32)` - for movement
  - `Energy(f32)` - current energy level
  - `Genome(Vec<f32>)` - genetic data (consider `SmallVec<[f32; 32]>` for small genomes)
  - `Age(u32)` - ticks since birth
  - `Size(f32)` - affects collision detection
  - `SpeciesId(u32)` - for clustering/speciation tracking
- **Sparse Components:** Use optional components for traits not all organisms have (e.g., `ReproductionCooldown`, `Territory`)

**Spatial Indexing:**
- **Grid-Based Spatial Hash:** Maintain a spatial hash grid (e.g., 16×16 cell buckets) for O(1) neighbor queries
- **Update Strategy:** Rebuild spatial index every N ticks or incrementally update on movement
- **Chunk-Based Queries:** Limit spatial queries to current chunk + neighbors (9 chunks max)

**Lifecycle Management:**
- **Death Queue:** Collect dead entities in a Vec, process removals at end of tick (avoids mutation during iteration)
- **Spawn Pool:** Pre-allocate entity pools for common organism types to reduce allocations
- **Component Archetypes:** Group organisms by component sets (e.g., animals vs plants) for better cache locality

**Performance Optimizations:**
- **Batch Processing:** Process organisms in chunks (e.g., 1000 at a time) for better cache locality
- **Parallel Processing:** Use `Query::par_for_each` for independent organism updates (Bevy supports this)
- **Early Exit:** Skip organisms with zero energy or invalid state before full processing

---

### 4. **Genetic System Implementation**

**Genome Representation:**
- **Fixed-Size Arrays:** Use `[f32; GENOME_SIZE]` or `SmallVec` for small genomes (avoids heap allocations)
- **Encoding Scheme:** Each gene index maps to a trait (e.g., `gene[0] = speed`, `gene[1] = size`, `gene[2] = eye_sensitivity`)
- **Normalization:** Keep genes in `[0.0, 1.0]` range via clamping or sigmoid for predictable trait mapping

**Mutation:**
- **Gaussian Noise:** `new_gene = old_gene + rng.sample(Gaussian(0.0, mutation_rate))`
- **Per-Gene Mutation:** Use bitmask or probability check to mutate individual genes (not all at once)
- **Adaptive Mutation:** Store mutation rate in genome itself (evolvable trait)

**Crossover (Sexual Reproduction):**
- **Uniform Crossover:** For each gene, randomly choose from parent A or B (50/50)
- **Single-Point Crossover:** Choose random split point, take genes [0..split] from parent A, [split..end] from parent B
- **Averaging:** For some traits, average parent genes: `offspring_gene = (parent_a + parent_b) / 2.0`

**Trait Expression:**
- **Lazy Evaluation:** Compute traits from genome only when needed (cache traits in component)
- **Trait Mapping Functions:** Use lookup table or match statement for gene → trait mapping
- **Sigmoid/Activation Functions:** Apply sigmoid to genes for non-linear trait scaling: `trait = sigmoid(gene * scale + bias)`
- **Multi-Gene Blends:** Weighted helper functions combine primary and modifier loci (e.g., fast-twitch vs. endurance, risk tolerance, migration drive) to create richer trade-offs without changing genome length.

**Performance Optimizations:**
- **SIMD for Crossover:** Use vectorized operations for bulk gene copying/merging
- **Mutation Cache:** Pre-generate mutation noise arrays for batch mutations
- **Genome Pooling:** Reuse genome vectors via object pooling to reduce allocations

---

### 5. **Behavior System Implementation**

**Decision-Making Architecture:**
- **State Machine:** Simple enum-based state (Wandering, Chasing, Eating, Fleeing, Mating, Resting)
- **Priority System:** Evaluate actions in priority order (survival > reproduction > exploration)
- **Sensory Input:** Query spatial hash for nearby organisms/resources within sensory radius
- **Behavior Memory:** Persist hunger/threat memories and migration targets to bias priorities over multiple ticks (e.g., sustained fleeing, opportunistic migration).

**Rule-Based Logic:**
- **Energy Thresholds:** `if energy < 0.3 * max_energy { prioritize_food() }`
- **Predator Detection:** Query spatial grid for nearby organisms with `predator_flag` component
- **Mate Detection:** Query for same species within range, check reproduction cooldown
- **Resource Seeking:** Query spatial grid for resource-rich cells within view distance

**Action Execution:**
- **Movement:** Update velocity toward target, clamp to max speed (genome-based)
- **Consumption:** Check collision with target, transfer energy, mark resource as consumed
- **Reproduction:** Check cooldown, find mate, create offspring entity with crossover genome

**Performance Optimizations:**
- **Spatial Query Caching:** Cache nearby entities/resources for multiple behavior checks
- **Early Termination:** Skip low-priority behaviors if high-priority action is selected
- **Batch Sensory Queries:** Process all sensory inputs in one spatial query pass
- **Behavior Archetypes:** Pre-compile behavior trees for common organism types (plants simpler than animals)

---

### 6. **Metabolism & Energy System Implementation**

**Energy Storage:**
- **Component:** `Energy(f32)` component updated each tick
- **Max Energy:** Store as `MaxEnergy(f32)` component or derive from size/genome
- **Energy Efficiency:** Trait derived from genome (affects conversion rate)

**Consumption:**
- **Resource → Energy:** `energy_gained = resource_consumed * conversion_efficiency * genome.digestion_rate`
- **Collision Detection:** Use spatial hash to find organisms/resources at same position
- **Consumption Rate:** Limit consumption per tick based on size/genome (prevents instant consumption)

**Expenditure:**
- **Base Metabolism:** `energy_cost = base_rate * size * genome.metabolism_rate * dt`
- **Movement Cost:** `energy_cost += velocity.length() * genome.movement_cost_multiplier * dt`
- **Growth Cost:** `energy_cost += growth_rate * size * dt` (if growing)
- **Reproduction Cost:** One-time cost when reproducing (deduct from parent)

**Death & Recycling:**
- **Starvation:** `if energy <= 0.0 { mark_for_death() }`
- **Decomposition:** On death, convert organism biomass to detritus in current cell
- **Nutrient Recycling:** `detritus += organism.size * nutrient_value`

**Performance Optimizations:**
- **Batch Energy Updates:** Process all energy updates in parallel (independent operations)
- **Lazy Death Checks:** Only check death condition when energy changes significantly
- **Quantization:** Round very small energy values to zero to avoid micro-fluctuations

---

### 7. **Reproduction & Heredity Implementation**

**Reproduction Types:**
- **Asexual:** Clone parent genome, apply mutations, spawn at parent location
- **Sexual:** Find mate within range, perform crossover, apply mutations, spawn near parents

**Parent Selection:**
- **Fitness-Weighted:** `fitness = energy * age * size` (or more complex formula)
- **Tournament Selection:** Randomly sample N organisms, pick fittest
- **Roulette Wheel:** Use cumulative fitness for weighted random selection

**Offspring Generation:**
- **Genome Creation:** Allocate new genome vector, perform crossover/mutation
- **Energy Split:** Parent(s) lose energy (e.g., `parent_energy *= 0.7`)
- **Spawn Position:** Place near parent with small random offset (avoid overlap)
- **Initial Energy:** Set offspring energy to fraction of parent's energy
- **Clutch Planning:** Use trait-driven clutch size, per-offspring energy shares, and averaged parental mutation rates to shape reproductive strategy.

**Mutation Rate Evolution:**
- **Self-Modifying:** Store mutation rate as gene in genome (mutates itself)
- **Clamp Rates:** Keep mutation rate in reasonable range (e.g., 0.001 to 0.1)

**Cooldown System:**
- **Reproduction Cooldown Component:** `ReproductionCooldown(u32)` ticks remaining
- **Decrement:** Reduce cooldown each tick, allow reproduction when zero
- **Genome-Based:** Cooldown duration derived from genome (evolvable trait)

**Performance Optimizations:**
- **Batch Reproduction:** Collect all reproduction events, process at end of tick
- **Offspring Pooling:** Pre-allocate entity pools for common organism types
- **Genome Pooling:** Reuse genome vectors to reduce allocations

---

### 8. **Ecosystem & Emergent Dynamics Implementation**

**Trophic Layers:**
- **Trait-Based Classification:** Use genome traits to determine trophic level (producer/consumer/decomposer)
- **Producer Logic:** Plants generate energy from sunlight/resources (no consumption needed)
- **Consumer Logic:** Animals consume other organisms/resources
- **Decomposer Logic:** Fungi/microbes consume detritus, convert to nutrients

**Biome Emergence:**
- **Resource Clustering:** Use K-means or density-based clustering on resource distribution
- **Biome Classification:** Label chunks based on dominant resource type and climate
- **Biome Persistence:** Update biome labels periodically (not every tick) to reduce overhead

**Population Dynamics:**
- **Species Tracking:** Use `SpeciesId` component to group organisms
- **Population Counts:** Maintain `HashMap<SpeciesId, u32>` updated incrementally
- **Extinction Detection:** Mark species as extinct if population = 0 for N consecutive ticks
- **Speciation:** Detect when populations diverge (genetic distance > threshold) and assign new species IDs

**Feedback Loops:**
- **Plant → Herbivore:** More plants → more herbivores can survive → more herbivore reproduction
- **Herbivore → Predator:** More herbivores → more predators can survive
- **Detritus → Plants:** More decomposers → more nutrients → more plant growth

**Performance Optimizations:**
- **Lazy Biome Updates:** Update biome classifications every N ticks (not every tick)
- **Incremental Population Tracking:** Update counts only when organisms spawn/die
- **Spatial Clustering:** Use spatial hash to accelerate population density calculations

---

### 9. **Statistics, Logging & Visualization Implementation**

**Data Collection:**
- **Stats Struct:** `SimulationStats { population_counts: HashMap<SpeciesId, u32>, mean_traits: Vec<f32>, mutation_rate: f32, ... }`
- **Incremental Updates:** Update stats during simulation (not post-processing)
- **Sampling:** Collect stats every N ticks (not every tick) to reduce overhead

**Logging:**
- **Async Logging:** Use `tokio` or `std::thread` for background CSV/JSON writing
- **Buffered Writes:** Batch log entries, flush periodically (e.g., every 100 ticks)
- **Compression:** Use `flate2` or `zstd` for compressed log files
- **Format:** CSV for analytics, JSON for structured queries

**Visualization (Bevy):**
- **Separate Render Thread:** Run visualization in separate thread/process if possible
- **Level of Detail (LOD):** Render fewer organisms at zoom out (sample organisms for display)
- **Heatmap Rendering:** Use texture/color map for resource density visualization
- **Time-Series Graphs:** Use `bevy_ui` or external plotting library (e.g., `plotters`)

**Performance Optimizations:**
- **Stats Sampling:** Only collect stats every 10-100 ticks (configurable)
- **Selective Logging:** Log only essential metrics in production, full logs in debug mode
- **Visualization Culling:** Only render organisms visible in viewport
- **Texture Caching:** Cache heatmap textures, update only when data changes

#### Advanced: Data Collection & Analytics Architecture

**Segmentation Keys (how data is grouped):**
- `biome_id` (stable label per chunk, updated periodically)
- `chunk_id` (coarse spatial bin; useful for debugging hotspots)
- `species_id` and `trophic_level` (producer/consumer/decomposer)
- `env_bucket = (temp_bin, humidity_bin, elevation_bin, terrain)` for environment stratification
- `time_bucket` (tick, 100-tick, 1000-tick windows)
- `event_type`/`event_id` for impact analysis during climate/events

**Hierarchical Aggregation Pipeline:**
- Per-thread, per-chunk collectors accumulate `ChunkStats` (lock-free, thread-local) during the tick.
- A biome reducer merges `ChunkStats` → `BiomeStats` keyed by `biome_id` and optionally `(biome_id, env_bucket)`.
- A world reducer merges `BiomeStats` → `WorldStats` for global summaries.
- Use mergeable structures at all levels: Welford mean/variance, histograms (fixed-bin/hdr), t-digest, counters, sketches.

**Biome & Environment Bucketing:**
- Assign a stable `BiomeId` to each chunk; recompute labels every N ticks; when a chunk’s biome changes, re-key future aggregates (don’t rewrite history).
- Discretize environment into small integer bins: `temp_bin`, `humidity_bin`, `elev_bin` (e.g., 8–16 bins each) plus `terrain`.
- Maintain stats both by `biome_id` and by `env_bucket` to answer within-biome and cross-biome questions.

**Metrics Taxonomy (examples):**
- Population & demography: alive, births, deaths, age distribution, lifespan histogram.
- Traits: per-trait mean/variance, fixed-bin histograms over [0,1], optional sampled correlations.
- Energy/metabolism: energy mean/variance, intake vs expenditure by source/action.
- Resources: production, regeneration, consumption, net balance per resource type.
- Behavior: action selection rates, time-in-state, step-length distribution, migration rate.
- Fitness proxies: reproduction success per species/biome, offspring count distribution.
- Diversity: Shannon/Simpson indices; mean genetic distance (sample-based) per biome/species.
- Interactions: predator→prey edges (counts, biomass), competition encounters, symbiosis events.
- Event impacts: before/after deltas for climate/events in affected regions.

**Online Algorithms & Sketches (mergeable):**
- Welford for mean/variance; counters for totals; EMAs for smoothed trends.
- Histograms: fixed-bin for bounded traits; `hdrhistogram` for wide-range values.
- Quantiles: `t-digest` for P50/P90/P99; merge digests across threads/regions.
- Cardinality: HyperLogLog for unique counts (e.g., unique species per biome/window).
- Top-K: Count-Min Sketch + small heap per key for heavy hitters (species, interactions).
- Reservoir sampling per key for exemplar agents/traits.
- Ring buffers for sliding windows; configurable window sizes (e.g., 100/1k/10k ticks).

**Windows & Sampling Strategy:**
- Multi-scale time buckets: tick-level (debug), 100-tick (ops), 1000-tick (research).
- Stratified agent sampling by `(biome_id, species_id)` at 1–5% to cap cost; auto-increase during anomalies/events.
- Burst mode: temporarily elevate sampling/logging near events, then decay back.
- Backpressure: if writer queue is full, drop non-critical metrics first.

**Interaction & Migration Flows:**
- Record migrations as edges `(src_biome, dst_biome, species_id, time_bucket)` with counts and optional biomass.
- Maintain OD matrices per species and aggregated; expose flux heatmaps.
- Predator-prey matrix `(pred_species, prey_species, biome_id, time_bucket)` with counts/biomass consumed.

**Storage Layout & Schema (append-only):**
- Directory partitioning:
  - `data/logs/{run_id}/metrics={name}/biome={biome_id}/tick={start}-{end}.parquet`
  - `data/logs/{run_id}/events/{tick}.jsonl`
- Core schemas:
  - biome_stats: `tick_start, tick_end, biome_id, env_bucket, metric, value`
  - species_stats: `tick_start, tick_end, biome_id, species_id, metric, value`
  - interactions: `tick_start, tick_end, biome_id, src_species, dst_species, type, count, biomass`
  - migration: `tick_start, tick_end, src_biome, dst_biome, species_id, count`
- Compression: zstd (level 3–6); row groups sized ~5–20 MB for scan efficiency.

**Implementation in ECS:**
- `StatsPlugin` schedules:
  - `collect_chunk_stats_system` (parallel; thread-local accumulators)
  - `reduce_biome_stats_system` (merge by keys)
  - `reduce_world_stats_system`
  - `flush_stats_system` (async writer with bounded queue)
- Compact IDs: `BiomeId: u16`, `SpeciesId: u32`, `EnvBucketId: u16`.

**Controls & Overhead Management:**
- Config-driven toggles/frequencies (TOML in `data/configs/stats.toml`).
- Per-metric budgets (max keys, top-K truncation) and watchdog to disable expensive metrics if frame time exceeds threshold.

**Example Aggregation Keys:**
- `(time=1k, biome_id=Desert, species_id=5)`
- `(time=100, env_bucket=(temp=2, humidity=1, elev=3), trophic=consumer)`
- `(tick=now, src_biome=Forest, dst_biome=Grassland, species_id=12)`

---

### 10. **Event & Climate System Implementation**

**Event System:**
- **Event Queue:** Use `VecDeque<Event>` for discrete events (volcano, meteor, flood)
- **Event Types:** Enum with parameters (e.g., `Volcano { location: (f32, f32), intensity: f32 }`)
- **Scheduled Events:** Store events with timestamps, process when time reached
- **Random Events:** Use Poisson process or weighted random selection for event generation

**Climate Simulation:**
- **Global Climate State:** `GlobalClimate { temperature: f32, humidity: f32, season: f32, ... }`
- **Regional Variation:** Apply noise/perlin noise to global climate for regional variation
- **Time-Based Cycles:**
  - **Seasons:** `temperature = base_temp + season_amplitude * sin(2π * time / season_period)`
  - **Long-Term Drift:** Add slow random walk to base temperature (ice ages, global warming)

**Event Effects:**
- **Volcano:** Increase temperature in radius, block sunlight, create ash resources
- **Meteor:** Destroy organisms in impact radius, create crater, add minerals
- **Flood:** Increase water resources, kill low-lying organisms
- **Drought:** Reduce water resources, increase mortality

**Performance Optimizations:**
- **Event Batching:** Process all events in one pass at start/end of tick
- **Lazy Climate Updates:** Only update climate-dependent systems when climate changes significantly
- **Spatial Event Queries:** Use spatial hash to find affected cells quickly

---

## 🧩 Development Timeline (Suggested Order)

| Stage | System | Description |
|-------|---------|-------------|
| 1 | **Core Framework** | Create project structure, ECS framework, world grid. |
| 2 | **World & Resource Simulation** | Implement terrain, resources, and climate updates. |
| 3 | **Organisms (Basic)** | Add agents with position, energy, metabolism, simple behavior. |
| 4 | **Genetics & Reproduction** | Add genome encoding, mutation, crossover. |
| 5 | **Behavior System** | Implement decision rules (wander, eat, flee, mate). |
| 6 | **Resource-Organism Interaction** | Link eating/metabolism with resource map. |
| 7 | **Visualization & Logging** | Add real-time data collection and map visualization. |
| 8 | **Emergent Ecosystem Tuning** | Tune rates until emergent biomes and dynamics form. |
| 9 | **Advanced Systems** | Add speciation, climate events, disease, co-evolution. |
| 10 | **Performance Scaling** | Parallelize updates, optimize data layout, add chunking. |

---

## 🧠 Design Notes

- **ECS architecture** ensures modularity and parallelism.  
- **Stateless updates** enable parallel chunk computation.  
- **Rust + Bevy ECS** provides scalability to millions of agents.  
- **Open-ended evolution** emerges from fitness landscape + resource feedback loops.  
- **Visualization** handled by separate async process for performance.

---

## 📦 Recommended Dependencies & Libraries

**Core Framework:**
- `bevy` - ECS framework and visualization engine
- `bevy_ecs` - Entity Component System (if using standalone)
- `rayon` - Data parallelism and work-stealing scheduler

**Spatial Indexing:**
- `spatial-rs` or custom spatial hash grid - Fast neighbor queries
- `rstar` - R-tree for spatial queries (alternative to hash grid)

**Random Number Generation:**
- `rand` - Random number generation
- `fastrand` - Faster RNG for large-scale simulations
- `noise` or `libnoise` - Perlin/simplex noise for climate generation

**Data Structures:**
- `smallvec` - Stack-allocated vectors for small genomes
- `slotmap` - Efficient entity ID management (alternative to Bevy's entities)
- `dashmap` - Concurrent hashmap for thread-safe stats

**Serialization & Logging:**
- `serde` + `serde_json` - JSON serialization
- `csv` - CSV writing for analytics
- `flate2` or `zstd` - Compression for log files
- `tracing` - Structured logging framework

**Mathematics & Simulation:**
- `nalgebra` or `glam` - Vector math (Bevy uses `glam` by default)
- `simd` or `std::arch` - SIMD operations for vectorized calculations

**Visualization Extensions:**
- `plotters` - Plotting library for time-series graphs
- `bevy_mod_picking` - Mouse interaction for Bevy
- `bevy_inspector_egui` - Runtime inspection of ECS components

**Performance Profiling:**
- `puffin` - Profiling tool compatible with Bevy
- `tracing` - Performance tracing
- `criterion` - Benchmarking framework

**Analytics & Sketching:**
- `hdrhistogram` - low-overhead histograms and mergeable quantiles
- `tdigest` - streaming percentile estimation (mergeable across reducers)
- `quantiles` - additional streaming quantile estimators
- `probabilistic-collections` - HyperLogLog, Count-Min Sketch for unique counts/top-K
- `arrow2`, `parquet`, `polars` - columnar storage and offline analytics

---

## ⚡ Performance & Memory Management Strategies

**Memory Allocation:**
- **Object Pooling:** Pre-allocate pools for frequently created/destroyed objects (organisms, genomes, events)
- **Arena Allocators:** Use `bumpalo` or custom arena allocators for temporary allocations during tick processing
- **Stack Allocation:** Use `SmallVec` or fixed-size arrays for small collections (genomes, trait vectors)
- **Avoid Allocations in Hot Paths:** Minimize `Vec::push`, `HashMap::insert` during simulation ticks

**Cache Optimization:**
- **Structure of Arrays (SoA):** Store components in contiguous arrays (Bevy does this automatically)
- **Data Locality:** Keep related data together (position + velocity, energy + metabolism)
- **Prefetching:** Use `std::hint::black_box` and explicit prefetch hints for predictable access patterns
- **False Sharing Prevention:** Separate frequently-written data to different cache lines

**Parallelization Strategy:**
- **Chunk Independence:** Ensure chunks can be processed in parallel without synchronization
- **Read-Write Separation:** Use double-buffering or immutable snapshots for world state
- **Work Granularity:** Balance chunk size (too small = overhead, too large = load imbalance)
- **Thread Affinity:** Pin threads to CPU cores for better cache locality

**Update Frequency Optimization:**
- **Variable Timestep:** Use fixed timestep for physics, variable for non-critical systems
- **LOD by Distance:** Update distant organisms less frequently (e.g., every 10 ticks)
- **Dirty Tracking:** Only update systems/components that have changed
- **Hierarchical Updates:** Update global systems less frequently than local ones

**Algorithmic Optimizations:**
- **Spatial Culling:** Only process organisms/resources in visible/viewport chunks
- **Early Exit:** Break loops when conditions are met (e.g., found food, detected predator)
- **Approximation:** Use approximate algorithms for non-critical calculations (e.g., distance approximation)
- **Batch Operations:** Group similar operations (e.g., all energy updates, all movement updates)

**Profiling & Measurement:**
- **Identify Bottlenecks:** Use `cargo flamegraph` or `puffin` to find hot paths
- **Benchmark Critical Paths:** Use `criterion` to measure optimization impact
- **Memory Profiling:** Use `dhat` or `valgrind` to track memory usage and leaks
- **Metrics Collection:** Track frame time, tick time, memory usage, entity count in real-time

---

## 📊 Future Extensions

- Genetic drift and speciation tracking.  
- Climate-driven migration and extinction events.  
- Procedural terrain and biome generation.  
- Trophic network visualizer.  
- AI-assisted ecosystem summarizer (auto-analyzes evolutionary trends).  

---

## 🧩 Repository Layout (Planned)
evo_sim/
├── Cargo.toml
├── src/
│   ├── main.rs 
│   ├── world/
│   ├── resources/
│   ├── organisms/
│   ├── genetics/
│   ├── behavior/
│   ├── metabolism/
│   ├── visualization/
│   └── utils/
├── data/
│   ├── logs/
│   ├── configs/
│   └── outputs/
└── docs/
└── PROJECT_OVERVIEW.md


---

**Author’s Note:**  
This project is designed for scientific and creative exploration — to watch complexity and life emerge from simple, local rules.


