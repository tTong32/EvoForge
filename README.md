# 🧬 Evolution Simulator 2.0

A modular, open-ended simulation of biological evolution and ecosystems featuring dynamic environments, emergent behaviors, and millions of autonomous agents.

## 🚀 Current Status

**Steps 1-10: Core Systems + Advanced Systems + Performance** ✅ **COMPLETE**

The simulation now includes:
- ✅ **Step 1: Core Framework** - Project structure, Bevy ECS, world grid, chunks, cells
- ✅ **Step 2: World & Resource Simulation** - Climate system, resource regeneration, terrain
- ✅ **Step 3: Organisms (Basic)** - Organism components, spawning, metabolism, energy
- ✅ **Step 4: Genetics & Reproduction** - Genome encoding, mutation, crossover, trait expression
- ✅ **Step 5: Behavior System** - State machine, decision-making, sensory data, memory
- ✅ **Step 6: Resource-Organism Interaction** - Eating, metabolism, energy flow
- ✅ **Step 7: Visualization & Logging** - Real-time rendering, CSV logging, camera controls
- ✅ **Step 8: Emergent Ecosystem Tuning** - COMPLETE
  - ✅ Speciation system - tracks and differentiates species based on genetic distance
  - ✅ Tuning parameters - centralized ecosystem balance configuration
  - ✅ Ecosystem statistics - population and trait tracking
  - ✅ Species assignment during spawning and reproduction
  - ✅ Balanced resource regeneration/consumption rates
  - ✅ Tuned reproduction rates for stability (prevents instant spawning)
  - ✅ Improved behavior differentiation between producers, consumers, and decomposers
- ✅ **Step 9: Advanced Systems** - COMPLETE
  - ✅ Major disasters system - volcanoes, meteors, floods, droughts that affect organisms and terrain
  - ✅ Disease system - spreading mechanics, resistance traits, species-specific diseases
  - ✅ Co-evolution system - tracks predator-prey, competitive, and mutualistic relationships
  - ✅ Evolvable defenses - physical, chemical, behavioral defenses that evolve over time
- ✅ **Step 10: Performance Scaling** - COMPLETE
  - ✅ Parallelized chunk processing using rayon for independent chunk updates
  - ✅ Optimized resource flow with parallel chunk processing
  - ✅ Leveraged Bevy's automatic system-level parallelization for organism updates

## 📁 Project Structure

```
evolution-sim/
├── Cargo.toml              # Project dependencies
├── src/
│   ├── main.rs             # Application entry point
│   ├── world/              # World system module
│   │   ├── mod.rs          # World plugin and module exports
│   │   ├── cell.rs         # Cell data structure (environment, resources)
│   │   ├── chunk.rs        # Chunk management (64x64 cells)
│   │   ├── grid.rs         # Sparse world grid with HashMap storage
│   │   ├── climate.rs      # Climate simulation
│   │   ├── resources.rs    # Resource regeneration and flow
│   │   ├── terrain.rs      # Terrain generation
│   │   └── events.rs       # Major disaster events (Step 9)
│   ├── organisms/          # Organism system module
│   │   ├── mod.rs          # Organism plugin
│   │   ├── components.rs   # Organism components
│   │   ├── genetics.rs     # Genome and trait expression
│   │   ├── behavior.rs     # Behavior system and decision-making
│   │   ├── systems.rs      # Organism update systems
│   │   ├── speciation.rs   # Species tracking and differentiation (Step 8)
│   │   ├── tuning.rs       # Ecosystem tuning parameters (Step 8)
│   │   ├── ecosystem_stats.rs # Ecosystem statistics (Step 8)
│   │   ├── disease.rs      # Disease system with spreading mechanics (Step 9)
│   │   └── coevolution.rs  # Co-evolution system tracking species interactions (Step 9)
│   ├── visualization/      # Visualization module
│   │   ├── mod.rs          # Visualization plugin
│   │   ├── organisms.rs    # Organism sprite rendering
│   │   └── camera.rs       # Camera controls
│   └── utils/              # Utility functions
│       ├── mod.rs          # Coordinate conversion, math utilities
│       └── spatial_hash.rs # Spatial hashing for efficient queries
├── data/
│   ├── logs/               # Simulation logs (CSV files)
│   ├── configs/            # Configuration files
│   └── outputs/            # Output data
└── docs/
    └── PROJECT_OVERVIEW.md # Complete project documentation
```

## 🏗️ Architecture

### World System

The world is divided into **chunks** (64×64 cells each), stored sparsely in a `HashMap`. This allows:
- Memory efficiency (only active chunks in memory)
- Parallel processing of independent chunks
- Lazy loading of distant regions

### Cell Structure

Each cell contains:
- **Environmental data**: temperature, humidity, elevation, terrain type
- **Resource densities**: 6 resource types (Plant, Mineral, Sunlight, Water, Detritus, Prey)

### ECS Framework

Using Bevy ECS for:
- Component-based architecture
- Parallel system execution
- Efficient data storage (Structure of Arrays)

## 🛠️ Building

```bash
# Check compilation
cargo check

# Build in release mode
cargo build --release

# Run the simulator
cargo run
```

## 🎮 Controls

- **Arrow Keys / WASD**: Pan camera
- **+ / -**: Zoom in/out
- **0**: Reset zoom
- **R**: Reset camera position

## 👁️ Visualization

The simulator displays organisms as colored sprites:
- **Green**: Producers (plants, algae)
- **Red**: Consumers (animals)
- **Purple**: Decomposers (fungi, bacteria)

Colors vary based on:
- Energy level (brighter = more energy)
- Species ID (slight hue variation)
- Disease status (sickly yellow-green tint for infected organisms)

**Disease Visualization:**
- Infected organisms show a sickly yellow-green color tint
- Orange-red pulsing indicators appear around infected organisms
- More severe infections = more pronounced color shift

**Disaster Visualization:**
- **Volcanoes**: Red/orange circles with pulsing effects (heat and ash)
- **Meteors**: Dark red/brown circles (impact craters)
- **Floods**: Blue circles (water accumulation)
- **Droughts**: Yellow/brown circles (dry conditions)
- Disasters pulse to show activity and fade out as they expire

## 📋 Next Steps

Following the development timeline:

1. ✅ **Core Framework** - Complete
2. ✅ **World & Resource Simulation** - Complete
3. ✅ **Organisms (Basic)** - Complete
4. ✅ **Genetics & Reproduction** - Complete
5. ✅ **Behavior System** - Complete
6. ✅ **Resource-Organism Interaction** - Complete
7. ✅ **Visualization & Logging** - Complete
8. ✅ **Emergent Ecosystem Tuning** - COMPLETE
   - ✅ Speciation system implemented
   - ✅ Tuning parameters resource created
   - ✅ Ecosystem statistics collection
   - ✅ Balanced resource regeneration/consumption rates
   - ✅ Tuned reproduction rates for stability
   - ✅ Improved behavior differentiation between organism types
9. ✅ **Advanced Systems** - COMPLETE
   - ✅ Major disasters system (volcanoes, meteors, floods, droughts)
   - ✅ Disease system with spreading mechanics and resistance
   - ✅ Co-evolution system tracking species interactions
   - ✅ Evolvable defenses (physical, chemical, behavioral, escape capability)
   - ✅ Predator-prey, competitive, and mutualistic relationships
10. ✅ **Performance Scaling** - COMPLETE
   - ✅ Parallelized chunk processing using rayon (climate updates, resource regeneration, resource flow)
   - ✅ Optimized world update systems for multi-threaded execution
   - ✅ Leveraged Bevy's automatic system-level parallelization
   - ✅ Improved data locality and cache efficiency

## 📚 Documentation

See `PROJECT_OVERVIEW.md` for complete system documentation, implementation strategies, and design decisions.

## 🧪 Testing

```bash
# Run tests (when implemented)
cargo test
```

## 📝 License

See LICENSE file for details.

