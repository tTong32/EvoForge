# 🧬 Evolution Simulator 2.0

A modular, open-ended simulation of biological evolution and ecosystems featuring dynamic environments, emergent behaviors, and millions of autonomous agents.

## 🚀 Current Status

**Steps 1-7: Core Systems** ✅ **COMPLETE**

The simulation now includes:
- ✅ **Step 1: Core Framework** - Project structure, Bevy ECS, world grid, chunks, cells
- ✅ **Step 2: World & Resource Simulation** - Climate system, resource regeneration, terrain
- ✅ **Step 3: Organisms (Basic)** - Organism components, spawning, metabolism, energy
- ✅ **Step 4: Genetics & Reproduction** - Genome encoding, mutation, crossover, trait expression
- ✅ **Step 5: Behavior System** - State machine, decision-making, sensory data, memory
- ✅ **Step 6: Resource-Organism Interaction** - Eating, metabolism, energy flow
- ✅ **Step 7: Visualization & Logging** - Real-time rendering, CSV logging, camera controls

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
│   │   └── terrain.rs      # Terrain generation
│   ├── organisms/          # Organism system module
│   │   ├── mod.rs          # Organism plugin
│   │   ├── components.rs   # Organism components
│   │   ├── genetics.rs     # Genome and trait expression
│   │   ├── behavior.rs     # Behavior system and decision-making
│   │   └── systems.rs      # Organism update systems
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

## 📋 Next Steps

Following the development timeline:

1. ✅ **Core Framework** - Complete
2. ✅ **World & Resource Simulation** - Complete
3. ✅ **Organisms (Basic)** - Complete
4. ✅ **Genetics & Reproduction** - Complete
5. ✅ **Behavior System** - Complete
6. ✅ **Resource-Organism Interaction** - Complete
7. ✅ **Visualization & Logging** - Complete
8. ⏭️ **Emergent Ecosystem Tuning** - Tune rates until emergent biomes form
9. ⏭️ **Advanced Systems** - Add speciation, climate events, disease, co-evolution
10. ⏭️ **Performance Scaling** - Parallelize updates, optimize data layout

## 📚 Documentation

See `PROJECT_OVERVIEW.md` for complete system documentation, implementation strategies, and design decisions.

## 🧪 Testing

```bash
# Run tests (when implemented)
cargo test
```

## 📝 License

See LICENSE file for details.

