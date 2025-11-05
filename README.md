# 🧬 Evolution Simulator 2.0

A modular, open-ended simulation of biological evolution and ecosystems featuring dynamic environments, emergent behaviors, and millions of autonomous agents.

## 🚀 Current Status

**Step 1: Core Framework** ✅ **COMPLETE**

The core framework has been established with:
- ✅ Project structure with Cargo.toml and dependencies
- ✅ Bevy ECS framework integration
- ✅ World grid system with sparse chunk storage
- ✅ Cell and Chunk data structures
- ✅ Basic plugin architecture

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
│   │   └── grid.rs         # Sparse world grid with HashMap storage
│   └── utils/              # Utility functions
│       └── mod.rs          # Coordinate conversion, math utilities
├── data/
│   ├── logs/               # Simulation logs (future)
│   ├── configs/            # Configuration files (future)
│   └── outputs/            # Output data (future)
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

## 📋 Next Steps

Following the development timeline:

1. ✅ **Core Framework** - Complete
2. ⏭️ **World & Resource Simulation** - Implement terrain, resources, and climate updates
3. ⏭️ **Organisms (Basic)** - Add agents with position, energy, metabolism, simple behavior
4. ⏭️ **Genetics & Reproduction** - Add genome encoding, mutation, crossover
5. ⏭️ **Behavior System** - Implement decision rules
6. ⏭️ **Resource-Organism Interaction** - Link eating/metabolism with resource map
7. ⏭️ **Visualization & Logging** - Add real-time data collection and map visualization
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

