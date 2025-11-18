## Evolution Simulator 2.0

### Overview

Evolution Simulator 2.0 is a modular, open-ended ecosystem simulator written in Rust. It models large populations of autonomous organisms evolving in dynamic environments, with support for genetics, behavior, speciation, co-evolution, disease, and environmental disasters. The project is designed both as an experimental platform for artificial life research and as a performant sandbox for exploring emergent ecosystem dynamics.

### Key Features

- **Agent-based evolution at scale**: Simulates large populations of organisms using Bevy ECS and data-oriented design, with each organism represented by components for metabolism, movement, perception, memory, and decision-making.
- **Genetics and reproduction**: Encodes genomes as structured traits, supports mutation and crossover, and expresses traits into observable phenotypes that affect survival, reproduction, and niche specialization.
- **Behavior system**: Uses a behavior state machine and sensory inputs to drive foraging, fleeing, hunting, and other behaviors, influenced by energy constraints and environmental conditions.
- **Speciation and ecosystem dynamics**: Tracks species via genetic distance, manages speciation events, and records ecosystem statistics such as population sizes, trait distributions, and species turnover.
- **Co-evolution and disease**: Models predator–prey, competitive, and mutualistic interactions, along with disease spread, resistance traits, and evolvable physical, chemical, and behavioral defenses.
- **Environmental simulation**: Simulates climate, terrain, and resource regeneration across a chunked world grid, with major disasters such as volcanoes, meteors, floods, and droughts that reshape the environment and selective pressures.
- **Visualization and logging**: Renders organisms and environmental effects in real time and logs simulation data to CSV for offline analysis and AI model training.

### Architecture

The simulator is built on top of the Bevy game engine and its Entity-Component-System (ECS) framework for performance and modularity.

- **World system**: The world is divided into chunks (each composed of cells), stored in a sparse `HashMap` for memory efficiency. This layout supports parallel updates and lazy loading of only active regions.
- **Cell structure**: Each cell stores environmental data (temperature, humidity, elevation, terrain type) and resource densities (such as plant biomass, minerals, sunlight, water, detritus, and prey).
- **Parallelism and performance**: Uses Rayon for parallel chunk processing and leverages Bevy’s automatic system-level parallelization. World updates and resource flows are designed to benefit from cache-friendly, data-oriented layouts.
- **Visualization layer**: A dedicated visualization module renders organisms, camera controls, and overlays for disasters and diseases while remaining decoupled from the core simulation logic.

The project is organized into modules:

- **`world`**: world grid, cells, chunks, climate, resources, terrain, and events  
- **`organisms`**: organism components, genetics, behavior, systems, speciation, tuning, ecosystem statistics, disease, and co-evolution  
- **`visualization`**: rendering of organisms and camera controls  
- **`utils`**: spatial hashing, coordinate conversion, and math utilities  

### AI Analysis and XAI

The project includes a plan for an interpretable AI (Explainable AI, XAI) module designed to analyze simulation data and answer the question: **“Why does this genome thrive in this environment?”** The AI component will:

- Train models on logged simulation data (genomes, environments, and fitness outcomes) to predict genome-environment fitness.
- Provide feature attribution and explanations that highlight which genes and environmental factors contributed most to a given outcome.
- Help explore niche specialization, trade-offs, and evolutionary strategies discovered by the simulation.

This makes the simulator not only a generator of complex evolutionary data, but also a platform for interpretable analysis of evolutionary dynamics.

### Project Structure

```text
evolution-sim/
├── Cargo.toml              # Project dependencies and build configuration
├── src/
│   ├── main.rs             # Application entry point
│   ├── world/              # World system module
│   │   ├── mod.rs          # World plugin and module exports
│   │   ├── cell.rs         # Cell data structure (environment, resources)
│   │   ├── chunk.rs        # Chunk management
│   │   ├── grid.rs         # Sparse world grid with HashMap storage
│   │   ├── climate.rs      # Climate simulation
│   │   ├── resources.rs    # Resource regeneration and flow
│   │   ├── terrain.rs      # Terrain generation
│   │   └── events.rs       # Major disaster events
│   ├── organisms/          # Organism system module
│   │   ├── mod.rs          # Organism plugin
│   │   ├── components.rs   # Organism components
│   │   ├── genetics.rs     # Genome and trait expression
│   │   ├── behavior.rs     # Behavior system and decision-making
│   │   ├── systems.rs      # Organism update systems
│   │   ├── speciation.rs   # Species tracking and differentiation
│   │   ├── tuning.rs       # Ecosystem tuning parameters
│   │   ├── ecosystem_stats.rs # Ecosystem statistics
│   │   ├── disease.rs      # Disease system
│   │   └── coevolution.rs  # Co-evolution system
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
    └── PROJECT_OVERVIEW.md # Detailed project documentation (if present)
```

### Controls and Visualization

When running with visualization enabled, the simulator provides basic camera controls:

- **Arrow keys / WASD**: pan the camera  
- **Plus / minus**: zoom in and out  
- **0**: reset zoom  
- **R**: reset camera position  

Organisms are displayed as colored sprites that encode their ecological roles and health state. Additional overlays may indicate disease, disasters, and other environmental effects.

### Building and Running

This is a standard Rust project managed with Cargo.

```bash
# Check compilation
cargo check

# Build in release mode
cargo build --release

# Run the simulator
cargo run
```

### Status and Roadmap

The core framework, world and resource simulation, organism systems, genetics, behavior, speciation, co-evolution, disease system, disasters, visualization, logging, and performance scaling are implemented. Ongoing work focuses on expanding analysis tools, improving ecosystem tuning, and developing the AI/XAI module for genome-environment fitness prediction and explanation.

### License

See the `LICENSE` file for licensing information.

