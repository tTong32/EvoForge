mod organisms;
mod utils;
mod visualization;
mod world;

use bevy::prelude::*;
use organisms::OrganismPlugin;
use tracing_subscriber::EnvFilter;
use visualization::VisualizationPlugin;
use world::WorldPlugin;

fn main() {
    // Initialize tracing subscriber for better error visibility
    // Default to INFO level if RUST_LOG is not set
    let filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info"));

    tracing_subscriber::fmt().with_env_filter(filter).init();

    App::new()
        .add_plugins(DefaultPlugins.set(WindowPlugin {
            primary_window: Some(Window {
                title: "Evolution Simulator".into(),
                resolution: (1280.0, 720.0).into(),
                ..default()
            }),
            ..default()
        }))
        .add_plugins(WorldPlugin)
        .add_plugins(OrganismPlugin)
        .add_plugins(VisualizationPlugin)
        .add_systems(Startup, setup)
        .add_systems(Update, update_simulation)
        .run();
}

fn setup(mut commands: Commands) {
    commands.spawn(Camera2dBundle::default());

    info!("Evolution Simulator initialized");
    info!("Core framework ready");
}

fn update_simulation(_time: Res<Time>) {
    // Placeholder for simulation tick updates
    // This will be replaced with proper simulation loop
}
