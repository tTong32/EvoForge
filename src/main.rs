mod world;
mod utils;

use bevy::prelude::*;
use world::WorldPlugin;

fn main() {
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

