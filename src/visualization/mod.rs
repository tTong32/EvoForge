mod camera;
mod organisms;
mod disasters;

pub use camera::*;
pub use organisms::*;
pub use disasters::*;

use bevy::prelude::*;

pub struct VisualizationPlugin;

impl Plugin for VisualizationPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<CameraConfig>()
            .add_systems(Startup, setup_visualization)
            .add_systems(
                Update,
                (
                    // Organism visualization
                    spawn_organism_sprites,
                    update_organism_sprites,
                    update_organism_colors,
                    update_disease_indicators, // Step 9: Disease visualization
                    cleanup_dead_organism_sprites,
                    // Disaster visualization
                    spawn_and_update_disaster_sprites, // Step 9: Disaster visualization
                    cleanup_expired_disaster_sprites, // Step 9: Cleanup expired disasters
                    // Camera controls
                    handle_camera_controls,
                ),
            );
    }
}

fn setup_visualization(mut commands: Commands) {
    // Spawn a background to show the world bounds
    commands.spawn(SpriteBundle {
        sprite: Sprite {
            color: Color::rgb(0.05, 0.05, 0.1), // Dark blue background
            custom_size: Some(Vec2::new(2000.0, 2000.0)),
            ..default()
        },
        transform: Transform::from_translation(Vec3::new(0.0, 0.0, 0.0)),
        ..default()
    });

    info!("Visualization system initialized");
    info!("Camera controls: Arrow Keys/WASD = Pan, +/- = Zoom, 0 = Reset Zoom, R = Reset Camera");
    info!("Organism colors: Green = Producer, Red = Consumer, Purple = Decomposer");
    info!("Disease visualization: Infected organisms show sickly colors and pulsing effects");
    info!("Disaster visualization: Disasters appear as colored circles with pulsing effects");
}

