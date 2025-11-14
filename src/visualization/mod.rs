mod camera;
mod organisms;

pub use camera::*;
pub use organisms::*;

use bevy::prelude::*;

pub struct VisualizationPlugin;

impl Plugin for VisualizationPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<CameraConfig>()
            .add_systems(Startup, setup_visualization)
            .add_systems(
                Update,
                (
                    spawn_organism_sprites,
                    update_organism_sprites,
                    update_organism_colors,
                    cleanup_dead_organism_sprites,
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
}

