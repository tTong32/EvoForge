use bevy::prelude::*;

/// Camera configuration
#[derive(Resource)]
pub struct CameraConfig {
    pub zoom_speed: f32,
    pub pan_speed: f32,
    pub min_zoom: f32,
    pub max_zoom: f32,
    pub default_zoom: f32,
}

impl Default for CameraConfig {
    fn default() -> Self {
        Self {
            zoom_speed: 0.1,
            pan_speed: 500.0,
            min_zoom: 0.1,
            max_zoom: 5.0,
            default_zoom: 1.0,
        }
    }
}

/// Handle camera controls (panning and zooming)
/// Using Bevy 0.12 Input<KeyCode> API
pub fn handle_camera_controls(
    mut camera_query: Query<(&mut Transform, &mut OrthographicProjection), With<Camera2d>>,
    keyboard_input: Res<Input<KeyCode>>,
    time: Res<Time>,
    config: Res<CameraConfig>,
) {
    let Ok((mut transform, mut projection)) = camera_query.get_single_mut() else {
        // Camera might not be ready yet, skip this frame
        return;
    };

    let dt = time.delta_seconds();

    // Keyboard panning - using WASD keys
    // Note: Bevy 0.12 uses different KeyCode variant names
    let mut pan_direction = Vec2::ZERO;
    
    // WASD controls - using single letter variants (W, S, A, D) which should exist in Bevy 0.12
    if keyboard_input.pressed(KeyCode::W) {
        pan_direction.y += 1.0;
    }
    if keyboard_input.pressed(KeyCode::S) {
        pan_direction.y -= 1.0;
    }
    if keyboard_input.pressed(KeyCode::A) {
        pan_direction.x -= 1.0;
    }
    if keyboard_input.pressed(KeyCode::D) {
        pan_direction.x += 1.0;
    }

    // Apply panning
    if pan_direction.length() > 0.0 {
        let pan_amount = pan_direction.normalize() * config.pan_speed * dt / projection.scale;
        transform.translation.x += pan_amount.x;
        transform.translation.y += pan_amount.y;
    }

    // Keyboard zooming - using +/- keys
    // Try Equals instead of Equal, and check if Minus exists
    if keyboard_input.pressed(KeyCode::Equals) {
        projection.scale = (projection.scale - config.zoom_speed * dt).max(config.min_zoom);
    }
    if keyboard_input.pressed(KeyCode::Minus) {
        projection.scale = (projection.scale + config.zoom_speed * dt).min(config.max_zoom);
    }
    
    // Reset zoom with 0 key - try Key0 instead of Digit0
    if keyboard_input.just_pressed(KeyCode::Key0) {
        projection.scale = config.default_zoom;
    }

    // Reset camera position with R key
    if keyboard_input.just_pressed(KeyCode::R) {
        transform.translation = Vec3::ZERO;
        projection.scale = config.default_zoom;
    }
}

