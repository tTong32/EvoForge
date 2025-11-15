use bevy::prelude::*;
use glam::Vec2;
use crate::world::{DisasterEvents, Disaster, DisasterType};

/// Marker component for disaster sprite entities
#[derive(Component)]
pub struct DisasterSprite {
    pub disaster_id: u32, // Unique ID of the disaster
}

/// Spawn and update disaster visualizations
pub fn spawn_and_update_disaster_sprites(
    mut commands: Commands,
    disaster_events: Res<DisasterEvents>,
    mut sprite_query: Query<(Entity, &DisasterSprite, &mut Transform, &mut Sprite)>,
    time: Res<Time>,
) {
    let existing_disasters: std::collections::HashSet<u32> = sprite_query
        .iter()
        .map(|(_, sprite, _, _)| sprite.disaster_id)
        .collect();

    // Spawn sprites for new disasters
    for disaster in disaster_events.active_disasters.iter() {
        if existing_disasters.contains(&disaster.id) {
            continue;
        }

        let (color, _size, alpha) = get_disaster_visual(disaster);
        let sprite_size = disaster.radius * 2.0; // Show radius as diameter

        commands
            .spawn((
                SpriteBundle {
                    sprite: Sprite {
                        color: Color::rgba(color.x, color.y, color.z, alpha),
                        custom_size: Some(Vec2::new(sprite_size, sprite_size)),
                        ..default()
                    },
                    transform: Transform::from_translation(Vec3::new(
                        disaster.center.x,
                        disaster.center.y,
                        0.5, // Render below organisms but above background
                    )),
                    ..default()
                },
                DisasterSprite {
                    disaster_id: disaster.id,
                },
            ))
            .insert(Name::new(format!("Disaster-{:?}-{}", disaster.disaster_type, disaster.id)));
    }

    // Update existing disaster sprites
    for (entity, sprite, mut transform, mut sprite_component) in sprite_query.iter_mut() {
        // Find disaster by ID
        if let Some(disaster) = disaster_events.active_disasters.iter()
            .find(|d| d.id == sprite.disaster_id) {
            // Update position
            transform.translation.x = disaster.center.x;
            transform.translation.y = disaster.center.y;

            // Update visual properties based on disaster state
            let (color, _size, alpha) = get_disaster_visual(disaster);
            let sprite_size = disaster.radius * 2.0;

            // Fade out as disaster expires
            let lifetime_ratio = disaster.time_remaining / disaster.duration;
            let final_alpha = alpha * lifetime_ratio.max(0.3); // Don't fade completely

            sprite_component.color = Color::rgba(color.x, color.y, color.z, final_alpha);
            sprite_component.custom_size = Some(Vec2::new(sprite_size, sprite_size));

            // Add pulsing effect for active disasters
            let pulse = (time.elapsed_seconds() * 2.0).sin() * 0.1 + 1.0;
            let pulse_size = sprite_size * pulse;
            sprite_component.custom_size = Some(Vec2::new(pulse_size, pulse_size));
        } else {
            // Disaster expired, remove sprite
            commands.entity(entity).despawn_recursive();
        }
    }
}

/// Get visual properties for a disaster
fn get_disaster_visual(disaster: &Disaster) -> (Vec3, f32, f32) {
    match disaster.disaster_type {
        DisasterType::Volcano => {
            // Red/orange with high intensity
            let intensity = disaster.intensity;
            let color = Vec3::new(0.9, 0.3 * intensity, 0.1 * intensity);
            let alpha = 0.4 * intensity;
            (color, disaster.radius, alpha)
        },
        DisasterType::Meteor => {
            // Dark red/brown for impact crater
            let color = Vec3::new(0.4, 0.2, 0.1);
            let alpha = 0.6;
            (color, disaster.radius, alpha)
        },
        DisasterType::Flood => {
            // Blue for water
            let intensity = disaster.intensity;
            let color = Vec3::new(0.2, 0.4, 0.8 * intensity);
            let alpha = 0.3 * intensity;
            (color, disaster.radius, alpha)
        },
        DisasterType::Drought => {
            // Yellow/brown for dry conditions
            let intensity = disaster.intensity;
            let color = Vec3::new(0.8 * intensity, 0.6 * intensity, 0.2);
            let alpha = 0.25 * intensity;
            (color, disaster.radius, alpha)
        },
    }
}

/// Draw disaster effects on the world (optional: add particle effects or overlays)
pub fn draw_disaster_effects(
    _commands: Commands,
    _disaster_events: Res<DisasterEvents>,
) {
    // Placeholder for future particle effects or visual overlays
    // Could add:
    // - Ash particles for volcanoes
    // - Water ripples for floods
    // - Dust clouds for meteors
    // - Heat waves for droughts
}

/// Clean up expired disaster sprites
pub fn cleanup_expired_disaster_sprites(
    mut commands: Commands,
    sprite_query: Query<(Entity, &DisasterSprite)>,
    disaster_events: Res<DisasterEvents>,
) {
    // Create set of active disaster IDs
    let active_disaster_ids: std::collections::HashSet<u32> = disaster_events.active_disasters
        .iter()
        .map(|d| d.id)
        .collect();

    for (entity, sprite) in sprite_query.iter() {
        // Check if disaster still exists
        if !active_disaster_ids.contains(&sprite.disaster_id) {
            commands.entity(entity).despawn_recursive();
        }
    }
}

