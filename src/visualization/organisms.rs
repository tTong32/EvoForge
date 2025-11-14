use crate::organisms::*;
use bevy::prelude::*;

/// Marker component for organism sprite entities
#[derive(Component)]
pub struct OrganismSprite {
    pub organism_entity: bevy::ecs::entity::Entity,
}

/// Spawn sprites for organisms that don't have sprites yet
pub fn spawn_organism_sprites(
    mut commands: Commands,
    organism_query: Query<
        (Entity, &Position, &OrganismType, &Energy, &Size, &SpeciesId),
        With<Alive>,
    >,
    sprite_query: Query<&OrganismSprite>,
) {
    // Get all organism entities that already have sprites
    let existing_organisms: std::collections::HashSet<_> = sprite_query
        .iter()
        .map(|sprite| sprite.organism_entity)
        .collect();

    // Spawn sprites for organisms without sprites
    for (organism_entity, position, organism_type, energy, size, species_id) in organism_query.iter()
    {
        if existing_organisms.contains(&organism_entity) {
            continue;
        }

        let color = get_organism_color(organism_type, energy, species_id);
        let sprite_size = (size.value() * 3.0).max(2.0).min(15.0); // Clamp size for visibility

        // Use SpriteBundle with a colored rectangle (simpler than mesh for now)
        commands
            .spawn((
                SpriteBundle {
                    sprite: Sprite {
                        color,
                        custom_size: Some(Vec2::new(sprite_size, sprite_size)),
                        ..default()
                    },
                    transform: Transform::from_translation(Vec3::new(
                        position.x(),
                        position.y(),
                        1.0, // Render above background
                    )),
                    ..default()
                },
                OrganismSprite {
                    organism_entity,
                },
            ))
            .insert(Name::new(format!("OrganismSprite-{:?}", organism_entity)));
    }
}

/// Update sprite positions to match organism positions
/// This runs every frame to ensure sprites stay in sync with organisms
pub fn update_organism_sprites(
    mut sprite_query: Query<(&OrganismSprite, &mut Transform)>,
    organism_query: Query<&Position, With<Alive>>,
) {
    for (sprite, mut transform) in sprite_query.iter_mut() {
        if let Ok(position) = organism_query.get(sprite.organism_entity) {
            // Update position (Z remains the same for render order)
            transform.translation.x = position.x();
            transform.translation.y = position.y();
        }
    }
}

/// Update sprite colors and sizes based on organism properties (energy, type, species)
pub fn update_organism_colors(
    mut sprite_query: Query<(&OrganismSprite, &mut Sprite)>,
    organism_query: Query<(&OrganismType, &Energy, &Size, &SpeciesId), With<Alive>>,
) {
    for (sprite, mut sprite_component) in sprite_query.iter_mut() {
        if let Ok((organism_type, energy, size, species_id)) =
            organism_query.get(sprite.organism_entity)
        {
            let new_color = get_organism_color(organism_type, energy, species_id);
            let sprite_size = (size.value() * 3.0).max(2.0).min(15.0);
            
            sprite_component.color = new_color;
            sprite_component.custom_size = Some(Vec2::new(sprite_size, sprite_size));
        }
    }
}

/// Get color for an organism based on its properties
fn get_organism_color(
    organism_type: &OrganismType,
    energy: &Energy,
    species_id: &SpeciesId,
) -> Color {
    // Base color based on organism type
    let (r_base, g_base, b_base) = match organism_type {
        OrganismType::Producer => (0.2, 0.8, 0.2),   // Green
        OrganismType::Consumer => (0.8, 0.2, 0.2),   // Red
        OrganismType::Decomposer => (0.6, 0.4, 0.8), // Purple
    };

    // Modulate by energy level (darker = lower energy)
    let energy_factor = energy.ratio().max(0.4); // Minimum brightness
    let brightness = 0.5 + (energy_factor * 0.5); // Range from 0.5 to 1.0

    // Add slight color variation based on species ID for visual distinction
    let species_hue_shift = ((species_id.value() as f32 * 137.508) % 360.0).to_radians();
    let species_factor = 0.15; // How much species affects color
    
    // Apply brightness and hue variation
    let r: f32 = (r_base * brightness + (species_hue_shift.sin() * species_factor * 0.2)).clamp(0.0, 1.0);
    let g: f32 = (g_base * brightness + (species_hue_shift.cos() * species_factor * 0.2)).clamp(0.0, 1.0);
    let b: f32 = (b_base * brightness + ((species_hue_shift * 1.5).sin() * species_factor * 0.2)).clamp(0.0, 1.0);

    Color::rgb(r, g, b)
}

/// Clean up sprites for dead organisms
pub fn cleanup_dead_organism_sprites(
    mut commands: Commands,
    sprite_query: Query<(Entity, &OrganismSprite)>,
    organism_query: Query<&Alive>,
) {
    for (sprite_entity, sprite) in sprite_query.iter() {
        if organism_query.get(sprite.organism_entity).is_err() {
            // Organism is dead, remove sprite
            commands.entity(sprite_entity).despawn_recursive();
        }
    }
}

