use crate::organisms::*;
use crate::organisms::Infected;
use bevy::prelude::*;

/// Marker component for organism sprite entities
#[derive(Component)]
pub struct OrganismSprite {
    pub organism_entity: bevy::ecs::entity::Entity,
}

/// Marker component for disease indicator sprite (child of organism sprite)
#[derive(Component)]
pub struct DiseaseIndicator {
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
    infected_query: Query<&Infected, With<Alive>>,
) {
    // Get all organism entities that already have sprites
    let existing_organisms: std::collections::HashSet<_> = sprite_query
        .iter()
        .map(|sprite| sprite.organism_entity)
        .collect();

    // Spawn sprites for organisms without sprites
    for (organism_entity, position, organism_type, energy, size, species_id) in organism_query.iter()
    {
        // Phase 1: only visualize consumer organisms; plants are cell-based.
        if *organism_type != OrganismType::Consumer {
            continue;
        }

        if existing_organisms.contains(&organism_entity) {
            continue;
        }

        let color = get_organism_color(organism_type, energy, species_id);
        let sprite_size = (size.value() * 3.0).max(2.0).min(15.0); // Clamp size for visibility

        // Check if infected to apply initial visual
        let final_color = if let Ok(infected) = infected_query.get(organism_entity) {
            apply_disease_visual_effect(color, infected)
        } else {
            color
        };

        // Use SpriteBundle with a colored rectangle (simpler than mesh for now)
        let _sprite_entity = commands
            .spawn((
                SpriteBundle {
                    sprite: Sprite {
                        color: final_color,
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
            .insert(Name::new(format!("OrganismSprite-{:?}", organism_entity)))
            .id();
    }
}

/// Update disease indicators (spawn/despawn as infection status changes)
pub fn update_disease_indicators(
    mut commands: Commands,
    sprite_query: Query<(Entity, &OrganismSprite)>,
    infected_query: Query<&Infected, With<Alive>>,
    disease_indicator_query: Query<(Entity, &DiseaseIndicator)>,
    time: Res<Time>,
    mut indicator_sprite_query: Query<&mut Sprite, (With<DiseaseIndicator>, Without<OrganismSprite>)>,
) {
    // Create a map of organism entities to their sprite entities
    let organism_to_sprite: std::collections::HashMap<_, _> = sprite_query
        .iter()
        .map(|(sprite_entity, sprite)| (sprite.organism_entity, sprite_entity))
        .collect();

    // Create a set of organisms that have indicators
    let organisms_with_indicators: std::collections::HashSet<_> = disease_indicator_query
        .iter()
        .map(|(_, indicator)| indicator.organism_entity)
        .collect();

    // Spawn indicators for newly infected organisms
    for (organism_entity, sprite_entity) in &organism_to_sprite {
        let is_infected = infected_query.get(*organism_entity).is_ok();
        let has_indicator = organisms_with_indicators.contains(organism_entity);

        if is_infected && !has_indicator {
            // Spawn disease indicator as child of sprite
            commands
                .entity(*sprite_entity)
                .with_children(|parent| {
                    parent.spawn((
                        SpriteBundle {
                            sprite: Sprite {
                                color: Color::rgba(1.0, 0.5, 0.0, 0.7), // Orange-red overlay
                                custom_size: Some(Vec2::new(12.0, 12.0)),
                                ..default()
                            },
                            transform: Transform::from_translation(Vec3::new(0.0, 0.0, 0.1)),
                            ..default()
                        },
                        DiseaseIndicator {
                            organism_entity: *organism_entity,
                        },
                    ));
                });
        }
    }

    // Remove indicators for organisms that are no longer infected
    for (indicator_entity, indicator) in disease_indicator_query.iter() {
        if infected_query.get(indicator.organism_entity).is_err() {
            commands.entity(indicator_entity).despawn_recursive();
        }
    }

    // Update indicator visuals (pulsing effect)
    let pulse = (time.elapsed_seconds() * 4.0).sin() * 0.3 + 0.7;
    for mut sprite in indicator_sprite_query.iter_mut() {
        // Pulse the alpha channel for visibility
        let base_alpha = 0.7;
        sprite.color.set_a(base_alpha * pulse);
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

/// Update sprite colors and sizes based on organism properties (energy, type, species, disease)
pub fn update_organism_colors(
    _commands: Commands,
    mut sprite_query: Query<(&OrganismSprite, &mut Sprite)>,
    organism_query: Query<(&OrganismType, &Energy, &Size, &SpeciesId), With<Alive>>,
    infected_query: Query<&Infected, With<Alive>>,
    time: Res<Time>,
) {
    for (sprite, mut sprite_component) in sprite_query.iter_mut() {
        if let Ok((organism_type, energy, size, species_id)) =
            organism_query.get(sprite.organism_entity)
        {
            let mut new_color = get_organism_color(organism_type, energy, species_id);
            let mut sprite_size = (size.value() * 3.0).max(2.0).min(15.0);
            
            // Check if organism is infected
            if let Ok(infected) = infected_query.get(sprite.organism_entity) {
                // Modify color to show infection
                new_color = apply_disease_visual_effect(new_color, infected);
                
                // Add pulsing effect for infected organisms
                let pulse = (time.elapsed_seconds() * 3.0).sin() * 0.15 + 1.0;
                sprite_size *= pulse;
                
                // Spawn disease indicator sprite if it doesn't exist
                // We'll add this as a child entity with a different visual
            }
            
            sprite_component.color = new_color;
            sprite_component.custom_size = Some(Vec2::new(sprite_size, sprite_size));
        }
    }
}

/// Apply visual effect to show disease infection
fn apply_disease_visual_effect(base_color: Color, infected: &Infected) -> Color {
    // Shift color towards sickly green/yellow for infection
    // More severe infection = more color shift
    let severity = (infected.damage_accumulated / 10.0).min(1.0);
    
    // Blend towards sickly yellow-green
    let sick_color = Color::rgb(0.6, 0.7, 0.3);
    
    // Interpolate between base color and sick color
    let r = base_color.r() * (1.0 - severity * 0.4) + sick_color.r() * (severity * 0.4);
    let g = base_color.g() * (1.0 - severity * 0.4) + sick_color.g() * (severity * 0.4);
    let b = base_color.b() * (1.0 - severity * 0.4) + sick_color.b() * (severity * 0.4);
    
    Color::rgb(r, g, b)
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

