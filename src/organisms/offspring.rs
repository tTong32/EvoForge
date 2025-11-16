use bevy::prelude::*;
use std::collections::HashMap;

use crate::organisms::components::{
    Age, ChildGrowth, Egg, Energy, IndividualLearning, Metabolism, OrganismType,
    ParentalAttachment, ParentChildRelationship, Position, Size,
};
use crate::organisms::Behavior;
use crate::organisms::Genome;
use crate::organisms::{CachedTraits, SpeciesId};

/// Update egg incubation timers and hatch eggs into baby organisms.
pub fn update_egg_incubation(
    mut commands: Commands,
    time: Res<Time>,
    mut eggs: Query<
        (
            Entity,
            &mut Egg,
            &Position,
            &Genome,
            &CachedTraits,
            &SpeciesId,
            &OrganismType,
        ),
        Without<crate::organisms::components::Alive>,
    >,
) {
    let dt = time.delta_seconds();

    for (egg_entity, mut egg, position, genome, cached, species_id, organism_type) in
        eggs.iter_mut()
    {
        egg.incubation_time_remaining -= dt;
        if egg.incubation_time_remaining > 0.0 {
            continue;
        }

        // Hatch: spawn a baby organism with 30% adult stats.
        let adult_size = cached.size;
        let adult_max_energy = cached.max_energy;

        let size_factor = 0.3;
        let child_size = adult_size * size_factor;
        let child_max_energy = adult_max_energy * 0.3;

        let initial_energy = (child_max_energy * 0.6).max(child_max_energy * 0.3);

        let metabolism_rate = cached.metabolism_rate;
        let movement_cost = cached.movement_cost;
        let reproduction_cooldown = cached.reproduction_cooldown.max(1.0) as u32;

        let child_entity = commands
            .spawn((
                Position::new(position.x(), position.y()),
                crate::organisms::components::Velocity::new(0.0, 0.0),
                Energy::with_energy(child_max_energy, initial_energy),
                Age::new(),
                Size::new(child_size),
                Metabolism::new(metabolism_rate, movement_cost),
                crate::organisms::components::ReproductionCooldown::new(reproduction_cooldown),
                genome.clone(),
                cached.clone(),
                *species_id,
                *organism_type,
                IndividualLearning::new(cached.learning_rate),
                Behavior::new(),
                crate::organisms::components::Alive,
            ))
            .id();

        // Parent-child relationship & attachment.
        commands.entity(child_entity).insert(ParentChildRelationship {
            parent: egg.parent,
            child: child_entity,
            time_together: 0.0,
        });
        commands.entity(child_entity).insert(ParentalAttachment {
            parent: egg.parent,
            care_until_age: cached.parental_care_age,
        });
        commands.entity(child_entity).insert(ChildGrowth {
            growth: size_factor,
            base_rate: cached.growth_rate,
            max_rate: cached.max_growth_rate,
            food_deficit: 0.0,
            independence_age: cached.parental_care_age,
        });

        // Remove egg.
        commands.entity(egg_entity).despawn_recursive();
    }
}

/// Make attached children follow their parent and apply speed penalty to parents.
pub fn update_attached_children_movement(
    mut parents: Query<
        (Entity, &Position, &mut crate::organisms::components::Velocity),
        (With<crate::organisms::components::Alive>, Without<ParentalAttachment>),
    >,
    mut children: Query<
        (
            &mut Position,
            &mut crate::organisms::components::Velocity,
            &ParentalAttachment,
            &ChildGrowth,
        ),
        With<crate::organisms::components::Alive>,
    >,
) {
    // First, compute speed penalty factors per parent based on attached children.
    let mut parent_penalty: HashMap<Entity, f32> = HashMap::new();

    for (_child_pos, _child_vel, attachment, growth) in children.iter_mut() {
        let p = 0.5 + growth.growth.clamp(0.0, 1.0) * 0.5; // 0.5..1.0, older kids slow less
        parent_penalty
            .entry(attachment.parent)
            .and_modify(|v| *v = v.min(p))
            .or_insert(p);
    }

    // Apply penalty and gather parent positions.
    let mut parent_positions: HashMap<Entity, Vec2> = HashMap::new();
    for (parent_entity, parent_pos, mut vel) in parents.iter_mut() {
        parent_positions.insert(parent_entity, parent_pos.as_vec2());
        if let Some(factor) = parent_penalty.get(&parent_entity) {
            vel.0 *= *factor;
        }
    }

    // Make children follow their parent with a small offset and minimal velocity.
    for (mut child_pos, mut child_vel, attachment, _growth) in children.iter_mut() {
        if let Some(parent_world) = parent_positions.get(&attachment.parent) {
            let offset = Vec2::new(3.0, 0.0);
            child_pos.0 = *parent_world + offset;
            child_vel.0 = Vec2::ZERO;
        }
    }
}

/// Update child growth and independence / starvation.
pub fn update_child_growth_and_independence(
    mut commands: Commands,
    time: Res<Time>,
    mut query: Query<
        (
            Entity,
            &mut ChildGrowth,
            &mut Size,
            &mut Energy,
            &Age,
            &CachedTraits,
            Option<&mut ParentalAttachment>,
        ),
        With<crate::organisms::components::Alive>,
    >,
) {
    let dt = time.delta_seconds();

    for (entity, mut growth, mut size, mut energy, age, cached, attachment_opt) in query.iter_mut()
    {
        let food_ratio = energy.ratio();

        // Growth rate depends on base rate and food availability.
        let mut rate = growth.base_rate * (0.5 + food_ratio * 0.5);
        rate = rate.min(growth.max_rate);

        growth.growth = (growth.growth + rate * dt).clamp(0.0, 1.0);

        // Scale size and max energy with growth.
        size.0 = cached.size * growth.growth.max(0.1);
        energy.max = cached.max_energy * growth.growth.max(0.3);
        energy.current = energy.current.min(energy.max);

        // Starvation tracking.
        if food_ratio < 0.2 {
            growth.food_deficit += dt;
        } else {
            growth.food_deficit = (growth.food_deficit - dt).max(0.0);
        }

        // If deficit exceeds threshold, child dies of starvation.
        if growth.food_deficit > 30.0 {
            energy.current = 0.0;
        }

        // Independence check based on age.
        if (age.0 as f32) >= growth.independence_age {
            if attachment_opt.is_some() {
                commands
                    .entity(entity)
                    .remove::<ParentalAttachment>();
            }
        }
    }
}



