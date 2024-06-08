use std::sync::Arc;
use bevy::math::vec3;
use bevy::prelude::*;
use bevy::render::view::NoFrustumCulling;
use crate::point_cloud::{PointCloudPlugin, PointCloud};

pub mod point_cloud;

fn main() {
    App::new()
        .add_plugins((
            DefaultPlugins,
            PointCloudPlugin,
        ))
        .add_systems(Startup, startup)
        .run();
}

fn startup(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    commands.spawn((
        Name::new("PointCloud"),
        SpatialBundle::INHERITED_IDENTITY,
        NoFrustumCulling,
    ));

    commands.spawn((
        Name::new("Camera"),
        Camera3dBundle {
            transform: Transform::from_xyz(2.0, 2.0, 2.0)
                .looking_at(vec3(0.0, 1.5, 0.0), Vec3::Y),
            ..default()
        },
    ));

    commands.spawn((
        Name::new("Light"),
        PointLightBundle {
            transform: Transform::from_translation(Vec3::ONE),
            ..default()
        },
    ));

    commands.spawn((
        Name::new("GroundPlane"),
        PbrBundle {
            mesh: meshes.add(Plane3d::default().mesh().size(20., 20.)),
            material: materials.add(Color::srgb(0.3, 0.5, 0.3)),
            ..default()
        },
    ));

    commands.spawn((
        Name::new("TestCloud"),
        SpatialBundle::INHERITED_IDENTITY,
        PointCloud {
            points: Arc::new(vec![
                vec3(0.0, 0.0, 0.0),
                vec3(0.0, 1.0, 0.0),
                vec3(1.0, 0.0, 0.0),
                vec3(0.0, 0.0, 1.0),
                vec3(0.0, 0.0, 1.0),
                vec3(0.0, 1.5, 0.0),
            ]),
        },
    ));
}
