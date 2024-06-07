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
        .run()
}

fn startup(
    mut commands: Commands,
) {
    commands.spawn((
        Name::new("PointCloud"),
        SpatialBundle::INHERITED_IDENTITY,
        NoFrustumCulling,
    ));

    commands.spawn((
        Name::new("Camera"),
        Camera3dBundle {
            transform: Transform::from_xyz(0.0, 0.0, 15.0)
                .looking_at(Vec3::ZERO, Vec3::Y),
            ..default()
        },
    ));

    commands.spawn((
        Name::new("TestCloud"),
        TransformBundle::default(),
        VisibilityBundle::default(),
        PointCloud {
            points: Arc::new(vec! [
                vec3(0.0, 0.0, 0.0),
                vec3(0.0, 1.0, 0.0),
                vec3(1.0, 0.0, 0.0),
                vec3(0.0, 0.0, 1.0),
            ]),
        },
    ));
}
