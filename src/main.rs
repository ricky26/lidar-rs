use std::sync::Arc;

use bevy::input::mouse::MouseMotion;
use bevy::math::{vec3, vec4};
use bevy::prelude::*;
use bevy::render::view::NoFrustumCulling;
use bevy::window::CursorGrabMode;

use crate::point_cloud::{PointCloud, PointCloudPlugin};
use crate::scanner::{Scanner, ScannerPlugin};
use crate::transparency::OrderIndependentTransparencyPlugin;

pub mod transparency;
pub mod point_cloud;
pub mod scanner;

fn main() {
    App::new()
        .add_plugins((
            DefaultPlugins,
            OrderIndependentTransparencyPlugin,
            PointCloudPlugin,
            ScannerPlugin,
        ))
        .add_systems(Startup, startup)
        .add_systems(Update, (
            grab_cursor,
            move_free_cam,
        ))
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

    let point_cloud = commands
        .spawn((
            Name::new("PointCloud"),
            SpatialBundle::INHERITED_IDENTITY,
            PointCloud::default(),
        ))
        .id();

    commands
        .spawn((
            Name::new("Camera"),
            Camera3dBundle {
                transform: Transform::from_xyz(2.0, 2.0, 2.0)
                    .looking_at(vec3(0.0, 1.5, 0.0), Vec3::Y),
                ..default()
            },
            VisibilityBundle::default(),
            FreeCam::default(),
        ))
        .with_children(|children| {
            children
                .spawn((
                    Name::new("Scanner"),
                    SpatialBundle {
                        transform: Transform::from_xyz(0.2, -0.1, 0.1),
                        ..default()
                    },
                    Scanner {
                        point_cloud,
                        ..default()
                    },
                ));
        });

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

    // commands.spawn((
    //     Name::new("TestCloud"),
    //     SpatialBundle::INHERITED_IDENTITY,
    //     PointCloud {
    //         points: Arc::new(vec![
    //             vec4(0.0, 0.0, 0.0, 1.0),
    //             vec4(0.0, 1.0, 0.0, 1.0),
    //             vec4(1.0, 0.0, 0.0, 1.0),
    //             vec4(0.0, 0.0, 1.0, 1.0),
    //             vec4(0.0, 0.0, 1.0, 1.0),
    //             vec4(0.0, 1.5, 0.0, 1.0),
    //         ]),
    //     },
    // ));
}

pub enum FreeCamBinding {
    Move(Vec3),
    MoveModify(f32),
}

#[derive(Component)]
pub struct FreeCam {
    pub look: Vec2,
    pub max_look: f32,
    pub move_speed: f32,
    pub look_speed: f32,
    pub key_bindings: Vec<(KeyCode, FreeCamBinding)>,
}

impl Default for FreeCam {
    fn default() -> Self {
        FreeCam {
            look: Vec2::ZERO,
            max_look: std::f32::consts::PI * 0.4,
            move_speed: 2.0,
            look_speed: 0.1,
            key_bindings: vec![
                (KeyCode::KeyW, FreeCamBinding::Move(Vec3::NEG_Z)),
                (KeyCode::KeyS, FreeCamBinding::Move(Vec3::Z)),
                (KeyCode::KeyQ, FreeCamBinding::Move(Vec3::NEG_Y)),
                (KeyCode::KeyE, FreeCamBinding::Move(Vec3::Y)),
                (KeyCode::KeyA, FreeCamBinding::Move(Vec3::NEG_X)),
                (KeyCode::KeyD, FreeCamBinding::Move(Vec3::X)),
                (KeyCode::ShiftLeft, FreeCamBinding::MoveModify(5.)),
            ],
        }
    }
}

pub fn grab_cursor(
    mut windows: Query<&mut Window>,
) {
    for mut window in &mut windows {
        let (grab_mode, visible) = if !window.focused {
            (CursorGrabMode::None, true)
        } else {
            (CursorGrabMode::Locked, false)
        };

        window.cursor.grab_mode = grab_mode;
        window.cursor.visible = visible;
    }
}

pub fn move_free_cam(
    time: Res<Time>,
    key_input: Res<ButtonInput<KeyCode>>,
    mut mouse_motion: EventReader<MouseMotion>,
    mut cameras: Query<(&mut FreeCam, &mut Transform)>,
) {
    let look_input = mouse_motion.read()
        .fold(Vec2::ZERO, |acc, input| acc + input.delta)
        * time.delta_seconds() * -1.0;

    for (mut free_cam, mut transform) in &mut cameras {
        let (move_input, move_modifier) = free_cam.key_bindings.iter()
            .fold((Vec3::ZERO, 1.), |(input, modifier), (key_code, binding)| {
                if key_input.pressed(*key_code) {
                    match binding {
                        FreeCamBinding::Move(x) => (input + *x, modifier),
                        FreeCamBinding::MoveModify(x) => (input, modifier * *x),
                    }
                } else {
                    (input, modifier)
                }
            });
        let mut look = free_cam.look + look_input * free_cam.look_speed;
        look.y = look.y.clamp(-free_cam.max_look, free_cam.max_look);
        free_cam.look = look;
        transform.rotation = Quat::from_rotation_y(look.x)
            * Quat::from_rotation_x(look.y);

        let move_delta = transform.rotation * move_input * move_modifier * free_cam.move_speed * time.delta_seconds();
        transform.translation += move_delta;
    }
}
