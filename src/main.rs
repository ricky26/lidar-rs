use std::fmt::Write;
use std::sync::Arc;

use bevy::input::common_conditions::input_just_pressed;
use bevy::input::mouse::MouseMotion;
use bevy::math::{vec2, vec3};
use bevy::prelude::*;
use bevy::render::render_resource::encase::private::RuntimeSizedArray;
use bevy::window::{CursorGrabMode, WindowMode};

use crate::physics::{PhysicsPlugin, PhysicsScene};
use crate::point_cloud::{PointCloud, PointCloudMaterialPlugin, PointCloudPlugin};
use crate::point_cloud::distance_material::PointCloudDistanceMaterial;
use crate::scanner::{Scanner, ScannerPlugin};
use crate::transparency::OrderIndependentTransparencyPlugin;

pub mod transparency;
pub mod point_cloud;
pub mod scanner;
pub mod physics;

fn main() {
    App::new()
        .add_plugins((
            DefaultPlugins,
            OrderIndependentTransparencyPlugin,
            PointCloudPlugin,
            PointCloudMaterialPlugin::<PointCloudDistanceMaterial>::default(),
            PhysicsPlugin,
            ScannerPlugin,
        ))
        .add_systems(Startup, startup)
        .add_systems(Update, (
            move_free_cam,
            toggle_cursor_grab.run_if(input_just_pressed(KeyCode::KeyG)),
            toggle_lights.run_if(input_just_pressed(KeyCode::KeyL)),
            clear_scan.run_if(input_just_pressed(KeyCode::KeyR)),
            toggle_boost.run_if(input_just_pressed(KeyCode::KeyB)),
            toggle_fullscreen.run_if(input_just_pressed(KeyCode::F11)),
            update_debug_text,
            remove_emissive,
        ))
        .insert_resource(ClearColor(Color::BLACK))
        .insert_resource(AmbientLight::NONE)
        .run();
}

fn startup(
    asset_server: Res<AssetServer>,
    mut commands: Commands,
    mut distance_materials: ResMut<Assets<PointCloudDistanceMaterial>>,
    mut windows: Query<&mut Window>,
) {
    for mut window in &mut windows {
        window.cursor.grab_mode = CursorGrabMode::Locked;
        window.cursor.visible = false;
    }

    let distance_material = distance_materials.add(PointCloudDistanceMaterial::default());
    let point_cloud = commands
        .spawn((
            Name::new("PointCloud"),
            SpatialBundle::INHERITED_IDENTITY,
            PointCloud::default(),
            distance_material,
            ClearPointCloud,
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
        Name::new("Scene"),
        SceneBundle {
            scene: asset_server.load("models/scene.glb#Scene0"),
            ..default()
        },
        PhysicsScene,
    ));

    commands.spawn((
        Name::new("DebugText"),
        TextBundle {
            text: Text {
                sections: vec![TextSection::new("", TextStyle::default())],
                ..default()
            },
            style: Style {
                position_type: PositionType::Absolute,
                bottom: Val::ZERO,
                left: Val::ZERO,
                ..default()
            },
            ..default()
        },
        DebugText,
    ));
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

pub fn toggle_cursor_grab(
    mut windows: Query<&mut Window>,
) {
    for mut window in &mut windows {
        let (grab_mode, visible) = if window.cursor.visible {
            (CursorGrabMode::Locked, false)
        } else {
            (CursorGrabMode::None, true)
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

fn toggle_lights(
    mut lights: Query<
        &mut Visibility,
        Or<(With<PointLight>, With<SpotLight>, With<DirectionalLight>)>,
    >,
) {
    for mut visibility in &mut lights {
        let new_visibility = match *visibility {
            Visibility::Inherited | Visibility::Visible => Visibility::Hidden,
            Visibility::Hidden => Visibility::Inherited,
        };
        *visibility = new_visibility;
    }
}

#[derive(Component, Reflect)]
#[reflect(Component)]
struct ClearPointCloud;

fn clear_scan(
    mut point_clouds: Query<&mut PointCloud, With<ClearPointCloud>>,
) {
    for mut point_cloud in &mut point_clouds {
        let points = Arc::make_mut(&mut point_cloud.points);
        points.clear();
    }
}

fn toggle_boost(
    mut scanners: Query<&mut Scanner>,
) {
    let default = Scanner::default();
    for mut scanner in &mut scanners {
        if scanner.interval_range[0] == default.interval_range[0] {
            scanner.interval_range = vec2(0.00001, 0.00001);
        } else {
            scanner.interval_range = default.interval_range;
        }
    }
}

fn toggle_fullscreen(
    mut windows: Query<&mut Window>,
) {
    for mut window in &mut windows {
        let new_mode = if window.mode == WindowMode::Windowed {
            WindowMode::BorderlessFullscreen
        } else {
            WindowMode::Windowed
        };
        window.mode = new_mode;
    }
}

#[derive(Component, Reflect)]
#[reflect(Component)]
struct DebugText;

fn update_debug_text(
    mut text_query: Query<&mut Text, With<DebugText>>,
    point_cloud_query: Query<&PointCloud, With<ClearPointCloud>>,
) {
    let Ok(mut text) = text_query.get_single_mut() else {
        return;
    };

    let section = &mut text.sections[0];
    section.value.clear();

    if let Ok(point_cloud) = point_cloud_query.get_single() {
        write!(&mut section.value, "Points: {}", point_cloud.points.len()).unwrap();
    }
}

fn remove_emissive(
    mut commands: Commands,
    materials: Res<Assets<StandardMaterial>>,
    query: Query<(Entity, &Handle<StandardMaterial>), Changed<Handle<StandardMaterial>>>,
) {
    for (entity, handle) in &query {
        let Some(material) = materials.get(handle) else {
            continue;
        };

        let max_c = material.emissive.red
            .max(material.emissive.green)
            .max(material.emissive.blue);
        if material.emissive_texture.is_some() || max_c * material.emissive.alpha > 0.0 {
            commands.entity(entity).remove::<Handle<StandardMaterial>>();
        }
    }
}
