use std::f32::consts::PI;
use std::sync::Arc;

use bevy::color::palettes::css::{LIME, SKY_BLUE};
use bevy::input::mouse::{MouseScrollUnit, MouseWheel};
use bevy::math::{vec2, vec3};
use bevy::prelude::*;
use rand::Rng;
use crate::physics::PhysicsWorld;

use crate::point_cloud::PointCloud;

#[derive(Component, Reflect)]
#[reflect(Component)]
pub struct Scanner {
    pub size_setting: f32,
    pub angle_range: Vec2,
    pub interval_range: Vec2,
    pub progress: f32,
    pub active: bool,
    pub burst_trigger: bool,
    pub burst_count: u32,
    pub burst_interval: f32,
    pub burst_lines: u32,
    pub burst_size: f32,
    pub point_cloud: Entity,
}

impl Default for Scanner {
    fn default() -> Self {
        Scanner {
            size_setting: 0.6,
            angle_range: vec2(PI * 0.02, PI * 0.1),
            interval_range: vec2(0.0011, 0.001),
            progress: 0.0,
            active: false,
            burst_trigger: false,
            burst_count: 0,
            burst_interval: 0.01,
            burst_lines: 128,
            burst_size: 0.05,
            point_cloud: Entity::PLACEHOLDER,
        }
    }
}

pub fn update_scan_input(
    mouse_input: Res<ButtonInput<MouseButton>>,
    mut scroll_events: EventReader<MouseWheel>,
    mut scanners: Query<&mut Scanner>,
) {
    let scroll = scroll_events.read()
        .fold(0.0, |acc, event| acc + event.y * match event.unit {
            MouseScrollUnit::Line => 0.1,
            MouseScrollUnit::Pixel => 0.005,
        });

    for mut scanner in &mut scanners {
        let active = mouse_input.pressed(MouseButton::Left);
        if active != scanner.active {
            scanner.active = active;
        }

        let burst = mouse_input.pressed(MouseButton::Right);
        if burst != scanner.burst_trigger {
            scanner.burst_trigger = burst;
        }

        let size_setting = scanner.size_setting + scroll;
        let size_setting = size_setting.clamp(0., 1.);
        if size_setting != scanner.size_setting {
            scanner.size_setting = size_setting;
        }
    }
}

pub fn scan(
    time: Res<Time>,
    physics_world: Res<PhysicsWorld>,
    mut gizmos: Gizmos,
    mut scanners: Query<(&mut Scanner, &GlobalTransform)>,
    mut point_clouds: Query<&mut PointCloud>,
) {
    for (mut scanner, transform) in &mut scanners {
        scanner.progress += time.delta_seconds();
        if scanner.progress < 0. {
            continue;
        }

        // HACK: later gizmos are not drawn without this.
        gizmos.line(transform.translation(), transform.translation(), LIME);

        let mut rng = rand::thread_rng();
        let Ok(mut point_cloud) = point_clouds.get_mut(scanner.point_cloud) else {
            continue;
        };
        let points = Arc::make_mut(&mut point_cloud.points);

        if scanner.burst_count == 0 && scanner.burst_trigger {
            scanner.burst_count = scanner.burst_lines << 2;
        }

        let scan = |
            gizmos: &mut Gizmos,
            physics_world: &PhysicsWorld,
            points: &mut Vec<Vec4>,
            transform: &GlobalTransform,
            local_dir: Vec3,
        | {
            let global_dir = transform.affine()
                .transform_vector3(local_dir)
                .normalize();

            let max_dist = 200.;
            let start = transform.translation();

            let (end, hit) = if let Some(end) = physics_world.ray_cast(start, start + global_dir * max_dist) {
                (end, true)
            } else {
                (start + global_dir * max_dist, false)
            };

            if hit {
                points.push(end.extend(0.025));
            }

            gizmos.line(start, end, SKY_BLUE);
        };

        while scanner.burst_count > 0 {
            if scanner.progress < scanner.burst_interval {
                break;
            }
            scanner.progress -= scanner.burst_interval;
            scanner.burst_count -= 1;

            let axis = scanner.burst_count & 3;
            let major_offset = ((scanner.burst_count >> 2) as f32) / (scanner.burst_lines as f32) * 0.5;

            for i in 0..scanner.burst_lines {
                let minor_offset = (i as f32) / (scanner.burst_lines as f32 - 1.) - 0.5;
                let (x, y) = match axis {
                    0 => (major_offset, minor_offset),
                    1 => (minor_offset, major_offset),
                    2 => (-major_offset, -minor_offset),
                    _ => (-minor_offset, -major_offset),
                };

                let local_dir = vec3(x, y, -1.).normalize();
                scan(&mut gizmos, &physics_world, points, transform, local_dir);
            }
        }

        if scanner.burst_count > 0 {
            continue;
        }

        if scanner.active {
            let interval = scanner.interval_range.x.lerp(scanner.interval_range.y, scanner.size_setting);
            let angle = scanner.angle_range.x.lerp(scanner.angle_range.y, scanner.size_setting);

            while scanner.progress > interval {
                scanner.progress -= interval;

                let p = rng.gen_range(0.0..(2.0 * PI));
                let r = rng.gen_range(0.0..1.0f32).sqrt() * angle;
                let (sp, cp) = p.sin_cos();
                let (sr, cr) = r.sin_cos();
                let local_dir = vec3(sr * cp, sr * sp, -cr);
                scan(&mut gizmos, &physics_world, points, transform, local_dir);
            }
            continue;
        }

        scanner.progress = 0.;
    }
}

pub struct ScannerPlugin;

impl Plugin for ScannerPlugin {
    fn build(&self, app: &mut App) {
        app
            .add_systems(Update, (
                (
                    update_scan_input,
                    scan,
                ).chain(),
            ));
    }
}
