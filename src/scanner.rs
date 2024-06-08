use std::f32::consts::PI;
use std::sync::Arc;

use bevy::color::palettes::css::{LIME, SKY_BLUE};
use bevy::input::mouse::{MouseScrollUnit, MouseWheel};
use bevy::math::{vec2, vec3};
use bevy::prelude::*;
use rand::Rng;

use crate::point_cloud::PointCloud;

#[derive(Component, Reflect)]
#[reflect(Component)]
pub struct Scanner {
    pub size_setting: f32,
    pub angle_range: Vec2,
    pub interval_range: Vec2,
    pub progress: f32,
    pub burst_duration: f32,
    pub active: bool,
    pub trigger_burst: bool,
    pub point_cloud: Entity,
}

impl Default for Scanner {
    fn default() -> Self {
        Scanner {
            size_setting: 0.75,
            angle_range: vec2(PI * 0.01, PI * 0.1),
            interval_range: vec2(0.005, 0.003),
            progress: 0.0,
            burst_duration: 3.0,
            active: false,
            trigger_burst: false,
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
        if burst != scanner.trigger_burst {
            scanner.trigger_burst = burst;
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

        if scanner.trigger_burst {
            scanner.progress -= scanner.burst_duration;
            continue;
        }

        if scanner.active {
            let interval = scanner.interval_range.x.lerp(scanner.interval_range.y, scanner.size_setting);
            let angle = scanner.angle_range.x.lerp(scanner.angle_range.y, scanner.size_setting);

            while scanner.progress > interval {
                let p = rng.gen_range(0.0..(2.0 * PI));
                let r = rng.gen_range(0.0..angle);
                let (sp, cp) = p.sin_cos();
                let (sr, cr) = r.sin_cos();
                let local_dir = vec3(sr * cp, sr * sp, -cr);
                let global_dir = transform.affine()
                    .transform_vector3(local_dir)
                    .normalize();

                let max_dist = 200.;
                let start = transform.translation();
                let (end, hit) = if start.y * global_dir.y < 0. {
                    let t = start.y / global_dir.y;
                    (start - t * global_dir, t >= -max_dist)
                } else {
                    (start + global_dir * max_dist, false)
                };

                if hit {
                    points.push(end.extend(0.1));
                }

                gizmos.line(Vec3::ZERO, Vec3::Y, SKY_BLUE);
                gizmos.line(start, end, SKY_BLUE);
                scanner.progress -= interval;
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
