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
    pub angle: f32,
    pub angle_limits: Vec2,
    pub progress: f32,
    pub interval: f32,
    pub burst_duration: f32,
    pub active: bool,
    pub trigger_burst: bool,
    pub point_cloud: Entity,
}

impl Default for Scanner {
    fn default() -> Self {
        Scanner {
            angle: PI * 0.1,
            angle_limits: vec2(PI * 0.01, PI * 0.1),
            progress: 0.0,
            interval: 0.05,
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
            MouseScrollUnit::Line => 14.0,
            MouseScrollUnit::Pixel => 1.0,
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

        let angle = scanner.angle + scroll * 0.0005;
        let angle = angle.clamp(scanner.angle_limits.x, scanner.angle_limits.y);
        if angle != scanner.angle {
            scanner.angle = angle;
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
        scanner.progress += time.delta_seconds() / scanner.interval;
        if scanner.progress < 0. {
            continue;
        }

        gizmos.line(
            transform.translation(),
            transform.translation() + transform.forward().as_vec3(),
            LIME,
        );

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
            while scanner.progress > scanner.interval {
                let p = rng.gen_range(0.0..(2.0 * PI));
                let r = rng.gen_range(0.0..scanner.angle);
                let (sp, cp) = p.sin_cos();
                let (sr, cr) = r.sin_cos();
                let local_dir = vec3(sr * cp, sr * sp, cr);
                let global_dir = transform.affine()
                    .transform_vector3(local_dir)
                    .normalize();

                let start = transform.translation();
                if start.y * global_dir.y >= 0. {
                    let t = -start.y / global_dir.y;
                    let end = start + t * global_dir;
                    points.push(end.extend(0.3));
                    gizmos.line(start, end, SKY_BLUE);
                }

                scanner.progress -= scanner.interval;
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
