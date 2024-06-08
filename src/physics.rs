use bevy::prelude::*;
use bevy::render::mesh::VertexAttributeValues;
use bevy::scene::SceneInstance;
use parry3d::math::{Point, Vector};
use parry3d::query::Ray;
use parry3d::shape::{SharedShape, TriMesh};

#[derive(Resource)]
pub struct PhysicsWorld(Option<SharedShape>);

impl Default for PhysicsWorld {
    fn default() -> Self {
        PhysicsWorld(None)
    }
}

impl PhysicsWorld {
    pub fn ray_cast(&self, start: Vec3, end: Vec3) -> Option<Vec3> {
        let Some(world) = self.0.as_ref() else {
            return None;
        };
        let dir = end - start;
        let ray = Ray {
            origin: Point::from(start.to_array()),
            dir: Vector::from(dir.to_array()),
        };
        let t = world.cast_local_ray(&ray, 1.0, true)?;
        Some(start + t * dir)
    }
}

#[derive(Component)]
pub struct PhysicsScene;

#[derive(Component)]
pub struct LoadedPhysicsScene;

pub fn build_physics_world(
    mut commands: Commands,
    mut physics_world: ResMut<PhysicsWorld>,
    meshes: Res<Assets<Mesh>>,
    scenes: Query<Entity, (With<PhysicsScene>, With<SceneInstance>, Without<LoadedPhysicsScene>)>,
    colliders: Query<(&GlobalTransform, &Handle<Mesh>)>,
) {
    for entity in &scenes {
        if colliders.is_empty() {
            continue;
        }

        commands.entity(entity).insert(LoadedPhysicsScene);

        info!("Loading physics world...");
        let mut vertices = Vec::new();
        let mut indices = Vec::new();
        for (transform, mesh_handle) in &colliders {
            let Some(mesh) = meshes.get(mesh_handle) else {
                continue;
            };

            let mut mesh = mesh.clone();
            mesh.duplicate_vertices();

            let Some(VertexAttributeValues::Float32x3(positions)) = mesh.attribute(Mesh::ATTRIBUTE_POSITION) else {
                continue;
            };

            for chunk in positions.chunks_exact(3) {
                let first_vertex = vertices.len() as u32;
                let a = Point::from(transform.transform_point(chunk[0].into()).to_array());
                let b = Point::from(transform.transform_point(chunk[1].into()).to_array());
                let c = Point::from(transform.transform_point(chunk[2].into()).to_array());
                vertices.extend([a, b, c]);
                indices.push([first_vertex, first_vertex + 1, first_vertex + 2]);
            }
        }

        info!("Loaded {} vertices.", vertices.len());
        let new_world = if vertices.is_empty() {
            None
        } else {
            Some(SharedShape::new(TriMesh::new(vertices, indices)))
        };
        physics_world.0 = new_world;
    }
}

pub struct PhysicsPlugin;

impl Plugin for PhysicsPlugin {
    fn build(&self, app: &mut App) {
        app
            .init_resource::<PhysicsWorld>()
            .add_systems(Update, (
                build_physics_world,
            ));
    }
}
