#import bevy_pbr::{
    skinning,
    morph::morph,
    view_transformations::position_world_to_clip,
    mesh_view_bindings::view,
}
#import bevy_render::maths::affine3_to_square

struct Vertex {
    @builtin(vertex_index) index: u32,
    @builtin(instance_index) instance_index: u32,
}

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) world_position: vec4<f32>,
    @location(1) world_normal: vec3<f32>,
}

struct PointCloud {
    world_from_local: mat3x4<f32>,
    previous_world_from_local: mat3x4<f32>,
}

struct PointCloudPoints {
    points: array<vec3<f32>>,
};

struct PointCloudIndices {
    indices: array<u32>,
};

@group(1) @binding(0) var<storage> point_clouds: array<PointCloud>;
@group(1) @binding(1) var<storage> point_cloud_points: array<vec4<f32>>;

@vertex
fn vertex(in: Vertex) -> VertexOutput {
    let point_cloud = point_clouds[in.instance_index];
    let world_from_local = affine3_to_square(point_cloud.world_from_local);
    let point_local = point_cloud_points[vert.index / 6];
    let point_world = (world_from_local * vec4(point_local.xyz, 1.0)).xyz;
    let vert_index = vert.index % 6;
    var uv = vec2(f32((vert_index & 4) != 0), f32(vert_index & 1));
    if vertex.index >= 3 {
        uv = vec2(1, 1) - uv;
    }

    let right = view.world_from_view[0].xyz;
    let up = view.world_from_view[1].xyz;

    let vert_local = vec3(uv - 0.5, 0.0);
    let vert_world = (world_from_local * vec4(vert_local, 0.0)).xyz * point_local.w;
    let world_position = point_world + right * vert_world.x + up * vert_world.y;
    let world_normal = normalize(view.world_position - point_position);

    var out: VertexOutput;
    out.world_position = vec4(world_position, 0);
    out.world_normal = world_normal;
    out.clip_position = position_world_to_clip(world_position);
    return out;
}

struct FragmentOutput {
    @location(0) colour: vec4<f32>,
    @location(1) alpha: f32,
}

fn calculate_fragment_output(z: f32, colour: vec4<f32>) -> FragmentOutput {
    let weight = max(min(1.0, max(max(colour.r, colour.g), colour.b) * colour.a), colour.a) *
        clamp(0.03 / (1e-5 + pow(z / 200.0, 4.0)), 1e-2, 3e3);
    var out: FragmentOutput;
    out.colour = vec4(colour.rgb * colour.a, colour.a) * weight;
    out.alpha = colour.a;
    return out;
}

@fragment
fn fragment(in: VertexOutput) -> FragmentOutput {
    let colour = vec4(1.0, 0.0, 1.0, 1.0);
    // TODO: populate colour!
    return calculate_fragment_output(in.clip_position.z, colour);
}
