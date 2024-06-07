#import bevy_pbr::{
    skinning,
    morph::morph,
    view_transformations::position_world_to_clip,
    mesh_view_bindings::view,
}
#import bevy_render::maths::affine_to_square

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
//@group(1) @binding(1) var<storage> point_cloud_points: array<PointCloudPoints>;
//@group(1) @binding(2) var<storage> point_cloud_indices: array<PointCloudIndices>;

@vertex
fn vertex(vertex: Vertex) -> VertexOutput {
//    let point_index = point_cloud_indices[vertex.instance_index].indices[vertex.index / 6];
//    let point_position = point_cloud_points[vertex.instance_index].points[point_index];
    let point_position = vec3<f32>(0, 0, 0);

    var uv = vec2(f32((vertex.index & 4) != 0), f32(vertex.index & 1));
    if vertex.index >= 3 {
        uv = vec2(1, 1) - uv;
    }

    let local_position = vec3(uv - 0.5, 0.0);
    let world_position = point_position + (view.view * vec4(local_position, 0.0)).xyz;
    let world_normal = normalize(view.world_position - point_position);

    var out: VertexOutput;
    out.world_position = vec4(world_position, 0);
    out.world_normal = world_normal;
    out.clip_position = position_world_to_clip(world_position);
    return out;
}

struct FragmentInput {
    @location(0) world_position: vec4<f32>,
    @location(1) world_normal: vec3<f32>,
};

struct FragmentOutput {
    @location(0) colour: vec4<f32>,
    @location(0) alpha: f32,
}

fn calculate_fragment_output(colour: vec4<f32>) -> FragmentOutput {
    let weight = max(min(1.0, max(max(colour.r, colour.g), colour.b) * colour.a)) *
        clamp(0.03 / (1e-5 + pow(z / 200.0, 4.0)), 1e-2, 3e3);
    var out: FragmentOutput;
    out.colour = vec4(colour.rgb * colour.a, colour.a) * weight;
    out.alpha = colour.a;
    return out;
}

@fragment
fn fragment(in: FragmentInput) -> FragmentOutput {
    let colour = vec4(1.0, 0.0, 1.0, 1.0);
    // TODO: populate colour!
    return calculate_fragment_output(colour);
}
