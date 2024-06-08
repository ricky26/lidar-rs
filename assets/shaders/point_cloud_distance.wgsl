#import bevy_pbr::{
    view_transformations::position_world_to_clip,
    mesh_view_bindings::view,
}
#import bevy_render::color_operations::hsv_to_rgb
#import "shaders/point_cloud.wgsl"::{
    VertexOutput, FragmentOutput,
    calculate_fragment_output,
}

struct DistanceMaterial {
    distance_min: f32,
    distance_max: f32,
    hue_min: f32,
    hue_max: f32,
}

@group(2) @binding(0) var<uniform> material: DistanceMaterial;
@group(2) @binding(1) var base_color_texture: texture_2d<f32>;
@group(2) @binding(2) var base_color_sampler: sampler;

@fragment
fn fragment(in: VertexOutput) -> FragmentOutput {
    let dist = length(in.world_position.xyz - view.world_position);
    let frac = smoothstep(material.distance_min, material.distance_max, dist);
    let hue = mix(material.hue_min, material.hue_max, frac);
    let distance_color = vec4(hsv_to_rgb(vec3(hue, 1.0, 1.0)), 0.5);
    let color = distance_color * textureSample(base_color_texture, base_color_sampler, in.uv);
    return calculate_fragment_output(in.clip_position.z, color);
}
