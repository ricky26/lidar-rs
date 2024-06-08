#import bevy_pbr::{
    view_transformations::position_world_to_clip,
}
#import "shaders/point_cloud.wgsl"::{
    Vertex, VertexOutput, FragmentOutput,
    point_cloud_vertex, calculate_fragment_output,
}

@vertex
fn vertex(in: Vertex) -> VertexOutput {
    let v = point_cloud_vertex(in.index, in.instance_index);
    var out: VertexOutput;
    out.uv = v.uv;
    out.world_position = vec4(v.world_position, 0);
    out.world_normal = v.world_normal;
    out.clip_position = position_world_to_clip(v.world_position);
    return out;
}

@fragment
fn fragment(in: VertexOutput) -> FragmentOutput {
    let colour = vec4(1.0, 1.0, 1.0, 1.0);
    return calculate_fragment_output(in.clip_position.z, colour);
}
