#import bevy_core_pipeline::fullscreen_vertex_shader::FullscreenVertexOutput

@group(0) @binding(0) var transparency_colour: texture_multisampled_2d<f32>;
@group(0) @binding(1) var transparency_alpha: texture_multisampled_2d<f32>;

@fragment
fn fs_main(
    in: FullscreenVertexOutput,
    @builtin(sample_index) sample_index: u32,
) -> @location(0) vec4<f32> {
    let pixel = vec2<u32>(in.position.xy);
    let colour = textureLoad(transparency_colour, pixel, i32(sample_index));
    let alpha = textureLoad(transparency_alpha, pixel, i32(sample_index)).r;
    return vec4(colour.rgb / max(colour.a, 1e-5), alpha);
}
