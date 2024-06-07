#import bevy_core_pipeline::fullscreen_vertex_shader::FullscreenVertexOutput

//#ifdef MULTISAMPLED
@group(0) @binding(0) var transparency_colour: texture_multisampled_2d<f32>;
@group(0) @binding(1) var transparency_alpha: texture_multisampled_2d<f32>;
//#else
//@group(0) @binding(0) var transparency_colour: texture_2d<f32>;
//@group(0) @binding(1) var transparency_alpha: texture_2d<f32>;
//#endif

@fragment
fn fs_main(
    in: FullscreenVertexOutput,
    @builtin(sample_index) sample_index: i32,
) -> @location(0) vec4<f32> {
    let pixel = vec2<u32>(in.position.xy);
    let colour = textureLoad(transparency_colour, pixel, sample_index);
    let alpha = textureLoad(transparency_alpha, pixel, sample_index).r;
    return vec4(1.0, 0.0, 1.0, alpha);
}
