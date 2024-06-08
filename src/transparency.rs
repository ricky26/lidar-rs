use std::ops::Range;

use bevy::core_pipeline::core_3d::graph::{Core3d, Node3d};
use bevy::core_pipeline::fullscreen_vertex_shader::fullscreen_shader_vertex_state;
use bevy::ecs::query::QueryItem;
use bevy::pbr::MeshPipelineViewLayoutKey;
use bevy::prelude::*;
use bevy::render::{Render, RenderApp, RenderSet};
use bevy::render::camera::ExtractedCamera;
use bevy::render::render_graph::{NodeRunError, RenderGraphApp, RenderGraphContext, RenderLabel, ViewNode, ViewNodeRunner};
use bevy::render::render_phase::{BinnedPhaseItem, CachedRenderPipelinePhaseItem, DrawFunctionId, DrawFunctions, PhaseItem, PhaseItemExtraIndex, ViewBinnedRenderPhases};
use bevy::render::render_resource::{BindGroupEntries, BindGroupLayout, BindGroupLayoutEntries, BlendComponent, BlendFactor, BlendOperation, BlendState, CachedRenderPipelineId, ColorTargetState, ColorWrites, Extent3d, FragmentState, MultisampleState, PipelineCache, PrimitiveState, RenderPassDescriptor, RenderPipelineDescriptor, ShaderStages, SpecializedRenderPipeline, SpecializedRenderPipelines, TextureDescriptor, TextureDimension, TextureFormat, TextureSampleType, TextureUsages};
use bevy::render::render_resource::binding_types::texture_2d_multisampled;
use bevy::render::renderer::{RenderContext, RenderDevice};
use bevy::render::texture::{ColorAttachment, TextureCache};
use bevy::render::view::{ExtractedView, ViewTarget};

#[derive(Clone, Hash, PartialEq, Eq)]
pub struct OrderIndependentTransparencyPipelineKey {
    msaa_samples: u32,
    view_key: MeshPipelineViewLayoutKey,
}

#[derive(Resource)]
pub struct OrderIndependentTransparencyPipeline {
    shader: Handle<Shader>,
    layout: BindGroupLayout,
}

impl FromWorld for OrderIndependentTransparencyPipeline {
    fn from_world(world: &mut World) -> Self {
        let asset_server = world.resource::<AssetServer>();
        let shader = asset_server.load("shaders/oit_blit.wgsl");
        let render_device = world.resource::<RenderDevice>();
        let layout = render_device.create_bind_group_layout(
            "order_independent_transparency_layout",
            &BindGroupLayoutEntries::sequential(
                ShaderStages::VERTEX_FRAGMENT,
                (
                    texture_2d_multisampled(TextureSampleType::Float { filterable: false }),
                    texture_2d_multisampled(TextureSampleType::Float { filterable: false }),
                ),
            ),
        );
        OrderIndependentTransparencyPipeline {
            shader,
            layout,
        }
    }
}

impl SpecializedRenderPipeline for OrderIndependentTransparencyPipeline {
    type Key = OrderIndependentTransparencyPipelineKey;

    fn specialize(
        &self,
        key: Self::Key,
    ) -> RenderPipelineDescriptor {
        let layout = vec![self.layout.clone()];
        let mut shader_defs = vec![];

        if key.msaa_samples > 1 {
            shader_defs.push("MULTISAMPLED".into());
        }

        let blend = BlendComponent {
            src_factor: BlendFactor::OneMinusSrcAlpha,
            dst_factor: BlendFactor::SrcAlpha,
            operation: BlendOperation::Add,
        };
        RenderPipelineDescriptor {
            vertex: fullscreen_shader_vertex_state(),
            fragment: Some(FragmentState {
                shader: self.shader.clone(),
                shader_defs,
                entry_point: "fs_main".into(),
                targets: vec![Some(ColorTargetState {
                    format: TextureFormat::Rgba8UnormSrgb,
                    blend: Some(BlendState {
                        color: blend,
                        alpha: blend,
                    }),
                    write_mask: ColorWrites::ALL,
                })],
            }),
            layout,
            primitive: PrimitiveState::default(),
            depth_stencil: None,
            multisample: MultisampleState {
                count: key.msaa_samples,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            label: Some("Order Independent Transparency Pipeline".into()),
            push_constant_ranges: vec![],
        }
    }
}

#[derive(Component)]
pub struct OrderIndependentTransparencyPipelineId(pub CachedRenderPipelineId);

pub fn prepare_order_independent_transparency_pipeline(
    mut commands: Commands,
    pipeline_cache: Res<PipelineCache>,
    mut pipelines: ResMut<SpecializedRenderPipelines<OrderIndependentTransparencyPipeline>>,
    pipeline: Res<OrderIndependentTransparencyPipeline>,
    msaa: Res<Msaa>,
    views: Query<Entity, With<ExtractedView>>,
) {
    for entity in &views {
        let view_key = if msaa.samples() > 1 {
            MeshPipelineViewLayoutKey::MULTISAMPLED
        } else {
            MeshPipelineViewLayoutKey::empty()
        };
        let pipeline_id = pipelines.specialize(
            &pipeline_cache,
            &pipeline,
            OrderIndependentTransparencyPipelineKey {
                msaa_samples: msaa.samples(),
                view_key,
            },
        );

        commands
            .entity(entity)
            .insert(OrderIndependentTransparencyPipelineId(pipeline_id));
    }
}

#[derive(Component)]
pub struct TransparentAccumulationTexture {
    pub color_attachment: ColorAttachment,
    pub alpha_attachment: ColorAttachment,
}

pub fn prepare_transparent_accumulation_texture(
    mut commands: Commands,
    mut texture_cache: ResMut<TextureCache>,
    msaa: Res<Msaa>,
    render_device: Res<RenderDevice>,
    views: Query<(Entity, &ExtractedCamera)>,
) {
    for (entity, camera) in &views {
        let Some(physical_target_size) = camera.physical_target_size else {
            continue;
        };

        let size = Extent3d {
            depth_or_array_layers: 1,
            width: physical_target_size.x,
            height: physical_target_size.y,
        };

        let colour_texture = {
            let descriptor = TextureDescriptor {
                label: Some("transparency colour texture"),
                size,
                mip_level_count: 1,
                sample_count: msaa.samples(),
                dimension: TextureDimension::D2,
                format: TextureFormat::Rgba16Float,
                usage: TextureUsages::RENDER_ATTACHMENT | TextureUsages::TEXTURE_BINDING,
                view_formats: &[TextureFormat::Rgba16Float],
            };

            texture_cache.get(&render_device, descriptor)
        };

        let alpha_texture = {
            let descriptor = TextureDescriptor {
                label: Some("transparency alpha texture"),
                size,
                mip_level_count: 1,
                sample_count: msaa.samples(),
                dimension: TextureDimension::D2,
                format: TextureFormat::R16Float,
                usage: TextureUsages::RENDER_ATTACHMENT | TextureUsages::TEXTURE_BINDING,
                view_formats: &[TextureFormat::R16Float],
            };

            texture_cache.get(&render_device, descriptor)
        };

        commands.entity(entity).insert(TransparentAccumulationTexture {
            color_attachment: ColorAttachment::new(colour_texture, None, Some(LinearRgba::NONE)),
            alpha_attachment: ColorAttachment::new(alpha_texture, None, Some(LinearRgba::WHITE)),
        });
    }
}

#[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct OrderIndependentTransparent3dBinKey {
    pub pipeline: CachedRenderPipelineId,
    pub draw_function: DrawFunctionId,
}

pub struct OrderIndependentTransparent3d {
    pub key: OrderIndependentTransparent3dBinKey,
    pub entity: Entity,
    pub batch_range: Range<u32>,
    pub extra_index: PhaseItemExtraIndex,
}

impl PhaseItem for OrderIndependentTransparent3d {
    #[inline]
    fn entity(&self) -> Entity {
        self.entity
    }

    #[inline]
    fn draw_function(&self) -> DrawFunctionId {
        self.key.draw_function
    }

    #[inline]
    fn batch_range(&self) -> &Range<u32> {
        &self.batch_range
    }

    #[inline]
    fn batch_range_mut(&mut self) -> &mut Range<u32> {
        &mut self.batch_range
    }

    #[inline]
    fn extra_index(&self) -> PhaseItemExtraIndex {
        self.extra_index
    }

    #[inline]
    fn batch_range_and_extra_index_mut(&mut self) -> (&mut Range<u32>, &mut PhaseItemExtraIndex) {
        (&mut self.batch_range, &mut self.extra_index)
    }
}

impl BinnedPhaseItem for OrderIndependentTransparent3d {
    type BinKey = OrderIndependentTransparent3dBinKey;

    fn new(
        key: Self::BinKey,
        representative_entity: Entity,
        batch_range: Range<u32>,
        extra_index: PhaseItemExtraIndex,
    ) -> Self {
        OrderIndependentTransparent3d {
            key,
            entity: representative_entity,
            batch_range,
            extra_index,
        }
    }
}

impl CachedRenderPipelinePhaseItem for OrderIndependentTransparent3d {
    #[inline]
    fn cached_pipeline(&self) -> CachedRenderPipelineId {
        self.key.pipeline
    }
}

#[derive(Debug, Hash, PartialEq, Eq, Clone, RenderLabel)]
pub struct OrderIndependentCopyPass;

#[derive(Default)]
pub struct OrderIndependentCopyNode;

impl ViewNode for OrderIndependentCopyNode {
    type ViewQuery = (
        &'static ExtractedCamera,
        &'static ViewTarget,
        &'static TransparentAccumulationTexture,
        &'static OrderIndependentTransparencyPipelineId,
    );

    fn run(
        &self,
        graph: &mut RenderGraphContext,
        render_context: &mut RenderContext,
        (camera, target, temp_texture, copy_pipeline): QueryItem<Self::ViewQuery>,
        world: &World,
    ) -> Result<(), NodeRunError> {
        let Some(transparent_phases) =
            world.get_resource::<ViewBinnedRenderPhases<OrderIndependentTransparent3d>>()
            else {
                return Ok(());
            };

        let view_entity = graph.view_entity();
        let Some(transparent_phase) = transparent_phases.get(&view_entity) else {
            return Ok(());
        };
        let view_entity = graph.view_entity();

        if !transparent_phase.is_empty() {
            let _oit_transparent_pass_3d_span = info_span!("oit_transparent_pass_3d").entered();

            {
                let mut render_pass = render_context.begin_tracked_render_pass(RenderPassDescriptor {
                    label: Some("oit_transparent_pass_3d"),
                    color_attachments: &[
                        Some(temp_texture.color_attachment.get_attachment()),
                        Some(temp_texture.alpha_attachment.get_attachment()),
                    ],
                    depth_stencil_attachment: None,
                    timestamp_writes: None,
                    occlusion_query_set: None,
                });

                if let Some(viewport) = camera.viewport.as_ref() {
                    render_pass.set_camera_viewport(viewport);
                }

                transparent_phase.render(&mut render_pass, world, view_entity);
            }

            {
                let pipeline = world.resource::<OrderIndependentTransparencyPipeline>();
                let bind_group = render_context.render_device().create_bind_group(
                    "oit_copy_bind_group",
                    &pipeline.layout,
                    &BindGroupEntries::sequential((
                        &temp_texture.color_attachment.texture.default_view,
                        &temp_texture.alpha_attachment.texture.default_view,
                    )),
                );

                let mut copy_pass = render_context.begin_tracked_render_pass(RenderPassDescriptor {
                    label: Some("oit_transparent_pass_3d"),
                    color_attachments: &[Some(target.get_color_attachment())],
                    depth_stencil_attachment: None,
                    timestamp_writes: None,
                    occlusion_query_set: None,
                });

                if let Some(viewport) = camera.viewport.as_ref() {
                    copy_pass.set_camera_viewport(viewport);
                }

                let pipeline_cache = world.resource::<PipelineCache>();
                if let Some(pipeline) = pipeline_cache.get_render_pipeline(copy_pipeline.0) {
                    copy_pass.set_render_pipeline(pipeline);
                    copy_pass.set_bind_group(0, &bind_group, &[]);
                    copy_pass.draw(0..3, 0..1);
                }
            }
        }

        Ok(())
    }
}

pub struct OrderIndependentTransparencyPlugin;

impl Plugin for OrderIndependentTransparencyPlugin {
    fn build(&self, app: &mut App) {
        app.sub_app_mut(RenderApp)
            .init_resource::<SpecializedRenderPipelines<OrderIndependentTransparencyPipeline>>()
            .init_resource::<DrawFunctions<OrderIndependentTransparent3d>>()
            .add_systems(Render, (
                prepare_order_independent_transparency_pipeline.in_set(RenderSet::Prepare),
                prepare_transparent_accumulation_texture.in_set(RenderSet::PrepareResources),
            ))
            .add_render_graph_node::<ViewNodeRunner<OrderIndependentCopyNode>>(
                Core3d,
                OrderIndependentCopyPass,
            )
            .add_render_graph_edges(
                Core3d,
                (
                    Node3d::MainTransparentPass,
                    OrderIndependentCopyPass,
                    Node3d::EndMainPass,
                ),
            );
    }

    fn finish(&self, app: &mut App) {
        if let Some(render_app) = app.get_sub_app_mut(RenderApp) {
            render_app.init_resource::<OrderIndependentTransparencyPipeline>();
        }
    }
}
