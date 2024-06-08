use std::mem::size_of;
use std::num::NonZeroU32;
use std::ops::Range;
use std::sync::Arc;

use bevy::core_pipeline::core_3d::graph::{Core3d, Node3d};
use bevy::core_pipeline::fullscreen_vertex_shader::fullscreen_shader_vertex_state;
use bevy::ecs::entity::EntityHashMap;
use bevy::ecs::query::QueryItem;
use bevy::ecs::system::lifetimeless::SRes;
use bevy::ecs::system::SystemParamItem;
use bevy::math::Affine3;
use bevy::pbr::{MeshInputUniform, MeshPipeline, MeshPipelineViewLayoutKey, MeshPipelineViewLayouts, PreviousGlobalTransform, SetMeshViewBindGroup};
use bevy::prelude::*;
use bevy::render::{Extract, Render, RenderApp, RenderSet};
use bevy::render::batching::{GetBatchData, GetFullBatchData};
use bevy::render::batching::gpu_preprocessing::IndirectParametersBuffer;
use bevy::render::batching::no_gpu_preprocessing::{BatchedInstanceBuffer, write_batched_instance_buffer};
use bevy::render::camera::ExtractedCamera;
use bevy::render::render_graph::{NodeRunError, RenderGraphApp, RenderGraphContext, RenderLabel, ViewNode, ViewNodeRunner};
use bevy::render::render_phase::{AddRenderCommand, BinnedPhaseItem, BinnedRenderPhasePlugin, CachedRenderPipelinePhaseItem, DrawFunctionId, DrawFunctions, PhaseItem, PhaseItemExtraIndex, RenderCommand, RenderCommandResult, SetItemPipeline, TrackedRenderPass, ViewBinnedRenderPhases};
use bevy::render::render_resource::{BindGroup, BindGroupEntries, BindGroupLayout, BindGroupLayoutEntries, BlendComponent, BlendFactor, BlendOperation, BlendState, Buffer, BufferAddress, BufferDescriptor, BufferUsages, CachedRenderPipelineId, ColorTargetState, ColorWrites, Extent3d, FragmentState, MultisampleState, PipelineCache, PrimitiveState, RenderPassDescriptor, RenderPipelineDescriptor, ShaderStages, ShaderType, SpecializedRenderPipeline, SpecializedRenderPipelines, TextureDescriptor, TextureDimension, TextureFormat, TextureSampleType, TextureUsages, VertexState};
use bevy::render::render_resource::binding_types::{storage_buffer_read_only, texture_2d_multisampled};
use bevy::render::renderer::{RenderContext, RenderDevice, RenderQueue};
use bevy::render::texture::{ColorAttachment, TextureCache};
use bevy::render::view::{check_visibility, ExtractedView, ViewTarget, VisibilitySystems};
use nonmax::NonMaxU32;
use offset_allocator::Allocation;

#[derive(Clone, Debug, Reflect, Component)]
#[reflect(Component)]
pub struct PointCloud {
    pub points: Arc<Vec<Vec4>>,
}

/*
pub struct PointCloudBuffers {
    pub capacity: usize,
    pub len: usize,
    pub buffer: Buffer,
}
 */

pub struct PointCloudInstance {
    pub world_from_local: Affine3,
    pub previous_world_from_local: Affine3,
    pub allocation: Option<Allocation>,
    // pub buffers: Option<PointCloudBuffers>,
}

#[derive(Clone, ShaderType)]
pub struct PointCloudUniform {
    pub world_from_local: [Vec4; 3],
    pub previous_world_from_local: [Vec4; 3],
}

#[derive(Default, Resource, Deref, DerefMut)]
pub struct PointCloudInstances(EntityHashMap<PointCloudInstance>);

#[derive(Default, Resource, Deref, DerefMut)]
pub struct PendingPointClouds(Vec<(Entity, Arc<Vec<Vec4>>)>);

pub fn extract_point_clouds(
    mut point_cloud_instances: ResMut<PointCloudInstances>,
    mut pending_point_clouds: ResMut<PendingPointClouds>,
    clouds_query: Extract<
        Query<(
            Entity,
            &ViewVisibility,
            &GlobalTransform,
            Option<&PreviousGlobalTransform>,
            Ref<PointCloud>,
        )>,
    >,
) {
    point_cloud_instances.retain(|entity, _| clouds_query.contains(*entity));
    for (entity, view_visibility, transform, previous_transform, point_cloud) in &clouds_query {
        if !view_visibility.get() {
            point_cloud_instances.remove(&entity);
            continue;
        }
        let transform = transform.affine();
        let previous_transform = previous_transform.map(|t| t.0).unwrap_or(transform);
        let is_new = if let Some(existing) = point_cloud_instances.get_mut(&entity) {
            existing.world_from_local = (&transform).into();
            existing.previous_world_from_local = (&previous_transform).into();
            false
        } else {
            point_cloud_instances.insert(
                entity,
                PointCloudInstance {
                    world_from_local: (&transform).into(),
                    previous_world_from_local: (&previous_transform).into(),
                    allocation: None,
                },
            );
            true
        };

        if is_new || point_cloud.is_changed() {
            pending_point_clouds.push((entity, point_cloud.points.clone()));
        }
    }
}

pub fn upload_point_clouds(
    render_device: Res<RenderDevice>,
    render_queue: Res<RenderQueue>,
    mut point_clouds: ResMut<PointCloudInstances>,
    mut pending_point_clouds: ResMut<PendingPointClouds>,
) {
    for (entity, points) in pending_point_clouds.drain(..) {
        let Some(point_cloud) = point_clouds.get_mut(&entity) else {
            continue;
        };

        if let Some(allocation) = point_cloud.allocation.take() {
            // TODO: free allocation
        }

        // TODO: allocate
        if let Some(allocation) = None {
            // render_queue.write_buffer(&buffer.buffer, BufferAddress::default(), bytemuck::cast_slice(&points));
            point_cloud.allocation = allocation;
        }

        /*
        let buffer = match &mut point_cloud.buffers {
            Some(buffers) if buffers.capacity >= points.len() => {
                buffers.len = points.len();
                buffers
            }
            _ => {
                point_cloud.buffers.get_or_insert_with(|| {
                    let block_size = 1 << 20;
                    let init_capacity = points.len().div_ceil(block_size) * block_size;
                    let point_buffer = render_device.create_buffer(&BufferDescriptor {
                        label: Some("point buffer"),
                        size: (size_of::<Vec3>() * init_capacity) as BufferAddress,
                        usage: BufferUsages::COPY_DST | BufferUsages::STORAGE,
                        mapped_at_creation: false,
                    });
                    PointCloudBuffers {
                        capacity: init_capacity,
                        len: 0,
                        buffer: point_buffer,
                    }
                })
            }
        };*/
    }
}

pub fn queue_point_clouds(
    draw_functions: Res<DrawFunctions<OrderIndependentTransparent3d>>,
    point_cloud_pipeline: Res<PointCloudPipeline>,
    msaa: Res<Msaa>,
    mut pipelines: ResMut<SpecializedRenderPipelines<PointCloudPipeline>>,
    pipeline_cache: Res<PipelineCache>,
    point_cloud_instances: Res<PointCloudInstances>,
    mut transparent_phases: ResMut<ViewBinnedRenderPhases<OrderIndependentTransparent3d>>,
    mut views: Query<Entity, With<ExtractedView>>,
) {
    let draw_point_cloud = draw_functions.read().id::<DrawPointCloud>();
    let view_key = if msaa.samples() > 1 {
        MeshPipelineViewLayoutKey::MULTISAMPLED
    } else {
        MeshPipelineViewLayoutKey::empty()
    };
    let pipeline_key = PointCloudPipelineKey {
        msaa_samples: msaa.samples(),
        view_key,
    };
    for view_entity in &mut views {
        let Some(transparent_phase) = transparent_phases.get_mut(&view_entity) else {
            continue;
        };

        for entity in point_cloud_instances.keys().copied() {
            let pipeline = pipelines
                .specialize(&pipeline_cache, &point_cloud_pipeline, pipeline_key.clone());
            let key = OrderIndependentTransparent3dBinKey {
                pipeline,
                draw_function: draw_point_cloud,
            };
            transparent_phase.add(key, entity, true);
        }
    }
}

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

#[derive(Clone, Hash, PartialEq, Eq)]
pub struct PointCloudPipelineKey {
    msaa_samples: u32,
    view_key: MeshPipelineViewLayoutKey,
}

#[derive(Resource)]
pub struct PointCloudPipeline {
    shader: Handle<Shader>,
    view_layouts: MeshPipelineViewLayouts,
    point_cloud_layout: BindGroupLayout,
}

impl FromWorld for PointCloudPipeline {
    fn from_world(world: &mut World) -> Self {
        let asset_server = world.resource::<AssetServer>();
        let shader = asset_server.load("shaders/point_cloud.wgsl");
        let render_device = world.resource::<RenderDevice>();
        let mesh_pipeline = world.resource::<MeshPipeline>();
        let point_cloud_layout = render_device.create_bind_group_layout(
            "point_cloud_layout",
            &BindGroupLayoutEntries::sequential(
                ShaderStages::VERTEX_FRAGMENT,
                (
                    storage_buffer_read_only::<PointCloudUniform>(false).count(NonZeroU32::new(16).unwrap()),
                    storage_buffer_read_only::<Vec3>(false).count(NonZeroU32::new(16).unwrap()),
                    storage_buffer_read_only::<u32>(false).count(NonZeroU32::new(16).unwrap()),
                ),
            ),
        );

        PointCloudPipeline {
            shader,
            view_layouts: mesh_pipeline.view_layouts.clone(),
            point_cloud_layout,
        }
    }
}

impl SpecializedRenderPipeline for PointCloudPipeline {
    type Key = PointCloudPipelineKey;

    fn specialize(
        &self,
        key: Self::Key,
    ) -> RenderPipelineDescriptor {
        let layout = vec![
            self.view_layouts[key.view_key.bits() as usize].bind_group_layout.clone(),
            self.point_cloud_layout.clone(),
        ];

        let blend_add = BlendComponent {
            src_factor: BlendFactor::One,
            dst_factor: BlendFactor::One,
            operation: BlendOperation::Add,
        };
        let blend_dissolve = BlendComponent {
            src_factor: BlendFactor::Zero,
            dst_factor: BlendFactor::OneMinusSrcAlpha,
            operation: BlendOperation::Add,
        };
        let shader_defs = vec![];
        RenderPipelineDescriptor {
            vertex: VertexState {
                shader: self.shader.clone(),
                entry_point: "vertex".into(),
                shader_defs: shader_defs.clone(),
                buffers: vec![],
            },
            fragment: Some(FragmentState {
                shader: self.shader.clone(),
                shader_defs,
                entry_point: "fragment".into(),
                targets: vec![
                    Some(ColorTargetState {
                        format: TextureFormat::Rgba16Float,
                        blend: Some(BlendState {
                            color: blend_add,
                            alpha: blend_add,
                        }),
                        write_mask: ColorWrites::ALL,
                    }),
                    Some(ColorTargetState {
                        format: TextureFormat::R16Float,
                        blend: Some(BlendState {
                            color: blend_dissolve,
                            alpha: blend_dissolve,
                        }),
                        write_mask: ColorWrites::ALL,
                    }),
                ],
            }),
            layout,
            primitive: PrimitiveState::default(),
            depth_stencil: None,
            multisample: MultisampleState {
                count: key.msaa_samples,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            label: Some("Point Cloud Pipeline".into()),
            push_constant_ranges: vec![],
        }
    }
}

impl GetBatchData for PointCloudPipeline {
    type Param = SRes<PointCloudInstances>;
    type CompareData = ();
    type BufferData = PointCloudUniform;

    fn get_batch_data(
        point_cloud_instances: &SystemParamItem<Self::Param>,
        entity: Entity,
    ) -> Option<(Self::BufferData, Option<Self::CompareData>)> {
        let point_cloud = point_cloud_instances.get(&entity)?;
        Some((
            PointCloudUniform {
                world_from_local: point_cloud.world_from_local.to_transpose(),
                previous_world_from_local: point_cloud.previous_world_from_local.to_transpose(),
            },
            Some(())
        ))
    }
}

impl GetFullBatchData for PointCloudPipeline {
    type BufferInputData = MeshInputUniform;

    fn get_binned_batch_data(
        point_cloud_instances: &SystemParamItem<Self::Param>,
        entity: Entity,
    ) -> Option<Self::BufferData> {
        let instance = point_cloud_instances.get(&entity)?;
        Some(PointCloudUniform {
            world_from_local: instance.world_from_local.to_transpose(),
            previous_world_from_local: instance.previous_world_from_local.to_transpose(),
        })
    }

    fn get_index_and_compare_data(
        _point_cloud_instances: &SystemParamItem<Self::Param>,
        _entity: Entity,
    ) -> Option<(NonMaxU32, Option<Self::CompareData>)> {
        unreachable!();
        /*
        let point_cloud_instance = point_cloud_instances.get(&entity)?;
        Some((
            point_cloud_instance.current_uniform_index,
            Some(())
        ))
         */
    }

    fn get_binned_index(
        _point_cloud_instances: &SystemParamItem<Self::Param>,
        _entity: Entity,
    ) -> Option<NonMaxU32> {
        unreachable!();
        /*
        point_cloud_instances
            .get(&entity)
            .map(|entity| entity.current_uniform_index)
         */
    }

    fn get_batch_indirect_parameters_index(
        _point_cloud_instances: &SystemParamItem<Self::Param>,
        _indirect_parameters_buffer: &mut IndirectParametersBuffer,
        _entity: Entity,
        _instance_index: u32,
    ) -> Option<NonMaxU32> {
        unreachable!();
        /*get_batch_indirect_parameters_index(
            mesh_instances,
            meshes,
            indirect_parameters_buffer,
            entity,
            instance_index,
        )*/
    }
}

#[derive(Resource)]
pub struct PointCloudBindGroup {
    pub value: BindGroup,
}

pub fn prepare_point_cloud_bind_group(
    mut commands: Commands,
    point_cloud_pipeline: Res<PointCloudPipeline>,
    render_device: Res<RenderDevice>,
    point_cloud_uniforms: Res<BatchedInstanceBuffer<PointCloudUniform>>,
) {
    let Some(point_cloud_uniform) = point_cloud_uniforms.binding() else {
        return;
    };

    commands.insert_resource(PointCloudBindGroup {
        value: render_device.create_bind_group(
            "point_cloud_bind_group",
            &point_cloud_pipeline.point_cloud_layout,
            &BindGroupEntries::sequential((
                point_cloud_uniform,
            )),
        ),
    });
}

type DrawPointCloud = (
    SetItemPipeline,
    SetMeshViewBindGroup<0>,
    SetPointCloudBindGroup<1>,
    DrawPointCloudMesh,
);

pub struct SetPointCloudBindGroup<const I: usize>;

impl<P: PhaseItem, const I: usize> RenderCommand<P> for SetPointCloudBindGroup<I> {
    type Param = SRes<PointCloudBindGroup>;
    type ViewQuery = ();
    type ItemQuery = ();

    fn render<'w>(
        _item: &P,
        _view: (),
        _entity: Option<()>,
        bind_group: SystemParamItem<'w, '_, Self::Param>,
        pass: &mut TrackedRenderPass<'w>,
    ) -> RenderCommandResult {
        pass.set_bind_group(I, &bind_group.into_inner().value, &[]);
        RenderCommandResult::Success
    }
}

struct DrawPointCloudMesh;

impl<P: PhaseItem> RenderCommand<P> for DrawPointCloudMesh {
    type Param = SRes<PointCloudInstances>;
    type ViewQuery = ();
    type ItemQuery = ();

    #[inline]
    fn render<'w>(
        item: &P,
        _view: (),
        _entity: Option<()>,
        point_cloud_instances: SystemParamItem<'w, '_, Self::Param>,
        pass: &mut TrackedRenderPass<'w>,
    ) -> RenderCommandResult {
        let Some(point_cloud) = point_cloud_instances.get(&item.entity()) else {
            return RenderCommandResult::Success;
        };

        if let Some(buffers) = point_cloud.buffers.as_ref() {
            pass.draw(0..6, 0..buffers.len as u32);
        }

        RenderCommandResult::Success
    }
}

pub fn extract_camera_phases(
    mut transparent_phases: ResMut<ViewBinnedRenderPhases<OrderIndependentTransparent3d>>,
    cameras: Extract<Query<(Entity, &Camera), With<Camera3d>>>,
) {
    for (entity, camera) in &cameras {
        if !camera.is_active {
            continue;
        }

        transparent_phases.insert_or_clear(entity);
    }

    transparent_phases.retain(|e, _| cameras.contains(*e));
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

pub struct PointCloudPlugin;

impl Plugin for PointCloudPlugin {
    fn build(&self, app: &mut App) {
        app
            .add_plugins((
                BinnedRenderPhasePlugin::<OrderIndependentTransparent3d, PointCloudPipeline>::default(),
            ))
            .add_systems(PostUpdate, (
                check_visibility::<With<PointCloud>>.in_set(VisibilitySystems::CheckVisibility),
            ));
        app.sub_app_mut(RenderApp)
            .init_resource::<SpecializedRenderPipelines<PointCloudPipeline>>()
            .init_resource::<SpecializedRenderPipelines<OrderIndependentTransparencyPipeline>>()
            .init_resource::<DrawFunctions<OrderIndependentTransparent3d>>()
            .add_render_command::<OrderIndependentTransparent3d, DrawPointCloud>()
            .add_systems(ExtractSchedule, (
                extract_point_clouds,
                extract_camera_phases,
            ))
            .add_systems(Render, (
                queue_point_clouds.in_set(RenderSet::QueueMeshes),
                prepare_order_independent_transparency_pipeline.in_set(RenderSet::Prepare),
                upload_point_clouds.in_set(RenderSet::PrepareResources),
                prepare_transparent_accumulation_texture.in_set(RenderSet::PrepareResources),
                write_batched_instance_buffer::<PointCloudPipeline>
                    .in_set(RenderSet::PrepareResourcesFlush),
                prepare_point_cloud_bind_group.in_set(RenderSet::PrepareBindGroups),
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
            let render_device = render_app.world().resource::<RenderDevice>();
            let batch_instance_buffer = BatchedInstanceBuffer::<PointCloudUniform>::new(render_device);
            render_app
                .insert_resource(batch_instance_buffer)
                .init_resource::<PointCloudPipeline>()
                .init_resource::<PointCloudInstances>()
                .init_resource::<PendingPointClouds>()
                .init_resource::<OrderIndependentTransparencyPipeline>();
        }
    }
}
