use std::mem::size_of;
use std::ops::Range;
use std::sync::Arc;

use bevy::core_pipeline::core_3d::graph::{Core3d, Node3d};
use bevy::core_pipeline::fullscreen_vertex_shader::fullscreen_shader_vertex_state;
use bevy::ecs::entity::EntityHashMap;
use bevy::ecs::query::QueryItem;
use bevy::ecs::system::lifetimeless::SRes;
use bevy::ecs::system::SystemParamItem;
use bevy::math::Affine3;
use bevy::pbr::{CLUSTERED_FORWARD_STORAGE_BUFFER_COUNT, generate_view_layouts, MeshPipelineViewLayout, MeshPipelineViewLayoutKey, PreviousGlobalTransform, SetMeshViewBindGroup};
use bevy::prelude::*;
use bevy::render::{Extract, Render, RenderApp, RenderSet};
use bevy::render::batching::{batch_and_prepare_render_phase, GetBatchData, write_batched_instance_buffer};
use bevy::render::camera::ExtractedCamera;
use bevy::render::render_graph::{NodeRunError, RenderGraphApp, RenderGraphContext, RenderLabel, ViewNode, ViewNodeRunner};
use bevy::render::render_phase::{AddRenderCommand, CachedRenderPipelinePhaseItem, DrawFunctionId, DrawFunctions, PhaseItem, RenderCommand, RenderCommandResult, RenderPhase, SetItemPipeline, sort_phase_system, TrackedRenderPass};
use bevy::render::render_resource::{BindGroup, BindGroupEntries, BindGroupLayout, BindGroupLayoutEntries, BlendComponent, BlendFactor, BlendOperation, BlendState, Buffer, BufferAddress, BufferDescriptor, BufferUsages, CachedRenderPipelineId, ColorTargetState, ColorWrites, Extent3d, FragmentState, GpuArrayBuffer, MultisampleState, PipelineCache, PrimitiveState, RenderPassDescriptor, RenderPipelineDescriptor, Sampler, SamplerBindingType, SamplerDescriptor, ShaderStages, ShaderType, SpecializedRenderPipeline, SpecializedRenderPipelines, TextureDescriptor, TextureDimension, TextureFormat, TextureSampleType, TextureUsages, VertexState};
use bevy::render::render_resource::binding_types::{sampler, texture_2d, texture_2d_multisampled};
use bevy::render::renderer::{RenderContext, RenderDevice, RenderQueue};
use bevy::render::texture::{ColorAttachment, TextureCache};
use bevy::render::view::{ExtractedView, ViewTarget};
use bevy::utils::{FloatOrd, HashMap};
use bevy::utils::nonmax::NonMaxU32;

#[derive(Clone, Debug, Reflect, Component)]
#[reflect(Component)]
pub struct PointCloud {
    pub points: Arc<Vec<Vec3>>,
}

#[derive(Clone, Copy, Eq, PartialEq)]
enum Axis {
    XPositive,
    XNegative,
    YPositive,
    YNegative,
    ZPositive,
    ZNegative,
}

impl Axis {
    const ALL: [Axis; 6] = [
        Self::XPositive,
        Self::XNegative,
        Self::YPositive,
        Self::YNegative,
        Self::ZPositive,
        Self::ZNegative,
    ];

    #[inline(always)]
    fn from_f32(value: f32, positive: Axis, negative: Axis) -> Axis {
        if value < 0. {
            negative
        } else {
            positive
        }
    }

    pub fn direction(self) -> Vec3 {
        match self {
            Axis::XPositive => Vec3::X,
            Axis::XNegative => Vec3::NEG_X,
            Axis::YPositive => Vec3::Y,
            Axis::YNegative => Vec3::NEG_Y,
            Axis::ZPositive => Vec3::Z,
            Axis::ZNegative => Vec3::NEG_Z,
        }
    }

    pub fn index(self) -> usize {
        match self {
            Axis::XPositive => 0,
            Axis::XNegative => 1,
            Axis::YPositive => 2,
            Axis::YNegative => 3,
            Axis::ZPositive => 4,
            Axis::ZNegative => 5,
        }
    }
}

impl From<Vec3> for Axis {
    fn from(value: Vec3) -> Self {
        let abs = value.abs();
        let max = abs.max_element();
        if max == abs.z {
            Self::from_f32(value.z, Self::ZPositive, Self::ZNegative)
        } else if max == abs.y {
            Self::from_f32(value.y, Self::YPositive, Self::YNegative)
        } else {
            Self::from_f32(value.x, Self::XPositive, Self::XNegative)
        }
    }
}

pub struct PointCloudBuffers {
    pub capacity: usize,
    pub len: usize,
    pub point_buffer: Buffer,
    pub index_buffer: Buffer,
}

pub struct PointCloudInstance {
    pub world_from_local: Affine3,
    pub previous_world_from_local: Affine3,
    pub buffers: Option<PointCloudBuffers>,
}

#[derive(Clone, ShaderType)]
pub struct PointCloudUniform {
    pub world_from_local: [Vec4; 3],
    pub previous_world_from_local: [Vec4; 3],
}

#[derive(Default, Resource, Deref, DerefMut)]
pub struct PointCloudInstances(EntityHashMap<PointCloudInstance>);

#[derive(Default, Resource, Deref, DerefMut)]
pub struct PendingPointClouds(Vec<(Entity, Arc<Vec<Vec3>>)>);

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
                    buffers: None,
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
    mut scratch_distance: Local<Vec<f32>>,
    mut scratch_indices: Local<Vec<u32>>,
) {
    for (entity, points) in pending_point_clouds.drain(..) {
        let Some(point_cloud) = point_clouds.get_mut(&entity) else {
            continue;
        };

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
                    let index_buffer = render_device.create_buffer(&BufferDescriptor {
                        label: Some("index buffer"),
                        size: (size_of::<u32>() * init_capacity * 6) as BufferAddress,
                        usage: BufferUsages::COPY_DST | BufferUsages::STORAGE,
                        mapped_at_creation: false,
                    });
                    PointCloudBuffers {
                        capacity: init_capacity,
                        len: 0,
                        point_buffer,
                        index_buffer,
                    }
                })
            }
        };

        for axis in Axis::ALL {
            let start = scratch_indices.len();
            scratch_indices.extend(0..(points.len() as u32));
            let end = scratch_indices.len();
            let slice = &mut scratch_indices[start..end];
            let dir = axis.direction();
            scratch_distance.extend(points.iter()
                .copied()
                .map(|p| p.dot(dir)));
            slice.sort_by(|idx_a, idx_b| {
                let d_a = scratch_distance[*idx_a as usize];
                let d_b = scratch_distance[*idx_b as usize];
                d_a.partial_cmp(&d_b).unwrap()
            });
            scratch_distance.clear();
        }

        render_queue.write_buffer(&buffer.point_buffer, BufferAddress::default(), bytemuck::cast_slice(&points));
        render_queue.write_buffer(&buffer.index_buffer, BufferAddress::default(), bytemuck::cast_slice(&scratch_indices));
        scratch_indices.clear();
    }
}

pub fn queue_point_clouds(
    transparent_3d_draw_functions: Res<DrawFunctions<OrderIndependentTransparent3d>>,
    point_cloud_pipeline: Res<PointCloudPipeline>,
    msaa: Res<Msaa>,
    mut pipelines: ResMut<SpecializedRenderPipelines<PointCloudPipeline>>,
    pipeline_cache: Res<PipelineCache>,
    point_cloud_instances: Res<PointCloudInstances>,
    mut views: Query<(&ExtractedView, &mut RenderPhase<OrderIndependentTransparent3d>)>,
) {
    let draw_point_cloud = transparent_3d_draw_functions.read().id::<DrawPointCloud>();
    let view_key = if msaa.samples() > 1 {
        MeshPipelineViewLayoutKey::MULTISAMPLED
    } else {
        MeshPipelineViewLayoutKey::empty()
    };
    let key = PointCloudPipelineKey {
        msaa_samples: msaa.samples(),
        view_key,
    };
    for (view, mut transparent_phase) in &mut views {
        let rangefinder = view.rangefinder3d();
        for (entity, instance) in &point_cloud_instances.0 {
            let pipeline = pipelines
                .specialize(&pipeline_cache, &point_cloud_pipeline, key.clone());
            transparent_phase.add(OrderIndependentTransparent3d {
                entity: *entity,
                pipeline,
                draw_function: draw_point_cloud,
                distance: rangefinder
                    .distance_translation(&instance.world_from_local.translation),
                batch_range: 0..1,
                dynamic_offset: None,
            });
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
    sampler: Sampler,
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
                    sampler(SamplerBindingType::Filtering),
                ),
            ),
        );
        let sampler = render_device.create_sampler(&SamplerDescriptor::default());
        OrderIndependentTransparencyPipeline {
            shader,
            layout,
            sampler,
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

        RenderPipelineDescriptor {
            vertex: fullscreen_shader_vertex_state(),
            fragment: Some(FragmentState {
                shader: self.shader.clone(),
                shader_defs,
                entry_point: "fs_main".into(),
                targets: vec![Some(ColorTargetState {
                    format: TextureFormat::Rgba8UnormSrgb,
                    blend: Some(BlendState::REPLACE),
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

const ADDITIVE_BLENDING: BlendState = BlendState {
    color: BlendComponent {
        src_factor: BlendFactor::One,
        dst_factor: BlendFactor::One,
        operation: BlendOperation::Add,
    },
    alpha: BlendComponent::OVER,
};

#[derive(Clone, Hash, PartialEq, Eq)]
pub struct PointCloudPipelineKey {
    msaa_samples: u32,
    view_key: MeshPipelineViewLayoutKey,
}

#[derive(Resource)]
pub struct PointCloudPipeline {
    shader: Handle<Shader>,
    view_layouts: [MeshPipelineViewLayout; MeshPipelineViewLayoutKey::COUNT],
    point_cloud_layout: BindGroupLayout,
}

impl FromWorld for PointCloudPipeline {
    fn from_world(world: &mut World) -> Self {
        let asset_server = world.resource::<AssetServer>();
        let shader = asset_server.load("shaders/point_cloud.wgsl");
        let render_device = world.resource::<RenderDevice>();
        let clustered_forward_buffer_binding_type = render_device
            .get_supported_read_only_binding_type(CLUSTERED_FORWARD_STORAGE_BUFFER_COUNT);
        let view_layouts =
            generate_view_layouts(&render_device, clustered_forward_buffer_binding_type);
        let point_cloud_layout = render_device.create_bind_group_layout(
            "point_cloud_layout",
            &BindGroupLayoutEntries::sequential(
                ShaderStages::VERTEX_FRAGMENT,
                (
                    GpuArrayBuffer::<PointCloudUniform>::binding_layout(render_device),
                ),
            ),
        );

        PointCloudPipeline {
            shader,
            view_layouts,
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
                targets: vec![Some(ColorTargetState {
                    format: TextureFormat::Rgba16Float,
                    blend: Some(ADDITIVE_BLENDING),
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

#[derive(Resource)]
pub struct PointCloudBindGroup {
    pub value: BindGroup,
}

pub fn prepare_point_cloud_bind_group(
    mut commands: Commands,
    point_cloud_pipeline: Res<PointCloudPipeline>,
    render_device: Res<RenderDevice>,
    point_cloud_uniforms: Res<GpuArrayBuffer<PointCloudUniform>>,
) {
    if let Some(binding) = point_cloud_uniforms.binding() {
        commands.insert_resource(PointCloudBindGroup {
            value: render_device.create_bind_group(
                "point_cloud_bind_group",
                &point_cloud_pipeline.point_cloud_layout,
                &BindGroupEntries::single(binding),
            ),
        });
    }
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
    mut commands: Commands,
    cameras_3d: Extract<Query<(Entity, &Camera), With<Camera3d>>>,
) {
    for (entity, camera) in &cameras_3d {
        if camera.is_active {
            commands.get_or_spawn(entity).insert((
                RenderPhase::<OrderIndependentTransparent3d>::default(),
            ));
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
    views_3d: Query<
        (Entity, &ExtractedCamera),
        With<RenderPhase<OrderIndependentTransparent3d>>,
    >,
) {
    let mut textures = HashMap::default();
    for (entity, camera) in &views_3d {
        let Some(physical_target_size) = camera.physical_target_size else {
            continue;
        };

        let size = Extent3d {
            depth_or_array_layers: 1,
            width: physical_target_size.x,
            height: physical_target_size.y,
        };

        let colour_texture = textures
            .entry(camera.target.clone())
            .or_insert_with(|| {
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
            })
            .clone();

        let alpha_texture = textures
            .entry(camera.target.clone())
            .or_insert_with(|| {
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
            })
            .clone();

        commands.entity(entity).insert(TransparentAccumulationTexture {
            color_attachment: ColorAttachment::new(colour_texture, None, Some(Color::NONE)),
            alpha_attachment: ColorAttachment::new(alpha_texture, None, Some(Color::NONE)),
        });
    }
}

pub struct OrderIndependentTransparent3d {
    pub distance: f32,
    pub pipeline: CachedRenderPipelineId,
    pub entity: Entity,
    pub draw_function: DrawFunctionId,
    pub batch_range: Range<u32>,
    pub dynamic_offset: Option<NonMaxU32>,
}

impl PhaseItem for OrderIndependentTransparent3d {
    type SortKey = FloatOrd;

    #[inline]
    fn entity(&self) -> Entity {
        self.entity
    }

    #[inline]
    fn sort_key(&self) -> Self::SortKey {
        FloatOrd(self.distance)
    }

    #[inline]
    fn draw_function(&self) -> DrawFunctionId {
        self.draw_function
    }

    #[inline]
    fn sort(_items: &mut [Self]) {}

    #[inline]
    fn batch_range(&self) -> &Range<u32> {
        &self.batch_range
    }

    #[inline]
    fn batch_range_mut(&mut self) -> &mut Range<u32> {
        &mut self.batch_range
    }

    #[inline]
    fn dynamic_offset(&self) -> Option<NonMaxU32> {
        self.dynamic_offset
    }

    #[inline]
    fn dynamic_offset_mut(&mut self) -> &mut Option<NonMaxU32> {
        &mut self.dynamic_offset
    }
}

impl CachedRenderPipelinePhaseItem for OrderIndependentTransparent3d {
    #[inline]
    fn cached_pipeline(&self) -> CachedRenderPipelineId {
        self.pipeline
    }
}

#[derive(Debug, Hash, PartialEq, Eq, Clone, RenderLabel)]
pub struct OrderIndependentCopyPass;

#[derive(Default)]
pub struct OrderIndependentCopyNode;

impl ViewNode for OrderIndependentCopyNode {
    type ViewQuery = (
        &'static ExtractedCamera,
        &'static RenderPhase<OrderIndependentTransparent3d>,
        &'static ViewTarget,
        &'static TransparentAccumulationTexture,
        &'static OrderIndependentTransparencyPipelineId,
    );

    fn run(
        &self,
        graph: &mut RenderGraphContext,
        render_context: &mut RenderContext,
        (camera, phase, target, temp_texture, copy_pipeline): QueryItem<Self::ViewQuery>,
        world: &World,
    ) -> Result<(), NodeRunError> {
        let view_entity = graph.view_entity();

        if !phase.items.is_empty() {
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

                phase.render(&mut render_pass, world, view_entity);
            }

            {
                let pipeline = world.resource::<OrderIndependentTransparencyPipeline>();
                let bind_group = render_context.render_device().create_bind_group(
                    "oit_copy_bind_group",
                    &pipeline.layout,
                    &BindGroupEntries::sequential((
                        &temp_texture.color_attachment.texture.default_view,
                        &temp_texture.alpha_attachment.texture.default_view,
                        &pipeline.sampler,
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
                sort_phase_system::<OrderIndependentTransparent3d>.in_set(RenderSet::PhaseSort),
                prepare_order_independent_transparency_pipeline.in_set(RenderSet::Prepare),
                upload_point_clouds.in_set(RenderSet::PrepareResources),
                prepare_transparent_accumulation_texture.in_set(RenderSet::PrepareResources),
                batch_and_prepare_render_phase::<OrderIndependentTransparent3d, PointCloudPipeline>
                    .in_set(RenderSet::PrepareResources),
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
        if let Ok(render_app) = app.get_sub_app_mut(RenderApp) {
            render_app
                .insert_resource(GpuArrayBuffer::<PointCloudUniform>::new(
                    render_app.world.resource::<RenderDevice>(),
                ))
                .init_resource::<PointCloudPipeline>()
                .init_resource::<PointCloudInstances>()
                .init_resource::<PendingPointClouds>()
                .init_resource::<OrderIndependentTransparencyPipeline>();
        }
    }
}
