use std::mem::size_of;
use std::sync::Arc;

use bevy::ecs::entity::EntityHashMap;
use bevy::ecs::query::QueryItem;
use bevy::ecs::system::lifetimeless::{SRes, SResMut};
use bevy::ecs::system::SystemParamItem;
use bevy::math::Affine3;
use bevy::pbr::{MeshInputUniform, MeshPipeline, MeshPipelineViewLayoutKey, MeshPipelineViewLayouts, PreviousGlobalTransform, SetMeshViewBindGroup};
use bevy::prelude::*;
use bevy::render::{Extract, Render, RenderApp, RenderSet};
use bevy::render::batching::{GetBatchData, GetFullBatchData};
use bevy::render::batching::gpu_preprocessing::IndirectParametersBuffer;
use bevy::render::batching::no_gpu_preprocessing::{BatchedInstanceBuffer, clear_batched_cpu_instance_buffers, write_batched_instance_buffer};
use bevy::render::camera::ExtractedCamera;
use bevy::render::render_phase::{AddRenderCommand, BinnedRenderPhasePlugin, DrawFunctions, PhaseItem, RenderCommand, RenderCommandResult, SetItemPipeline, TrackedRenderPass, ViewBinnedRenderPhases};
use bevy::render::render_resource::{BindGroup, BindGroupEntries, BindGroupLayout, BindGroupLayoutEntries, BlendComponent, BlendFactor, BlendOperation, BlendState, Buffer, BufferAddress, BufferDescriptor, BufferUsages, ColorTargetState, ColorWrites, Extent3d, FragmentState, GpuArrayBuffer, MultisampleState, PipelineCache, PrimitiveState, RawBufferVec, RenderPipelineDescriptor, ShaderStages, ShaderType, SpecializedRenderPipeline, SpecializedRenderPipelines, TextureDescriptor, TextureDimension, TextureFormat, TextureUsages, VertexState};
use bevy::render::render_resource::binding_types::storage_buffer_read_only;
use bevy::render::renderer::{RenderDevice, RenderQueue};
use bevy::render::texture::{ColorAttachment, TextureCache};
use bevy::render::view::{check_visibility, ExtractedView, VisibilitySystems};
use bytemuck::{Pod, Zeroable};
use nonmax::NonMaxU32;
use offset_allocator::{Allocation, Allocator};

use crate::transparency::{OrderIndependentTransparent3d, OrderIndependentTransparent3dBinKey};

#[derive(Clone, Debug, Default, Reflect, Component)]
#[reflect(Component)]
pub struct PointCloud {
    pub points: Arc<Vec<Vec4>>,
}

pub struct PointCloudInstance {
    pub world_from_local: Affine3,
    pub previous_world_from_local: Affine3,
    pub num_points: u32,
    pub allocation: Option<Allocation>,
}

#[derive(Clone, ShaderType)]
pub struct PointCloudUniform {
    pub world_from_local: [Vec4; 3],
    pub previous_world_from_local: [Vec4; 3],
}

#[derive(Resource)]
pub struct PointCloudBuffers {
    pub point_buffer: Buffer,
    pub allocator: Allocator,
}

impl PointCloudBuffers {
    pub fn new(render_device: &RenderDevice) -> PointCloudBuffers {
        Self::with_capacity(render_device, 1024 * 1024 * 16)
    }

    pub fn with_capacity(render_device: &RenderDevice, capacity: u32) -> PointCloudBuffers {
        let point_buffer = render_device.create_buffer(&BufferDescriptor {
            label: Some("point cloud buffer"),
            size: capacity as BufferAddress * size_of::<Vec4>() as BufferAddress,
            usage: BufferUsages::COPY_SRC | BufferUsages::COPY_DST | BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        let allocator = Allocator::new(capacity);
        PointCloudBuffers {
            point_buffer,
            allocator,
        }
    }

    pub fn allocate(
        &mut self,
        _render_device: &RenderDevice,
        render_queue: &RenderQueue,
        points: &[Vec4],
    ) -> Allocation {
        let allocation = self.allocator.allocate(points.len() as u32)
            .expect("failed to allocate point buffer");
        let offset = allocation.offset as BufferAddress
            * size_of::<Vec4>() as BufferAddress;
        render_queue.write_buffer(&self.point_buffer, offset, bytemuck::cast_slice(points));
        allocation
    }

    pub fn free(&mut self, allocation: Allocation) {
        self.allocator.free(allocation);
    }
}

impl FromWorld for PointCloudBuffers {
    fn from_world(world: &mut World) -> Self {
        let render_device = world.resource::<RenderDevice>();
        Self::new(render_device)
    }
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
            existing.num_points = point_cloud.points.len() as u32;
            false
        } else {
            point_cloud_instances.insert(
                entity,
                PointCloudInstance {
                    world_from_local: (&transform).into(),
                    previous_world_from_local: (&previous_transform).into(),
                    num_points: point_cloud.points.len() as u32,
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
    mut point_cloud_buffers: ResMut<PointCloudBuffers>,
) {
    for (entity, points) in pending_point_clouds.drain(..) {
        let Some(point_cloud) = point_clouds.get_mut(&entity) else {
            continue;
        };

        if let Some(allocation) = point_cloud.allocation.take() {
            point_cloud_buffers.free(allocation);
        }

        point_cloud.allocation = Some(point_cloud_buffers.allocate(&render_device, &render_queue, &points));
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
        let shader = asset_server.load("shaders/point_cloud_distance.wgsl");
        let render_device = world.resource::<RenderDevice>();
        let mesh_pipeline = world.resource::<MeshPipeline>();
        let point_cloud_layout = render_device.create_bind_group_layout(
            "point_cloud_layout",
            &BindGroupLayoutEntries::sequential(
                ShaderStages::VERTEX_FRAGMENT,
                (
                    GpuArrayBuffer::<PointCloudUniform>::binding_layout(render_device),
                    storage_buffer_read_only::<Vec4>(false),
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
            primitive: PrimitiveState {
                cull_mode: None,
                ..default()
            },
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
    type Param = (
        SRes<PointCloudInstances>,
        SResMut<PointCloudIndirect>,
    );
    type CompareData = ();
    type BufferData = PointCloudUniform;

    fn get_batch_data(
        (ref point_cloud_instances, ref mut indirect): &mut SystemParamItem<Self::Param>,
        entity: Entity,
    ) -> Option<(Self::BufferData, Option<Self::CompareData>)> {
        let instance = point_cloud_instances.get(&entity)?;
        indirect.push(instance);
        Some((
            PointCloudUniform {
                world_from_local: instance.world_from_local.to_transpose(),
                previous_world_from_local: instance.previous_world_from_local.to_transpose(),
            },
            Some(())
        ))
    }
}

impl GetFullBatchData for PointCloudPipeline {
    type BufferInputData = MeshInputUniform;

    fn get_binned_batch_data(
        (point_cloud_instances, ref mut indirect): &mut SystemParamItem<Self::Param>,
        entity: Entity,
    ) -> Option<Self::BufferData> {
        let instance = point_cloud_instances.get(&entity)?;
        indirect.push(instance);
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

pub fn write_point_cloud_indirect(
    render_device: Res<RenderDevice>,
    render_queue: Res<RenderQueue>,
    mut indirect: ResMut<PointCloudIndirect>,
) {
    indirect.write_buffer(&render_device, &render_queue);
    indirect.clear();
}

pub fn prepare_point_cloud_bind_group(
    mut commands: Commands,
    point_cloud_pipeline: Res<PointCloudPipeline>,
    render_device: Res<RenderDevice>,
    point_cloud_uniforms: Res<BatchedInstanceBuffer<PointCloudUniform>>,
    point_cloud_buffers: Res<PointCloudBuffers>,
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
                point_cloud_buffers.point_buffer.as_entire_binding(),
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
    type Param = SRes<PointCloudIndirect>;
    type ViewQuery = ();
    type ItemQuery = ();

    #[inline]
    fn render<'w>(
        item: &P,
        _view: QueryItem<'w, Self::ViewQuery>,
        _entity: Option<()>,
        indirect: SystemParamItem<'w, '_, Self::Param>,
        pass: &mut TrackedRenderPass<'w>,
    ) -> RenderCommandResult {
        let Some(indirect_buffer) = indirect.into_inner().0.buffer() else {
            return RenderCommandResult::Failure;
        };

        let range = item.batch_range();
        let indirect_offset = range.start as BufferAddress * size_of::<DrawIndirect>() as BufferAddress;
        pass.multi_draw_indirect(indirect_buffer, indirect_offset, range.len() as u32);
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

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct DrawIndirect {
    pub vertex_count: u32,
    pub instance_count: u32,
    pub first_vertex: u32,
    pub first_instance: u32,
}

#[derive(Resource, Deref, DerefMut)]
pub struct PointCloudIndirect(RawBufferVec<DrawIndirect>);

impl Default for PointCloudIndirect {
    fn default() -> Self {
        PointCloudIndirect(RawBufferVec::new(BufferUsages::INDIRECT))
    }
}

impl PointCloudIndirect {
    pub fn push(&mut self, instance: &PointCloudInstance) {
        let first_instance = self.len() as u32;
        self.0.push(DrawIndirect {
            vertex_count: instance.num_points * 6,
            instance_count: 1,
            first_vertex: 0,
            first_instance,
        });
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
            .add_render_command::<OrderIndependentTransparent3d, DrawPointCloud>()
            .add_systems(ExtractSchedule, (
                extract_point_clouds,
                extract_camera_phases,
            ))
            .add_systems(Render, (
                queue_point_clouds.in_set(RenderSet::QueueMeshes),
                upload_point_clouds.in_set(RenderSet::PrepareResources),
                prepare_transparent_accumulation_texture.in_set(RenderSet::PrepareResources),
                write_batched_instance_buffer::<PointCloudPipeline>
                    .in_set(RenderSet::PrepareResourcesFlush),
                write_point_cloud_indirect.in_set(RenderSet::PrepareResourcesFlush),
                prepare_point_cloud_bind_group.in_set(RenderSet::PrepareBindGroups),
                clear_batched_cpu_instance_buffers::<PointCloudPipeline>
                    .in_set(RenderSet::Cleanup)
                    .after(RenderSet::Render),
            ));
    }

    fn finish(&self, app: &mut App) {
        if let Some(render_app) = app.get_sub_app_mut(RenderApp) {
            let render_device = render_app.world().resource::<RenderDevice>();
            let batch_instance_buffer = BatchedInstanceBuffer::<PointCloudUniform>::new(render_device);
            render_app
                .insert_resource(batch_instance_buffer)
                .init_resource::<PointCloudPipeline>()
                .init_resource::<PointCloudInstances>()
                .init_resource::<PointCloudBuffers>()
                .init_resource::<PointCloudIndirect>()
                .init_resource::<PendingPointClouds>();
        }
    }
}
