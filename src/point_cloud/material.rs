use std::hash::{Hash, Hasher};
use std::marker::PhantomData;
use bevy::ecs::system::lifetimeless::SRes;
use bevy::ecs::system::SystemParamItem;
use bevy::pbr::{MeshPipelineViewLayoutKey, SetMeshViewBindGroup};
use bevy::prelude::*;
use bevy::render::extract_instances::{ExtractedInstances, ExtractInstancesPlugin};
use bevy::render::render_asset::{prepare_assets, PrepareAssetError, RenderAsset, RenderAssetPlugin, RenderAssets};
use bevy::render::render_resource::{AsBindGroup, AsBindGroupError, BindGroup, BindGroupLayout, OwnedBindingResource, PipelineCache, RenderPipelineDescriptor, ShaderRef, SpecializedRenderPipeline, SpecializedRenderPipelines};
use bevy::render::{Render, RenderApp, RenderSet};
use bevy::render::render_phase::{AddRenderCommand, DrawFunctions, PhaseItem, RenderCommand, RenderCommandResult, SetItemPipeline, TrackedRenderPass, ViewBinnedRenderPhases};
use bevy::render::renderer::RenderDevice;
use bevy::render::texture::{FallbackImage, GpuImage};
use bevy::render::view::ExtractedView;
use crate::point_cloud::{DrawPointCloudMesh, PointCloudInstances, PointCloudPipeline, PointCloudPipelineKey, SetPointCloudBindGroup};
use crate::transparency::{OrderIndependentTransparent3d, OrderIndependentTransparent3dBinKey};

pub trait PointCloudMaterial: Asset + AsBindGroup + Clone + Sized {
    fn vertex_shader() -> ShaderRef {
        ShaderRef::Default
    }

    #[allow(unused_variables)]
    fn fragment_shader() -> ShaderRef {
        ShaderRef::Default
    }

    #[inline]
    fn specialize(
        _pipeline: &PointCloudMaterialPipeline<Self>,
        _descriptor: &mut RenderPipelineDescriptor,
        _key: PointCloudMaterialPipelineKey<Self>,
    ) {}
}

pub struct PointCloudMaterialPlugin<M: PointCloudMaterial> {
    pub _marker: PhantomData<fn() -> M>,
}

impl<M: PointCloudMaterial> Default for PointCloudMaterialPlugin<M> {
    fn default() -> Self {
        Self {
            _marker: Default::default(),
        }
    }
}

impl<M: PointCloudMaterial> Plugin for PointCloudMaterialPlugin<M>
    where
        M::Data: PartialEq + Eq + Hash + Clone,
{
    fn build(&self, app: &mut App) {
        app.init_asset::<M>().add_plugins((
            ExtractInstancesPlugin::<AssetId<M>>::extract_visible(),
            RenderAssetPlugin::<PreparedPointCloudMaterial<M>>::default(),
        ));

        if let Some(render_app) = app.get_sub_app_mut(RenderApp) {
            render_app
                .add_render_command::<OrderIndependentTransparent3d, DrawPointCloudMaterial<M>>()
                .init_resource::<SpecializedRenderPipelines<PointCloudMaterialPipeline<M>>>()
                .add_systems(Render, (
                    queue_material_point_clouds::<M>
                        .in_set(RenderSet::QueueMeshes)
                        .after(prepare_assets::<PreparedPointCloudMaterial<M>>),
                ));
        }
    }

    fn finish(&self, app: &mut App) {
        if let Some(render_app) = app.get_sub_app_mut(RenderApp) {
            render_app.init_resource::<PointCloudMaterialPipeline<M>>();
        }
    }
}

pub struct PointCloudMaterialPipelineKey<M: PointCloudMaterial> {
    pub point_key: PointCloudPipelineKey,
    pub bind_group_data: M::Data,
}

impl<M: PointCloudMaterial> Clone for PointCloudMaterialPipelineKey<M>
    where
        <M as AsBindGroup>::Data: Clone
{
    fn clone(&self) -> Self {
        PointCloudMaterialPipelineKey {
            point_key: self.point_key.clone(),
            bind_group_data: self.bind_group_data.clone(),
        }
    }
}

impl<M: PointCloudMaterial> PartialEq for PointCloudMaterialPipelineKey<M>
    where
        <M as AsBindGroup>::Data: PartialEq
{
    fn eq(&self, other: &Self) -> bool {
        self.point_key == other.point_key && self.bind_group_data == other.bind_group_data
    }
}

impl<M: PointCloudMaterial> Eq for PointCloudMaterialPipelineKey<M>
    where
        <M as AsBindGroup>::Data: Eq
{}

impl<M: PointCloudMaterial> Hash for PointCloudMaterialPipelineKey<M>
    where
        <M as AsBindGroup>::Data: Hash
{
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.point_key.hash(state);
        self.bind_group_data.hash(state);
    }
}

#[derive(Resource)]
pub struct PointCloudMaterialPipeline<M: PointCloudMaterial> {
    pub point_pipeline: PointCloudPipeline,
    pub material_layout: BindGroupLayout,
    pub vertex_shader: Option<Handle<Shader>>,
    pub fragment_shader: Option<Handle<Shader>>,
    pub marker: PhantomData<M>,
}

impl<M: PointCloudMaterial> Clone for PointCloudMaterialPipeline<M> {
    fn clone(&self) -> Self {
        Self {
            point_pipeline: self.point_pipeline.clone(),
            material_layout: self.material_layout.clone(),
            vertex_shader: self.vertex_shader.clone(),
            fragment_shader: self.fragment_shader.clone(),
            marker: PhantomData,
        }
    }
}

impl<M: PointCloudMaterial> SpecializedRenderPipeline for PointCloudMaterialPipeline<M>
    where
        M::Data: PartialEq + Eq + Hash + Clone,
{
    type Key = PointCloudMaterialPipelineKey<M>;

    fn specialize(
        &self,
        key: Self::Key,
    ) -> RenderPipelineDescriptor {
        let mut descriptor = self.point_pipeline.specialize(key.point_key);
        descriptor.label = Some("Point Cloud Material Pipeline".into());
        if let Some(vertex_shader) = &self.vertex_shader {
            descriptor.vertex.shader = vertex_shader.clone();
        }

        if let Some(fragment_shader) = &self.fragment_shader {
            descriptor.fragment.as_mut().unwrap().shader = fragment_shader.clone();
        }

        descriptor.layout.insert(2, self.material_layout.clone());

        M::specialize(self, &mut descriptor, key);
        descriptor
    }
}

impl<M: PointCloudMaterial> FromWorld for PointCloudMaterialPipeline<M> {
    fn from_world(world: &mut World) -> Self {
        let asset_server = world.resource::<AssetServer>();
        let render_device = world.resource::<RenderDevice>();

        PointCloudMaterialPipeline {
            point_pipeline: world.resource::<PointCloudPipeline>().clone(),
            material_layout: M::bind_group_layout(render_device),
            vertex_shader: match M::vertex_shader() {
                ShaderRef::Default => None,
                ShaderRef::Handle(handle) => Some(handle),
                ShaderRef::Path(path) => Some(asset_server.load(path)),
            },
            fragment_shader: match M::fragment_shader() {
                ShaderRef::Default => None,
                ShaderRef::Handle(handle) => Some(handle),
                ShaderRef::Path(path) => Some(asset_server.load(path)),
            },
            marker: PhantomData,
        }
    }
}

pub type RenderPointCloudMaterialInstances<M> = ExtractedInstances<AssetId<M>>;

type DrawPointCloudMaterial<M> = (
    SetItemPipeline,
    SetMeshViewBindGroup<0>,
    SetPointCloudBindGroup<1>,
    SetPointCloudMaterialBindGroup<M, 2>,
    DrawPointCloudMesh,
);

pub struct SetPointCloudMaterialBindGroup<M: PointCloudMaterial, const I: usize>(PhantomData<M>);

impl<P: PhaseItem, M: PointCloudMaterial, const I: usize> RenderCommand<P> for SetPointCloudMaterialBindGroup<M, I> {
    type Param = (
        SRes<RenderAssets<PreparedPointCloudMaterial<M>>>,
        SRes<RenderPointCloudMaterialInstances<M>>,
    );
    type ViewQuery = ();
    type ItemQuery = ();

    #[inline]
    fn render<'w>(
        item: &P,
        _view: (),
        _item_query: Option<()>,
        (materials, material_instances): SystemParamItem<'w, '_, Self::Param>,
        pass: &mut TrackedRenderPass<'w>,
    ) -> RenderCommandResult {
        let materials = materials.into_inner();
        let material_instances = material_instances.into_inner();

        let Some(material_asset_id) = material_instances.get(&item.entity()) else {
            return RenderCommandResult::Failure;
        };
        let Some(material) = materials.get(*material_asset_id) else {
            return RenderCommandResult::Failure;
        };
        pass.set_bind_group(I, &material.bind_group, &[]);
        RenderCommandResult::Success
    }
}

pub struct PreparedPointCloudMaterial<T: PointCloudMaterial> {
    pub bindings: Vec<(u32, OwnedBindingResource)>,
    pub bind_group: BindGroup,
    pub key: T::Data,
}

impl<M: PointCloudMaterial> RenderAsset for PreparedPointCloudMaterial<M> {
    type SourceAsset = M;

    type Param = (
        SRes<RenderDevice>,
        SRes<RenderAssets<GpuImage>>,
        SRes<FallbackImage>,
        SRes<PointCloudMaterialPipeline<M>>,
    );

    fn prepare_asset(
        material: Self::SourceAsset,
        (render_device, images, fallback_image, pipeline): &mut SystemParamItem<Self::Param>,
    ) -> Result<Self, PrepareAssetError<Self::SourceAsset>> {
        match material.as_bind_group(
            &pipeline.material_layout,
            render_device,
            images,
            fallback_image,
        ) {
            Ok(prepared) => {
                Ok(PreparedPointCloudMaterial {
                    bindings: prepared.bindings,
                    bind_group: prepared.bind_group,
                    key: prepared.data,
                })
            }
            Err(AsBindGroupError::RetryNextUpdate) => {
                Err(PrepareAssetError::RetryNextUpdate(material))
            }
        }
    }
}

pub type RenderMaterialInstances<M> = ExtractedInstances<AssetId<M>>;

pub fn queue_material_point_clouds<M: PointCloudMaterial>(
    draw_functions: Res<DrawFunctions<OrderIndependentTransparent3d>>,
    point_cloud_pipeline: Res<PointCloudMaterialPipeline<M>>,
    msaa: Res<Msaa>,
    mut pipelines: ResMut<SpecializedRenderPipelines<PointCloudMaterialPipeline<M>>>,
    pipeline_cache: Res<PipelineCache>,
    point_cloud_instances: Res<PointCloudInstances>,
    render_materials: Res<RenderAssets<PreparedPointCloudMaterial<M>>>,
    render_material_instances: Res<RenderMaterialInstances<M>>,
    mut transparent_phases: ResMut<ViewBinnedRenderPhases<OrderIndependentTransparent3d>>,
    mut views: Query<Entity, With<ExtractedView>>,
) where <M as AsBindGroup>::Data: Clone + Hash + Eq {
    let draw_point_cloud = draw_functions.read().id::<DrawPointCloudMaterial<M>>();
    let view_key = if msaa.samples() > 1 {
        MeshPipelineViewLayoutKey::MULTISAMPLED
    } else {
        MeshPipelineViewLayoutKey::empty()
    };
    let point_key = PointCloudPipelineKey {
        msaa_samples: msaa.samples(),
        view_key,
    };
    for view_entity in &mut views {
        let Some(transparent_phase) = transparent_phases.get_mut(&view_entity) else {
            continue;
        };

        for entity in point_cloud_instances.keys().copied() {
            let Some(material_asset_id) = render_material_instances.get(&entity) else {
                continue;
            };
            let Some(material) = render_materials.get(*material_asset_id) else {
                continue;
            };

            let pipeline_key = PointCloudMaterialPipelineKey {
                point_key,
                bind_group_data: material.key.clone(),
            };
            let pipeline = pipelines
                .specialize(&pipeline_cache, &point_cloud_pipeline, pipeline_key);
            let key = OrderIndependentTransparent3dBinKey {
                pipeline,
                draw_function: draw_point_cloud,
            };
            transparent_phase.add(key, entity, true);
        }
    }
}
