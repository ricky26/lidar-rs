use std::f32::consts::PI;

use bevy::prelude::*;
use bevy::render::render_asset::RenderAssets;
use bevy::render::render_resource::{AsBindGroup, AsBindGroupShaderType, ShaderRef, ShaderType};
use bevy::render::texture::GpuImage;

use crate::point_cloud::PointCloudMaterial;

#[derive(Clone, Default, ShaderType)]
pub struct PointCloudDistanceMaterialUniform {
    pub distance_min: f32,
    pub distance_max: f32,
    pub hue_min: f32,
    pub hue_max: f32,
}

impl AsBindGroupShaderType<PointCloudDistanceMaterialUniform> for PointCloudDistanceMaterial {
    fn as_bind_group_shader_type(
        &self,
        _images: &RenderAssets<GpuImage>,
    ) -> PointCloudDistanceMaterialUniform {
        PointCloudDistanceMaterialUniform {
            distance_min: self.distance_min,
            distance_max: self.distance_max,
            hue_min: self.hue_min,
            hue_max: self.hue_max,
        }
    }
}

#[derive(Clone, Asset, AsBindGroup, Reflect)]
#[uniform(0, PointCloudDistanceMaterialUniform)]
pub struct PointCloudDistanceMaterial {
    pub distance_min: f32,
    pub distance_max: f32,
    pub hue_min: f32,
    pub hue_max: f32,
    #[texture(1)]
    #[sampler(2)]
    pub base_color: Option<Handle<Image>>,
}

impl Default for PointCloudDistanceMaterial {
    fn default() -> Self {
        PointCloudDistanceMaterial {
            distance_min: 0.0,
            distance_max: 100.0,
            hue_min: 0.0,
            hue_max: PI * 1.1,
            base_color: None,
        }
    }
}

impl PointCloudMaterial for PointCloudDistanceMaterial {
    fn fragment_shader() -> ShaderRef {
        ShaderRef::Path("shaders/point_cloud_distance.wgsl".into())
    }
}
