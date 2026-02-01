use serde::Serialize;

use crate::gltf::GltfIndex;

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Channel {
    #[serde(rename = "sampler")]
    sampler_index: GltfIndex,
    target: Target,
}

#[derive(Debug, Clone, Default, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Sampler {}

#[derive(Debug, Clone, Serialize)]
pub enum TargetPath {
    Translation,
    Rotation,
    Scale,
    Weights,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Target {
    node: GltfIndex,
    #[serde(rename = "path")]
    target_type: TargetPath,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Animation {
    channels: Vec<Channel>,
    samplers: Vec<Sampler>,
    name: String,
}
