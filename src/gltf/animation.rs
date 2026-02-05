use serde::Serialize;

use crate::gltf::GltfIndex;

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Channel {
    #[serde(rename = "sampler")]
    sampler_index: GltfIndex,
    target: Target,
}

impl Channel {
    pub fn new(sampler_index: GltfIndex, target: Target) -> Self {
        Self {
            sampler_index,
            target,
        }
    }

    pub fn target(&self) -> &Target {
        &self.target
    }

    pub fn sampler_index(&self) -> u32 {
        self.sampler_index
    }
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "UPPERCASE")]
pub enum Interpolation {
    Linear,
    Step,
    CubicSpline,
}

#[derive(Debug, Clone, Default, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Sampler {
    #[serde(rename = "input")]
    timestamps_accessor: GltfIndex,
    interpolation: Option<Interpolation>,
    /// The values which are sampled
    #[serde(rename = "output")]
    values_accessor: GltfIndex,
}

impl Sampler {
    pub fn new(
        timestamps_accessor: GltfIndex,
        interpolation: Option<Interpolation>,
        values_accessor: GltfIndex,
    ) -> Self {
        Self {
            timestamps_accessor,
            interpolation,
            values_accessor,
        }
    }
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "lowercase")]
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

impl Target {
    pub fn new(node: GltfIndex, target_type: TargetPath) -> Self {
        Self { node, target_type }
    }

    pub fn node(&self) -> u32 {
        self.node
    }

    pub fn target_type(&self) -> &TargetPath {
        &self.target_type
    }
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Animation {
    channels: Vec<Channel>,
    samplers: Vec<Sampler>,
    name: String,
}

impl Animation {
    pub fn new(name: Option<String>, channels: &[Channel], samplers: &[Sampler]) -> Self {
        Self {
            channels: channels.to_vec(),
            samplers: samplers.to_vec(),
            name: name.unwrap_or("Unnamed Animation".to_string()),
        }
    }

    pub fn channels(&self) -> &[Channel] {
        &self.channels
    }

    pub fn samplers(&self) -> &[Sampler] {
        &self.samplers
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn channels_mut(&mut self) -> &mut Vec<Channel> {
        &mut self.channels
    }

    pub fn samplers_mut(&mut self) -> &mut Vec<Sampler> {
        &mut self.samplers
    }

    pub fn name_mut(&mut self) -> &mut String {
        &mut self.name
    }
}
