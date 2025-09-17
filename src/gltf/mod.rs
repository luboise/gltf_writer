use std::{collections::HashMap, fs};

use serde::{
    Serialize, Serializer,
    ser::{Error, SerializeMap},
};
use serde_repr::{Deserialize_repr, Serialize_repr};

pub type GltfIndex = u32;

type Vec3<T> = [T; 3];
type Vec4<T> = [T; 4];

type Mat4<T> = [T; 16];

#[derive(Debug, Clone)]
struct Quaternion<T> {
    x: T,
    y: T,
    z: T,
    w: T,
}

#[derive(Debug, Clone)]
pub enum GltfError {
    OutOfRange,
}

#[derive(Debug, Clone, Default)]
pub struct Buffer {
    data: Vec<u8>,
    pub(crate) index: Option<GltfIndex>,
}

impl Serialize for Buffer {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        if self.data.is_empty() {
            Err(S::Error::custom(
                "Unable to serialise a buffer with no data in it.",
            ))
        } else if let Some(index) = self.index {
            let mut map = serializer.serialize_map(None)?;

            let filename = format!("{}.bin", index);

            fs::write(&filename, &self.data).map_err(S::Error::custom)?;
            map.serialize_entry("byteLength", &self.data.len())?;
            map.serialize_entry("uri", &filename)?;
            map.end()
        } else {
            Err(S::Error::custom(
                "Unable to serialise a buffer without an index.",
            ))
        }
    }
}

impl Buffer {
    pub fn new<T: AsRef<[u8]>>(bytes: T) -> Buffer {
        Buffer {
            data: bytes.as_ref().to_vec(),
            ..Self::default()
        }
    }

    pub fn data_from_view(&self, view: &BufferView) -> Result<Vec<u8>, GltfError> {
        todo!();
    }
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct BufferView {
    #[serde(rename = "buffer")]
    pub buffer_index: GltfIndex,
    pub byte_offset: usize,
    pub byte_length: usize,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub byte_stride: Option<usize>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub target: Option<u32>,
}

impl BufferView {
    pub fn new(
        buffer_index: GltfIndex,
        byte_offset: usize,
        byte_length: usize,
        byte_stride: Option<usize>,
        target: Option<u32>,
    ) -> Self {
        Self {
            buffer_index,
            byte_offset,
            byte_length,
            byte_stride,
            target,
        }
    }
}

#[derive(Debug, Clone)]

pub enum NodeTransform {
    Precomputed(Mat4<f32>),
    TRS(Vec3<f32>, Vec3<f32>, Vec3<f32>),
}

impl Serialize for NodeTransform {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let mut map = serializer.serialize_map(None)?;

        match self {
            NodeTransform::Precomputed(pc) => map.serialize_entry("matrix", &pc)?,
            NodeTransform::TRS(translation, rotation, scale) => {
                map.serialize_entry("translation", &translation)?;
                map.serialize_entry("rotation", &rotation)?;
                map.serialize_entry("scale", &scale)?;
            }
        };

        map.end()
    }
}

#[derive(Debug, Clone, Serialize, Default)]
pub struct Node {
    name: Option<String>,

    // Child node indices
    #[serde(rename = "children", skip_serializing_if = "Vec::is_empty")]
    child_indices: Vec<GltfIndex>,

    // Accessors
    #[serde(rename = "camera", skip_serializing_if = "Option::is_none")]
    camera_index: Option<GltfIndex>,

    #[serde(rename = "mesh", skip_serializing_if = "Option::is_none")]
    mesh_index: Option<GltfIndex>,

    #[serde(rename = "skin", skip_serializing_if = "Option::is_none")]
    skin_index: Option<GltfIndex>,

    #[serde(flatten, skip_serializing_if = "Option::is_none")]
    transform: Option<NodeTransform>,
}

impl Node {
    pub fn new(name: Option<String>) -> Self {
        Node {
            name,
            ..Default::default()
        }
    }

    pub fn name(&self) -> Option<&String> {
        self.name.as_ref()
    }
    pub fn set_name(&mut self, name: Option<String>) {
        self.name = name;
    }

    pub fn mesh_index(&self) -> Option<GltfIndex> {
        self.mesh_index
    }
    pub fn set_mesh_index(&mut self, mesh_index: Option<GltfIndex>) {
        self.mesh_index = mesh_index;
    }

    pub fn camera_index(&self) -> Option<GltfIndex> {
        self.camera_index
    }
    pub fn set_camera_index(&mut self, camera_index: Option<GltfIndex>) {
        self.camera_index = camera_index;
    }

    pub fn skin_index(&self) -> Option<GltfIndex> {
        self.skin_index
    }
    pub fn set_skin_index(&mut self, skin_index: Option<GltfIndex>) {
        self.skin_index = skin_index;
    }

    pub fn transform(&self) -> &Option<NodeTransform> {
        &self.transform
    }
    pub fn set_transform(&mut self, transform: Option<NodeTransform>) {
        self.transform = transform;
    }

    pub fn add_child(&mut self, child_index: GltfIndex) {
        self.child_indices.push(child_index);
    }
    pub fn children(&self) -> &[GltfIndex] {
        &self.child_indices
    }
    pub fn children_mut(&mut self) -> &mut [GltfIndex] {
        &mut self.child_indices
    }
}

#[derive(Debug, Clone, Default, Serialize)]
pub struct Scene {
    name: String,

    #[serde(rename = "nodes")]
    root_nodes: Vec<GltfIndex>,
}

impl Scene {
    pub fn new(name: String) -> Self {
        Scene {
            name,
            root_nodes: vec![],
        }
    }
}

#[derive(Debug, Clone, Serialize_repr, Deserialize_repr)]
#[repr(u32)]
pub enum TopologyMode {
    Points = 0,
    Lines = 1,
    LineLoop = 2,
    LineStrip = 3,
    Triangles = 4,
    TriangleStrip = 5,
    TriangleFan = 6,
}

#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub enum VertexAttribute {
    Position,
    Normal,
    Tangent,
    TexCoord(GltfIndex),
    Colour(GltfIndex),
    Joints(GltfIndex),
    Weights(GltfIndex),
    User(String),
}

impl ToString for VertexAttribute {
    fn to_string(&self) -> String {
        match self {
            VertexAttribute::Position => "POSITION".to_string(),
            VertexAttribute::Normal => "NORMAL".to_string(),
            VertexAttribute::Tangent => "TANGENT".to_string(),
            VertexAttribute::TexCoord(num) => format!("TEXCOORD_{}", num),
            VertexAttribute::Colour(num) => format!("COLOR_{}", num),
            VertexAttribute::Joints(num) => format!("JOINTS_{}", num),
            VertexAttribute::Weights(num) => format!("WEIGHTS_{}", num),
            VertexAttribute::User(string) => string.clone(),
        }
    }
}

impl Serialize for VertexAttribute {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_str(&self.to_string())
    }
}

fn serialize_attributes<S>(
    attributes: &HashMap<VertexAttribute, GltfIndex>,
    serializer: S,
) -> Result<S::Ok, S::Error>
where
    S: Serializer,
{
    let mut map = serializer.serialize_map(None)?;

    for (key, value) in attributes {
        map.serialize_entry(&key.to_string(), &value);
    }

    map.end()
}

#[derive(Debug, Clone, Default, Serialize)]
pub struct Primitive {
    #[serde(rename = "indices", skip_serializing_if = "Option::is_none")]
    pub indices_accessor: Option<GltfIndex>,
    #[serde(rename = "mode", skip_serializing_if = "Option::is_none")]
    pub topology_type: Option<TopologyMode>,

    // Attributes
    #[serde(serialize_with = "serialize_attributes")]
    pub attributes: HashMap<VertexAttribute, GltfIndex>,
}

impl Primitive {
    pub fn set_attribute(&mut self, attribute: VertexAttribute, value: GltfIndex) {
        self.attributes.insert(attribute, value);
    }
}

#[derive(Debug, Default, Clone, Serialize)]
pub struct Mesh {
    name: String,
    primitives: Vec<Primitive>,
}

impl Mesh {
    pub fn new(name: String) -> Self {
        Self {
            name,
            primitives: vec![],
        }
    }

    pub fn add_primitive(&mut self, primitive: Primitive) -> &mut Primitive {
        let new_index = self.primitives.len();
        self.primitives.push(primitive);

        self.primitives.get_mut(new_index).unwrap()
    }
    pub fn primitives(&self) -> &[Primitive] {
        &self.primitives
    }
    pub fn primitives_mut(&mut self) -> &mut Vec<Primitive> {
        &mut self.primitives
    }
}

#[derive(Debug, Clone, Serialize_repr, Deserialize_repr)]
#[repr(usize)]
pub enum AccessorDataType {
    I8 = 5120,
    U8 = 5121,
    I16 = 5122,
    U16 = 5123,
    I32 = 5124,
    U32 = 5125,
    F32 = 5126,
}

impl AccessorDataType {
    pub fn is_signed(&self) -> bool {
        match self {
            Self::U8 | Self::U16 | Self::U32 => false,
            Self::I8 | Self::I16 | Self::I32 | Self::F32 => true,
        }
    }

    pub fn bit_size(&self) -> u8 {
        match self {
            Self::I8 | Self::U8 => 8,
            Self::I16 | Self::U16 => 16,
            Self::I32 | Self::U32 | Self::F32 => 32,
        }
    }
}

#[derive(Debug, Clone, Serialize)]
pub enum AccessorComponentCount {
    SCALAR,
    VEC2,
    VEC3,
    VEC4,
    MAT2,
    MAT3,
    MAT4,
}

impl AccessorComponentCount {
    pub fn component_count(&self) -> GltfIndex {
        match self {
            Self::SCALAR => 1,
            Self::VEC2 => 2,
            Self::VEC3 => 3,
            Self::VEC4 | Self::MAT2 => 4,
            Self::MAT3 => 9,
            Self::MAT4 => 16,
        }
    }
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Accessor {
    #[serde(rename = "bufferView")]
    pub buffer_view_index: GltfIndex,
    pub byte_offset: usize,

    #[serde(rename = "componentType")]
    pub data_type: AccessorDataType,

    #[serde(rename = "count")]
    pub data_count: usize,

    #[serde(rename = "type")]
    pub component_count: AccessorComponentCount,
}

impl Accessor {
    pub fn new(
        buffer_view_index: GltfIndex,
        byte_offset: usize,
        data_type: AccessorDataType,
        data_count: usize,
        parts_per_data: AccessorComponentCount,
    ) -> Self {
        Self {
            buffer_view_index,
            byte_offset,
            data_type,
            data_count,
            component_count: parts_per_data,
        }
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct Asset {
    version: String,

    #[serde(skip_serializing_if = "Option::is_none")]
    copyright: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    generator: Option<String>,
}

impl Default for Asset {
    fn default() -> Self {
        Self {
            version: "2.0".to_string(),
            copyright: Default::default(),
            generator: Default::default(),
        }
    }
}

impl Asset {
    pub fn version(&self) -> &str {
        &self.version
    }
    pub fn set_version(&mut self, version: String) {
        self.version = version;
    }

    pub fn copyright(&self) -> Option<&String> {
        self.copyright.as_ref()
    }
    pub fn set_copyright(&mut self, copyright: Option<String>) {
        self.copyright = copyright;
    }

    pub fn generator(&self) -> Option<&String> {
        self.generator.as_ref()
    }
    pub fn set_generator(&mut self, generator: Option<String>) {
        self.generator = generator;
    }
}

#[derive(Debug, Clone, Default, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Gltf {
    accessors: Vec<Accessor>,

    buffer_views: Vec<BufferView>,
    buffers: Vec<Buffer>,
    scenes: Vec<Scene>,
    meshes: Vec<Mesh>,
    nodes: Vec<Node>,

    asset: Asset,
}

/*
impl serde::Serialize for Gltf {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        let mut map = serializer.serialize_map(None)?;

        map.serialize_entry("version", "2.0")?;
        map.serialize_entry("accessors", &self.accessors)?;
        map.serialize_entry("buffers", &self.buffers);

        // map.serialize_entry("buffers", &self.buffers)?;

        todo!()
    }
}
*/

impl Gltf {
    pub fn add_buffer(&mut self, buffer: Buffer) -> GltfIndex {
        self.buffers.push(buffer);
        (self.buffers.len() - 1) as GltfIndex
    }
    pub fn buffers(&self) -> &[Buffer] {
        &self.buffers
    }
    pub fn buffers_mut(&mut self) -> &mut Vec<Buffer> {
        &mut self.buffers
    }

    pub fn add_buffer_view(&mut self, bv: BufferView) -> GltfIndex {
        self.buffer_views.push(bv);
        (self.buffer_views.len() - 1) as GltfIndex
    }
    pub fn buffer_views(&self) -> &[BufferView] {
        &self.buffer_views
    }
    pub fn buffer_views_mut(&mut self) -> &mut Vec<BufferView> {
        &mut self.buffer_views
    }

    pub fn add_scene(&mut self, scene: Scene) -> GltfIndex {
        self.scenes.push(scene);
        (self.scenes.len() - 1) as GltfIndex
    }
    pub fn scenes(&self) -> &[Scene] {
        &self.scenes
    }
    pub fn scenes_mut(&mut self) -> &mut Vec<Scene> {
        &mut self.scenes
    }

    pub fn add_accessor(&mut self, accessor: Accessor) -> GltfIndex {
        self.accessors.push(accessor);
        (self.accessors.len() - 1) as GltfIndex
    }
    pub fn accessors(&self) -> &[Accessor] {
        &self.accessors
    }
    pub fn accessors_mut(&mut self) -> &mut Vec<Accessor> {
        &mut self.accessors
    }

    pub fn add_mesh(&mut self, mesh: Mesh) -> GltfIndex {
        self.meshes.push(mesh);
        (self.meshes.len() - 1) as GltfIndex
    }
    pub fn meshes(&self) -> &[Mesh] {
        &self.meshes
    }
    pub fn meshes_mut(&mut self) -> &mut Vec<Mesh> {
        &mut self.meshes
    }

    pub fn add_node(&mut self, node: Node) -> GltfIndex {
        self.nodes.push(node);
        (self.nodes.len() - 1) as GltfIndex
    }
    pub fn nodes(&self) -> &[Node] {
        &self.nodes
    }
    pub fn nodes_mut(&mut self) -> &mut Vec<Node> {
        &mut self.nodes
    }

    pub fn prepare_for_export(&mut self) -> Result<(), String> {
        for (i, buffer) in self.buffers.iter_mut().enumerate() {
            buffer.index = Some(i as GltfIndex);
        }

        Ok(())
    }
}

/*
pub(crate) trait GltfValue {
     fn get_value(&self, gltf: &Gltf, index: u32);
     fn get_value_mut(&self, gltf:  &mut Gltf, index: u32);
}

pub(crate) trait CanGetTheThing<T>{
    fn get_thing(&self, gltf: &Gltf) -> &[T];
    fn get_thing_mut(&self, gltf: &mut Gltf) -> &mut [T];
}


#[derive(Debug, Clone, Default)]
pub struct Index<T> where T: CanGetTheThing<T>{
    index: u32
}



impl<T> Index<T> {
    pub fn value(&self, Gltf)
}


impl<T> for Index<T> {

}

impl<T> GltfValue for Index<T> {
    fn get_value(&self, gltf: &Gltf) {
        T::get_value()
    }

    fn get_value_mut(&self, gltf: &mut Gltf) {
        todo!()
    }
}
*/
