#[derive(Debug, Clone)]
pub enum GltfError {
    OutOfRange,
}

#[derive(Debug, Clone, Default)]
pub struct Buffer {
    data: Vec<u8>,
}

#[derive(Debug, Clone)]
pub struct BufferView {
    buffer_index: usize,
    byte_offset: usize,
    byte_length: usize,
    byte_stride: Option<usize>,
    target: Option<u32>,
}

impl Buffer {
    pub fn new<T: AsRef<[u8]>>(bytes: T) -> Buffer {
        Buffer {
            data: bytes.as_ref().to_vec(),
        }
    }

    pub fn data_from_view(&self, view: &BufferView) -> Result<Vec<u8>, GltfError> {
        Err(GltfError::OutOfRange)
    }
}

#[derive(Debug, Clone)]
pub struct Node {
    children: Vec<u32>,
}

#[derive(Debug, Clone, Default)]
pub struct Scene {}

#[derive(Debug, Clone, Default)]
pub struct Mesh {
    name: String,
    primitives: Vec<u32>,
}

#[derive(Debug, Clone, Default)]
pub struct Primitive {}

#[derive(Debug, Clone)]
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

#[derive(Debug, Clone)]
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
    pub fn component_count(&self) -> usize {
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

#[derive(Debug, Clone)]
pub struct Accessor {
    buffer_view: usize,
    byte_offset: usize,
    component_type: AccessorDataType,
    count: usize,
    component_count: AccessorComponentCount,
}

#[derive(Debug, Clone, Default)]
pub struct Gltf {
    buffers: Vec<Buffer>,
    buffer_views: Vec<BufferView>,
    scenes: Vec<Scene>,
}

impl Gltf {
    pub fn buffers(&self) -> &[Buffer] {
        &self.buffers
    }

    pub fn buffers_mut(&mut self) -> &mut Vec<Buffer> {
        &mut self.buffers
    }

    pub fn buffer_views(&self) -> &[BufferView] {
        &self.buffer_views
    }

    pub fn buffer_views_mut(&mut self) -> &mut Vec<BufferView> {
        &mut self.buffer_views
    }

    pub fn scenes(&self) -> &[Scene] {
        &self.scenes
    }

    pub fn scenes_mut(&mut self) -> &mut Vec<Scene> {
        &mut self.scenes
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
