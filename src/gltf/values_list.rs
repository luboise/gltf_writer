use crate::gltf::AccessorDataType;

use serde::{Serialize, Serializer, ser::SerializeSeq};

#[derive(Debug, Clone)]
pub(crate) struct ValuesList {
    bytes: Vec<u8>,
    data_type: AccessorDataType,
}

impl Serialize for ValuesList {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let mut chunks = self.bytes.chunks_exact(self.value_size());

        if self.len() == 1 {
            return match self.data_type {
                AccessorDataType::I8 => {
                    let byte = chunks.nth(0).expect("Unable to get first byte.")[0] as i8;
                    serializer.serialize_i8(byte)
                }
                AccessorDataType::U8 => {
                    let byte = chunks.nth(0).expect("Unable to get first byte.")[0];
                    serializer.serialize_u8(byte)
                }
                AccessorDataType::I16 => {
                    let bytes = (chunks.nth(0).expect("Unable to get chunk."))
                        .try_into()
                        .unwrap();
                    serializer.serialize_i16(i16::from_le_bytes(bytes))
                }
                AccessorDataType::U16 => {
                    let bytes = (chunks.nth(0).expect("Unable to get chunk."))
                        .try_into()
                        .unwrap();
                    serializer.serialize_u16(u16::from_le_bytes(bytes))
                }
                AccessorDataType::I32 => {
                    let bytes = (chunks.nth(0).expect("Unable to get chunk."))
                        .try_into()
                        .unwrap();
                    serializer.serialize_i32(i32::from_le_bytes(bytes))
                }
                AccessorDataType::U32 => {
                    let bytes = (chunks.nth(0).expect("Unable to get chunk."))
                        .try_into()
                        .unwrap();
                    serializer.serialize_u32(u32::from_le_bytes(bytes))
                }
                AccessorDataType::F32 => {
                    let bytes = (chunks.nth(0).expect("Unable to get chunk."))
                        .try_into()
                        .unwrap();
                    serializer.serialize_f32(f32::from_le_bytes(bytes))
                }
            };
        }

        let mut seq = serializer.serialize_seq(Some(self.len()))?;

        match self.data_type {
            AccessorDataType::I8 => {
                chunks.for_each(|val| {
                    seq.serialize_element(&val)
                        .expect("Failed to serialize i8.")
                });
            }
            AccessorDataType::U8 => {
                chunks.for_each(|val| {
                    seq.serialize_element(&val)
                        .expect("Failed to serialize u8.")
                });
            }
            AccessorDataType::I16 => {
                chunks
                    .map(|chunk| i16::from_le_bytes(chunk.try_into().unwrap()))
                    .for_each(|val| {
                        seq.serialize_element(&val)
                            .expect("Failed to serialize i16.")
                    });
            }
            AccessorDataType::U16 => {
                chunks
                    .map(|chunk| u16::from_le_bytes(chunk.try_into().unwrap()))
                    .for_each(|val| {
                        seq.serialize_element(&val)
                            .expect("Failed to serialize u16.")
                    });
            }
            AccessorDataType::I32 => {
                chunks
                    .map(|chunk| i32::from_le_bytes(chunk.try_into().unwrap()))
                    .for_each(|val| {
                        seq.serialize_element(&val)
                            .expect("Failed to serialize i32.")
                    });
            }
            AccessorDataType::U32 => {
                chunks
                    .map(|chunk| u32::from_le_bytes(chunk.try_into().unwrap()))
                    .for_each(|val| {
                        seq.serialize_element(&val)
                            .expect("Failed to serialize u32.")
                    });
            }
            AccessorDataType::F32 => {
                chunks
                    .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
                    .for_each(|val| {
                        seq.serialize_element(&val)
                            .expect("Failed to serialize f32.")
                    });
            }
        };

        seq.end()
    }
}

impl ValuesList {
    pub fn new(bytes: Vec<u8>, data_type: AccessorDataType) -> Self {
        Self { bytes, data_type }
    }

    pub fn value_size(&self) -> usize {
        let val_size = self.data_type.byte_size() as usize;

        assert!(val_size > 0, "Value size should never be 0.");
        val_size
    }

    pub fn len(&self) -> usize {
        self.bytes.len() / self.value_size()
    }
}
