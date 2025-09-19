use std::{fs, path::Path};

use crate::gltf::GltfError;

#[derive(Debug, Clone)]
pub struct GltfSerialJSON {
    gltf_bytes: Vec<u8>,
    buffers: Vec<Vec<u8>>,
}

impl GltfSerialJSON {
    pub fn new(gltf_bytes: Vec<u8>) -> Self {
        Self {
            gltf_bytes,
            buffers: vec![],
        }
    }

    pub fn add_buffer<B: AsRef<[u8]>>(&mut self, bytes: B) {
        self.buffers.push(bytes.as_ref().to_vec())
    }

    pub fn gltf_bytes(&self) -> &[u8] {
        &self.gltf_bytes
    }
    pub fn buffers(&self) -> &[Vec<u8>] {
        &self.buffers
    }

    pub fn export<P: AsRef<Path>>(&self, path: P) -> Result<(), GltfError> {
        let export_path = path.as_ref();

        let error_mapper = |e| {
            GltfError::SerialisationError(format!(
                "Failed to check if path {} exists. {}",
                export_path.display(),
                e
            ))
        };

        if fs::exists(export_path).map_err(error_mapper)? {
            return Err(GltfError::SerialisationError(format!(
                "Path {} already exists.",
                export_path.display()
            )));
        }

        let file_stem = export_path
            .file_stem()
            .ok_or(format!(
                "The specified path {}, did not have a file stem.",
                export_path.display()
            ))?
            .to_str()
            .ok_or("Error converting OsString to regular string.".to_string())?
            .to_string();

        let out_dir = export_path.join(file_stem);

        if !out_dir.exists() {
            fs::create_dir(&out_dir)?;
        }
        // If its an existing file or non-directory, then we can't write files into it
        else if !out_dir.is_dir() {
            return Err(GltfError::SerialisationError(format!(
                "Out dir {} exists and is not a file. Unable to export to that folder.",
                out_dir.display()
            )));
        }

        fs::write(export_path, &self.gltf_bytes)?;

        for (i, buffer) in self.buffers.iter().enumerate() {
            let bin_path = out_dir.join(format!("{}.bin", i));

            fs::write(bin_path, buffer)?;
        }

        Ok(())
    }
}

pub type GltfExportType = SerialGltfType;

#[derive(Debug, Clone)]
pub enum SerialGltfType {
    JSON,
    // TODO: Implement binary .glb exporting
    // Binary,
}
