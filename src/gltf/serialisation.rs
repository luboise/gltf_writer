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
        let gltf_path = path.as_ref();

        let error_mapper = |e| {
            GltfError::SerialisationError(format!(
                "Failed to check if path {} exists. {}",
                gltf_path.display(),
                e
            ))
        };

        if fs::exists(gltf_path).map_err(error_mapper)? {
            return Err(GltfError::SerialisationError(format!(
                "Path {} already exists.",
                gltf_path.display()
            )));
        }

        let export_dir = gltf_path.parent().ok_or(format!(
            "Unable to get parent from export path {}",
            gltf_path.display()
        ))?;

        println!("Export dir: {}", export_dir.display());

        if !export_dir.exists() {
            println!("  export dir was not found. Creating it.");
            fs::create_dir_all(export_dir)?;
        }

        let file_stem = gltf_path
            .file_stem()
            .ok_or(format!(
                "The specified path {}, did not have a file stem.",
                gltf_path.display()
            ))?
            .to_str()
            .ok_or("Error converting OsString to regular string.".to_string())?
            .to_string();

        println!("File stem: {}", &file_stem);

        fs::write(gltf_path, &self.gltf_bytes)?;

        // TODO: Implement asset dir semantics into the buffers in the gltf bytes
        /*
        let asset_dir = export_dir.join(file_stem);

        if !asset_dir.exists() {
            println!("Creating dir {}", asset_dir.display());
            fs::create_dir(&asset_dir)?;
        }
        // If its an existing file or non-directory, then we can't write files into it

        if !asset_dir.is_dir() {
            return Err(GltfError::SerialisationError(format!(
                "Out dir {} exists and is not a file. Unable to export to that folder.",
                asset_dir.display()
            )));
        }
        */

        for (i, buffer) in self.buffers.iter().enumerate() {
            let bin_path = export_dir.join(format!("{}.bin", i));

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
