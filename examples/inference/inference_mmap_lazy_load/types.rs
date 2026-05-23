//! Support module for the sibling `main.rs` recipe.
//!
//! Contract: contracts/recipe-iiur-v1.yaml (inherited from main.rs — Invariant B)
#![allow(
    dead_code,
    unused_imports,
    unreachable_pub,
    clippy::trivially_copy_pass_by_ref,
    clippy::wildcard_imports
)]
#[allow(unused_imports)]
use apr_cookbook::prelude::*;
use proptest::prelude::*;
use rand::Rng;
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::Path;
use std::time::{Duration, Instant};

// ============================================================================
// Constants
// ============================================================================

/// Magic bytes identifying our synthetic APR-like binary format.
pub const MAGIC: &[u8; 4] = b"APRM";

/// Header size in bytes: magic(4) + version(4) + tensor_count(4) = 12.
pub const HEADER_SIZE: usize = 12;

// Bytes per tensor metadata entry: name_len(4) + name(64) + ndims(4) +
/// dims(4*4) + dtype(4) + offset(8) + length(8) = 108.
pub const TENSOR_META_SIZE: usize = 108;

/// Maximum tensor name length stored in the file.
pub const MAX_NAME_LEN: usize = 64;

// ============================================================================
// Data Structures
// ============================================================================

/// Metadata for a single tensor stored inside the model file.
#[derive(Debug, Clone)]
#[allow(dead_code)] // shape, dtype used for format fidelity; read in tests
pub struct ModelTensor {
    pub name: String,
    pub shape: Vec<usize>,
    pub dtype: String,
    pub offset: u64,
    pub length: u64,
}

/// Tracks a region that has been "mapped" (read on demand).
#[derive(Debug, Clone)]
#[allow(dead_code)] // offset, length retained for diagnostic introspection
pub struct MappedRegion {
    pub name: String,
    pub offset: u64,
    pub length: u64,
    pub access_count: u64,
}

// Simulated memory-mapped model loader.
//
// Instead of true OS-level mmap (which requires `unsafe`), this loader
// opens a file handle and seeks to requested byte ranges, achieving the
/// same selective-read semantics without any `unsafe` code.
#[derive(Debug)]
#[allow(dead_code)] // file_size used for diagnostics in tests
pub struct MmapModelLoader {
    pub path: std::path::PathBuf,
    pub file_size: u64,
    pub mapped_regions: Vec<MappedRegion>,
    pub total_bytes_read: u64,
}

// ============================================================================
// Model File Creation
// ============================================================================

// Write a synthetic APR-like binary file with a header and `size_mb` worth
/// of tensor data.  Returns the tensor index for later selective loading.
pub fn create_test_model(
    path: &Path,
    size_mb: usize,
    rng: &mut rand::rngs::StdRng,
) -> Result<Vec<ModelTensor>> {
    let total_bytes = size_mb * 1024 * 1024;
    // Each tensor holds 256 KiB of f32 data (65 536 floats).
    let floats_per_tensor: usize = 65_536;
    let bytes_per_tensor = floats_per_tensor * 4;
    let n_tensors = total_bytes / bytes_per_tensor;

    if n_tensors == 0 {
        return Err(CookbookError::invalid_format(
            "model size too small to hold even one tensor",
        ));
    }

    let mut file = std::fs::File::create(path)?;

    // --- Header ---
    file.write_all(MAGIC)?;
    file.write_all(&1u32.to_le_bytes())?; // version
    file.write_all(&(n_tensors as u32).to_le_bytes())?;

    // --- Tensor metadata ---
    let meta_block_size = n_tensors * TENSOR_META_SIZE;
    let data_offset_base = (HEADER_SIZE + meta_block_size) as u64;

    let mut tensors = Vec::with_capacity(n_tensors);
    for i in 0..n_tensors {
        let name = format!("layer_{i:04}.weight");
        let offset = data_offset_base + (i as u64) * (bytes_per_tensor as u64);
        let length = bytes_per_tensor as u64;

        // Write fixed-size name field
        let name_bytes = name.as_bytes();
        let name_len = name_bytes.len().min(MAX_NAME_LEN);
        file.write_all(&(name_len as u32).to_le_bytes())?;
        let mut name_buf = [0u8; MAX_NAME_LEN];
        name_buf[..name_len].copy_from_slice(&name_bytes[..name_len]);
        file.write_all(&name_buf)?;

        // ndims + shape (up to 4 dims, padded)
        let shape = vec![256, 256];
        file.write_all(&(shape.len() as u32).to_le_bytes())?;
        for d in 0..4 {
            let val = if d < shape.len() { shape[d] as u32 } else { 0 };
            file.write_all(&val.to_le_bytes())?;
        }

        // dtype tag (0 = f32)
        file.write_all(&0u32.to_le_bytes())?;
        file.write_all(&offset.to_le_bytes())?;
        file.write_all(&length.to_le_bytes())?;

        tensors.push(ModelTensor {
            name,
            shape,
            dtype: "f32".to_string(),
            offset,
            length,
        });
    }

    // --- Tensor data ---
    let mut buf = vec![0u8; bytes_per_tensor];
    for _ in 0..n_tensors {
        // Fill with deterministic random f32 values
        for chunk in buf.chunks_exact_mut(4) {
            let val: f32 = rng.gen_range(-1.0..1.0);
            chunk.copy_from_slice(&val.to_le_bytes());
        }
        file.write_all(&buf)?;
    }

    file.flush()?;
    Ok(tensors)
}

// ============================================================================
// Mmap-Style Loading
// ============================================================================

/// Open a model file and prepare a loader without reading tensor data.
pub fn simulate_mmap_load(path: &Path, tensors: &[ModelTensor]) -> Result<MmapModelLoader> {
    let metadata = std::fs::metadata(path)?;

    // Validate that the file is large enough for all declared tensors
    for t in tensors {
        let end = t.offset + t.length;
        if end > metadata.len() {
            return Err(CookbookError::invalid_format(format!(
                "tensor '{}' extends past end of file ({} > {})",
                t.name,
                end,
                metadata.len()
            )));
        }
    }

    Ok(MmapModelLoader {
        path: path.to_path_buf(),
        file_size: metadata.len(),
        mapped_regions: Vec::new(),
        total_bytes_read: 0,
    })
}

/// Load exactly one tensor on demand, tracking the byte range accessed.
pub fn lazy_load_tensor(loader: &mut MmapModelLoader, tensor: &ModelTensor) -> Result<Vec<f32>> {
    let mut file = std::fs::File::open(&loader.path)?;
    file.seek(SeekFrom::Start(tensor.offset))?;

    let n_bytes = tensor.length as usize;
    let mut raw = vec![0u8; n_bytes];
    file.read_exact(&mut raw)?;

    // Decode little-endian f32 values
    let n_floats = n_bytes / 4;
    let mut floats = Vec::with_capacity(n_floats);
    for chunk in raw.chunks_exact(4) {
        let bytes: [u8; 4] = [chunk[0], chunk[1], chunk[2], chunk[3]];
        floats.push(f32::from_le_bytes(bytes));
    }

    // Track the mapped region
    if let Some(region) = loader
        .mapped_regions
        .iter_mut()
        .find(|r| r.name == tensor.name)
    {
        region.access_count += 1;
    } else {
        loader.mapped_regions.push(MappedRegion {
            name: tensor.name.clone(),
            offset: tensor.offset,
            length: tensor.length,
            access_count: 1,
        });
        loader.total_bytes_read += tensor.length;
    }

    Ok(floats)
}

// ============================================================================
// Benchmarks
// ============================================================================

/// Eager load: read the entire file into memory.
pub fn benchmark_eager_load(path: &Path) -> Result<(Duration, usize)> {
    let start = Instant::now();
    let data = std::fs::read(path)?;
    let elapsed = start.elapsed();
    Ok((elapsed, data.len()))
}

/// Lazy load: read only the tensors at the given indices.
pub fn benchmark_lazy_load(
    path: &Path,
    tensors: &[ModelTensor],
    indices: &[usize],
) -> Result<(Duration, usize)> {
    let mut loader = simulate_mmap_load(path, tensors)?;
    let start = Instant::now();

    for &idx in indices {
        if idx < tensors.len() {
            let _ = lazy_load_tensor(&mut loader, &tensors[idx])?;
        }
    }

    let elapsed = start.elapsed();
    Ok((elapsed, loader.total_bytes_read as usize))
}

// ============================================================================
// Main
// ============================================================================
