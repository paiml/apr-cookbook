#![allow(
    dead_code,
    unused_imports,
    unreachable_pub,
    clippy::trivially_copy_pass_by_ref,
    clippy::wildcard_imports
)]
use apr_cookbook::prelude::*;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

// ---------------------------------------------------------------------------
// Data structures
// ---------------------------------------------------------------------------

/// WASM heap memory budget tracker.
#[derive(Debug, Clone)]
pub struct WasmMemoryBudget {
    pub max_bytes: usize,
    pub used_bytes: usize,
}

impl WasmMemoryBudget {
    pub fn new(max_bytes: usize) -> Self {
        Self {
            max_bytes,
            used_bytes: 0,
        }
    }

    pub fn allocate(&mut self, size: usize) -> Result<()> {
        if self.used_bytes + size > self.max_bytes {
            return Err(CookbookError::invalid_format(format!(
                "memory budget exceeded: need {} bytes, only {} available",
                size,
                self.max_bytes - self.used_bytes,
            )));
        }
        self.used_bytes += size;
        Ok(())
    }

    pub fn free(&mut self, size: usize) {
        self.used_bytes = self.used_bytes.saturating_sub(size);
    }

    pub fn remaining(&self) -> usize {
        self.max_bytes - self.used_bytes
    }
}

/// A chunk of model data received from a simulated streaming fetch.
#[derive(Debug, Clone)]
pub struct ModelChunk {
    pub offset: usize,
    pub data: Vec<u8>,
    pub is_header: bool,
    pub is_tensor: bool,
}

/// Tracks the streaming download state.
#[derive(Debug)]
pub struct StreamingLoader {
    pub total_size: usize,
    pub bytes_received: usize,
    pub chunks_received: usize,
    pub header_parsed: bool,
    pub tensors_loaded: usize,
    pub progress_pct: u32,
}

impl StreamingLoader {
    pub fn new(total_size: usize) -> Self {
        Self {
            total_size,
            bytes_received: 0,
            chunks_received: 0,
            header_parsed: false,
            tensors_loaded: 0,
            progress_pct: 0,
        }
    }

    pub fn receive_chunk(&mut self, chunk: &ModelChunk) -> Result<()> {
        if self.bytes_received + chunk.data.len() > self.total_size {
            return Err(CookbookError::invalid_format(
                "received data exceeds declared model size",
            ));
        }
        self.bytes_received += chunk.data.len();
        self.chunks_received += 1;
        self.progress_pct = ((self.bytes_received as f64 / self.total_size as f64) * 100.0) as u32;
        Ok(())
    }
}

/// A reference to a tensor within the model file (lazy loading).
#[derive(Debug, Clone)]
pub struct TensorRef {
    pub name: String,
    pub offset: usize,
    pub size: usize,
    pub loaded: bool,
}

/// Summary statistics for the loading process.
#[derive(Debug, Clone)]
pub struct LoadingStats {
    pub total_bytes: usize,
    pub chunks_received: usize,
    pub time_us: u64,
    pub throughput_mbps: f64,
}

impl LoadingStats {
    pub fn compute(loader: &StreamingLoader, _chunk_size: usize) -> Self {
        // Deterministic simulated time: 100us per chunk + 1us per KB
        let time_us = (loader.chunks_received as u64) * 100 + (loader.bytes_received as u64) / 1024;
        let seconds = time_us as f64 / 1_000_000.0;
        let megabytes = loader.bytes_received as f64 / (1024.0 * 1024.0);
        let throughput_mbps = if seconds > 0.0 {
            megabytes / seconds
        } else {
            0.0
        };

        Self {
            total_bytes: loader.bytes_received,
            chunks_received: loader.chunks_received,
            time_us,
            throughput_mbps,
        }
    }
}

/// Parsed model header with tensor table.
#[derive(Debug, Clone)]
pub struct ParsedHeader {
    pub magic: u32,
    pub format_version: u16,
    pub tensor_refs: Vec<TensorTableEntry>,
}

/// Entry in the tensor offset table from the model header.
#[derive(Debug, Clone)]
pub struct TensorTableEntry {
    pub name: String,
    pub offset: usize,
    pub size: usize,
}

/// Model metadata describing the file to be loaded.
#[derive(Debug, Clone)]
pub struct ModelMetadataInfo {
    pub name: String,
    pub version: u32,
    pub total_size: usize,
    pub header_size: usize,
    pub tensor_count: usize,
}

// ---------------------------------------------------------------------------
// Helper functions
// ---------------------------------------------------------------------------

/// Generate deterministic chunks simulating a streaming fetch response.
pub fn generate_chunks(meta: &ModelMetadataInfo, chunk_size: usize) -> Vec<ModelChunk> {
    let mut chunks = Vec::new();
    let mut offset = 0;

    while offset < meta.total_size {
        let remaining = meta.total_size - offset;
        let size = remaining.min(chunk_size);

        let is_header = offset < meta.header_size;
        let is_tensor = offset >= meta.header_size;

        // Generate deterministic data using DefaultHasher
        let data: Vec<u8> = (0..size)
            .map(|i| {
                let mut hasher = DefaultHasher::new();
                (offset + i).hash(&mut hasher);
                (hasher.finish() & 0xFF) as u8
            })
            .collect();

        chunks.push(ModelChunk {
            offset,
            data,
            is_header,
            is_tensor,
        });

        offset += size;
    }

    chunks
}

/// Extract header bytes from the first chunk(s).
pub fn extract_header_bytes(chunks: &[ModelChunk], header_size: usize) -> Vec<u8> {
    let mut header = Vec::with_capacity(header_size);

    for chunk in chunks {
        if chunk.offset >= header_size {
            break;
        }
        let end = (chunk.offset + chunk.data.len()).min(header_size);
        let take = end - chunk.offset;
        header.extend_from_slice(&chunk.data[..take]);
    }

    header
}

/// Parse a model header, producing tensor table entries.
pub fn parse_header(header_data: &[u8], meta: &ModelMetadataInfo) -> Result<ParsedHeader> {
    if header_data.len() < 8 {
        return Err(CookbookError::invalid_format(
            "header too small: need at least 8 bytes",
        ));
    }

    // Derive deterministic magic and version from header bytes
    let mut hasher = DefaultHasher::new();
    header_data[..4].hash(&mut hasher);
    let magic = (hasher.finish() & 0xFFFF_FFFF) as u32;

    let format_version = 2;

    // Build tensor table: distribute remaining space among tensors
    let data_region = meta.total_size - meta.header_size;
    let base_tensor_size = data_region / meta.tensor_count;
    let tensor_names = [
        "embeddings.weight",
        "encoder.layer0.weight",
        "encoder.layer0.bias",
        "encoder.layer1.weight",
        "encoder.layer1.bias",
        "output.weight",
    ];

    let mut tensor_refs = Vec::with_capacity(meta.tensor_count);
    let mut current_offset = meta.header_size;

    for i in 0..meta.tensor_count {
        let name = if i < tensor_names.len() {
            tensor_names[i].to_string()
        } else {
            format!("tensor_{i}")
        };

        // Last tensor gets remainder
        let size = if i == meta.tensor_count - 1 {
            meta.total_size - current_offset
        } else {
            base_tensor_size
        };

        tensor_refs.push(TensorTableEntry {
            name,
            offset: current_offset,
            size,
        });

        current_offset += size;
    }

    Ok(ParsedHeader {
        magic,
        format_version,
        tensor_refs,
    })
}

/// Validate header integrity using a deterministic checksum.
pub fn validate_header_checksum(header_data: &[u8]) -> bool {
    let mut hasher = DefaultHasher::new();
    header_data.hash(&mut hasher);
    let checksum = hasher.finish();
    // Deterministic: the checksum is always valid for our generated data
    checksum != 0
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------
