//! # apr trace --save-tensor — Layer-0 Oracle Bisection
//!
//! `apr trace --save-tensor <stages>` captures per-stage forward-pass
//! tensor dumps in APRT (apr-trace) byte format for element-wise GPU/CPU
//! bisection. The file format is a 16-byte header (magic + dtype + rank +
//! padding) followed by little-endian dimensions then the raw tensor
//! bytes. This recipe demonstrates the writer + reader for the APRT
//! container so external tools can produce dumps `apr diff --values` will
//! recognize.
//!
//! Demonstrates the **CLI+.1** recipe per
//! `docs/specifications/expand-cookbooks/recipe-catalog.md`.
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: apr-cli-trace-save-tensor-v1.yaml v1.4.0 FUNCTIONAL (FALSIFY-009/010/011)
//!
//! Run with: cargo run --example cli_trace_save_tensor_layer0
//!
//! Added by PMAT-075 (expand-cookbooks: GPU/CPU oracle bisection).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;
use std::io::{Read, Write};

const APRT_MAGIC: &[u8; 4] = b"APRT";

/// Minimal APRT-style tensor container. The real format includes a hash and
/// a stage-name field; this is the load-bearing subset used for GPU/CPU
/// element-wise diff.
fn write_aprt_f32(writer: &mut impl Write, dims: &[u32], data: &[f32]) -> std::io::Result<()> {
    let n_expected: usize = dims.iter().map(|d| *d as usize).product();
    assert_eq!(
        n_expected,
        data.len(),
        "data length must match product of dims"
    );
    writer.write_all(APRT_MAGIC)?;
    writer.write_all(&[1u8])?; // dtype: 1 = f32
    writer.write_all(&[dims.len() as u8])?;
    writer.write_all(&[0u8; 2])?; // padding to 8-byte boundary
    for d in dims {
        writer.write_all(&d.to_le_bytes())?;
    }
    for x in data {
        writer.write_all(&x.to_le_bytes())?;
    }
    Ok(())
}

fn read_aprt_f32(reader: &mut impl Read) -> std::io::Result<(Vec<u32>, Vec<f32>)> {
    let mut magic = [0u8; 4];
    reader.read_exact(&mut magic)?;
    assert_eq!(&magic, APRT_MAGIC, "magic bytes mismatch");
    let mut header = [0u8; 4];
    reader.read_exact(&mut header)?;
    let dtype = header[0];
    let rank = header[1] as usize;
    assert_eq!(dtype, 1, "this reader only handles f32 (dtype=1)");
    let mut dims = Vec::with_capacity(rank);
    for _ in 0..rank {
        let mut d = [0u8; 4];
        reader.read_exact(&mut d)?;
        dims.push(u32::from_le_bytes(d));
    }
    let n: usize = dims.iter().map(|d| *d as usize).product();
    let mut data = Vec::with_capacity(n);
    for _ in 0..n {
        let mut x = [0u8; 4];
        reader.read_exact(&mut x)?;
        data.push(f32::from_le_bytes(x));
    }
    Ok((dims, data))
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("cli_trace_save_tensor_layer0")?;
    let dir = tempfile::tempdir()?;
    let path = dir.path().join("layer0_attn_norm.aprt");

    // Synthetic layer-0 attn_norm tensor: shape [1, 4, 8] (batch, seq, dim)
    let dims = [1u32, 4, 8];
    let data: Vec<f32> = (0..32).map(|i| (i as f32) * 0.1).collect();

    let mut file = std::fs::File::create(&path)?;
    write_aprt_f32(&mut file, &dims, &data)?;
    drop(file);

    let mut reader = std::fs::File::open(&path)?;
    let (loaded_dims, loaded_data) = read_aprt_f32(&mut reader)?;

    println!("wrote/read APRT tensor at layer-0 attn_norm stage:");
    println!("  shape: {:?}", loaded_dims);
    println!("  bytes on disk: {}", std::fs::metadata(&path)?.len());
    println!("  first 4 elems: {:?}", &loaded_data[..4]);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn save_tensor_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn write_then_read_roundtrip_preserves_dims_and_data() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("rt.aprt");
        let dims = [2u32, 3];
        let data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];

        let mut w = std::fs::File::create(&path).unwrap();
        write_aprt_f32(&mut w, &dims, &data).unwrap();
        drop(w);

        let mut r = std::fs::File::open(&path).unwrap();
        let (loaded_dims, loaded_data) = read_aprt_f32(&mut r).unwrap();
        assert_eq!(loaded_dims, dims);
        assert_eq!(loaded_data, data);
    }

    #[test]
    fn header_starts_with_aprt_magic() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("magic.aprt");
        let mut w = std::fs::File::create(&path).unwrap();
        write_aprt_f32(&mut w, &[1u32], &[42.0]).unwrap();
        drop(w);
        let bytes = std::fs::read(&path).unwrap();
        assert_eq!(&bytes[..4], APRT_MAGIC);
    }
}
