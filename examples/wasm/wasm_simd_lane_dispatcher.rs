//! # WASM SIMD Lane-Count Dispatcher
//!
//! WASM SIMD128 = single 128-bit register. Pick lane shape based on
//! element type:
//!   i8x16: 16 lanes (byte-wise pixel ops)
//!   i16x8: 8 lanes (audio sample ops)
//!   i32x4 / f32x4: 4 lanes (general arith)
//!   i64x2 / f64x2: 2 lanes (high-precision math)
//!
//! Plus tail handling: if length not divisible by lane count, fall back
//! to scalar for the tail.
//!
//! Demonstrates the **WASM.20** recipe for PMAT-146 (wasm round 4).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: WebAssembly fixed-width SIMD specification.
//!
//! Run with: cargo run --example wasm_simd_lane_dispatcher
//!
//! Added by PMAT-146 (expand-cookbooks followup).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ElementType {
    I8,
    I16,
    I32,
    I64,
    F32,
    F64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LaneShape {
    I8x16,
    I16x8,
    I32x4,
    I64x2,
    F32x4,
    F64x2,
}

#[derive(Debug, PartialEq)]
pub enum DispatchVerdict {
    Ok {
        shape: LaneShape,
        full_vectors: u32,
        scalar_tail: u32,
    },
    InvalidLength,
}

pub fn pick(element: ElementType, total_elements: u32) -> DispatchVerdict {
    if total_elements == 0 {
        return DispatchVerdict::InvalidLength;
    }
    let (shape, lanes) = match element {
        ElementType::I8 => (LaneShape::I8x16, 16),
        ElementType::I16 => (LaneShape::I16x8, 8),
        ElementType::I32 => (LaneShape::I32x4, 4),
        ElementType::I64 => (LaneShape::I64x2, 2),
        ElementType::F32 => (LaneShape::F32x4, 4),
        ElementType::F64 => (LaneShape::F64x2, 2),
    };
    let full_vectors = total_elements / lanes;
    let scalar_tail = total_elements % lanes;
    DispatchVerdict::Ok {
        shape,
        full_vectors,
        scalar_tail,
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("wasm_simd_lane_dispatcher")?;

    println!("F32, 100 elem: {:?}", pick(ElementType::F32, 100));
    println!("I8, 1024 elem: {:?}", pick(ElementType::I8, 1024));
    println!("F64, 10 elem: {:?}", pick(ElementType::F64, 10));
    println!("invalid: {:?}", pick(ElementType::F32, 0));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dispatcher_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn i8_picks_x16() {
        let v = pick(ElementType::I8, 32);
        if let DispatchVerdict::Ok { shape, .. } = v {
            assert_eq!(shape, LaneShape::I8x16);
        }
    }

    #[test]
    fn i16_picks_x8() {
        let v = pick(ElementType::I16, 32);
        if let DispatchVerdict::Ok { shape, .. } = v {
            assert_eq!(shape, LaneShape::I16x8);
        }
    }

    #[test]
    fn i32_picks_x4() {
        let v = pick(ElementType::I32, 32);
        if let DispatchVerdict::Ok { shape, .. } = v {
            assert_eq!(shape, LaneShape::I32x4);
        }
    }

    #[test]
    fn f32_picks_x4() {
        let v = pick(ElementType::F32, 32);
        if let DispatchVerdict::Ok { shape, .. } = v {
            assert_eq!(shape, LaneShape::F32x4);
        }
    }

    #[test]
    fn f64_picks_x2() {
        let v = pick(ElementType::F64, 32);
        if let DispatchVerdict::Ok { shape, .. } = v {
            assert_eq!(shape, LaneShape::F64x2);
        }
    }

    #[test]
    fn full_vectors_correct_for_f32_100() {
        // 100 / 4 = 25 vectors, 0 tail.
        let v = pick(ElementType::F32, 100);
        if let DispatchVerdict::Ok {
            full_vectors,
            scalar_tail,
            ..
        } = v
        {
            assert_eq!(full_vectors, 25);
            assert_eq!(scalar_tail, 0);
        }
    }

    #[test]
    fn scalar_tail_for_uneven_length() {
        // 10 / 4 = 2 vectors, tail 2.
        let v = pick(ElementType::F32, 10);
        if let DispatchVerdict::Ok {
            full_vectors,
            scalar_tail,
            ..
        } = v
        {
            assert_eq!(full_vectors, 2);
            assert_eq!(scalar_tail, 2);
        }
    }

    #[test]
    fn invalid_zero_elements() {
        assert_eq!(pick(ElementType::F32, 0), DispatchVerdict::InvalidLength);
    }

    #[test]
    fn i64_two_lanes_correct() {
        // 7 / 2 = 3 vectors, tail 1.
        let v = pick(ElementType::I64, 7);
        if let DispatchVerdict::Ok {
            full_vectors,
            scalar_tail,
            ..
        } = v
        {
            assert_eq!(full_vectors, 3);
            assert_eq!(scalar_tail, 1);
        }
    }

    #[test]
    fn i8_sixteen_lanes_correct() {
        // 1024 / 16 = 64 vectors, 0 tail.
        let v = pick(ElementType::I8, 1024);
        if let DispatchVerdict::Ok {
            full_vectors,
            scalar_tail,
            ..
        } = v
        {
            assert_eq!(full_vectors, 64);
            assert_eq!(scalar_tail, 0);
        }
    }
}
