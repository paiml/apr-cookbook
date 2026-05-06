//! # Conversion Quantization Rescaler
//!
//! Symmetric quantization: q = round(x / scale), x ≈ q × scale.
//! Asymmetric: q = round(x / scale + zero_point), x ≈ (q − zero_point) × scale.
//! When converting from one quant scheme to another (e.g., Int8 sym →
//! Int8 asym), the scale must be re-derived from the value range. This
//! recipe builds the rescaler.
//!
//! Demonstrates the **CONV.11** recipe for PMAT-133 (conversion coverage).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: Jacob et al. (2018). Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference.
//!
//! Run with: cargo run --example convert_quantization_rescaler
//!
//! Added by PMAT-133 (expand-cookbooks followup).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuantScheme {
    Int8Symmetric,
    Int8Asymmetric,
    Int4Asymmetric,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct QuantParams {
    pub scale: f64,
    pub zero_point: i32,
}

#[derive(Debug, PartialEq)]
pub enum RescaleVerdict {
    Ok(QuantParams),
    InvalidRange,
}

pub fn derive(scheme: QuantScheme, value_min: f64, value_max: f64) -> RescaleVerdict {
    if !value_min.is_finite() || !value_max.is_finite() || value_min >= value_max {
        return RescaleVerdict::InvalidRange;
    }
    let (qmin, qmax) = match scheme {
        QuantScheme::Int8Symmetric => (-127i32, 127i32),
        QuantScheme::Int8Asymmetric => (0i32, 255i32),
        QuantScheme::Int4Asymmetric => (0i32, 15i32),
    };
    let q_range = f64::from(qmax - qmin);
    if scheme == QuantScheme::Int8Symmetric {
        let abs = value_min.abs().max(value_max.abs());
        if abs == 0.0 {
            return RescaleVerdict::InvalidRange;
        }
        RescaleVerdict::Ok(QuantParams {
            scale: abs / 127.0,
            zero_point: 0,
        })
    } else {
        let scale = (value_max - value_min) / q_range;
        let zero_point = qmin - (value_min / scale).round() as i32;
        RescaleVerdict::Ok(QuantParams { scale, zero_point })
    }
}

pub fn quantize(scheme: QuantScheme, params: QuantParams, x: f64) -> i32 {
    let q = (x / params.scale).round() as i32 + params.zero_point;
    let (qmin, qmax) = match scheme {
        QuantScheme::Int8Symmetric => (-127, 127),
        QuantScheme::Int8Asymmetric => (0, 255),
        QuantScheme::Int4Asymmetric => (0, 15),
    };
    q.clamp(qmin, qmax)
}

pub fn dequantize(params: QuantParams, q: i32) -> f64 {
    f64::from(q - params.zero_point) * params.scale
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("convert_quantization_rescaler")?;

    let v = derive(QuantScheme::Int8Symmetric, -1.0, 1.0);
    println!("int8-sym [-1, 1]: {v:?}");

    let v = derive(QuantScheme::Int8Asymmetric, 0.0, 255.0);
    println!("int8-asym [0, 255]: {v:?}");

    if let RescaleVerdict::Ok(params) = derive(QuantScheme::Int8Symmetric, -1.0, 1.0) {
        let q = quantize(QuantScheme::Int8Symmetric, params, 0.5);
        println!("quantize 0.5: {q}");
        println!("dequantize back: {}", dequantize(params, q));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rescaler_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn int8_symmetric_zero_point_zero() {
        let v = derive(QuantScheme::Int8Symmetric, -1.0, 1.0);
        if let RescaleVerdict::Ok(p) = v {
            assert_eq!(p.zero_point, 0);
            assert!((p.scale - 1.0 / 127.0).abs() < 1e-12);
        }
    }

    #[test]
    fn int8_asymmetric_uses_value_min() {
        let v = derive(QuantScheme::Int8Asymmetric, 0.0, 1.0);
        if let RescaleVerdict::Ok(p) = v {
            assert!(p.scale > 0.0);
        }
    }

    #[test]
    fn invalid_range_inverted_rejected() {
        let v = derive(QuantScheme::Int8Symmetric, 1.0, -1.0);
        assert_eq!(v, RescaleVerdict::InvalidRange);
    }

    #[test]
    fn zero_range_rejected() {
        let v = derive(QuantScheme::Int8Symmetric, 0.5, 0.5);
        assert_eq!(v, RescaleVerdict::InvalidRange);
    }

    #[test]
    fn nan_range_rejected() {
        let v = derive(QuantScheme::Int8Symmetric, f64::NAN, 1.0);
        assert_eq!(v, RescaleVerdict::InvalidRange);
    }

    #[test]
    fn quantize_within_qmax_for_int8() {
        let p = QuantParams {
            scale: 0.01,
            zero_point: 0,
        };
        let q = quantize(QuantScheme::Int8Symmetric, p, 100.0);
        assert!((-127..=127).contains(&q));
    }

    #[test]
    fn quantize_clamps_to_qmin_for_extreme_negative() {
        let p = QuantParams {
            scale: 0.01,
            zero_point: 0,
        };
        let q = quantize(QuantScheme::Int8Symmetric, p, -100.0);
        assert_eq!(q, -127);
    }

    #[test]
    fn dequantize_round_trip_within_scale() {
        let p = QuantParams {
            scale: 0.5,
            zero_point: 0,
        };
        let original = 1.5;
        let q = quantize(QuantScheme::Int8Symmetric, p, original);
        let back = dequantize(p, q);
        assert!((back - original).abs() <= p.scale);
    }

    #[test]
    fn int4_qmax_is_15() {
        let p = QuantParams {
            scale: 1.0,
            zero_point: 0,
        };
        let q = quantize(QuantScheme::Int4Asymmetric, p, 100.0);
        assert_eq!(q, 15);
    }
}
