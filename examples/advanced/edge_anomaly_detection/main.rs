#![allow(unused_imports)]
//! # Demo L: Edge Anomaly Detection
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Micro-autoencoder for sensor anomaly detection on resource-constrained edge devices.
//! Fixed-point arithmetic (Q8.8), <1KB model, <100us inference.
//!
//!
//! ## Format Variants
//! ```bash
//! apr run model.apr          # APR native format
//! apr run model.gguf         # GGUF (llama.cpp compatible)
//! apr run model.safetensors  # SafeTensors (HuggingFace)
//! ```
//! ## References
//! - Sculley, D. et al. (2015). *Hidden Technical Debt in Machine Learning Systems*. NeurIPS. arXiv:1503.05991

use std::fmt;
use std::mem::size_of;

mod types;
#[allow(unused_imports)]
#[allow(clippy::wildcard_imports)]
use types::*;

fn main() {
    println!("=== Demo L: Edge Anomaly Detection ===\n");
    let weights = MicroAutoencoderWeights::new_xavier(42);
    println!(
        "Model size: {} bytes ({:.1} KB)",
        weights.size_bytes(),
        weights.size_bytes() as f32 / 1024.0
    );
    let model = MicroAutoencoder::new(weights).with_threshold(0.12);
    let mut detector = AdaptiveAnomalyDetector::new(model);
    let mut sim = SensorSimulator::new(12345);
    println!("\n--- Processing 100 normal + 20 anomalous readings ---");
    for _ in 0..100 {
        detector.process(&sim.generate_normal());
    }
    println!(
        "Normal anomaly rate: {:.2}%",
        detector.anomaly_rate() * 100.0
    );
    let pre = detector.counts().0;
    for _ in 0..20 {
        detector.process(&sim.generate_anomaly(0.5));
    }
    println!(
        "Detection rate: {:.1}%",
        (detector.counts().0 - pre) as f32 / 20.0 * 100.0
    );
    println!("Adaptive threshold: {:.4}", detector.adaptive_threshold());
    println!("\n=== Demo L Complete ===");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fixed_point_roundtrip() {
        let fp = FixedPoint::from_f32(0.5);
        assert_eq!(fp.raw(), 128);
        assert!((fp.to_f32() - 0.5).abs() < 0.01);
    }
    #[test]
    fn test_fixed_point_ops() {
        assert!(
            (FixedPoint::from_f32(0.5)
                .fixed_mul(FixedPoint::from_f32(0.5))
                .to_f32()
                - 0.25)
                .abs()
                < 0.02
        );
        assert!(
            (FixedPoint::from_f32(0.3)
                .fixed_add(FixedPoint::from_f32(0.2))
                .to_f32()
                - 0.5)
                .abs()
                < 0.02
        );
        assert!(FixedPoint::from_f32(0.5).relu().to_f32() > 0.0);
        assert_eq!(FixedPoint::from_f32(-0.5).relu().to_f32(), 0.0);
    }
    #[test]
    fn test_fixed_point_overflow() {
        assert!(FixedPoint::from_f32(1000.0).raw() <= i16::MAX);
        assert!(FixedPoint::from_f32(-1000.0).raw() >= i16::MIN);
    }
    #[test]
    fn test_sensor_reading() {
        let r = SensorReading::new([0.5; NUM_SENSORS], 0);
        assert!(r.is_valid());
        assert!(!SensorReading::new([1.5; NUM_SENSORS], 0).is_valid());
        let mut nan_vals = [0.5; NUM_SENSORS];
        nan_vals[0] = f32::NAN;
        assert!(!SensorReading::new(nan_vals, 0).is_valid());
        assert!(r
            .to_fixed_point()
            .iter()
            .all(|fp| (fp.to_f32() - 0.5).abs() < 0.01));
    }
    #[test]
    fn test_weights_serialization() {
        let w = MicroAutoencoderWeights::new_xavier(42);
        assert!(w.size_bytes() < 1024);
        let bytes = w.to_bytes();
        assert_eq!(bytes.len(), MicroAutoencoderWeights::expected_size());
        let restored = MicroAutoencoderWeights::from_bytes(&bytes).expect("deser");
        assert_eq!(w.encoder_w1[0][0].raw(), restored.encoder_w1[0][0].raw());
        assert!(MicroAutoencoderWeights::from_bytes(&[0u8; 10]).is_err());
    }
    #[test]
    fn test_autoencoder() {
        let w = MicroAutoencoderWeights::new_xavier(42);
        let mut ae = MicroAutoencoder::new(w).with_threshold(0.5);
        assert!((ae.threshold() - 0.5).abs() < 0.001);
        let out = ae.forward(&[FixedPoint::from_f32(0.5); NUM_SENSORS]);
        assert!(out.iter().all(|&v| (0.0..=1.0).contains(&v)));
        assert_eq!(ae.get_latent().len(), LATENT_DIM);
        assert!(ae.memory_footprint() < 2048);
        let result = ae.detect(&SensorReading::new([0.5; NUM_SENSORS], 0));
        assert!(result.reconstruction_error >= 0.0);
    }
    #[test]
    fn test_rolling_stats() {
        let mut stats = RollingStats::new(0.1);
        for _ in 0..100 {
            stats.update(&SensorReading::new([0.6; NUM_SENSORS], 0));
        }
        assert!(stats.mean.iter().all(|&m| (m - 0.6).abs() < 0.1));
        assert!(stats.is_drift(&SensorReading::new([0.9; NUM_SENSORS], 0), 2.0));
        let z = stats.z_scores(&SensorReading::new([0.6; NUM_SENSORS], 0));
        assert!(z.iter().all(|&z| z.abs() < 1.0));
    }
    #[test]
    fn test_adaptive_detector() {
        let w = MicroAutoencoderWeights::new_xavier(42);
        let mut det = AdaptiveAnomalyDetector::new(MicroAutoencoder::new(w));
        assert_eq!(det.counts(), (0, 0));
        det.process(&SensorReading::new([0.5; NUM_SENSORS], 0));
        assert_eq!(det.counts().1, 1);
        assert!(
            det.process(&SensorReading::new([f32::NAN; NUM_SENSORS], 0))
                .is_anomaly
        );
    }
    #[test]
    fn test_simulator() {
        let mut sim = SensorSimulator::new(42);
        let r = sim.generate_normal();
        assert!(r.is_valid());
        assert!(sim.generate_anomaly(0.5).is_valid());
        assert!(sim.generate_drift(0.1).is_valid());
        let mut s2 = SensorSimulator::new(42);
        assert_eq!(SensorSimulator::new(42).generate_normal().values, {
            let _ = s2;
            SensorSimulator::new(42).generate_normal().values
        });
    }
    #[test]
    fn test_anomaly_result() {
        let n = AnomalyResult::normal(0.05, [0.1, 0.2]);
        assert!(!n.is_anomaly && n.anomalous_sensors.is_empty());
        let a = AnomalyResult::anomaly(0.5, [0.1, 0.2], vec![0, 2]);
        assert!(a.is_anomaly && a.anomalous_sensors.len() == 2);
    }
    #[test]
    fn test_edge_error_display() {
        assert!(EdgeError::InvalidModelSize {
            expected: 100,
            got: 50
        }
        .to_string()
        .contains("100"));
    }
}

#[cfg(test)]
mod proptests {
    use super::*;
    use proptest::prelude::*;
    proptest! {
        #![proptest_config(ProptestConfig::with_cases(50))]
        #[test] fn prop_fixed_point_roundtrip(v in -10.0f32..10.0) { prop_assert!((FixedPoint::from_f32(v).to_f32() - v).abs() < 0.1); }
        #[test] fn prop_fixed_point_relu_non_negative(v in -10.0f32..10.0) { prop_assert!(FixedPoint::from_f32(v).relu().to_f32() >= 0.0); }
        #[test] fn prop_autoencoder_output_bounded(seed in 0u64..1000) {
            let mut ae = MicroAutoencoder::new(MicroAutoencoderWeights::new_xavier(seed));
            for &v in &ae.forward(&[FixedPoint::from_f32(0.5); NUM_SENSORS]) { prop_assert!(v >= 0.0 && v <= 1.0); }
        }
        #[test] fn prop_weights_serialization_roundtrip(seed in 0u64..1000) {
            prop_assert!(MicroAutoencoderWeights::from_bytes(&MicroAutoencoderWeights::new_xavier(seed).to_bytes()).is_ok());
        }
        #[test] fn prop_detector_counts_consistent(n in 1usize..30) {
            let mut det = AdaptiveAnomalyDetector::new(MicroAutoencoder::new(MicroAutoencoderWeights::new_xavier(42)));
            for _ in 0..n { det.process(&SensorReading::new([0.5; NUM_SENSORS], 0)); }
            let (a, t) = det.counts(); prop_assert!(a <= t && t == n as u64);
        }
    }
}
