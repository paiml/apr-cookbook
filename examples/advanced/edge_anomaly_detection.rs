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

pub const NUM_SENSORS: usize = 8;
pub const LATENT_DIM: usize = 2;
pub const HIDDEN_DIM: usize = 4;
pub const FIXED_POINT_SCALE: i32 = 256;
pub const DEFAULT_THRESHOLD: f32 = 0.15;
pub const MAX_HISTORY_SIZE: usize = 100;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct FixedPoint(i16);

impl FixedPoint {
    #[must_use]
    pub fn from_f32(v: f32) -> Self {
        Self((v * FIXED_POINT_SCALE as f32).clamp(-32768.0, 32767.0) as i16)
    }
    #[must_use]
    pub fn to_f32(self) -> f32 {
        f32::from(self.0) / FIXED_POINT_SCALE as f32
    }
    #[must_use]
    pub fn fixed_mul(self, o: Self) -> Self {
        Self(((i32::from(self.0) * i32::from(o.0)) / FIXED_POINT_SCALE).clamp(-32768, 32767) as i16)
    }
    #[must_use]
    pub fn fixed_add(self, o: Self) -> Self {
        Self(self.0.saturating_add(o.0))
    }
    #[must_use]
    pub fn relu(self) -> Self {
        Self(self.0.max(0))
    }
    #[must_use]
    pub fn raw(self) -> i16 {
        self.0
    }
}

#[derive(Debug, Clone)]
pub struct SensorReading {
    pub values: [f32; NUM_SENSORS],
    pub timestamp_ms: u64,
    pub sensor_id: u8,
}

impl SensorReading {
    #[must_use]
    pub fn new(values: [f32; NUM_SENSORS], timestamp_ms: u64) -> Self {
        Self {
            values,
            timestamp_ms,
            sensor_id: 0,
        }
    }
    #[must_use]
    pub fn with_sensor_id(mut self, id: u8) -> Self {
        self.sensor_id = id;
        self
    }
    #[must_use]
    pub fn is_valid(&self) -> bool {
        self.values
            .iter()
            .all(|&v| (0.0..=1.0).contains(&v) && v.is_finite())
    }
    #[must_use]
    pub fn to_fixed_point(&self) -> [FixedPoint; NUM_SENSORS] {
        let mut r = [FixedPoint::default(); NUM_SENSORS];
        for (i, &v) in self.values.iter().enumerate() {
            r[i] = FixedPoint::from_f32(v);
        }
        r
    }
}

#[derive(Debug, Clone)]
pub struct AnomalyResult {
    pub reconstruction_error: f32,
    pub is_anomaly: bool,
    pub confidence: f32,
    pub anomalous_sensors: Vec<usize>,
    pub latent_code: [f32; LATENT_DIM],
}

impl AnomalyResult {
    #[must_use]
    pub fn normal(e: f32, l: [f32; LATENT_DIM]) -> Self {
        Self {
            reconstruction_error: e,
            is_anomaly: false,
            confidence: 1.0 - (e * 5.0).min(1.0),
            anomalous_sensors: Vec::new(),
            latent_code: l,
        }
    }
    #[must_use]
    pub fn anomaly(e: f32, l: [f32; LATENT_DIM], s: Vec<usize>) -> Self {
        Self {
            reconstruction_error: e,
            is_anomaly: true,
            confidence: (e * 5.0).min(1.0),
            anomalous_sensors: s,
            latent_code: l,
        }
    }
}

#[derive(Debug, Clone)]
pub struct MicroAutoencoderWeights {
    pub encoder_w1: [[FixedPoint; NUM_SENSORS]; HIDDEN_DIM],
    pub encoder_b1: [FixedPoint; HIDDEN_DIM],
    pub encoder_w2: [[FixedPoint; HIDDEN_DIM]; LATENT_DIM],
    pub encoder_b2: [FixedPoint; LATENT_DIM],
    pub decoder_w1: [[FixedPoint; LATENT_DIM]; HIDDEN_DIM],
    pub decoder_b1: [FixedPoint; HIDDEN_DIM],
    pub decoder_w2: [[FixedPoint; HIDDEN_DIM]; NUM_SENSORS],
    pub decoder_b2: [FixedPoint; NUM_SENSORS],
}

impl MicroAutoencoderWeights {
    fn init_layer<const R: usize, const C: usize>(
        rng: &mut SimpleRng,
        scale: f32,
    ) -> [[FixedPoint; C]; R] {
        let mut w = [[FixedPoint::default(); C]; R];
        for row in &mut w {
            for v in row {
                *v = FixedPoint::from_f32(rng.next_gaussian() * scale);
            }
        }
        w
    }
    #[must_use]
    pub fn new_xavier(seed: u64) -> Self {
        let mut rng = SimpleRng::new(seed);
        Self {
            encoder_w1: Self::init_layer(
                &mut rng,
                (2.0 / (NUM_SENSORS + HIDDEN_DIM) as f32).sqrt(),
            ),
            encoder_b1: [FixedPoint::default(); HIDDEN_DIM],
            encoder_w2: Self::init_layer(&mut rng, (2.0 / (HIDDEN_DIM + LATENT_DIM) as f32).sqrt()),
            encoder_b2: [FixedPoint::default(); LATENT_DIM],
            decoder_w1: Self::init_layer(&mut rng, (2.0 / (LATENT_DIM + HIDDEN_DIM) as f32).sqrt()),
            decoder_b1: [FixedPoint::default(); HIDDEN_DIM],
            decoder_w2: Self::init_layer(
                &mut rng,
                (2.0 / (HIDDEN_DIM + NUM_SENSORS) as f32).sqrt(),
            ),
            decoder_b2: [FixedPoint::default(); NUM_SENSORS],
        }
    }
    #[must_use]
    pub fn size_bytes(&self) -> usize {
        Self::expected_size()
    }
    #[must_use]
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut b = Vec::with_capacity(self.size_bytes());
        macro_rules! ser {
            ($arr:expr) => {
                for row in &$arr {
                    for v in row {
                        b.extend_from_slice(&v.raw().to_le_bytes());
                    }
                }
            };
            (bias $arr:expr) => {
                for v in &$arr {
                    b.extend_from_slice(&v.raw().to_le_bytes());
                }
            };
        }
        ser!(self.encoder_w1);
        ser!(bias self.encoder_b1);
        ser!(self.encoder_w2);
        ser!(bias self.encoder_b2);
        ser!(self.decoder_w1);
        ser!(bias self.decoder_b1);
        ser!(self.decoder_w2);
        ser!(bias self.decoder_b2);
        b
    }
    pub fn from_bytes(bytes: &[u8]) -> Result<Self> {
        if bytes.len() != Self::expected_size() {
            return Err(EdgeError::InvalidModelSize {
                expected: Self::expected_size(),
                got: bytes.len(),
            });
        }
        let mut off = 0;
        let mut read_i16 = || {
            let v = i16::from_le_bytes([bytes[off], bytes[off + 1]]);
            off += 2;
            FixedPoint(v)
        };
        macro_rules! deser {
            ($R:expr, $C:expr) => {{
                let mut a = [[FixedPoint::default(); $C]; $R];
                for row in &mut a {
                    for v in row {
                        *v = read_i16();
                    }
                }
                a
            }};
            (bias $N:expr) => {{
                let mut a = [FixedPoint::default(); $N];
                for v in &mut a {
                    *v = read_i16();
                }
                a
            }};
        }
        Ok(Self {
            encoder_w1: deser!(HIDDEN_DIM, NUM_SENSORS),
            encoder_b1: deser!(bias HIDDEN_DIM),
            encoder_w2: deser!(LATENT_DIM, HIDDEN_DIM),
            encoder_b2: deser!(bias LATENT_DIM),
            decoder_w1: deser!(HIDDEN_DIM, LATENT_DIM),
            decoder_b1: deser!(bias HIDDEN_DIM),
            decoder_w2: deser!(NUM_SENSORS, HIDDEN_DIM),
            decoder_b2: deser!(bias NUM_SENSORS),
        })
    }
    #[must_use]
    pub fn expected_size() -> usize {
        2 * (HIDDEN_DIM * NUM_SENSORS
            + HIDDEN_DIM
            + LATENT_DIM * HIDDEN_DIM
            + LATENT_DIM
            + HIDDEN_DIM * LATENT_DIM
            + HIDDEN_DIM
            + NUM_SENSORS * HIDDEN_DIM
            + NUM_SENSORS)
    }
}

pub struct MicroAutoencoder {
    weights: MicroAutoencoderWeights,
    threshold: f32,
    hidden1: [FixedPoint; HIDDEN_DIM],
    latent: [FixedPoint; LATENT_DIM],
    hidden2: [FixedPoint; HIDDEN_DIM],
    output: [FixedPoint; NUM_SENSORS],
}

impl MicroAutoencoder {
    #[must_use]
    pub fn new(weights: MicroAutoencoderWeights) -> Self {
        Self {
            weights,
            threshold: DEFAULT_THRESHOLD,
            hidden1: [FixedPoint::default(); HIDDEN_DIM],
            latent: [FixedPoint::default(); LATENT_DIM],
            hidden2: [FixedPoint::default(); HIDDEN_DIM],
            output: [FixedPoint::default(); NUM_SENSORS],
        }
    }
    #[must_use]
    pub fn with_threshold(mut self, t: f32) -> Self {
        self.threshold = t;
        self
    }
    #[must_use]
    pub fn threshold(&self) -> f32 {
        self.threshold
    }

    fn encode(&mut self, input: &[FixedPoint; NUM_SENSORS]) {
        for (h, row) in self.hidden1.iter_mut().zip(self.weights.encoder_w1.iter()) {
            let mut s = FixedPoint::default();
            for (w, &x) in row.iter().zip(input.iter()) {
                s = s.fixed_add(w.fixed_mul(x));
            }
            *h = s;
        }
        for (h, &b) in self.hidden1.iter_mut().zip(self.weights.encoder_b1.iter()) {
            *h = h.fixed_add(b).relu();
        }
        for (l, row) in self.latent.iter_mut().zip(self.weights.encoder_w2.iter()) {
            let mut s = FixedPoint::default();
            for (w, &h) in row.iter().zip(self.hidden1.iter()) {
                s = s.fixed_add(w.fixed_mul(h));
            }
            *l = s;
        }
        for (l, &b) in self.latent.iter_mut().zip(self.weights.encoder_b2.iter()) {
            *l = l.fixed_add(b);
        }
    }

    fn decode(&mut self) {
        for (h, row) in self.hidden2.iter_mut().zip(self.weights.decoder_w1.iter()) {
            let mut s = FixedPoint::default();
            for (w, &l) in row.iter().zip(self.latent.iter()) {
                s = s.fixed_add(w.fixed_mul(l));
            }
            *h = s;
        }
        for (h, &b) in self.hidden2.iter_mut().zip(self.weights.decoder_b1.iter()) {
            *h = h.fixed_add(b).relu();
        }
        for (o, row) in self.output.iter_mut().zip(self.weights.decoder_w2.iter()) {
            let mut s = FixedPoint::default();
            for (w, &h) in row.iter().zip(self.hidden2.iter()) {
                s = s.fixed_add(w.fixed_mul(h));
            }
            *o = s;
        }
        for (o, &b) in self.output.iter_mut().zip(self.weights.decoder_b2.iter()) {
            *o = FixedPoint::from_f32(o.fixed_add(b).to_f32().clamp(0.0, 1.0));
        }
    }

    pub fn forward(&mut self, input: &[FixedPoint; NUM_SENSORS]) -> [f32; NUM_SENSORS] {
        self.encode(input);
        self.decode();
        let mut r = [0.0_f32; NUM_SENSORS];
        for (r, &o) in r.iter_mut().zip(self.output.iter()) {
            *r = o.to_f32();
        }
        r
    }
    #[must_use]
    pub fn get_latent(&self) -> [f32; LATENT_DIM] {
        let mut r = [0.0_f32; LATENT_DIM];
        for (r, &l) in r.iter_mut().zip(self.latent.iter()) {
            *r = l.to_f32();
        }
        r
    }
    pub fn detect(&mut self, reading: &SensorReading) -> AnomalyResult {
        let input = reading.to_fixed_point();
        let recon = self.forward(&input);
        let latent = self.get_latent();
        let mut mse = 0.0_f32;
        let mut bad = Vec::new();
        for (i, (&orig, &rec)) in reading.values.iter().zip(recon.iter()).enumerate() {
            let e = (orig - rec).powi(2);
            mse += e;
            if e > self.threshold / NUM_SENSORS as f32 {
                bad.push(i);
            }
        }
        mse /= NUM_SENSORS as f32;
        if mse > self.threshold {
            AnomalyResult::anomaly(mse, latent, bad)
        } else {
            AnomalyResult::normal(mse, latent)
        }
    }
    #[must_use]
    pub fn memory_footprint(&self) -> usize {
        self.weights.size_bytes()
            + size_of::<[FixedPoint; HIDDEN_DIM]>() * 2
            + size_of::<[FixedPoint; LATENT_DIM]>()
            + size_of::<[FixedPoint; NUM_SENSORS]>()
    }
}

impl fmt::Debug for MicroAutoencoder {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("MicroAutoencoder")
            .field("threshold", &self.threshold)
            .finish_non_exhaustive()
    }
}

#[derive(Debug, Clone)]
pub struct RollingStats {
    pub mean: [f32; NUM_SENSORS],
    pub variance: [f32; NUM_SENSORS],
    pub count: usize,
    pub alpha: f32,
}

impl RollingStats {
    #[must_use]
    pub fn new(alpha: f32) -> Self {
        Self {
            mean: [0.5; NUM_SENSORS],
            variance: [0.1; NUM_SENSORS],
            count: 0,
            alpha: alpha.clamp(0.001, 0.5),
        }
    }
    pub fn update(&mut self, r: &SensorReading) {
        self.count += 1;
        for (i, &v) in r.values.iter().enumerate() {
            let d = v - self.mean[i];
            self.mean[i] += self.alpha * d;
            self.variance[i] = (1.0 - self.alpha) * self.variance[i] + self.alpha * d * d;
        }
    }
    #[must_use]
    pub fn is_drift(&self, r: &SensorReading, z_thresh: f32) -> bool {
        r.values
            .iter()
            .enumerate()
            .any(|(i, &v)| (v - self.mean[i]).abs() / self.variance[i].sqrt().max(0.001) > z_thresh)
    }
    #[must_use]
    pub fn z_scores(&self, r: &SensorReading) -> [f32; NUM_SENSORS] {
        let mut s = [0.0_f32; NUM_SENSORS];
        for (i, &v) in r.values.iter().enumerate() {
            s[i] = (v - self.mean[i]) / self.variance[i].sqrt().max(0.001);
        }
        s
    }
}

pub struct AdaptiveAnomalyDetector {
    model: MicroAutoencoder,
    stats: RollingStats,
    score_history: Vec<f32>,
    anomaly_count: u64,
    total_count: u64,
}

impl AdaptiveAnomalyDetector {
    #[must_use]
    pub fn new(model: MicroAutoencoder) -> Self {
        Self {
            model,
            stats: RollingStats::new(0.01),
            score_history: Vec::with_capacity(MAX_HISTORY_SIZE),
            anomaly_count: 0,
            total_count: 0,
        }
    }
    pub fn process(&mut self, reading: &SensorReading) -> AnomalyResult {
        self.total_count += 1;
        if !reading.is_valid() {
            return AnomalyResult::anomaly(1.0, [0.0; LATENT_DIM], (0..NUM_SENSORS).collect());
        }
        self.stats.update(reading);
        let result = self.model.detect(reading);
        if self.score_history.len() >= MAX_HISTORY_SIZE {
            self.score_history.remove(0);
        }
        self.score_history.push(result.reconstruction_error);
        if result.is_anomaly {
            self.anomaly_count += 1;
        }
        result
    }
    #[must_use]
    pub fn anomaly_rate(&self) -> f32 {
        if self.total_count == 0 {
            0.0
        } else {
            self.anomaly_count as f32 / self.total_count as f32
        }
    }
    #[must_use]
    pub fn adaptive_threshold(&self) -> f32 {
        if self.score_history.len() < 10 {
            return self.model.threshold();
        }
        let mut sorted = self.score_history.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        sorted
            .get((sorted.len() as f32 * 0.95) as usize)
            .copied()
            .unwrap_or(DEFAULT_THRESHOLD)
    }
    #[must_use]
    pub fn stats(&self) -> &RollingStats {
        &self.stats
    }
    #[must_use]
    pub fn counts(&self) -> (u64, u64) {
        (self.anomaly_count, self.total_count)
    }
}

pub struct SensorSimulator {
    sensor_ranges: [(f32, f32, f32); NUM_SENSORS],
    rng: SimpleRng,
    time_ms: u64,
}

impl SensorSimulator {
    #[must_use]
    pub fn new(seed: u64) -> Self {
        Self {
            sensor_ranges: [
                (0.2, 0.4, 0.05),
                (0.45, 0.55, 0.02),
                (0.0, 0.1, 0.03),
                (0.1, 0.3, 0.05),
                (0.3, 0.6, 0.08),
                (0.2, 0.8, 0.1),
                (0.1, 0.4, 0.05),
                (0.0, 0.2, 0.03),
            ],
            rng: SimpleRng::new(seed),
            time_ms: 0,
        }
    }
    pub fn generate_normal(&mut self) -> SensorReading {
        let mut values = [0.0_f32; NUM_SENSORS];
        for (i, &(min, max, std)) in self.sensor_ranges.iter().enumerate() {
            values[i] = ((min + max) / 2.0 + self.rng.next_gaussian() * std).clamp(0.0, 1.0);
        }
        self.time_ms += 100;
        SensorReading::new(values, self.time_ms)
    }
    pub fn generate_anomaly(&mut self, severity: f32) -> SensorReading {
        let mut r = self.generate_normal();
        for _ in 0..=(self.rng.next_u64() % 3) {
            let idx = (self.rng.next_u64() as usize) % NUM_SENSORS;
            let dir = if self.rng.next_f32() > 0.5 { 1.0 } else { -1.0 };
            r.values[idx] = (r.values[idx] + dir * severity * 0.5).clamp(0.0, 1.0);
        }
        r
    }
    pub fn generate_drift(&mut self, amount: f32) -> SensorReading {
        let mut r = self.generate_normal();
        for v in &mut r.values {
            *v = (*v + amount).clamp(0.0, 1.0);
        }
        r
    }
}

#[derive(Debug, Clone)]
pub enum EdgeError {
    InvalidModelSize { expected: usize, got: usize },
    InvalidReading { reason: String },
    ModelNotInitialized,
}
impl fmt::Display for EdgeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidModelSize { expected, got } => {
                write!(f, "Invalid model size: expected {expected}, got {got}")
            }
            Self::InvalidReading { reason } => write!(f, "Invalid reading: {reason}"),
            Self::ModelNotInitialized => write!(f, "Model not initialized"),
        }
    }
}
impl std::error::Error for EdgeError {}
pub type Result<T> = std::result::Result<T, EdgeError>;

struct SimpleRng {
    state: u64,
}
impl SimpleRng {
    fn new(seed: u64) -> Self {
        Self { state: seed.max(1) }
    }
    fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.state = x;
        x
    }
    fn next_f32(&mut self) -> f32 {
        (self.next_u64() as f64 / u64::MAX as f64) as f32
    }
    fn next_gaussian(&mut self) -> f32 {
        let u1 = self.next_f32().max(1e-10);
        let u2 = self.next_f32();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos()
    }
}

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
