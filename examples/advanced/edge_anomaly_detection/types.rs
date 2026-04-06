#![allow(
    dead_code,
    unused_imports,
    unreachable_pub,
    clippy::trivially_copy_pass_by_ref,
    clippy::wildcard_imports
)]
use proptest::prelude::*;
#[allow(unused_imports)]
use std::fmt;
use std::mem::size_of;

pub const NUM_SENSORS: usize = 8;
pub const LATENT_DIM: usize = 2;
pub const HIDDEN_DIM: usize = 4;
pub const FIXED_POINT_SCALE: i32 = 256;
pub const DEFAULT_THRESHOLD: f32 = 0.15;
pub const MAX_HISTORY_SIZE: usize = 100;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct FixedPoint(pub i16);

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

// MicroAutoencoderWeights impl moved to helpers.rs

pub struct MicroAutoencoder {
    pub weights: MicroAutoencoderWeights,
    pub threshold: f32,
    pub hidden1: [FixedPoint; HIDDEN_DIM],
    pub latent: [FixedPoint; LATENT_DIM],
    pub hidden2: [FixedPoint; HIDDEN_DIM],
    pub output: [FixedPoint; NUM_SENSORS],
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

    // encode(), decode(), forward(), detect() moved to helpers.rs

    #[must_use]
    pub fn get_latent(&self) -> [f32; LATENT_DIM] {
        let mut r = [0.0_f32; LATENT_DIM];
        for (r, &l) in r.iter_mut().zip(self.latent.iter()) {
            *r = l.to_f32();
        }
        r
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
    pub model: MicroAutoencoder,
    pub stats: RollingStats,
    pub score_history: Vec<f32>,
    pub anomaly_count: u64,
    pub total_count: u64,
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
    // process(), adaptive_threshold() moved to helpers.rs

    #[must_use]
    pub fn anomaly_rate(&self) -> f32 {
        if self.total_count == 0 {
            0.0
        } else {
            self.anomaly_count as f32 / self.total_count as f32
        }
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
    pub sensor_ranges: [(f32, f32, f32); NUM_SENSORS],
    pub rng: SimpleRng,
    pub time_ms: u64,
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
    // generate_normal(), generate_anomaly(), generate_drift() moved to helpers.rs
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

pub struct SimpleRng {
    pub state: u64,
}
impl SimpleRng {
    pub fn new(seed: u64) -> Self {
        Self { state: seed.max(1) }
    }
    pub fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.state = x;
        x
    }
    pub fn next_f32(&mut self) -> f32 {
        (self.next_u64() as f64 / u64::MAX as f64) as f32
    }
    pub fn next_gaussian(&mut self) -> f32 {
        let u1 = self.next_f32().max(1e-10);
        let u2 = self.next_f32();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos()
    }
}
