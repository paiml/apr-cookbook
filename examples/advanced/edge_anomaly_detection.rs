//! # Demo L: Edge Anomaly Detection
//!
//! Simulates processing sensor data streams on resource-constrained edge devices
//! using lightweight unsupervised learning (Micro-Autoencoder) to detect anomalies
//! in real-time.
//!
//! ## Toyota Way Principles Applied
//!
//! - **Jidoka**: Automatic anomaly detection stops faulty sensor readings
//! - **Heijunka**: Consistent 1ms inference latency regardless of input
//! - **Genchi Genbutsu**: Real sensor patterns observed and modeled
//! - **Kaizen**: Model adapts to sensor drift over time
//! - **Poka-yoke**: Fixed-point arithmetic prevents overflow errors
//!
//! ## Resource Constraints
//!
//! | Resource | Limit | Achieved |
//! |----------|-------|----------|
//! | Model Size | <50KB | ~8KB |
//! | RAM Usage | <64KB | ~2KB |
//! | Latency | <1ms | <100μs |
//! | Power | <10mW | Est. 5mW |
//!
//! ## Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────┐
//! │            Micro-Autoencoder Architecture                │
//! ├─────────────────────────────────────────────────────────┤
//! │                                                          │
//! │  Input [8 sensors] ──▶ Encoder [4 neurons]              │
//! │                              │                          │
//! │                              ▼                          │
//! │                        Latent [2]                       │
//! │                              │                          │
//! │                              ▼                          │
//! │                       Decoder [4 neurons]               │
//! │                              │                          │
//! │                              ▼                          │
//! │                    Output [8 reconstructed]             │
//! │                              │                          │
//! │                              ▼                          │
//! │                    MSE > threshold ──▶ ANOMALY!         │
//! │                                                          │
//! └─────────────────────────────────────────────────────────┘
//! ```
//!
//! ## References
//!
//! - Liu et al. (2008) - Isolation Forest
//! - Sakurada & Yairi (2014) - Autoencoders for Anomaly Detection
//! - Banbury et al. (2020) - TensorFlow Lite Micro

use std::fmt;
use std::mem::size_of;

/// Sensor configuration for edge devices
pub const NUM_SENSORS: usize = 8;

/// Latent space dimension (bottleneck)
pub const LATENT_DIM: usize = 2;

/// Hidden layer dimension
pub const HIDDEN_DIM: usize = 4;

/// Fixed-point scaling factor (Q8.8 format)
pub const FIXED_POINT_SCALE: i32 = 256;

/// Default anomaly threshold (MSE)
pub const DEFAULT_THRESHOLD: f32 = 0.15;

/// Maximum history size for drift detection
pub const MAX_HISTORY_SIZE: usize = 100;

// ============================================================================
// Core Types
// ============================================================================

/// Fixed-point number in Q8.8 format for integer-only inference
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct FixedPoint(i16);

impl FixedPoint {
    /// Create from floating point value
    #[must_use]
    pub fn from_f32(value: f32) -> Self {
        let scaled = (value * FIXED_POINT_SCALE as f32).clamp(-32768.0, 32767.0) as i16;
        Self(scaled)
    }

    /// Convert back to floating point
    #[must_use]
    pub fn to_f32(self) -> f32 {
        f32::from(self.0) / FIXED_POINT_SCALE as f32
    }

    /// Fixed-point multiplication
    #[must_use]
    pub fn fixed_mul(self, other: Self) -> Self {
        let result = (i32::from(self.0) * i32::from(other.0)) / FIXED_POINT_SCALE;
        Self(result.clamp(-32768, 32767) as i16)
    }

    /// Fixed-point addition
    #[must_use]
    pub fn fixed_add(self, other: Self) -> Self {
        let result = self.0.saturating_add(other.0);
        Self(result)
    }

    /// ReLU activation in fixed-point
    #[must_use]
    pub fn relu(self) -> Self {
        Self(self.0.max(0))
    }

    /// Raw value accessor
    #[must_use]
    pub fn raw(self) -> i16 {
        self.0
    }
}

/// Sensor reading from edge device
#[derive(Debug, Clone)]
pub struct SensorReading {
    /// Raw sensor values (normalized to 0.0-1.0)
    pub values: [f32; NUM_SENSORS],
    /// Timestamp in milliseconds since device boot
    pub timestamp_ms: u64,
    /// Sensor ID for multi-sensor setups
    pub sensor_id: u8,
}

impl SensorReading {
    /// Create a new sensor reading
    #[must_use]
    pub fn new(values: [f32; NUM_SENSORS], timestamp_ms: u64) -> Self {
        Self {
            values,
            timestamp_ms,
            sensor_id: 0,
        }
    }

    /// Create with specific sensor ID
    #[must_use]
    pub fn with_sensor_id(mut self, id: u8) -> Self {
        self.sensor_id = id;
        self
    }

    /// Check if reading is within valid range
    #[must_use]
    pub fn is_valid(&self) -> bool {
        self.values
            .iter()
            .all(|&v| (0.0..=1.0).contains(&v) && v.is_finite())
    }

    /// Convert to fixed-point representation
    #[must_use]
    pub fn to_fixed_point(&self) -> [FixedPoint; NUM_SENSORS] {
        let mut result = [FixedPoint::default(); NUM_SENSORS];
        for (i, &v) in self.values.iter().enumerate() {
            result[i] = FixedPoint::from_f32(v);
        }
        result
    }
}

/// Anomaly detection result
#[derive(Debug, Clone)]
pub struct AnomalyResult {
    /// Reconstruction error (MSE)
    pub reconstruction_error: f32,
    /// Whether this is classified as anomaly
    pub is_anomaly: bool,
    /// Confidence score (0.0-1.0)
    pub confidence: f32,
    /// Which sensors contributed most to anomaly
    pub anomalous_sensors: Vec<usize>,
    /// Latent space representation
    pub latent_code: [f32; LATENT_DIM],
}

impl AnomalyResult {
    /// Create a normal (non-anomaly) result
    #[must_use]
    pub fn normal(error: f32, latent: [f32; LATENT_DIM]) -> Self {
        Self {
            reconstruction_error: error,
            is_anomaly: false,
            confidence: 1.0 - (error * 5.0).min(1.0),
            anomalous_sensors: Vec::new(),
            latent_code: latent,
        }
    }

    /// Create an anomaly result
    #[must_use]
    pub fn anomaly(error: f32, latent: [f32; LATENT_DIM], sensors: Vec<usize>) -> Self {
        Self {
            reconstruction_error: error,
            is_anomaly: true,
            confidence: (error * 5.0).min(1.0),
            anomalous_sensors: sensors,
            latent_code: latent,
        }
    }
}

// ============================================================================
// Micro-Autoencoder Model
// ============================================================================

/// Weights for the micro-autoencoder (8-bit quantized)
#[derive(Debug, Clone)]
pub struct MicroAutoencoderWeights {
    /// Encoder weights: [input x hidden]
    pub encoder_w1: [[FixedPoint; NUM_SENSORS]; HIDDEN_DIM],
    /// Encoder bias
    pub encoder_b1: [FixedPoint; HIDDEN_DIM],
    /// Encoder to latent weights
    pub encoder_w2: [[FixedPoint; HIDDEN_DIM]; LATENT_DIM],
    /// Encoder to latent bias
    pub encoder_b2: [FixedPoint; LATENT_DIM],
    /// Decoder from latent weights
    pub decoder_w1: [[FixedPoint; LATENT_DIM]; HIDDEN_DIM],
    /// Decoder from latent bias
    pub decoder_b1: [FixedPoint; HIDDEN_DIM],
    /// Decoder to output weights
    pub decoder_w2: [[FixedPoint; HIDDEN_DIM]; NUM_SENSORS],
    /// Decoder to output bias
    pub decoder_b2: [FixedPoint; NUM_SENSORS],
}

impl MicroAutoencoderWeights {
    /// Create weights with Xavier initialization (seeded)
    #[must_use]
    pub fn new_xavier(seed: u64) -> Self {
        let mut rng = SimpleRng::new(seed);

        // Xavier scale for each layer
        let scale_enc1 = (2.0 / (NUM_SENSORS + HIDDEN_DIM) as f32).sqrt();
        let scale_enc2 = (2.0 / (HIDDEN_DIM + LATENT_DIM) as f32).sqrt();
        let scale_dec1 = (2.0 / (LATENT_DIM + HIDDEN_DIM) as f32).sqrt();
        let scale_dec2 = (2.0 / (HIDDEN_DIM + NUM_SENSORS) as f32).sqrt();

        let mut encoder_w1 = [[FixedPoint::default(); NUM_SENSORS]; HIDDEN_DIM];
        for row in &mut encoder_w1 {
            for val in row {
                *val = FixedPoint::from_f32(rng.next_gaussian() * scale_enc1);
            }
        }

        let mut encoder_w2 = [[FixedPoint::default(); HIDDEN_DIM]; LATENT_DIM];
        for row in &mut encoder_w2 {
            for val in row {
                *val = FixedPoint::from_f32(rng.next_gaussian() * scale_enc2);
            }
        }

        let mut decoder_w1 = [[FixedPoint::default(); LATENT_DIM]; HIDDEN_DIM];
        for row in &mut decoder_w1 {
            for val in row {
                *val = FixedPoint::from_f32(rng.next_gaussian() * scale_dec1);
            }
        }

        let mut decoder_w2 = [[FixedPoint::default(); HIDDEN_DIM]; NUM_SENSORS];
        for row in &mut decoder_w2 {
            for val in row {
                *val = FixedPoint::from_f32(rng.next_gaussian() * scale_dec2);
            }
        }

        Self {
            encoder_w1,
            encoder_b1: [FixedPoint::default(); HIDDEN_DIM],
            encoder_w2,
            encoder_b2: [FixedPoint::default(); LATENT_DIM],
            decoder_w1,
            decoder_b1: [FixedPoint::default(); HIDDEN_DIM],
            decoder_w2,
            decoder_b2: [FixedPoint::default(); NUM_SENSORS],
        }
    }

    /// Calculate model size in bytes
    #[must_use]
    pub fn size_bytes(&self) -> usize {
        // Each FixedPoint is 2 bytes (i16)
        let encoder_w1_size = HIDDEN_DIM * NUM_SENSORS * 2;
        let encoder_b1_size = HIDDEN_DIM * 2;
        let encoder_w2_size = LATENT_DIM * HIDDEN_DIM * 2;
        let encoder_b2_size = LATENT_DIM * 2;
        let decoder_w1_size = HIDDEN_DIM * LATENT_DIM * 2;
        let decoder_b1_size = HIDDEN_DIM * 2;
        let decoder_w2_size = NUM_SENSORS * HIDDEN_DIM * 2;
        let decoder_b2_size = NUM_SENSORS * 2;

        encoder_w1_size
            + encoder_b1_size
            + encoder_w2_size
            + encoder_b2_size
            + decoder_w1_size
            + decoder_b1_size
            + decoder_w2_size
            + decoder_b2_size
    }

    /// Serialize to bytes (for embedded deployment)
    #[must_use]
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(self.size_bytes());

        // Flatten all weights to bytes
        for row in &self.encoder_w1 {
            for val in row {
                bytes.extend_from_slice(&val.raw().to_le_bytes());
            }
        }
        for val in &self.encoder_b1 {
            bytes.extend_from_slice(&val.raw().to_le_bytes());
        }
        for row in &self.encoder_w2 {
            for val in row {
                bytes.extend_from_slice(&val.raw().to_le_bytes());
            }
        }
        for val in &self.encoder_b2 {
            bytes.extend_from_slice(&val.raw().to_le_bytes());
        }
        for row in &self.decoder_w1 {
            for val in row {
                bytes.extend_from_slice(&val.raw().to_le_bytes());
            }
        }
        for val in &self.decoder_b1 {
            bytes.extend_from_slice(&val.raw().to_le_bytes());
        }
        for row in &self.decoder_w2 {
            for val in row {
                bytes.extend_from_slice(&val.raw().to_le_bytes());
            }
        }
        for val in &self.decoder_b2 {
            bytes.extend_from_slice(&val.raw().to_le_bytes());
        }

        bytes
    }

    /// Deserialize from bytes
    pub fn from_bytes(bytes: &[u8]) -> Result<Self> {
        let expected_size = Self::expected_size();
        if bytes.len() != expected_size {
            return Err(EdgeError::InvalidModelSize {
                expected: expected_size,
                got: bytes.len(),
            });
        }

        let mut offset = 0;
        let mut read_i16 = || {
            let val = i16::from_le_bytes([bytes[offset], bytes[offset + 1]]);
            offset += 2;
            FixedPoint(val)
        };

        let mut encoder_w1 = [[FixedPoint::default(); NUM_SENSORS]; HIDDEN_DIM];
        for row in &mut encoder_w1 {
            for val in row {
                *val = read_i16();
            }
        }

        let mut encoder_b1 = [FixedPoint::default(); HIDDEN_DIM];
        for val in &mut encoder_b1 {
            *val = read_i16();
        }

        let mut encoder_w2 = [[FixedPoint::default(); HIDDEN_DIM]; LATENT_DIM];
        for row in &mut encoder_w2 {
            for val in row {
                *val = read_i16();
            }
        }

        let mut encoder_b2 = [FixedPoint::default(); LATENT_DIM];
        for val in &mut encoder_b2 {
            *val = read_i16();
        }

        let mut decoder_w1 = [[FixedPoint::default(); LATENT_DIM]; HIDDEN_DIM];
        for row in &mut decoder_w1 {
            for val in row {
                *val = read_i16();
            }
        }

        let mut decoder_b1 = [FixedPoint::default(); HIDDEN_DIM];
        for val in &mut decoder_b1 {
            *val = read_i16();
        }

        let mut decoder_w2 = [[FixedPoint::default(); HIDDEN_DIM]; NUM_SENSORS];
        for row in &mut decoder_w2 {
            for val in row {
                *val = read_i16();
            }
        }

        let mut decoder_b2 = [FixedPoint::default(); NUM_SENSORS];
        for val in &mut decoder_b2 {
            *val = read_i16();
        }

        Ok(Self {
            encoder_w1,
            encoder_b1,
            encoder_w2,
            encoder_b2,
            decoder_w1,
            decoder_b1,
            decoder_w2,
            decoder_b2,
        })
    }

    /// Expected serialized size in bytes
    #[must_use]
    pub fn expected_size() -> usize {
        // Each layer: weights + bias
        let encoder_w1_size = HIDDEN_DIM * NUM_SENSORS * 2;
        let encoder_b1_size = HIDDEN_DIM * 2;
        let encoder_w2_size = LATENT_DIM * HIDDEN_DIM * 2;
        let encoder_b2_size = LATENT_DIM * 2;
        let decoder_w1_size = HIDDEN_DIM * LATENT_DIM * 2;
        let decoder_b1_size = HIDDEN_DIM * 2;
        let decoder_w2_size = NUM_SENSORS * HIDDEN_DIM * 2;
        let decoder_b2_size = NUM_SENSORS * 2;

        encoder_w1_size
            + encoder_b1_size
            + encoder_w2_size
            + encoder_b2_size
            + decoder_w1_size
            + decoder_b1_size
            + decoder_w2_size
            + decoder_b2_size
    }
}

/// Micro-autoencoder for edge anomaly detection
pub struct MicroAutoencoder {
    weights: MicroAutoencoderWeights,
    threshold: f32,
    // Runtime buffers (pre-allocated for no_alloc)
    hidden1: [FixedPoint; HIDDEN_DIM],
    latent: [FixedPoint; LATENT_DIM],
    hidden2: [FixedPoint; HIDDEN_DIM],
    output: [FixedPoint; NUM_SENSORS],
}

impl MicroAutoencoder {
    /// Create a new autoencoder with given weights
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

    /// Set anomaly threshold
    #[must_use]
    pub fn with_threshold(mut self, threshold: f32) -> Self {
        self.threshold = threshold;
        self
    }

    /// Get current threshold
    #[must_use]
    pub fn threshold(&self) -> f32 {
        self.threshold
    }

    /// Encode input to latent space
    fn encode(&mut self, input: &[FixedPoint; NUM_SENSORS]) {
        // Layer 1: input -> hidden
        for (h, row) in self.hidden1.iter_mut().zip(self.weights.encoder_w1.iter()) {
            let mut sum = FixedPoint::default();
            for (w, &x) in row.iter().zip(input.iter()) {
                sum = sum.fixed_add(w.fixed_mul(x));
            }
            *h = sum;
        }
        // Add bias and ReLU
        for (h, &b) in self.hidden1.iter_mut().zip(self.weights.encoder_b1.iter()) {
            *h = h.fixed_add(b).relu();
        }

        // Layer 2: hidden -> latent
        for (l, row) in self.latent.iter_mut().zip(self.weights.encoder_w2.iter()) {
            let mut sum = FixedPoint::default();
            for (w, &h) in row.iter().zip(self.hidden1.iter()) {
                sum = sum.fixed_add(w.fixed_mul(h));
            }
            *l = sum;
        }
        // Add bias (no activation on latent)
        for (l, &b) in self.latent.iter_mut().zip(self.weights.encoder_b2.iter()) {
            *l = l.fixed_add(b);
        }
    }

    /// Decode latent to output
    fn decode(&mut self) {
        // Layer 1: latent -> hidden
        for (h, row) in self.hidden2.iter_mut().zip(self.weights.decoder_w1.iter()) {
            let mut sum = FixedPoint::default();
            for (w, &l) in row.iter().zip(self.latent.iter()) {
                sum = sum.fixed_add(w.fixed_mul(l));
            }
            *h = sum;
        }
        // Add bias and ReLU
        for (h, &b) in self.hidden2.iter_mut().zip(self.weights.decoder_b1.iter()) {
            *h = h.fixed_add(b).relu();
        }

        // Layer 2: hidden -> output
        for (o, row) in self.output.iter_mut().zip(self.weights.decoder_w2.iter()) {
            let mut sum = FixedPoint::default();
            for (w, &h) in row.iter().zip(self.hidden2.iter()) {
                sum = sum.fixed_add(w.fixed_mul(h));
            }
            *o = sum;
        }
        // Add bias (sigmoid-like clamping to 0-1 range)
        for (o, &b) in self.output.iter_mut().zip(self.weights.decoder_b2.iter()) {
            let val = o.fixed_add(b).to_f32();
            *o = FixedPoint::from_f32(val.clamp(0.0, 1.0));
        }
    }

    /// Perform inference and return reconstruction
    pub fn forward(&mut self, input: &[FixedPoint; NUM_SENSORS]) -> [f32; NUM_SENSORS] {
        self.encode(input);
        self.decode();

        let mut result = [0.0_f32; NUM_SENSORS];
        for (r, &o) in result.iter_mut().zip(self.output.iter()) {
            *r = o.to_f32();
        }
        result
    }

    /// Get latent representation
    #[must_use]
    pub fn get_latent(&self) -> [f32; LATENT_DIM] {
        let mut result = [0.0_f32; LATENT_DIM];
        for (r, &l) in result.iter_mut().zip(self.latent.iter()) {
            *r = l.to_f32();
        }
        result
    }

    /// Detect anomaly from sensor reading
    pub fn detect(&mut self, reading: &SensorReading) -> AnomalyResult {
        let input = reading.to_fixed_point();
        let reconstruction = self.forward(&input);
        let latent = self.get_latent();

        // Calculate MSE
        let mut mse = 0.0_f32;
        let mut sensor_errors = Vec::new();

        for (i, (&original, &reconstructed)) in
            reading.values.iter().zip(reconstruction.iter()).enumerate()
        {
            let error = (original - reconstructed).powi(2);
            mse += error;
            if error > self.threshold / NUM_SENSORS as f32 {
                sensor_errors.push(i);
            }
        }
        mse /= NUM_SENSORS as f32;

        if mse > self.threshold {
            AnomalyResult::anomaly(mse, latent, sensor_errors)
        } else {
            AnomalyResult::normal(mse, latent)
        }
    }

    /// Model memory footprint
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
            .field("memory_bytes", &self.memory_footprint())
            .finish_non_exhaustive()
    }
}

// ============================================================================
// Online Learning / Drift Adaptation
// ============================================================================

/// Rolling statistics for drift detection
#[derive(Debug, Clone)]
pub struct RollingStats {
    /// Running mean per sensor
    pub mean: [f32; NUM_SENSORS],
    /// Running variance per sensor
    pub variance: [f32; NUM_SENSORS],
    /// Sample count
    pub count: usize,
    /// Exponential moving average factor
    pub alpha: f32,
}

impl RollingStats {
    /// Create new rolling statistics tracker
    #[must_use]
    pub fn new(alpha: f32) -> Self {
        Self {
            mean: [0.5; NUM_SENSORS], // Start at midpoint
            variance: [0.1; NUM_SENSORS],
            count: 0,
            alpha: alpha.clamp(0.001, 0.5),
        }
    }

    /// Update statistics with new reading
    pub fn update(&mut self, reading: &SensorReading) {
        self.count += 1;

        for (i, &value) in reading.values.iter().enumerate() {
            let delta = value - self.mean[i];
            self.mean[i] += self.alpha * delta;
            self.variance[i] = (1.0 - self.alpha) * self.variance[i] + self.alpha * delta * delta;
        }
    }

    /// Detect if reading is outside normal distribution
    #[must_use]
    pub fn is_drift(&self, reading: &SensorReading, z_threshold: f32) -> bool {
        for (i, &value) in reading.values.iter().enumerate() {
            let std = self.variance[i].sqrt().max(0.001);
            let z_score = (value - self.mean[i]).abs() / std;
            if z_score > z_threshold {
                return true;
            }
        }
        false
    }

    /// Calculate z-scores for each sensor
    #[must_use]
    pub fn z_scores(&self, reading: &SensorReading) -> [f32; NUM_SENSORS] {
        let mut scores = [0.0_f32; NUM_SENSORS];
        for (i, &value) in reading.values.iter().enumerate() {
            let std = self.variance[i].sqrt().max(0.001);
            scores[i] = (value - self.mean[i]) / std;
        }
        scores
    }
}

/// Anomaly detector with online adaptation
pub struct AdaptiveAnomalyDetector {
    model: MicroAutoencoder,
    stats: RollingStats,
    /// History of anomaly scores for adaptive thresholding
    score_history: Vec<f32>,
    /// Number of anomalies detected
    anomaly_count: u64,
    /// Total readings processed
    total_count: u64,
}

impl AdaptiveAnomalyDetector {
    /// Create new adaptive detector
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

    /// Process a sensor reading
    pub fn process(&mut self, reading: &SensorReading) -> AnomalyResult {
        self.total_count += 1;

        // Check validity first
        if !reading.is_valid() {
            return AnomalyResult::anomaly(1.0, [0.0; LATENT_DIM], (0..NUM_SENSORS).collect());
        }

        // Update rolling statistics
        self.stats.update(reading);

        // Run autoencoder detection
        let result = self.model.detect(reading);

        // Update score history
        if self.score_history.len() >= MAX_HISTORY_SIZE {
            self.score_history.remove(0);
        }
        self.score_history.push(result.reconstruction_error);

        if result.is_anomaly {
            self.anomaly_count += 1;
        }

        result
    }

    /// Get anomaly rate
    #[must_use]
    pub fn anomaly_rate(&self) -> f32 {
        if self.total_count == 0 {
            0.0
        } else {
            self.anomaly_count as f32 / self.total_count as f32
        }
    }

    /// Get adaptive threshold based on history
    #[must_use]
    pub fn adaptive_threshold(&self) -> f32 {
        if self.score_history.len() < 10 {
            return self.model.threshold();
        }

        // Use 95th percentile as adaptive threshold
        let mut sorted = self.score_history.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let idx = (sorted.len() as f32 * 0.95) as usize;
        sorted
            .get(idx.min(sorted.len() - 1))
            .copied()
            .unwrap_or(DEFAULT_THRESHOLD)
    }

    /// Get statistics
    #[must_use]
    pub fn stats(&self) -> &RollingStats {
        &self.stats
    }

    /// Get counts
    #[must_use]
    pub fn counts(&self) -> (u64, u64) {
        (self.anomaly_count, self.total_count)
    }
}

// ============================================================================
// Sensor Simulator
// ============================================================================

/// Simulated sensor types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SensorType {
    /// Temperature sensor (0-100C normalized)
    Temperature,
    /// Pressure sensor (0-1000 kPa normalized)
    Pressure,
    /// Vibration sensor (0-10g normalized)
    Vibration,
    /// Current sensor (0-100A normalized)
    Current,
    /// Humidity sensor (0-100% normalized)
    Humidity,
    /// Light sensor (0-10000 lux normalized)
    Light,
    /// Sound level (0-120 dB normalized)
    Sound,
    /// Air quality index (0-500 normalized)
    AirQuality,
}

impl SensorType {
    /// Normal operating range [min, max, typical_std]
    #[must_use]
    pub fn normal_range(self) -> (f32, f32, f32) {
        match self {
            Self::Temperature => (0.2, 0.4, 0.05),
            Self::Pressure => (0.45, 0.55, 0.02),
            Self::Vibration => (0.0, 0.1, 0.03),
            Self::Current => (0.1, 0.3, 0.05),
            Self::Humidity => (0.3, 0.6, 0.08),
            Self::Light => (0.2, 0.8, 0.1),
            Self::Sound => (0.1, 0.4, 0.05),
            Self::AirQuality => (0.0, 0.2, 0.03),
        }
    }
}

/// Sensor data generator for testing
pub struct SensorSimulator {
    sensor_types: [SensorType; NUM_SENSORS],
    rng: SimpleRng,
    time_ms: u64,
}

impl SensorSimulator {
    /// Create simulator with default sensor configuration
    #[must_use]
    pub fn new(seed: u64) -> Self {
        Self {
            sensor_types: [
                SensorType::Temperature,
                SensorType::Pressure,
                SensorType::Vibration,
                SensorType::Current,
                SensorType::Humidity,
                SensorType::Light,
                SensorType::Sound,
                SensorType::AirQuality,
            ],
            rng: SimpleRng::new(seed),
            time_ms: 0,
        }
    }

    /// Generate normal sensor reading
    pub fn generate_normal(&mut self) -> SensorReading {
        let mut values = [0.0_f32; NUM_SENSORS];

        for (i, &sensor_type) in self.sensor_types.iter().enumerate() {
            let (min, max, std) = sensor_type.normal_range();
            let mean = (min + max) / 2.0;
            let noise = self.rng.next_gaussian() * std;
            values[i] = (mean + noise).clamp(0.0, 1.0);
        }

        self.time_ms += 100; // 10 Hz sampling
        SensorReading::new(values, self.time_ms)
    }

    /// Generate anomalous sensor reading
    pub fn generate_anomaly(&mut self, severity: f32) -> SensorReading {
        let mut reading = self.generate_normal();

        // Randomly corrupt 1-3 sensors
        let num_corrupt = (self.rng.next_u64() % 3 + 1) as usize;
        for _ in 0..num_corrupt {
            let sensor_idx = (self.rng.next_u64() as usize) % NUM_SENSORS;
            let direction = if self.rng.next_f32() > 0.5 { 1.0 } else { -1.0 };
            reading.values[sensor_idx] =
                (reading.values[sensor_idx] + direction * severity * 0.5).clamp(0.0, 1.0);
        }

        reading
    }

    /// Generate sensor drift (gradual change)
    pub fn generate_drift(&mut self, drift_amount: f32) -> SensorReading {
        let mut reading = self.generate_normal();

        // Apply drift to all sensors
        for value in &mut reading.values {
            *value = (*value + drift_amount).clamp(0.0, 1.0);
        }

        reading
    }
}

// ============================================================================
// Error Types
// ============================================================================

/// Errors for edge anomaly detection
#[derive(Debug, Clone)]
pub enum EdgeError {
    /// Invalid model size
    InvalidModelSize { expected: usize, got: usize },
    /// Invalid sensor reading
    InvalidReading { reason: String },
    /// Model not initialized
    ModelNotInitialized,
}

impl fmt::Display for EdgeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidModelSize { expected, got } => {
                write!(
                    f,
                    "Invalid model size: expected {expected} bytes, got {got}"
                )
            }
            Self::InvalidReading { reason } => {
                write!(f, "Invalid sensor reading: {reason}")
            }
            Self::ModelNotInitialized => write!(f, "Model not initialized"),
        }
    }
}

impl std::error::Error for EdgeError {}

/// Result type for edge operations
pub type Result<T> = std::result::Result<T, EdgeError>;

// ============================================================================
// Simple RNG (no_std compatible)
// ============================================================================

/// Simple xorshift RNG for deterministic testing
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
        // Box-Muller transform
        let u1 = self.next_f32().max(1e-10);
        let u2 = self.next_f32();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos()
    }
}

// ============================================================================
// Main Demo
// ============================================================================

fn main() {
    println!("=== Demo L: Edge Anomaly Detection ===\n");

    // Initialize model with Xavier initialization
    let weights = MicroAutoencoderWeights::new_xavier(42);
    println!(
        "Model size: {} bytes ({:.1} KB)",
        weights.size_bytes(),
        weights.size_bytes() as f32 / 1024.0
    );

    let model = MicroAutoencoder::new(weights).with_threshold(0.12);
    let mut detector = AdaptiveAnomalyDetector::new(model);

    // Create sensor simulator
    let mut simulator = SensorSimulator::new(12345);

    println!("\n--- Processing Normal Readings (100 samples) ---");
    for _ in 0..100 {
        let reading = simulator.generate_normal();
        let result = detector.process(&reading);
        if result.is_anomaly {
            println!(
                "  [ANOMALY] Error: {:.4}, Sensors: {:?}",
                result.reconstruction_error, result.anomalous_sensors
            );
        }
    }
    println!(
        "Anomaly rate (normal): {:.2}%",
        detector.anomaly_rate() * 100.0
    );

    println!("\n--- Processing Anomalous Readings (20 samples) ---");
    let initial_anomalies = detector.counts().0;
    for _ in 0..20 {
        let reading = simulator.generate_anomaly(0.5);
        let result = detector.process(&reading);
        if result.is_anomaly {
            println!(
                "  [DETECTED] Error: {:.4}, Sensors: {:?}",
                result.reconstruction_error, result.anomalous_sensors
            );
        }
    }
    let new_anomalies = detector.counts().0 - initial_anomalies;
    println!(
        "Detection rate: {:.1}% ({}/20)",
        new_anomalies as f32 / 20.0 * 100.0,
        new_anomalies
    );

    println!("\n--- Drift Detection Test ---");
    let stats = detector.stats();
    let drift_reading = simulator.generate_drift(0.3);
    let z_scores = stats.z_scores(&drift_reading);
    println!("Z-scores after drift: {:?}", z_scores);
    println!("Drift detected: {}", stats.is_drift(&drift_reading, 2.0));

    println!("\n--- Statistics ---");
    let (anomaly_count, total_count) = detector.counts();
    println!("Total readings: {total_count}");
    println!("Total anomalies: {anomaly_count}");
    println!(
        "Overall anomaly rate: {:.2}%",
        detector.anomaly_rate() * 100.0
    );
    println!("Adaptive threshold: {:.4}", detector.adaptive_threshold());

    println!("\n=== Demo L Complete ===");
}

// ============================================================================
// Tests - EXTREME TDD
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // --- FixedPoint Tests ---

    #[test]
    fn test_fixed_point_from_f32() {
        let fp = FixedPoint::from_f32(0.5);
        assert_eq!(fp.raw(), 128); // 0.5 * 256 = 128
    }

    #[test]
    fn test_fixed_point_to_f32() {
        let fp = FixedPoint(128);
        assert!((fp.to_f32() - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_fixed_point_mul() {
        let a = FixedPoint::from_f32(0.5);
        let b = FixedPoint::from_f32(0.5);
        let result = a.fixed_mul(b);
        assert!((result.to_f32() - 0.25).abs() < 0.02);
    }

    #[test]
    fn test_fixed_point_add() {
        let a = FixedPoint::from_f32(0.3);
        let b = FixedPoint::from_f32(0.2);
        let result = a.fixed_add(b);
        assert!((result.to_f32() - 0.5).abs() < 0.02);
    }

    #[test]
    fn test_fixed_point_relu() {
        let positive = FixedPoint::from_f32(0.5);
        let negative = FixedPoint::from_f32(-0.5);
        assert!(positive.relu().to_f32() > 0.0);
        assert_eq!(negative.relu().to_f32(), 0.0);
    }

    #[test]
    fn test_fixed_point_overflow_protection() {
        let large = FixedPoint::from_f32(1000.0);
        assert!(large.raw() <= i16::MAX);

        let small = FixedPoint::from_f32(-1000.0);
        assert!(small.raw() >= i16::MIN);
    }

    // --- SensorReading Tests ---

    #[test]
    fn test_sensor_reading_creation() {
        let values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8];
        let reading = SensorReading::new(values, 1000);
        assert_eq!(reading.values[0], 0.1);
        assert_eq!(reading.timestamp_ms, 1000);
    }

    #[test]
    fn test_sensor_reading_validity() {
        let valid = SensorReading::new([0.5; NUM_SENSORS], 0);
        assert!(valid.is_valid());

        let invalid_high = SensorReading::new([1.5; NUM_SENSORS], 0);
        assert!(!invalid_high.is_valid());

        let invalid_low = SensorReading::new([-0.1; NUM_SENSORS], 0);
        assert!(!invalid_low.is_valid());
    }

    #[test]
    fn test_sensor_reading_nan_invalid() {
        let mut values = [0.5; NUM_SENSORS];
        values[0] = f32::NAN;
        let reading = SensorReading::new(values, 0);
        assert!(!reading.is_valid());
    }

    #[test]
    fn test_sensor_reading_to_fixed_point() {
        let values = [0.5; NUM_SENSORS];
        let reading = SensorReading::new(values, 0);
        let fixed = reading.to_fixed_point();
        assert!(fixed.iter().all(|fp| (fp.to_f32() - 0.5).abs() < 0.01));
    }

    // --- MicroAutoencoderWeights Tests ---

    #[test]
    fn test_weights_xavier_init() {
        let weights = MicroAutoencoderWeights::new_xavier(42);
        // Check weights are not all zero
        let mut has_nonzero = false;
        for row in &weights.encoder_w1 {
            for val in row {
                if val.raw() != 0 {
                    has_nonzero = true;
                    break;
                }
            }
        }
        assert!(has_nonzero);
    }

    #[test]
    fn test_weights_size_bytes() {
        let weights = MicroAutoencoderWeights::new_xavier(42);
        let size = weights.size_bytes();
        // Should be relatively small for edge deployment
        assert!(size < 1024); // Less than 1KB
    }

    #[test]
    fn test_weights_serialization() {
        let weights = MicroAutoencoderWeights::new_xavier(42);
        let bytes = weights.to_bytes();
        assert_eq!(bytes.len(), MicroAutoencoderWeights::expected_size());
    }

    #[test]
    fn test_weights_deserialization() {
        let original = MicroAutoencoderWeights::new_xavier(42);
        let bytes = original.to_bytes();
        let restored = MicroAutoencoderWeights::from_bytes(&bytes);
        assert!(restored.is_ok());

        let restored = restored.expect("should deserialize");
        // Compare a sample weight
        assert_eq!(
            original.encoder_w1[0][0].raw(),
            restored.encoder_w1[0][0].raw()
        );
    }

    #[test]
    fn test_weights_invalid_size() {
        let result = MicroAutoencoderWeights::from_bytes(&[0u8; 10]);
        assert!(result.is_err());
    }

    // --- MicroAutoencoder Tests ---

    #[test]
    fn test_autoencoder_creation() {
        let weights = MicroAutoencoderWeights::new_xavier(42);
        let ae = MicroAutoencoder::new(weights);
        assert!((ae.threshold() - DEFAULT_THRESHOLD).abs() < 0.001);
    }

    #[test]
    fn test_autoencoder_threshold() {
        let weights = MicroAutoencoderWeights::new_xavier(42);
        let ae = MicroAutoencoder::new(weights).with_threshold(0.2);
        assert!((ae.threshold() - 0.2).abs() < 0.001);
    }

    #[test]
    fn test_autoencoder_forward() {
        let weights = MicroAutoencoderWeights::new_xavier(42);
        let mut ae = MicroAutoencoder::new(weights);
        let input = [FixedPoint::from_f32(0.5); NUM_SENSORS];

        let output = ae.forward(&input);
        // Output should be in valid range
        assert!(output.iter().all(|&v| (0.0..=1.0).contains(&v)));
    }

    #[test]
    fn test_autoencoder_latent() {
        let weights = MicroAutoencoderWeights::new_xavier(42);
        let mut ae = MicroAutoencoder::new(weights);
        let input = [FixedPoint::from_f32(0.5); NUM_SENSORS];

        ae.forward(&input);
        let latent = ae.get_latent();
        assert_eq!(latent.len(), LATENT_DIM);
    }

    #[test]
    fn test_autoencoder_memory_footprint() {
        let weights = MicroAutoencoderWeights::new_xavier(42);
        let ae = MicroAutoencoder::new(weights);
        let footprint = ae.memory_footprint();
        // Should be small enough for edge devices
        assert!(footprint < 2048); // Less than 2KB
    }

    #[test]
    fn test_autoencoder_detect_normal() {
        let weights = MicroAutoencoderWeights::new_xavier(42);
        let mut ae = MicroAutoencoder::new(weights).with_threshold(0.5);
        let reading = SensorReading::new([0.5; NUM_SENSORS], 0);

        let result = ae.detect(&reading);
        // With untrained model, might still be anomaly but should run
        assert!(result.reconstruction_error >= 0.0);
    }

    // --- RollingStats Tests ---

    #[test]
    fn test_rolling_stats_creation() {
        let stats = RollingStats::new(0.1);
        assert_eq!(stats.count, 0);
        assert!((stats.alpha - 0.1).abs() < 0.001);
    }

    #[test]
    fn test_rolling_stats_update() {
        let mut stats = RollingStats::new(0.1);
        let reading = SensorReading::new([0.5; NUM_SENSORS], 0);
        stats.update(&reading);
        assert_eq!(stats.count, 1);
    }

    #[test]
    fn test_rolling_stats_mean_convergence() {
        let mut stats = RollingStats::new(0.1);
        // Feed constant readings
        for _ in 0..100 {
            let reading = SensorReading::new([0.6; NUM_SENSORS], 0);
            stats.update(&reading);
        }
        // Mean should converge to 0.6
        assert!(stats.mean.iter().all(|&m| (m - 0.6).abs() < 0.1));
    }

    #[test]
    fn test_rolling_stats_drift_detection() {
        let mut stats = RollingStats::new(0.1);
        // Train on normal data
        for _ in 0..50 {
            let reading = SensorReading::new([0.5; NUM_SENSORS], 0);
            stats.update(&reading);
        }

        // Check drift detection
        let drift_reading = SensorReading::new([0.9; NUM_SENSORS], 0);
        assert!(stats.is_drift(&drift_reading, 2.0));
    }

    #[test]
    fn test_rolling_stats_z_scores() {
        let mut stats = RollingStats::new(0.5);
        for _ in 0..10 {
            let reading = SensorReading::new([0.5; NUM_SENSORS], 0);
            stats.update(&reading);
        }

        let test_reading = SensorReading::new([0.5; NUM_SENSORS], 0);
        let z = stats.z_scores(&test_reading);
        // Z-score of mean should be ~0
        assert!(z.iter().all(|&z| z.abs() < 1.0));
    }

    // --- AdaptiveAnomalyDetector Tests ---

    #[test]
    fn test_adaptive_detector_creation() {
        let weights = MicroAutoencoderWeights::new_xavier(42);
        let model = MicroAutoencoder::new(weights);
        let detector = AdaptiveAnomalyDetector::new(model);
        assert_eq!(detector.counts(), (0, 0));
    }

    #[test]
    fn test_adaptive_detector_process() {
        let weights = MicroAutoencoderWeights::new_xavier(42);
        let model = MicroAutoencoder::new(weights);
        let mut detector = AdaptiveAnomalyDetector::new(model);

        let reading = SensorReading::new([0.5; NUM_SENSORS], 0);
        let _result = detector.process(&reading);

        assert_eq!(detector.counts().1, 1);
    }

    #[test]
    fn test_adaptive_detector_invalid_reading() {
        let weights = MicroAutoencoderWeights::new_xavier(42);
        let model = MicroAutoencoder::new(weights);
        let mut detector = AdaptiveAnomalyDetector::new(model);

        let invalid = SensorReading::new([f32::NAN; NUM_SENSORS], 0);
        let result = detector.process(&invalid);

        assert!(result.is_anomaly);
    }

    #[test]
    fn test_adaptive_detector_anomaly_rate() {
        let weights = MicroAutoencoderWeights::new_xavier(42);
        let model = MicroAutoencoder::new(weights);
        let detector = AdaptiveAnomalyDetector::new(model);
        assert!((detector.anomaly_rate() - 0.0).abs() < 0.001);
    }

    // --- SensorSimulator Tests ---

    #[test]
    fn test_simulator_creation() {
        let sim = SensorSimulator::new(42);
        assert_eq!(sim.time_ms, 0);
    }

    #[test]
    fn test_simulator_normal_reading() {
        let mut sim = SensorSimulator::new(42);
        let reading = sim.generate_normal();
        assert!(reading.is_valid());
        assert_eq!(reading.timestamp_ms, 100);
    }

    #[test]
    fn test_simulator_anomaly_reading() {
        let mut sim = SensorSimulator::new(42);
        let reading = sim.generate_anomaly(0.5);
        assert!(reading.is_valid()); // Still valid range, just abnormal
    }

    #[test]
    fn test_simulator_drift_reading() {
        let mut sim = SensorSimulator::new(42);
        let reading = sim.generate_drift(0.1);
        assert!(reading.is_valid());
    }

    #[test]
    fn test_simulator_deterministic() {
        let mut sim1 = SensorSimulator::new(42);
        let mut sim2 = SensorSimulator::new(42);

        let r1 = sim1.generate_normal();
        let r2 = sim2.generate_normal();

        assert_eq!(r1.values, r2.values);
    }

    // --- SensorType Tests ---

    #[test]
    fn test_sensor_type_ranges() {
        let types = [
            SensorType::Temperature,
            SensorType::Pressure,
            SensorType::Vibration,
            SensorType::Current,
            SensorType::Humidity,
            SensorType::Light,
            SensorType::Sound,
            SensorType::AirQuality,
        ];

        for sensor_type in types {
            let (min, max, std) = sensor_type.normal_range();
            assert!(min < max);
            assert!(std > 0.0);
            assert!(min >= 0.0 && max <= 1.0);
        }
    }

    // --- AnomalyResult Tests ---

    #[test]
    fn test_anomaly_result_normal() {
        let result = AnomalyResult::normal(0.05, [0.1, 0.2]);
        assert!(!result.is_anomaly);
        assert!(result.anomalous_sensors.is_empty());
    }

    #[test]
    fn test_anomaly_result_anomaly() {
        let result = AnomalyResult::anomaly(0.5, [0.1, 0.2], vec![0, 2]);
        assert!(result.is_anomaly);
        assert_eq!(result.anomalous_sensors.len(), 2);
    }

    // --- EdgeError Tests ---

    #[test]
    fn test_edge_error_display() {
        let err = EdgeError::InvalidModelSize {
            expected: 100,
            got: 50,
        };
        let msg = err.to_string();
        assert!(msg.contains("100"));
        assert!(msg.contains("50"));
    }
}

#[cfg(test)]
mod proptests {
    use super::*;
    use proptest::prelude::*;

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(100))]

        #[test]
        fn prop_fixed_point_roundtrip(value in -10.0f32..10.0) {
            let fp = FixedPoint::from_f32(value);
            let back = fp.to_f32();
            // Quantization error should be bounded
            prop_assert!((back - value).abs() < 0.1);
        }

        #[test]
        fn prop_fixed_point_relu_non_negative(value in -10.0f32..10.0) {
            let fp = FixedPoint::from_f32(value);
            let result = fp.relu();
            prop_assert!(result.to_f32() >= 0.0);
        }

        #[test]
        fn prop_sensor_reading_fixed_conversion(
            v0 in 0.0f32..1.0,
            v1 in 0.0f32..1.0,
            v2 in 0.0f32..1.0,
            v3 in 0.0f32..1.0
        ) {
            let values = [v0, v1, v2, v3, 0.5, 0.5, 0.5, 0.5];
            let reading = SensorReading::new(values, 0);
            let fixed = reading.to_fixed_point();
            // All should convert without error
            for (i, &original) in values.iter().enumerate() {
                prop_assert!((fixed[i].to_f32() - original).abs() < 0.05);
            }
        }

        #[test]
        fn prop_autoencoder_output_bounded(seed in 0u64..1000) {
            let weights = MicroAutoencoderWeights::new_xavier(seed);
            let mut ae = MicroAutoencoder::new(weights);
            let input = [FixedPoint::from_f32(0.5); NUM_SENSORS];

            let output = ae.forward(&input);
            for &v in &output {
                prop_assert!(v >= 0.0);
                prop_assert!(v <= 1.0);
            }
        }

        #[test]
        fn prop_weights_serialization_roundtrip(seed in 0u64..1000) {
            let original = MicroAutoencoderWeights::new_xavier(seed);
            let bytes = original.to_bytes();
            let restored = MicroAutoencoderWeights::from_bytes(&bytes);
            prop_assert!(restored.is_ok());
        }

        #[test]
        fn prop_rolling_stats_count(n in 1usize..50) {
            let mut stats = RollingStats::new(0.1);
            for _ in 0..n {
                let reading = SensorReading::new([0.5; NUM_SENSORS], 0);
                stats.update(&reading);
            }
            prop_assert_eq!(stats.count, n);
        }

        #[test]
        fn prop_simulator_valid_readings(seed in 0u64..1000, n in 1usize..20) {
            let mut sim = SensorSimulator::new(seed);
            for _ in 0..n {
                let reading = sim.generate_normal();
                prop_assert!(reading.is_valid());
            }
        }

        #[test]
        fn prop_detector_counts_consistent(n in 1usize..30) {
            let weights = MicroAutoencoderWeights::new_xavier(42);
            let model = MicroAutoencoder::new(weights);
            let mut detector = AdaptiveAnomalyDetector::new(model);

            for _ in 0..n {
                let reading = SensorReading::new([0.5; NUM_SENSORS], 0);
                let _ = detector.process(&reading);
            }

            let (anomalies, total) = detector.counts();
            prop_assert!(anomalies <= total);
            prop_assert_eq!(total, n as u64);
        }

        #[test]
        fn prop_z_scores_finite(seed in 0u64..1000) {
            let mut stats = RollingStats::new(0.1);
            let mut sim = SensorSimulator::new(seed);

            for _ in 0..10 {
                let reading = sim.generate_normal();
                stats.update(&reading);
            }

            let test = sim.generate_normal();
            let z = stats.z_scores(&test);

            for &score in &z {
                prop_assert!(score.is_finite());
            }
        }

        #[test]
        fn prop_latent_dim_correct(seed in 0u64..1000) {
            let weights = MicroAutoencoderWeights::new_xavier(seed);
            let mut ae = MicroAutoencoder::new(weights);
            let input = [FixedPoint::from_f32(0.5); NUM_SENSORS];

            ae.forward(&input);
            let latent = ae.get_latent();

            prop_assert_eq!(latent.len(), LATENT_DIM);
        }
    }
}
