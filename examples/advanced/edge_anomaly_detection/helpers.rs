#![allow(
    dead_code,
    unused_imports,
    unreachable_pub,
    clippy::trivially_copy_pass_by_ref,
    clippy::wildcard_imports
)]
#[allow(unused_imports, clippy::wildcard_imports)]
use super::types::*;

impl MicroAutoencoderWeights {
    pub fn init_layer<const R: usize, const C: usize>(
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

impl MicroAutoencoder {
    pub fn encode(&mut self, input: &[FixedPoint; NUM_SENSORS]) {
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

    pub fn decode(&mut self) {
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
}

impl AdaptiveAnomalyDetector {
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
}

impl SensorSimulator {
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
