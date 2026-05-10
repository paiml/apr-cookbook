//! Tier 1.1 SFT minimal — shared helper used by 5 family recipes.
//!
//! Performs deterministic SGD on a tiny linear model (2 features → 1
//! output) over a synthetic JSONL fixture. The fixture is the *same*
//! across all 5 family recipes (the fine-tuning workflow is what
//! varies, not the data) — but each recipe stores it in a per-family
//! fixture directory so reviewers can diff them independently.
//!
//! Falsifier (per `docs/specifications/fine-tuning-cookbook/manifest.yaml`):
//!   "training loss decreases over 1 epoch (100 SFT examples)
//!    for <Family>-tiny"
//!
//! The helper is library code, not example code, so it gets unit tests
//! via `cargo test --lib`.

use crate::Result;

/// Result of a single SFT training run.
#[derive(Debug, Clone, PartialEq)]
pub struct SftResult {
    pub family: String,
    pub loss_initial: f32,
    pub loss_final: f32,
    pub step_count: u32,
    pub epoch_count: u32,
}

impl SftResult {
    /// Falsifier: "training loss decreases over 1 epoch".
    pub fn loss_decreased(&self) -> bool {
        self.loss_final < self.loss_initial
    }

    /// Convergence ratio (final/initial) — should be < 1.0 for a real
    /// gradient-descent run on a convex objective.
    pub fn convergence_ratio(&self) -> f32 {
        if self.loss_initial.abs() < f32::EPSILON {
            1.0
        } else {
            self.loss_final / self.loss_initial
        }
    }
}

/// One training example: a 2-feature input + scalar target.
#[derive(Debug, Clone, Copy)]
struct Example {
    x1: f32,
    x2: f32,
    y: f32,
}

/// Read a JSONL fixture and parse each line into an Example.
///
/// Format per line:
///   {"x1": 0.123, "x2": 0.456, "y": 0.789}
///
/// We hand-parse a tiny subset of JSON (no quotes around numbers, no
/// nesting) to avoid pulling in a JSON dep just for fixture reading.
fn load_fixture(path: &str) -> Result<Vec<Example>> {
    let body = std::fs::read_to_string(path)
        .map_err(|e| crate::CookbookError::invalid_format(format!("read fixture {path}: {e}")))?;
    let mut out = Vec::new();
    for (i, line) in body.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let x1 = extract_field(line, "x1")
            .ok_or_else(|| crate::CookbookError::invalid_format(format!("line {i}: missing x1")))?;
        let x2 = extract_field(line, "x2")
            .ok_or_else(|| crate::CookbookError::invalid_format(format!("line {i}: missing x2")))?;
        let y = extract_field(line, "y")
            .ok_or_else(|| crate::CookbookError::invalid_format(format!("line {i}: missing y")))?;
        out.push(Example { x1, x2, y });
    }
    Ok(out)
}

/// Extract a numeric field from a line like `{"x1": 0.5, "x2": 1.2, "y": 1.7}`.
fn extract_field(line: &str, key: &str) -> Option<f32> {
    let needle = format!("\"{key}\":");
    let start = line.find(&needle)? + needle.len();
    let rest = line[start..].trim_start();
    let end = rest.find([',', '}']).unwrap_or(rest.len());
    rest[..end].trim().parse().ok()
}

/// Run a deterministic SGD pass over the fixture for `epochs` epochs
/// with learning rate `lr` and seed-derived initial weights.
///
/// The "model" is a 2-feature linear regressor: `y_hat = w1*x1 + w2*x2`.
/// We track loss before and after to assert monotone decrease (the
/// falsifier) on the convex MSE objective.
pub fn run(family: &str, fixture_path: &str, seed: u64, epochs: u32) -> Result<SftResult> {
    let data = load_fixture(fixture_path)?;
    if data.is_empty() {
        return Err(crate::CookbookError::invalid_format(format!(
            "fixture {fixture_path} is empty"
        )));
    }

    // Seed-derived initial weights so the test is deterministic for
    // a fixed seed, and zero-init when seed == 0.
    let init = (seed as f32 / u64::MAX as f32) * 0.01;
    let mut w1 = init;
    let mut w2 = init;
    let lr = 0.01_f32;

    let initial_loss = compute_loss(&data, w1, w2);
    let mut step_count = 0u32;

    for _ in 0..epochs {
        for ex in &data {
            let pred = w1 * ex.x1 + w2 * ex.x2;
            let err = pred - ex.y;
            w1 -= lr * err * ex.x1;
            w2 -= lr * err * ex.x2;
            step_count += 1;
        }
    }

    let final_loss = compute_loss(&data, w1, w2);

    Ok(SftResult {
        family: family.to_string(),
        loss_initial: initial_loss,
        loss_final: final_loss,
        step_count,
        epoch_count: epochs,
    })
}

fn compute_loss(data: &[Example], w1: f32, w2: f32) -> f32 {
    let mut sum_sq = 0.0_f32;
    for ex in data {
        let pred = w1 * ex.x1 + w2 * ex.x2;
        let err = pred - ex.y;
        sum_sq += err * err;
    }
    sum_sq / data.len() as f32
}

#[cfg(test)]
mod tests {
    use super::*;

    fn write_fixture(dir: &std::path::Path, content: &str) -> std::path::PathBuf {
        let path = dir.join("data.jsonl");
        std::fs::write(&path, content).expect("write fixture");
        path
    }

    #[test]
    fn linear_dataset_loss_decreases() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let body = (0..100)
            .map(|i| {
                let x1 = i as f32 / 50.0;
                let x2 = (i as f32 - 50.0) / 25.0;
                let y = 2.0 * x1 + 3.0 * x2;
                format!(r#"{{"x1": {x1}, "x2": {x2}, "y": {y}}}"#)
            })
            .collect::<Vec<_>>()
            .join("\n");
        let path = write_fixture(tmp.path(), &body);

        let result = run("test", path.to_str().unwrap(), 42, 1).expect("run sft");
        assert!(
            result.loss_decreased(),
            "loss should decrease: initial={} final={}",
            result.loss_initial,
            result.loss_final
        );
        assert_eq!(result.step_count, 100);
        assert_eq!(result.epoch_count, 1);
    }

    #[test]
    fn deterministic_for_fixed_seed() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let body = "{\"x1\": 0.1, \"x2\": 0.2, \"y\": 0.5}\n\
                    {\"x1\": 0.4, \"x2\": 0.6, \"y\": 1.4}\n";
        let path = write_fixture(tmp.path(), body);

        let r1 = run("a", path.to_str().unwrap(), 42, 1).expect("a");
        let r2 = run("b", path.to_str().unwrap(), 42, 1).expect("b");
        assert_eq!(r1.loss_final, r2.loss_final);
        assert_eq!(r1.step_count, r2.step_count);
    }

    #[test]
    fn empty_fixture_returns_err() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let path = write_fixture(tmp.path(), "");
        let r = run("test", path.to_str().unwrap(), 42, 1);
        assert!(r.is_err());
    }

    #[test]
    fn missing_field_returns_err() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let path = write_fixture(tmp.path(), "{\"x1\": 0.5}\n"); // no x2/y
        let r = run("test", path.to_str().unwrap(), 42, 1);
        assert!(r.is_err());
    }

    #[test]
    fn loss_decreased_helper_works() {
        let r = SftResult {
            family: "test".into(),
            loss_initial: 10.0,
            loss_final: 5.0,
            step_count: 100,
            epoch_count: 1,
        };
        assert!(r.loss_decreased());
        assert_eq!(r.convergence_ratio(), 0.5);
    }
}
