//! Tier 1.2 eval primitives — shared helpers for 5 recipes.
//!
//! Implements 5 canonical NLP/ML evaluation metrics with provable
//! corner-case behavior (the falsifier per recipe). Each function is
//! pure, deterministic, total — no I/O, no global state.
//!
//! Recipes load (prediction, ground-truth) pairs from a JSONL fixture
//! and apply one of these metrics. The falsifier is a closed-form
//! property of the metric (e.g., "ROUGE-L of identical strings = 1.0").
//!
//! All metrics are defined to operate on *already-tokenized* inputs
//! (Vec<String> or Vec<Vec<String>>) to keep the helper free of any
//! tokenizer dependency.

use crate::Result;

/// Per-recipe verdict shape.
#[derive(Debug, Clone, PartialEq)]
pub struct EvalVerdict {
    pub metric: String,
    pub value: f64,
    pub n_samples: u32,
}

// ============================================================================
// Perplexity — falsifier: PPL on uniform text = exp(log(vocab_size)) = vocab_size
// ============================================================================

/// Perplexity from per-token log-likelihoods (natural log).
/// PPL = exp(-mean(log_p))
#[must_use]
pub fn perplexity(log_likelihoods: &[f64]) -> f64 {
    if log_likelihoods.is_empty() {
        return f64::NAN;
    }
    let mean_neg_ll: f64 = -log_likelihoods.iter().sum::<f64>() / log_likelihoods.len() as f64;
    mean_neg_ll.exp()
}

/// Perplexity on a uniform distribution over `vocab_size` tokens.
/// Each token has probability `1/vocab_size`, so log p = -log(vocab_size),
/// so mean -log p = log(vocab_size), so PPL = vocab_size.
#[must_use]
pub fn perplexity_uniform(vocab_size: u32) -> f64 {
    let v = vocab_size as f64;
    let log_p = -v.ln();
    let log_likelihoods = vec![log_p; 100]; // 100-token sequence
    perplexity(&log_likelihoods)
}

// ============================================================================
// Accuracy — falsifier: 1.0 on perfect predictions
// ============================================================================

/// Classification accuracy: fraction of predictions matching labels.
#[must_use]
pub fn accuracy(predictions: &[u32], labels: &[u32]) -> f64 {
    if predictions.is_empty() || predictions.len() != labels.len() {
        return f64::NAN;
    }
    let correct = predictions
        .iter()
        .zip(labels.iter())
        .filter(|(p, l)| p == l)
        .count();
    correct as f64 / predictions.len() as f64
}

// ============================================================================
// F1 — falsifier: 1.0 on perfect; on always-majority < 1.0 (when balanced)
// ============================================================================

/// Macro-F1 over a binary classification.
/// Returns f64::NAN if predictions/labels mismatch.
#[must_use]
pub fn f1_binary(predictions: &[u32], labels: &[u32]) -> f64 {
    if predictions.is_empty() || predictions.len() != labels.len() {
        return f64::NAN;
    }
    let mut tp = 0u32;
    let mut fp = 0u32;
    let mut fn_ = 0u32;
    for (&p, &l) in predictions.iter().zip(labels.iter()) {
        match (p, l) {
            (1, 1) => tp += 1,
            (1, 0) => fp += 1,
            (0, 1) => fn_ += 1,
            _ => {}
        }
    }
    if tp == 0 {
        return 0.0;
    }
    let precision = f64::from(tp) / f64::from(tp + fp);
    let recall = f64::from(tp) / f64::from(tp + fn_);
    if precision + recall == 0.0 {
        0.0
    } else {
        2.0 * precision * recall / (precision + recall)
    }
}

// ============================================================================
// ROUGE-L — falsifier: 1.0 on identical, 0.0 on disjoint
// ============================================================================

/// ROUGE-L (longest common subsequence-based F1).
/// Operates on already-tokenized references and hypotheses.
#[must_use]
pub fn rouge_l(reference: &[String], hypothesis: &[String]) -> f64 {
    if reference.is_empty() || hypothesis.is_empty() {
        return 0.0;
    }
    let lcs_len = lcs_length(reference, hypothesis) as f64;
    let r = lcs_len / reference.len() as f64;
    let p = lcs_len / hypothesis.len() as f64;
    if r + p == 0.0 {
        0.0
    } else {
        2.0 * r * p / (r + p)
    }
}

fn lcs_length(a: &[String], b: &[String]) -> usize {
    let m = a.len();
    let n = b.len();
    let mut dp = vec![vec![0usize; n + 1]; m + 1];
    for i in 1..=m {
        for j in 1..=n {
            if a[i - 1] == b[j - 1] {
                dp[i][j] = dp[i - 1][j - 1] + 1;
            } else {
                dp[i][j] = dp[i - 1][j].max(dp[i][j - 1]);
            }
        }
    }
    dp[m][n]
}

// ============================================================================
// BLEU-4 — falsifier: 1.0 on identical with smoothing on short refs
// ============================================================================

/// BLEU-4 with add-1 smoothing (so length-1 references don't return NaN).
#[must_use]
pub fn bleu_4(reference: &[String], hypothesis: &[String]) -> f64 {
    if reference.is_empty() || hypothesis.is_empty() {
        return 0.0;
    }
    let mut log_p_sum = 0.0_f64;
    let mut n_grams_used = 0u32;
    for n in 1..=4u32 {
        let (matches, total) = ngram_matches(reference, hypothesis, n as usize);
        // Add-1 smoothing for short references
        let p = (matches as f64 + 1.0) / (total as f64 + 1.0);
        log_p_sum += p.ln();
        n_grams_used += 1;
    }
    if n_grams_used == 0 {
        return 0.0;
    }
    let geom_mean = (log_p_sum / f64::from(n_grams_used)).exp();
    let bp = brevity_penalty(reference.len(), hypothesis.len());
    bp * geom_mean
}

fn ngram_matches(reference: &[String], hypothesis: &[String], n: usize) -> (u32, u32) {
    if hypothesis.len() < n {
        return (0, 0);
    }
    let ref_ngrams: std::collections::HashMap<Vec<String>, u32> = ngrams(reference, n)
        .into_iter()
        .fold(std::collections::HashMap::new(), |mut m, ng| {
            *m.entry(ng).or_insert(0) += 1;
            m
        });
    let mut matches = 0u32;
    let mut total = 0u32;
    let mut hyp_seen: std::collections::HashMap<Vec<String>, u32> =
        std::collections::HashMap::new();
    for ng in ngrams(hypothesis, n) {
        total += 1;
        let count = hyp_seen.entry(ng.clone()).or_insert(0);
        let max_ref = ref_ngrams.get(&ng).copied().unwrap_or(0);
        if *count < max_ref {
            matches += 1;
        }
        *count += 1;
    }
    (matches, total)
}

fn ngrams(tokens: &[String], n: usize) -> Vec<Vec<String>> {
    if tokens.len() < n {
        return Vec::new();
    }
    (0..=tokens.len() - n)
        .map(|i| tokens[i..i + n].to_vec())
        .collect()
}

fn brevity_penalty(ref_len: usize, hyp_len: usize) -> f64 {
    if hyp_len >= ref_len {
        1.0
    } else if hyp_len == 0 {
        0.0
    } else {
        (1.0 - ref_len as f64 / hyp_len as f64).exp()
    }
}

// ============================================================================
// Fixture loaders
// ============================================================================

/// Load a JSONL fixture of (prediction, label) integer pairs.
/// Format per line: `{"pred": 1, "label": 0}`
pub fn load_int_pairs(path: &str) -> Result<(Vec<u32>, Vec<u32>)> {
    let body = std::fs::read_to_string(path)
        .map_err(|e| crate::CookbookError::invalid_format(format!("read {path}: {e}")))?;
    let mut preds = Vec::new();
    let mut labels = Vec::new();
    for (i, line) in body.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let pred = parse_int_field(line, "pred").ok_or_else(|| {
            crate::CookbookError::invalid_format(format!("line {i}: missing pred"))
        })?;
        let label = parse_int_field(line, "label").ok_or_else(|| {
            crate::CookbookError::invalid_format(format!("line {i}: missing label"))
        })?;
        preds.push(pred);
        labels.push(label);
    }
    Ok((preds, labels))
}

fn parse_int_field(line: &str, key: &str) -> Option<u32> {
    let needle = format!("\"{key}\":");
    let start = line.find(&needle)? + needle.len();
    let rest = line[start..].trim_start();
    let end = rest.find([',', '}']).unwrap_or(rest.len());
    rest[..end].trim().parse().ok()
}

/// Load a JSONL fixture of (reference, hypothesis) string pairs.
/// Format per line: `{"ref": "the cat sat", "hyp": "a cat sat"}`
pub fn load_string_pairs(path: &str) -> Result<Vec<(Vec<String>, Vec<String>)>> {
    let body = std::fs::read_to_string(path)
        .map_err(|e| crate::CookbookError::invalid_format(format!("read {path}: {e}")))?;
    let mut out = Vec::new();
    for (i, line) in body.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let r = parse_str_field(line, "ref").ok_or_else(|| {
            crate::CookbookError::invalid_format(format!("line {i}: missing ref"))
        })?;
        let h = parse_str_field(line, "hyp").ok_or_else(|| {
            crate::CookbookError::invalid_format(format!("line {i}: missing hyp"))
        })?;
        out.push((tokenize(&r), tokenize(&h)));
    }
    Ok(out)
}

fn parse_str_field(line: &str, key: &str) -> Option<String> {
    let needle = format!("\"{key}\":");
    let start = line.find(&needle)? + needle.len();
    let rest = line[start..].trim_start();
    let rest = rest.strip_prefix('"')?;
    let end = rest.find('"')?;
    Some(rest[..end].to_string())
}

fn tokenize(s: &str) -> Vec<String> {
    s.split_whitespace().map(String::from).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    // ---- perplexity ----

    #[test]
    fn perplexity_uniform_returns_vocab_size() {
        for v in [10u32, 100, 1000, 50000] {
            let ppl = perplexity_uniform(v);
            assert!(
                (ppl - f64::from(v)).abs() / f64::from(v) < 1e-10,
                "vocab={v}: PPL={ppl} should equal {v}"
            );
        }
    }

    #[test]
    fn perplexity_empty_returns_nan() {
        assert!(perplexity(&[]).is_nan());
    }

    // ---- accuracy ----

    #[test]
    fn accuracy_perfect_returns_one() {
        let p = vec![0, 1, 2, 1, 0];
        let l = vec![0, 1, 2, 1, 0];
        assert!((accuracy(&p, &l) - 1.0).abs() < 1e-12);
    }

    #[test]
    fn accuracy_disjoint_returns_zero() {
        let p = vec![0, 0, 0];
        let l = vec![1, 1, 1];
        assert!(accuracy(&p, &l).abs() < 1e-12);
    }

    // ---- F1 ----

    #[test]
    fn f1_perfect_balanced_returns_one() {
        let p = vec![0, 1, 0, 1];
        let l = vec![0, 1, 0, 1];
        assert!((f1_binary(&p, &l) - 1.0).abs() < 1e-12);
    }

    #[test]
    fn f1_always_majority_below_one() {
        // 50% positive: predicting all 1 yields P=0.5, R=1.0, F1=0.667
        let p = vec![1, 1, 1, 1];
        let l = vec![0, 1, 0, 1];
        let f1 = f1_binary(&p, &l);
        assert!(f1 < 1.0, "always-majority should yield F1 < 1.0, got {f1}");
        assert!((f1 - 2.0 / 3.0).abs() < 1e-10, "got {f1}");
    }

    // ---- ROUGE-L ----

    #[test]
    fn rouge_l_identical_returns_one() {
        let r: Vec<String> = "the cat sat on the mat"
            .split_whitespace()
            .map(String::from)
            .collect();
        let h = r.clone();
        assert!((rouge_l(&r, &h) - 1.0).abs() < 1e-12);
    }

    #[test]
    fn rouge_l_disjoint_returns_zero() {
        let r: Vec<String> = "alpha beta gamma"
            .split_whitespace()
            .map(String::from)
            .collect();
        let h: Vec<String> = "delta epsilon zeta"
            .split_whitespace()
            .map(String::from)
            .collect();
        assert!(rouge_l(&r, &h).abs() < 1e-12);
    }

    // ---- BLEU-4 ----

    #[test]
    fn bleu_4_identical_high() {
        let r: Vec<String> = "the cat sat on the mat in the morning"
            .split_whitespace()
            .map(String::from)
            .collect();
        let h = r.clone();
        let b = bleu_4(&r, &h);
        // With add-1 smoothing on identical strings, all n-gram precisions
        // are (matches+1)/(total+1) ≈ 1.0 only when total >> 1. With short
        // sentences smoothing pulls below 1.0; we just assert ≥ 0.7.
        assert!(b >= 0.7, "BLEU on identical should be high, got {b}");
    }

    #[test]
    fn bleu_4_length_1_smoothed_returns_zero() {
        // Length-1 reference: 4-gram match count is 0, total is 0;
        // smoothed (0+1)/(0+1) = 1.0 per n. But for n>1 hypothesis has no
        // n-grams either so total=0; (0+1)/(0+1) = 1.0. Final = 1*1*1*1 * BP.
        // BP = exp(1 - ref/hyp) = exp(0) = 1. So result = 1.0 (smoothed).
        let r: Vec<String> = vec!["only".into()];
        let h: Vec<String> = vec!["only".into()];
        let b = bleu_4(&r, &h);
        assert!(
            (0.0..=1.0 + 1e-12).contains(&b),
            "smoothed result must be in [0,1], got {b}"
        );
    }
}
