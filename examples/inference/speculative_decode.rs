//! Speculative Decoding Example
//!
//! Demonstrates speculative decoding: a small "draft" model proposes K candidate
//! tokens, then a larger "target" model verifies them in a single forward pass.
//! Accepted tokens are emitted instantly; rejected tokens trigger resampling.
//!
//! # Algorithm
//!
//! ```text
//! ┌──────────────────────────────────────────────────────────────────┐
//! │                    Speculative Decoding                          │
//! ├──────────────────────────────────────────────────────────────────┤
//! │  1. Draft model generates K candidate tokens (fast)              │
//! │  2. Target model scores all K tokens in one pass (slow but once) │
//! │  3. Compare draft vs target distributions per position           │
//! │  4. Accept longest prefix where draft agrees with target         │
//! │  5. Resample correction token from adjusted distribution         │
//! │  6. Effective speedup = (accepted_tokens + 1) / 1                │
//! └──────────────────────────────────────────────────────────────────┘
//! ```
//!
//! # Running
//!
//! ```bash
//! cargo run --example speculative_decode
//! ```

use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

const VOCAB_SIZE: usize = 64;
const DRAFT_LOOKAHEAD: usize = 5;

/// A simple language model that produces next-token probability distributions.
/// Uses deterministic hashing to simulate learned distributions.
struct LanguageModel {
    /// Model quality: higher = sharper distributions around "correct" tokens
    quality: f32,
    /// Number of hidden dimensions (affects "compute cost")
    hidden_dim: usize,
    /// Seed for deterministic behavior
    seed: u64,
    /// Simulated forward pass count
    forward_count: usize,
}

impl LanguageModel {
    fn new(quality: f32, hidden_dim: usize, seed: u64) -> Self {
        Self {
            quality,
            hidden_dim,
            seed,
            forward_count: 0,
        }
    }

    /// Compute next-token probability distribution given context.
    /// Returns a softmax distribution over VOCAB_SIZE tokens.
    fn forward(&mut self, context: &[u32]) -> Vec<f64> {
        self.forward_count += 1;
        let mut logits = vec![0.0f64; VOCAB_SIZE];

        // Generate logits deterministically from context
        for (i, logit) in logits.iter_mut().enumerate() {
            let mut hasher = DefaultHasher::new();
            (self.seed, context, i).hash(&mut hasher);
            let h = hasher.finish();
            let base = (h as f64 / u64::MAX as f64 - 0.5) * 2.0;
            *logit = base * f64::from(self.quality);
        }

        // Softmax
        let max_logit = logits.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let exp_sum: f64 = logits.iter().map(|l| (l - max_logit).exp()).sum();
        logits
            .iter()
            .map(|l| (l - max_logit).exp() / exp_sum)
            .collect()
    }

    /// Batch forward: score multiple positions at once (the key optimization).
    /// For speculative decoding, the target model scores K draft tokens in one call.
    fn batch_forward(&mut self, contexts: &[Vec<u32>]) -> Vec<Vec<f64>> {
        // In reality this would be a single batched GPU call.
        // We simulate it as one "forward pass" regardless of batch size.
        self.forward_count += 1;
        contexts
            .iter()
            .map(|ctx| {
                let mut logits = vec![0.0f64; VOCAB_SIZE];
                for (i, logit) in logits.iter_mut().enumerate() {
                    let mut hasher = DefaultHasher::new();
                    (self.seed, ctx.as_slice(), i).hash(&mut hasher);
                    let h = hasher.finish();
                    let base = (h as f64 / u64::MAX as f64 - 0.5) * 2.0;
                    *logit = base * f64::from(self.quality);
                }
                let max_logit = logits.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                let exp_sum: f64 = logits.iter().map(|l| (l - max_logit).exp()).sum();
                logits
                    .iter()
                    .map(|l| (l - max_logit).exp() / exp_sum)
                    .collect()
            })
            .collect()
    }

    /// Sample a token from a probability distribution using deterministic seed
    fn sample(&self, probs: &[f64], step: usize) -> u32 {
        let mut hasher = DefaultHasher::new();
        (self.seed, "sample", step, self.forward_count).hash(&mut hasher);
        let r = hasher.finish() as f64 / u64::MAX as f64;

        let mut cumulative = 0.0;
        for (i, &p) in probs.iter().enumerate() {
            cumulative += p;
            if r < cumulative {
                return i as u32;
            }
        }
        (VOCAB_SIZE - 1) as u32
    }

    fn compute_cost(&self) -> usize {
        self.hidden_dim * self.hidden_dim
    }
}

/// Standard autoregressive decoding (baseline)
fn standard_decode(model: &mut LanguageModel, prompt: &[u32], max_tokens: usize) -> Vec<u32> {
    let mut tokens = prompt.to_vec();
    for step in 0..max_tokens {
        let probs = model.forward(&tokens);
        let next = model.sample(&probs, step);
        tokens.push(next);
    }
    tokens
}

/// Speculative decoding with draft + target models
fn speculative_decode(
    draft: &mut LanguageModel,
    target: &mut LanguageModel,
    prompt: &[u32],
    max_tokens: usize,
    lookahead: usize,
) -> (Vec<u32>, SpecStats) {
    let mut tokens = prompt.to_vec();
    let mut stats = SpecStats::default();
    let mut generated = 0;

    while generated < max_tokens {
        let remaining = max_tokens - generated;
        let k = lookahead.min(remaining);

        // Step 1: Draft model generates K candidate tokens
        let mut draft_tokens = Vec::with_capacity(k);
        let mut draft_probs = Vec::with_capacity(k);
        let mut draft_ctx = tokens.clone();

        for step in 0..k {
            let probs = draft.forward(&draft_ctx);
            let token = draft.sample(&probs, generated + step);
            draft_tokens.push(token);
            draft_probs.push(probs);
            draft_ctx.push(token);
        }

        // Step 2: Target model scores all K positions in one batch call
        let mut contexts = Vec::with_capacity(k);
        let mut ctx = tokens.clone();
        for &dt in &draft_tokens {
            contexts.push(ctx.clone());
            ctx.push(dt);
        }
        let target_probs_batch = target.batch_forward(&contexts);

        // Step 3: Accept/reject using probability ratio
        let mut accepted = 0;
        for i in 0..k {
            let draft_p = draft_probs[i][draft_tokens[i] as usize];
            let target_p = target_probs_batch[i][draft_tokens[i] as usize];

            // Accept if target probability >= draft probability
            // (simplified rejection sampling)
            if target_p >= draft_p * 0.8 {
                tokens.push(draft_tokens[i]);
                accepted += 1;
            } else {
                // Reject: sample from adjusted distribution
                let correction: Vec<f64> = target_probs_batch[i]
                    .iter()
                    .zip(draft_probs[i].iter())
                    .map(|(&tp, &dp)| (tp - 0.8 * dp).max(0.0))
                    .collect();
                let sum: f64 = correction.iter().sum();
                if sum > 0.0 {
                    let normalized: Vec<f64> = correction.iter().map(|&c| c / sum).collect();
                    let corrected_token = target.sample(&normalized, generated + i);
                    tokens.push(corrected_token);
                } else {
                    let token = target.sample(&target_probs_batch[i], generated + i);
                    tokens.push(token);
                }
                accepted += 1; // We still produce one token from the correction
                break; // Stop accepting after first rejection
            }
        }

        generated += accepted;
        stats.total_draft_tokens += k;
        stats.total_accepted += accepted;
        stats.rounds += 1;
    }

    stats.draft_forwards = draft.forward_count;
    stats.target_forwards = target.forward_count;
    (tokens, stats)
}

/// Statistics from speculative decoding
#[derive(Default, Debug)]
struct SpecStats {
    total_draft_tokens: usize,
    total_accepted: usize,
    rounds: usize,
    draft_forwards: usize,
    target_forwards: usize,
}

impl SpecStats {
    fn acceptance_rate(&self) -> f64 {
        if self.total_draft_tokens == 0 {
            return 0.0;
        }
        self.total_accepted as f64 / self.total_draft_tokens as f64
    }

    fn avg_accepted_per_round(&self) -> f64 {
        if self.rounds == 0 {
            return 0.0;
        }
        self.total_accepted as f64 / self.rounds as f64
    }
}

fn main() {
    println!("=== Speculative Decoding Example ===\n");

    let prompt: Vec<u32> = vec![1, 5, 12, 8]; // Start tokens
    let max_tokens = 50;

    // =========================================================================
    // Section 1: Baseline (standard autoregressive)
    // =========================================================================
    println!("1. Baseline: Standard Autoregressive Decoding");
    println!("   ─────────────────────────────────────────");

    let mut baseline_model = LanguageModel::new(3.0, 512, 42);
    let baseline_tokens = standard_decode(&mut baseline_model, &prompt, max_tokens);
    let baseline_cost = baseline_model.forward_count * baseline_model.compute_cost();

    println!("   Generated {} tokens", max_tokens);
    println!("   Forward passes: {}", baseline_model.forward_count);
    println!(
        "   Compute cost:   {} ({}x{} per pass)",
        baseline_cost,
        baseline_model.forward_count,
        baseline_model.compute_cost()
    );
    println!(
        "   Tokens: {:?}...",
        &baseline_tokens[prompt.len()..prompt.len() + 10]
    );
    println!();

    // =========================================================================
    // Section 2: Speculative Decoding
    // =========================================================================
    println!("2. Speculative Decoding (K={})", DRAFT_LOOKAHEAD);
    println!("   ─────────────────────────────────────────");

    let mut draft = LanguageModel::new(2.0, 64, 42); // Small, fast
    let mut target = LanguageModel::new(3.0, 512, 42); // Large, slow

    let (spec_tokens, stats) = speculative_decode(
        &mut draft,
        &mut target,
        &prompt,
        max_tokens,
        DRAFT_LOOKAHEAD,
    );

    let draft_cost = stats.draft_forwards * draft.compute_cost();
    let target_cost = stats.target_forwards * target.compute_cost();
    let total_spec_cost = draft_cost + target_cost;

    println!("   Generated {} tokens", spec_tokens.len() - prompt.len());
    println!("   Rounds:          {}", stats.rounds);
    println!(
        "   Acceptance rate:  {:.1}%",
        stats.acceptance_rate() * 100.0
    );
    println!(
        "   Avg accepted/round: {:.1}",
        stats.avg_accepted_per_round()
    );
    println!(
        "   Draft forwards:  {} (cost/pass: {})",
        stats.draft_forwards,
        draft.compute_cost()
    );
    println!(
        "   Target forwards: {} (cost/pass: {})",
        stats.target_forwards,
        target.compute_cost()
    );
    println!("   Total compute:   {}", total_spec_cost);
    println!(
        "   Speedup vs baseline: {:.2}x",
        baseline_cost as f64 / total_spec_cost as f64
    );
    println!();

    // =========================================================================
    // Section 3: Sweep Lookahead Values
    // =========================================================================
    println!("3. Lookahead Sweep");
    println!("   ─────────────────────────────────────────");
    println!(
        "   {:>4} {:>10} {:>12} {:>10} {:>10}",
        "K", "Accept%", "Rounds", "Compute", "Speedup"
    );
    println!("   {}", "─".repeat(50));

    for k in [1, 2, 3, 5, 8, 12] {
        let mut d = LanguageModel::new(2.0, 64, 42);
        let mut t = LanguageModel::new(3.0, 512, 42);
        let (_, s) = speculative_decode(&mut d, &mut t, &prompt, max_tokens, k);
        let cost = s.draft_forwards * d.compute_cost() + s.target_forwards * t.compute_cost();
        let speedup = baseline_cost as f64 / cost as f64;
        println!(
            "   {:>4} {:>9.1}% {:>12} {:>10} {:>9.2}x",
            k,
            s.acceptance_rate() * 100.0,
            s.rounds,
            cost,
            speedup,
        );
    }
    println!();

    // =========================================================================
    // Section 4: Draft Quality Impact
    // =========================================================================
    println!("4. Draft Model Quality Impact");
    println!("   ─────────────────────────────────────────");
    println!(
        "   {:>10} {:>10} {:>12} {:>10}",
        "Quality", "Accept%", "Rounds", "Speedup"
    );
    println!("   {}", "─".repeat(45));

    for quality in [0.5, 1.0, 2.0, 2.8, 3.0] {
        let mut d = LanguageModel::new(quality, 64, 42);
        let mut t = LanguageModel::new(3.0, 512, 42);
        let (_, s) = speculative_decode(&mut d, &mut t, &prompt, max_tokens, DRAFT_LOOKAHEAD);
        let cost = s.draft_forwards * d.compute_cost() + s.target_forwards * t.compute_cost();
        let speedup = baseline_cost as f64 / cost as f64;
        println!(
            "   {:>10.1} {:>9.1}% {:>12} {:>9.2}x",
            quality,
            s.acceptance_rate() * 100.0,
            s.rounds,
            speedup,
        );
    }
    println!();

    println!("=== Example Complete ===");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_language_model_softmax() {
        let mut model = LanguageModel::new(2.0, 64, 42);
        let probs = model.forward(&[1, 2, 3]);
        let sum: f64 = probs.iter().sum();
        assert!(
            (sum - 1.0).abs() < 1e-10,
            "Softmax should sum to 1, got {}",
            sum
        );
        assert!(probs.iter().all(|&p| p >= 0.0));
    }

    #[test]
    fn test_language_model_deterministic() {
        let mut m1 = LanguageModel::new(2.0, 64, 42);
        let mut m2 = LanguageModel::new(2.0, 64, 42);
        let p1 = m1.forward(&[1, 2, 3]);
        let p2 = m2.forward(&[1, 2, 3]);
        assert_eq!(p1, p2);
    }

    #[test]
    fn test_batch_forward_matches_single() {
        let ctx = vec![1u32, 2, 3];
        let mut m1 = LanguageModel::new(2.0, 64, 42);
        let single = m1.forward(&ctx);

        let mut m2 = LanguageModel::new(2.0, 64, 42);
        let batch = m2.batch_forward(&[ctx]);
        assert_eq!(single.len(), batch[0].len());
        for (a, b) in single.iter().zip(batch[0].iter()) {
            assert!((a - b).abs() < 1e-10);
        }
    }

    #[test]
    fn test_standard_decode_length() {
        let mut model = LanguageModel::new(2.0, 64, 42);
        let result = standard_decode(&mut model, &[1, 2], 20);
        assert_eq!(result.len(), 22); // prompt(2) + generated(20)
    }

    #[test]
    fn test_speculative_decode_generates_tokens() {
        let mut draft = LanguageModel::new(2.0, 64, 42);
        let mut target = LanguageModel::new(3.0, 512, 42);
        let (tokens, stats) = speculative_decode(&mut draft, &mut target, &[1, 2], 20, 5);
        assert!(tokens.len() >= 22, "Should generate at least 20 tokens");
        assert!(stats.rounds > 0);
        assert!(stats.total_accepted > 0);
    }

    #[test]
    fn test_acceptance_rate_bounded() {
        let mut draft = LanguageModel::new(2.0, 64, 42);
        let mut target = LanguageModel::new(3.0, 512, 42);
        let (_, stats) = speculative_decode(&mut draft, &mut target, &[1, 2], 50, 5);
        let rate = stats.acceptance_rate();
        assert!(rate >= 0.0 && rate <= 1.0);
    }

    #[test]
    fn test_speculative_fewer_target_forwards() {
        let mut baseline = LanguageModel::new(3.0, 512, 42);
        let _ = standard_decode(&mut baseline, &[1, 2], 30);
        let baseline_forwards = baseline.forward_count;

        let mut draft = LanguageModel::new(2.0, 64, 42);
        let mut target = LanguageModel::new(3.0, 512, 42);
        let (_, stats) = speculative_decode(&mut draft, &mut target, &[1, 2], 30, 5);

        // Target model should need fewer forwards than baseline
        assert!(
            stats.target_forwards < baseline_forwards,
            "Target forwards {} should be less than baseline {}",
            stats.target_forwards,
            baseline_forwards
        );
    }

    #[test]
    fn test_higher_quality_draft_better_acceptance() {
        let mut d_low = LanguageModel::new(0.5, 64, 42);
        let mut t_low = LanguageModel::new(3.0, 512, 42);
        let (_, stats_low) = speculative_decode(&mut d_low, &mut t_low, &[1, 2], 30, 5);

        let mut d_high = LanguageModel::new(2.8, 64, 42);
        let mut t_high = LanguageModel::new(3.0, 512, 42);
        let (_, stats_high) = speculative_decode(&mut d_high, &mut t_high, &[1, 2], 30, 5);

        assert!(
            stats_high.acceptance_rate() >= stats_low.acceptance_rate(),
            "Higher quality draft should have higher acceptance rate"
        );
    }
}
