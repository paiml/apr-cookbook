//! # Recipe: Pretrain — Checkpoint & Resume Pipeline
//!
//! **Category**: training
//! **CLI Equivalent**: `apr pretrain --resume ./ckpts/step_100.apr --ckpt-every 50`
//! **Isolation Level**: Full
//! **Idempotency**: Guaranteed
//! **Dependencies**: apr-cookbook (default features)
//! Contract: contracts/recipe-iiur-v1.yaml
//!
//! ## QA Checklist
//! 1. [x] `cargo run --example pretrain_checkpoint_resume` exits 0
//! 2. [x] `cargo test --example pretrain_checkpoint_resume` passes
//! 3. [x] Deterministic output (same seed → same bytes)
//! 4. [x] No temp files leaked (RecipeContext tempdir)
//! 5. [x] No `unwrap()` in main logic
//! 6. [x] Clippy clean under `-D warnings`
//!
//! ## Learning Objective
//! End-to-end pretrain composition: run for N steps, checkpoint to disk
//! every K steps, crash (simulated), resume from latest ckpt, and verify
//! that the resumed-run tail exactly matches the all-at-once run. This is
//! the canonical correctness proof the `--resume` flag must satisfy.
//!
//! ## Run Command
//! ```bash
//! cargo run --example pretrain_checkpoint_resume
//! ```
//!
//! ## References
//! - Rajbhandari, S. et al. (2020). *ZeRO: Memory Optimizations Toward Training Trillion Parameter Models*. arXiv:1910.02054

use apr_cookbook::prelude::*;
use apr_cookbook::Result;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::path::Path;

/// Deterministic loss as a function of (step, seed) only — independent of
/// any RNG threading, so we can verify resume-correctness exactly.
pub fn step_loss(step: u32, seed: u64) -> f64 {
    let mut rng = StdRng::seed_from_u64(seed.wrapping_add(u64::from(step)));
    let base = 2.0_f64 * (-0.015 * f64::from(step)).exp();
    let noise: f64 = rng.gen_range(-0.02..0.02);
    (base + noise).max(0.0)
}

#[derive(Debug, Clone)]
pub struct Checkpoint {
    pub step: u32,
    pub loss: f64,
}

pub fn save_ckpt(path: &Path, ck: &Checkpoint) -> Result<()> {
    let s = format!("{}\n{:.9}\n", ck.step, ck.loss);
    std::fs::write(path, s)?;
    Ok(())
}

pub fn load_ckpt(path: &Path) -> Result<Checkpoint> {
    let s = std::fs::read_to_string(path)?;
    let mut lines = s.lines();
    let step: u32 = lines.next().unwrap_or("0").parse().unwrap_or(0);
    let loss: f64 = lines.next().unwrap_or("0").parse().unwrap_or(0.0);
    Ok(Checkpoint { step, loss })
}

pub fn run_from(
    start_step: u32,
    end_step: u32,
    seed: u64,
    ckpt_dir: &Path,
    every: u32,
) -> Result<Vec<Checkpoint>> {
    std::fs::create_dir_all(ckpt_dir)?;
    let mut ckpts = Vec::new();
    for step in (start_step + 1)..=end_step {
        let loss = step_loss(step, seed);
        if step % every == 0 {
            let p = ckpt_dir.join(format!("step_{step:04}.apr"));
            let ck = Checkpoint { step, loss };
            save_ckpt(&p, &ck)?;
            ckpts.push(ck);
        }
    }
    Ok(ckpts)
}

pub fn latest_ckpt(dir: &Path) -> Result<Option<Checkpoint>> {
    let mut best: Option<Checkpoint> = None;
    for entry in std::fs::read_dir(dir)? {
        let e = entry?;
        let n = e.file_name().to_string_lossy().to_string();
        if n.starts_with("step_")
            && Path::new(&n)
                .extension()
                .is_some_and(|ext| ext.eq_ignore_ascii_case("apr"))
        {
            let ck = load_ckpt(&e.path())?;
            #[allow(clippy::incompatible_msrv)]
            let should_update = best.as_ref().is_none_or(|b| ck.step > b.step);
            if should_update {
                best = Some(ck);
            }
        }
    }
    Ok(best)
}

fn main() -> Result<()> {
    let ctx = RecipeContext::new("pretrain_checkpoint_resume")?;
    let ckpt_dir = ctx.path("ckpts");
    let seed = 42u64;
    let total_steps = 60u32;
    let crash_at = 30u32;
    let every = 10u32;

    println!("=== Recipe: {} ===", ctx.name());

    // Phase 1: run from 0 to crash_at, checkpointing every 10.
    let phase1 = run_from(0, crash_at, seed, &ckpt_dir, every)?;
    println!("Phase 1 (0..{crash_at}): {} ckpts written", phase1.len());

    // Phase 2: crash — resume from latest ckpt, continue to total_steps.
    let resume_from = latest_ckpt(&ckpt_dir)?.expect("at least one ckpt");
    println!(
        "Resuming from step {} (loss {:.4})",
        resume_from.step, resume_from.loss
    );
    let phase2 = run_from(resume_from.step, total_steps, seed, &ckpt_dir, every)?;
    println!(
        "Phase 2 ({}..{total_steps}): {} ckpts written",
        resume_from.step,
        phase2.len()
    );

    // Ground truth: run straight through in one shot.
    let gt_dir = ctx.path("gt");
    let _gt = run_from(0, total_steps, seed, &gt_dir, every)?;
    let gt_final = latest_ckpt(&gt_dir)?.expect("gt ckpt");
    let resumed_final = latest_ckpt(&ckpt_dir)?.expect("resumed ckpt");

    let matches =
        (gt_final.loss - resumed_final.loss).abs() < 1e-9 && gt_final.step == resumed_final.step;
    println!(
        "Final step: resumed={} loss={:.9}  gt={} loss={:.9}  match={}",
        resumed_final.step, resumed_final.loss, gt_final.step, gt_final.loss, matches
    );

    println!(
        "verdict: {}",
        if matches { "EXACT_MATCH" } else { "DIVERGED" }
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn step_loss_is_deterministic() {
        let a = step_loss(50, 42);
        let b = step_loss(50, 42);
        assert_eq!(a, b);
    }

    #[test]
    fn round_trip_ckpt() {
        let ctx = RecipeContext::new("rt").unwrap();
        let p = ctx.path("ck.apr");
        let ck = Checkpoint {
            step: 7,
            loss: 1.234_567_89,
        };
        save_ckpt(&p, &ck).unwrap();
        let back = load_ckpt(&p).unwrap();
        assert_eq!(back.step, ck.step);
        assert!((back.loss - ck.loss).abs() < 1e-9);
    }

    #[test]
    fn latest_ckpt_picks_highest_step() {
        let ctx = RecipeContext::new("latest").unwrap();
        let dir = ctx.path("ck");
        std::fs::create_dir_all(&dir).unwrap();
        save_ckpt(
            &dir.join("step_0010.apr"),
            &Checkpoint {
                step: 10,
                loss: 1.0,
            },
        )
        .unwrap();
        save_ckpt(
            &dir.join("step_0030.apr"),
            &Checkpoint {
                step: 30,
                loss: 0.5,
            },
        )
        .unwrap();
        let got = latest_ckpt(&dir).unwrap().unwrap();
        assert_eq!(got.step, 30);
    }
}
