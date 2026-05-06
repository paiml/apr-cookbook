//! # Distillation LR Schedule Match
//!
//! Compute student LR schedule that mirrors teacher's: teacher peak
//! LR and warmup steps scaled by the train-step ratio. Common pattern:
//! cosine schedule with warmup; student uses same shape, scaled steps.
//!
//! Demonstrates the **DIST.35** recipe for PMAT-157 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: Loshchilov & Hutter (2017) SGDR cosine schedule.
//!
//! Run with: cargo run --example distill_lr_schedule_match
//!
//! Added by PMAT-157 (catalog 1036→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum ScheduleVerdict {
    Ok {
        student_peak_lr: f64,
        student_warmup_steps: u32,
        student_total_steps: u32,
    },
    InvalidConfig,
}

pub fn pick(
    teacher_peak_lr: f64,
    teacher_warmup_steps: u32,
    teacher_total_steps: u32,
    student_total_steps: u32,
) -> ScheduleVerdict {
    if !teacher_peak_lr.is_finite()
        || teacher_peak_lr <= 0.0
        || teacher_total_steps == 0
        || student_total_steps == 0
        || teacher_warmup_steps > teacher_total_steps
    {
        return ScheduleVerdict::InvalidConfig;
    }
    let ratio = f64::from(student_total_steps) / f64::from(teacher_total_steps);
    let student_warmup_steps = (f64::from(teacher_warmup_steps) * ratio).round() as u32;
    // Smaller students benefit from slightly higher peak LR.
    let lr_multiplier = match student_total_steps.cmp(&teacher_total_steps) {
        std::cmp::Ordering::Less => 1.5,
        std::cmp::Ordering::Equal => 1.0,
        std::cmp::Ordering::Greater => 0.75,
    };
    ScheduleVerdict::Ok {
        student_peak_lr: teacher_peak_lr * lr_multiplier,
        student_warmup_steps,
        student_total_steps,
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("distill_lr_schedule_match")?;

    println!("smaller student: {:?}", pick(1e-4, 1000, 100_000, 50_000));
    println!("same size: {:?}", pick(1e-4, 1000, 100_000, 100_000));
    println!("longer student: {:?}", pick(1e-4, 1000, 100_000, 200_000));
    println!("invalid: {:?}", pick(1e-4, 0, 0, 1000));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn picker_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn smaller_student_higher_lr() {
        let v = pick(1e-4, 1000, 100_000, 50_000);
        if let ScheduleVerdict::Ok {
            student_peak_lr, ..
        } = v
        {
            // Smaller student → 1.5× LR.
            assert!((student_peak_lr - 1.5e-4).abs() < 1e-12);
        }
    }

    #[test]
    fn same_size_same_lr() {
        let v = pick(1e-4, 1000, 100_000, 100_000);
        if let ScheduleVerdict::Ok {
            student_peak_lr, ..
        } = v
        {
            assert!((student_peak_lr - 1e-4).abs() < 1e-12);
        }
    }

    #[test]
    fn longer_student_lower_lr() {
        let v = pick(1e-4, 1000, 100_000, 200_000);
        if let ScheduleVerdict::Ok {
            student_peak_lr, ..
        } = v
        {
            assert!((student_peak_lr - 0.75e-4).abs() < 1e-12);
        }
    }

    #[test]
    fn warmup_scaled_proportionally() {
        let v = pick(1e-4, 1000, 100_000, 50_000);
        if let ScheduleVerdict::Ok {
            student_warmup_steps,
            ..
        } = v
        {
            // 1000 × 0.5 = 500.
            assert_eq!(student_warmup_steps, 500);
        }
    }

    #[test]
    fn invalid_zero_steps() {
        assert_eq!(pick(1e-4, 0, 0, 1000), ScheduleVerdict::InvalidConfig);
    }

    #[test]
    fn invalid_zero_student_steps() {
        assert_eq!(pick(1e-4, 100, 1000, 0), ScheduleVerdict::InvalidConfig);
    }

    #[test]
    fn invalid_negative_lr() {
        assert_eq!(pick(-1e-4, 100, 1000, 1000), ScheduleVerdict::InvalidConfig);
    }

    #[test]
    fn warmup_exceeds_total_invalid() {
        assert_eq!(pick(1e-4, 2000, 1000, 1000), ScheduleVerdict::InvalidConfig);
    }

    #[test]
    fn nan_lr_invalid() {
        assert_eq!(
            pick(f64::NAN, 100, 1000, 1000),
            ScheduleVerdict::InvalidConfig
        );
    }

    #[test]
    fn deterministic() {
        let a = pick(1e-4, 1000, 100_000, 50_000);
        let b = pick(1e-4, 1000, 100_000, 50_000);
        assert_eq!(a, b);
    }
}
