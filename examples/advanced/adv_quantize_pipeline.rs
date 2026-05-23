//! # Advanced Quantization Pipeline Planner
//!
//! Quantization workflow: Calibrate → Quantize → Validate → Deploy.
//! Each step has prerequisites:
//!   Calibrate needs raw model + calibration dataset
//!   Quantize needs calibration result + target precision
//!   Validate needs quantized model + eval dataset
//!   Deploy needs validate-passed marker
//!
//! Picker reports next step + missing prerequisites if blocked.
//!
//! Demonstrates the **ADV.13** recipe for PMAT-142 (advanced round 5).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: PyTorch quantization workflow docs (post-training quantization).
//!
//! Run with: cargo run --example adv_quantize_pipeline
//!
//! Added by PMAT-142 (expand-cookbooks followup).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Step {
    Calibrate,
    Quantize,
    Validate,
    Deploy,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct PipelineState {
    pub raw_model_loaded: bool,
    pub calibration_dataset_ready: bool,
    pub eval_dataset_ready: bool,
    pub calibrated: bool,
    pub quantized: bool,
    pub validated: bool,
    pub deployed: bool,
}

#[derive(Debug, PartialEq)]
pub enum NextStepVerdict {
    DoNext {
        step: Step,
    },
    Complete,
    Blocked {
        step: Step,
        missing: Vec<&'static str>,
    },
}

pub fn next_step(state: PipelineState) -> NextStepVerdict {
    if !state.calibrated {
        let mut missing = Vec::new();
        if !state.raw_model_loaded {
            missing.push("raw_model_loaded");
        }
        if !state.calibration_dataset_ready {
            missing.push("calibration_dataset_ready");
        }
        if !missing.is_empty() {
            return NextStepVerdict::Blocked {
                step: Step::Calibrate,
                missing,
            };
        }
        return NextStepVerdict::DoNext {
            step: Step::Calibrate,
        };
    }
    if !state.quantized {
        return NextStepVerdict::DoNext {
            step: Step::Quantize,
        };
    }
    if !state.validated {
        let mut missing = Vec::new();
        if !state.eval_dataset_ready {
            missing.push("eval_dataset_ready");
        }
        if !missing.is_empty() {
            return NextStepVerdict::Blocked {
                step: Step::Validate,
                missing,
            };
        }
        return NextStepVerdict::DoNext {
            step: Step::Validate,
        };
    }
    if !state.deployed {
        return NextStepVerdict::DoNext { step: Step::Deploy };
    }
    NextStepVerdict::Complete
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("adv_quantize_pipeline")?;

    let fresh = PipelineState {
        raw_model_loaded: true,
        calibration_dataset_ready: true,
        eval_dataset_ready: true,
        ..Default::default()
    };
    println!("fresh: {:?}", next_step(fresh));

    let after_calibrate = PipelineState {
        calibrated: true,
        ..fresh
    };
    println!("after calibrate: {:?}", next_step(after_calibrate));

    let blocked = PipelineState::default();
    println!("blocked: {:?}", next_step(blocked));

    let done = PipelineState {
        raw_model_loaded: true,
        calibration_dataset_ready: true,
        eval_dataset_ready: true,
        calibrated: true,
        quantized: true,
        validated: true,
        deployed: true,
    };
    println!("done: {:?}", next_step(done));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ready() -> PipelineState {
        PipelineState {
            raw_model_loaded: true,
            calibration_dataset_ready: true,
            eval_dataset_ready: true,
            ..Default::default()
        }
    }

    #[test]
    fn pipeline_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn fresh_state_does_calibrate() {
        let v = next_step(ready());
        assert_eq!(
            v,
            NextStepVerdict::DoNext {
                step: Step::Calibrate
            }
        );
    }

    #[test]
    fn after_calibrate_does_quantize() {
        let s = PipelineState {
            calibrated: true,
            ..ready()
        };
        assert_eq!(
            next_step(s),
            NextStepVerdict::DoNext {
                step: Step::Quantize
            }
        );
    }

    #[test]
    fn after_quantize_does_validate() {
        let s = PipelineState {
            calibrated: true,
            quantized: true,
            ..ready()
        };
        assert_eq!(
            next_step(s),
            NextStepVerdict::DoNext {
                step: Step::Validate
            }
        );
    }

    #[test]
    fn after_validate_does_deploy() {
        let s = PipelineState {
            calibrated: true,
            quantized: true,
            validated: true,
            ..ready()
        };
        assert_eq!(next_step(s), NextStepVerdict::DoNext { step: Step::Deploy });
    }

    #[test]
    fn fully_done_is_complete() {
        let s = PipelineState {
            raw_model_loaded: true,
            calibration_dataset_ready: true,
            eval_dataset_ready: true,
            calibrated: true,
            quantized: true,
            validated: true,
            deployed: true,
        };
        assert_eq!(next_step(s), NextStepVerdict::Complete);
    }

    #[test]
    fn no_raw_model_blocks_calibrate() {
        let s = PipelineState::default();
        if let NextStepVerdict::Blocked { step, missing } = next_step(s) {
            assert_eq!(step, Step::Calibrate);
            assert!(missing.contains(&"raw_model_loaded"));
        }
    }

    #[test]
    fn no_calibration_dataset_blocks_calibrate() {
        let s = PipelineState {
            raw_model_loaded: true,
            ..Default::default()
        };
        if let NextStepVerdict::Blocked { missing, .. } = next_step(s) {
            assert!(missing.contains(&"calibration_dataset_ready"));
        }
    }

    #[test]
    fn no_eval_dataset_blocks_validate() {
        let s = PipelineState {
            raw_model_loaded: true,
            calibration_dataset_ready: true,
            calibrated: true,
            quantized: true,
            ..Default::default()
        };
        if let NextStepVerdict::Blocked { step, missing } = next_step(s) {
            assert_eq!(step, Step::Validate);
            assert!(missing.contains(&"eval_dataset_ready"));
        }
    }

    #[test]
    fn order_is_calibrate_quantize_validate_deploy() {
        let mut s = ready();
        for expected in [
            Step::Calibrate,
            Step::Quantize,
            Step::Validate,
            Step::Deploy,
        ] {
            if let NextStepVerdict::DoNext { step } = next_step(s) {
                assert_eq!(step, expected);
                match step {
                    Step::Calibrate => s.calibrated = true,
                    Step::Quantize => s.quantized = true,
                    Step::Validate => s.validated = true,
                    Step::Deploy => s.deployed = true,
                }
            }
        }
        assert_eq!(next_step(s), NextStepVerdict::Complete);
    }

    #[test]
    fn blocked_lists_all_missing_prereqs() {
        // Empty state lists both raw_model and calibration_dataset.
        if let NextStepVerdict::Blocked { missing, .. } = next_step(PipelineState::default()) {
            assert_eq!(missing.len(), 2);
        }
    }
}
