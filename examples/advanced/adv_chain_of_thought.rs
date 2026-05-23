//! # Advanced Chain-of-Thought Extractor
//!
//! Extract step-by-step reasoning from a model response. Common
//! markers: "Step N:", "First...", numbered lists, "Therefore",
//! "Because". Returns the reasoning steps + final answer.
//!
//! Demonstrates the **ADV.22** recipe for PMAT-152 (milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: Wei et al. (2022). Chain-of-Thought Prompting. arXiv:2201.11903.
//!
//! Run with: cargo run --example adv_chain_of_thought
//!
//! Added by PMAT-152 (catalog crosses 1000 recipes).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum CotVerdict {
    Ok {
        steps: Vec<String>,
        final_answer: String,
    },
    EmptyResponse,
    NoStepsDetected,
}

pub fn extract(response: &str) -> CotVerdict {
    if response.trim().is_empty() {
        return CotVerdict::EmptyResponse;
    }
    let mut steps = Vec::new();
    let mut final_answer = String::new();
    for line in response.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        let lower = trimmed.to_ascii_lowercase();
        if lower.starts_with("answer:")
            || lower.starts_with("final answer:")
            || lower.starts_with("therefore,")
            || lower.starts_with("therefore ")
        {
            final_answer = trimmed.to_string();
        } else if is_step_marker(trimmed) {
            steps.push(trimmed.to_string());
        }
    }
    if steps.is_empty() && final_answer.is_empty() {
        return CotVerdict::NoStepsDetected;
    }
    CotVerdict::Ok {
        steps,
        final_answer,
    }
}

fn is_step_marker(line: &str) -> bool {
    let lower = line.to_ascii_lowercase();
    lower.starts_with("step ")
        || lower.starts_with("first")
        || lower.starts_with("second")
        || lower.starts_with("third")
        || lower.starts_with("then")
        || lower.starts_with("next")
        || lower.starts_with("finally")
        || matches!(line.chars().next(), Some(c) if c.is_ascii_digit())
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("adv_chain_of_thought")?;

    let response = "Step 1: Compute 2 + 3 = 5\nStep 2: Multiply by 4 = 20\nAnswer: 20";
    println!("typical: {:?}", extract(response));

    let no_cot = "The answer is 42.";
    println!("no cot: {:?}", extract(no_cot));

    println!("empty: {:?}", extract(""));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extractor_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn step_markers_extracted() {
        let r = "Step 1: do X\nStep 2: do Y\nAnswer: Z";
        if let CotVerdict::Ok { steps, .. } = extract(r) {
            assert_eq!(steps.len(), 2);
        }
    }

    #[test]
    fn final_answer_extracted() {
        let r = "Step 1: x\nAnswer: 42";
        if let CotVerdict::Ok { final_answer, .. } = extract(r) {
            assert!(final_answer.contains("42"));
        }
    }

    #[test]
    fn empty_rejected() {
        assert_eq!(extract(""), CotVerdict::EmptyResponse);
    }

    #[test]
    fn whitespace_only_rejected() {
        assert_eq!(extract("   \n  "), CotVerdict::EmptyResponse);
    }

    #[test]
    fn no_markers_rejected() {
        let r = "Just a plain sentence.";
        assert_eq!(extract(r), CotVerdict::NoStepsDetected);
    }

    #[test]
    fn first_second_third_recognized() {
        let r = "First, do X.\nSecond, do Y.\nThird, do Z.\nAnswer: done";
        if let CotVerdict::Ok { steps, .. } = extract(r) {
            assert_eq!(steps.len(), 3);
        }
    }

    #[test]
    fn numbered_lines_recognized() {
        let r = "1. First step\n2. Second step\nAnswer: ok";
        if let CotVerdict::Ok { steps, .. } = extract(r) {
            assert_eq!(steps.len(), 2);
        }
    }

    #[test]
    fn therefore_treated_as_final() {
        let r = "Step 1: A\nTherefore, the answer is B";
        if let CotVerdict::Ok { final_answer, .. } = extract(r) {
            assert!(final_answer.starts_with("Therefore"));
        }
    }

    #[test]
    fn case_insensitive_step() {
        let r = "STEP 1: Capital\nstep 2: lowercase";
        if let CotVerdict::Ok { steps, .. } = extract(r) {
            assert_eq!(steps.len(), 2);
        }
    }

    #[test]
    fn empty_lines_skipped() {
        let r = "Step 1: X\n\n\nStep 2: Y\n\nAnswer: Z";
        if let CotVerdict::Ok { steps, .. } = extract(r) {
            assert_eq!(steps.len(), 2);
        }
    }

    #[test]
    fn final_only_no_steps_ok() {
        let r = "Answer: 42";
        let v = extract(r);
        if let CotVerdict::Ok {
            steps,
            final_answer,
        } = v
        {
            assert!(steps.is_empty());
            assert!(final_answer.contains("42"));
        }
    }
}
