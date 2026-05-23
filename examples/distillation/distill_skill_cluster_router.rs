//! # Distillation Skill Cluster Router
//!
//! Route inputs to specialist students:
//!   math input → math student
//!   code input → code student
//!   general → general student
//!
//! Detection by token pattern. Returns student_index or fallback.
//!
//! Demonstrates the **DIST.23** recipe for PMAT-152 (milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: Mixture-of-Experts routing (Shazeer et al., 2017).
//!
//! Run with: cargo run --example distill_skill_cluster_router
//!
//! Added by PMAT-152 (catalog crosses 1000 recipes).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SkillCluster {
    Math,
    Code,
    Translation,
    General,
}

#[derive(Debug, PartialEq)]
pub enum RouteVerdict {
    Ok {
        cluster: SkillCluster,
        confidence: f64,
    },
    EmptyInput,
}

pub fn route(input: &str) -> RouteVerdict {
    if input.is_empty() {
        return RouteVerdict::EmptyInput;
    }
    let lower = input.to_ascii_lowercase();
    let math_score = count_keywords(
        &lower,
        &[
            "equation",
            "solve",
            "derivative",
            "integral",
            "matrix",
            "calculate",
            "compute",
            "+",
            "−",
            "÷",
            "×",
        ],
    );
    let code_score = count_keywords(
        &lower,
        &[
            "function", "variable", "loop", "fn ", "def ", "class ", "import ", "return", "()",
            "{}", ";",
        ],
    );
    let translation_score = count_keywords(
        &lower,
        &[
            "translate",
            "english",
            "spanish",
            "french",
            "german",
            "japanese",
            "from ",
            " to ",
        ],
    );
    let total = math_score + code_score + translation_score;
    if total == 0 {
        return RouteVerdict::Ok {
            cluster: SkillCluster::General,
            confidence: 0.5,
        };
    }
    let max_score = math_score.max(code_score).max(translation_score);
    let confidence = f64::from(max_score) / f64::from(total);
    let cluster = if math_score == max_score {
        SkillCluster::Math
    } else if code_score == max_score {
        SkillCluster::Code
    } else {
        SkillCluster::Translation
    };
    RouteVerdict::Ok {
        cluster,
        confidence,
    }
}

fn count_keywords(text: &str, keywords: &[&str]) -> u32 {
    keywords
        .iter()
        .map(|kw| text.matches(kw).count() as u32)
        .sum()
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("distill_skill_cluster_router")?;

    println!("math: {:?}", route("solve the equation 2x + 5 = 11 for x"));
    println!(
        "code: {:?}",
        route("write a fn that loops over a Vec and returns the sum")
    );
    println!(
        "translation: {:?}",
        route("translate from english to spanish: hello")
    );
    println!("general: {:?}", route("what is the capital of france"));
    println!("empty: {:?}", route(""));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn router_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn math_input_picks_math() {
        let v = route("solve the equation x + 5 = 10");
        if let RouteVerdict::Ok { cluster, .. } = v {
            assert_eq!(cluster, SkillCluster::Math);
        }
    }

    #[test]
    fn code_input_picks_code() {
        let v = route("write a fn that loops");
        if let RouteVerdict::Ok { cluster, .. } = v {
            assert_eq!(cluster, SkillCluster::Code);
        }
    }

    #[test]
    fn translation_picks_translation() {
        let v = route("translate from english to spanish");
        if let RouteVerdict::Ok { cluster, .. } = v {
            assert_eq!(cluster, SkillCluster::Translation);
        }
    }

    #[test]
    fn general_no_keywords() {
        let v = route("what is the capital of france");
        if let RouteVerdict::Ok { cluster, .. } = v {
            assert_eq!(cluster, SkillCluster::General);
        }
    }

    #[test]
    fn empty_input_rejected() {
        assert_eq!(route(""), RouteVerdict::EmptyInput);
    }

    #[test]
    fn confidence_in_zero_one() {
        for input in [
            "math solve",
            "fn loop",
            "translate spanish",
            "weather today",
        ] {
            if let RouteVerdict::Ok { confidence, .. } = route(input) {
                assert!((0.0..=1.0).contains(&confidence));
            }
        }
    }

    #[test]
    fn deterministic() {
        let a = route("solve equation");
        let b = route("solve equation");
        assert_eq!(a, b);
    }

    #[test]
    fn case_insensitive() {
        let lower = route("solve equation");
        let upper = route("SOLVE EQUATION");
        assert_eq!(lower, upper);
    }

    #[test]
    fn general_default_confidence() {
        let v = route("hello there");
        if let RouteVerdict::Ok {
            cluster,
            confidence,
        } = v
        {
            assert_eq!(cluster, SkillCluster::General);
            assert!((confidence - 0.5).abs() < 1e-9);
        }
    }

    #[test]
    fn high_confidence_dominant_signal() {
        // Very math-heavy input.
        let v = route("equation derivative integral matrix solve compute");
        if let RouteVerdict::Ok { confidence, .. } = v {
            assert!(confidence > 0.5);
        }
    }
}
