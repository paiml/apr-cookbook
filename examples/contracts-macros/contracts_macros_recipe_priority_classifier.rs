//! # Contracts-Macros Recipe Priority Classifier
//!
//! Classify a recipe into P0..P3 based on (citation count, contract
//! grade, complexity score). P0 is highest priority. Returns the
//! priority and the score breakdown.
//!
//! Demonstrates the **CMM.78** recipe for PMAT-183 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: ICE scoring (Impact/Confidence/Ease, Sean Ellis ~2010).
//!
//! Run with: cargo run --example contracts_macros_recipe_priority_classifier
//!
//! Added by PMAT-183 (catalog 1270→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum Priority {
    P0,
    P1,
    P2,
    P3,
}

#[derive(Debug, PartialEq)]
pub enum ClassifyVerdict {
    Ok { priority: Priority, score: u32 },
    InvalidConfig,
}

pub fn classify(citations: u32, grade: char, complexity: u32) -> ClassifyVerdict {
    if !"ABCDF".contains(grade) || complexity > 100 {
        return ClassifyVerdict::InvalidConfig;
    }
    let cite_score = citations.min(10);
    let grade_score = match grade {
        'A' => 10,
        'B' => 7,
        'C' => 4,
        'D' => 2,
        _ => 0,
    };
    let complexity_score = if complexity <= 20 {
        10
    } else if complexity <= 50 {
        5
    } else {
        0
    };
    let score = cite_score + grade_score + complexity_score;
    let priority = if score >= 25 {
        Priority::P0
    } else if score >= 18 {
        Priority::P1
    } else if score >= 10 {
        Priority::P2
    } else {
        Priority::P3
    };
    ClassifyVerdict::Ok { priority, score }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("contracts_macros_recipe_priority_classifier")?;

    println!("flagship: {:?}", classify(10, 'A', 10));
    println!("backlog: {:?}", classify(0, 'F', 80));
    println!("invalid: {:?}", classify(0, 'X', 0));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn classifier_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn high_score_p0() {
        let v = classify(10, 'A', 10);
        if let ClassifyVerdict::Ok { priority, .. } = v {
            assert_eq!(priority, Priority::P0);
        }
    }

    #[test]
    fn low_score_p3() {
        let v = classify(0, 'F', 80);
        if let ClassifyVerdict::Ok { priority, .. } = v {
            assert_eq!(priority, Priority::P3);
        }
    }

    #[test]
    fn invalid_grade_rejected() {
        assert_eq!(classify(5, 'X', 30), ClassifyVerdict::InvalidConfig);
    }

    #[test]
    fn invalid_complexity_rejected() {
        assert_eq!(classify(5, 'A', 200), ClassifyVerdict::InvalidConfig);
    }

    #[test]
    fn medium_score_p1_or_p2() {
        let v = classify(3, 'B', 30);
        if let ClassifyVerdict::Ok { priority, .. } = v {
            assert!(matches!(priority, Priority::P1 | Priority::P2));
        }
    }

    #[test]
    fn citations_capped_at_ten() {
        let v_ten = classify(10, 'B', 60);
        let v_thousand = classify(1000, 'B', 60);
        assert_eq!(v_ten, v_thousand);
    }

    #[test]
    fn deterministic() {
        let r1 = classify(5, 'A', 10);
        let r2 = classify(5, 'A', 10);
        assert_eq!(r1, r2);
    }

    #[test]
    fn higher_grade_higher_score() {
        let a = classify(5, 'A', 30);
        let c = classify(5, 'C', 30);
        if let (ClassifyVerdict::Ok { score: a_s, .. }, ClassifyVerdict::Ok { score: c_s, .. }) =
            (a, c)
        {
            assert!(a_s > c_s);
        }
    }

    #[test]
    fn lower_complexity_higher_score() {
        let easy = classify(5, 'B', 10);
        let hard = classify(5, 'B', 80);
        if let (ClassifyVerdict::Ok { score: e_s, .. }, ClassifyVerdict::Ok { score: h_s, .. }) =
            (easy, hard)
        {
            assert!(e_s > h_s);
        }
    }

    #[test]
    fn boundary_complexity_50_partial_credit() {
        let v = classify(0, 'F', 50);
        if let ClassifyVerdict::Ok { score, .. } = v {
            // 0 citations + F=0 + complexity_score=5 = 5.
            assert_eq!(score, 5);
        }
    }

    #[test]
    fn extreme_complexity_zero_score_for_complexity() {
        let v = classify(0, 'F', 100);
        if let ClassifyVerdict::Ok { score, .. } = v {
            assert_eq!(score, 0);
        }
    }
}
