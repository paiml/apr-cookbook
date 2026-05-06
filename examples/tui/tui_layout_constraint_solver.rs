//! # TUI Layout Constraint Solver
//!
//! Solve a layout: given total width, distribute to children with
//! `(min, max, flex_weight)` constraints. Minimums first, then flex-
//! weighted distribution of remaining space, capped at maximums.
//!
//! Demonstrates the **TUI.52** recipe for PMAT-177 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: CSS flexbox flex-grow algorithm.
//!
//! Run with: cargo run --example tui_layout_constraint_solver
//!
//! Added by PMAT-177 (catalog 1216→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum SolveVerdict {
    Ok { widths: Vec<u32> },
    InfeasibleMinimums,
    InvalidConfig,
}

pub fn solve(total: u32, children: &[(u32, u32, u32)]) -> SolveVerdict {
    if total == 0 || children.is_empty() {
        return SolveVerdict::InvalidConfig;
    }
    for (min, max, _) in children {
        if min > max {
            return SolveVerdict::InvalidConfig;
        }
    }
    let min_total: u32 = children.iter().map(|(m, _, _)| *m).sum();
    if min_total > total {
        return SolveVerdict::InfeasibleMinimums;
    }
    let mut widths: Vec<u32> = children.iter().map(|(m, _, _)| *m).collect();
    let mut remaining = total - min_total;
    while remaining > 0 {
        let weight_total: u32 = children
            .iter()
            .enumerate()
            .filter(|(i, (_, max, w))| widths[*i] < *max && *w > 0)
            .map(|(_, (_, _, w))| *w)
            .sum();
        if weight_total == 0 {
            break;
        }
        let mut allocated = 0u32;
        for (i, (_, max, w)) in children.iter().enumerate() {
            if widths[i] >= *max || *w == 0 {
                continue;
            }
            let share = (remaining * w) / weight_total;
            let cap = max - widths[i];
            let add = share.min(cap);
            widths[i] += add;
            allocated += add;
        }
        if allocated == 0 {
            // Rounding leftover: dump on the first flex child not at max.
            if let Some((i, _)) = children
                .iter()
                .enumerate()
                .find(|(i, (_, max, w))| widths[*i] < *max && *w > 0)
            {
                let cap = children[i].1 - widths[i];
                let add = remaining.min(cap);
                widths[i] += add;
                allocated = add;
            } else {
                break;
            }
        }
        remaining = remaining.saturating_sub(allocated);
    }
    SolveVerdict::Ok { widths }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("tui_layout_constraint_solver")?;

    let children = [(10u32, 100u32, 1u32), (10, 100, 2)];
    println!("typical: {:?}", solve(60, &children));

    let infeasible = [(50u32, 100u32, 1u32), (50, 100, 1)];
    println!("infeasible: {:?}", solve(80, &infeasible));

    println!("invalid: {:?}", solve(0, &children));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn solver_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn flex_weights_distribute() {
        let v = solve(60, &[(10, 100, 1), (10, 100, 2)]);
        if let SolveVerdict::Ok { widths } = v {
            // 60 - 20 = 40 extra; 1:2 split → 13 + 27.
            assert!((widths[0] as i32 - 23).abs() <= 2);
            assert!((widths[1] as i32 - 37).abs() <= 2);
            assert_eq!(widths.iter().sum::<u32>(), 60);
        }
    }

    #[test]
    fn min_respected() {
        let v = solve(60, &[(10, 100, 0), (10, 100, 1)]);
        if let SolveVerdict::Ok { widths } = v {
            // First child has weight 0 → only its min.
            assert_eq!(widths[0], 10);
        }
    }

    #[test]
    fn max_capped() {
        let v = solve(100, &[(10, 30, 1), (10, 100, 1)]);
        if let SolveVerdict::Ok { widths } = v {
            assert_eq!(widths[0], 30);
            assert_eq!(widths[1], 70);
        }
    }

    #[test]
    fn infeasible_minimums_rejected() {
        let v = solve(50, &[(30, 100, 1), (30, 100, 1)]);
        assert_eq!(v, SolveVerdict::InfeasibleMinimums);
    }

    #[test]
    fn invalid_zero_total() {
        assert_eq!(solve(0, &[(10, 100, 1)]), SolveVerdict::InvalidConfig);
    }

    #[test]
    fn invalid_empty_children() {
        assert_eq!(solve(100, &[]), SolveVerdict::InvalidConfig);
    }

    #[test]
    fn invalid_min_above_max() {
        assert_eq!(solve(100, &[(50, 30, 1)]), SolveVerdict::InvalidConfig);
    }

    #[test]
    fn widths_sum_to_total() {
        let v = solve(100, &[(10, 80, 1), (10, 80, 1)]);
        if let SolveVerdict::Ok { widths } = v {
            assert_eq!(widths.iter().sum::<u32>(), 100);
        }
    }

    #[test]
    fn single_child_full_width() {
        let v = solve(100, &[(0, 100, 1)]);
        if let SolveVerdict::Ok { widths } = v {
            assert_eq!(widths[0], 100);
        }
    }

    #[test]
    fn deterministic() {
        let c = [(10u32, 100u32, 1u32), (10, 100, 2)];
        let a = solve(60, &c);
        let b = solve(60, &c);
        assert_eq!(a, b);
    }
}
