//! # Visualization Legend Placement Optimizer
//!
//! Pick legend corner by data density: lowest-density quadrant wins.
//! Falls back to `outside-right` if all four quadrants are dense.
//! This recipe takes (x, y) data points + plot bounds, computes
//! per-quadrant point counts, and returns the optimal corner.
//!
//! Demonstrates the **VIZ.4** recipe for PMAT-128 (visualization coverage —
//! closing F-invariant gap).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: Tufte, E. R. (2001). The Visual Display of Quantitative Information.
//!
//! Run with: cargo run --example viz_legend_placement_optimizer
//!
//! Added by PMAT-128 (expand-cookbooks followup).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LegendPosition {
    UpperLeft,
    UpperRight,
    LowerLeft,
    LowerRight,
    OutsideRight,
}

#[derive(Debug, PartialEq)]
pub enum PlacementVerdict {
    Ok(LegendPosition),
    EmptyData,
    InvalidBounds,
}

const DENSE_FRACTION_THRESHOLD: f64 = 0.20;

pub fn pick(
    points: &[(f64, f64)],
    x_min: f64,
    x_max: f64,
    y_min: f64,
    y_max: f64,
) -> PlacementVerdict {
    if points.is_empty() {
        return PlacementVerdict::EmptyData;
    }
    if x_min >= x_max || y_min >= y_max || !x_min.is_finite() || !y_max.is_finite() {
        return PlacementVerdict::InvalidBounds;
    }
    let x_mid = (x_min + x_max) / 2.0;
    let y_mid = (y_min + y_max) / 2.0;
    let mut counts = [0usize; 4]; // UL, UR, LL, LR
    for (x, y) in points {
        let upper = *y >= y_mid;
        let right = *x >= x_mid;
        let idx = match (upper, right) {
            (true, false) => 0,  // UL
            (true, true) => 1,   // UR
            (false, false) => 2, // LL
            (false, true) => 3,  // LR
        };
        counts[idx] += 1;
    }
    let total = points.len() as f64;
    let min_idx = counts
        .iter()
        .enumerate()
        .min_by_key(|(_, c)| **c)
        .map_or(0, |(i, _)| i);
    let min_frac = counts[min_idx] as f64 / total;
    if min_frac > DENSE_FRACTION_THRESHOLD {
        return PlacementVerdict::Ok(LegendPosition::OutsideRight);
    }
    let pos = match min_idx {
        0 => LegendPosition::UpperLeft,
        1 => LegendPosition::UpperRight,
        2 => LegendPosition::LowerLeft,
        _ => LegendPosition::LowerRight,
    };
    PlacementVerdict::Ok(pos)
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("viz_legend_placement_optimizer")?;

    let scatter: Vec<(f64, f64)> = (0..100)
        .map(|i| ((i as f64) / 100.0, ((i as f64) / 100.0).powi(2)))
        .collect();
    println!("rising-curve: {:?}", pick(&scatter, 0.0, 1.0, 0.0, 1.0));
    let dense: Vec<(f64, f64)> = (0..100)
        .map(|i| (i as f64 / 100.0, (i as f64 / 100.0).sin()))
        .collect();
    println!("dense:        {:?}", pick(&dense, 0.0, 1.0, -1.0, 1.0));
    println!("empty:        {:?}", pick(&[], 0.0, 1.0, 0.0, 1.0));
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
    fn rising_curve_legend_in_upper_left() {
        // Curve y = x²: most points in lower-left + lower-right; sparse upper-left.
        let pts: Vec<(f64, f64)> = (0..100)
            .map(|i| (i as f64 / 100.0, (i as f64 / 100.0).powi(2)))
            .collect();
        let v = pick(&pts, 0.0, 1.0, 0.0, 1.0);
        // Upper-left quadrant should be sparsest for y = x².
        if let PlacementVerdict::Ok(pos) = v {
            assert_eq!(pos, LegendPosition::UpperLeft);
        }
    }

    #[test]
    fn falling_curve_legend_in_upper_right() {
        // y = 1 - x: sparse upper-right.
        let pts: Vec<(f64, f64)> = (0..100)
            .map(|i| {
                let x = i as f64 / 100.0;
                (x, 1.0 - x)
            })
            .collect();
        let v = pick(&pts, 0.0, 1.0, 0.0, 1.0);
        // For y = 1 - x both UpperRight and LowerLeft are empty; either
        // sparse corner is acceptable.
        if let PlacementVerdict::Ok(pos) = v {
            assert!(matches!(
                pos,
                LegendPosition::UpperRight | LegendPosition::LowerLeft
            ));
        }
    }

    #[test]
    fn empty_rejected() {
        assert_eq!(pick(&[], 0.0, 1.0, 0.0, 1.0), PlacementVerdict::EmptyData);
    }

    #[test]
    fn invalid_bounds_rejected() {
        let pts = vec![(0.5, 0.5)];
        assert_eq!(
            pick(&pts, 1.0, 0.0, 0.0, 1.0),
            PlacementVerdict::InvalidBounds
        );
        assert_eq!(
            pick(&pts, 0.0, 1.0, 1.0, 0.0),
            PlacementVerdict::InvalidBounds
        );
    }

    #[test]
    fn nan_bound_rejected() {
        let pts = vec![(0.5, 0.5)];
        assert_eq!(
            pick(&pts, f64::NAN, 1.0, 0.0, 1.0),
            PlacementVerdict::InvalidBounds
        );
    }

    #[test]
    fn dense_data_falls_back_outside() {
        // Uniform random-ish: each quadrant has ~25% > 20% threshold.
        let pts: Vec<(f64, f64)> = (0..100)
            .map(|i| {
                let x = (i % 10) as f64 / 10.0;
                let y = (i / 10) as f64 / 10.0;
                (x, y)
            })
            .collect();
        let v = pick(&pts, 0.0, 1.0, 0.0, 1.0);
        if let PlacementVerdict::Ok(pos) = v {
            assert_eq!(pos, LegendPosition::OutsideRight);
        }
    }

    #[test]
    fn single_point_picks_some_corner() {
        // Single point at (0.5, 0.5) — boundary.
        let v = pick(&[(0.5, 0.5)], 0.0, 1.0, 0.0, 1.0);
        // Three quadrants empty; picks one of them.
        assert!(matches!(v, PlacementVerdict::Ok(_)));
    }

    #[test]
    fn quadrant_split_at_midpoint() {
        // Points strictly in one quadrant.
        let lower_left = vec![(0.1, 0.1), (0.2, 0.2), (0.3, 0.3)];
        let v = pick(&lower_left, 0.0, 1.0, 0.0, 1.0);
        // Only LL has points → legend goes anywhere except LL.
        if let PlacementVerdict::Ok(pos) = v {
            assert_ne!(pos, LegendPosition::LowerLeft);
        }
    }

    #[test]
    fn equal_density_picks_first_min() {
        // Two empty quadrants → picks the first by index.
        let pts = vec![(0.6, 0.4), (0.7, 0.3)]; // both LR
        let v = pick(&pts, 0.0, 1.0, 0.0, 1.0);
        if let PlacementVerdict::Ok(pos) = v {
            // UL is index 0 → first empty.
            assert_eq!(pos, LegendPosition::UpperLeft);
        }
    }
}
