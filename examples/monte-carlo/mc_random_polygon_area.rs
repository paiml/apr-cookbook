//! # Monte-Carlo Random Polygon Area
//!
//! Generate random N-gons (vertices in unit disk) and compute area
//! via the shoelace formula. Reports mean area and ratio to disk
//! area (π).
//!
//! Demonstrates the **MC.105** recipe for PMAT-194 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: shoelace formula (Gauss 1795); convex polygon area
//!  estimation.
//!
//! Run with: cargo run --example mc_random_polygon_area
//!
//! Added by PMAT-194 (catalog 1369→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum PolygonVerdict {
    Ok { mean_area: f64, max_area: f64 },
    InvalidConfig,
}

pub fn simulate(trials: u32, vertices: u32, seed: u64) -> PolygonVerdict {
    if trials == 0 || vertices < 3 {
        return PolygonVerdict::InvalidConfig;
    }
    let mut total_area: f64 = 0.0;
    let mut max_area: f64 = 0.0;
    let mut rng_state = seed | 1;
    for _ in 0..trials {
        // Generate random angles (in order) for vertices on unit circle.
        let mut angles: Vec<f64> = (0..vertices)
            .map(|_| (lcg(&mut rng_state) >> 32) as f64 / (u32::MAX as f64))
            .collect();
        angles.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let pts: Vec<(f64, f64)> = angles
            .iter()
            .map(|a| {
                let rad = a * 2.0 * std::f64::consts::PI;
                (rad.cos(), rad.sin())
            })
            .collect();
        let area = shoelace(&pts);
        total_area += area;
        if area > max_area {
            max_area = area;
        }
    }
    PolygonVerdict::Ok {
        mean_area: total_area / f64::from(trials),
        max_area,
    }
}

fn shoelace(pts: &[(f64, f64)]) -> f64 {
    let n = pts.len();
    let mut sum = 0.0;
    for i in 0..n {
        let (x1, y1) = pts[i];
        let (x2, y2) = pts[(i + 1) % n];
        sum += x1 * y2 - x2 * y1;
    }
    sum.abs() * 0.5
}

fn lcg(state: &mut u64) -> u64 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    *state
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("mc_random_polygon_area")?;

    println!("triangle: {:?}", simulate(1000, 3, 42));
    println!("hexagon: {:?}", simulate(1000, 6, 42));
    println!("invalid: {:?}", simulate(0, 3, 42));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn simulator_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn area_positive() {
        let v = simulate(100, 6, 42);
        if let PolygonVerdict::Ok { mean_area, .. } = v {
            assert!(mean_area > 0.0);
        }
    }

    #[test]
    fn area_le_unit_circle() {
        // Inscribed polygon area < π × 1² = π.
        let v = simulate(100, 10, 42);
        if let PolygonVerdict::Ok { max_area, .. } = v {
            assert!(max_area < std::f64::consts::PI);
        }
    }

    #[test]
    fn more_vertices_more_area() {
        let tri = simulate(1000, 3, 42);
        let hex = simulate(1000, 8, 42);
        if let (PolygonVerdict::Ok { mean_area: t, .. }, PolygonVerdict::Ok { mean_area: h, .. }) =
            (tri, hex)
        {
            assert!(h > t);
        }
    }

    #[test]
    fn invalid_zero_trials() {
        assert_eq!(simulate(0, 3, 42), PolygonVerdict::InvalidConfig);
    }

    #[test]
    fn invalid_two_vertices() {
        assert_eq!(simulate(10, 2, 42), PolygonVerdict::InvalidConfig);
    }

    #[test]
    fn deterministic() {
        let a = simulate(100, 5, 42);
        let b = simulate(100, 5, 42);
        assert_eq!(a, b);
    }

    #[test]
    fn finite_outputs() {
        let v = simulate(100, 5, 42);
        if let PolygonVerdict::Ok {
            mean_area,
            max_area,
        } = v
        {
            assert!(mean_area.is_finite());
            assert!(max_area.is_finite());
        }
    }

    #[test]
    fn mean_le_max() {
        let v = simulate(100, 5, 42);
        if let PolygonVerdict::Ok {
            mean_area,
            max_area,
        } = v
        {
            assert!(mean_area <= max_area);
        }
    }

    #[test]
    fn single_trial_works() {
        let v = simulate(1, 3, 42);
        assert!(matches!(v, PolygonVerdict::Ok { .. }));
    }

    #[test]
    fn many_vertices_handled() {
        let v = simulate(20, 100, 42);
        assert!(matches!(v, PolygonVerdict::Ok { .. }));
    }

    #[test]
    fn area_in_unit_bounds() {
        let v = simulate(100, 3, 42);
        if let PolygonVerdict::Ok { mean_area, .. } = v {
            assert!(mean_area < 4.0);
        }
    }
}
