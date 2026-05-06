//! # CGP Proof-Status Dispatcher
//!
//! Lean theorem statuses: WIP (theorem name reserved, no `.lean` file
//! yet), Sorry (proof scaffolded with `sorry`), Proved (full proof
//! verified), NotApplicable (theorem doesn't apply). This recipe
//! dispatches per-status badges + aggregate scoring.
//!
//! Demonstrates the **CGP.6** recipe for PMAT-128 (cgp coverage).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: aprender contracts § Lean status discipline.
//!
//! Run with: cargo run --example cgp_proof_status_dispatcher
//!
//! Added by PMAT-128 (expand-cookbooks followup).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProofStatus {
    Wip,
    Sorry,
    Proved,
    NotApplicable,
}

impl ProofStatus {
    pub fn from_str_strict(s: &str) -> Option<Self> {
        match s.to_ascii_lowercase().as_str() {
            "wip" => Some(ProofStatus::Wip),
            "sorry" => Some(ProofStatus::Sorry),
            "proved" => Some(ProofStatus::Proved),
            "not-applicable" | "n/a" | "na" => Some(ProofStatus::NotApplicable),
            _ => None,
        }
    }

    pub fn badge(self) -> &'static str {
        match self {
            ProofStatus::Wip => "[WIP]",
            ProofStatus::Sorry => "[SORRY]",
            ProofStatus::Proved => "[PROVED]",
            ProofStatus::NotApplicable => "[N/A]",
        }
    }
}

#[derive(Debug, PartialEq, Default)]
pub struct ProofTotals {
    pub wip: u32,
    pub sorry: u32,
    pub proved: u32,
    pub not_applicable: u32,
}

impl ProofTotals {
    pub fn total(&self) -> u32 {
        self.wip + self.sorry + self.proved + self.not_applicable
    }

    pub fn coverage_pct(&self) -> Option<f64> {
        let denom = self.total() - self.not_applicable;
        if denom == 0 {
            return None;
        }
        Some(f64::from(self.proved) / f64::from(denom) * 100.0)
    }
}

pub fn aggregate<I>(statuses: I) -> ProofTotals
where
    I: IntoIterator<Item = ProofStatus>,
{
    let mut t = ProofTotals::default();
    for s in statuses {
        match s {
            ProofStatus::Wip => t.wip += 1,
            ProofStatus::Sorry => t.sorry += 1,
            ProofStatus::Proved => t.proved += 1,
            ProofStatus::NotApplicable => t.not_applicable += 1,
        }
    }
    t
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("cgp_proof_status_dispatcher")?;

    for s in ["wip", "sorry", "proved", "not-applicable", "typo"] {
        println!("{s} → {:?}", ProofStatus::from_str_strict(s));
    }
    let t = aggregate([
        ProofStatus::Proved,
        ProofStatus::Proved,
        ProofStatus::Sorry,
        ProofStatus::Wip,
        ProofStatus::NotApplicable,
    ]);
    println!("totals: {t:?}");
    println!("coverage: {:?}%", t.coverage_pct());
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dispatcher_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn known_statuses_round_trip() {
        for s in ["wip", "sorry", "proved", "not-applicable"] {
            assert!(ProofStatus::from_str_strict(s).is_some());
        }
    }

    #[test]
    fn case_insensitive_status_parse() {
        assert_eq!(
            ProofStatus::from_str_strict("PROVED"),
            Some(ProofStatus::Proved)
        );
        assert_eq!(ProofStatus::from_str_strict("Wip"), Some(ProofStatus::Wip));
    }

    #[test]
    fn unknown_status_rejected() {
        assert!(ProofStatus::from_str_strict("typo").is_none());
    }

    #[test]
    fn na_aliases_accepted() {
        assert_eq!(
            ProofStatus::from_str_strict("n/a"),
            Some(ProofStatus::NotApplicable)
        );
        assert_eq!(
            ProofStatus::from_str_strict("na"),
            Some(ProofStatus::NotApplicable)
        );
    }

    #[test]
    fn badges_match_status() {
        assert_eq!(ProofStatus::Wip.badge(), "[WIP]");
        assert_eq!(ProofStatus::Sorry.badge(), "[SORRY]");
        assert_eq!(ProofStatus::Proved.badge(), "[PROVED]");
        assert_eq!(ProofStatus::NotApplicable.badge(), "[N/A]");
    }

    #[test]
    fn aggregate_counts_correctly() {
        let t = aggregate([ProofStatus::Proved, ProofStatus::Proved, ProofStatus::Wip]);
        assert_eq!(t.proved, 2);
        assert_eq!(t.wip, 1);
        assert_eq!(t.total(), 3);
    }

    #[test]
    fn coverage_excludes_not_applicable() {
        let t = aggregate([
            ProofStatus::Proved,
            ProofStatus::Sorry,
            ProofStatus::NotApplicable,
        ]);
        // 1 proved out of 2 applicable = 50%.
        let pct = t.coverage_pct().unwrap();
        assert!((pct - 50.0).abs() < 1e-9);
    }

    #[test]
    fn coverage_all_na_returns_none() {
        let t = aggregate([ProofStatus::NotApplicable]);
        assert!(t.coverage_pct().is_none());
    }

    #[test]
    fn empty_aggregate_zero_totals() {
        let t = aggregate(std::iter::empty());
        assert_eq!(t.total(), 0);
    }
}
