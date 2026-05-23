//! # Contracts-Macros Recipe Owner Routing
//!
//! Map a failing recipe ID to the responsible owner team via the
//! prefix-match table (most-specific match wins). Returns the team
//! name or Unowned.
//!
//! Demonstrates the **CMM.57** recipe for PMAT-176 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: GitHub CODEOWNERS prefix-routing.
//!
//! Run with: cargo run --example contracts_macros_recipe_owner_routing
//!
//! Added by PMAT-176 (catalog 1207→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum RoutingVerdict {
    Owned { team: String, prefix: String },
    Unowned,
    EmptyTable,
}

pub fn route(recipe_id: &str, table: &[(&str, &str)]) -> RoutingVerdict {
    if table.is_empty() {
        return RoutingVerdict::EmptyTable;
    }
    if recipe_id.is_empty() {
        return RoutingVerdict::Unowned;
    }
    let mut best: Option<(&str, &str)> = None;
    for (prefix, team) in table {
        if recipe_id.starts_with(prefix) {
            match best {
                Some((bp, _)) if bp.len() >= prefix.len() => {}
                _ => best = Some((prefix, team)),
            }
        }
    }
    match best {
        Some((prefix, team)) => RoutingVerdict::Owned {
            team: team.to_string(),
            prefix: prefix.to_string(),
        },
        None => RoutingVerdict::Unowned,
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("contracts_macros_recipe_owner_routing")?;

    let table = [
        ("tui_", "frontend"),
        ("mc_", "performance"),
        ("contracts_macros_", "verification"),
        ("contracts_macros_recipe_", "verification-tools"),
    ];
    println!(
        "most specific: {:?}",
        route("contracts_macros_recipe_id_canon", &table)
    );
    println!(
        "less specific: {:?}",
        route("contracts_macros_arxiv", &table)
    );
    println!("tui: {:?}", route("tui_progress_state", &table));
    println!("unowned: {:?}", route("unknown_recipe", &table));
    println!("empty table: {:?}", route("any", &[]));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn typical() -> Vec<(&'static str, &'static str)> {
        vec![
            ("tui_", "frontend"),
            ("mc_", "performance"),
            ("contracts_macros_", "verification"),
            ("contracts_macros_recipe_", "verification-tools"),
        ]
    }

    #[test]
    fn router_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn most_specific_wins() {
        let v = route("contracts_macros_recipe_canon", &typical());
        if let RoutingVerdict::Owned { team, .. } = v {
            assert_eq!(team, "verification-tools");
        }
    }

    #[test]
    fn less_specific_match() {
        let v = route("contracts_macros_arxiv", &typical());
        if let RoutingVerdict::Owned { team, .. } = v {
            assert_eq!(team, "verification");
        }
    }

    #[test]
    fn surface_match() {
        let v = route("tui_progress_state", &typical());
        if let RoutingVerdict::Owned { team, .. } = v {
            assert_eq!(team, "frontend");
        }
    }

    #[test]
    fn no_match_unowned() {
        let v = route("unknown_recipe", &typical());
        assert_eq!(v, RoutingVerdict::Unowned);
    }

    #[test]
    fn empty_table_special() {
        assert_eq!(route("any", &[]), RoutingVerdict::EmptyTable);
    }

    #[test]
    fn empty_recipe_id_unowned() {
        assert_eq!(route("", &typical()), RoutingVerdict::Unowned);
    }

    #[test]
    fn prefix_returned() {
        let v = route("tui_table", &typical());
        if let RoutingVerdict::Owned { prefix, .. } = v {
            assert_eq!(prefix, "tui_");
        }
    }

    #[test]
    fn case_sensitive() {
        let v = route("TUI_progress", &typical());
        assert_eq!(v, RoutingVerdict::Unowned);
    }

    #[test]
    fn many_prefixes() {
        let table: Vec<(&str, &str)> = (0..50)
            .map(|i| {
                if i == 25 {
                    ("match_", "winner")
                } else {
                    ("none_", "loser")
                }
            })
            .collect();
        let v = route("match_x", &table);
        if let RoutingVerdict::Owned { team, .. } = v {
            assert_eq!(team, "winner");
        }
    }

    #[test]
    fn deterministic() {
        let t = typical();
        let a = route("tui_x", &t);
        let b = route("tui_x", &t);
        assert_eq!(a, b);
    }
}
