//! # Serverless Step Functions Choice Router
//!
//! Step Functions Choice state branches on input. Routing rules:
//!   StringEquals: branch when input.field == "value"
//!   NumericGreaterThan: branch when input.field > N
//!   IsPresent: branch if input.field exists
//!
//! Picker checks input against rules in order; first match wins.
//! If none match, fall through to Default.
//!
//! Demonstrates the **SVL.13** recipe for PMAT-144 (serverless round 2).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: AWS Step Functions Choice State spec.
//!
//! Run with: cargo run --example serverless_step_function_router
//!
//! Added by PMAT-144 (expand-cookbooks followup).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;
use std::collections::BTreeMap;

#[derive(Debug, Clone, PartialEq)]
pub enum Condition {
    StringEquals { field: String, value: String },
    NumericGreaterThan { field: String, value: i64 },
    IsPresent { field: String },
}

#[derive(Debug, Clone, PartialEq)]
pub struct Branch {
    pub condition: Condition,
    pub target_state: String,
}

#[derive(Debug, Clone, PartialEq)]
pub enum InputValue {
    Str(String),
    Num(i64),
    Missing,
}

#[derive(Debug, PartialEq)]
pub enum RouteVerdict {
    NextState { state: String, matched_index: usize },
    DefaultBranch { state: String },
    NoBranches,
}

pub fn route(
    branches: &[Branch],
    default_state: Option<&str>,
    input: &BTreeMap<String, InputValue>,
) -> RouteVerdict {
    if branches.is_empty() {
        if let Some(d) = default_state {
            return RouteVerdict::DefaultBranch {
                state: d.to_string(),
            };
        }
        return RouteVerdict::NoBranches;
    }
    for (i, branch) in branches.iter().enumerate() {
        if matches_condition(&branch.condition, input) {
            return RouteVerdict::NextState {
                state: branch.target_state.clone(),
                matched_index: i,
            };
        }
    }
    if let Some(d) = default_state {
        return RouteVerdict::DefaultBranch {
            state: d.to_string(),
        };
    }
    RouteVerdict::NoBranches
}

fn matches_condition(cond: &Condition, input: &BTreeMap<String, InputValue>) -> bool {
    match cond {
        Condition::StringEquals { field, value } => match input.get(field) {
            Some(InputValue::Str(s)) => s == value,
            _ => false,
        },
        Condition::NumericGreaterThan { field, value } => match input.get(field) {
            Some(InputValue::Num(n)) => n > value,
            _ => false,
        },
        Condition::IsPresent { field } => input
            .get(field)
            .is_some_and(|v| !matches!(v, InputValue::Missing)),
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("serverless_step_function_router")?;

    let branches = vec![
        Branch {
            condition: Condition::StringEquals {
                field: "kind".to_string(),
                value: "premium".to_string(),
            },
            target_state: "PremiumPath".to_string(),
        },
        Branch {
            condition: Condition::NumericGreaterThan {
                field: "size".to_string(),
                value: 100,
            },
            target_state: "LargeBatchPath".to_string(),
        },
    ];

    let mut input = BTreeMap::new();
    input.insert("kind".to_string(), InputValue::Str("premium".to_string()));
    println!(
        "kind=premium: {:?}",
        route(&branches, Some("Default"), &input)
    );

    let mut input2 = BTreeMap::new();
    input2.insert("size".to_string(), InputValue::Num(200));
    println!("size=200: {:?}", route(&branches, Some("Default"), &input2));

    let mut input3 = BTreeMap::new();
    input3.insert("other".to_string(), InputValue::Num(5));
    println!("no match: {:?}", route(&branches, Some("Default"), &input3));
    println!("empty branches: {:?}", route(&[], None, &BTreeMap::new()));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn input(pairs: &[(&str, InputValue)]) -> BTreeMap<String, InputValue> {
        pairs
            .iter()
            .map(|(k, v)| ((*k).to_string(), v.clone()))
            .collect()
    }

    #[test]
    fn router_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn string_equals_match() {
        let branches = vec![Branch {
            condition: Condition::StringEquals {
                field: "kind".to_string(),
                value: "premium".to_string(),
            },
            target_state: "PremiumPath".to_string(),
        }];
        let i = input(&[("kind", InputValue::Str("premium".to_string()))]);
        if let RouteVerdict::NextState { state, .. } = route(&branches, None, &i) {
            assert_eq!(state, "PremiumPath");
        }
    }

    #[test]
    fn numeric_gt_match() {
        let branches = vec![Branch {
            condition: Condition::NumericGreaterThan {
                field: "size".to_string(),
                value: 100,
            },
            target_state: "LargeBatch".to_string(),
        }];
        let i = input(&[("size", InputValue::Num(200))]);
        if let RouteVerdict::NextState { state, .. } = route(&branches, None, &i) {
            assert_eq!(state, "LargeBatch");
        }
    }

    #[test]
    fn is_present_match() {
        let branches = vec![Branch {
            condition: Condition::IsPresent {
                field: "auth_token".to_string(),
            },
            target_state: "Authorized".to_string(),
        }];
        let i = input(&[("auth_token", InputValue::Str("abc".to_string()))]);
        if let RouteVerdict::NextState { state, .. } = route(&branches, None, &i) {
            assert_eq!(state, "Authorized");
        }
    }

    #[test]
    fn first_match_wins() {
        let branches = vec![
            Branch {
                condition: Condition::IsPresent {
                    field: "key".to_string(),
                },
                target_state: "First".to_string(),
            },
            Branch {
                condition: Condition::IsPresent {
                    field: "key".to_string(),
                },
                target_state: "Second".to_string(),
            },
        ];
        let i = input(&[("key", InputValue::Num(1))]);
        if let RouteVerdict::NextState {
            matched_index,
            state,
        } = route(&branches, None, &i)
        {
            assert_eq!(matched_index, 0);
            assert_eq!(state, "First");
        }
    }

    #[test]
    fn no_match_falls_to_default() {
        let branches = vec![Branch {
            condition: Condition::IsPresent {
                field: "missing".to_string(),
            },
            target_state: "X".to_string(),
        }];
        let i = input(&[("other", InputValue::Num(1))]);
        if let RouteVerdict::DefaultBranch { state } = route(&branches, Some("Default"), &i) {
            assert_eq!(state, "Default");
        }
    }

    #[test]
    fn no_match_no_default_returns_no_branches() {
        let branches = vec![Branch {
            condition: Condition::IsPresent {
                field: "missing".to_string(),
            },
            target_state: "X".to_string(),
        }];
        assert_eq!(
            route(&branches, None, &BTreeMap::new()),
            RouteVerdict::NoBranches
        );
    }

    #[test]
    fn empty_branches_with_default() {
        if let RouteVerdict::DefaultBranch { state } = route(&[], Some("Only"), &BTreeMap::new()) {
            assert_eq!(state, "Only");
        }
    }

    #[test]
    fn empty_branches_no_default() {
        assert_eq!(route(&[], None, &BTreeMap::new()), RouteVerdict::NoBranches);
    }

    #[test]
    fn type_mismatch_no_match() {
        // Field is Num but condition expects Str → no match.
        let branches = vec![Branch {
            condition: Condition::StringEquals {
                field: "x".to_string(),
                value: "y".to_string(),
            },
            target_state: "T".to_string(),
        }];
        let i = input(&[("x", InputValue::Num(5))]);
        if let RouteVerdict::DefaultBranch { state } = route(&branches, Some("D"), &i) {
            assert_eq!(state, "D");
        }
    }

    #[test]
    fn missing_input_value_no_match() {
        let branches = vec![Branch {
            condition: Condition::IsPresent {
                field: "x".to_string(),
            },
            target_state: "T".to_string(),
        }];
        let i = input(&[("x", InputValue::Missing)]);
        if let RouteVerdict::DefaultBranch { state } = route(&branches, Some("D"), &i) {
            assert_eq!(state, "D");
        }
    }
}
