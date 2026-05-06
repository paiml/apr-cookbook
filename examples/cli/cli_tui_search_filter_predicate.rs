//! # apr tui — Search Filter Predicate
//!
//! `apr tui` `/search` filter accepts a substring; case-sensitive by
//! default, prefix `\C` for explicit case-sensitive (defensive); empty
//! query restores the full list. This recipe builds the predicate and
//! asserts the contract.
//!
//! Demonstrates the **TUI.6** recipe for PMAT-108 (apr tui coverage).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: aprender TUI-003 + vim search prefix convention
//!
//! Run with: cargo run --example cli_tui_search_filter_predicate
//!
//! Added by PMAT-108 (expand-cookbooks followup).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParsedQuery {
    pub needle: String,
    pub case_sensitive: bool,
    pub is_empty: bool,
}

pub fn parse_query(raw: &str) -> ParsedQuery {
    if raw.is_empty() {
        return ParsedQuery {
            needle: String::new(),
            case_sensitive: false,
            is_empty: true,
        };
    }
    if let Some(stripped) = raw.strip_prefix("\\C") {
        return ParsedQuery {
            needle: stripped.to_string(),
            case_sensitive: true,
            is_empty: stripped.is_empty(),
        };
    }
    ParsedQuery {
        needle: raw.to_string(),
        case_sensitive: false,
        is_empty: false,
    }
}

pub fn matches(query: &ParsedQuery, item: &str) -> bool {
    if query.is_empty {
        return true;
    }
    if query.case_sensitive {
        item.contains(&query.needle)
    } else {
        item.to_ascii_lowercase()
            .contains(&query.needle.to_ascii_lowercase())
    }
}

pub fn filter_items<'a>(items: &'a [&'a str], raw_query: &str) -> Vec<&'a str> {
    let query = parse_query(raw_query);
    items
        .iter()
        .copied()
        .filter(|i| matches(&query, i))
        .collect()
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("cli_tui_search_filter_predicate")?;

    let items = [
        "model.embed_tokens.weight",
        "model.layers.0.self_attn.q_proj.weight",
        "lm_head.weight",
        "MODEL.LAYERS.0.MLP.gate.weight",
    ];

    for q in ["", "weight", "Q_PROJ", "\\CQ_PROJ", "\\Cself_attn"] {
        println!("query {q:>15}  →  {:?}", filter_items(&items, q));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn predicate_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn empty_query_matches_everything() {
        let q = parse_query("");
        assert!(q.is_empty);
        assert!(matches(&q, "anything"));
    }

    #[test]
    fn case_insensitive_default() {
        let q = parse_query("foo");
        assert!(!q.case_sensitive);
        assert!(matches(&q, "FOO_BAR"));
        assert!(matches(&q, "foo_bar"));
    }

    #[test]
    fn capital_c_prefix_enables_case_sensitive() {
        let q = parse_query("\\CFoo");
        assert!(q.case_sensitive);
        assert!(matches(&q, "Foo_bar"));
        assert!(!matches(&q, "FOO_BAR"));
        assert!(!matches(&q, "foo_bar"));
    }

    #[test]
    fn case_sensitive_with_empty_after_prefix_is_empty() {
        let q = parse_query("\\C");
        assert!(q.is_empty);
        // Empty prefix-only query matches everything (operator probably typed
        // \C and started typing — show all until they finish).
        assert!(matches(&q, "anything"));
    }

    #[test]
    fn filter_items_returns_subset() {
        let items = ["a.weight", "b.bias", "c.weight"];
        let kept = filter_items(&items, "weight");
        assert_eq!(kept.len(), 2);
    }

    #[test]
    fn filter_items_empty_query_returns_all() {
        let items = ["a", "b", "c"];
        let kept = filter_items(&items, "");
        assert_eq!(kept.len(), 3);
    }

    #[test]
    fn filter_items_no_match_returns_empty() {
        let items = ["a", "b", "c"];
        let kept = filter_items(&items, "nonexistent");
        assert!(kept.is_empty());
    }
}
