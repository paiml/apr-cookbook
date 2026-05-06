//! # Contracts-Macros Obligation Priority Sort
//!
//! Sort obligations by (severity desc, age_secs desc, name asc) so the
//! most-urgent items surface first. Returns the deterministic order.
//!
//! Demonstrates the **CMM.31** recipe for PMAT-168 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: priority queue sorting (Heap, CLRS).
//!
//! Run with: cargo run --example contracts_macros_priority_sort
//!
//! Added by PMAT-168 (catalog 1135→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Severity {
    Advisory,
    Required,
    Blocking,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Obligation {
    pub name: String,
    pub severity: Severity,
    pub age_secs: u64,
}

#[derive(Debug, PartialEq)]
pub enum SortVerdict {
    Ok { ordered: Vec<String> },
    EmptyList,
}

pub fn sort_priority(items: &[Obligation]) -> SortVerdict {
    if items.is_empty() {
        return SortVerdict::EmptyList;
    }
    let mut owned: Vec<Obligation> = items.to_vec();
    owned.sort_by(|a, b| {
        b.severity
            .cmp(&a.severity)
            .then(b.age_secs.cmp(&a.age_secs))
            .then(a.name.cmp(&b.name))
    });
    SortVerdict::Ok {
        ordered: owned.into_iter().map(|o| o.name).collect(),
    }
}

fn ob(name: &str, severity: Severity, age_secs: u64) -> Obligation {
    Obligation {
        name: name.to_string(),
        severity,
        age_secs,
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("contracts_macros_priority_sort")?;

    let items = vec![
        ob("low", Severity::Advisory, 100),
        ob("urgent", Severity::Blocking, 5),
        ob("medium", Severity::Required, 50),
        ob("oldest", Severity::Blocking, 200),
    ];
    println!("typical: {:?}", sort_priority(&items));
    println!("empty: {:?}", sort_priority(&[]));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sorter_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn blocking_before_required() {
        let v = sort_priority(&[
            ob("a", Severity::Required, 0),
            ob("b", Severity::Blocking, 0),
        ]);
        if let SortVerdict::Ok { ordered } = v {
            assert_eq!(ordered[0], "b");
        }
    }

    #[test]
    fn required_before_advisory() {
        let v = sort_priority(&[
            ob("a", Severity::Advisory, 0),
            ob("b", Severity::Required, 0),
        ]);
        if let SortVerdict::Ok { ordered } = v {
            assert_eq!(ordered[0], "b");
        }
    }

    #[test]
    fn older_first_within_severity() {
        let v = sort_priority(&[
            ob("new", Severity::Blocking, 5),
            ob("old", Severity::Blocking, 200),
        ]);
        if let SortVerdict::Ok { ordered } = v {
            assert_eq!(ordered[0], "old");
        }
    }

    #[test]
    fn name_asc_breaks_ties() {
        let v = sort_priority(&[
            ob("b", Severity::Blocking, 100),
            ob("a", Severity::Blocking, 100),
        ]);
        if let SortVerdict::Ok { ordered } = v {
            assert_eq!(ordered, vec!["a".to_string(), "b".to_string()]);
        }
    }

    #[test]
    fn empty_rejected() {
        assert_eq!(sort_priority(&[]), SortVerdict::EmptyList);
    }

    #[test]
    fn single_element_works() {
        let v = sort_priority(&[ob("only", Severity::Required, 0)]);
        if let SortVerdict::Ok { ordered } = v {
            assert_eq!(ordered, vec!["only".to_string()]);
        }
    }

    #[test]
    fn count_preserved() {
        let v = sort_priority(&[
            ob("a", Severity::Blocking, 0),
            ob("b", Severity::Required, 0),
            ob("c", Severity::Advisory, 0),
        ]);
        if let SortVerdict::Ok { ordered } = v {
            assert_eq!(ordered.len(), 3);
        }
    }

    #[test]
    fn full_ordering() {
        let v = sort_priority(&[
            ob("low", Severity::Advisory, 100),
            ob("urgent", Severity::Blocking, 5),
            ob("medium", Severity::Required, 50),
            ob("oldest", Severity::Blocking, 200),
        ]);
        if let SortVerdict::Ok { ordered } = v {
            assert_eq!(
                ordered,
                vec![
                    "oldest".to_string(),
                    "urgent".to_string(),
                    "medium".to_string(),
                    "low".to_string(),
                ]
            );
        }
    }

    #[test]
    fn deterministic() {
        let items = vec![ob("a", Severity::Blocking, 100)];
        let a = sort_priority(&items);
        let b = sort_priority(&items);
        assert_eq!(a, b);
    }

    #[test]
    fn many_obligations() {
        let items: Vec<Obligation> = (0..50)
            .map(|i| ob(&format!("o{i}"), Severity::Required, i as u64))
            .collect();
        if let SortVerdict::Ok { ordered } = sort_priority(&items) {
            assert_eq!(ordered.len(), 50);
        }
    }
}
