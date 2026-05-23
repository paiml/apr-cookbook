//! # apr validate-manifest — `--offline` Safety Override
//!
//! `apr validate-manifest --offline` overrides any `--live` flag and
//! refuses any network I/O. This recipe codifies the override priority:
//! `--offline` wins over `--live` and `APR_OFFLINE=1`. Output explicitly
//! marks PM-003 as NotApplicable rather than Failed (operator chose
//! offline by intent).
//!
//! Demonstrates the **VAL-MANIFEST.8** recipe for PMAT-110 (apr validate-manifest coverage).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: aprender SPEC-SHIP-TWO-001 §12.3 + Sovereign AI §9
//!
//! Run with: cargo run --example cli_validate_manifest_offline_safety
//!
//! Added by PMAT-110 (expand-cookbooks followup).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NetworkPolicy {
    Allowed,
    Forbidden,
}

#[derive(Debug, Default, Clone, Copy)]
pub struct OfflineInputs {
    pub cli_offline: bool,
    pub cli_live: bool,
    pub env_apr_offline: bool,
    pub env_hf_hub_offline: bool,
}

pub fn resolve_policy(inputs: OfflineInputs) -> NetworkPolicy {
    // --offline always wins; environment offline flags also force offline.
    if inputs.cli_offline || inputs.env_apr_offline || inputs.env_hf_hub_offline {
        return NetworkPolicy::Forbidden;
    }
    if inputs.cli_live {
        return NetworkPolicy::Allowed;
    }
    // Default: deferred = treated as forbidden (no spurious network).
    NetworkPolicy::Forbidden
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct OfflineOverrideAudit {
    pub policy: NetworkPolicy,
    pub forced_offline_by: Option<&'static str>,
}

pub fn audit_override(inputs: OfflineInputs) -> OfflineOverrideAudit {
    let policy = resolve_policy(inputs);
    let forced_offline_by = if inputs.cli_offline {
        Some("--offline flag")
    } else if inputs.env_apr_offline {
        Some("APR_OFFLINE=1")
    } else if inputs.env_hf_hub_offline {
        Some("HF_HUB_OFFLINE=1")
    } else {
        None
    };
    OfflineOverrideAudit {
        policy,
        forced_offline_by,
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("cli_validate_manifest_offline_safety")?;

    let cases = [
        ("default", OfflineInputs::default()),
        (
            "--live",
            OfflineInputs {
                cli_live: true,
                ..Default::default()
            },
        ),
        (
            "--offline",
            OfflineInputs {
                cli_offline: true,
                ..Default::default()
            },
        ),
        (
            "--offline --live (offline wins)",
            OfflineInputs {
                cli_offline: true,
                cli_live: true,
                ..Default::default()
            },
        ),
        (
            "APR_OFFLINE=1",
            OfflineInputs {
                cli_live: true,
                env_apr_offline: true,
                ..Default::default()
            },
        ),
        (
            "HF_HUB_OFFLINE=1",
            OfflineInputs {
                cli_live: true,
                env_hf_hub_offline: true,
                ..Default::default()
            },
        ),
    ];
    for (label, inputs) in cases {
        println!("{label:>32}  →  {:?}", audit_override(inputs));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn audit_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn default_is_forbidden() {
        // Default = no live flag set, no offline overrides → forbidden.
        let p = resolve_policy(OfflineInputs::default());
        assert_eq!(p, NetworkPolicy::Forbidden);
    }

    #[test]
    fn live_alone_allowed() {
        let p = resolve_policy(OfflineInputs {
            cli_live: true,
            ..Default::default()
        });
        assert_eq!(p, NetworkPolicy::Allowed);
    }

    #[test]
    fn cli_offline_wins_over_live() {
        let p = resolve_policy(OfflineInputs {
            cli_offline: true,
            cli_live: true,
            ..Default::default()
        });
        assert_eq!(p, NetworkPolicy::Forbidden);
    }

    #[test]
    fn apr_offline_env_wins_over_live() {
        let p = resolve_policy(OfflineInputs {
            cli_live: true,
            env_apr_offline: true,
            ..Default::default()
        });
        assert_eq!(p, NetworkPolicy::Forbidden);
    }

    #[test]
    fn hf_hub_offline_env_wins_over_live() {
        let p = resolve_policy(OfflineInputs {
            cli_live: true,
            env_hf_hub_offline: true,
            ..Default::default()
        });
        assert_eq!(p, NetworkPolicy::Forbidden);
    }

    #[test]
    fn audit_records_cli_flag_as_source() {
        let a = audit_override(OfflineInputs {
            cli_offline: true,
            cli_live: true,
            ..Default::default()
        });
        assert_eq!(a.forced_offline_by, Some("--offline flag"));
    }

    #[test]
    fn audit_records_env_var_as_source() {
        let a = audit_override(OfflineInputs {
            cli_live: true,
            env_apr_offline: true,
            ..Default::default()
        });
        assert_eq!(a.forced_offline_by, Some("APR_OFFLINE=1"));
    }

    #[test]
    fn audit_no_force_when_default() {
        let a = audit_override(OfflineInputs::default());
        assert!(a.forced_offline_by.is_none());
    }
}
