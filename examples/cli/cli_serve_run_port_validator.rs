//! # apr serve run — `--port` Validator
//!
//! `apr serve run --port <N>` requires a non-privileged port (1024-65535
//! per IANA), and rejects ports already commonly bound by other services
//! (5432 = postgres, 6379 = redis, 8080 = generic dev, 8000 = python).
//! Default 9090 is the recommended apr inference port.
//!
//! Demonstrates the **SERVE-RUN.4** recipe for PMAT-105 (apr serve coverage).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: aprender SERVE-002 + IANA port assignments
//!
//! Run with: cargo run --example cli_serve_run_port_validator
//!
//! Added by PMAT-105 (expand-cookbooks followup).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum PortVerdict {
    Ok,
    PrivilegedPort { port: u16 },
    OutOfRange,
    CommonlyBound { port: u16, service: &'static str },
}

const COMMON_PORTS: &[(u16, &str)] = &[
    (22, "ssh"),
    (80, "http"),
    (443, "https"),
    (3306, "mysql"),
    (5432, "postgres"),
    (6379, "redis"),
    (8000, "python dev"),
    (8080, "generic dev"),
    (11434, "ollama"),
];

pub fn validate_port(port: u16) -> PortVerdict {
    if port == 0 {
        return PortVerdict::OutOfRange;
    }
    if port < 1024 {
        return PortVerdict::PrivilegedPort { port };
    }
    if let Some((p, svc)) = COMMON_PORTS.iter().find(|(p, _)| *p == port) {
        return PortVerdict::CommonlyBound {
            port: *p,
            service: svc,
        };
    }
    PortVerdict::Ok
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("cli_serve_run_port_validator")?;

    for p in [
        0u16, 22, 80, 443, 1023, 1024, 5432, 8080, 9090, 11434, 65535,
    ] {
        println!("--port {p:>5}  →  {:?}", validate_port(p));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn validator_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn privileged_port_rejected() {
        assert!(matches!(
            validate_port(80),
            PortVerdict::PrivilegedPort { .. }
        ));
        assert!(matches!(
            validate_port(443),
            PortVerdict::PrivilegedPort { .. }
        ));
        assert!(matches!(
            validate_port(1023),
            PortVerdict::PrivilegedPort { .. }
        ));
    }

    #[test]
    fn boundary_at_1024_passes() {
        // First non-privileged port — must NOT be flagged as PrivilegedPort.
        assert_eq!(validate_port(1024), PortVerdict::Ok);
    }

    #[test]
    fn port_zero_rejected() {
        assert_eq!(validate_port(0), PortVerdict::OutOfRange);
    }

    #[test]
    fn commonly_bound_port_flagged() {
        // 5432 = postgres — operator probably has it bound already.
        let v = validate_port(5432);
        if let PortVerdict::CommonlyBound { port, service } = v {
            assert_eq!(port, 5432);
            assert_eq!(service, "postgres");
        } else {
            panic!("expected CommonlyBound");
        }
    }

    #[test]
    fn ollama_port_flagged() {
        // Specific to apr workflow — ollama uses 11434 by default.
        assert!(matches!(
            validate_port(11434),
            PortVerdict::CommonlyBound {
                service: "ollama",
                ..
            }
        ));
    }

    #[test]
    fn high_unassigned_port_passes() {
        assert_eq!(validate_port(9090), PortVerdict::Ok);
        assert_eq!(validate_port(65535), PortVerdict::Ok);
    }
}
