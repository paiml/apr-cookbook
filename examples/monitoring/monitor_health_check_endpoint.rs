//! # Monitoring Health-Check Endpoint Picker
//!
//! Three K8s probe types map to different rules:
//!   /healthz (livez) — am I alive at all? Restart on failure.
//!   /readyz — am I ready for traffic? Drop from load balancer on failure.
//!   /startupz — finished startup? K8s gates the other probes during startup.
//!
//! Picker maps (model_loaded, dependency_ok, startup_complete) →
//! per-endpoint health verdict.
//!
//! Demonstrates the **MON.25** recipe for PMAT-144 (monitoring round 5).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: Kubernetes liveness/readiness/startup probe spec.
//!
//! Run with: cargo run --example monitor_health_check_endpoint
//!
//! Added by PMAT-144 (expand-cookbooks followup).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Endpoint {
    Healthz,
    Readyz,
    Startupz,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Status {
    Ok,
    Failure,
}

#[derive(Debug, PartialEq)]
pub enum HealthVerdict {
    Ok {
        http_status: u16,
        body: &'static str,
    },
    InvalidEndpoint,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ProcessState {
    pub process_alive: bool,
    pub model_loaded: bool,
    pub dependency_ok: bool,
    pub startup_complete: bool,
}

pub fn check(endpoint: Endpoint, state: ProcessState) -> HealthVerdict {
    let status = match endpoint {
        Endpoint::Healthz => {
            if state.process_alive {
                Status::Ok
            } else {
                Status::Failure
            }
        }
        Endpoint::Readyz => {
            if state.process_alive
                && state.model_loaded
                && state.dependency_ok
                && state.startup_complete
            {
                Status::Ok
            } else {
                Status::Failure
            }
        }
        Endpoint::Startupz => {
            if state.startup_complete {
                Status::Ok
            } else {
                Status::Failure
            }
        }
    };
    let (http_status, body) = match status {
        Status::Ok => (200, "OK"),
        Status::Failure => (503, "FAIL"),
    };
    HealthVerdict::Ok { http_status, body }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("monitor_health_check_endpoint")?;

    println!(
        "all healthy: healthz: {:?}",
        check(
            Endpoint::Healthz,
            ProcessState {
                process_alive: true,
                model_loaded: true,
                dependency_ok: true,
                startup_complete: true
            }
        )
    );
    println!(
        "all healthy: readyz: {:?}",
        check(
            Endpoint::Readyz,
            ProcessState {
                process_alive: true,
                model_loaded: true,
                dependency_ok: true,
                startup_complete: true
            }
        )
    );
    println!(
        "model not loaded: readyz: {:?}",
        check(
            Endpoint::Readyz,
            ProcessState {
                process_alive: true,
                model_loaded: false,
                dependency_ok: true,
                startup_complete: true
            }
        )
    );
    println!(
        "process dead: healthz: {:?}",
        check(
            Endpoint::Healthz,
            ProcessState {
                process_alive: false,
                model_loaded: true,
                dependency_ok: true,
                startup_complete: true
            }
        )
    );
    println!(
        "still starting: startupz: {:?}",
        check(
            Endpoint::Startupz,
            ProcessState {
                process_alive: true,
                model_loaded: false,
                dependency_ok: false,
                startup_complete: false
            }
        )
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn checker_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn healthz_only_needs_process_alive() {
        let v = check(
            Endpoint::Healthz,
            ProcessState {
                process_alive: true,
                model_loaded: false,
                dependency_ok: false,
                startup_complete: false,
            },
        );
        if let HealthVerdict::Ok { http_status, .. } = v {
            assert_eq!(http_status, 200);
        }
    }

    #[test]
    fn readyz_needs_everything() {
        let v = check(
            Endpoint::Readyz,
            ProcessState {
                process_alive: true,
                model_loaded: true,
                dependency_ok: true,
                startup_complete: true,
            },
        );
        if let HealthVerdict::Ok { http_status, .. } = v {
            assert_eq!(http_status, 200);
        }
    }

    #[test]
    fn readyz_fails_without_model() {
        let v = check(
            Endpoint::Readyz,
            ProcessState {
                process_alive: true,
                model_loaded: false,
                dependency_ok: true,
                startup_complete: true,
            },
        );
        if let HealthVerdict::Ok { http_status, .. } = v {
            assert_eq!(http_status, 503);
        }
    }

    #[test]
    fn readyz_fails_without_deps() {
        let v = check(
            Endpoint::Readyz,
            ProcessState {
                process_alive: true,
                model_loaded: true,
                dependency_ok: false,
                startup_complete: true,
            },
        );
        if let HealthVerdict::Ok { http_status, .. } = v {
            assert_eq!(http_status, 503);
        }
    }

    #[test]
    fn readyz_fails_without_startup() {
        let v = check(
            Endpoint::Readyz,
            ProcessState {
                process_alive: true,
                model_loaded: true,
                dependency_ok: true,
                startup_complete: false,
            },
        );
        if let HealthVerdict::Ok { http_status, .. } = v {
            assert_eq!(http_status, 503);
        }
    }

    #[test]
    fn healthz_fails_when_process_dead() {
        let v = check(
            Endpoint::Healthz,
            ProcessState {
                process_alive: false,
                model_loaded: true,
                dependency_ok: true,
                startup_complete: true,
            },
        );
        if let HealthVerdict::Ok { http_status, .. } = v {
            assert_eq!(http_status, 503);
        }
    }

    #[test]
    fn startupz_only_cares_about_startup() {
        let v = check(
            Endpoint::Startupz,
            ProcessState {
                process_alive: false,
                model_loaded: false,
                dependency_ok: false,
                startup_complete: true,
            },
        );
        if let HealthVerdict::Ok { http_status, .. } = v {
            assert_eq!(http_status, 200);
        }
    }

    #[test]
    fn startupz_fails_during_startup() {
        let v = check(
            Endpoint::Startupz,
            ProcessState {
                process_alive: true,
                model_loaded: true,
                dependency_ok: true,
                startup_complete: false,
            },
        );
        if let HealthVerdict::Ok { http_status, .. } = v {
            assert_eq!(http_status, 503);
        }
    }

    #[test]
    fn ok_status_returns_ok_body() {
        if let HealthVerdict::Ok { body, .. } = check(
            Endpoint::Healthz,
            ProcessState {
                process_alive: true,
                model_loaded: true,
                dependency_ok: true,
                startup_complete: true,
            },
        ) {
            assert_eq!(body, "OK");
        }
    }

    #[test]
    fn failure_status_returns_fail_body() {
        if let HealthVerdict::Ok { body, .. } = check(
            Endpoint::Healthz,
            ProcessState {
                process_alive: false,
                model_loaded: false,
                dependency_ok: false,
                startup_complete: false,
            },
        ) {
            assert_eq!(body, "FAIL");
        }
    }
}
