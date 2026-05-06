//! # apr serve run — Endpoint Router
//!
//! `apr serve run` exposes a fixed endpoint surface: `POST /v1/completions`,
//! `POST /v1/chat/completions`, `POST /v1/embeddings`, `GET /v1/models`,
//! `GET /healthz`, `GET /metrics`. This recipe builds the router and
//! asserts the contract: known paths route to declared handlers, unknown
//! paths return 404, method mismatches return 405.
//!
//! Demonstrates the **SERVE-RUN.5** recipe for PMAT-105 (apr serve coverage).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: aprender SERVE-003 + OpenAI API compatibility
//!
//! Run with: cargo run --example cli_serve_run_endpoint_router
//!
//! Added by PMAT-105 (expand-cookbooks followup).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Handler {
    Completions,
    ChatCompletions,
    Embeddings,
    ListModels,
    Healthz,
    Metrics,
}

#[derive(Debug, PartialEq)]
pub enum RouteVerdict {
    Match(Handler),
    NotFound,
    MethodNotAllowed { allowed: &'static str },
}

pub fn route(method: &str, path: &str) -> RouteVerdict {
    match path {
        "/v1/completions" => match method {
            "POST" => RouteVerdict::Match(Handler::Completions),
            _ => RouteVerdict::MethodNotAllowed { allowed: "POST" },
        },
        "/v1/chat/completions" => match method {
            "POST" => RouteVerdict::Match(Handler::ChatCompletions),
            _ => RouteVerdict::MethodNotAllowed { allowed: "POST" },
        },
        "/v1/embeddings" => match method {
            "POST" => RouteVerdict::Match(Handler::Embeddings),
            _ => RouteVerdict::MethodNotAllowed { allowed: "POST" },
        },
        "/v1/models" => match method {
            "GET" => RouteVerdict::Match(Handler::ListModels),
            _ => RouteVerdict::MethodNotAllowed { allowed: "GET" },
        },
        "/healthz" => match method {
            "GET" => RouteVerdict::Match(Handler::Healthz),
            _ => RouteVerdict::MethodNotAllowed { allowed: "GET" },
        },
        "/metrics" => match method {
            "GET" => RouteVerdict::Match(Handler::Metrics),
            _ => RouteVerdict::MethodNotAllowed { allowed: "GET" },
        },
        _ => RouteVerdict::NotFound,
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("cli_serve_run_endpoint_router")?;

    let cases = [
        ("POST", "/v1/completions"),
        ("POST", "/v1/chat/completions"),
        ("GET", "/v1/models"),
        ("GET", "/healthz"),
        ("DELETE", "/v1/models"),
        ("GET", "/v1/completions"),
        ("POST", "/admin/shutdown"),
    ];

    for (m, p) in cases {
        println!("{m:>6} {p:>30}  →  {:?}", route(m, p));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn router_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn post_completions_routes() {
        assert_eq!(
            route("POST", "/v1/completions"),
            RouteVerdict::Match(Handler::Completions)
        );
    }

    #[test]
    fn get_models_routes() {
        assert_eq!(
            route("GET", "/v1/models"),
            RouteVerdict::Match(Handler::ListModels)
        );
    }

    #[test]
    fn healthz_and_metrics_route() {
        assert_eq!(
            route("GET", "/healthz"),
            RouteVerdict::Match(Handler::Healthz)
        );
        assert_eq!(
            route("GET", "/metrics"),
            RouteVerdict::Match(Handler::Metrics)
        );
    }

    #[test]
    fn method_mismatch_returns_405() {
        // GET /v1/completions → 405 (only POST allowed).
        let v = route("GET", "/v1/completions");
        assert!(matches!(
            v,
            RouteVerdict::MethodNotAllowed { allowed: "POST" }
        ));
    }

    #[test]
    fn delete_on_models_returns_405() {
        let v = route("DELETE", "/v1/models");
        assert!(matches!(
            v,
            RouteVerdict::MethodNotAllowed { allowed: "GET" }
        ));
    }

    #[test]
    fn unknown_path_returns_404() {
        assert_eq!(route("POST", "/admin/shutdown"), RouteVerdict::NotFound);
        assert_eq!(route("GET", "/api/v1/foo"), RouteVerdict::NotFound);
    }

    #[test]
    fn empty_path_returns_404() {
        assert_eq!(route("GET", ""), RouteVerdict::NotFound);
    }

    #[test]
    fn case_sensitive_path_matching() {
        // /healthz != /Healthz
        assert_eq!(route("GET", "/Healthz"), RouteVerdict::NotFound);
    }
}
