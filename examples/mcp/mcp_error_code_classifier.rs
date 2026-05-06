//! # MCP JSON-RPC Error Code Classifier
//!
//! JSON-RPC 2.0 reserves -32768..=-32000 for the protocol; -32099..=-32000
//! is the "implementation-defined server-errors" range. Standard codes:
//! -32700 ParseError, -32600 InvalidRequest, -32601 MethodNotFound,
//! -32602 InvalidParams, -32603 InternalError. Anything outside the
//! reserved band is application-domain. This recipe builds the
//! classifier.
//!
//! Demonstrates the **MCP.16** recipe for PMAT-135 (mcp coverage).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: JSON-RPC 2.0 specification § 5.1 (Error Object).
//!
//! Run with: cargo run --example mcp_error_code_classifier
//!
//! Added by PMAT-135 (expand-cookbooks followup).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ErrorClass {
    ParseError,
    InvalidRequest,
    MethodNotFound,
    InvalidParams,
    InternalError,
    ServerError,
    ApplicationError,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Retryability {
    Retry,
    DoNotRetry,
    BackoffRetry,
}

pub fn classify(code: i32) -> ErrorClass {
    match code {
        -32700 => ErrorClass::ParseError,
        -32600 => ErrorClass::InvalidRequest,
        -32601 => ErrorClass::MethodNotFound,
        -32602 => ErrorClass::InvalidParams,
        -32603 => ErrorClass::InternalError,
        -32099..=-32000 => ErrorClass::ServerError,
        _ => ErrorClass::ApplicationError,
    }
}

pub fn retryable(class: ErrorClass) -> Retryability {
    match class {
        ErrorClass::InternalError | ErrorClass::ServerError => Retryability::BackoffRetry,
        ErrorClass::ParseError
        | ErrorClass::InvalidRequest
        | ErrorClass::MethodNotFound
        | ErrorClass::InvalidParams => Retryability::DoNotRetry,
        ErrorClass::ApplicationError => Retryability::Retry,
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("mcp_error_code_classifier")?;

    for code in [-32700, -32600, -32601, -32602, -32603, -32050, -1, 1000] {
        let cls = classify(code);
        println!("{code:>6} → {cls:?} ({:?})", retryable(cls));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn classifier_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn parse_error_classified() {
        assert_eq!(classify(-32700), ErrorClass::ParseError);
    }

    #[test]
    fn invalid_request_classified() {
        assert_eq!(classify(-32600), ErrorClass::InvalidRequest);
    }

    #[test]
    fn method_not_found_classified() {
        assert_eq!(classify(-32601), ErrorClass::MethodNotFound);
    }

    #[test]
    fn invalid_params_classified() {
        assert_eq!(classify(-32602), ErrorClass::InvalidParams);
    }

    #[test]
    fn internal_error_classified() {
        assert_eq!(classify(-32603), ErrorClass::InternalError);
    }

    #[test]
    fn server_error_band_classified() {
        assert_eq!(classify(-32050), ErrorClass::ServerError);
        assert_eq!(classify(-32099), ErrorClass::ServerError);
        assert_eq!(classify(-32000), ErrorClass::ServerError);
    }

    #[test]
    fn application_error_outside_reserved_band() {
        assert_eq!(classify(-1), ErrorClass::ApplicationError);
        assert_eq!(classify(1000), ErrorClass::ApplicationError);
        assert_eq!(classify(-31999), ErrorClass::ApplicationError);
    }

    #[test]
    fn parse_error_not_retryable() {
        assert_eq!(retryable(ErrorClass::ParseError), Retryability::DoNotRetry);
    }

    #[test]
    fn internal_error_backoff_retry() {
        assert_eq!(
            retryable(ErrorClass::InternalError),
            Retryability::BackoffRetry
        );
    }

    #[test]
    fn server_error_backoff_retry() {
        assert_eq!(
            retryable(ErrorClass::ServerError),
            Retryability::BackoffRetry
        );
    }

    #[test]
    fn application_error_retry() {
        assert_eq!(retryable(ErrorClass::ApplicationError), Retryability::Retry);
    }

    #[test]
    fn invalid_params_not_retryable() {
        // Bad params won't get better on retry.
        assert_eq!(
            retryable(ErrorClass::InvalidParams),
            Retryability::DoNotRetry
        );
    }
}
