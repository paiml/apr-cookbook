//! # apr mcp — JSON-RPC Error Code Mapper
//!
//! `apr mcp` returns standard JSON-RPC error codes per the spec (-32700
//! parse error, -32600 invalid request, -32601 method not found, -32602
//! invalid params, -32603 internal error). This recipe maps internal
//! error categories to standard codes and asserts the contract.
//!
//! Demonstrates the **MCP.7** recipe for PMAT-107 (apr mcp coverage).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: aprender MCP-003 + JSON-RPC 2.0 spec §5.1
//!
//! Run with: cargo run --example cli_mcp_error_response_codes
//!
//! Added by PMAT-107 (expand-cookbooks followup).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InternalError {
    InvalidJson,
    UnknownMethod,
    BadArguments,
    UnexpectedFailure,
    AuthRequired, // server-defined: -32000
    RateLimited,  // server-defined: -32001
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ErrorResponse {
    pub code: i32,
    pub message: &'static str,
}

pub fn map_error(e: InternalError) -> ErrorResponse {
    match e {
        InternalError::InvalidJson => ErrorResponse {
            code: -32700,
            message: "Parse error",
        },
        InternalError::UnknownMethod => ErrorResponse {
            code: -32601,
            message: "Method not found",
        },
        InternalError::BadArguments => ErrorResponse {
            code: -32602,
            message: "Invalid params",
        },
        InternalError::UnexpectedFailure => ErrorResponse {
            code: -32603,
            message: "Internal error",
        },
        InternalError::AuthRequired => ErrorResponse {
            code: -32000,
            message: "Authentication required",
        },
        InternalError::RateLimited => ErrorResponse {
            code: -32001,
            message: "Rate limit exceeded",
        },
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("cli_mcp_error_response_codes")?;

    for e in [
        InternalError::InvalidJson,
        InternalError::UnknownMethod,
        InternalError::BadArguments,
        InternalError::UnexpectedFailure,
        InternalError::AuthRequired,
        InternalError::RateLimited,
    ] {
        let r = map_error(e);
        println!("{e:?}  →  code {} \"{}\"", r.code, r.message);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mapper_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn parse_error_is_minus_32700() {
        assert_eq!(map_error(InternalError::InvalidJson).code, -32700);
    }

    #[test]
    fn method_not_found_is_minus_32601() {
        assert_eq!(map_error(InternalError::UnknownMethod).code, -32601);
    }

    #[test]
    fn invalid_params_is_minus_32602() {
        assert_eq!(map_error(InternalError::BadArguments).code, -32602);
    }

    #[test]
    fn internal_error_is_minus_32603() {
        assert_eq!(map_error(InternalError::UnexpectedFailure).code, -32603);
    }

    #[test]
    fn server_defined_errors_in_minus_32000_range() {
        // Per spec, -32000 to -32099 reserved for server-defined errors.
        let codes = [
            map_error(InternalError::AuthRequired).code,
            map_error(InternalError::RateLimited).code,
        ];
        for c in codes {
            assert!(
                (-32099..=-32000).contains(&c),
                "code {c} not in server-defined range"
            );
        }
    }

    #[test]
    fn no_two_errors_share_a_code() {
        let codes: Vec<i32> = [
            InternalError::InvalidJson,
            InternalError::UnknownMethod,
            InternalError::BadArguments,
            InternalError::UnexpectedFailure,
            InternalError::AuthRequired,
            InternalError::RateLimited,
        ]
        .iter()
        .map(|e| map_error(*e).code)
        .collect();
        let unique: std::collections::HashSet<i32> = codes.iter().copied().collect();
        assert_eq!(unique.len(), codes.len(), "duplicate codes: {codes:?}");
    }

    #[test]
    fn every_error_has_nonempty_message() {
        for e in [
            InternalError::InvalidJson,
            InternalError::UnknownMethod,
            InternalError::BadArguments,
            InternalError::UnexpectedFailure,
            InternalError::AuthRequired,
            InternalError::RateLimited,
        ] {
            assert!(!map_error(e).message.is_empty());
        }
    }
}
