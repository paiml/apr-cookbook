#![allow(unused_imports)]
//! Chat Tool Use / Function Calling Example
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Demonstrates a chat system with tool dispatch where the model can invoke
//! registered functions (calculator, weather lookup, unit converter) during
//! a multi-turn conversation and incorporate their results.
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────┐
//! │  User Message                                                   │
//! │       ▼                                                         │
//! │  ┌──────────┐   ┌────────────┐   ┌───────────────────┐        │
//! │  │Classifier │──▶│Tool Router │──▶│ Tool Execution    │        │
//! │  └──────────┘   └────────────┘   └───────────────────┘        │
//! │       │                                   │                    │
//! │       ▼              ┌────────────┐       │                    │
//! │  No-tool path ──────▶│ Assistant   │◀─────┘                    │
//! │                      │ Response    │                           │
//! │                      └────────────┘──▶ Message History         │
//! └─────────────────────────────────────────────────────────────────┘
//! ```
//!
//! # Running
//!
//! ```bash
//! cargo run --example chat_tool_use
//! ```
//!
//!
//! ## Format Variants
//! ```bash
//! apr run model.apr          # APR native format
//! apr run model.gguf         # GGUF (llama.cpp compatible)
//! apr run model.safetensors  # SafeTensors (HuggingFace)
//! ```
//! ## References
//! - Crankshaw, D. et al. (2017). *Clipper: A Low-Latency Online Prediction Serving System*. NSDI. arXiv:1612.03079

use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::time::Instant;

mod helpers;
mod types;
#[allow(unused_imports)]
#[allow(clippy::wildcard_imports)]
use helpers::*;
#[allow(unused_imports)]
#[allow(clippy::wildcard_imports)]
use types::*;

fn main() {
    println!("=== Chat Tool Use / Function Calling Example ===\n");
    demo_tool_registration();
    demo_single_turn_dispatch();
    demo_multi_turn_conversation();
    demo_parallel_tool_calls();
    demo_error_handling();
    demo_throughput_benchmark();
    println!("\n=== Example Complete ===");
}

#[cfg(test)]
mod tests {
    #[allow(unused_imports, clippy::wildcard_imports)]
    use super::helpers::*;
    use super::*;

    #[test]
    fn test_calculator_basic_ops() {
        for (input, expected) in [
            ("3 + 4", "7"),
            ("10 - 3", "7"),
            ("6 * 7", "42"),
            ("100 / 4", "25"),
        ] {
            let r = execute_calculator(input);
            assert!(r.success, "failed: {input}");
            assert_eq!(r.output, expected);
        }
    }

    #[test]
    fn test_calculator_errors() {
        let div0 = execute_calculator("10 / 0");
        assert!(!div0.success);
        assert!(div0.output.contains("division by zero"));
        let bad = execute_calculator("not a number");
        assert!(!bad.success);
        let op = execute_calculator("5 ^ 3");
        assert!(!op.success);
        assert!(op.output.contains("unknown op"));
    }

    #[test]
    fn test_weather_deterministic() {
        let r1 = execute_weather("Tokyo");
        let r2 = execute_weather("Tokyo");
        assert_eq!(r1.output, r2.output);
        assert!(r1.success);
    }

    #[test]
    fn test_weather_varies_by_city() {
        assert_ne!(
            execute_weather("Tokyo").output,
            execute_weather("London").output
        );
    }

    #[test]
    fn test_weather_empty_city() {
        assert!(!execute_weather("").success);
    }

    #[test]
    fn test_unit_converter_known() {
        for input in [
            "100 cm to in",
            "1 km to mi",
            "0 c to f",
            "212 f to c",
            "10 kg to lb",
        ] {
            assert!(execute_unit_converter(input).success, "failed: {input}");
        }
    }

    #[test]
    fn test_unit_converter_errors() {
        assert!(!execute_unit_converter("garbage").success);
        let r = execute_unit_converter("100 foo to bar");
        assert!(!r.success);
        assert!(r.output.contains("unsupported"));
    }

    #[test]
    fn test_unit_converter_round_trip() {
        let r1 = execute_unit_converter("100 km to mi");
        let mi: f64 = r1
            .output
            .split('=')
            .nth(1)
            .unwrap()
            .trim()
            .split_whitespace()
            .next()
            .unwrap()
            .parse()
            .unwrap();
        let r2 = execute_unit_converter(&format!("{mi} mi to km"));
        let km: f64 = r2
            .output
            .split('=')
            .nth(1)
            .unwrap()
            .trim()
            .split_whitespace()
            .next()
            .unwrap()
            .parse()
            .unwrap();
        assert!((km - 100.0).abs() < 0.01, "round-trip: 100 -> {mi} -> {km}");
    }

    #[test]
    fn test_registry_find_and_dispatch() {
        let reg = build_registry();
        assert!(reg.find("calculator").is_some());
        assert!(reg.find("weather").is_some());
        assert!(reg.find("unit_converter").is_some());
        assert!(reg.find("nope").is_none());
        assert_eq!(reg.names().len(), 3);
        let unknown = reg.dispatch(&ToolCall {
            name: "fake".into(),
            arguments: "x".into(),
        });
        assert!(!unknown.success);
        assert!(unknown.output.contains("unknown tool"));
    }

    #[test]
    fn test_classify_routes_correctly() {
        let calc = classify_tool_calls("please calculate 5 + 3", 42);
        assert_eq!(calc[0].name, "calculator");
        let wx = classify_tool_calls("weather in Paris?", 42);
        assert_eq!(wx[0].name, "weather");
        assert!(classify_tool_calls("hello there", 42).is_empty());
    }

    #[test]
    fn test_classify_multiple_tools() {
        let calls = classify_tool_calls("calculate 5+3 and check weather in Tokyo", 42);
        assert!(calls.len() >= 2);
        let names: Vec<&str> = calls.iter().map(|c| c.name.as_str()).collect();
        assert!(names.contains(&"calculator"));
        assert!(names.contains(&"weather"));
    }

    #[test]
    fn test_chat_engine_multi_turn() {
        let mut eng = ChatEngine::new(build_registry(), 42);
        let r1 = eng.process_message("Hello!");
        assert!(!r1.is_empty());
        assert_eq!(eng.count_role(&Role::User), 1);
        let r2 = eng.process_message("Calculate 10 + 20");
        assert!(r2.contains("30"), "expected 30 in: {r2}");
        assert!(eng.count_role(&Role::Tool) >= 1);
    }

    #[test]
    fn test_chat_engine_history() {
        let mut eng = ChatEngine::new(build_registry(), 42);
        eng.process_message("Hi");
        assert_eq!(eng.message_count(), 2);
        eng.process_message("Calculate 1 + 1");
        assert!(eng.message_count() >= 4);
        assert_eq!(eng.last_message().unwrap().role, Role::Assistant);
    }

    #[test]
    fn test_message_constructors() {
        let u = Message::user("hi");
        assert_eq!(u.role, Role::User);
        assert!(u.tool_name.is_none());
        let a = Message::assistant("ok");
        assert_eq!(a.role, Role::Assistant);
        let t = Message::tool("calc", "42");
        assert_eq!(t.tool_name.as_deref(), Some("calc"));
        assert_eq!(t.role, Role::Tool);
    }
}
