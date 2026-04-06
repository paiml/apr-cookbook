//! Chat Tool Use / Function Calling Example
//!
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

// ---------------------------------------------------------------------------
// Message types
// ---------------------------------------------------------------------------

#[derive(Clone, Debug, PartialEq)]
enum Role {
    User,
    Assistant,
    Tool,
}

#[derive(Clone, Debug)]
struct Message {
    role: Role,
    content: String,
    tool_name: Option<String>,
}

impl Message {
    fn user(content: &str) -> Self {
        Self {
            role: Role::User,
            content: content.to_string(),
            tool_name: None,
        }
    }
    fn assistant(content: &str) -> Self {
        Self {
            role: Role::Assistant,
            content: content.to_string(),
            tool_name: None,
        }
    }
    fn tool(name: &str, content: &str) -> Self {
        Self {
            role: Role::Tool,
            content: content.to_string(),
            tool_name: Some(name.to_string()),
        }
    }
}

// ---------------------------------------------------------------------------
// Tool infrastructure
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
struct ToolCall {
    name: String,
    arguments: String,
}

#[derive(Clone, Debug)]
struct ToolResult {
    name: String,
    output: String,
    success: bool,
}

impl ToolResult {
    fn ok(name: &str, output: String) -> Self {
        Self {
            name: name.to_string(),
            output,
            success: true,
        }
    }
    fn err(name: &str, output: String) -> Self {
        Self {
            name: name.to_string(),
            output,
            success: false,
        }
    }
}

struct Tool {
    name: &'static str,
    description: &'static str,
    execute: fn(&str) -> ToolResult,
}

struct ToolRegistry {
    tools: Vec<Tool>,
}

impl ToolRegistry {
    fn new() -> Self {
        Self { tools: Vec::new() }
    }
    fn register(&mut self, tool: Tool) {
        self.tools.push(tool);
    }

    fn find(&self, name: &str) -> Option<&Tool> {
        self.tools.iter().find(|t| t.name == name)
    }
    fn names(&self) -> Vec<&str> {
        self.tools.iter().map(|t| t.name).collect()
    }

    fn dispatch(&self, call: &ToolCall) -> ToolResult {
        match self.find(&call.name) {
            Some(tool) => (tool.execute)(&call.arguments),
            None => ToolResult::err(&call.name, format!("unknown tool '{}'", call.name)),
        }
    }
}

// ---------------------------------------------------------------------------
// Built-in tools
// ---------------------------------------------------------------------------

fn execute_calculator(args: &str) -> ToolResult {
    let parts: Vec<&str> = args.split_whitespace().collect();
    if parts.len() != 3 {
        return ToolResult::err("calculator", "expected 'a op b'".into());
    }
    let a: f64 = match parts[0].parse() {
        Ok(v) => v,
        Err(_) => return ToolResult::err("calculator", format!("bad number '{}'", parts[0])),
    };
    let b: f64 = match parts[2].parse() {
        Ok(v) => v,
        Err(_) => return ToolResult::err("calculator", format!("bad number '{}'", parts[2])),
    };
    let r = match parts[1] {
        "+" => a + b,
        "-" => a - b,
        "*" => a * b,
        "/" if b == 0.0 => return ToolResult::err("calculator", "division by zero".into()),
        "/" => a / b,
        op => return ToolResult::err("calculator", format!("unknown op '{op}'")),
    };
    ToolResult::ok("calculator", format!("{r}"))
}

fn execute_weather(args: &str) -> ToolResult {
    let city = args.trim();
    if city.is_empty() {
        return ToolResult::err("weather", "city name required".into());
    }
    let mut hasher = DefaultHasher::new();
    city.to_lowercase().hash(&mut hasher);
    let h = hasher.finish();
    let temp_c = (h % 45) as i32 - 10;
    let conds = [
        "sunny",
        "cloudy",
        "rainy",
        "partly cloudy",
        "windy",
        "foggy",
    ];
    let cond = conds[(h / 45 % conds.len() as u64) as usize];
    let humidity = 30 + (h / 1000 % 60);
    ToolResult::ok(
        "weather",
        format!("{city}: {temp_c}C, {cond}, {humidity}% humidity"),
    )
}

fn execute_unit_converter(args: &str) -> ToolResult {
    let parts: Vec<&str> = args.split_whitespace().collect();
    if parts.len() != 4 || parts[2] != "to" {
        return ToolResult::err("unit_converter", "expected 'value unit to unit'".into());
    }
    let value: f64 = match parts[0].parse() {
        Ok(v) => v,
        Err(_) => return ToolResult::err("unit_converter", format!("bad number '{}'", parts[0])),
    };
    let (from, to) = (parts[1], parts[3]);
    let converted = match (from, to) {
        ("km", "mi") => Some(value * 0.621_371),
        ("mi", "km") => Some(value * 1.609_344),
        ("cm", "in") => Some(value * 0.393_701),
        ("in", "cm") => Some(value * 2.54),
        ("kg", "lb") => Some(value * 2.204_623),
        ("lb", "kg") => Some(value * 0.453_592),
        ("c", "f") => Some(value * 1.8 + 32.0),
        ("f", "c") => Some((value - 32.0) / 1.8),
        ("m", "ft") => Some(value * 3.280_84),
        ("ft", "m") => Some(value * 0.3048),
        _ => None,
    };
    match converted {
        Some(r) => ToolResult::ok("unit_converter", format!("{value} {from} = {r:.4} {to}")),
        None => ToolResult::err("unit_converter", format!("unsupported '{from}' to '{to}'")),
    }
}

// ---------------------------------------------------------------------------
// Classifier: keyword-based tool routing
// ---------------------------------------------------------------------------

const TOOL_KEYWORDS: [(&str, &str); 12] = [
    ("calculate", "calculator"),
    ("compute", "calculator"),
    ("math", "calculator"),
    ("plus", "calculator"),
    ("weather", "weather"),
    ("temperature in", "weather"),
    ("forecast", "weather"),
    ("climate", "weather"),
    ("convert", "unit_converter"),
    ("how many", "unit_converter"),
    ("miles", "unit_converter"),
    ("kilometers", "unit_converter"),
];

fn classify_tool_calls(message: &str, seed: u64) -> Vec<ToolCall> {
    let lower = message.to_lowercase();
    let mut calls = Vec::new();
    let mut seen: Vec<&str> = Vec::new();
    for &(keyword, tool) in &TOOL_KEYWORDS {
        if lower.contains(keyword) && !seen.contains(&tool) {
            let args = extract_arguments(&lower, keyword, tool, seed);
            calls.push(ToolCall {
                name: tool.to_string(),
                arguments: args,
            });
            seen.push(tool);
        }
    }
    calls
}

fn extract_arguments(message: &str, keyword: &str, tool: &str, seed: u64) -> String {
    let after = message
        .find(keyword)
        .map_or(message, |pos| &message[pos + keyword.len()..]);
    let trimmed = after.trim().trim_start_matches(':').trim();
    match tool {
        "calculator" => {
            let c = trimmed.replace("what is", "").replace("what's", "");
            let c = c.trim();
            if c.is_empty() {
                let mut h = DefaultHasher::new();
                (seed, message).hash(&mut h);
                let v = h.finish();
                format!("{} + {}", v % 100, (v / 100) % 50 + 1)
            } else {
                c.to_string()
            }
        }
        "weather" => {
            let c = trimmed
                .replace("in ", "")
                .replace("for ", "")
                .replace("at ", "")
                .replace('?', "");
            let c = c.trim();
            if c.is_empty() {
                "London".into()
            } else {
                c.to_string()
            }
        }
        "unit_converter" => {
            let c = trimmed.replace('?', "");
            let c = c.trim();
            if c.is_empty() {
                "100 cm to in".into()
            } else {
                c.to_string()
            }
        }
        _ => trimmed.to_string(),
    }
}

// ---------------------------------------------------------------------------
// Chat engine
// ---------------------------------------------------------------------------

struct ChatEngine {
    registry: ToolRegistry,
    history: Vec<Message>,
    seed: u64,
}

impl ChatEngine {
    fn new(registry: ToolRegistry, seed: u64) -> Self {
        Self {
            registry,
            history: Vec::new(),
            seed,
        }
    }

    fn process_message(&mut self, user_input: &str) -> String {
        self.history.push(Message::user(user_input));
        let tool_calls = classify_tool_calls(user_input, self.seed);
        if tool_calls.is_empty() {
            let resp = self.generate_direct_response(user_input);
            self.history.push(Message::assistant(&resp));
            return resp;
        }
        let mut outputs = Vec::new();
        for call in &tool_calls {
            let result = self.registry.dispatch(call);
            self.history
                .push(Message::tool(&result.name, &result.output));
            outputs.push(result);
        }
        let resp = Self::synthesize_response(&outputs);
        self.history.push(Message::assistant(&resp));
        resp
    }

    fn generate_direct_response(&self, input: &str) -> String {
        let mut hasher = DefaultHasher::new();
        (self.seed, input, self.history.len()).hash(&mut hasher);
        let h = hasher.finish();
        let responses = [
            "I can help with calculations, weather lookups, and unit conversions.",
            "Could you be more specific? I have tools for math, weather, and conversions.",
            "I understand. Let me know if you need any calculations or lookups.",
            "Try asking me to calculate, check weather, or convert units.",
        ];
        responses[(h % responses.len() as u64) as usize].to_string()
    }

    fn synthesize_response(results: &[ToolResult]) -> String {
        let parts: Vec<String> = results
            .iter()
            .map(|r| {
                if r.success {
                    format!("[{}] {}", r.name, r.output)
                } else {
                    format!("[{}] failed: {}", r.name, r.output)
                }
            })
            .collect();
        parts.join(" | ")
    }

    fn message_count(&self) -> usize {
        self.history.len()
    }

    fn count_role(&self, role: &Role) -> usize {
        self.history.iter().filter(|m| m.role == *role).count()
    }

    #[cfg(test)]
    fn last_message(&self) -> Option<&Message> {
        self.history.last()
    }
}

fn build_registry() -> ToolRegistry {
    let mut r = ToolRegistry::new();
    r.register(Tool {
        name: "calculator",
        description: "Evaluate 'a op b' where op is +,-,*,/",
        execute: execute_calculator,
    });
    r.register(Tool {
        name: "weather",
        description: "Look up current weather for a city",
        execute: execute_weather,
    });
    r.register(Tool {
        name: "unit_converter",
        description: "Convert units: 'value from to target'",
        execute: execute_unit_converter,
    });
    r
}

// ---------------------------------------------------------------------------
// Section helpers (extracted from main to reduce cyclomatic complexity)
// ---------------------------------------------------------------------------

fn demo_tool_registration() {
    println!("1. Tool Registration & Discovery");
    println!("   ─────────────────────────────────────────");
    let registry = build_registry();
    println!("   Registered {} tools:", registry.tools.len());
    for (i, tool) in registry.tools.iter().enumerate() {
        println!("   {}. {:16} - {}", i + 1, tool.name, tool.description);
    }
    println!("   Names: {:?}\n", registry.names());
}

fn demo_single_turn_dispatch() {
    println!("2. Single-Turn Tool Dispatch");
    println!("   ─────────────────────────────────────────");
    let registry = build_registry();
    let dispatches = [
        ("calculator", "15 * 7"),
        ("calculator", "100 / 3"),
        ("weather", "Tokyo"),
        ("weather", "New York"),
        ("unit_converter", "42 km to mi"),
        ("unit_converter", "98.6 f to c"),
    ];
    for (name, args) in dispatches {
        let call = ToolCall {
            name: name.to_string(),
            arguments: args.to_string(),
        };
        let r = registry.dispatch(&call);
        let tag = if r.success { " OK" } else { "ERR" };
        println!("   [{tag}] {name}({args}) -> {}", r.output);
    }
    println!();
}

fn demo_multi_turn_conversation() {
    println!("3. Multi-Turn Conversation");
    println!("   ─────────────────────────────────────────");
    let mut engine = ChatEngine::new(build_registry(), 42);
    let turns = [
        "Hello, what can you do?",
        "Calculate 256 * 3",
        "What's the weather in Paris?",
        "Now convert 30 c to f",
        "Thanks, that's all!",
    ];
    for (i, msg) in turns.iter().enumerate() {
        let resp = engine.process_message(msg);
        println!("   Turn {}: User: {msg:?}", i + 1);
        println!("           Asst: {resp:?}");
        println!(
            "           [msgs={}, user={}, tool={}]",
            engine.message_count(),
            engine.count_role(&Role::User),
            engine.count_role(&Role::Tool)
        );
    }
    println!();
}

fn demo_parallel_tool_calls() {
    println!("4. Parallel Tool Calls");
    println!("   ─────────────────────────────────────────");
    let mut engine = ChatEngine::new(build_registry(), 99);
    let queries = [
        "Calculate 50 + 25 and check the weather in London",
        "Convert 10 miles to kilometers and compute 7 * 8",
    ];
    for (i, query) in queries.iter().enumerate() {
        let resp = engine.process_message(query);
        println!("   Query {}: {query:?}", i + 1);
        println!("   Response: {resp:?}");
        let tool_msgs: Vec<_> = engine
            .history
            .iter()
            .filter(|m| m.role == Role::Tool)
            .collect();
        for (j, tm) in tool_msgs.iter().enumerate() {
            println!(
                "     {}. [{}] {}",
                j + 1,
                tm.tool_name.as_deref().unwrap_or("?"),
                tm.content
            );
        }
        println!();
    }
}

fn demo_error_handling() {
    println!("5. Error Handling & Fallbacks");
    println!("   ─────────────────────────────────────────");
    let registry = build_registry();
    let errors: [(&str, &str); 5] = [
        ("calculator", "not a number"),
        ("calculator", "10 / 0"),
        ("weather", ""),
        ("unit_converter", "bad"),
        ("nonexistent_tool", "hello"),
    ];
    for (name, args) in errors {
        let call = ToolCall {
            name: name.to_string(),
            arguments: args.to_string(),
        };
        let r = registry.dispatch(&call);
        let tag = if r.success { " OK" } else { "ERR" };
        println!("   [{tag}] {name}({args:>16}) -> {}", r.output);
    }
    println!();
}

fn demo_throughput_benchmark() {
    println!("6. Throughput Benchmark");
    println!("   ─────────────────────────────────────────");
    let n_conv = 100;
    let templates = [
        "Calculate 42 + 17",
        "What's the weather in Berlin?",
        "Convert 100 km to mi",
        "Compute 99 * 11",
        "Weather forecast for Seattle",
        "How many miles is 50 kilometers?",
    ];
    let start = Instant::now();
    let mut total_turns = 0u32;
    for c in 0..n_conv {
        let mut eng = ChatEngine::new(build_registry(), c as u64);
        for template in &templates {
            eng.process_message(template);
            total_turns += 1;
        }
    }
    let elapsed = start.elapsed();
    let tps = f64::from(total_turns) / elapsed.as_secs_f64();
    println!("   Conversations:  {n_conv}");
    println!("   Turns/conv:     {}", templates.len());
    println!("   Total turns:    {total_turns}");
    println!("   Time:           {} ms", elapsed.as_millis());
    println!("   Throughput:     {tps:.0} turns/sec");
    println!(
        "   Avg latency:    {:.1} us/turn",
        elapsed.as_micros() as f64 / f64::from(total_turns)
    );
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

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
