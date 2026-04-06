#![allow(
    dead_code,
    unused_imports,
    unreachable_pub,
    clippy::trivially_copy_pass_by_ref,
    clippy::wildcard_imports
)]
#[allow(unused_imports)]
use super::types::*;

#[allow(unused_imports)]
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::time::Instant;

pub fn execute_calculator(args: &str) -> ToolResult {
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

pub fn execute_weather(args: &str) -> ToolResult {
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

pub fn execute_unit_converter(args: &str) -> ToolResult {
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

pub fn classify_tool_calls(message: &str, seed: u64) -> Vec<ToolCall> {
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

pub fn extract_arguments(message: &str, keyword: &str, tool: &str, seed: u64) -> String {
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

pub fn build_registry() -> ToolRegistry {
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

pub fn demo_tool_registration() {
    println!("1. Tool Registration & Discovery");
    println!("   ─────────────────────────────────────────");
    let registry = build_registry();
    println!("   Registered {} tools:", registry.tools.len());
    for (i, tool) in registry.tools.iter().enumerate() {
        println!("   {}. {:16} - {}", i + 1, tool.name, tool.description);
    }
    println!("   Names: {:?}\n", registry.names());
}

pub fn demo_single_turn_dispatch() {
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

pub fn demo_multi_turn_conversation() {
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

pub fn demo_parallel_tool_calls() {
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

pub fn demo_error_handling() {
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

pub fn demo_throughput_benchmark() {
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
