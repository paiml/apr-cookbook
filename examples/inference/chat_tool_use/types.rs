#![allow(
    dead_code,
    unused_imports,
    unreachable_pub,
    clippy::trivially_copy_pass_by_ref,
    clippy::wildcard_imports
)]
#[allow(unused_imports, clippy::wildcard_imports)]
use super::helpers::*;
#[allow(unused_imports)]
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::time::Instant;

// ---------------------------------------------------------------------------
// Message types
// ---------------------------------------------------------------------------

#[derive(Clone, Debug, PartialEq)]
pub enum Role {
    User,
    Assistant,
    Tool,
}

#[derive(Clone, Debug)]
pub struct Message {
    pub role: Role,
    pub content: String,
    pub tool_name: Option<String>,
}

impl Message {
    pub fn user(content: &str) -> Self {
        Self {
            role: Role::User,
            content: content.to_string(),
            tool_name: None,
        }
    }
    pub fn assistant(content: &str) -> Self {
        Self {
            role: Role::Assistant,
            content: content.to_string(),
            tool_name: None,
        }
    }
    pub fn tool(name: &str, content: &str) -> Self {
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
pub struct ToolCall {
    pub name: String,
    pub arguments: String,
}

#[derive(Clone, Debug)]
pub struct ToolResult {
    pub name: String,
    pub output: String,
    pub success: bool,
}

impl ToolResult {
    pub fn ok(name: &str, output: String) -> Self {
        Self {
            name: name.to_string(),
            output,
            success: true,
        }
    }
    pub fn err(name: &str, output: String) -> Self {
        Self {
            name: name.to_string(),
            output,
            success: false,
        }
    }
}

pub struct Tool {
    pub name: &'static str,
    pub description: &'static str,
    pub execute: fn(&str) -> ToolResult,
}

pub struct ToolRegistry {
    pub tools: Vec<Tool>,
}

impl ToolRegistry {
    pub fn new() -> Self {
        Self { tools: Vec::new() }
    }
    pub fn register(&mut self, tool: Tool) {
        self.tools.push(tool);
    }

    pub fn find(&self, name: &str) -> Option<&Tool> {
        self.tools.iter().find(|t| t.name == name)
    }
    pub fn names(&self) -> Vec<&str> {
        self.tools.iter().map(|t| t.name).collect()
    }

    pub fn dispatch(&self, call: &ToolCall) -> ToolResult {
        match self.find(&call.name) {
            Some(tool) => (tool.execute)(&call.arguments),
            None => ToolResult::err(&call.name, format!("unknown tool '{}'", call.name)),
        }
    }
}

// ---------------------------------------------------------------------------
// Built-in tools
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Classifier: keyword-based tool routing
// ---------------------------------------------------------------------------

pub const TOOL_KEYWORDS: [(&str, &str); 12] = [
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

// ---------------------------------------------------------------------------
// Chat engine
// ---------------------------------------------------------------------------

pub struct ChatEngine {
    pub registry: ToolRegistry,
    pub history: Vec<Message>,
    pub seed: u64,
}

impl ChatEngine {
    pub fn new(registry: ToolRegistry, seed: u64) -> Self {
        Self {
            registry,
            history: Vec::new(),
            seed,
        }
    }

    pub fn process_message(&mut self, user_input: &str) -> String {
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

    pub fn generate_direct_response(&self, input: &str) -> String {
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

    pub fn synthesize_response(results: &[ToolResult]) -> String {
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

    pub fn message_count(&self) -> usize {
        self.history.len()
    }

    pub fn count_role(&self, role: &Role) -> usize {
        self.history.iter().filter(|m| m.role == *role).count()
    }

    #[cfg(test)]
    pub fn last_message(&self) -> Option<&Message> {
        self.history.last()
    }
}

// ---------------------------------------------------------------------------
// Section helpers (extracted from main to reduce cyclomatic complexity)
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------
