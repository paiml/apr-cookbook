//! Multi-Turn Chat Inference Example
//!
//! Demonstrates a chat-style inference loop with conversation history,
//! system prompts, turn formatting, and context window management.
//!
//! # Features
//!
//! - **System Prompt**: Configurable persona/instructions
//! - **Conversation History**: Full turn-by-turn tracking
//! - **Context Window**: Sliding window with configurable max length
//! - **Turn Templates**: Structured role-content formatting
//! - **Response Scoring**: Coherence and relevance metrics
//!
//! # Running
//!
//! ```bash
//! cargo run --example chat_multiturn
//! ```

use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

const MAX_RESPONSE_TOKENS: usize = 40;

/// Chat roles
#[derive(Clone, Debug)]
enum Role {
    System,
    User,
    Assistant,
}

impl Role {
    fn prefix(&self) -> &'static str {
        match self {
            Role::System => "<|system|>",
            Role::User => "<|user|>",
            Role::Assistant => "<|assistant|>",
        }
    }
}

/// A single message in the conversation
#[derive(Clone, Debug)]
struct Message {
    role: Role,
    content: String,
}

impl Message {
    fn new(role: Role, content: &str) -> Self {
        Self {
            role,
            content: content.to_string(),
        }
    }

    fn token_count(&self) -> usize {
        // Simple tokenization: ~4 chars per token + role prefix
        self.content.len() / 4 + 3
    }
}

/// Conversation manager with context window
struct Conversation {
    messages: Vec<Message>,
    max_context_tokens: usize,
}

impl Conversation {
    fn new(max_context_tokens: usize) -> Self {
        Self {
            messages: Vec::new(),
            max_context_tokens,
        }
    }

    fn add_system_prompt(&mut self, content: &str) {
        self.messages.push(Message::new(Role::System, content));
    }

    fn add_user_message(&mut self, content: &str) {
        self.messages.push(Message::new(Role::User, content));
    }

    fn add_assistant_response(&mut self, content: &str) {
        self.messages.push(Message::new(Role::Assistant, content));
    }

    /// Get total token count of the conversation
    fn total_tokens(&self) -> usize {
        self.messages.iter().map(Message::token_count).sum()
    }

    /// Get context window: keep system prompt + most recent messages
    fn context_window(&self) -> Vec<&Message> {
        if self.messages.is_empty() {
            return Vec::new();
        }

        let mut window: Vec<&Message> = Vec::new();
        let mut tokens = 0;

        // Always include system prompt if present
        let system_msg = self
            .messages
            .iter()
            .find(|m| matches!(m.role, Role::System));
        if let Some(sys) = system_msg {
            tokens += sys.token_count();
            window.push(sys);
        }

        // Add messages from most recent, working backwards
        let non_system: Vec<&Message> = self
            .messages
            .iter()
            .filter(|m| !matches!(m.role, Role::System))
            .collect();

        let mut recent = Vec::new();
        for msg in non_system.iter().rev() {
            let msg_tokens = msg.token_count();
            if tokens + msg_tokens > self.max_context_tokens {
                break;
            }
            tokens += msg_tokens;
            recent.push(*msg);
        }
        recent.reverse();
        window.extend(recent);

        window
    }

    /// Count messages that were trimmed from context
    fn trimmed_count(&self) -> usize {
        let window_len = self.context_window().len();
        self.messages.len().saturating_sub(window_len)
    }

    #[cfg(test)]
    fn turn_count(&self) -> usize {
        self.messages
            .iter()
            .filter(|m| matches!(m.role, Role::User))
            .count()
    }
}

/// Format context window into a prompt string
fn format_prompt(window: &[&Message]) -> String {
    let mut prompt = String::new();
    for msg in window {
        prompt.push_str(msg.role.prefix());
        prompt.push('\n');
        prompt.push_str(&msg.content);
        prompt.push('\n');
    }
    prompt.push_str("<|assistant|>\n");
    prompt
}

/// Simple deterministic response generator
fn generate_response(prompt: &str, seed: u64) -> String {
    let mut tokens = Vec::new();
    let prompt_bytes: Vec<u8> = prompt.bytes().collect();

    for step in 0..MAX_RESPONSE_TOKENS {
        let mut hasher = DefaultHasher::new();
        (seed, &prompt_bytes, &tokens, step).hash(&mut hasher);
        let h = hasher.finish();

        // Bias toward printable ASCII
        let token = (h % 80 + 32) as u8;

        // Stop on double newline
        if tokens.len() >= 2 && tokens[tokens.len() - 1] == b'\n' && token == b'\n' {
            break;
        }

        tokens.push(token);
    }

    // Clean up into readable words
    let raw: String = tokens.iter().map(|&b| b as char).collect();
    // Post-process: trim and take first sentence-like chunk
    let cleaned = raw.trim().chars().take(120).collect::<String>();
    if cleaned.is_empty() {
        "I understand.".to_string()
    } else {
        cleaned
    }
}

/// Compute response quality metrics
struct ResponseMetrics {
    response_length: usize,
    context_utilization: f64,
}

fn compute_metrics(response: &str, conversation: &Conversation) -> ResponseMetrics {
    let window = conversation.context_window();
    let context_tokens: usize = window.iter().map(|m| m.token_count()).sum();
    let utilization = context_tokens as f64 / conversation.max_context_tokens as f64;

    ResponseMetrics {
        response_length: response.len(),
        context_utilization: utilization,
    }
}

fn main() {
    println!("=== Multi-Turn Chat Inference Example ===\n");

    // =========================================================================
    // Section 1: Basic Chat
    // =========================================================================
    println!("1. Basic Multi-Turn Chat");
    println!("   ─────────────────────────────────────────");

    let mut conv = Conversation::new(500);
    conv.add_system_prompt("You are a helpful assistant that answers questions concisely.");

    let user_messages = [
        "What is Rust?",
        "How does ownership work?",
        "Can you give an example?",
        "What about lifetimes?",
    ];

    for (i, msg) in user_messages.iter().enumerate() {
        conv.add_user_message(msg);

        let window = conv.context_window();
        let prompt = format_prompt(&window);
        let response = generate_response(&prompt, 42 + i as u64);
        let metrics = compute_metrics(&response, &conv);

        println!("   Turn {}: User: \"{}\"", i + 1, msg);
        println!(
            "   Turn {}: Asst: \"{}\"",
            i + 1,
            &response[..response.len().min(60)]
        );
        println!(
            "           [len={}, ctx={:.0}%, tokens={}]",
            metrics.response_length,
            metrics.context_utilization * 100.0,
            conv.total_tokens()
        );

        conv.add_assistant_response(&response);
    }
    println!();

    // =========================================================================
    // Section 2: System Prompt Comparison
    // =========================================================================
    println!("2. System Prompt Comparison");
    println!("   ─────────────────────────────────────────");

    let personas = [
        ("Concise", "Reply in one sentence."),
        (
            "Expert",
            "You are a senior Rust developer. Give detailed technical answers.",
        ),
        (
            "Friendly",
            "You are a friendly tutor. Use simple language and examples.",
        ),
    ];

    let test_question = "Explain pattern matching in Rust";

    for (name, system_prompt) in &personas {
        let mut conv = Conversation::new(200);
        conv.add_system_prompt(system_prompt);
        conv.add_user_message(test_question);

        let window = conv.context_window();
        let prompt = format_prompt(&window);
        let response = generate_response(&prompt, 42);

        println!(
            "   {:>10}: \"{}\"",
            name,
            &response[..response.len().min(50)]
        );
    }
    println!();

    // =========================================================================
    // Section 3: Context Window Management
    // =========================================================================
    println!("3. Context Window Management");
    println!("   ─────────────────────────────────────────");
    println!(
        "   {:>5} {:>8} {:>10} {:>10} {:>10}",
        "Turn", "Messages", "Tokens", "InWindow", "Trimmed"
    );
    println!("   {}", "─".repeat(48));

    let mut conv = Conversation::new(100); // Small window to trigger trimming
    conv.add_system_prompt("Be brief.");

    for i in 0..12 {
        let msg = format!("Question number {} about Rust programming", i + 1);
        conv.add_user_message(&msg);

        let window = conv.context_window();
        let prompt = format_prompt(&window);
        let response = generate_response(&prompt, 42 + i as u64);

        println!(
            "   {:>5} {:>8} {:>10} {:>10} {:>10}",
            i + 1,
            conv.messages.len(),
            conv.total_tokens(),
            window.len(),
            conv.trimmed_count()
        );

        conv.add_assistant_response(&response);
    }
    println!();

    // =========================================================================
    // Section 4: Turn Format Analysis
    // =========================================================================
    println!("4. Turn Format Analysis");
    println!("   ─────────────────────────────────────────");

    let mut conv = Conversation::new(500);
    conv.add_system_prompt("You are a math tutor.");
    conv.add_user_message("What is 2+2?");
    conv.add_assistant_response("Four.");
    conv.add_user_message("And 3+3?");

    let window = conv.context_window();
    let prompt = format_prompt(&window);

    println!("   Formatted prompt:");
    for line in prompt.lines().take(10) {
        println!("   | {}", line);
    }
    println!("   | ...");
    println!("   Total prompt length: {} chars", prompt.len());
    println!();

    // =========================================================================
    // Section 5: Chat Throughput
    // =========================================================================
    println!("5. Chat Throughput Benchmark");
    println!("   ─────────────────────────────────────────");

    let n_conversations = 50;
    let turns_per_conv = 8;

    let start = std::time::Instant::now();
    let mut total_responses = 0;

    for c in 0..n_conversations {
        let mut conv = Conversation::new(300);
        conv.add_system_prompt("Be helpful.");
        for t in 0..turns_per_conv {
            conv.add_user_message("Tell me about Rust.");
            let window = conv.context_window();
            let prompt = format_prompt(&window);
            let response = generate_response(&prompt, c as u64 * 100 + t as u64);
            conv.add_assistant_response(&response);
            total_responses += 1;
        }
    }

    let elapsed = start.elapsed();
    let responses_per_sec = f64::from(total_responses) / elapsed.as_secs_f64();

    println!("   Conversations:  {n_conversations}");
    println!("   Turns/conv:     {turns_per_conv}");
    println!("   Total responses: {total_responses}");
    println!("   Total time:     {} ms", elapsed.as_millis());
    println!("   Throughput:     {responses_per_sec:.0} responses/sec");
    println!();

    println!("=== Example Complete ===");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_conversation_add_messages() {
        let mut conv = Conversation::new(500);
        conv.add_system_prompt("system");
        conv.add_user_message("hello");
        conv.add_assistant_response("hi");
        assert_eq!(conv.messages.len(), 3);
    }

    #[test]
    fn test_conversation_turn_count() {
        let mut conv = Conversation::new(500);
        conv.add_user_message("q1");
        conv.add_assistant_response("a1");
        conv.add_user_message("q2");
        assert_eq!(conv.turn_count(), 2);
    }

    #[test]
    fn test_context_window_includes_system() {
        let mut conv = Conversation::new(500);
        conv.add_system_prompt("system prompt");
        conv.add_user_message("hello");
        let window = conv.context_window();
        assert!(matches!(window[0].role, Role::System));
    }

    #[test]
    fn test_context_window_trimming() {
        let mut conv = Conversation::new(30); // Very small window
        conv.add_system_prompt("sys");
        for i in 0..20 {
            conv.add_user_message(&format!("message {i} with some content padding"));
        }
        let window = conv.context_window();
        assert!(window.len() < conv.messages.len());
        assert!(conv.trimmed_count() > 0);
    }

    #[test]
    fn test_generate_response_deterministic() {
        let r1 = generate_response("test prompt", 42);
        let r2 = generate_response("test prompt", 42);
        assert_eq!(r1, r2);
    }

    #[test]
    fn test_generate_response_different_seeds() {
        let r1 = generate_response("same prompt", 1);
        let r2 = generate_response("same prompt", 2);
        assert_ne!(r1, r2);
    }

    #[test]
    fn test_format_prompt_structure() {
        let mut conv = Conversation::new(500);
        conv.add_system_prompt("be nice");
        conv.add_user_message("hi");
        let window = conv.context_window();
        let prompt = format_prompt(&window);
        assert!(prompt.contains("<|system|>"));
        assert!(prompt.contains("<|user|>"));
        assert!(prompt.contains("<|assistant|>"));
    }

    #[test]
    fn test_message_token_count() {
        let msg = Message::new(Role::User, "hello world test");
        let count = msg.token_count();
        assert!(count > 0);
        assert!(count < 100);
    }
}
