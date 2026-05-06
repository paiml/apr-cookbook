//! # Chat Template Renderer
//!
//! Different model families expect different chat-completion templates:
//! ChatML (`<|im_start|>role\ncontent<|im_end|>`), Llama-2 (`[INST]
//! ... [/INST]`), Mistral (similar to Llama-2 but bos-anchored).
//! This recipe builds the renderer + dispatcher by family name.
//!
//! Demonstrates the **CHAT.6** recipe for PMAT-125 (chat coverage).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: HuggingFace chat templates spec.
//!
//! Run with: cargo run --example chat_template_renderer
//!
//! Added by PMAT-125 (expand-cookbooks followup).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TemplateFamily {
    ChatMl,
    Llama2,
    Mistral,
    Phi,
}

impl TemplateFamily {
    pub fn from_str_strict(s: &str) -> Option<Self> {
        match s.to_ascii_lowercase().as_str() {
            "chatml" => Some(TemplateFamily::ChatMl),
            "llama2" | "llama-2" => Some(TemplateFamily::Llama2),
            "mistral" => Some(TemplateFamily::Mistral),
            "phi" => Some(TemplateFamily::Phi),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Role {
    System,
    User,
    Assistant,
}

pub fn render(family: TemplateFamily, role: Role, content: &str) -> String {
    match family {
        TemplateFamily::ChatMl => {
            let role_str = match role {
                Role::System => "system",
                Role::User => "user",
                Role::Assistant => "assistant",
            };
            format!("<|im_start|>{role_str}\n{content}<|im_end|>")
        }
        TemplateFamily::Llama2 => match role {
            Role::System => format!("<<SYS>>\n{content}\n<</SYS>>"),
            Role::User => format!("[INST] {content} [/INST]"),
            Role::Assistant => content.to_string(),
        },
        TemplateFamily::Mistral => match role {
            Role::System => format!("<s>[INST] <<SYS>>\n{content}\n<</SYS>> [/INST]"),
            Role::User => format!("<s>[INST] {content} [/INST]"),
            Role::Assistant => format!("{content}</s>"),
        },
        TemplateFamily::Phi => match role {
            Role::System => format!("<|system|>\n{content}<|end|>"),
            Role::User => format!("<|user|>\n{content}<|end|>"),
            Role::Assistant => format!("<|assistant|>\n{content}<|end|>"),
        },
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("chat_template_renderer")?;

    for family in [
        TemplateFamily::ChatMl,
        TemplateFamily::Llama2,
        TemplateFamily::Mistral,
        TemplateFamily::Phi,
    ] {
        println!("=== {family:?} ===");
        for role in [Role::System, Role::User, Role::Assistant] {
            println!("{}", render(family, role, "Hello, world."));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn renderer_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn chatml_user_uses_im_start_token() {
        let r = render(TemplateFamily::ChatMl, Role::User, "Hi");
        assert!(r.starts_with("<|im_start|>user"));
        assert!(r.ends_with("<|im_end|>"));
    }

    #[test]
    fn llama2_user_uses_inst_brackets() {
        let r = render(TemplateFamily::Llama2, Role::User, "Hi");
        assert_eq!(r, "[INST] Hi [/INST]");
    }

    #[test]
    fn llama2_system_uses_sys_brackets() {
        let r = render(TemplateFamily::Llama2, Role::System, "be helpful");
        assert!(r.contains("<<SYS>>"));
        assert!(r.contains("<</SYS>>"));
    }

    #[test]
    fn mistral_anchors_with_bos() {
        let r = render(TemplateFamily::Mistral, Role::User, "Hi");
        assert!(r.starts_with("<s>"));
    }

    #[test]
    fn mistral_assistant_has_eos() {
        let r = render(TemplateFamily::Mistral, Role::Assistant, "Hello");
        assert!(r.ends_with("</s>"));
    }

    #[test]
    fn phi_uses_end_token() {
        let r = render(TemplateFamily::Phi, Role::User, "Hi");
        assert!(r.contains("<|user|>"));
        assert!(r.ends_with("<|end|>"));
    }

    #[test]
    fn family_names_round_trip() {
        for s in ["chatml", "llama2", "llama-2", "mistral", "phi"] {
            assert!(TemplateFamily::from_str_strict(s).is_some());
        }
    }

    #[test]
    fn case_insensitive_family_names() {
        assert_eq!(
            TemplateFamily::from_str_strict("ChatML"),
            Some(TemplateFamily::ChatMl)
        );
        assert_eq!(
            TemplateFamily::from_str_strict("MISTRAL"),
            Some(TemplateFamily::Mistral)
        );
    }

    #[test]
    fn unknown_family_returns_none() {
        assert!(TemplateFamily::from_str_strict("alpaca").is_none());
    }

    #[test]
    fn content_appears_verbatim() {
        let content = "test 123";
        for family in [
            TemplateFamily::ChatMl,
            TemplateFamily::Llama2,
            TemplateFamily::Mistral,
            TemplateFamily::Phi,
        ] {
            let r = render(family, Role::User, content);
            assert!(r.contains(content), "{family:?} did not include content");
        }
    }
}
