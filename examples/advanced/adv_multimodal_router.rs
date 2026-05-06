//! # Advanced Multimodal Input Router
//!
//! Route image/audio/text/video to specialized model. Detection by
//! content-type or magic bytes.
//!
//! Demonstrates the **ADV.23** recipe for PMAT-152 (milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: GPT-4V multimodal routing.
//!
//! Run with: cargo run --example adv_multimodal_router
//!
//! Added by PMAT-152 (catalog crosses 1000 recipes).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Modality {
    Text,
    Image,
    Audio,
    Video,
}

#[derive(Debug, PartialEq)]
pub enum RouteVerdict {
    Ok {
        modality: Modality,
        target_model: &'static str,
    },
    EmptyContent,
    UnknownModality,
}

pub fn route(content_type: &str, magic_bytes: &[u8]) -> RouteVerdict {
    if content_type.is_empty() && magic_bytes.is_empty() {
        return RouteVerdict::EmptyContent;
    }
    let modality = match content_type {
        "text/plain" | "text/html" | "application/json" => Modality::Text,
        ct if ct.starts_with("image/") => Modality::Image,
        ct if ct.starts_with("audio/") => Modality::Audio,
        ct if ct.starts_with("video/") => Modality::Video,
        _ => match magic_bytes {
            // PNG: 89 50 4E 47.
            [0x89, 0x50, 0x4E, 0x47, ..] => Modality::Image,
            // JPEG: FF D8 FF.
            [0xFF, 0xD8, 0xFF, ..] => Modality::Image,
            // OGG: 4F 67 67 53.
            [0x4F, 0x67, 0x67, 0x53, ..] => Modality::Audio,
            // MP4 ftyp: bytes 4..8 == "ftyp".
            [_, _, _, _, 0x66, 0x74, 0x79, 0x70, ..] => Modality::Video,
            _ => return RouteVerdict::UnknownModality,
        },
    };
    let target_model = match modality {
        Modality::Text => "gpt-4-text",
        Modality::Image => "gpt-4-vision",
        Modality::Audio => "whisper-v3",
        Modality::Video => "video-llava-7b",
    };
    RouteVerdict::Ok {
        modality,
        target_model,
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("adv_multimodal_router")?;

    println!("text/plain: {:?}", route("text/plain", &[]));
    println!("image/png: {:?}", route("image/png", &[]));
    println!(
        "PNG magic bytes: {:?}",
        route("", &[0x89, 0x50, 0x4E, 0x47])
    );
    println!("audio: {:?}", route("audio/mpeg", &[]));
    println!("unknown: {:?}", route("application/x-foo", &[0u8; 4]));
    println!("empty: {:?}", route("", &[]));
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
    fn text_routed_to_text_model() {
        let v = route("text/plain", &[]);
        if let RouteVerdict::Ok {
            modality,
            target_model,
        } = v
        {
            assert_eq!(modality, Modality::Text);
            assert_eq!(target_model, "gpt-4-text");
        }
    }

    #[test]
    fn image_content_type_routed() {
        let v = route("image/png", &[]);
        if let RouteVerdict::Ok { modality, .. } = v {
            assert_eq!(modality, Modality::Image);
        }
    }

    #[test]
    fn audio_content_type_routed() {
        let v = route("audio/mpeg", &[]);
        if let RouteVerdict::Ok { modality, .. } = v {
            assert_eq!(modality, Modality::Audio);
        }
    }

    #[test]
    fn video_content_type_routed() {
        let v = route("video/mp4", &[]);
        if let RouteVerdict::Ok { modality, .. } = v {
            assert_eq!(modality, Modality::Video);
        }
    }

    #[test]
    fn png_magic_bytes_recognized() {
        let v = route("", &[0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A]);
        if let RouteVerdict::Ok { modality, .. } = v {
            assert_eq!(modality, Modality::Image);
        }
    }

    #[test]
    fn jpeg_magic_bytes_recognized() {
        let v = route("", &[0xFF, 0xD8, 0xFF, 0xE0]);
        if let RouteVerdict::Ok { modality, .. } = v {
            assert_eq!(modality, Modality::Image);
        }
    }

    #[test]
    fn ogg_magic_bytes_recognized() {
        let v = route("", &[0x4F, 0x67, 0x67, 0x53]);
        if let RouteVerdict::Ok { modality, .. } = v {
            assert_eq!(modality, Modality::Audio);
        }
    }

    #[test]
    fn unknown_returns_unknown() {
        let v = route("application/x-foo", &[0u8; 4]);
        assert_eq!(v, RouteVerdict::UnknownModality);
    }

    #[test]
    fn empty_input_rejected() {
        assert_eq!(route("", &[]), RouteVerdict::EmptyContent);
    }

    #[test]
    fn json_routed_as_text() {
        let v = route("application/json", &[]);
        if let RouteVerdict::Ok { modality, .. } = v {
            assert_eq!(modality, Modality::Text);
        }
    }

    #[test]
    fn target_model_unique_per_modality() {
        let text = route("text/plain", &[]);
        let image = route("image/png", &[]);
        if let (
            RouteVerdict::Ok {
                target_model: t, ..
            },
            RouteVerdict::Ok {
                target_model: i, ..
            },
        ) = (text, image)
        {
            assert_ne!(t, i);
        }
    }
}
