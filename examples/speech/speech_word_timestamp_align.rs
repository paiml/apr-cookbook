//! # Speech Word-Timestamp Alignment
//!
//! Whisper's segment-level timestamps cover whole utterances. To get
//! per-word timestamps, distribute the segment duration proportionally
//! to character count of each word. Optional: weight by phoneme count
//! for better accuracy on long syllables.
//!
//! Demonstrates the **SPEECH.5** recipe for PMAT-140 (speech round 2).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: WhisperX cross-attention timestamp alignment.
//!
//! Run with: cargo run --example speech_word_timestamp_align
//!
//! Added by PMAT-140 (expand-cookbooks followup).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, Clone, PartialEq)]
pub struct WordSpan {
    pub word: String,
    pub start_ms: u32,
    pub end_ms: u32,
}

#[derive(Debug, PartialEq)]
pub enum AlignVerdict {
    Ok(Vec<WordSpan>),
    EmptyWords,
    InvalidSegment,
}

pub fn align(words: &[&str], segment_start_ms: u32, segment_end_ms: u32) -> AlignVerdict {
    if words.is_empty() {
        return AlignVerdict::EmptyWords;
    }
    if segment_end_ms <= segment_start_ms {
        return AlignVerdict::InvalidSegment;
    }
    let total_chars: u32 = words.iter().map(|w| w.chars().count() as u32).sum();
    if total_chars == 0 {
        return AlignVerdict::InvalidSegment;
    }
    let total_dur = segment_end_ms - segment_start_ms;
    let mut spans = Vec::with_capacity(words.len());
    let mut cursor = segment_start_ms;
    for (i, w) in words.iter().enumerate() {
        let chars = w.chars().count() as u32;
        let dur = if i == words.len() - 1 {
            segment_end_ms - cursor
        } else {
            (u64::from(total_dur) * u64::from(chars) / u64::from(total_chars)) as u32
        };
        spans.push(WordSpan {
            word: (*w).to_string(),
            start_ms: cursor,
            end_ms: cursor + dur,
        });
        cursor += dur;
    }
    AlignVerdict::Ok(spans)
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("speech_word_timestamp_align")?;

    println!("typical: {:?}", align(&["hello", "world"], 0, 1000));
    println!("uneven length: {:?}", align(&["a", "bb", "ccc"], 0, 600));
    println!("empty words: {:?}", align(&[], 0, 1000));
    println!("invalid segment: {:?}", align(&["a"], 100, 50));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn align_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn equal_chars_equal_durations() {
        // "ab" "cd": each 2 chars; total 4. 1000ms → 500/500.
        let v = align(&["ab", "cd"], 0, 1000);
        if let AlignVerdict::Ok(spans) = v {
            assert_eq!(spans[0].end_ms - spans[0].start_ms, 500);
            assert_eq!(spans[1].end_ms - spans[1].start_ms, 500);
        }
    }

    #[test]
    fn proportional_durations() {
        // "a" "bb" "ccc" = 1+2+3 = 6 chars. 600ms → 100/200/300.
        let v = align(&["a", "bb", "ccc"], 0, 600);
        if let AlignVerdict::Ok(spans) = v {
            assert_eq!(spans[0].end_ms - spans[0].start_ms, 100);
            assert_eq!(spans[1].end_ms - spans[1].start_ms, 200);
            assert_eq!(spans[2].end_ms - spans[2].start_ms, 300);
        }
    }

    #[test]
    fn first_starts_at_segment_start() {
        let v = align(&["a", "b"], 100, 500);
        if let AlignVerdict::Ok(spans) = v {
            assert_eq!(spans[0].start_ms, 100);
        }
    }

    #[test]
    fn last_ends_at_segment_end() {
        let v = align(&["a", "bb", "ccc"], 0, 600);
        if let AlignVerdict::Ok(spans) = v {
            assert_eq!(spans[2].end_ms, 600);
        }
    }

    #[test]
    fn empty_words_rejected() {
        assert_eq!(align(&[], 0, 1000), AlignVerdict::EmptyWords);
    }

    #[test]
    fn inverted_segment_rejected() {
        assert_eq!(align(&["a"], 500, 100), AlignVerdict::InvalidSegment);
    }

    #[test]
    fn equal_segment_bounds_rejected() {
        assert_eq!(align(&["a"], 100, 100), AlignVerdict::InvalidSegment);
    }

    #[test]
    fn single_word_full_duration() {
        let v = align(&["only"], 0, 1000);
        if let AlignVerdict::Ok(spans) = v {
            assert_eq!(spans[0].start_ms, 0);
            assert_eq!(spans[0].end_ms, 1000);
        }
    }

    #[test]
    fn spans_contiguous_no_gaps() {
        let v = align(&["a", "bb", "ccc"], 0, 600);
        if let AlignVerdict::Ok(spans) = v {
            for w in spans.windows(2) {
                assert_eq!(w[0].end_ms, w[1].start_ms);
            }
        }
    }

    #[test]
    fn last_word_absorbs_rounding() {
        // 7 chars total in 100 ms: each char = ~14ms but rounding accumulates.
        // Last word should pick up the slack so spans cover [0, 100].
        let v = align(&["aaa", "bbb", "c"], 0, 100);
        if let AlignVerdict::Ok(spans) = v {
            assert_eq!(spans.last().unwrap().end_ms, 100);
        }
    }

    #[test]
    fn unicode_chars_counted_correctly() {
        // "héllo" has 5 chars (é is one char), "wörld" has 5.
        // Should split duration evenly.
        let v = align(&["héllo", "wörld"], 0, 1000);
        if let AlignVerdict::Ok(spans) = v {
            assert!((500..=510).contains(&(spans[0].end_ms - spans[0].start_ms)));
        }
    }
}
