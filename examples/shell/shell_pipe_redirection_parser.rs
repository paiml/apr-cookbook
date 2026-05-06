//! # Shell Pipe & Redirection Parser
//!
//! Parses POSIX shell pipelines: `cmd1 | cmd2 > out.txt 2> err.txt
//! < in.txt`. Returns the command sequence + per-stage redirections.
//! Constraints: balanced redirection operators (`>`, `<`, `>>`, `2>`,
//! `2>&1`), at least one command, no trailing pipe.
//!
//! Demonstrates the **SHELL.4** recipe for PMAT-126 (shell coverage).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: POSIX 1003.1-2017 Shell & Utilities §2.7.
//!
//! Run with: cargo run --example shell_pipe_redirection_parser
//!
//! Added by PMAT-126 (expand-cookbooks followup).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq, Eq, Clone)]
pub struct Redirection {
    pub op: String,
    pub target: String,
}

#[derive(Debug, PartialEq, Eq, Clone)]
pub struct Stage {
    pub argv: Vec<String>,
    pub redirections: Vec<Redirection>,
}

#[derive(Debug, PartialEq)]
pub enum ParseError {
    EmptyPipeline,
    TrailingPipe,
    DanglingRedirection,
    EmptyStage,
}

pub fn parse(input: &str) -> std::result::Result<Vec<Stage>, ParseError> {
    let trimmed = input.trim();
    if trimmed.is_empty() {
        return Err(ParseError::EmptyPipeline);
    }
    if trimmed.ends_with('|') {
        return Err(ParseError::TrailingPipe);
    }
    let mut stages = Vec::new();
    for stage_str in trimmed.split('|') {
        let stage = parse_stage(stage_str.trim())?;
        stages.push(stage);
    }
    Ok(stages)
}

fn parse_stage(stage_str: &str) -> std::result::Result<Stage, ParseError> {
    if stage_str.is_empty() {
        return Err(ParseError::EmptyStage);
    }
    let tokens: Vec<&str> = stage_str.split_whitespace().collect();
    let mut argv = Vec::new();
    let mut redirections = Vec::new();
    let mut i = 0;
    while i < tokens.len() {
        let tok = tokens[i];
        if matches!(tok, ">" | "<" | ">>" | "2>" | "2>>") {
            if i + 1 >= tokens.len() {
                return Err(ParseError::DanglingRedirection);
            }
            redirections.push(Redirection {
                op: tok.to_string(),
                target: tokens[i + 1].to_string(),
            });
            i += 2;
            continue;
        }
        if tok == "2>&1" {
            redirections.push(Redirection {
                op: "2>&1".into(),
                target: "stderr-to-stdout".into(),
            });
            i += 1;
            continue;
        }
        argv.push(tok.to_string());
        i += 1;
    }
    if argv.is_empty() {
        return Err(ParseError::EmptyStage);
    }
    Ok(Stage { argv, redirections })
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("shell_pipe_redirection_parser")?;

    for input in [
        "ls -la | grep .rs > out.txt",
        "cat foo.txt | wc -l",
        "cmd > out.txt 2>&1",
        "ls |",
        "",
        "ls > ",
    ] {
        println!("{input:<35}  →  {:?}", parse(input));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parser_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn typical_pipeline_parses() {
        let stages = parse("ls -la | grep .rs").unwrap();
        assert_eq!(stages.len(), 2);
        assert_eq!(stages[0].argv, vec!["ls", "-la"]);
        assert_eq!(stages[1].argv, vec!["grep", ".rs"]);
    }

    #[test]
    fn redirection_attached_to_stage() {
        let stages = parse("cat > out.txt").unwrap();
        assert_eq!(stages.len(), 1);
        assert_eq!(stages[0].argv, vec!["cat"]);
        assert_eq!(stages[0].redirections.len(), 1);
        assert_eq!(stages[0].redirections[0].op, ">");
        assert_eq!(stages[0].redirections[0].target, "out.txt");
    }

    #[test]
    fn append_redirect_recognised() {
        let stages = parse("cat >> out.txt").unwrap();
        assert_eq!(stages[0].redirections[0].op, ">>");
    }

    #[test]
    fn stderr_redirect_recognised() {
        let stages = parse("cat 2> err.txt").unwrap();
        assert_eq!(stages[0].redirections[0].op, "2>");
    }

    #[test]
    fn stderr_to_stdout_recognised() {
        let stages = parse("cmd 2>&1").unwrap();
        assert_eq!(stages[0].redirections[0].op, "2>&1");
    }

    #[test]
    fn empty_pipeline_rejected() {
        assert_eq!(parse(""), Err(ParseError::EmptyPipeline));
        assert_eq!(parse("   "), Err(ParseError::EmptyPipeline));
    }

    #[test]
    fn trailing_pipe_rejected() {
        assert_eq!(parse("ls |"), Err(ParseError::TrailingPipe));
    }

    #[test]
    fn dangling_redirect_rejected() {
        let v = parse("ls >");
        assert_eq!(v, Err(ParseError::DanglingRedirection));
    }

    #[test]
    fn empty_stage_in_pipeline_rejected() {
        // `ls |  | grep` has an empty middle stage.
        let v = parse("ls |  | grep");
        assert_eq!(v, Err(ParseError::EmptyStage));
    }

    #[test]
    fn three_stage_pipeline() {
        let stages = parse("cat foo | sort | uniq").unwrap();
        assert_eq!(stages.len(), 3);
    }
}
