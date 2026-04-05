---
name: qa
description: QA the installed apr CLI binary — pull a model, test every subcommand, hunt for bugs, file GitHub issues. Use when asked to QA apr, test apr, or check apr-cli.
allowed-tools: Bash Glob Grep Read Agent TaskCreate TaskUpdate TaskList
---

# APR CLI QA Skill

Perform exhaustive QA of the **installed** `apr` binary (not source builds). Pull a real model, exercise every subcommand, and file GitHub issues for bugs found.

## Arguments

$ARGUMENTS

If arguments include a model path, use that model. Otherwise pull the default:
```
apr pull hf://Qwen/Qwen2.5-Coder-1.5B-Instruct-GGUF/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf
```

## Setup

1. Verify `apr` is installed: `which apr && apr --version`
2. Pull/locate the test model
3. Set `M=/path/to/model.gguf` for all subsequent commands

## Test Plan

Work through every category below. For each command, capture stdout+stderr and the exit code. Track bugs as you go.

### 1. Inspection commands
Test each with the model file. Also test `--json` and `--quiet` variants where supported.

```
apr inspect $M
apr inspect --json $M
apr inspect --vocab $M
apr tensors $M
apr tensors --json $M
apr tree $M
apr flow $M
apr debug $M
apr debug --drama $M
apr hex $M
apr explain $M
apr explain E001
apr explain --tensor q_proj
apr explain --kernel qwen2
apr oracle $M
apr oracle --json $M
```

### 2. Validation commands
```
apr validate $M
apr check $M
apr lint $M
apr qa --skip-throughput --skip-ollama --skip-golden --skip-gpu-speedup --skip-gpu-state --skip-ptx-parity --skip-format-parity --skip-capability $M
apr qualify $M                    # timeout 90s
```

### 3. Inference commands
Use `timeout 60` for all inference commands.

```
apr run $M "What is 2+2?" --max-tokens 32
apr bench $M --iterations 3 --warmup 1 --max-tokens 16
apr eval $M --max-tokens 16
apr serve plan $M
```

### 4. Transform commands
Use `timeout 30` — these may hang (known issue pattern).

```
apr convert $M --output /tmp/apr-qa-convert.apr
apr import $M
apr export $M -o /tmp/apr-qa-export.safetensors
apr quantize $M --output /tmp/apr-qa-quant.gguf
apr merge $M $M --output /tmp/apr-qa-merged.gguf
apr prune $M --output /tmp/apr-qa-pruned.gguf
apr compile $M
echo "test123" | apr encrypt $M --output /tmp/apr-qa-enc.enc
echo "test123" | apr decrypt /tmp/apr-qa-enc.enc --output /tmp/apr-qa-dec.gguf
```

### 5. Training/data commands
```
# Create test data
echo '{"input": "hello world", "label": 0}' > /tmp/apr-qa-data.jsonl
echo '{"input": "foo bar", "label": 1}' >> /tmp/apr-qa-data.jsonl

apr finetune $M
apr distill $M
apr train plan
apr data audit /tmp/apr-qa-data.jsonl
apr tokenize plan --data /tmp/apr-qa-data.jsonl
```

### 6. Operational commands
```
apr list
apr list --json
apr gpu
apr gpu --json
apr diff $M $M
apr trace $M
apr profile $M                    # timeout 60
apr parity $M                     # timeout 30
apr ptx-map $M
apr rosetta inspect $M
apr rosetta fingerprint $M
apr compare-hf $M --hf <appropriate-repo>   # timeout 30
apr runs ls
apr diagnose --help
apr probar --help
apr publish --help
```

### 7. Error handling & edge cases
```
apr inspect /nonexistent/file.gguf       # should exit non-zero
echo "bad" > /tmp/apr-qa-bad.gguf
apr inspect /tmp/apr-qa-bad.gguf         # should exit non-zero
apr foobar                               # unknown subcommand
apr list --json --json                   # duplicate flag
```

## Bug Detection Checklist

Flag anything matching these patterns:

- **Panics/crashes** — any `thread 'main' panicked` output
- **Exit code lies** — output says error/fail but exit code is 0
- **Hangs** — command doesn't complete within timeout
- **Wrong data** — quantization showing "0", dtypes as numbers, etc.
- **No-op flags** — flags that are accepted but have no effect (compare output with/without)
- **Missing fallbacks** — GPU failures with no CPU fallback
- **Cache inconsistencies** — pull/list/rm disagree about what's cached
- **Misleading messages** — labels like "Garbage" for valid models

## Filing Issues

For each bug found, file a GitHub issue using `gh`:

```bash
gh issue create --repo paiml/aprender --title "<concise title>" --body "$(cat <<'EOF'
## Description
<what's wrong>

## Reproduction
<exact commands>

## Expected
<what should happen>

## Version
`apr <version>`
EOF
)"
```

Group related exit-code bugs into a single issue. Use severity labels:
- **P0**: Panics, crashes, data corruption
- **P1**: Core workflow broken (run/pull/list)
- **P2**: Wrong output, misleading messages
- **P3**: Cosmetic, minor UX

## Cleanup

Remove all temp files when done:
```bash
rm -f /tmp/apr-qa-*.{gguf,apr,enc,jsonl,safetensors}
```

## Report

End with a summary table of all bugs found and issues filed, organized by severity.
