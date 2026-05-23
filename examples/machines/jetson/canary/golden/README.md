# Golden Reference Outputs

Generated on first successful canary run, committed as regression baselines.

## Files

- `qwen-coder-1.5b-fibonacci.txt` — expected output from
  `apr run model.apr --prompt "Write a Python fibonacci function"`
  at temperature=0, seed=42, max-tokens=128
- `matmul-128x128-f32.bin` — reference f64 matmul output
  for kernel correctness (Tier 2)

## Regeneration

```bash
# On Jetson after successful canary:
ssh jetson 'apr run ~/data/models/canary/qwen-1.5b-q4k.apr \
    --prompt "Write a Python fibonacci function" \
    --max-tokens 128 --temperature 0' > golden/qwen-coder-1.5b-fibonacci.txt
```
