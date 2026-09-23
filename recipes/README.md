# `apr` binary recipes (`cookbook-recipe/v1`)

Each YAML file here is one command you can run against a released `apr` binary: the argv,
the model it needs (pinned by sha256), what a correct run prints, and the minimum apr
version. The format is the one decided in
[aprender#3769](https://github.com/paiml/aprender/issues/3769).

**What is validated today, and what is not.**

| Check | Status |
|---|---|
| Every `--flag` in a recipe's argv is listed by `apr <verb> --help` on the release binary | **Checked** by `scripts/check_recipe_argv_surface.py`: 5/5 against `apr 0.69.1 (d8a6df53a)`. A misspelt flag fails, and that negative control was run |
| The `cookbook-recipe-v1` SHACL shape (argv against the CLI surface JSON, receipt present, a recipe for every model-taking verb) | **Not built yet**: aprender#3769 done_when 1 |
| A receipt from the release binary on lambda and gx10 showing `expect` passed | **Done for 0.69.1**: all 5 recipes PASS on the published binaries on both hosts, found by convention at `receipts/<CURRENT>/<host>/<id>.json` (#439, #452) |

So these recipes are a draft. They are not SHACL-validated recipes yet, and the table
above says so instead of implying otherwise.

```bash
python3 scripts/check_recipe_argv_surface.py /path/to/apr
```
