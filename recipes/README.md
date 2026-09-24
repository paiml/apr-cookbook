# `apr` binary recipes (`cookbook-recipe/v1`)

Each YAML file here is one command you can run against a released `apr` binary: the argv,
the model it needs (pinned by sha256), what a correct run prints, and the minimum apr
version. The format is the one decided in
[aprender#3769](https://github.com/paiml/aprender/issues/3769).

**What is validated today, and what is not.**

| Check | Status |
|---|---|
| Every `--flag` in a recipe's argv is listed by `apr <verb> --help` on the release binary | **Checked** by `scripts/check_recipe_argv_surface.py`: 5/5 against the published `apr 0.69.1` release asset (x86_64-cuda, sha256 `c4bbd7ae…`, prints `v0.69.1+no-git`). A misspelt flag fails (exit 1), and that negative control was run |
| The `cookbook-recipe-v1` SHACL shape (argv against the CLI surface JSON, receipt present, a recipe for every model-taking verb) | **Not built yet**: aprender#3769 done_when 1 |
| A receipt from the release binary on lambda and gx10 showing `expect` passed | **Not in this change**: the recipe runner that writes receipts to `receipts/<CURRENT>/<host>/<id>.json` lands next (#439, PR #452), and the check that requires one per recipe after it (#442) |

So these recipes are a draft. They are not SHACL-validated or receipted yet, and the table
above says so instead of implying otherwise.

```bash
python3 scripts/check_recipe_argv_surface.py /path/to/apr
```
