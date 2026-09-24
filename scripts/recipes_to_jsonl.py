#!/usr/bin/env python3
"""Derive evidence/recipes/recipes.jsonl from recipes/apr/**.yaml for the cookbook-recipe-v1 shape (#440).

pv's shapes gate validates RDF built by its generic `json` extractor, which turns each JSONL line
into one node of scalar properties. It cannot see list positions ("argv[0] is apr") or join two
fields ("every {model:slot} names a pinned model"). So this script computes those facts, and the
receipt coverage, as explicit fields. contracts/cookbook-recipe-v1.yaml then REQUIRES them
(`in: [true]`, patterns, counts). The JSONL is DERIVED and never hand-edited: `--check`
regenerates it and fails on any difference, so a recipe cannot drift from what the shape judged.

A row per recipe:
  id, schema, verb, recipe_sha256
  argv0_is_apr          argv[0] == "apr"
  placeholders_resolve  every {model:<slot>} in argv names a slot in models, and every slot is used
  model_sha256 / model_ref (repeated, one per model slot)
  expect_exit           the expected exit code
  has_output_check      expect also judges output (stdout_contains / stdout_json / stdout_json_is_object)
  min_apr, host (repeated), backend (repeated)
  receipt_current_pass  for EVERY declared host: receipts/<CURRENT>/<host>/<id>.json says PASS,
                        names the CURRENT binary, and carries this recipe's sha256 (#442's rule)

Zero recipes is REFUSED (rc 2): an empty corpus would let the shape gate decline quietly.
Usage: recipes_to_jsonl.py [--recipes DIR] [--receipts DIR] [--out FILE] [--check]
"""
import argparse
import hashlib
import json
import pathlib
import re
import sys

import yaml

SLOT = re.compile(r"^\{model:([A-Za-z0-9_-]+)\}$")


def receipt_current_pass(rid, hosts, recipe_sha, receipts_dir):
    try:
        cur = (pathlib.Path(receipts_dir) / "CURRENT").read_text().strip()
    except OSError:
        return False
    sha = cur.split("-", 1)[1] if "-" in cur else None
    if not sha or not hosts:
        return False
    for h in hosts:
        p = pathlib.Path(receipts_dir) / cur / h / ("%s.json" % rid)
        try:
            r = json.loads(p.read_text())
        except (OSError, ValueError):
            return False
        if r.get("verdict") != "PASS" or sha not in (r.get("apr_version") or "") or r.get("recipe_sha256") != recipe_sha:
            return False
    return True


def row(path, receipts_dir):
    raw = path.read_bytes()
    r = yaml.safe_load(raw)
    argv = r.get("argv") or []
    models = r.get("models") or {}
    used = {m.group(1) for a in argv if isinstance(a, str) for m in [SLOT.match(a)] if m}
    expect = r.get("expect") or {}
    rid = r.get("id")
    sha = hashlib.sha256(raw).hexdigest()
    out = {
        "id": rid,
        "schema": r.get("schema"),
        "verb": argv[1] if len(argv) > 1 else None,
        "recipe_sha256": sha,
        "argv0_is_apr": bool(argv) and argv[0] == "apr",
        "placeholders_resolve": used == set(models),
        "model_sha256": [(m or {}).get("sha256") for m in models.values()] or None,
        "model_ref": [(m or {}).get("ref") for m in models.values()] or None,
        "expect_exit": expect.get("exit"),
        "has_output_check": any(k in expect for k in ("stdout_contains", "stdout_json", "stdout_json_is_object")),
        "min_apr": r.get("min_apr"),
        "host": r.get("hosts") or None,
        "backend": r.get("backends") or None,
        "receipt_current_pass": receipt_current_pass(rid, r.get("hosts") or [], sha, receipts_dir),
    }
    # null means "absent" to the json extractor (it emits nothing), which minCount then catches.
    return {k: v for k, v in out.items() if v is not None}


def build(recipes_dir, receipts_dir):
    files = sorted(pathlib.Path(recipes_dir).rglob("*.yaml"))
    return [json.dumps(row(f, receipts_dir), sort_keys=True) for f in files]


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("--recipes", default="recipes/apr")
    ap.add_argument("--receipts", default="receipts")
    ap.add_argument("--out", default="evidence/recipes/recipes.jsonl")
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args(argv)
    lines = build(a.recipes, a.receipts)
    if not lines:
        print("REFUSED: no recipes under %s -- an empty corpus would let the shape gate decline quietly" % a.recipes)
        return 2
    text = "\n".join(lines) + "\n"
    out = pathlib.Path(a.out)
    if a.check:
        have = out.read_text() if out.exists() else ""
        if have != text:
            print("FAIL  %s is not what the recipes derive -- run scripts/recipes_to_jsonl.py (never hand-edit it)" % out)
            return 1
        print("PASS  %s matches the %d recipe(s)" % (out, len(lines)))
        return 0
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text)
    print("wrote %s: %d recipe row(s)" % (out, len(lines)))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
