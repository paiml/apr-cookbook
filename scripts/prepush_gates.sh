#!/usr/bin/env bash
# prepush_gates.sh -- run the pre-push quality gates BEFORE `git push` opens the connection (#437).
#
# The pre-push hook ran fmt, clippy and the full `cargo test --all-features` INSIDE `git push`,
# after git had opened the SSH connection to GitHub. The full test outlives GitHub's idle
# timeout ("Connection to github.com closed by remote host"), so a green branch could not be
# pushed without --no-verify, which skips every gate at once.
#
# Now the gates run here, with no connection open. On success they STAMP the exact commit
# (HEAD's sha) under the git common dir. The hook passes a pushed sha that carries a stamp,
# and for any other sha it still runs the gates itself (the old behaviour). So no gate is
# skipped, only moved in front of the connection. A branch DELETE pushes no code and runs no
# gates.
#
# A stamp describes a COMMIT, so the tree must equal HEAD: a tracked change, or ANY untracked
# non-ignored file, refuses to stamp (rc 2), because the gates would have measured inputs that
# are not the commit being pushed.
#
# SCOPE (#462): the full `cargo test` linked all ~1800 examples, and ONE target dir reached
# 667G (debug/examples alone was 636G). A second session starting a fresh CARGO_TARGET_DIR added
# another 639G and filled the build RAID to 99%. So the gates now build:
#   - fmt: the whole tree (it is cheap);
#   - clippy + test: the lib, bins and tests (plus doctests), plus ONLY the examples that the pushed
#     range (BASE..HEAD) touches. clippy also lints the benches (as --all-targets did); test never ran them.
# CI still builds every example; this gate is the fast local net, not the release gate.
# An example is "touched" when its own file changed, or, for a directory example (.../main.rs),
# any file under its directory changed. A change to src/ or Cargo.toml builds no example here,
# and CI covers that.
# One shared target dir (PREPUSH_TARGET_DIR; otherwise the build RAID when it exists, otherwise
# <git common dir>/prepush-target), so it deliberately IGNORES a per-session CARGO_TARGET_DIR.
# PREPUSH_JOBS (default 4) caps cargo -j. PREPUSH_MIN_FREE_GB (default 100) is the floor: below
# it the gates refuse to start (rc 2) instead of filling the disk.
# BASE is PREPUSH_BASE, else merge-base(HEAD, origin/main); the hook passes the remote sha.
#
# Usage:
#   scripts/prepush_gates.sh             run the gates on HEAD, stamp it on success
#   scripts/prepush_gates.sh --check SHA rc 0 iff SHA carries a stamp
#   scripts/prepush_gates.sh --self-test case table (stub cargo, throwaway repo)
# Then: git push (the hook sees the stamp and passes at once).
# Exit: 0 gates passed | 1 a gate failed | 2 refused (dirty tree, disk below the floor) / usage
set -euo pipefail

CARGO=${PREPUSH_CARGO:-cargo}

stamp_dir() {
    local common
    common=$(git rev-parse --git-common-dir)
    printf '%s/prepush-gates\n' "$common"
}

JOBS=${PREPUSH_JOBS:-4}
MIN_FREE_GB=${PREPUSH_MIN_FREE_GB:-100}

target_dir() {
    if [ -n "${PREPUSH_TARGET_DIR:-}" ]; then
        printf '%s\n' "$PREPUSH_TARGET_DIR"
    elif [ -d /mnt/nvme-raid0/targets ] && [ -w /mnt/nvme-raid0/targets ]; then
        printf '%s\n' /mnt/nvme-raid0/targets/apr-cookbook-prepush
    else
        printf '%s/prepush-target\n' "$(git rev-parse --path-format=absolute --git-common-dir)"
    fi
}

# Free GB on the filesystem that holds DIR (its nearest existing ancestor).
free_gb() {
    local d=$1
    while [ ! -d "$d" ]; do d=$(dirname "$d"); done
    df -Pk "$d" | awk 'NR == 2 { printf "%d\n", $4 / 1048576 }'
}

# The base the pushed range is measured from: an explicit arg (the hook's remote sha), else
# PREPUSH_BASE, else merge-base with origin/main. Empty when none resolves.
resolve_base() {
    local b=${1:-}
    if [ -n "$b" ] && [ "$b" != 0000000000000000000000000000000000000000 ] && git cat-file -e "$b^{commit}" 2> /dev/null; then
        printf '%s\n' "$b"; return
    fi
    git merge-base HEAD "${PREPUSH_BASE:-origin/main}" 2> /dev/null || true
}

# Example target names whose sources BASE..HEAD touches, one per line.
changed_examples() {
    local base=$1 files
    files=$(git diff --name-only "$base" HEAD -- examples/)
    [ -n "$files" ] || return 0
    "$CARGO" metadata --no-deps --format-version 1 2> /dev/null | CHANGED="$files" python3 -c '
import json, os, sys
root = os.path.realpath(".")
changed = set(os.environ["CHANGED"].split("\n"))
names = set()
for pkg in json.load(sys.stdin)["packages"]:
    for t in pkg["targets"]:
        if "example" not in t["kind"]:
            continue
        src = os.path.relpath(os.path.realpath(t["src_path"]), root)
        own_dir = os.path.dirname(src) + "/" if os.path.basename(src) == "main.rs" else None
        if src in changed or (own_dir and any(f.startswith(own_dir) for f in changed)):
            names.add(t["name"])
print("\n".join(sorted(names)))
'
}

# The gates. $1 = the base of the pushed range (optional; see resolve_base).
run_gates() {
    local base tdir free ex_args=() ex n=0
    tdir=$(target_dir)
    free=$(free_gb "$tdir")
    if [ "$free" -lt "$MIN_FREE_GB" ]; then
        echo "REFUSED: ${free}G free under $tdir, below the ${MIN_FREE_GB}G floor (PREPUSH_MIN_FREE_GB); free space first -- a full build of this repo's examples is ~650G (#462)" >&2
        return 2
    fi
    export CARGO_TARGET_DIR=$tdir
    base=$(resolve_base "${1:-}")
    if [ -n "$base" ]; then
        while IFS= read -r ex; do
            [ -n "$ex" ] || continue
            ex_args+=(--example "$ex"); n=$((n + 1))
        done < <(changed_examples "$base")
        echo "  scope: lib+bins+tests+doc and $n changed example(s) since ${base:0:9}; target $tdir (${free}G free), -j$JOBS"
    else
        echo "  scope: lib+bins+tests+doc only (no base resolved, so no examples); target $tdir (${free}G free), -j$JOBS"
    fi
    printf '  cargo fmt... '
    if "$CARGO" fmt --all -- --check > /dev/null 2>&1; then echo ok; else echo FAIL; echo "   Run: cargo fmt --all"; return 1; fi
    printf '  cargo clippy... '
    if "$CARGO" clippy -j "$JOBS" --lib --bins --tests --benches "${ex_args[@]}" --all-features -- -D warnings > /dev/null 2>&1; then echo ok; else echo FAIL; echo "   Run: cargo clippy --lib --bins --tests --benches ${ex_args[*]} --all-features -- -D warnings"; return 1; fi
    printf '  cargo test... '
    if "$CARGO" test --all-features -j "$JOBS" --lib --bins --tests "${ex_args[@]}" > /dev/null 2>&1; then echo ok; else echo FAIL; echo "   Run: cargo test --all-features --lib --bins --tests ${ex_args[*]}"; return 1; fi
    printf '  cargo test --doc... '
    if "$CARGO" test --all-features -j "$JOBS" --doc > /dev/null 2>&1; then echo ok; else echo FAIL; echo "   Run: cargo test --all-features --doc"; return 1; fi
}

# The tree must equal HEAD: no tracked change AND no untracked, non-ignored file of ANY kind.
# Builds and tests read more than .rs (include_str!/include_bytes! of .json/.yaml, fixtures read
# at run time), so an untracked data file would let the gates measure inputs the stamped commit
# does not contain (quorum finding, #453). Ignored files (target/) are fine: git status omits them.
tree_is_head() {
    [ -z "$(git status --porcelain --untracked-files=normal)" ]
}

stamp_head() {
    local sha dir
    sha=$(git rev-parse HEAD)
    if ! tree_is_head; then
        echo "REFUSED: the working tree differs from HEAD ${sha:0:9}; commit or clean it first -- a stamp names a commit, and the gates would measure other code" >&2
        return 2
    fi
    echo "Pre-push gates on ${sha:0:9} (no connection open)"
    local rc=0
    run_gates || rc=$?
    [ "$rc" = 0 ] || return "$rc"
    dir=$(stamp_dir)
    mkdir -p "$dir"
    printf '%s\nfmt=ok clippy=ok test=ok\n%s\n' "$sha" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$dir/$sha"
    echo "stamped ${sha:0:9}: git push will pass the hook without re-running the gates"
}

check_stamp() {
    local sha=$1 f
    f="$(stamp_dir)/$sha"
    [ -f "$f" ] && [ "$(head -n 1 "$f")" = "$sha" ]
}

# Deletes only a mktemp directory: non-empty, not /, under /tmp, and a directory.
cleanup_tmp() {
    local d=$1
    if [ -n "$d" ] && [ "$d" != "/" ] && [ -d "$d" ]; then
        case "$d" in /tmp/tmp.*) rm -rf -- "$d" ;; *) ;; esac
    fi
}

self_test() {
    local t bad=0 rc
    t=$(mktemp -d)
    case "$t" in /tmp/tmp.*) ;; *) echo "REFUSED: mktemp gave $t" >&2; return 2 ;; esac
    trap 'cleanup_tmp "$t"' RETURN
    cat > "$t/cargo" << 'STUB'
#!/usr/bin/env bash
[ "${STUB_FAIL:-}" = "$1" ] && exit 1
exit 0
STUB
    chmod +x "$t/cargo"
    # The disk floor is exercised by its own rows; every other row must not depend on this host's
    # free space, and nothing may build outside the throwaway dir.
    MIN_FREE_GB=0
    export PREPUSH_MIN_FREE_GB=0 PREPUSH_TARGET_DIR="$t/target"
    git init -q "$t/repo"
    git -C "$t/repo" -c core.hooksPath=/dev/null -c user.name=t -c user.email=t@t commit -q --allow-empty -m one
    row() { if [ "$2" = "$3" ]; then echo "ok    $1"; else echo "FAIL  $1 (rc $3, wanted $2)"; bad=$((bad + 1)); fi; }
    has() { case "$1" in *"$2"*) echo 1 ;; *) echo 0 ;; esac; }
    # Every case runs in the throwaway repo with the stub cargo and THIS script's functions.
    rc=0; ( cd "$t/repo" && PREPUSH_CARGO="$t/cargo" CARGO="$t/cargo" stamp_head > /dev/null 2>&1 ) || rc=$?
    row "clean tree, gates pass -> stamped (rc 0)" 0 "$rc"
    rc=0; ( cd "$t/repo" && check_stamp "$(git rev-parse HEAD)" ) || rc=$?
    row "  ...and --check finds the stamp for HEAD" 0 "$rc"
    rc=0; ( cd "$t/repo" && check_stamp 0000000000000000000000000000000000000001 ) || rc=$?
    row "a sha with no stamp -> --check rc 1 (the hook then runs the gates)" 1 "$rc"
    git -C "$t/repo" -c core.hooksPath=/dev/null -c user.name=t -c user.email=t@t commit -q --allow-empty -m two
    rc=0; ( cd "$t/repo" && STUB_FAIL=test CARGO="$t/cargo" stamp_head > /dev/null 2>&1 ) || rc=$?
    row "MUST-RED: a failing cargo test -> rc 1, no stamp" 1 "$rc"
    rc=0; ( cd "$t/repo" && check_stamp "$(git rev-parse HEAD)" ) || rc=$?
    row "  ...and that commit is NOT stamped" 1 "$rc"
    printf 'fn main() {}\n' > "$t/repo/new.rs"
    rc=0; ( cd "$t/repo" && CARGO="$t/cargo" stamp_head > /dev/null 2>&1 ) || rc=$?
    row "MUST-RED: an untracked .rs file -> refused (rc 2): the gates would measure other code" 2 "$rc"
    rm -f "$t/repo/new.rs"
    printf '{"k": 1}\n' > "$t/repo/data.json"
    rc=0; ( cd "$t/repo" && CARGO="$t/cargo" stamp_head > /dev/null 2>&1 ) || rc=$?
    row "MUST-RED: an untracked NON-.rs file (data.json, readable by include_str!/tests) -> refused (rc 2)" 2 "$rc"
    rm -f "$t/repo/data.json"
    printf 'target/\n' > "$t/repo/.gitignore"; git -C "$t/repo" add .gitignore
    git -C "$t/repo" -c core.hooksPath=/dev/null -c user.name=t -c user.email=t@t commit -q -m ignore
    mkdir -p "$t/repo/target"; printf 'x' > "$t/repo/target/build.o"
    rc=0; ( cd "$t/repo" && CARGO="$t/cargo" stamp_head > /dev/null 2>&1 ) || rc=$?
    row "an IGNORED build artifact (target/) does not block a stamp" 0 "$rc"
    rm -f "$t/repo/target/build.o"; rmdir "$t/repo/target"
    printf 'x\n' > "$t/repo/f"; git -C "$t/repo" add f
    rc=0; ( cd "$t/repo" && CARGO="$t/cargo" stamp_head > /dev/null 2>&1 ) || rc=$?
    row "MUST-RED: a staged change -> refused (rc 2)" 2 "$rc"
    # SCOPE (#462): a stub cargo that LOGS each call (its CARGO_TARGET_DIR and argv) and answers
    # `cargo metadata` with three examples: a file example foo, a file example bar, and a
    # directory example dirx (examples/dirx/main.rs).
    cat > "$t/slog" << 'STUB'
#!/usr/bin/env bash
printf '%s|%s\n' "${CARGO_TARGET_DIR:-}" "$*" >> "$SCOPE_LOG"
if [ "$1" = metadata ]; then
    printf '{"packages":[{"targets":[{"kind":["lib"],"name":"l","src_path":"%s/src/lib.rs"},{"kind":["example"],"name":"foo","src_path":"%s/examples/foo.rs"},{"kind":["example"],"name":"bar","src_path":"%s/examples/bar.rs"},{"kind":["example"],"name":"dirx","src_path":"%s/examples/dirx/main.rs"}]}]}\n' "$PWD" "$PWD" "$PWD" "$PWD"
fi
exit 0
STUB
    chmod +x "$t/slog"
    local s="$t/scope" b0 log="$t/scope.log" tl
    git init -q "$s"; mkdir -p "$s/src" "$s/examples/dirx"
    for f in Cargo.toml Cargo.lock src/lib.rs examples/foo.rs examples/bar.rs examples/dirx/main.rs examples/dirx/util.rs; do printf '// %s\n' "$f" > "$s/$f"; done
    git -C "$s" add -A; git -C "$s" -c core.hooksPath=/dev/null -c user.name=t -c user.email=t@t commit -q -m base
    b0=$(git -C "$s" rev-parse HEAD)
    printf '// edit\n' >> "$s/examples/foo.rs"; printf '// edit\n' >> "$s/examples/dirx/util.rs"
    git -C "$s" -c core.hooksPath=/dev/null -c user.name=t -c user.email=t@t commit -q -am examples
    : > "$log"; rc=0; ( cd "$s" && SCOPE_LOG="$log" CARGO="$t/slog" CARGO_TARGET_DIR=/somewhere/else run_gates "$b0" > /dev/null 2>&1 ) || rc=$?
    tl=$(grep -- '|test .*--lib' "$log" || true)
    row "scope: gates pass on an examples change (rc 0)" 0 "$rc"
    row "scope: the changed FILE example is built (--example foo)" 1 "$(has "$tl" "--example foo")"
    row "scope: a change INSIDE a directory example builds it (--example dirx)" 1 "$(has "$tl" "--example dirx")"
    row "MUST-RED: an UNCHANGED example is not built (no --example bar)" 0 "$(has "$tl" "--example bar")"
    row "MUST-RED: a per-session CARGO_TARGET_DIR is ignored; every call uses the shared target" 0 "$(grep -vc "^$t/target|" "$log" || true)"
    row "scope: cargo is capped at -j$JOBS" 1 "$(has "$tl" "-j $JOBS ")"
    row "scope: clippy still lints the benches (--all-targets covered them)" 1 "$(has "$(grep -- '|clippy ' "$log" || true)" "--benches")"
    b0=$(git -C "$s" rev-parse HEAD)
    printf '// edit\n' >> "$s/src/lib.rs"
    git -C "$s" -c core.hooksPath=/dev/null -c user.name=t -c user.email=t@t commit -q -am lib
    : > "$log"; rc=0; ( cd "$s" && SCOPE_LOG="$log" CARGO="$t/slog" run_gates "$b0" > /dev/null 2>&1 ) || rc=$?
    row "scope: a src/-only change builds NO example (rc $rc)" 0 "$(grep -c -- '--example' "$log" || true)"
    row "  ...and still runs lib/bins/tests and the doctests" 2 "$(grep -cE -- '\|test .*(--lib|--doc)' "$log" || true)"
    b0=$(git -C "$s" rev-parse HEAD)
    printf '# bump\n' >> "$s/Cargo.toml"; printf '# bump\n' >> "$s/Cargo.lock"
    git -C "$s" -c core.hooksPath=/dev/null -c user.name=t -c user.email=t@t commit -q -am deps
    : > "$log"; rc=0; ( cd "$s" && SCOPE_LOG="$log" CARGO="$t/slog" run_gates "$b0" > /dev/null 2>&1 ) || rc=$?
    row "scope: a deps bump (Cargo.toml + Cargo.lock) builds lib/tests and NO example (rc $rc)" "0/0/2" "$rc/$(grep -c -- '--example' "$log" || true)/$(grep -cE -- '\|test .*(--lib|--doc)' "$log" || true)"
    : > "$log"; rc=0; ( cd "$s" && MIN_FREE_GB=999999999 SCOPE_LOG="$log" CARGO="$t/slog" run_gates "$b0" > /dev/null 2>&1 ) || rc=$?
    row "MUST-RED: free space below the floor -> refused (rc 2) before any cargo call (ran $(grep -c . "$log" || true))" "2/0" "$rc/$(grep -c . "$log" || true)"
    # END TO END through the REAL hook: a throwaway clone of HEAD's hook + this script, a bare
    # remote, and a stub cargo that COUNTS its calls, so "the gates did not run" is measured.
    local src e calls
    src=$(git rev-parse --show-toplevel)
    e="$t/e2e"; git init -q --bare "$t/remote.git"; git init -q "$e"
    mkdir -p "$e/.githooks" "$e/scripts"
    cp "$src/.githooks/pre-push" "$e/.githooks/pre-push"; cp "$src/scripts/prepush_gates.sh" "$e/scripts/prepush_gates.sh"
    chmod +x "$e/.githooks/pre-push" "$e/scripts/prepush_gates.sh"
    printf '#!/usr/bin/env bash\necho x >> "%s/calls"\n[ "${STUB_FAIL:-}" = "$1" ] && exit 1\nexit 0\n' "$t" > "$t/ccount"; chmod +x "$t/ccount"
    git -C "$e" add -A; git -C "$e" -c core.hooksPath=/dev/null -c user.name=t -c user.email=t@t commit -q -m e2e
    git -C "$e" config core.hooksPath .githooks; git -C "$e" remote add origin "$t/remote.git"
    : > "$t/calls"; rc=0; ( cd "$e" && PREPUSH_CARGO="$t/ccount" bash scripts/prepush_gates.sh > /dev/null 2>&1 ) || rc=$?
    row "e2e: prepush_gates.sh on a clean clone -> stamped" 0 "$rc"
    : > "$t/calls"; rc=0; ( cd "$e" && PREPUSH_CARGO="$t/ccount" git push -q origin HEAD:refs/heads/a > /dev/null 2>&1 ) || rc=$?
    calls=$(grep -c x "$t/calls" || true)
    row "e2e: a STAMPED push succeeds and runs 0 gate commands (ran $calls)" "0/0" "$rc/$calls"
    git -C "$e" -c core.hooksPath=/dev/null -c user.name=t -c user.email=t@t commit -q --allow-empty -m unstamped
    : > "$t/calls"; rc=0; ( cd "$e" && PREPUSH_CARGO="$t/ccount" git push -q origin HEAD:refs/heads/b > /dev/null 2>&1 ) || rc=$?
    calls=$(grep -c x "$t/calls" || true)
    row "e2e: an UNSTAMPED push runs all 4 gates in the hook (ran $calls)" "0/4" "$rc/$calls"
    rc=0; ( cd "$e" && STUB_FAIL=test PREPUSH_CARGO="$t/ccount" git push -q origin HEAD:refs/heads/c > /dev/null 2>&1 ) || rc=$?
    row "e2e MUST-RED: an unstamped push whose test FAILS is refused" 1 "$([ "$rc" != 0 ] && echo 1 || echo 0)"
    : > "$t/calls"; rc=0; ( cd "$e" && STUB_FAIL=test PREPUSH_CARGO="$t/ccount" git push -q origin --delete a > /dev/null 2>&1 ) || rc=$?
    calls=$(grep -c x "$t/calls" || true)
    row "e2e: a branch DELETE runs 0 gates, even with a failing test (ran $calls)" "0/0" "$rc/$calls"
    echo "prepush_gates self-test: $([ "$bad" = 0 ] && echo PASS || echo FAIL) ($bad failed)"
    [ "$bad" = 0 ]
}

case "${1:-}" in
    "") stamp_head ;;
    --check) [ -n "${2:-}" ] || { echo "usage: $0 --check SHA" >&2; exit 2; }; check_stamp "$2" ;;
    --self-test) self_test ;;
    --run-gates) run_gates "${2:-}" ;;
    *) echo "usage: $0 [--check SHA | --self-test]" >&2; exit 2 ;;
esac
