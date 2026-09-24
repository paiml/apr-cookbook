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
# Usage:
#   scripts/prepush_gates.sh             run the gates on HEAD, stamp it on success
#   scripts/prepush_gates.sh --check SHA rc 0 iff SHA carries a stamp
#   scripts/prepush_gates.sh --self-test case table (stub cargo, throwaway repo)
# Then: git push (the hook sees the stamp and passes at once).
# Exit: 0 gates passed | 1 a gate failed | 2 refused (dirty tree) / usage
set -euo pipefail

CARGO=${PREPUSH_CARGO:-cargo}

stamp_dir() {
    local common
    common=$(git rev-parse --git-common-dir)
    printf '%s/prepush-gates\n' "$common"
}

# The gates, byte-for-byte the hook's commands.
run_gates() {
    printf '  cargo fmt... '
    if "$CARGO" fmt --all -- --check > /dev/null 2>&1; then echo ok; else echo FAIL; echo "   Run: cargo fmt --all"; return 1; fi
    printf '  cargo clippy... '
    if "$CARGO" clippy --all-targets --all-features -- -D warnings > /dev/null 2>&1; then echo ok; else echo FAIL; echo "   Run: cargo clippy --all-targets --all-features -- -D warnings"; return 1; fi
    printf '  cargo test... '
    if "$CARGO" test --all-features > /dev/null 2>&1; then echo ok; else echo FAIL; echo "   Run: cargo test --all-features"; return 1; fi
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
    run_gates || return 1
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
    git init -q "$t/repo"
    git -C "$t/repo" -c core.hooksPath=/dev/null -c user.name=t -c user.email=t@t commit -q --allow-empty -m one
    row() { if [ "$2" = "$3" ]; then echo "ok    $1"; else echo "FAIL  $1 (rc $3, wanted $2)"; bad=$((bad + 1)); fi; }
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
    row "e2e: an UNSTAMPED push runs all 3 gates in the hook (ran $calls)" "0/3" "$rc/$calls"
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
    --run-gates) run_gates ;;
    *) echo "usage: $0 [--check SHA | --self-test]" >&2; exit 2 ;;
esac
