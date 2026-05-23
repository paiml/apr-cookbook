# Code — `apr code` Agentic Surface

Recipes for the **`apr code`** Claude-Code-parity surface. Each demonstrates the file layout and discovery/parsing/validation pattern an `apr code` install uses for one of the 7 P0/P1 SHIPPED rows of [`apr-code-parity-v1.yaml`](https://github.com/paiml/aprender/blob/main/contracts/apr-code-parity-v1.yaml) v5.1.

The recipes are CLI-independent — they implement the same parsing/discovery logic in Rust without invoking the `apr code` binary, so they ship correct documentation now and continue working when `apr code` lands in the next aprender release.

## Recipes

| # | Recipe | Parity row |
|---|--------|------------|
| C.1 | [`code_mcp_client_config`](https://github.com/paiml/apr-cookbook/blob/main/examples/code/code_mcp_client_config.rs) | PMAT-CODE-MCP-CLIENT-001 |
| C.2 | [`code_slash_command_extension`](https://github.com/paiml/apr-cookbook/blob/main/examples/code/code_slash_command_extension.rs) | PMAT-CODE-SLASH-PARITY-001 |
| C.3 | [`code_hook_session_start`](https://github.com/paiml/apr-cookbook/blob/main/examples/code/code_hook_session_start.rs) | PMAT-CODE-HOOKS-001 |
| C.4 | [`code_subagent_spawn_payload`](https://github.com/paiml/apr-cookbook/blob/main/examples/code/code_subagent_spawn_payload.rs) | PMAT-CODE-SPAWN-PARITY-001 |
| C.5 | [`code_custom_agent_definition`](https://github.com/paiml/apr-cookbook/blob/main/examples/code/code_custom_agent_definition.rs) | PMAT-CODE-CUSTOM-AGENTS-001 |
| C.6 | [`code_skill_discovery`](https://github.com/paiml/apr-cookbook/blob/main/examples/code/code_skill_discovery.rs) | PMAT-CODE-SKILLS-001 |
| C.7 | [`code_worktree_isolation_permission_mode`](https://github.com/paiml/apr-cookbook/blob/main/examples/code/code_worktree_isolation_permission_mode.rs) | PMAT-CODE-WORKTREE-001 + PMAT-CODE-PERMISSIONS-001 (combined) |

## Conventions covered

- **`.apr/agents/<name>.md`** — `---`-fenced YAML frontmatter, hand-rolled parser
- **`.apr/skills/{<name>.md, <name>/SKILL.md}`** — flat + nested layouts; `.apr` precedence over `.claude` on collision
- **`.apr/hooks/<event>/<name>.sh`** — executable shell scripts at lifecycle events
- **`.apr/commands/<name>.md`** — project-local slash command extensions
- **`.mcp.json`** — MCP server registry (stdio / sse / http transports)
- **Task spawn payload** — JSON envelope with `subagent_type` + `prompt` + `description`
- **`<repo>/.apr/worktrees/<branch>/HEAD`** — worktree marker; permission-mode lattice (`deny < ask < allow < always_allow`)

## Provenance

Added during PMAT-074 (expand-cookbooks initiative, v6.1.0).
