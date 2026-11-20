# Agents and branches

| WS | Agent work | Where | State |
|---|---|---|---|
| A1 | Optuna loop owner: fixes, rounds, ingest | main ~/openjev | running (round 4) |
| A2 | option sub-sampling + packing | worktree-agent-a2d0d8e624957f092 | running |
| A3 | PLE coverage + hot/cold table | worktree-agent-a6144d4df19d02ef5 | paused mid-code |
| A4 | DDP 2×T4 | worktree-agent-a4bd9a6cb3cd6f0a5 | paused |
| A7.4 | NOTA/elimination/inconclusive twins | worktree-agent-a21a5c6ede756b5ea @3f061c3 | done; full pytest gate pending |
| B | XLA fork core + JAX proto | ~/openjev-xla branch xla | paused |

Worktrees: ~/openjev/.claude/worktrees/agent-*. Merge order: one at a time into main,
full pytest + tiny-Gemma preflight after each. A2/A3/A4 all touch train_prism.py, so rebase.
Resume a paused agent with SendMessage to its id (the ids are in the conversation; if lost,
start a new agent on the worktree branch and tell it to check git status/log first).
