# How the user wants work done

- Keep the main thread clear: delegate all work to background agents and answer questions in a few lines.
- Follow ~/.claude/CLAUDE.md: haiku for checks and logs, sonnet for routine work, opus for core code;
  effort low/med/high only (xhigh/max need approval); no gh without approval; no raw DB queries.
- Standing approval for Kaggle pushes on the OpenJev kernels. The final long run needs the user's OK.
- Small Optuna rounds (4 trials), fixed micro-batch 8, fair experiments, log every decision with evidence.
- Simplified Technical English output style.
