---
name: kaiwu-diy-upload
description: Prepare and upload code for this Tencent Kaiwu intelligent-traffic repo when the work primarily touches `agent_diy/`, `train_test.py`, or related `conf/**` files. Use when Codex needs repo-specific guardrails for remote upload limits, generate a timestamped upload batch, or sync changes through `chrome-devtools-mcp` instead of Playwright.
---

# Kaiwu DIY Upload

Use this skill for repo-local Tencent Kaiwu sync work.

## Core Rules

- Prefer editing `agent_diy/**`.
- Only touch `train_test.py` or `conf/**` when the change really requires it.
- Do not upload or sync files outside the remote allowlist:
  - `agent_diy/**`
  - `agent_ppo/**`
  - `conf/**`
  - `log/**`
  - `train_test.py`
- Never sync `PI-eLight-main/`, `upload_diy_minimal/`, docs, archives, or automation outputs.
- Use `chrome-devtools-mcp` for the live Tencent IDE. Do not reintroduce Playwright flows.

## Default Workflow

1. Make the code change, usually under `agent_diy/`.
2. Build a DIY-focused upload batch:

```bash
python automation/build_diy_sync_batch.py
```

3. If config files changed too, include `conf/**`:

```bash
python automation/build_diy_sync_batch.py --with-conf
```

4. If the change spans every allowed remote path, use the full batch:

```bash
python automation/build_diy_sync_batch.py --all-allowed
```

5. Use `chrome-devtools-mcp` to open the current Tencent Kaiwu IDE page and write files from the generated batch directory.

## Upload Batch Rules

- Generated files live under `automation/data/sync_batch_diy/` by default.
- `.py` files in the batch receive a fresh `# Uploaded at: ...` header.
- Local source files must remain untouched.
- The default DIY batch only includes changed files from:
  - `agent_diy/**`
  - `train_test.py`
- `--with-conf` extends that to changed `conf/**`.

## Verification

- Confirm the batch manifest exists:
  - `automation/data/sync_batch_diy_manifest.json`
- In the remote IDE, reopen the changed file after save.
- For `.py` files, verify the remote file contains the new `# Uploaded at: ...` header.

## References

- Read `references/remote-rules.md` before broad uploads or when uncertain about scope.
