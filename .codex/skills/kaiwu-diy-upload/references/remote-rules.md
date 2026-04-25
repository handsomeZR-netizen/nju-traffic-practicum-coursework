# Remote Rules

## Allowed Remote Paths

- `agent_diy/**`
- `agent_ppo/**`
- `conf/**`
- `log/**`
- `train_test.py`

## Preferred Edit Scope

- First choice: `agent_diy/**`
- Second choice: `train_test.py`
- Only when needed: `conf/**`

## Do Not Upload

- `PI-eLight-main/**`
- `upload_diy_minimal/**`
- `automation/data/**`
- `automation/logs/**`
- Markdown docs
- Zip files
- Conversation logs

## Recommended Commands

DIY-only changed files:

```bash
python automation/build_diy_sync_batch.py
```

DIY + changed config:

```bash
python automation/build_diy_sync_batch.py --with-conf
```

All changed allowed files:

```bash
python automation/build_diy_sync_batch.py --all-allowed
```

All allowed files regardless of change scope:

```bash
python automation/build_sync_batch.py --scope full
```
