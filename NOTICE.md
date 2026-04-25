# Notice

This repository is packaged for educational reuse of the Tencent Kaiwu
intelligent traffic light scheduling coursework.

## Provenance

- `agent_diy/`, `agent_ppo/`, `conf/`, `train_test.py`, and
  `upload_diy_minimal/` include Tencent Kaiwu platform scaffold code. Files
  carrying Tencent copyright headers retain those notices.
- `PI-eLight-main/` is retained as third-party reference material for traffic
  signal control strategy comparison. It keeps its upstream acknowledgements
  and dependency notes.
- Repository-authored packaging, documentation, synchronization helpers, and
  glue code are released under the MIT license in `LICENSE`, where legally
  permitted and where no stronger file-level notice applies.

## Reuse Guidance

For Tencent Kaiwu submissions, use `agent_diy/` as the main editable strategy
slot and generate a minimal upload copy with `build_upload_diy_package.py` or
the scripts under `automation/`.

For research or coursework reuse, keep external platform dependencies separate:
`kaiwu_agent`, `kaiwudrl`, and the `tools/` scripts are expected to be supplied
by the Tencent Kaiwu runtime.
