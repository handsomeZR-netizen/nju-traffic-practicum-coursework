# DIY 最小上传说明

这份说明对应 `upload_diy_minimal/`。

## 核心规则

- 平台页面算法选择：`DIY`
- 代码目录必须使用：`agent_diy/`
- 不要再使用 `agent_pielight/` 作为云端主入口

## 最小上传包包含

- `agent_diy/`
- `conf/algo_conf_intelligent_traffic_lights.toml`
- `conf/app_conf_intelligent_traffic_lights.toml`
- `conf/configure_app.toml`
- `kaiwu.json`
- `train_test.py`

## 不包含

- `conf/kaiwudrl/`
- `tools/`

这两个目录必须保留云端现有版本，不能被最小上传包覆盖。

## 本地生成方式

```bash
python build_upload_diy_package.py
```

生成后目录为：

```text
upload_diy_minimal/
```

## 云端训练方式

上传 `upload_diy_minimal/` 中的文件后，在平台创建训练任务时直接选 `DIY`。

如果在云端终端做冒烟测试，使用：

```bash
cd /data/projects/intelligent_traffic_lights
/bin/python3 - <<'PY'
import train_test as t
t.algorithm_name = "diy"
t.train()
PY
```
