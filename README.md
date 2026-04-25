# 南京大学交通实训课程作业

[![Python](https://img.shields.io/badge/python-3.8%2B-blue.svg)](pyproject.toml)
[![Package Check](https://github.com/handsomeZR-netizen/nju-traffic-practicum-coursework/actions/workflows/package-check.yml/badge.svg)](https://github.com/handsomeZR-netizen/nju-traffic-practicum-coursework/actions/workflows/package-check.yml)
[![Platform](https://img.shields.io/badge/platform-Tencent%20Kaiwu-0052D9.svg)](https://github.com/handsomeZR-netizen/nju-traffic-practicum-coursework)
[![Algorithm](https://img.shields.io/badge/algorithm-DIY%20%2F%20PPO-2ea44f.svg)](agent_diy)
[![License](https://img.shields.io/badge/license-MIT%20%2B%20notices-orange.svg)](NOTICE.md)

基于腾讯开悟平台智能交通信号灯调度任务完成的课程实训项目。仓库围绕 `DIY` 策略槽位进行了代码收敛、特征设计、训练配置迁移、云端评测联调与实验记录整理，并补齐了可复用的安装元数据、最小上传包生成脚本、同步批次工具和 GitHub Actions 校验。

当前主线已经统一切换到 `normal` 环境，并进入“低评测消耗”的 checkpoint 确认阶段。此前固定环境 / `easy` 线结果保留为探索历史，不再直接参与当前主线成绩判断。

## 快速复用

```bash
git clone https://github.com/handsomeZR-netizen/nju-traffic-practicum-coursework.git
cd nju-traffic-practicum-coursework
python -m pip install -r requirements.txt
python -m pip install -e .
```

注意：`kaiwu_agent`、`kaiwudrl` 和平台 `tools/` 脚本由腾讯开悟运行环境提供，本地普通 Python 环境主要用于阅读、静态检查、打包和复用策略代码。

生成腾讯开悟 `DIY` 槽位的最小上传目录：

```bash
python build_upload_diy_package.py
```

生成只包含当前改动的网页端上传批次：

```bash
python automation/build_diy_sync_batch.py
```

如果本次也改了平台配置：

```bash
python automation/build_diy_sync_batch.py --with-conf
```

生成完整允许范围清单：

```bash
python automation/sync_manifest.py --json-out automation/data/sync_manifest.json
```

## 当前状态

- 当前主线算法槽位：`DIY`
- 当前主线代码目录：`agent_diy/`
- 当前主线环境层级：`normal`
- 当前主线配置文件：`agent_diy/conf/train_env_conf.toml`
- 当前主线实验记录：`实验记录与下一步计划.md`
- 当前剩余评测次数：`46`

截至 `2026-04-18`，`normal` 主线已确认的关键结果为：

| 候选 | 说明 | 得分 | 当前结论 |
| --- | --- | ---: | --- |
| `normal-diy-30-30` | 当前主基线来源 | `2048.24` 左右均值 | 保留 |
| `27min checkpoint` | 从主基线续训筛出 | `2058.41` | 强候选 |
| `60min checkpoint` | 从主基线续训筛出 | `2059.20` | 强候选 |

已淘汰 checkpoint：`14min`、`56min`、`85min`。

下一阶段重点不是继续横向筛更多 checkpoint，而是用尽可能少的评测次数确认 `27min` 和 `60min` 哪个更适合作为后续代码修改前的保底基线。

## 项目亮点

- 基于腾讯开悟智能交通环境完成 `DIY` 智能体训练与评测闭环
- 在平台目录限制下将策略实现集中到 `agent_diy/`，便于云端直接替换与回滚
- 建立训练记录、评测复盘、配置迁移、最小上传包生成和同步批次生成流程
- 明确区分 `easy` 与 `normal` 两条实验线，避免不同环境下结果被错误混用
- 在评测预算受限的情况下，形成低评测消耗的 checkpoint 筛选流程
- 保留 `PI-eLight-main/` 作为公开策略思路参考，实际提交主线仍以 `agent_diy/` 为准

## 目录结构

| 路径 | 用途 |
| --- | --- |
| `agent_diy/` | 当前主线策略实现 |
| `agent_ppo/` | 平台原始 PPO 参考实现 |
| `conf/` | 平台算法与应用配置 |
| `train_test.py` | 腾讯开悟训练与联调入口 |
| `build_upload_diy_package.py` | 生成 `upload_diy_minimal/` 的辅助脚本 |
| `upload_diy_minimal/` | 云端复制粘贴使用的最小上传包 |
| `automation/` | 同步清单与上传批次生成脚本 |
| `.codex/skills/kaiwu-diy-upload/` | 可选的仓库专用 Codex skill，用于远端上传约束 |
| `PI-eLight-main/` | 第三方公开思路参考 |
| `实验记录与下一步计划.md` | 当前 `normal` 主线实验结论与后续计划 |
| `训练评测协作模板.md` | 后续协作时统一的信息模板 |
| `腾讯开悟网页同步说明.md` | Browser MCP / Chrome DevTools MCP 同步说明 |

## 工程说明

本项目除了策略优化外，还重点完成了以下工程整理工作：

- 将原先分散的实现整理到 `DIY` 官方槽位，避免自定义目录在云端失效
- 保留平台默认结构，尽量减少对 `conf/kaiwudrl/` 与平台工具链的侵入式修改
- 补充初始化文档、上传说明和实验记录，降低平台环境切换成本
- 对训练环境配置进行显式管理，支持按不同评测标准快速切换
- 通过 `pyproject.toml`、`requirements.txt`、`MANIFEST.in` 和 GitHub Actions 提供基础复用入口

## 训练与评测指标

项目当前围绕以下核心指标进行训练与评测：

- 总得分 `score`
- 平均排队长度 `avg_queue_len`
- 平均车辆延误 `avg_delay`
- 平均车辆等待时间 `avg_waiting_time`
- 信号变化频率 `signal_change_frequency`

由于不同环境层级会直接影响结果可比性，仓库内的实验记录会显式标注：

- 当前结果属于 `easy` 还是 `normal`
- 是否来自续训 checkpoint
- 当前剩余评测次数
- 是否可以作为当前主线结论

## 网页端同步工作流

当前仓库使用本地脚本生成同步批次，再由 `Browser MCP` 或 `Chrome DevTools MCP` 接管已经登录的腾讯开悟网页 IDE 执行写入。

日常只同步 `agent_diy/` 和 `train_test.py`：

```bash
python automation/build_diy_sync_batch.py
```

连同改动过的 `conf/**` 一起同步：

```bash
python automation/build_diy_sync_batch.py --with-conf
```

同步所有允许范围内的改动：

```bash
python automation/build_diy_sync_batch.py --all-allowed
```

当前允许同步的远端范围固定为：

- `agent_diy/`
- `agent_ppo/`
- `conf/`
- `log/`
- `train_test.py`

其中 `.py` 文件在生成上传批次时会自动插入最新的 `# Uploaded at: ...` 时间戳，但不会修改本地源码。详细说明见 `腾讯开悟网页同步说明.md`。

## 本地校验

```bash
python -m compileall agent_diy agent_ppo automation build_upload_diy_package.py train_test.py
python automation/sync_manifest.py --json-out automation/data/sync_manifest.json
python automation/build_diy_sync_batch.py --dry-run --with-conf
```

GitHub Actions 会在 push 和 pull request 时执行同类轻量检查。

## 许可证与第三方代码

仓库自写的包装、文档和辅助脚本在可授权范围内按 `LICENSE` 中的 MIT 条款开放。带有腾讯、PI-eLight、TinyLight 或其他第三方声明的文件保留其原始声明和约束。复用前请阅读 `NOTICE.md`。
