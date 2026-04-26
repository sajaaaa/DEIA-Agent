---
name: ITDP-Agent Project Overview
description: 项目基本信息、架构、研究目标和关键文件
type: project
---

ITDP-Agent 是一个多智能体协作研究项目，在 Overcooked-AI 烹饪游戏环境中验证"意图感知的任务驱动优先级"协调方法。

**Why:** 现有方法（ProAgent/TDP/RACE）要么被动配合队友，要么可能和队友抢同一任务。ITDP 结合瓶颈分析和贝叶斯意图推断，主动做"队友没在做的最重要的事"。

**How to apply:** 理解用户的修改请求时，优先考虑对协调效果的影响，而不只是代码正确性。

## 核心文件

- `src/proagent/proagent.py` — ITDPAgent 类（继承 ProMediumLevelAgent），当前 generate_ml_action() 纯规则不调 LLM，LLM 版本已注释
- `src/proagent/itdp_module.py` — ITDPCoordinator 核心决策逻辑，贝叶斯意图推断
- `src/main.py` — 实验入口，Logger 类同时写终端和文件
- `src/utils.py` — make_agent() 工厂，layout 名称映射

## 技术栈

- Python 3.7，overcooked_ai（支持 0.0.1 和 1.1.0 两个版本，代码中有大量版本分支）
- LLM：硅基流动 API（兼容 OpenAI 接口），key 在 `src/siliconflow_key.txt`（不提交 git）
- BC/SP 等策略依赖 tensorflow + stable_baselines

## 当前版本状态（v12）

- 修复了 forced_coordination 地图上"结构不可达"被误判为"被队友阻挡"的 bug
- ITDPAgent 当前 100% 使用规则决策，LLM 增强版本注释在 proagent.py:1180-1376
- 实验结果自动保存到 `src/experiments/<timestamp>_<config>/`
