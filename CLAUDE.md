# CLAUDE.md — proagent-DEIA

## 项目概述

**ITDP-Agent**：意图感知的任务驱动优先级智能体（Intent-aware Task-Driven Priority Agent），基于 Overcooked-AI 环境的多智能体协作研究项目。

核心创新：结合任务瓶颈分析（TDP）和贝叶斯意图推断，让智能体做"队友没在做的最重要的事"。

## 环境依赖

- Python 3.7（`__pycache__` 显示 cpython-37）
- `overcooked_ai` 支持两个版本：`1.1.0` 和 `0.0.1`，代码中有大量版本分支
- `stable_baselines`（BC agent 依赖）
- `tensorflow`（itdp_module 和 BC 模型依赖）
- `importlib_metadata`（替代 pkg_resources 获取版本号）
- `rich`（终端彩色输出）
- API：硅基流动（SiliconFlow）兼容 OpenAI 接口，key 存于 `src/siliconflow_key.txt`

## 文件结构

```
src/
├── main.py              # 入口，含 Logger 类（同时写终端和文件）
├── utils.py             # make_agent() 工厂函数，NEW_LAYOUTS/OLD_LAYOUTS 映射
├── run.sh / run_comparison.sh  # 实验脚本
├── proagent/
│   ├── proagent.py      # ProAgent、ProMediumLevelAgent、ITDPAgent 类
│   ├── itdp_module.py   # ITDPCoordinator、贝叶斯意图推断核心逻辑
│   └── modules.py       # LLM 调用封装（Module 类）
├── prompts/gpt/planner/ # LLM prompt 文件，按 layout 和 agent_index 命名
├── models/bc_runs/      # BC agent 预训练模型
└── experiments/         # 实验日志自动保存目录
```

## 运行方式

```bash
cd src

# ITDP（纯规则，不调用 LLM）
python main.py --layout cramped_room --p0 ITDP --p1 BC --horizon 400

# ProAgent（调用 LLM）
python main.py --layout cramped_room --p0 ProAgent --p1 BC \
    --gpt_model Qwen/Qwen2.5-7B-Instruct --horizon 400 -pl l2-ap

# 多 episode
python main.py --layout cramped_room --p0 ITDP --p1 BC --horizon 400 --episode 5
```

支持的 layout：`cramped_room`, `asymmetric_advantages`, `coordination_ring`, `forced_coordination`, `counter_circuit`

支持的 agent：`ITDP`, `ProAgent`, `Greedy`, `BC`, `SP`, `FCP`, `MEP`, `PBT`, `COLE`, `Random`, `Stay`, `Human`

## 关键设计决策

### ITDPAgent.generate_ml_action()
当前版本（v12）**不调用 LLM**，100% 使用 `ITDPCoordinator.decide()` 规则决策。原来的 LLM 增强版本已注释掉（proagent.py:1180-1376）。

### 可达性检测（v12 修复）
`_check_reachability()` 区分两种不可达：
- 结构不可达（有墙）→ `can_reach=False`
- 被队友暂时阻挡 → `can_reach=True, blocked=True`

### API 适配
`ProAgent._is_siliconflow_model()` 检测模型名关键词（Qwen/deepseek/glm/THUDM），自动切换到硅基流动 API key 文件。

### 实验日志
`main.py` 的 `Logger` 类重定向 `sys.stdout`，所有 print 同时写入 `experiments/<timestamp>_<config>/experiment_log.txt`。

## 注意事项

- `src/siliconflow_key.txt` 含 API key，不要提交到 git
- BC agent 只对标准 5 个 layout 有预训练模型，自定义 layout 会 fallback 到 Greedy
- `overcooked_ai` 版本差异较大，修改时注意两个版本分支都要处理
- `itdp_module.py` 依赖 tensorflow，但只用于加载 BC/SP 等策略模型
