# 操作日志

记录每次对话中对代码库的操作，包括任务描述、完成方式和修改内容。

---

## 2026-04-26

### 初始化项目文档

**任务：** 阅读 README，了解项目，进行 Claude 初始化

**完成方式：**
- 阅读 `README.md`、`src/main.py`、`src/proagent/proagent.py`、`src/utils.py` 等核心文件
- 了解项目架构、技术栈、运行方式

**创建的文件：**
- `CLAUDE.md` — 项目概览文档，供 Claude 在未来对话中快速了解项目结构、运行方式、关键设计决策
- `.claude/memory/MEMORY.md` — 记忆索引文件
- `.claude/memory/project_overview.md` — 项目记忆文件，记录项目背景、核心文件、技术栈、当前版本状态

**修改的文件：** 无

---

## 2026-04-26（第二次对话）

### 配置权限 & 实现 DEIA-Agent & 实验分析

---

#### 1. 配置 Claude Code 权限

**任务：** 设置文件读取/新建/修改自动允许，只对删除操作询问

**完成方式：** 创建 `.claude/settings.json`，配置 `permissions.allow` 和 `permissions.deny`

**创建的文件：**
- `.claude/settings.json` — 权限配置：Read/Write/Edit/mkdir/cp/mv 自动允许，rm/rmdir/unlink 拒绝

---

#### 2. 实现 DEIA-Agent（核心功能）

**任务：** 基于 ProAgent 框架，加入队友意图辅助识别和任务优先级分析，解决 LLM 协作三大痛点：
1. LLM 推理时间长，无法实时协作
2. LLM 识别队友意图耗时
3. 被动等待队友行动，协作效率低

**设计思路：**
- 用 `ITDPCoordinator` 快速预计算：① 队友意图贝叶斯概率分布 ② 任务瓶颈优先级队列
- 将预计算结果作为结构化上下文注入 LLM 提示词
- LLM 不再从头分析，只做最终决策 → 推理更快、更准

**修改的文件：**
- `src/proagent/proagent.py` — 新增 `DEIAAgent` 类（继承 `ITDPAgent`），包含：
  - `generate_ml_action()` — ITDP 预分析 + LLM 最终决策
  - `_build_deia_prompt_block()` — 构建结构化分析块（意图分布 + 瓶颈队列 + 可达性 + 推荐动作）
  - `_format_intent_distribution_block()` — ASCII 可视化队友意图概率分布
  - `_format_bottleneck_block()` — 格式化任务优先级队列
- `src/utils.py` — 导入 `DEIAAgent`，新增 `"DEIA"` 分支到 `make_agent()`
- `src/main.py` — argparse choices 加入 `DEIA`，episode 循环、结果保存逻辑同步支持 DEIA

---

#### 3. 首次实验：DEIA vs BC（cramped_room）

**配置：** DEIA (Qwen2.5-7B, l2-ap) vs BC | cramped_room | 400步 | 1局

**结果：** 得分 180（9次送餐），耗时 702秒，处于同类组合高位区间

**发现的问题：**
| 问题 | 严重程度 |
|------|---------|
| BC 无 `ml_actions`，贝叶斯信念无法更新，置信度全程 LOW | 高 |
| LLM 输出格式错误 10%（混淆 agent_index，动作名拼写错） | 高 |
| wait 占比 22%，过于保守 | 中 |

**创建的文件：**
- `src/experiments/analysis/experiment_analysis.md` — 实验分析记录文件

---

#### 4. 修复：BC 队友意图推断失效

**根因：**
- BC 不产生 `ml_actions`，`state.ml_actions[1-agent_index]` 始终为 `None`
- `predict_intent()` 每次检测到强信号（队友手持物品）后触发 `_soft_reset`，信念被反复清空回均匀分布

**修复方案（DEIAAgent 专属，不影响 ITDPAgent）：**
- `action()` 每步追踪队友手持物品变化，变化时调用 `_infer_tm_action_from_held()` 合成中层动作字符串（`pickup_onion`、`deliver_soup` 等），传给 `ITDPCoordinator`
- `_reinforce_belief_from_tm_held()` 在 `decide()` 返回后重注入信念：将队友手持物品对应瓶颈概率强制拉到 65%，覆盖 `soft_reset` 的破坏
- 刷新 `debug_info['intent_confidence']`，让 prompt 中显示重注入后的真实置信度

**修改的文件：**
- `src/proagent/proagent.py` — `DEIAAgent` 新增：
  - `reset()` — 初始化 `_prev_tm_held = None`
  - `action()` — 手持物品变化追踪 + 合成动作更新
  - `_infer_tm_action_from_held()` — 手持物品前后状态 → 中层动作映射
  - `_reinforce_belief_from_tm_held()` — 信念重注入，覆盖 soft_reset
  - `generate_ml_action()` — `decide()` 后调用重注入并刷新置信度

---

## 2026-04-26（第三次对话）

### 修复 LLM 格式错误 & wait 过高 & 启动批量实验

---

#### 5. 修复：LLM 格式错误（parse 失败回退）

**问题：** Qwen2.5-7B 约 10% 概率输出格式错误（混淆 agent_index，动作名拼写错），`parse_ml_action` 失败后返回 `wait(1)`，导致不必要的停顿

**修复方案：** 双层 fallback
- Fallback #1：`parse_ml_action` 返回 `wait(1)` 但响应中无 `wait` 关键词 → 直接用 `itdp_action`
- Fallback #2：解析结果包含 `wait` 但当前无锅/送餐堵塞，且 ITDP 推荐非 wait 动作 → 用 `itdp_action` 覆盖

**修改的文件：**
- `src/proagent/proagent.py` — `DEIAAgent.generate_ml_action()` 末尾加双层 fallback 逻辑

---

#### 6. 修复：wait 比例过高（22%）

**问题：** LLM 过于保守，在无堵塞场景仍频繁选 wait

**修复方案：**
- 提示词增加第 6 条明确规则："WAIT 是最后手段，只有在移动路径被完全阻挡时才选 wait"
- Fallback #2（见上）进一步兜底：ITDP 推荐非 wait 时强制覆盖 LLM 的 wait

**修改的文件：**
- `src/proagent/proagent.py` — `_build_deia_prompt_block()` 提示词增加明确反 wait 规则

---

#### 7. Baseline 策略决定

**决定：** BC+BC / Greedy+BC / SP / FCP / MEP / PBT / COLE 直接引用 ProAgent(AAAI 2024)论文数据，不重跑；ProAgent+BC(Qwen) 可选跑

**原因：** 节省实验时间，这些 baseline 不依赖 LLM，结果确定性高，ProAgent 原文数据可信

---

#### 8. 创建批量实验脚本 & 调试运行

**任务：** 5 layout × P0+P1 × 5 episode，全自动运行并汇总结果表

**调试过程：**
- `python3` → `ModuleNotFoundError: numpy`（系统 Python 无依赖）
- `python` → `PackageNotFoundError: overcooked_ai`（指向 Python 3.13）
- 改为完整 conda 路径：`/home/aj/miniconda3/envs/proagent/bin/python` → 成功

**创建的文件：**
- `src/run_deia_tonight.sh` — 批量实验脚本，输出 `results_table.txt`（Mean±Std 格式），`run.log` 完整日志

**实验状态：** 已启动（后台运行），10 组（5 layout × 2 position），预计耗时约 10 小时

---

## 2026-04-27

### 配置优化 & 忽略 paper 目录

---

#### 9. 扩展 Claude Code 权限配置

**任务：** 减少 Bash 命令的权限弹窗，避免每次都需要手动点 Yes

**修改方案：** 将 `.claude/settings.json` 的 `allow` 从枚举具体命令改为 `Bash(*)`（放行所有），`deny` 列表保留 `rm/rmdir/unlink` 拦截删除操作

**修改的文件：**
- `.claude/settings.json` — `allow` 改为 `["Read(*)", "Write(*)", "Edit(*)", "Bash(*)"]`，`deny` 保留删除拦截

---

#### 10. 忽略 paper 相关内容

**任务：** 将论文草稿目录加入 git 忽略，避免提交到仓库

**修改的文件：**
- `.gitignore` — 新增 `paper/` 规则

---

