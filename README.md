# ITDP-Agent: Intent-aware Task-Driven Priority Agent

## 意图感知的任务驱动优先级智能体 - 零样本协调新方法

---

## 核心创新

### 方法对比

| 方法 | 决策依据 | 核心问题 |
|------|----------|----------|
| **ProAgent** | 推断队友意图 → 配合 | 被动配合，依赖准确推断 |
| **RACE** | 学习队友偏好 → 互补 | 适应滞后，偏好可能变化 |
| **TDP** | 识别任务瓶颈 → 解决 | 可能和队友抢同一任务 |
| **Bayesian Delegation** | 贝叶斯逆规划推断意图 | 计算复杂，需要精确Q值 |
| **ITDP (本文)** | **瓶颈分析 + 贝叶斯意图** | **主动且协调，避免冲突** |

### 核心思想

```
TDP的问题：我去解决瓶颈，但队友可能也在解决同一个瓶颈
ITDP的解决：我去解决"队友没在处理的"最高优先级瓶颈
```

**决策公式**：
```
我的任务 = argmax { 优先级(瓶颈) | 瓶颈 ∉ 队友正在处理的瓶颈集合 }
```

### Bayesian Delegation集成（v12新增）

基于论文 **"Too many cooks: Coordinating multi-agent collaboration through inverse planning"** (Wu et al., CogSci 2020)

**核心公式**：
```
P(ta|H₀:T) ∝ P(ta) × ∏ₜ P(aₜ|sₜ, ta)     [后验更新]
P(aₜ|sₜ, ta) ∝ exp(β × Q*(s, a))         [逆规划似然]
```

**实现方式**：
- 强信号（如拿着soup）→ 直接规则判断（快速、确定）
- 弱信号（如手空）→ 贝叶斯逆规划推断（概率化、考虑历史）

**参数**：
- β = 0.9（温度参数，论文默认值）
- 历史长度 = 10步

---

## 文件结构

```
proagent/
├── proagent.py      # 替换，包含ITDPAgent类
├── itdp_module.py   # 新增！ITDP核心模块
└── modules.py       # 原文件

src/
├── main.py          # 替换，添加ITDP选项 + 日志功能
└── utils.py         # 替换，添加ITDPAgent支持
```

---

## 使用方法

```bash
# ITDP vs BC
python main.py --layout cramped_room --p0 ITDP --p1 BC --horizon 400

# ITDP vs Greedy
python main.py --layout cramped_room --p0 ITDP --p1 Greedy --horizon 400

# 使用大模型
python main.py --layout cramped_room --p0 ITDP --p1 BC \
    --gpt_model Qwen/Qwen2.5-72B-Instruct --horizon 400

# 多episode实验
python main.py --layout cramped_room --p0 ITDP --p1 BC --horizon 400 --episode 5
```

---

## 日志保存功能（新增）

### 自动保存目录结构

```
experiments/
└── 20241229_153000_cramped_room_ITDP_vs_BC_400steps_1ep/
    ├── experiment_log.txt    # 完整实验日志（所有终端输出）
    └── results_Qwen_Qwen2.5-7B-Instruct_l2-ap.json  # 结果JSON
```

### 日志文件内容

`experiment_log.txt` 包含：
- 每个timestep的游戏状态
- ITDP的决策分析过程
- LLM的输入prompt和输出
- 动作执行结果
- 最终得分统计

### 结果JSON内容

```json
{
    "input": {...},           // 实验参数
    "raw_results": [120],     // 每个episode的得分
    "mean_result": 120.0,     // 平均分
    "std_result": 0.0,        // 标准差
    "max_result": 120,        // 最高分
    "min_result": 120,        // 最低分
    "total_time_seconds": 234.5  // 总耗时
}
```

---

## 决策流程

```
┌─────────────────────────────────────────┐
│  手持物品检查 (强制动作)                 │
│  ─────────────────────────              │
│  soup → deliver_soup                    │
│  dish+汤好 → fill_dish_with_soup        │
│  dish+汤煮 → wait(5)                    │
│  onion+汤好 → place_obj_on_counter      │
│  手空 → 继续...                         │
└─────────────────────┬───────────────────┘
                      │ (手空时)
                      ▼
┌─────────────────────────────────────────┐
│  1. 瓶颈分析 → [B1, B2, B3...]          │
│  2. 意图推断 → 队友正在处理 {Bx}         │
│  3. 互补选择 → 选择队友没做的最高优先级   │
│  4. LLM规划 → 生成具体动作               │
│  5. 验证修正 → 确保动作合法              │
└─────────────────────────────────────────┘
```

---

## 版本更新

### v12 (当前版本) - 修复"结构不可达"误判为"被队友阻挡"

**问题根源**（导致得分为0）：
```
forced_coordination地图：
- P1在左边区域(1,2)
- Pot在右边区域(3,0)和(4,1)
- 队友P0在(3,1)，距离pot仅1格

v11的错误判断：
  BFS找不到从P1到pot的路径
  但检测到队友在pot旁边（距离=1）
  → 错误地认为"结构可达，只是被队友阻挡"
  → 决策: wait(3) "等队友移开"

实际情况：
  即使队友移开，P1也到不了pot（有墙隔开）
  → 正确决策: place_obj_on_counter (放柜台让队友接力)
```

**v12修复**：先检测"忽略队友时能否到达"（结构可达性）
```python
# 1. 忽略队友时能否到达（检测结构可达性）
can_reach_pot_structural = _can_reach_any_location(my_pos, pot_locations, None)

# 2. 考虑队友时能否到达（检测当前可达性）
can_reach_pot_now = _can_reach_any_location(my_pos, pot_locations, teammate_pos)

if can_reach_pot_now:
    can_reach_pot = True, pot_blocked = False  # 当前就能到
elif can_reach_pot_structural:
    can_reach_pot = True, pot_blocked = True   # 被队友阻挡
else:
    can_reach_pot = False, pot_blocked = False # 结构不可达
```

### 决策对比

| 场景 | v11判断 | v11决策 | v12判断 | v12决策 |
|------|---------|---------|---------|---------|
| forced_coordination, P1拿洋葱 | pot=✓⏳ | wait(3) ❌ | pot=✗ | place_obj_on_counter ✓ |
| cramped_room, 队友挡住pot | pot=✓⏳ | wait(3) ✓ | pot=✓⏳ | wait(3) ✓ |
| cramped_room, 没人挡 | pot=✓ | put_onion_in_pot ✓ | pot=✓ | put_onion_in_pot ✓ |

### 预期日志输出（v12）

```
forced_coordination地图：
### ITDP-Agent Analysis
My holding: onion, Teammate holding: nothing
Kitchen: pot=0/3, ready=False, cooking=False
Reachability: pot=✗ serve=✗ onion=✓ dish=✓   ← 正确：pot结构不可达
  My pos: (1, 2), Teammate pos: (3, 1)
ITDP Decision: place_obj_on_counter           ← 正确：放柜台让队友拿

[FORCED ACTION] Holding ingredient but cannot reach pot → place on counter for teammate
```

### 预期得分提升

- **v11**: 0分（一直wait，从不传递物品）
- **v12**: 60-80分（正确放柜台，形成传递链）

---

## 联系方式

如有问题，请随时反馈！
