# DEIA-Agent 研究进度

## 第一阶段：DEIA 基础搭建与验证 ✅

- [x] **实现 DEIAAgent**：ITDP 预分析 + LLM 最终决策框架
- [x] **修复 BC 队友意图推断**：手持物品变化追踪 + 信念重注入
- [x] **修复 LLM 格式错误**：双层 fallback（parse 失败 / 无意义 wait）
- [x] **降低 wait 比例**：提示词反 wait 规则 + Fallback 兜底
- [x] **首次实验**：cramped_room DEIA vs BC，得分 180（基准验证）
- [x] **批量实验脚本**：5 layout × 5 episode 全自动运行

---

## 第二阶段：完整实验与结果分析 ✅

- [x] **DEIA+BC 全场景实验**：5 layout × 2 position × 5 ep
- [x] **ProAgent+BC(Qwen7B) 对比实验**：全场景完成，均值 93
- [x] **ITDP+BC 消融基线**：全场景完成，均值 139
- [x] **forced_coordination 修复**：wait(3)→wait(1) + 柜台传递 prompt（P0 从60→100）
- [x] **forced_coordination P1 问题**：决定不修，原始数据（ITDP 22 vs DEIA 54）已足够

---

## 第三阶段：消融实验（进行中）

- [x] **消融框架实现**：`ablation_mode` 参数，支持 `no_intent` / `no_priority`
- [x] **消融基线（DEIA w/o LLM = ITDP+BC）**：LLM 在 forced_coordination 贡献 +145%
- [ ] **消融实验运行中**：DEIA_no_intent + DEIA_no_priority，20 组，预计明天完成
- [ ] **消融结果分析**：量化贝叶斯意图模块和任务优先级队列各自贡献

---

## 第四阶段：论文撰写

- [ ] **整理完整实验数据表格**（等消融实验完成）
- [ ] **撰写方法部分**：DEIA 框架设计 + 算法描述
- [ ] **撰写实验部分**：主实验对比 + 消融分析
- [ ] **投稿目标**：CCF-C（待定）

---

## 当前关键数据

### 主实验对比（双向均值）

| Layout | BC+BC* | ProAgent+GPT4* | ProAgent+Qwen7B | ITDP+BC（消融） | DEIA+Qwen7B |
|--------|--------|---------------|-----------------|----------------|-------------|
| cramped_room | ~130 | ~190 | 128 | 180 | **180** |
| asymmetric_advantages | ~150 | ~230 | 156 | 232 | **232** |
| coordination_ring | ~100 | ~150 | 60 | 144 | **152** |
| forced_coordination | ~80 | ~100 | 12 | 22 | **54** |
| counter_circuit | ~90 | ~120 | 108 | 116 | **120** |
| **均值** | ~110 | ~158 | **93** | **139** | **148** |

*引用 ProAgent 论文（AAAI 2024）数据

### 消融实验（进行中）

| Layout | DEIA_no_intent | DEIA_no_priority | DEIA（完整） |
|--------|---------------|-----------------|-------------|
| cramped_room | 172 | 进行中 | 180 |
| 其余 layout | 待完成 | 待完成 | — |
