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
- [x] **消融实验运行中**：DEIA_no_intent + DEIA_no_priority，20 组，已完成
- [x] **消融结果分析**：量化贝叶斯意图模块和任务优先级队列各自贡献（no_intent均值143，no_priority均值135）

---

## 第四阶段：论文撰写

- [x] **整理完整实验数据表格**（消融实验已完成）
- [x] **撰写方法部分**：DEIA 框架设计 + 算法描述
- [x] **撰写实验部分**：主实验对比 + 消融分析
- [x] **撰写讨论与结论**：计算开销、泛化能力、局限性、未来工作
- [ ] **投稿目标**：CCF-C（待定）
- [ ] **显著性检验**：从实验日志提取各 episode 原始得分，对 DEIA vs ProAgent 各场景做 Welch t-test，输出 p 值（n=5 较小，仅作参考，投稿前视审稿要求决定是否加入论文）

---

## 第五阶段：规则模块优化（任务驱动改进）

- [ ] **优先级队列加入时间维度**：锅剩余烹饪步数动态调整 `NEED_DISH` / `NEED_INGREDIENT` 优先级，改善 forced_coordination 等时序敏感场景
- [ ] **多步前瞻（Look-ahead）**：在 `ITDPCoordinator.decide()` 中加入 2–3 步 BFS/贪心前瞻，预测执行当前推荐动作后的下一个瓶颈，避免"解决当前瓶颈后立刻陷入新瓶颈"
- [ ] **消融验证**：对比加入时间维度 / 多步前瞻前后的得分，量化各优化的贡献（重点关注 forced_coordination 和 coordination_ring）

> 注：贝叶斯 Q 值矩阵在线学习属于第二个点核心贡献，在第六阶段实现。

---

## 第六阶段：第二个点——泛化到未知队友 + 在线适应

### 核心问题
当前 DEIA 的贝叶斯 Q 值矩阵是针对 BC 行为手工设计的，换队友后意图推断准确率下降。
目标：让意图推断模块在线自适应，从交互中学习队友行为模式，不依赖固定先验。

### 实现任务

- [x] **基线测试（SP/FCP/MEP）**：已完成，30 组全部跑完（均值 SP=137, FCP=139, MEP=152，vs BC=148）
- [ ] **基线测试（PBT/COLE）**：模型已有，需补跑 20 组（2 对手 × 5 layout × 2 position × 5 ep），完善泛化基线表格
- [ ] **在线 Q 值自适应**：根据队友实际动作序列，在线更新贝叶斯 Q 值矩阵（梯度或计数统计均可）
- [ ] **队友类型识别**：设计轻量分类器，在前 N 步内识别队友策略类型（BC/SP/FCP/MEP），切换对应先验
- [ ] **自适应 DEIA 实验**：5 layout × 4 队友类型（BC/SP/FCP/MEP）× 2 position × 5 ep
- [ ] **对比实验**：固定先验 DEIA vs 自适应 DEIA，量化在线适应的收益
- [ ] **消融**：有/无队友类型识别模块，验证识别模块的贡献

### 预期结论
- 固定先验 DEIA 在非 BC 队友上得分下降 X%（待测）
- 在线自适应后恢复至接近 BC 队友水平
- 队友类型识别在前 N 步内收敛（N 待定）

### 鲁棒性实验：对抗 Bad Partner

**动机**：ProAgent 完全依赖 LLM 推理队友意图，遇到无规律队友时意图推断失效，得分趋近 0。
DEIA 的 ITDP 兜底机制（强制动作 + 任务瓶颈分析）应能保证基本得分，验证框架鲁棒性。

- [ ] **直接可用**：用现有 `RandomAgent`、`StayAgent` 跑 DEIA vs ProAgent 对比实验
- [ ] **实现 SpinAgent**：只做方向动作，原地转圈，不拿任何物品
- [ ] **实现 PickDropAgent**：拿起物品后立刻放下，循环执行，干扰物品布局
- [ ] **实现 BlockerAgent**：追着对方走，主动占据对方目标位置（需访问对方坐标）
- [ ] **对比实验**：DEIA vs ProAgent，各 bad partner 类型，5 layout × 5 ep
- [ ] **预期结论**：ProAgent 遇到 bad partner 得分接近 0；DEIA 因 ITDP 兜底仍能维持基本得分

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

### 消融实验（已完成）

| Layout | DEIA_no_intent | DEIA_no_priority | DEIA（完整） |
|--------|---------------|-----------------|-------------|
| cramped_room | 172 | 118 | 180 |
| asymmetric_advantages | 218 | 220 | 232 |
| coordination_ring | 138 | 146 | 152 |
| forced_coordination | 66 | 82 | 54 |
| counter_circuit | 120 | 108 | 120 |
| **均值** | **143** | **135** | **148** |
