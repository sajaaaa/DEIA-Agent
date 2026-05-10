"""
Intent-aware Task-Driven Priority Module (ITDP)
意图感知的任务驱动优先级模块

核心创新：
1. TDP思想：以任务瓶颈为中心决策
2. 意图感知：推断队友正在处理哪个瓶颈，避免重复劳动
3. 动态分工：不是学习固定偏好，而是实时协调

决策流程：
任务分析 → 瓶颈列表 → 队友意图推断 → 选择互补瓶颈 → 执行

与其他方法对比：
- ProAgent: 推断队友意图 → 配合队友（被动）
- RACE: 学习队友偏好 → 互补分工（滞后）
- TDP: 识别瓶颈 → 解决瓶颈（可能冲突）
- ITDP: 识别瓶颈 + 推断队友意图 → 互补解决（主动且协调）

【新增文件】请放在 proagent/ 目录下
"""

from collections import defaultdict, deque
from enum import Enum
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass


# ============================================================
# 第一部分：任务瓶颈分析（来自TDP）
# ============================================================

class Bottleneck(Enum):
    """瓶颈类型 - 按流水线顺序"""
    NEED_INGREDIENT = "need_ingredient"          # 需要食材
    NEED_POT_FILLING = "need_pot_filling"        # 锅需要填充
    WAITING_COOK = "waiting_cook"                # 等待煮熟
    NEED_DISH = "need_dish"                      # 需要盘子
    NEED_PLATING = "need_plating"                # 需要装汤
    NEED_DELIVERY = "need_delivery"              # 需要送餐
    NONE = "none"


@dataclass
class BottleneckInfo:
    """瓶颈详细信息"""
    type: Bottleneck
    priority: int           # 优先级（越小越紧急）
    description: str
    required_action: str    # 解决该瓶颈需要的动作
    location_hint: str      # 位置提示（用于距离计算）


class TaskPipelineAnalyzer:
    """
    任务流水线分析器
    分析当前状态，返回所有瓶颈（按优先级排序）
    """
    
    def __init__(self):
        pass
    
    def analyze(self, kitchen_state: Dict) -> List[BottleneckInfo]:
        """
        分析厨房状态，返回所有瓶颈列表（按优先级排序）
        
        核心思想：不是只返回最紧急的瓶颈，而是返回所有瓶颈
        这样可以让agent选择队友没在处理的瓶颈
        """
        bottlenecks = []
        
        pot_items = kitchen_state.get('pot_items', 0)
        soup_ready = kitchen_state.get('soup_ready', False)
        soup_cooking = kitchen_state.get('soup_cooking', False)
        
        # === 从后往前分析流水线（后面的更紧急）===
        
        # 优先级1：汤好了需要送餐（最紧急，因为汤会变凉/游戏有时间限制）
        if soup_ready:
            bottlenecks.append(BottleneckInfo(
                type=Bottleneck.NEED_DISH,
                priority=1,
                description="Soup ready! Need dish to plate",
                required_action="pickup_dish",
                location_hint="dish_dispenser"
            ))
            bottlenecks.append(BottleneckInfo(
                type=Bottleneck.NEED_PLATING,
                priority=0,  # 如果有盘子，这个更紧急
                description="Soup ready! Need to plate",
                required_action="fill_dish_with_soup",
                location_hint="pot"
            ))
        
        # 优先级2：正在煮，可以准备盘子
        if soup_cooking:
            bottlenecks.append(BottleneckInfo(
                type=Bottleneck.WAITING_COOK,
                priority=3,
                description="Soup cooking, prepare dish",
                required_action="pickup_dish",
                location_hint="dish_dispenser"
            ))
            # 也可以为下一锅准备食材
            bottlenecks.append(BottleneckInfo(
                type=Bottleneck.NEED_INGREDIENT,
                priority=4,
                description="Prepare ingredients for next batch",
                required_action="pickup_onion",
                location_hint="onion_dispenser"
            ))
        
        # 优先级3：锅没满，需要食材
        if pot_items < 3 and not soup_cooking and not soup_ready:
            urgency = 2 if pot_items == 0 else 3  # 锅空更紧急
            bottlenecks.append(BottleneckInfo(
                type=Bottleneck.NEED_POT_FILLING,
                priority=urgency,
                description=f"Pot has {pot_items}/3, need more ingredients",
                required_action="pickup_onion",
                location_hint="onion_dispenser"
            ))
            # 如果已经有食材在手，需要放进锅
            bottlenecks.append(BottleneckInfo(
                type=Bottleneck.NEED_POT_FILLING,
                priority=urgency - 1,  # 放进锅比拿食材更紧急
                description=f"Put ingredient in pot ({pot_items}/3)",
                required_action="put_onion_in_pot",
                location_hint="pot"
            ))
        
        # 按优先级排序（数字越小越紧急）
        bottlenecks.sort(key=lambda b: b.priority)
        
        # 如果没有瓶颈，默认拿食材
        if not bottlenecks:
            bottlenecks.append(BottleneckInfo(
                type=Bottleneck.NEED_INGREDIENT,
                priority=5,
                description="Default: get ingredients",
                required_action="pickup_onion",
                location_hint="onion_dispenser"
            ))
        
        return bottlenecks


# ============================================================
# 第二部分：队友意图推断（集成Bayesian Delegation）
# ============================================================

# --- Bayesian Delegation 核心实现 ---
# 基于论文: "Too many cooks" (Wu et al., CogSci 2020)

import math

class BayesianTaskBelief:
    """
    贝叶斯任务信念 - 实现论文的核心算法
    
    P(ta|H₀:T) ∝ P(ta) × ∏ₜ P(aₜ|sₜ, ta)
    """
    
    def __init__(self, beta: float = 0.9):
        """
        Args:
            beta: 温度参数 (论文默认0.9)
        """
        self.beta = beta
        self.bottlenecks = list(Bottleneck)
        self.belief: Dict[Bottleneck, float] = {}
        self.last_state: Optional[Dict] = None
        self.reset()
    
    def reset(self):
        """重置为均匀先验"""
        n = len(self.bottlenecks)
        self.belief = {b: 1.0 / n for b in self.bottlenecks}
        self.last_state = None
    
    def update(self, teammate_state: Dict, kitchen_state: Dict) -> Dict[Bottleneck, float]:
        """
        贝叶斯更新 - 修复版
        添加衰减机制防止信念过度集中
        """
        held = teammate_state.get('held_object')
        pot_items = kitchen_state.get('pot_items', 0)
        soup_ready = kitchen_state.get('soup_ready', False)
        
        # 检测任务切换
        if self._detect_task_switch(teammate_state):
            self._soft_reset(0.3)
        
        # ========================================
        # 修复：每次更新前先应用衰减，防止信念过度集中
        # ========================================
        decay_factor = 0.98  # 每步衰减2%向均匀分布靠拢，允许信念更快收敛
        n = len(self.bottlenecks)
        uniform = 1.0 / n
        for b in self.bottlenecks:
            self.belief[b] = decay_factor * self.belief[b] + (1 - decay_factor) * uniform
        
        # 计算每个瓶颈的似然
        likelihoods = {}
        for b in self.bottlenecks:
            q = self._compute_q_value(b, held, pot_items, soup_ready)
            likelihoods[b] = math.exp(self.beta * q)
        
        # 贝叶斯更新
        total = 0.0
        for b in self.bottlenecks:
            self.belief[b] *= likelihoods[b]
            total += self.belief[b]
        
        # 归一化
        if total > 1e-10:
            for b in self.bottlenecks:
                self.belief[b] /= total
        else:
            self._soft_reset(0.5)

        self.last_state = teammate_state.copy()
        return self.belief.copy()

    def _compute_q_value(self, bottleneck: Bottleneck, held: Optional[str],
                         pot_items: int, soup_ready: bool) -> float:
        """近似Q值（模拟逆规划）"""
        # Q值矩阵：(held_object, bottleneck) → Q值
        q_matrix = {
            # 拿着soup
            ('soup', Bottleneck.NEED_DELIVERY): 3.0,
            ('soup', Bottleneck.NEED_PLATING): -2.0,
            ('soup', Bottleneck.NEED_POT_FILLING): -3.0,
            ('soup', Bottleneck.NEED_INGREDIENT): -3.0,
            ('soup', Bottleneck.NEED_DISH): -3.0,
            
            # 拿着dish
            ('dish', Bottleneck.NEED_PLATING): 2.5 if soup_ready else 1.5,
            ('dish', Bottleneck.NEED_DISH): -1.0,
            ('dish', Bottleneck.NEED_DELIVERY): 0.5 if soup_ready else -1.5,
            ('dish', Bottleneck.NEED_POT_FILLING): -2.5,
            ('dish', Bottleneck.NEED_INGREDIENT): -2.5,
            
            # 拿着onion
            ('onion', Bottleneck.NEED_POT_FILLING): 2.5 if pot_items < 3 else -1.5,
            ('onion', Bottleneck.NEED_INGREDIENT): -0.5,
            ('onion', Bottleneck.NEED_DISH): -2.5,
            ('onion', Bottleneck.NEED_PLATING): -2.5,
            ('onion', Bottleneck.NEED_DELIVERY): -2.5,
            
            # 手空
            (None, Bottleneck.NEED_INGREDIENT): 0.8 if pot_items < 3 else -0.5,
            (None, Bottleneck.NEED_DISH): 0.5 if soup_ready or pot_items >= 3 else 0.0,
            (None, Bottleneck.WAITING_COOK): 0.3,
            (None, Bottleneck.NONE): 0.2,
        }
        
        # 任务可行性调整
        feasibility = 0.0
        if bottleneck == Bottleneck.NEED_POT_FILLING and pot_items >= 3:
            feasibility = -2.0
        elif bottleneck == Bottleneck.NEED_PLATING and not soup_ready:
            feasibility = -1.0
        elif bottleneck == Bottleneck.NEED_DELIVERY and not soup_ready:
            feasibility = -1.5
        
        base_q = q_matrix.get((held, bottleneck), 0.0)
        return base_q + 0.3 * feasibility
    
    def _detect_task_switch(self, current_state: Dict) -> bool:
        """
        检测任务切换 - 增强版
        增加更多触发条件
        """
        if self.last_state is None:
            return False
        
        last_held = self.last_state.get('held_object')
        curr_held = current_state.get('held_object')
        
        ingredient_items = {'onion', 'tomato'}
        plating_items = {'dish', 'soup'}
        
        # 原有检测：手持物品类型变化
        if last_held in ingredient_items and curr_held in plating_items:
            return True
        if last_held in plating_items and curr_held in ingredient_items:
            return True
        if curr_held == 'soup' and last_held != 'soup':
            return True
        
        # 新增检测：手持物品从有到无，或从无到有
        # 已移除：捡起/放下本身不代表任务切换，过于敏感会导致信念频繁重置

        return False
    
    def _soft_reset(self, decay: float = 0.3):
        """软重置"""
        n = len(self.bottlenecks)
        uniform = 1.0 / n
        for b in self.bottlenecks:
            self.belief[b] = decay * self.belief[b] + (1 - decay) * uniform
    
    def get_most_likely(self) -> Tuple[Bottleneck, float]:
        """获取最可能的瓶颈"""
        best = max(self.belief, key=self.belief.get)
        return best, self.belief[best]
    
    def get_high_prob_bottlenecks(self, threshold: float = 0.15) -> Set[Bottleneck]:
        """获取高概率瓶颈集合"""
        return {b for b, p in self.belief.items() if p > threshold and b != Bottleneck.NONE}
    
    def get_confidence(self) -> float:
        """置信度（基于熵）"""
        entropy = 0.0
        for p in self.belief.values():
            if p > 1e-10:
                entropy -= p * math.log(p)
        max_entropy = math.log(len(self.bottlenecks))
        return 1.0 - (entropy / max_entropy) if max_entropy > 0 else 1.0


class TeammateIntentPredictor:
    """
    队友意图推断器（集成Bayesian Delegation）
    
    融合两种方法：
    1. 规则方法：对于强信号（如拿着soup），直接判断
    2. 贝叶斯方法：对于弱信号（如手空），使用概率推断
    
    理论基础：
    - 论文: "Too many cooks" (Wu et al., CogSci 2020)
    - 核心: P(ta|H) ∝ P(ta) × ∏ P(a|s,ta)
    """
    
    def __init__(self, history_size: int = 5, use_bayesian: bool = True, beta: float = 0.9):
        self.history_size = history_size
        self.use_bayesian = use_bayesian

        # 历史记录
        self.teammate_action_history = deque(maxlen=history_size)
        self.teammate_position_history = deque(maxlen=history_size)
        # 持物历史：用于检测反复拿放行为
        self.held_history = deque(maxlen=16)
        
        # 贝叶斯信念
        self.bayesian_belief = BayesianTaskBelief(beta=beta) if use_bayesian else None
        
        # 统计
        self.rule_count = 0
        self.bayesian_count = 0
        
    def update(self, teammate_action: Optional[str], teammate_pos: Optional[Tuple] = None):
        """更新队友观察历史"""
        if teammate_action:
            self.teammate_action_history.append(teammate_action)
        if teammate_pos:
            self.teammate_position_history.append(teammate_pos)
    
    def predict_intent(self, teammate_state: Dict, kitchen_state: Dict, 
                        my_agent_index: int = 0) -> Tuple[Set[Bottleneck], str]:
        """
        预测队友正在处理哪个/哪些瓶颈
        
        融合规则方法和贝叶斯方法
        """
        held = teammate_state.get('held_object')
        handling = set()
        reasons = []

        # 记录持物历史（每步都更新，用于反复拿放检测）
        self.held_history.append(held)

        # 更新贝叶斯信念（不管是否使用）
        if self.bayesian_belief:
            self.bayesian_belief.update(teammate_state, kitchen_state)

        # === 强信号：使用规则方法（最可靠）===

        if held == 'soup':
            self.rule_count += 1
            handling.add(Bottleneck.NEED_DELIVERY)
            reasons.append(f"[Rule] Holding soup → DELIVERY")
            if self.bayesian_belief:
                self.bayesian_belief._soft_reset(0.5)
            return handling, "; ".join(reasons)

        if held == 'dish':
            self.rule_count += 1
            if kitchen_state.get('soup_ready'):
                # 检测反复拿放：dish 在最近16步内出现又消失超过2次
                if self._detect_repeated_pickup_drop('dish', threshold=2):
                    # 队友反复拿放盘子，意图不可信，降级为弱信号
                    reasons.append(f"[Rule] Holding dish but repeated pickup/drop detected → intent unreliable")
                    if self.bayesian_belief:
                        self.bayesian_belief._soft_reset(0.7)
                    handling = self.bayesian_belief.get_high_prob_bottlenecks(threshold=0.15) if self.bayesian_belief else set()
                    return handling, "; ".join(reasons)
                handling.add(Bottleneck.NEED_PLATING)
                reasons.append(f"[Rule] Holding dish + soup ready → PLATING")
            else:
                handling.add(Bottleneck.NEED_DISH)
                handling.add(Bottleneck.WAITING_COOK)
                reasons.append(f"[Rule] Holding dish → DISH preparation")
            if self.bayesian_belief:
                self.bayesian_belief._soft_reset(0.5)
            return handling, "; ".join(reasons)
        
        if held in ['onion', 'tomato']:
            self.rule_count += 1
            handling.add(Bottleneck.NEED_POT_FILLING)
            handling.add(Bottleneck.NEED_INGREDIENT)
            reasons.append(f"[Rule] Holding {held} → POT_FILLING")
            # 规则方法被使用时，软重置贝叶斯信念
            if self.bayesian_belief:
                self.bayesian_belief._soft_reset(0.5)
            return handling, "; ".join(reasons)
        
        # === 弱信号：使用贝叶斯方法 ===
        if self.use_bayesian and self.bayesian_belief:
            self.bayesian_count += 1
            handling = self.bayesian_belief.get_high_prob_bottlenecks(threshold=0.15)
            best, prob = self.bayesian_belief.get_most_likely()
            conf = self.bayesian_belief.get_confidence()
            reasons.append(f"[Bayesian] P({best.value})={prob:.1%}, conf={conf:.1%}")
            
            if handling:
                return handling, "; ".join(reasons)
        
        # === 回退：基于历史的规则 ===
        if self.teammate_action_history:
            recent = list(self.teammate_action_history)[-3:]
            
            if any('pickup_onion' in str(a) or 'pickup_tomato' in str(a) for a in recent if a):
                handling.add(Bottleneck.NEED_INGREDIENT)
                reasons.append(f"[History] Recently picked ingredient")
            
            if any('pickup_dish' in str(a) for a in recent if a):
                handling.add(Bottleneck.NEED_DISH)
                reasons.append(f"[History] Recently picked dish")
        
        # === 最后回退：默认分工 ===
        if not handling:
            if my_agent_index == 0:
                handling.add(Bottleneck.NEED_DISH)
                handling.add(Bottleneck.WAITING_COOK)
                reasons.append("[Default] Teammate handles DISH")
            else:
                handling.add(Bottleneck.NEED_INGREDIENT)
                handling.add(Bottleneck.NEED_POT_FILLING)
                reasons.append("[Default] Teammate handles INGREDIENT")
        
        if not reasons:
            reasons.append("Cannot infer teammate intent")
        
        return handling, "; ".join(reasons)

    def _detect_repeated_pickup_drop(self, item: str, threshold: int = 2) -> bool:
        """检测队友是否反复拿起又放下某个物品。
        统计 held_history 中 item→非item 的转换次数，超过 threshold 则认为行为异常。
        """
        transitions = 0
        prev = None
        for h in self.held_history:
            if prev == item and h != item:
                transitions += 1
            prev = h
        return transitions >= threshold

    def get_confidence(self) -> float:
        """推断置信度"""
        if self.bayesian_belief:
            return self.bayesian_belief.get_confidence()
        return min(1.0, len(self.teammate_action_history) / self.history_size)
    
    def get_belief_summary(self) -> str:
        """获取贝叶斯信念摘要"""
        if not self.bayesian_belief:
            return "Bayesian disabled"
        
        sorted_beliefs = sorted(self.bayesian_belief.belief.items(), 
                               key=lambda x: x[1], reverse=True)
        lines = []
        for b, p in sorted_beliefs[:3]:
            if p > 0.05:
                lines.append(f"{b.value}:{p:.0%}")
        return " | ".join(lines) if lines else "uniform"
    
    def get_statistics(self) -> Dict:
        """获取统计信息"""
        total = self.rule_count + self.bayesian_count
        return {
            'total': total,
            'rule_ratio': self.rule_count / total if total > 0 else 0,
            'bayesian_ratio': self.bayesian_count / total if total > 0 else 0,
        }
    
    def reset(self):
        self.teammate_action_history.clear()
        self.teammate_position_history.clear()
        if self.bayesian_belief:
            self.bayesian_belief.reset()
        self.rule_count = 0
        self.bayesian_count = 0


# ============================================================
# 第三部分：ITDP协调器（核心）
# ============================================================

class ITDPCoordinator:
    """
    ITDP协调器 - 意图感知的任务驱动优先级
    
    决策流程：
    1. 分析任务 → 获取瓶颈列表
    2. 推断队友意图 → 知道队友在处理什么
    3. 选择瓶颈 → 队友没在处理的最高优先级瓶颈
    4. 生成动作 → 解决选中的瓶颈
    """
    
    def __init__(self):
        self.pipeline_analyzer = TaskPipelineAnalyzer()
        self.intent_predictor = TeammateIntentPredictor()
        
        # 动作历史（用于卡住检测）
        self.my_action_history = deque(maxlen=10)
        self.stuck_counter = 0
        
    def update_teammate_observation(self, teammate_action: Optional[str], 
                                     teammate_pos: Optional[Tuple] = None):
        """更新队友观察"""
        self.intent_predictor.update(teammate_action, teammate_pos)
    
    def decide(self, kitchen_state: Dict, my_state: Dict, 
               teammate_state: Dict, my_agent_index: int = 0,
               reachability: Dict = None) -> Tuple[str, str, dict]:
        """
        核心决策函数
        
        Args:
            kitchen_state: 厨房状态
            my_state: 自己的状态 {'held_object': ...}
            teammate_state: 队友状态 {'held_object': ..., 'position': ...}
            my_agent_index: 我的agent索引
            reachability: 可达性字典 {'pot': bool, 'serve': bool, 'onion': bool, 'dish': bool}
        
        Returns:
            (动作, 原因, 调试信息)
        """
        # 默认可达性
        if reachability is None:
            reachability = {'pot': True, 'serve': True, 'onion': True, 'dish': True,
                           'pot_blocked': False, 'serve_blocked': False}
        
        my_held = my_state.get('held_object')
        
        # === Step 0: 强制规则（手持物品决定必须动作）===
        forced_action, forced_reason = self._check_forced_action(
            my_held, kitchen_state, reachability
        )
        if forced_action:
            return forced_action, forced_reason, {"type": "forced", "reachability": reachability}
        
        # === Step 1: 分析任务瓶颈 ===
        bottlenecks = self.pipeline_analyzer.analyze(kitchen_state)
        
        # === Step 2: 推断队友意图 ===
        teammate_handling, intent_reason = self.intent_predictor.predict_intent(
            teammate_state, kitchen_state, my_agent_index
        )
        
        # === Step 3: 选择互补瓶颈 ===
        selected = None
        for bn in bottlenecks:
            # 跳过队友正在处理的瓶颈
            if bn.type in teammate_handling:
                continue
            
            # 检查动作是否可行（包括可达性）
            if self._is_action_feasible(bn.required_action, my_held, kitchen_state, reachability):
                selected = bn
                break
        
        # 如果所有瓶颈都被队友处理，选择最紧急的（协助队友）
        if selected is None and bottlenecks:
            for bn in bottlenecks:
                if self._is_action_feasible(bn.required_action, my_held, kitchen_state, reachability):
                    selected = bn
                    break
        
        # === Step 4: 生成动作 ===
        if selected:
            action = selected.required_action
            reason = f"[ITDP] {selected.description} | Teammate: {intent_reason}"
        else:
            # Fallback: 根据可达性选择
            if reachability.get('onion', True):
                action = "pickup_onion"
                reason = "[ITDP] Fallback: get ingredients"
            elif reachability.get('dish', True):
                action = "pickup_dish"
                reason = "[ITDP] Fallback: get dish (cannot reach onion)"
            else:
                action = "wait(1)"
                reason = "[ITDP] Fallback: waiting for counter delivery (limited reachability)"
        
        # 卡住检测
        action = self._check_stuck(action)
        
        debug_info = {
            "type": "itdp",
            "bottlenecks": [(b.type.value, b.priority) for b in bottlenecks],
            "teammate_handling": [b.value for b in teammate_handling],
            "selected": selected.type.value if selected else None,
            "intent_confidence": self.intent_predictor.get_confidence(),
            "reachability": reachability
        }
        
        return action, reason, debug_info
    
    def _check_forced_action(self, held: Optional[str], 
                              kitchen_state: Dict,
                              reachability: Dict = None) -> Tuple[Optional[str], str]:
        """
        检查是否有强制动作（基于手持物品和可达性）
        
        Args:
            held: 手持物品
            kitchen_state: 厨房状态
            reachability: 可达性字典 {
                'pot': bool,      # 结构上能否到锅
                'serve': bool,    # 结构上能否到送餐点
                'onion': bool,    # 能否拿食材
                'dish': bool,     # 能否拿盘子
                'pot_blocked': bool,   # 锅是否被队友暂时挡住
                'serve_blocked': bool  # 送餐点是否被队友暂时挡住
            }
        """
        if reachability is None:
            reachability = {'pot': True, 'serve': True, 'onion': True, 'dish': True,
                           'pot_blocked': False, 'serve_blocked': False}
        
        can_reach_pot = reachability.get('pot', True)
        can_reach_serve = reachability.get('serve', True)
        pot_blocked = reachability.get('pot_blocked', False)
        serve_blocked = reachability.get('serve_blocked', False)
        
        soup_ready = kitchen_state.get('soup_ready', False)
        soup_cooking = kitchen_state.get('soup_cooking', False)
        pot_items = kitchen_state.get('pot_items', 0)
        
        if held == 'soup':
            if can_reach_serve:
                if serve_blocked:
                    # 送餐点被队友暂时挡住，等待
                    return 'wait(1)', '[FORCED] Holding soup, serve blocked by teammate → wait'
                else:
                    return 'deliver_soup', '[FORCED] Holding soup → must deliver'
            else:
                # 结构上无法送餐，放柜台让队友送
                return 'place_obj_on_counter', '[FORCED] Holding soup but cannot reach serve → place on counter for teammate'
        
        if held == 'dish':
            if soup_ready:
                if can_reach_pot:
                    if pot_blocked:
                        return 'wait(1)', '[FORCED] Holding dish + soup ready, pot blocked → wait for teammate'
                    else:
                        return 'fill_dish_with_soup', '[FORCED] Holding dish + soup ready → plate it'
                else:
                    return 'place_obj_on_counter', '[FORCED] Holding dish but cannot reach pot → place on counter for teammate'
            elif soup_cooking:
                if can_reach_pot:
                    if pot_blocked:
                        return 'wait(1)', '[FORCED] Holding dish + soup cooking, pot blocked → wait for teammate'
                    else:
                        return 'fill_dish_with_soup', '[FORCED] Holding dish + soup cooking → go to pot and wait'
                else:
                    return 'place_obj_on_counter', '[FORCED] Holding dish but cannot reach pot → place on counter for teammate'
            elif pot_items >= 3:
                if can_reach_pot:
                    if pot_blocked:
                        return 'wait(1)', '[FORCED] Holding dish + pot full, blocked → wait for teammate'
                    else:
                        return 'fill_dish_with_soup', '[FORCED] Holding dish + pot full → go to pot'
                else:
                    return 'place_obj_on_counter', '[FORCED] Holding dish but cannot reach pot → place on counter for teammate'
            else:
                return 'place_obj_on_counter', '[FORCED] Holding dish but pot not ready → put down'
        
        if held in ['onion', 'tomato']:
            if soup_ready:
                return 'place_obj_on_counter', '[FORCED] Soup ready, store ingredient for later'
            elif soup_cooking:
                return 'place_obj_on_counter', '[FORCED] Soup cooking, store ingredient for next batch'
            elif pot_items < 3:
                if can_reach_pot:
                    if pot_blocked:
                        # 锅被队友暂时挡住，等待而不是放柜台
                        return 'wait(1)', '[FORCED] Holding ingredient, pot blocked by teammate → wait'
                    else:
                        return 'put_onion_in_pot', '[FORCED] Holding ingredient → put in pot'
                else:
                    # 结构上无法到达锅，放柜台让队友拿
                    return 'place_obj_on_counter', '[FORCED] Holding ingredient but cannot reach pot → place on counter for teammate'
            else:
                return 'place_obj_on_counter', '[FORCED] Pot full → store ingredient'
        
        return None, ''
    
    def _is_action_feasible(self, action: str, held: Optional[str], 
                            kitchen_state: Dict,
                            reachability: Dict = None) -> bool:
        """
        检查动作是否可行
        
        Args:
            action: 要执行的动作
            held: 手持物品
            kitchen_state: 厨房状态
            reachability: 可达性字典 {'pot': bool, 'serve': bool, 'onion': bool, 'dish': bool}
        """
        if reachability is None:
            reachability = {'pot': True, 'serve': True, 'onion': True, 'dish': True,
                           'pot_blocked': False, 'serve_blocked': False}
        
        soup_ready = kitchen_state.get('soup_ready', False)
        soup_cooking = kitchen_state.get('soup_cooking', False)
        pot_items = kitchen_state.get('pot_items', 0)
        
        # 手里有东西，不能pickup
        if held and action.startswith('pickup'):
            return False
        
        # 手里没东西，不能put/deliver/fill
        if not held:
            if action in ['put_onion_in_pot', 'put_tomato_in_pot', 
                         'deliver_soup', 'fill_dish_with_soup', 
                         'place_obj_on_counter']:
                return False
        
        # 可达性检查
        if action in ['put_onion_in_pot', 'put_tomato_in_pot', 'fill_dish_with_soup']:
            if not reachability.get('pot', True):
                return False
        
        if action == 'deliver_soup':
            if not reachability.get('serve', True):
                return False
        
        if action == 'pickup_onion':
            if not reachability.get('onion', True):
                return False
        
        if action == 'pickup_dish':
            if not reachability.get('dish', True):
                return False
        
        # 汤没好也没在煮，不能fill_dish
        if action == 'fill_dish_with_soup' and not soup_ready and not soup_cooking:
            return False
        
        # 手里没汤，不能deliver
        if action == 'deliver_soup' and held != 'soup':
            return False
        
        return True
    
    def _check_stuck(self, action: str) -> str:
        """卡住检测和逃逸"""
        self.my_action_history.append(action)
        
        # 检查是否重复同一动作
        if len(self.my_action_history) >= 5:
            recent = list(self.my_action_history)[-5:]
            if len(set(recent)) == 1:
                self.stuck_counter += 1
                if self.stuck_counter > 2:
                    self.stuck_counter = 0
                    # 尝试等待
                    return 'wait(3)'
            else:
                self.stuck_counter = 0
        
        return action
    
    def get_prompt(self, kitchen_state: Dict, my_state: Dict, 
                   teammate_state: Dict) -> str:
        """生成LLM提示"""
        bottlenecks = self.pipeline_analyzer.analyze(kitchen_state)
        teammate_handling, intent_reason = self.intent_predictor.predict_intent(
            teammate_state, kitchen_state
        )
        
        my_held = my_state.get('held_object') or 'nothing'
        tm_held = teammate_state.get('held_object') or 'nothing'
        
        bottleneck_str = "\n".join([
            f"  {i+1}. [{b.type.value}] {b.description} → {b.required_action}"
            for i, b in enumerate(bottlenecks[:5])
        ])
        
        teammate_str = ", ".join([b.value for b in teammate_handling]) or "unknown"
        
        prompt = f"""
{'='*60}
ITDP: Intent-aware Task-Driven Priority Analysis
{'='*60}

[Kitchen Status]
• Pot: {kitchen_state.get('pot_items', 0)}/3 items
• Soup cooking: {kitchen_state.get('soup_cooking', False)}
• Soup ready: {kitchen_state.get('soup_ready', False)}

[Agent Status]
• I am holding: {my_held}
• Teammate holding: {tm_held}

[Task Bottlenecks] (sorted by priority)
{bottleneck_str}

[Teammate Intent Prediction]
• Teammate likely handling: {teammate_str}
• Reasoning: {intent_reason}
• Confidence: {self.intent_predictor.get_confidence():.0%}

[ITDP Decision Rule]
1. If I'm holding something → forced action based on what I hold
2. Otherwise → find highest priority bottleneck that teammate is NOT handling
3. This avoids duplicate work and maximizes parallel efficiency

[Coordination Principle]
"Don't compete with teammate on the same bottleneck. 
 Find what NEEDS to be done that teammate ISN'T doing."
{'='*60}"""
        
        return prompt
    
    def reset(self):
        self.intent_predictor.reset()
        self.my_action_history.clear()
        self.stuck_counter = 0


# ============================================================
# 第四部分：工具函数
# ============================================================

def visualize_coordination(kitchen_state: Dict, my_state: Dict, 
                           teammate_state: Dict, decision: str) -> str:
    """可视化协调状态"""
    pot = kitchen_state.get('pot_items', 0)
    cooking = '🔥' if kitchen_state.get('soup_cooking') else ''
    ready = '✓' if kitchen_state.get('soup_ready') else ''
    
    my_held = my_state.get('held_object') or '∅'
    tm_held = teammate_state.get('held_object') or '∅'
    
    viz = f"""
┌─────────────────────────────────────────┐
│  Pipeline: [🧅]→[🍲{pot}/3{cooking}{ready}]→[🍽️]→[📦]  │
│  Me: [{my_held}]  Teammate: [{tm_held}]       │
│  Decision: {decision:<25} │
└─────────────────────────────────────────┘"""
    return viz