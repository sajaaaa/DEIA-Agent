"""
Human Agent for Overcooked-AI
允许真实人类通过终端输入控制游戏角色
"""

from overcooked_ai_py.mdp.actions import Action, Direction


class HumanAgent:
    """
    人类玩家Agent
    每个时间步暂停游戏，等待人类输入动作
    """
    
    def __init__(self):
        self.agent_index = None
        self.mdp = None
        
        # 动作映射
        self.action_map = {
            # 方向键 (支持多种输入方式)
            'w': Direction.NORTH,
            'up': Direction.NORTH,
            'n': Direction.NORTH,
            '↑': Direction.NORTH,
            
            's': Direction.SOUTH,
            'down': Direction.SOUTH,
            '↓': Direction.SOUTH,
            
            'a': Direction.WEST,
            'left': Direction.WEST,
            '←': Direction.WEST,
            
            'd': Direction.EAST,
            'right': Direction.EAST,
            '→': Direction.EAST,
            
            # 交互
            'e': Action.INTERACT,
            'interact': Action.INTERACT,
            'i': Action.INTERACT,
            'space': Action.INTERACT,
            
            # 停留
            'stay': Action.STAY,
            'wait': Action.STAY,
            'x': Action.STAY,
            '': Action.STAY,  # 直接回车 = 停留
        }
        
    def set_agent_index(self, agent_index):
        self.agent_index = agent_index
        
    def set_mdp(self, mdp):
        self.mdp = mdp
    
    def reset(self):
        pass
    
    def action(self, state):
        """
        获取人类玩家的动作输入
        """
        player = state.players[self.agent_index]
        held_obj = player.held_object.name if player.held_object else "nothing"
        
        print(f"\n{'='*50}")
        print(f"🎮 HUMAN PLAYER {self.agent_index} - YOUR TURN!")
        print(f"{'='*50}")
        print(f"Your position: {player.position}")
        print(f"Your orientation: {self._orientation_to_arrow(player.orientation)}")
        print(f"You are holding: {held_obj}")
        print(f"\n📋 Available actions:")
        print(f"  Movement: w/↑(up), s/↓(down), a/←(left), d/→(right)")
        print(f"  Interact: e/i/space (pickup, put down, use)")
        print(f"  Stay:     x/enter (do nothing)")
        print(f"{'='*50}")
        
        while True:
            try:
                user_input = input(f"Player {self.agent_index}, enter your action: ").strip().lower()
                
                if user_input in self.action_map:
                    action = self.action_map[user_input]
                    print(f"✓ Action: {self._action_to_string(action)}")
                    return action
                else:
                    print(f"❌ Invalid input '{user_input}'. Please try again.")
                    print(f"   Valid inputs: w/s/a/d (move), e/i (interact), x/enter (stay)")
                    
            except KeyboardInterrupt:
                print("\n\n⚠️ Game interrupted by user. Defaulting to STAY.")
                return Action.STAY
            except EOFError:
                print("\n\n⚠️ Input stream ended. Defaulting to STAY.")
                return Action.STAY
    
    def _orientation_to_arrow(self, orientation):
        """将方向转换为箭头符号"""
        arrows = {
            (0, -1): '↑ (North)',
            (0, 1): '↓ (South)',
            (-1, 0): '← (West)',
            (1, 0): '→ (East)',
        }
        return arrows.get(orientation, str(orientation))
    
    def _action_to_string(self, action):
        """将动作转换为可读字符串"""
        if action == Action.INTERACT:
            return "INTERACT (pickup/put/use)"
        elif action == Action.STAY:
            return "STAY (do nothing)"
        elif action == Direction.NORTH:
            return "MOVE UP ↑"
        elif action == Direction.SOUTH:
            return "MOVE DOWN ↓"
        elif action == Direction.WEST:
            return "MOVE LEFT ←"
        elif action == Direction.EAST:
            return "MOVE RIGHT →"
        else:
            return str(action)


class HumanAgentWithSkills(HumanAgent):
    """
    增强版人类玩家Agent
    支持输入高级技能（如pickup_onion），自动转换为底层动作序列
    """
    
    def __init__(self):
        super().__init__()
        
        # 高级技能映射
        self.skill_map = {
            'pickup_onion': 'pickup_onion',
            'pickup_dish': 'pickup_dish',
            'put_onion_in_pot': 'put_onion_in_pot',
            'fill_dish_with_soup': 'fill_dish_with_soup',
            'deliver_soup': 'deliver_soup',
            'place_obj_on_counter': 'place_obj_on_counter',
        }
        
        self.current_skill = None
        self.skill_action_queue = []
    
    def action(self, state):
        """
        获取人类玩家的动作输入
        支持底层动作和高级技能
        """
        player = state.players[self.agent_index]
        held_obj = player.held_object.name if player.held_object else "nothing"
        
        print(f"\n{'='*60}")
        print(f"🎮 HUMAN PLAYER {self.agent_index} - YOUR TURN!")
        print(f"{'='*60}")
        print(f"Your position: {player.position}")
        print(f"Your orientation: {self._orientation_to_arrow(player.orientation)}")
        print(f"You are holding: {held_obj}")
        print(f"\n📋 Available inputs:")
        print(f"  [Movement]  w/↑(up), s/↓(down), a/←(left), d/→(right)")
        print(f"  [Interact]  e/i/space")
        print(f"  [Stay]      x/enter")
        print(f"\n📋 Or enter a skill name:")
        print(f"  pickup_onion, pickup_dish, put_onion_in_pot")
        print(f"  fill_dish_with_soup, deliver_soup, place_obj_on_counter")
        print(f"{'='*60}")
        
        while True:
            try:
                user_input = input(f"Player {self.agent_index}, enter action/skill: ").strip().lower()
                
                # 检查是否是底层动作
                if user_input in self.action_map:
                    action = self.action_map[user_input]
                    print(f"✓ Action: {self._action_to_string(action)}")
                    return action
                
                # 检查是否是高级技能
                elif user_input in self.skill_map:
                    print(f"✓ Skill: {user_input}")
                    print(f"  (This will be executed as a sequence of low-level actions)")
                    # 返回技能名称，由上层处理
                    return ('skill', user_input)
                
                else:
                    print(f"❌ Invalid input '{user_input}'. Please try again.")
                    
            except KeyboardInterrupt:
                print("\n\n⚠️ Game interrupted. Defaulting to STAY.")
                return Action.STAY
            except EOFError:
                return Action.STAY
