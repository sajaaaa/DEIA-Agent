"""
Overcooked-AI 实验主程序 - 增强版
支持：
1. 自定义地图布局
2. 人类玩家交互
"""

import time
import datetime
import os
import json
import sys
from argparse import ArgumentParser
import numpy as np
from rich import print as rprint

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  
os.environ["CUDA_VISIBLE_DEVICES"] = "-1" 
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message=".*cuBLAS factory.*")


from distutils.util import strtobool
def boolean_argument(value):
    """Convert a string value to boolean."""
    return bool(strtobool(value))


# ============================================
# 日志记录器：同时输出到终端和文件
# ============================================
class Logger:
    def __init__(self, log_file):
        self.terminal = sys.stdout
        self.log_file = log_file
        self.log = open(log_file, "w", encoding="utf-8")
    
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()
    
    def flush(self):
        self.terminal.flush()
        self.log.flush()
    
    def close(self):
        self.log.close()


import importlib_metadata
VERSION = importlib_metadata.version("overcooked_ai")
print(f'\n----This overcook version is {VERSION}----\n')

from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld
from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv
from overcooked_ai_py.agents.agent import AgentGroup
from overcooked_ai_py.mdp.actions import Action

from utils import NEW_LAYOUTS, OLD_LAYOUTS, make_agent


# ============================================
# 自定义地图支持
# ============================================

# 预定义的自定义地图
# 注意：玩家位置使用1和2（不能用0）
CUSTOM_LAYOUTS = {
    # 简单的训练地图
    "simple_kitchen": [
        "XXPXX",
        "O   O",
        "X 2 X",
        "X1  X",
        "XDXSX"
    ],
    
    # 长走廊地图 - 测试协调
    "long_corridor": [
        "XXPXXXXXX",
        "O  1    S",
        "X       X",
        "X    2  X",
        "XXXXDXXXX"
    ],
    
    # 双锅地图 - 测试任务分配
    "dual_pots": [
        "XPXXXPX",
        "O     O",
        "X 1 2 X",
        "X     X",
        "X     X",
        "XXDXSXX"
    ],
    
    # 隔离地图 - 两个玩家在不同区域
    "separated": [
        "XXPXX",
        "O X O",
        "X1X2X",
        "X   X",
        "XDXSX"
    ],
    
    # 大厨房
    "large_kitchen": [
        "XXXPXPXXX",
        "O   1   O",
        "X       X",
        "X       X",
        "X   2   X",
        "O       O",
        "XXDXXXSXX"
    ],
}


def load_custom_layout(layout_name, custom_layout_file=None):
    """
    加载自定义地图
    
    Args:
        layout_name: 地图名称（预定义或内置）
        custom_layout_file: 自定义地图文件路径（可选）
    
    Returns:
        mdp: OvercookedGridworld对象
        layout_name: 使用的地图名称
    """
    
    # 1. 如果提供了自定义文件，从文件加载
    if custom_layout_file and os.path.exists(custom_layout_file):
        print(f"\n📂 Loading custom layout from file: {custom_layout_file}")
        with open(custom_layout_file, 'r') as f:
            content = f.read().strip()
        
        # 支持两种格式：纯文本网格 或 JSON
        if content.startswith('{'):
            # JSON格式
            layout_data = json.loads(content)
            grid = layout_data.get('grid', [])
            if isinstance(grid, str):
                grid = [line.strip() for line in grid.strip().split('\n')]
        else:
            # 纯文本格式，每行是地图的一行
            grid = [line.strip() for line in content.split('\n') if line.strip()]
        
        mdp = OvercookedGridworld.from_grid(grid)
        return mdp, f"custom_{os.path.basename(custom_layout_file)}"
    
    # 2. 检查是否是预定义的自定义地图
    if layout_name in CUSTOM_LAYOUTS:
        print(f"\n🗺️ Using predefined custom layout: {layout_name}")
        grid = CUSTOM_LAYOUTS[layout_name]
        mdp = OvercookedGridworld.from_grid(grid)
        return mdp, layout_name
    
    # 3. 使用内置地图
    if VERSION == '1.1.0':
        if layout_name in NEW_LAYOUTS:
            mdp = OvercookedGridworld.from_layout_name(NEW_LAYOUTS[layout_name])
        else:
            # 尝试直接使用layout_name
            mdp = OvercookedGridworld.from_layout_name(layout_name)
    elif VERSION == '0.0.1':
        if layout_name in OLD_LAYOUTS:
            mdp = OvercookedGridworld.from_layout_name(OLD_LAYOUTS[layout_name])
        else:
            mdp = OvercookedGridworld.from_layout_name(layout_name)
    
    return mdp, layout_name


def print_layout_help():
    """打印可用地图列表"""
    print("\n" + "="*60)
    print("📋 AVAILABLE LAYOUTS")
    print("="*60)
    
    print("\n[Built-in Layouts]")
    for name in ['cramped_room', 'asymmetric_advantages', 'coordination_ring', 
                 'forced_coordination', 'counter_circuit']:
        print(f"  • {name}")
    
    print("\n[Custom Layouts (predefined)]")
    for name, grid in CUSTOM_LAYOUTS.items():
        print(f"  • {name}")
        for row in grid[:3]:  # 只显示前3行
            print(f"      {row}")
        if len(grid) > 3:
            print(f"      ...")
    
    print("\n[Custom Layout from File]")
    print("  Use --custom_layout_file path/to/layout.txt")
    print("  Format: Each line is a row of the grid")
    print("  Symbols: X=wall, O=onion, D=dish, P=pot, S=serve, space=floor")
    print("="*60 + "\n")


# ============================================
# 主函数
# ============================================

def main(variant):
    layout = variant['layout']
    horizon = variant['horizon']
    episode = variant['episode']
    mode = variant['mode']
    custom_layout_file = variant.get('custom_layout_file')
    
    p0_algo = variant['p0']
    p1_algo = variant['p1']
    
    # 人类玩家检测
    has_human = p0_algo == "Human" or p1_algo == "Human"
    
    # ============================================
    # 创建实验目录和日志文件
    # ============================================
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    if variant['log_dir'] is None:
        log_dir = f"experiments/{timestamp}_{layout}_{p0_algo}_vs_{p1_algo}_{horizon}steps_{episode}ep"
    else:
        log_dir = variant['log_dir']
    
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    # 如果有人类玩家，不重定向stdout（保持交互）
    if not has_human:
        log_file = f"{log_dir}/experiment_log.txt"
        logger = Logger(log_file)
        sys.stdout = logger
    else:
        log_file = None
        logger = None
    
    print(f"=" * 60)
    print(f"Experiment Log")
    print(f"Timestamp: {timestamp}")
    print(f"Log Directory: {log_dir}")
    if has_human:
        print(f"🎮 HUMAN PLAYER MODE - Interactive input enabled")
    print(f"=" * 60)
    
    # ============================================
    # 加载地图
    # ============================================
    mdp, actual_layout_name = load_custom_layout(layout, custom_layout_file)
    
    print(f"\n🗺️ Layout: {actual_layout_name}")
    print(f"Grid size: {mdp.width} x {mdp.height}")
    
    env = OvercookedEnv(mdp, horizon=horizon)
    env.reset()

    print(f"\n===P0 agent: {p0_algo} | P1 agent: {p1_algo}===\n")

    start_time = time.time()
    results = []

    for i in range(episode):  
        print(f"\n{'#'*60}")
        print(f"# Episode {i+1}/{episode}")
        print(f"{'#'*60}\n")

        agents_list = []
        for idx, alg in enumerate([p0_algo, p1_algo]):
            if alg == "Human":
                # 创建人类玩家
                from human_agent import HumanAgent
                agent = HumanAgent()
                print(f"\n🎮 Player {idx} is controlled by HUMAN")
                
            elif alg == "ProAgent":
                assert variant['gpt_model'] is not None
                print(f"\n----Use {variant['gpt_model']}----\n")
                agent = make_agent(alg, mdp, actual_layout_name, 
                                   model=variant['gpt_model'], 
                                   prompt_level=variant['prompt_level'], 
                                   belief_revision=variant['belief_revision'], 
                                   retrival_method=variant['retrival_method'], 
                                   K=variant['K'])
                                   
            elif alg == "ITDP":
                assert variant['gpt_model'] is not None
                print(f"\n----Use {variant['gpt_model']} with ITDP-Agent----\n")
                agent = make_agent(alg, mdp, actual_layout_name, 
                                   model=variant['gpt_model'], 
                                   prompt_level=variant['prompt_level'], 
                                   belief_revision=variant['belief_revision'], 
                                   retrival_method=variant['retrival_method'], 
                                   K=variant['K'])
                                   
            elif alg == "BC":
                agent = make_agent(alg, mdp, actual_layout_name, seed_id=i)
            else:
                agent = make_agent(alg, mdp, actual_layout_name)
                
            agents_list.append(agent)

        team = AgentGroup(*agents_list)
        team.reset()

        env.reset()
        r_total = 0

        if mode == 'exp':
            for t in range(horizon):
                s_t = env.state
                
                # 打印当前状态
                print(f'\n{">"*15} time: {t} {"<"*15}\n')
                print(env.mdp.state_string(s_t).replace('ø', 'o'))
                print(f"\n💰 Current Score: {r_total}")
                
                # 如果有人类玩家，显示更多信息
                if has_human:
                    for idx, player in enumerate(s_t.players):
                        held = player.held_object.name if player.held_object else "nothing"
                        print(f"Player {idx}: pos={player.position}, holding={held}")

                # 获取动作
                a_t = team.joint_action(s_t) 
                
                print(f"\n-----------Controller-----------\n")    
                print(f"action: P0 {Action.to_char(a_t[0])} | P1 {Action.to_char(a_t[1])}")

                obs, reward, done, env_info = env.step(a_t)

                ml_actions = obs.ml_actions
                skills = ""
                for idx, ml_action in enumerate(ml_actions):
                    if ml_action is not None:
                        skills += f"P{idx} finished <{ml_action}>. "
                if skills:
                    print(skills)

                r_total += reward
                print(f'r: {reward} | total: {r_total}\n')
                
                if done:
                    print(f"\n🏁 Episode finished early at step {t}!")
                    break

            # 结束一个episode
            if p0_algo in ["ProAgent", "ITDP"] or p1_algo in ["ProAgent", "ITDP"]:
                print(f"\n================\n")
                try:
                    print(f"P1's real behavior: {team.agents[0].teammate_ml_actions_dict}")
                    print(f"The inferred P1's intention: {team.agents[0].teammate_intentions_dict}")
                except:
                    try:
                        print(f"P0's real behavior: {team.agents[1].teammate_ml_actions_dict}")
                        print(f"The inferred P0's intention: {team.agents[1].teammate_intentions_dict}")
                    except:
                        pass
                print(f"\n================\n")
            
        elif mode == 'demo':
            pass
         
        print(f"\n🏆 Episode {i+1}/{episode} Score: {r_total}\n")
        results.append(r_total)

    end_time = time.time()
    total_time = end_time - start_time
    print(f"\n⏱️ Total time: {total_time:.3f}s\n")

    # ============================================
    # 保存实验结果
    # ============================================
    result_dict = {
        "input": variant,
        "layout_used": actual_layout_name,
        "raw_results": results,
        "mean_result": float(np.mean(results)),
        "std_result": float(np.std(results)),
        "max_result": int(np.max(results)),
        "min_result": int(np.min(results)),
        "total_time_seconds": total_time,
    }
    
    print(f"\n{'='*60}")
    print(f"EXPERIMENT SUMMARY")
    print(f"{'='*60}")
    for k, v in result_dict.items():
        print(f'{k}: {v}')

    if variant['save']:
        if p0_algo in ["ProAgent", "ITDP"] or p1_algo in ["ProAgent", "ITDP"]:
            json_file = f"{log_dir}/results_{variant['gpt_model'].replace('/', '_')}_{variant['prompt_level']}.json"
        else:
            json_file = f"{log_dir}/results.json"
        
        with open(json_file, "w") as f:
            json.dump(result_dict, f, indent=4)
        
        print(f"\n📁 Results saved to: {json_file}")
    
    # 关闭日志
    if logger:
        sys.stdout = logger.terminal
        logger.close()
        print(f"\n[Experiment Complete] Log saved to: {log_dir}")
    else:
        print(f"\n[Experiment Complete]")
    
    return result_dict


if __name__ == '__main__':
    
    parser = ArgumentParser(description='OvercookedAI Experiment - Enhanced Version')

    # 基础参数
    parser.add_argument('--layout', '-l', type=str, default='cramped_room',
                        help='Layout name (built-in, custom predefined, or use with --custom_layout_file)')
    parser.add_argument('--custom_layout_file', type=str, default=None,
                        help='Path to custom layout file (overrides --layout)')
    parser.add_argument('--list_layouts', action='store_true',
                        help='List all available layouts and exit')
    
    parser.add_argument('--p0', type=str, default='Greedy', 
                        choices=['ITDP', 'ProAgent', 'Greedy', 'COLE', 'FCP', 'MEP', 'PBT', 'SP', 'BC', 'Random', 'Stay', 'Human'],
                        help='Algorithm for P0 (Player 0)')
    parser.add_argument('--p1', type=str, default='Greedy', 
                        choices=['ITDP', 'ProAgent', 'Greedy', 'COLE', 'FCP', 'MEP', 'PBT', 'SP', 'BC', 'Random', 'Stay', 'Human'],
                        help='Algorithm for P1 (Player 1)')
    parser.add_argument('--horizon', type=int, default=400, help='Horizon steps in one game')
    parser.add_argument('--episode', type=int, default=1, help='Number of episodes')

    # LLM相关参数
    parser.add_argument('--gpt_model', type=str, default='Qwen/Qwen2.5-7B-Instruct', 
                        choices=['text-davinci-003', 'gpt-3.5-turbo-16k', 'gpt-3.5-turbo-0301', 
                                 'gpt-3.5-turbo', 'gpt-4', 'gpt-4-0314',
                                 'Qwen/Qwen2-7B-Instruct', 'Qwen/Qwen2.5-7B-Instruct', 
                                 'Qwen/Qwen2.5-72B-Instruct', 'deepseek-ai/DeepSeek-V2.5', 
                                 'THUDM/glm-4-9b-chat'], 
                        help='LLM model to use')
    parser.add_argument('--prompt_level', '-pl', type=str, default='l2-ap', 
                        choices=['l1-p', 'l2-ap', 'l3-aip'])
    parser.add_argument('--belief_revision', '-br', type=boolean_argument, default=False)
    parser.add_argument('--retrival_method', type=str, default="recent_k", 
                        choices=['recent_k', 'bert_topk'])
    parser.add_argument('--K', type=int, default=1)

    # 其他参数
    parser.add_argument('--mode', type=str, default='exp', choices=['exp', 'demo'])
    parser.add_argument('--save', type=boolean_argument, default=True)
    parser.add_argument('--log_dir', type=str, default=None)
    parser.add_argument('--debug', type=boolean_argument, default=True)

    args = parser.parse_args()
    
    # 显示地图列表
    if args.list_layouts:
        print_layout_help()
        sys.exit(0)
    
    variant = vars(args)

    # 显示使用说明
    if args.p0 == 'Human' or args.p1 == 'Human':
        print("\n" + "🎮"*30)
        print("HUMAN PLAYER MODE")
        print("🎮"*30)
        print("\nControls:")
        print("  w / ↑  : Move Up (North)")
        print("  s / ↓  : Move Down (South)")  
        print("  a / ←  : Move Left (West)")
        print("  d / →  : Move Right (East)")
        print("  e / i  : Interact (pickup/put/use)")
        print("  x / Enter : Stay (do nothing)")
        print("\nPress Enter to start...\n")
        input()

    start_time = time.time()
    main(variant)
    end_time = time.time()
    print(f"\n{'='*30}")
    print(f"Finished! Total time: {end_time - start_time:.3f}s")
    print(f"{'='*30}\n")