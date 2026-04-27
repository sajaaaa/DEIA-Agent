import itertools, os, json, re
from collections import defaultdict
import numpy as np
import pkg_resources
import sys 
import copy 
from .modules import Module
from .itdp_module import ITDPCoordinator, visualize_coordination  # ITDP导入
from overcooked_ai_py.mdp.actions import Action, Direction
from overcooked_ai_py.planning.search import find_path 
from overcooked_ai_py.planning.search import get_intersect_counter 
from overcooked_ai_py.planning.search import query_counter_states 

cwd = os.getcwd()
openai_key_file = os.path.join(cwd, "openai_key.txt")
siliconflow_key_file = os.path.join(cwd, "siliconflow_key.txt")  # 新增
PROMPT_DIR = os.path.join(cwd, "prompts")

NAME_TO_ACTION = {
	"NORTH": Direction.NORTH,
	"SOUTH": Direction.SOUTH,
	"EAST": Direction.EAST,
	"WEST": Direction.WEST,
	"INTERACT": Action.INTERACT,
	"STAY": Action.STAY
}


class ProAgent(object):
	"""
	This agent uses GPT-3.5 to generate actions.
	"""
	def __init__(self, model="gpt-3.5-turbo"):
		self.agent_index = None
		self.model = model

		self.openai_api_keys = []
		self.load_openai_keys()
		self.key_rotation = True

	def _is_siliconflow_model(self):
		"""判断是否使用硅基流动模型"""
		keywords = ["Qwen", "deepseek", "glm", "THUDM"]
		return any(kw.lower() in self.model.lower() for kw in keywords)

	def load_openai_keys(self):
		if self._is_siliconflow_model():
			key_file = siliconflow_key_file
		else:
			key_file = openai_key_file
		
		if os.path.exists(key_file):
			with open(key_file, "r") as f:
				context = f.read()
			self.openai_api_keys = [k.strip() for k in context.split('\n') if k.strip()]
		else:
			raise FileNotFoundError(f"API key file not found: {key_file}")

	def openai_api_key(self):
		if self.key_rotation:
			self.update_openai_key()
		return self.openai_api_keys[0]

	def update_openai_key(self):
		self.openai_api_keys.append(self.openai_api_keys.pop(0))

	def set_agent_index(self, agent_index):
		raise NotImplementedError

	def action(self, state):
		raise NotImplementedError

	def reset(self):
		raise NotImplementedError


class ProMediumLevelAgent(ProAgent):
	"""
	This agent default to use GPT-3.5 to generate medium level actions.
	"""
	def __init__(
			self,
			mlam,
			layout,
			model='gpt-3.5-turbo',
			prompt_level='l2-ap', # ['l1-p', 'l2-ap', 'l3-aip']
			belief_revision=False,
			retrival_method="recent_k",
			K=1, 
			auto_unstuck=False,
			controller_mode='new', # the default overcooked-ai Greedy controller
			debug_mode='N', 
			agent_index=None,
			outdir = None 
	):
		super().__init__(model=model)

		self.trace = True 
		self.debug_mode = 'Y' 
		self.controller_mode = controller_mode 
		self.mlam = mlam
		self.layout = layout
		self.mdp = self.mlam.mdp
		
		self.out_dir = outdir 
		self.agent_index = agent_index

		self.prompt_level = prompt_level
		self.belief_revision = belief_revision

		self.retrival_method = retrival_method
		self.K = K
		
		self.prev_state = None
		self.auto_unstuck = auto_unstuck

		self.current_ml_action = None
		self.current_ml_action_steps = 0
		self.time_to_wait = 0
		self.possible_motion_goals = None
		self.pot_id_to_pos = []

		self.layout_prompt = self.generate_layout_prompt()


	def set_mdp(self, mdp):
		self.mdp = mdp

	def create_gptmodule(self, module_name, file_type='txt', retrival_method='recent_k', K=10):
		print(f"\n--->Initializing GPT {module_name}<---\n")    

		# prompt_file = os.path.join(PROMPT_DIR, self.model, module_name, self.layout+f'_{self.agent_index}.'+file_type)

		if "gpt" in self.model or "text-davinci" in self.model:
			model_name = "gpt"
		elif "claude" in self.model:
			model_name = "claude"
		else:
			model_name = "gpt"  # Qwen等使用gpt格式的prompt
	
		if module_name == "planner":
			prompt_file = os.path.join(PROMPT_DIR, model_name, module_name, self.prompt_level, f'{self.layout}_{self.agent_index}.{file_type}')
		elif module_name == "explainer":
			prompt_file = os.path.join(PROMPT_DIR, model_name, module_name, f'player{self.agent_index}.{file_type}')
		else:
			raise Exception(f"Module {module_name} not supported.")

		# print(prompt_file)
		with open(prompt_file, "r") as f:
			if file_type == 'json':
				messages = json.load(f)
			elif file_type == 'txt':
				messages = [{"role": "system", "content": f.read()}]
			else:
				print("Unsupported file format.")
		
		return Module(messages, self.model, retrival_method, K)

	def reset(self):
		self.planner.reset()
		self.explainer.reset()
		self.prev_state = None
		self.current_ml_action = None
		self.current_ml_action_steps = 0
		self.time_to_wait = 0
		self.possible_motion_goals = None
		self.current_timestep = 0
		self.teammate_ml_actions_dict = {}
		self.teammate_intentions_dict = {}

	def set_agent_index(self, agent_index):
		self.agent_index = agent_index
		self.planner = self.create_gptmodule("planner", retrival_method=self.retrival_method, K=self.K)
		self.explainer = self.create_gptmodule("explainer", retrival_method='recent_k', K=self.K)

		print(self.planner.instruction_head_list[0]['content'])

	def generate_layout_prompt(self):
		layout_prompt_dict = {
			"onion_dispenser": " <Onion Dispenser {id}>",
			"dish_dispenser": " <Dish Dispenser {id}>",
			"tomato_dispenser": " <Tomato Dispenser {id}>",
			"serving": " <Serving Loc {id}>",
			"pot": " <Pot {id}>",
		}
		layout_prompt = "Here's the layout of the kitchen:"
		for obj_type, prompt_template in layout_prompt_dict.items():
			locations = getattr(self.mdp, f"get_{obj_type}_locations")()
			for obj_id, obj_pos in enumerate(locations):
				layout_prompt += prompt_template.format(id=obj_id) + ","
				if obj_type == "pot":
					self.pot_id_to_pos.append(obj_pos)
		layout_prompt = layout_prompt[:-1] + ".\n"
		return layout_prompt
	  
	def generate_state_prompt(self, state):
		ego = state.players[self.agent_index]
		teammate = state.players[1 - self.agent_index]

		time_prompt = f"Scene {state.timestep}: "
		ego_object = ego.held_object.name if ego.held_object else "nothing"
		teammate_object = teammate.held_object.name if teammate.held_object else "nothing"
		ego_state_prompt = f"<Player {self.agent_index}> holds "
		if ego_object == 'soup':
			ego_state_prompt += f"a dish with {ego_object} and needs to deliver soup.  "
		elif ego_object == 'nothing':
			ego_state_prompt += f"{ego_object}. "
		else:
			ego_state_prompt += f"one {ego_object}. "
		
		teammate_state_prompt = f"<Player {1-self.agent_index}> holds "
		if teammate_object == 'soup':
			teammate_state_prompt += f"a dish with {teammate_object}. "
		elif teammate_object == "nothing":
			teammate_state_prompt += f"{teammate_object}. "
		else:
			teammate_state_prompt += f"one {teammate_object}. "

		
		kitchen_state_prompt = "Kitchen states: "
		prompt_dict = {
			"empty": "<Pot {id}> is empty; ",
			"cooking": "<Pot {id}> starts cooking, the soup will be ready after {t} timesteps; ",
			"ready": "<Pot {id}> has already cooked the soup; ",
			"1_items": "<Pot {id}> has 1 onion; ",
			"2_items": "<Pot {id}> has 2 onions; ",
			"3_items": "<Pot {id}> has 3 onions and is full; "
		}

		pot_states_dict = self.mdp.get_pot_states(state)   

		if pkg_resources.get_distribution("overcooked_ai").version == '1.1.0':
			for key in pot_states_dict.keys():
				if key == "cooking":
					for pos in pot_states_dict[key]:
						pot_id = self.pot_id_to_pos.index(pos)
						soup_object = state.get_object(pos)
						kitchen_state_prompt += prompt_dict[key].format(id=pot_id, t=soup_object.cook_time_remaining)
				else:
					for pos in pot_states_dict[key]:
						pot_id = self.pot_id_to_pos.index(pos)
						kitchen_state_prompt += prompt_dict[key].format(id=pot_id) 
		
		elif pkg_resources.get_distribution("overcooked_ai").version == '0.0.1':
			for key in pot_states_dict.keys():
				if key == "empty":
					for pos in pot_states_dict[key]: 
						pot_id = self.pot_id_to_pos.index(pos)
						kitchen_state_prompt += prompt_dict[key].format(id=pot_id)     
				else: # key = 'onion' or 'tomota'
					for soup_key in pot_states_dict[key].keys():
						# soup_key: ready, cooking, partially_full
						for pos in pot_states_dict[key][soup_key]:
							pot_id = self.pot_id_to_pos.index(pos)
							soup_object = state.get_object(pos)
							soup_type, num_items, cook_time = soup_object.state
							if soup_key == "cooking":
								kitchen_state_prompt += prompt_dict[soup_key].format(id=pot_id, t=self.mdp.soup_cooking_time-cook_time)
							elif soup_key == "partially_full":
								pass
							else:
								kitchen_state_prompt += prompt_dict[soup_key].format(id=pot_id)
 

		intersect_counters = get_intersect_counter(
								state.players_pos_and_or[self.agent_index], 
								state.players_pos_and_or[1 - self.agent_index], 
								self.mdp, 
								self.mlam
							)
		counter_states = query_counter_states(self.mdp, state)  

		if self.layout == 'forced_coordination': 
			kitchen_state_prompt += '{} counters can be visited by <Player {}>. Their states are as follows: '.format(len(intersect_counters), self.agent_index)
			count_states = {}  
			for i in intersect_counters:  
				obj_i = 'nothing' 
				if counter_states[i] != ' ': 
					obj_i = counter_states[i]                
				if obj_i in count_states:  
					count_states[obj_i] += 1
				else: 
					count_states[obj_i]  = 1 
			total_obj = ['onion', 'dish']
			for i in count_states:   
				if i == 'nothing': 
					continue 
				kitchen_state_prompt += f'{count_states[i]} counters have {i}. '   
			for i in total_obj: 
				if i not in count_states:        
					kitchen_state_prompt += f'No counters have {i}. ' 

		if self.layout == 'forced_coordination': 
			teammate_state_prompt = ""
		return (self.layout_prompt + time_prompt + ego_state_prompt +
				teammate_state_prompt + kitchen_state_prompt)

	def generate_belief_prompt(self):
		ego_id = self.agent_index
		intention_prompt = f"All <Player {ego_id}> infered intentions about <Player {1-ego_id}>: {self.teammate_intentions_dict}.\n"
		real_behavior_prompt = f"<Player {1-ego_id}> real behaviors: {self.teammate_ml_actions_dict}.\n"
		belief_prompt = intention_prompt + real_behavior_prompt
		return belief_prompt
	
	##################
	'''
	The followings are the Planner part
	'''
	##################

	def action(self, state):

		start_pos_and_or = state.players_pos_and_or[self.agent_index]

		# only use to record the teammate ml_action, 
		# if teammate finish ml_action in t-1, it will record in s_t, 
		# otherwise, s_t will just record None,
		# and we here check this information and store it into proagent
		self.current_timestep = state.timestep
		if state.ml_actions[1-self.agent_index] != None:
			self.teammate_ml_actions_dict[str(self.current_timestep-1)] = state.ml_actions[1-self.agent_index]

		# if current ml action does not exist, generate a new one
		if self.current_ml_action is None:
			self.current_ml_action = self.generate_ml_action(state)

		# if the current ml action is in process, Player{self.agent_index} done, else generate a new one
		if self.current_ml_action_steps > 0:
			current_ml_action_done = self.check_current_ml_action_done(state)
			if current_ml_action_done:
				# generate a new ml action
				self.generate_success_feedback(state)
				self.current_ml_action = self.generate_ml_action(state)

		count = 0
		while not self.validate_current_ml_action(state):

			self.trace = False
			self.generate_failure_feedback(state)
			self.current_ml_action = self.generate_ml_action(state)
			
			count += 1
			if count > 3:
				self.current_ml_action = "wait(1)"
				self.time_to_wait = 1

		
		self.trace = True 
		if "wait" in self.current_ml_action:
			self.current_ml_action_steps += 1
			self.time_to_wait -= 1
			lis_actions = self.mdp.get_valid_actions(state.players[self.agent_index])
			chosen_action =lis_actions[np.random.randint(0,len(lis_actions))]
			if pkg_resources.get_distribution("overcooked_ai").version == '1.1.0':
				self.prev_state = state
				return chosen_action, {}
			elif pkg_resources.get_distribution("overcooked_ai").version == '0.0.1':
				self.prev_state = state
				return chosen_action
		else:
			possible_motion_goals = self.find_motion_goals(state)    
			current_motion_goal, chosen_action = self.choose_motion_goal(
				start_pos_and_or, 
				possible_motion_goals, 
				state
			)
		# if "wait" in self.current_ml_action: 
		# 	print(f'current motion goal for P{self.agent_index} is wait') 
		# else: 
		# 	if current_motion_goal is None: 
		# 		current_motion_goal = 'None' 
		# 	print(f'current motion goal for P{self.agent_index} is {current_motion_goal}') 


		if self.auto_unstuck and chosen_action != Action.INTERACT:
			if (
					self.prev_state is not None
					and state.players
					== self.prev_state.players
			):
				if self.agent_index == 0:
					joint_actions = list(
						itertools.product(Action.ALL_ACTIONS, [Action.STAY])
					)
				elif self.agent_index == 1:
					joint_actions = list(
						itertools.product([Action.STAY], Action.ALL_ACTIONS)
					)
				else:
					raise ValueError("Player index not recognized")

				unblocking_joint_actions = []
				for j_a in joint_actions:
					if j_a != [Action.INTERACT,Action.STAY] and  j_a != [Action.STAY,Action.INTERACT]:
						if pkg_resources.get_distribution("overcooked_ai").version == '1.1.0':
							new_state, _ = self.mlam.mdp.get_state_transition(state, j_a)
						elif pkg_resources.get_distribution("overcooked_ai").version == '0.0.1':
							new_state, _, _ = self.mlam.mdp.get_state_transition(state, j_a)		
						if (
								new_state.players_pos_and_or
								!= self.prev_state.players_pos_and_or
							):
							unblocking_joint_actions.append(j_a)
				unblocking_joint_actions.append([Action.STAY, Action.STAY])
				chosen_action = unblocking_joint_actions[
					np.random.choice(len(unblocking_joint_actions))
				][self.agent_index]

		self.prev_state = state
		if chosen_action is None:
			self.current_ml_action = "wait(1)"
			self.time_to_wait = 1
			chosen_action = Action.STAY
		self.current_ml_action_steps += 1

		# print(f'ml_action = {self.current_ml_action}') 
		# print(f'P{self.agent_index} : {Action.to_char(chosen_action)}')
		if pkg_resources.get_distribution("overcooked_ai").version == '1.1.0':
			return chosen_action, {}
		elif pkg_resources.get_distribution("overcooked_ai").version == '0.0.1':
			return chosen_action

	def parse_ml_action(self, response, agent_index): 

		if agent_index == 0: 
			pattern = r'layer\s*0: (.+)'
		elif agent_index == 1: 
			pattern = r'layer\s*1: (.+)'
		else:
			raise ValueError("Unsupported agent index.")

		match = re.search(pattern, response)
		if match:
			action_string = match.group(1)
		else:
			# raise Exception("please check the query")
			action_string = response
			# print("please check the query")

		# Parse the response to get the medium level action string
		try: 
			ml_action = action_string.split()[0]
		except: 
			print('failed on 528') 
			action_string = 'wait(1)'
			ml_action = action_string
			# ml_action = 'wait(1)' 

		if "place" in action_string:
			ml_action = "place_obj_on_counter"
		elif "pick" in action_string:
			if "onion" in action_string:
				ml_action = "pickup_onion"
			elif "tomato" in action_string:
				ml_action = "pickup_tomato"
			elif "dish" in action_string:
				ml_action = "pickup_dish"
		elif "put" in action_string:
			if "onion" in action_string:
				ml_action = "put_onion_in_pot"
			elif "tomato" in action_string:
				ml_action = "put_tomato_in_pot"
		elif "fill" in action_string:   
			ml_action = "fill_dish_with_soup"
		elif "deliver" in action_string:
			ml_action = "deliver_soup"
		elif "wait" not in action_string:
			ml_action='wait(1)'  
			action_string = ml_action
		if "wait" in action_string:
			
			def parse_wait_string(s):
				# Check if it's just "wait"
				if s == "wait":
					return 1

				# Remove 'wait' and other characters from the string
				s = s.replace('wait', '').replace('(', '').replace(')', '').replace('"', '').replace('.', '') 

				# If it's a number, return it as an integer
				if s.isdigit():
					return int(s)

				# If it's not a number, return a default value or raise an exception
				return 1
			if self.layout == 'forced_coordination': 
				# 这里可以改一下试试 
				self.time_to_wait = max(3, parse_wait_string(action_string))
			else: 
				self.time_to_wait = parse_wait_string(action_string)    
			# print(ml_action) 
			# print(self.time_to_wait) 
			
			ml_action = f"wait({self.time_to_wait})"

		else:
			pass
		
		# aviod to generate two skill, eg, Plan for Player 0: "deliver_soup(), pickup(onion)".
		if "," in ml_action:
			ml_action = ml_action.split(',')[0].strip()

		            
		return ml_action    


	def generate_ml_action(self, state):
		"""
		Selects a medium level action for the current state.
		Motion goals can be thought of instructions of the form:
			[do X] at location [Y]

		In this method, X (e.g. deliver the soup, pick up an onion, etc) is chosen based on
		a simple set of  heuristics based on the current state.

		Effectively, will return a list of all possible locations Y in which the selected
		medium level action X can be performed.
		"""
		if self.prompt_level == "l3-aip" and self.belief_revision:
			belief_prompt = self.generate_belief_prompt()
		else:
			belief_prompt = ''
		state_prompt = belief_prompt + self.generate_state_prompt(state)

		print(f"\n\n### Observation module to GPT\n")   
		print(f"{state_prompt}")

		state_message = {"role": "user", "content": state_prompt}
		self.planner.current_user_message = state_message
		response = self.planner.query(key=self.openai_api_key(), stop='Scene', trace = self.trace)
		
		if 'wait' not in response:
			self.planner.add_msg_to_dialog_history(state_message) 
			self.planner.add_msg_to_dialog_history({"role": "assistant", "content": response})
		
		print(f"\n\n\n### GPT Planner module\n")   
		print("====== GPT Query ======")
		print(response)  


		print("\n===== Parser =====\n")
		## specific for prompt need intention
		if self.prompt_level == "l3-aip":
			generated_intention = self.parse_ml_action(response, 1-self.agent_index)
			self.teammate_intentions_dict[str(self.current_timestep)] = generated_intention
			print(f"Intention for Player {1 - self.agent_index}: {generated_intention}")  
			# if str(self.current_timestep) in self.teammate_intentions_dict:   
			# 	self.teammate_intentions_dict[str(self.current_timestep)].append(generated_intention)
			# else: 
			# 	self.teammate_intentions_dict[str(self.current_timestep)] = [] 
			# 	self.teammate_intentions_dict[str(self.current_timestep)].append(generated_intention) 

		ml_action = self.parse_ml_action(response, self.agent_index)

		if "wait" not in ml_action:
			self.planner.add_msg_to_dialog_history({"role": "assistant", "content": ml_action})
		
		print(f"Player {self.agent_index}: {ml_action}")
		self.current_ml_action_steps = 0
		return ml_action



	##################
	'''
	The followings are the Verificator part
	'''
	##################

	def check_current_ml_action_done(self,state):
		"""
		checks if the current ml action is done
		:return: True or False
		"""
		player = state.players[self.agent_index]
		# pot_states_dict = self.mlam.mdp.get_pot_states(state)
		if "pickup" in self.current_ml_action:
			pattern = r"pickup(?:[(]|_)(\w+)(?:[)]|)" # fit both pickup(onion) and pickup_onion
			obj_str = re.search(pattern, self.current_ml_action).group(1)
			return player.has_object() and player.get_object().name == obj_str
		
		elif "fill" in self.current_ml_action:
			return player.held_object.name == 'soup'
		
		elif "put" in self.current_ml_action or "place" in self.current_ml_action:
			return not player.has_object()
		
		elif "deliver" in self.current_ml_action:
			return not player.has_object()
		
		elif "wait" in self.current_ml_action:
			return self.time_to_wait == 0

	def validate_current_ml_action(self, state):
		"""
		make sure the current_ml_action exists and is valid
		"""
		if self.current_ml_action is None:
			return False

		pot_states_dict = self.mdp.get_pot_states(state)
		player = state.players[self.agent_index]
		if pkg_resources.get_distribution("overcooked_ai").version == '1.1.0':
			soup_cooking = len(pot_states_dict['cooking']) > 0
			soup_ready = len(pot_states_dict['ready']) > 0
			pot_not_full = pot_states_dict["empty"] + self.mdp.get_partially_full_pots(pot_states_dict)
			cookable_pots = self.mdp.get_full_but_not_cooking_pots(pot_states_dict)
		elif pkg_resources.get_distribution("overcooked_ai").version == '0.0.1':
			soup_cooking = len(pot_states_dict['onion']['cooking'])+len(pot_states_dict['tomato']['cooking']) > 0
			soup_ready = len(pot_states_dict['onion']['ready'])+len(pot_states_dict['tomato']['ready']) > 0
			pot_not_full = pot_states_dict["empty"] + pot_states_dict["onion"]['partially_full'] + pot_states_dict["tomato"]['partially_full']
			cookable_pots = pot_states_dict["onion"]['{}_items'.format(self.mdp.num_items_for_soup)] + pot_states_dict["tomato"]['{}_items'.format(self.mdp.num_items_for_soup)] # pot has max onions/tomotos

		
		has_onion = False
		has_tomato = False
		has_dish = False
		has_soup = False
		has_object = player.has_object()
		if has_object:
			has_onion = player.get_object().name == 'onion'
			has_tomato = player.get_object().name == 'tomato'
			has_dish = player.get_object().name == 'dish'
			has_soup = player.get_object().name == 'soup'
		empty_counter = self.mdp.get_empty_counter_locations(state)


		if self.current_ml_action in ["pickup(onion)", "pickup_onion"]:   

			flag2 = len(self.find_motion_goals(state)) == 0 
			if flag2: 
				return False 
			return not has_object and len(self.mdp.get_onion_dispenser_locations()) > 0
		if self.current_ml_action in ["pickup(tomato)", "pickup_tomato"]:
			return not has_object and len(self.mdp.get_tomato_dispenser_locations()) > 0
		elif self.current_ml_action in ["pickup(dish)", "pickup_dish"]:
			flag2 = len(self.find_motion_goals(state)) == 0 
			if flag2: 
				return False 
			return not has_object and len(self.mdp.get_dish_dispenser_locations()) > 0
		elif "put_onion_in_pot" in self.current_ml_action:
			return has_onion and len(pot_not_full) > 0
		elif "put_tomato_in_pot" in self.current_ml_action:
			return has_tomato and len(pot_not_full) > 0
		elif "place_obj_on_counter" in self.current_ml_action:
			return has_object and len(empty_counter) > 0
		elif "fill_dish_with_soup" in self.current_ml_action:
			return has_dish and (soup_ready or soup_cooking)
		elif "deliver_soup" in self.current_ml_action:
			return has_soup
		elif "wait" in self.current_ml_action:
			return 0 < int(self.current_ml_action.split('(')[1][:-1]) <= 20


	def generate_success_feedback(self, state):
		success_feedback = f"### Controller Validation\nPlayer {self.agent_index} succeeded at {self.current_ml_action}. \n"
		print(success_feedback)  
		if 'wait' not in success_feedback:
			self.planner.add_msg_to_dialog_history({"role": "user", "content": f'Player {self.agent_index} succeeded at {self.current_ml_action}.'})
		
	def generate_failure_feedback(self, state):
		failure_feedback = self.generate_state_prompt(state)
		failure_feedback += f" Player {self.agent_index} failed at {self.current_ml_action}."
		failure_feedback += f" Why did Player {self.agent_index} fail ?"     
		print(f"\n~~~~~~~~ Explainer~~~~~~~~\n{failure_feedback}")  
		failure_message = {"role": "user", "content": failure_feedback}
		self.explainer.current_user_message = failure_message
		failure_explanation = self.explainer.query(self.openai_api_key())
		print(failure_explanation)  
		if "wait" not in failure_explanation or self.layout == 'forced_coodination':
			self.explainer.add_msg_to_dialog_history({"role": "user", "content": failure_feedback})
			self.explainer.add_msg_to_dialog_history({"role": "assistant", "content": failure_explanation})
		self.planner.add_msg_to_dialog_history({"role": "user", "content": failure_explanation}) 

	##################
	'''
	The followings are the Controller part almost inherited from GreedyHumanModel class
	'''
	##################	
		
	def find_shared_counters(self, state, mlam):  
		counter_dicts = query_counter_states(self.mdp, state) 

		counter_list  = get_intersect_counter(state.players_pos_and_or[self.agent_index],
						state.players_pos_and_or[1 - self.agent_index], 
						self.mdp, 
						self.mlam
					)    

		print('counter_list = {}'.format(counter_list))  
		lis = [] 
		for i in counter_list:  
			if counter_dicts[i] == ' ':  
				lis.append(i)       
		available_plans = mlam._get_ml_actions_for_positions(lis)
		return available_plans          

	def find_motion_goals(self, state):
		"""
		Generates the motion goals for the given medium level action.
		:param state:
		:return:
		"""
		am = self.mlam
		motion_goals = []
		player = state.players[self.agent_index]
		pot_states_dict = self.mdp.get_pot_states(state)
		counter_objects = self.mdp.get_counter_objects_dict(
			state, list(self.mdp.terrain_pos_dict["X"])
		)
		if self.current_ml_action in ["pickup(onion)", "pickup_onion"]:
			motion_goals = am.pickup_onion_actions_new(state, counter_objects, state.players_pos_and_or, self.agent_index) 


		elif self.current_ml_action in ["pickup(tomato)", "pickup_tomato"]:
			motion_goals = am.pickup_tomato_actions(state, counter_objects)
		elif self.current_ml_action in ["pickup(dish)", "pickup_dish"]:
			motion_goals = am.pickup_dish_actions_new(state, counter_objects , state.players_pos_and_or, self.agent_index)
		elif "put_onion_in_pot" in self.current_ml_action:
			motion_goals = am.put_onion_in_pot_actions(pot_states_dict)
		elif "put_tomato_in_pot" in self.current_ml_action:
			motion_goals = am.put_tomato_in_pot_actions(pot_states_dict)
		elif "place_obj_on_counter" in self.current_ml_action:  
			motion_goals = self.find_shared_counters(state, self.mlam)     
			if len(motion_goals) == 0: 
				motion_goals = am.place_obj_on_counter_actions(state)

		elif "start_cooking" in self.current_ml_action:
			if pkg_resources.get_distribution("overcooked_ai").version == '1.1.0':
				next_order = list(state.all_orders)[0]
				soups_ready_to_cook_key = "{}_items".format(len(next_order.ingredients))
				soups_ready_to_cook = pot_states_dict[soups_ready_to_cook_key]
			elif pkg_resources.get_distribution("overcooked_ai").version == '0.0.1':
				soups_ready_to_cook = pot_states_dict["onion"]['{}_items'.format(self.mdp.num_items_for_soup)] + pot_states_dict["tomato"]['{}_items'.format(self.mdp.num_items_for_soup)]
			only_pot_states_ready_to_cook = defaultdict(list)
			only_pot_states_ready_to_cook[soups_ready_to_cook_key] = soups_ready_to_cook
			motion_goals = am.start_cooking_actions(only_pot_states_ready_to_cook)
		elif "fill_dish_with_soup" in self.current_ml_action:
			motion_goals = am.pickup_soup_with_dish_actions(pot_states_dict, only_nearly_ready=True)
		elif "deliver_soup" in self.current_ml_action:
			motion_goals = am.deliver_soup_actions()
		elif "wait" in self.current_ml_action:
			motion_goals = am.wait_actions(player)
		else:
			raise ValueError("Invalid action: {}".format(self.current_ml_action))

		motion_goals = [
			mg
			for mg in motion_goals
			if self.mlam.motion_planner.is_valid_motion_start_goal_pair(
				player.pos_and_or, mg
			)
		]

		return motion_goals

	def choose_motion_goal(self, start_pos_and_or, motion_goals, state = None):
		"""
		For each motion goal, consider the optimal motion plan that reaches the desired location.
		Based on the plan's cost, the method chooses a motion goal (either boltzmann rationally
		or rationally), and returns the plan and the corresponding first action on that plan.
		"""

		if self.controller_mode == 'new':
			(
				chosen_goal,
				chosen_goal_action,
			) = self.get_lowest_cost_action_and_goal_new(
				start_pos_and_or, motion_goals, state
			)
		else: 
			(
				chosen_goal,
				chosen_goal_action,
			) = self.get_lowest_cost_action_and_goal(
				start_pos_and_or, motion_goals
			)
		return chosen_goal, chosen_goal_action
	
	def get_lowest_cost_action_and_goal(self, start_pos_and_or, motion_goals):
		"""
		Chooses motion goal that has the lowest cost action plan.
		Returns the motion goal itself and the first action on the plan.
		"""
		min_cost = np.Inf
		best_action, best_goal = None, None
		for goal in motion_goals:
			action_plan, _, plan_cost = self.mlam.motion_planner.get_plan(
				start_pos_and_or, goal
			)
			if plan_cost < min_cost:
				best_action = action_plan[0]
				min_cost = plan_cost
				best_goal = goal
		return best_goal, best_action

	
	def get_lowest_cost_action_and_goal_new(self, start_pos_and_or, motion_goals, state): 
		"""
		Chooses motion goal that has the lowest cost action plan.
		Returns the motion goal itself and the first action on the plan.
		"""   
		min_cost = np.Inf
		best_action, best_goal = None, None
		for goal in motion_goals:   
			action_plan, plan_cost = self.real_time_planner(
				start_pos_and_or, goal, state
			)     
			if plan_cost < min_cost:
				best_action = action_plan
				min_cost = plan_cost
				best_goal = goal     
		if best_action is None: 
			# print('\n\n\nBlocking Happend, executing default path\n\n\n')
			# print('current position = {}'.format(start_pos_and_or)) 
			# print('goal position = {}'.format(motion_goals))        
			if np.random.rand() < 0.5:  
				return None, Action.STAY
			else: 
				return self.get_lowest_cost_action_and_goal(start_pos_and_or, motion_goals)
		return best_goal, best_action

	def real_time_planner(self, start_pos_and_or, goal, state):   
		terrain_matrix = {
			'matrix': copy.deepcopy(self.mlam.mdp.terrain_mtx), 
			'height': len(self.mlam.mdp.terrain_mtx), 
			'width' : len(self.mlam.mdp.terrain_mtx[0]) 
		}
		other_pos_and_or = state.players_pos_and_or[1 - self.agent_index]
		action_plan, plan_cost = find_path(start_pos_and_or, other_pos_and_or, goal, terrain_matrix) 

		return action_plan, plan_cost
	

class ProPlanningAgent(ProAgent):
	def __init__(self, model="gpt-3.5-turbo"):
		super().__init__(model=model)

# ============================================================
# ITDP-Agent: Intent-aware Task-Driven Priority Agent
# 意图感知的任务驱动优先级智能体
# ============================================================
#
# 核心创新：结合任务驱动和意图感知
#
# 与其他方法的对比：
# ┌──────────────┬─────────────────┬─────────────────────────┐
# │ 方法         │ 决策依据        │ 问题                    │
# ├──────────────┼─────────────────┼─────────────────────────┤
# │ ProAgent     │ 推断队友意图    │ 被动配合，依赖准确推断  │
# │ RACE         │ 学习队友偏好    │ 适应滞后，偏好可能变    │
# │ TDP          │ 识别任务瓶颈    │ 可能和队友抢同一任务    │
# │ ITDP (本文)  │ 瓶颈 + 意图     │ 主动且协调，避免冲突    │
# └──────────────┴─────────────────┴─────────────────────────┘
#
# 决策流程：
# 1. 分析任务 → 瓶颈列表 [B1, B2, B3...]
# 2. 观察队友 → 推断正在处理的瓶颈
# 3. 选择 → 队友没在处理的最高优先级瓶颈
# 4. 执行 → 解决选中的瓶颈
# ============================================================

from .itdp_module import ITDPCoordinator, visualize_coordination


class ITDPAgent(ProMediumLevelAgent):
    """
    ITDP-Agent: Intent-aware Task-Driven Priority Agent
    意图感知的任务驱动优先级智能体
    
    核心理念：
    "知道任务需要什么 + 知道队友在做什么 = 做队友没做的最重要的事"
    """
    
    def __init__(
            self,
            mlam,
            layout,
            model='Qwen/Qwen2.5-7B-Instruct',
            prompt_level='l2-ap',
            belief_revision=False,
            retrival_method="recent_k",
            K=1,
            auto_unstuck=True,
            controller_mode='new',
            debug_mode='N',
            agent_index=None,
            outdir=None
    ):
        super().__init__(
            mlam=mlam,
            layout=layout,
            model=model,
            prompt_level=prompt_level,
            belief_revision=belief_revision,
            retrival_method=retrival_method,
            K=K,
            auto_unstuck=auto_unstuck,
            controller_mode=controller_mode,
            debug_mode=debug_mode,
            agent_index=agent_index,
            outdir=outdir
        )
        
        # 初始化ITDP协调器
        self.itdp = ITDPCoordinator()
        
        print(f"\n{'='*60}")
        print(f"[ITDP-Agent] Intent-aware Task-Driven Priority Agent")
        print(f"  Innovation: Task bottleneck analysis + Teammate intent prediction")
        print(f"  Principle: Do what's needed AND what teammate isn't doing")
        print(f"{'='*60}\n")
    
    def reset(self):
        super().reset()
        self.itdp.reset()
        print("[ITDP-Agent] Reset - ready for intent-aware coordination")
    
    def _get_kitchen_state(self, state):
        """获取厨房状态"""
        pot_states_dict = self.mdp.get_pot_states(state)
        
        pot_items = 0
        soup_ready = False
        soup_cooking = False
        
        if pkg_resources.get_distribution("overcooked_ai").version == '1.1.0':
            soup_ready = len(pot_states_dict.get('ready', [])) > 0
            soup_cooking = len(pot_states_dict.get('cooking', [])) > 0
            
            # 计算pot_items - 需要检查多个来源
            for key in ['1_items', '2_items', '3_items']:
                if key in pot_states_dict and pot_states_dict[key]:
                    pot_items = max(pot_items, int(key[0]))
            
            # 如果正在煮或已经好了，pot_items至少是3
            if soup_cooking or soup_ready:
                pot_items = 3
                
        else:
            # 0.0.1版本 - 需要正确解析pot_states_dict
            # pot_states_dict结构: {'onion': {'empty': [], 'cooking': [...], 'ready': [...], 1: [...], 2: [...], 3: [...]}}
            for soup_type in ['onion', 'tomato']:
                if soup_type in pot_states_dict:
                    soup_data = pot_states_dict[soup_type]
                    soup_ready = soup_ready or len(soup_data.get('ready', [])) > 0
                    soup_cooking = soup_cooking or len(soup_data.get('cooking', [])) > 0
                    
                    # 检查各数量状态
                    for num in [3, 2, 1]:
                        if num in soup_data and len(soup_data[num]) > 0:
                            pot_items = max(pot_items, num)
                            break
            
            # 如果正在煮或已经好了，pot_items至少是3
            if soup_cooking or soup_ready:
                pot_items = 3
            
            # Backup: 直接从pot对象中获取数量
            if pot_items == 0:
                pot_locations = self.mdp.get_pot_locations()
                for pos in pot_locations:
                    if state.has_object(pos):
                        obj = state.get_object(pos)
                        if hasattr(obj, 'ingredients'):
                            pot_items = max(pot_items, len(obj.ingredients))
                        elif hasattr(obj, '_ingredients'):
                            pot_items = max(pot_items, len(obj._ingredients))
        
        return {
            'pot_items': pot_items,
            'soup_ready': soup_ready,
            'soup_cooking': soup_cooking
        }
    
    def _get_player_state(self, player):
        """获取玩家状态"""
        held_object = None
        if player.has_object():
            held_object = player.get_object().name
        return {
            'held_object': held_object,
            'position': player.position,
            'orientation': player.orientation
        }
    
    def _check_reachability(self, state):
        """
        实时可达性检测 v12
        
        关键修复：正确区分"结构不可达"和"被队友阻挡"
        - 结构不可达：即使没有队友也到不了（地图有墙）
        - 被队友阻挡：没有队友能到，有队友到不了
        
        Returns:
            reachability dict
        """
        try:
            player = state.players[self.agent_index]
            teammate = state.players[1 - self.agent_index]
            my_pos = player.position
            teammate_pos = teammate.position
            
            # 获取各种位置
            pot_locations = self.mdp.get_pot_locations()
            serve_locations = self.mdp.get_serving_locations()
            onion_dispenser_locations = self.mdp.get_onion_dispenser_locations()
            dish_dispenser_locations = self.mdp.get_dish_dispenser_locations()
            
            # 获取柜台上的物品位置
            counter_objects = self.mdp.get_counter_objects_dict(
                state, list(self.mdp.terrain_pos_dict["X"])
            )
            onions_on_counter = list(counter_objects.get('onion', []))
            dishes_on_counter = list(counter_objects.get('dish', []))
            
            # === 检测pot可达性 ===
            # 1. 忽略队友时能否到达（检测结构可达性）
            can_reach_pot_structural = self._can_reach_any_location(my_pos, pot_locations, None)
            # 2. 考虑队友时能否到达（检测当前可达性）
            can_reach_pot_now = self._can_reach_any_location(my_pos, pot_locations, teammate_pos)
            
            if can_reach_pot_now:
                can_reach_pot = True
                pot_blocked = False
            elif can_reach_pot_structural:
                # 结构上能到，但现在到不了 = 被队友阻挡
                can_reach_pot = True
                pot_blocked = True
            else:
                # 结构上也到不了 = 永久不可达
                can_reach_pot = False
                pot_blocked = False
            
            # === 检测serve可达性 ===
            can_reach_serve_structural = self._can_reach_any_location(my_pos, serve_locations, None)
            can_reach_serve_now = self._can_reach_any_location(my_pos, serve_locations, teammate_pos)
            
            if can_reach_serve_now:
                can_reach_serve = True
                serve_blocked = False
            elif can_reach_serve_structural:
                can_reach_serve = True
                serve_blocked = True
            else:
                can_reach_serve = False
                serve_blocked = False
            
            # === 检测onion可达性 ===
            onion_sources = list(onion_dispenser_locations) + onions_on_counter
            can_reach_onion = self._can_reach_any_location(my_pos, onion_sources, teammate_pos) if onion_sources else False
            
            # === 检测dish可达性 ===
            dish_sources = list(dish_dispenser_locations) + dishes_on_counter
            can_reach_dish = self._can_reach_any_location(my_pos, dish_sources, teammate_pos) if dish_sources else False
            
            result = {
                'pot': can_reach_pot,
                'serve': can_reach_serve,
                'onion': can_reach_onion,
                'dish': can_reach_dish,
                'pot_blocked': pot_blocked,
                'serve_blocked': serve_blocked
            }
            
            return result
            
        except Exception as e:
            print(f"[REACHABILITY ERROR] {e}")
            import traceback
            traceback.print_exc()
            return {
                'pot': True, 'serve': True, 'onion': True, 'dish': True,
                'pot_blocked': False, 'serve_blocked': False
            }
    
    def _can_reach_any_location(self, start_pos, target_locations, teammate_pos):
        """
        检测是否能到达任意一个目标位置
        使用BFS图搜索
        
        Args:
            start_pos: 起始位置
            target_locations: 目标位置列表
            teammate_pos: 队友位置（None表示忽略队友，用于检测结构可达性）
        """
        if not target_locations:
            return False
        
        try:
            # 获取所有可站立的位置
            graph = self.mlam.motion_planner.mdp.get_valid_player_positions()
            
            # BFS检测可达性
            from collections import deque
            visited = set()
            queue = deque([start_pos])
            visited.add(start_pos)
            
            # 目标位置的相邻格子（因为我们需要站在旁边操作）
            target_adjacent = set()
            for loc in target_locations:
                for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                    adj = (loc[0] + dx, loc[1] + dy)
                    # 如果teammate_pos是None，不排除任何位置
                    if adj in graph:
                        if teammate_pos is None or adj != teammate_pos:
                            target_adjacent.add(adj)
            
            # 如果没有可到达的目标相邻位置
            if not target_adjacent:
                return False
            
            while queue:
                current = queue.popleft()
                
                # 检查是否到达目标相邻位置
                if current in target_adjacent:
                    return True
                
                # 扩展邻居
                for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                    next_pos = (current[0] + dx, current[1] + dy)
                    # 如果teammate_pos是None，不排除任何位置
                    if next_pos in graph and next_pos not in visited:
                        if teammate_pos is None or next_pos != teammate_pos:
                            visited.add(next_pos)
                            queue.append(next_pos)
            
            return False
            
        except Exception as e:
            # 如果BFS失败，退回到Motion Planner方法
            print(f"[BFS ERROR] {e}, falling back to motion planner")
            return self._can_reach_via_motion_planner(start_pos, target_locations)
    
    def _can_reach_via_motion_planner(self, start_pos, target_locations):
        """使用Motion Planner检测可达性（备用方法）"""
        try:
            start_pos_and_or = (start_pos, (0, 1))  # 默认朝向
            
            for target in target_locations:
                # 尝试所有朝向
                for orientation in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                    goal = (target, orientation)
                    try:
                        if self.mlam.motion_planner.is_valid_motion_start_goal_pair(start_pos_and_or, goal):
                            return True
                    except:
                        continue
            return False
        except:
            return True  # 出错假设可达
    
    def _is_blocking_any(self, blocker_pos, target_locations):
        """检查是否阻挡任何目标位置"""
        if blocker_pos is None:
            return False
        for loc in target_locations:
            # 检查是否站在目标位置或其相邻位置
            dist = abs(blocker_pos[0] - loc[0]) + abs(blocker_pos[1] - loc[1])
            if dist <= 1:
                return True
        return False
    
    def action(self, state):
        """重写action，添加队友观察"""
        # 观察队友动作
        teammate_action = state.ml_actions[1 - self.agent_index]
        teammate_player = state.players[1 - self.agent_index]
        
        if teammate_action is not None:
            self.itdp.update_teammate_observation(
                teammate_action, 
                teammate_player.position
            )
        
        return super().action(state)
    
#     def generate_ml_action(self, state):
#         """ITDP决策生成"""
        
#         # 获取各方状态
#         kitchen_state = self._get_kitchen_state(state)
#         my_state = self._get_player_state(state.players[self.agent_index])
#         teammate_state = self._get_player_state(state.players[1 - self.agent_index])
        
#         # 检测可达性（通用方法，适用于任何地图）
#         # 返回字典，包含结构可达性和是否被队友暂时挡住
#         reachability = self._check_reachability(state)
        
#         # ITDP决策（传入agent_index和可达性信息）
#         itdp_action, itdp_reason, debug_info = self.itdp.decide(
#             kitchen_state, my_state, teammate_state, self.agent_index,
#             reachability=reachability
#         )
        
#         # 可视化
#         viz = visualize_coordination(kitchen_state, my_state, teammate_state, itdp_action)
        
#         my_held = my_state['held_object'] or 'nothing'
#         tm_held = teammate_state['held_object'] or 'nothing'
        
#         print(f"\n### ITDP-Agent Analysis")
#         print(f"My holding: {my_held}, Teammate holding: {tm_held}")
#         print(f"Kitchen: pot={kitchen_state['pot_items']}/3, ready={kitchen_state['soup_ready']}, cooking={kitchen_state['soup_cooking']}")
        
#         # ========================================================
#         # 打印队友意图预测详细信息
#         # ========================================================
#         teammate_handling, intent_reason = self.itdp.intent_predictor.predict_intent(
#             teammate_state, kitchen_state, self.agent_index
#         )
#         confidence = self.itdp.intent_predictor.get_confidence()
        
#         print(f"\n--- Teammate Intent Prediction ---")
#         print(f"Teammate likely handling: {[b.value for b in teammate_handling] if teammate_handling else 'unknown'}")
#         print(f"Reasoning: {intent_reason}")
#         print(f"Confidence: {confidence:.1%}")
        
#         # 如果使用贝叶斯推断，显示信念分布
#         if self.itdp.intent_predictor.bayesian_belief:
#             belief = self.itdp.intent_predictor.bayesian_belief.belief
#             # 只显示概率>5%的瓶颈
#             significant_beliefs = {b.value: f"{p:.1%}" for b, p in belief.items() if p > 0.05}
#             if significant_beliefs:
#                 print(f"Belief distribution: {significant_beliefs}")
#         print(f"----------------------------------")
        
#         print(f"\nTeammate handling: {debug_info.get('teammate_handling', [])}")
        
#         # 显示可达性状态
#         pot_ok = "✓" if reachability.get('pot', True) else "✗"
#         serve_ok = "✓" if reachability.get('serve', True) else "✗"
#         onion_ok = "✓" if reachability.get('onion', True) else "✗"
#         dish_ok = "✓" if reachability.get('dish', True) else "✗"
#         pot_blk = "⏳" if reachability.get('pot_blocked', False) else ""
#         serve_blk = "⏳" if reachability.get('serve_blocked', False) else ""
        
#         print(f"Reachability: pot={pot_ok}{pot_blk} serve={serve_ok}{serve_blk} onion={onion_ok} dish={dish_ok}")
        
#         # 如果有不可达的情况，显示详细信息
#         if not all([reachability.get('pot', True), reachability.get('serve', True), 
#                    reachability.get('onion', True), reachability.get('dish', True)]):
#             # 获取位置信息用于调试
#             try:
#                 pot_locs = self.mdp.get_pot_locations()
#                 serve_locs = self.mdp.get_serving_locations()
#                 onion_locs = self.mdp.get_onion_dispenser_locations()
#                 dish_locs = self.mdp.get_dish_dispenser_locations()
#                 my_pos = state.players[self.agent_index].position
#                 tm_pos = state.players[1 - self.agent_index].position
#                 print(f"  My pos: {my_pos}, Teammate pos: {tm_pos}")
#                 print(f"  Pot locs: {pot_locs}, Serve locs: {serve_locs}")
#                 print(f"  Onion dispensers: {onion_locs}, Dish dispensers: {dish_locs}")
#             except:
#                 pass
        
#         print(f"ITDP Decision: {itdp_action}")
#         print(viz)
        
#         # ========================================================
#         # 修改：始终调用LLM做决策，但把ITDP建议传给LLM参考
#         # ========================================================
#         is_forced = debug_info.get("type") == "forced"
#         if is_forced:
#             print(f"\n[ITDP FORCED RECOMMENDATION] {itdp_reason}")
#             print(f"ITDP strongly suggests: {itdp_action}")
#             print(f"Will ask LLM to make final decision with ITDP guidance...")
        
#         # 生成ITDP提示（所有情况都调用LLM）
#         itdp_prompt = self.itdp.get_prompt(kitchen_state, my_state, teammate_state)
        
#         # 生成基础状态描述
#         if self.prompt_level == "l3-aip" and self.belief_revision:
#             belief_prompt = self.generate_belief_prompt()
#         else:
#             belief_prompt = ''
#         state_prompt = belief_prompt + self.generate_state_prompt(state)
        
#         # 根据手持物品生成约束提示
#         hand_constraint = ""
#         if my_state['held_object']:
#             hand_constraint = f"\n⚠️ WARNING: You are holding [{my_held}], you CANNOT pickup anything else!"
        
#         # 根据是否是强制动作，生成额外提示
#         forced_hint = ""
#         if is_forced:
#             forced_hint = f"""
# ╔══════════════════════════════════════════════════════════════╗
# ║  ⚠️  ITDP FORCED RECOMMENDATION (STRONG SUGGESTION)  ⚠️      ║
# ╠══════════════════════════════════════════════════════════════╣
# ║  ITDP Analysis: {itdp_reason:<44} ║
# ║  Strongly Recommended Action: {itdp_action:<30} ║
# ║                                                              ║
# ║  This is a FORCED situation - the recommended action is      ║
# ║  almost certainly the best choice. If you choose differently,║
# ║  please provide a very good reason.                          ║
# ╚══════════════════════════════════════════════════════════════╝
# """
        
#         enhanced_prompt = f"""{state_prompt}

# {itdp_prompt}

# {viz}
# {forced_hint}
# ITDP RECOMMENDATION: {itdp_action}
# REASON: {itdp_reason}
# {hand_constraint}

# DECISION RULES:
# 1. If holding SOUP → deliver_soup (no exception!)
# 2. If holding DISH + soup ready → fill_dish_with_soup
# 3. If holding DISH + soup cooking → wait for soup
# 4. If holding DISH + no soup → place_obj_on_counter
# 5. If holding INGREDIENT + soup ready → place_obj_on_counter (CANNOT put in pot!)
# 6. If holding INGREDIENT + pot not full → put_onion_in_pot
# 7. If hands empty → pick the bottleneck that TEAMMATE IS NOT handling

# Current Status:
# - I (Player {self.agent_index}) am holding: {my_held}
# - Teammate (Player {1-self.agent_index}) is holding: {tm_held}

# Please respond in the following format (ALL THREE parts are required):

# Teammate Intent: [What do you think your teammate is trying to do? What task/bottleneck are they handling?]

# Cooperation Request: [What do you hope your teammate will do next to cooperate with you? What would be helpful?]

# My Action: [Your decision - choose ONE action from: pickup_onion, pickup_dish, put_onion_in_pot, fill_dish_with_soup, deliver_soup, place_obj_on_counter, wait(N)]"""
        
#         state_message = {"role": "user", "content": enhanced_prompt}
#         self.planner.current_user_message = state_message
#         response = self.planner.query(key=self.openai_api_key(), stop='Scene', trace=self.trace)
        
#         if 'wait' not in response:
#             self.planner.add_msg_to_dialog_history(state_message)
#             self.planner.add_msg_to_dialog_history({"role": "assistant", "content": response})
        
#         print(f"\n### LLM Response\n{response}")
        
#         # 解析三段式回复
#         print("\n===== Parsing LLM Response =====\n")
#         parsed = self._parse_itdp_response(response)
        
#         print(f"[Teammate Intent] {parsed['teammate_intent']}")
#         print(f"[Cooperation Request] {parsed['cooperation_request']}")
#         print(f"[My Action] {parsed['my_action']}")
        
#         # 保存队友意图推断（用于后续分析）
#         self.teammate_intentions_dict[str(self.current_timestep)] = parsed['teammate_intent']
        
#         # 从My Action部分提取最终动作
#         ml_action = self._extract_action_from_text(parsed['my_action'])
        
#         # === ITDP验证与修正（更严格）===
#         ml_action = self._itdp_validate(ml_action, kitchen_state, my_state, itdp_action)
        
#         if "wait" not in ml_action:
#             self.planner.add_msg_to_dialog_history({"role": "assistant", "content": ml_action})
        
#         print(f"Final Action for Player {self.agent_index}: {ml_action}")
#         self.current_ml_action_steps = 0
        
#         # 处理wait动作，设置time_to_wait
#         if "wait" in ml_action:
#             import re
#             match = re.search(r'wait\((\d+)\)', ml_action)
#             if match:
#                 self.time_to_wait = int(match.group(1))
#             else:
#                 self.time_to_wait = 3  # 默认等待3步
#             print(f"Setting time_to_wait = {self.time_to_wait}")
        
#         return ml_action
    def generate_ml_action(self, state):
        """ITDP-only决策生成（不使用LLM）"""
        
        # 获取各方状态
        kitchen_state = self._get_kitchen_state(state)
        my_state = self._get_player_state(state.players[self.agent_index])
        teammate_state = self._get_player_state(state.players[1 - self.agent_index])
        
        # 检测可达性
        reachability = self._check_reachability(state)
        
        # ITDP决策（100%使用ITDP模块，不调用LLM）
        itdp_action, itdp_reason, debug_info = self.itdp.decide(
            kitchen_state, my_state, teammate_state, self.agent_index,
            reachability=reachability
        )
        
        # 打印决策信息
        print(f"\n{'='*60}")
        print(f"[ITDP-ONLY DECISION] (No LLM)")
        print(f"{'='*60}")
        print(f"Agent {self.agent_index}: holding {my_state.get('held_object') or 'nothing'}")
        print(f"Teammate: holding {teammate_state.get('held_object') or 'nothing'}")
        print(f"Kitchen: pot={kitchen_state['pot_items']}/3, ready={kitchen_state['soup_ready']}, cooking={kitchen_state['soup_cooking']}")
        print(f"ITDP Decision: {itdp_action}")
        print(f"Reason: {itdp_reason}")
        print(f"{'='*60}\n")
        
        # 直接返回ITDP决策，不调用LLM
        ml_action = itdp_action
        
        # 处理wait动作，设置time_to_wait
        if "wait" in ml_action:
            import re
            match = re.search(r'wait\((\d+)\)', ml_action)
            if match:
                self.time_to_wait = int(match.group(1))
            else:
                self.time_to_wait = 3  # 默认等待3步
        
        self.current_ml_action_steps = 0
        return ml_action

    def _parse_itdp_response(self, response):
        """
        解析LLM的三段式回复
        
        格式：
        Teammate Intent: [...]
        Cooperation Request: [...]
        My Action: [...]
        """
        result = {
            'teammate_intent': '',
            'cooperation_request': '',
            'my_action': ''
        }
        
        # 尝试匹配各部分
        import re
        
        # 匹配 Teammate Intent
        teammate_match = re.search(r'Teammate\s*Intent[:\s]*(.+?)(?=Cooperation|My Action|$)', response, re.IGNORECASE | re.DOTALL)
        if teammate_match:
            result['teammate_intent'] = teammate_match.group(1).strip()
        
        # 匹配 Cooperation Request
        coop_match = re.search(r'Cooperation\s*Request[:\s]*(.+?)(?=My Action|$)', response, re.IGNORECASE | re.DOTALL)
        if coop_match:
            result['cooperation_request'] = coop_match.group(1).strip()
        
        # 匹配 My Action (最重要，这是最终决策)
        action_match = re.search(r'My\s*Action[:\s]*(.+?)$', response, re.IGNORECASE | re.DOTALL)
        if action_match:
            result['my_action'] = action_match.group(1).strip()
        else:
            # 如果没有匹配到格式，使用整个response
            result['my_action'] = response
        
        return result
    
    def _extract_action_from_text(self, text):
        """
        从文本中提取动作名称
        """
        text_lower = text.lower()
        
        # 按优先级检查各种动作
        if "deliver" in text_lower and "soup" in text_lower:
            return "deliver_soup"
        elif "fill" in text_lower and ("dish" in text_lower or "soup" in text_lower):
            return "fill_dish_with_soup"
        elif "put" in text_lower and "onion" in text_lower and "pot" in text_lower:
            return "put_onion_in_pot"
        elif "put" in text_lower and "tomato" in text_lower and "pot" in text_lower:
            return "put_tomato_in_pot"
        elif "pickup" in text_lower or "pick up" in text_lower or "pick_up" in text_lower:
            if "onion" in text_lower:
                return "pickup_onion"
            elif "tomato" in text_lower:
                return "pickup_tomato"
            elif "dish" in text_lower:
                return "pickup_dish"
        elif "place" in text_lower and "counter" in text_lower:
            return "place_obj_on_counter"
        elif "wait" in text_lower:
            # 提取等待时间
            import re
            match = re.search(r'wait\s*\(?(\d+)\)?', text_lower)
            if match:
                return f"wait({match.group(1)})"
            else:
                return "wait(3)"
        
        # 简化匹配（直接包含动作名）
        action_names = [
            'deliver_soup', 'fill_dish_with_soup', 
            'put_onion_in_pot', 'put_tomato_in_pot',
            'pickup_onion', 'pickup_tomato', 'pickup_dish',
            'place_obj_on_counter'
        ]
        for action in action_names:
            if action in text_lower:
                return action
        
        # 默认返回wait
        print(f"[WARNING] Could not parse action from: {text}")
        return "wait(1)"
    
    def _itdp_validate(self, ml_action, kitchen_state, my_state, itdp_recommended):
        """
        ITDP动作验证 - 修改版
        只打印警告信息，不覆盖LLM的决策
        让LLM完全负责最终决策
        """
        held = my_state.get('held_object')
        soup_ready = kitchen_state.get('soup_ready', False)
        soup_cooking = kitchen_state.get('soup_cooking', False)
        pot_items = kitchen_state.get('pot_items', 0)

        warning_msg = None

        # === 检查潜在问题（只警告，不覆盖）===

        # 规则0：手里有东西不能pickup
        if held is not None and ml_action.startswith('pickup'):
            warning_msg = f"[ITDP-WARNING] LLM chose '{ml_action}' but holding {held}! ITDP suggests: {itdp_recommended}"

        # 拿着soup应该deliver
        if held == 'soup' and ml_action != 'deliver_soup':
            warning_msg = f"[ITDP-WARNING] LLM chose '{ml_action}' but holding SOUP! ITDP suggests: deliver_soup"

        # 拿着dish的情况
        if held == 'dish':
            if soup_ready and ml_action != 'fill_dish_with_soup':
                warning_msg = f"[ITDP-WARNING] LLM chose '{ml_action}' but holding DISH + soup ready! ITDP suggests: fill_dish_with_soup"
            elif soup_cooking and ml_action not in ['wait', 'wait(3)', 'wait(5)', 'fill_dish_with_soup']:
                warning_msg = f"[ITDP-WARNING] LLM chose '{ml_action}' but holding DISH + soup cooking! ITDP suggests: wait or fill_dish_with_soup"

        # 拿着食材的情况
        if held in ['onion', 'tomato']:
            if (soup_ready or soup_cooking) and ml_action not in ['place_obj_on_counter', 'wait']:
                warning_msg = f"[ITDP-WARNING] LLM chose '{ml_action}' but soup ready/cooking! ITDP suggests: place_obj_on_counter"
            elif pot_items < 3 and ml_action not in ['put_onion_in_pot', 'put_tomato_in_pot', 'place_obj_on_counter', 'wait']:
                warning_msg = f"[ITDP-WARNING] LLM chose '{ml_action}' but holding ingredient! ITDP suggests: put_onion_in_pot"

        # 手空时的检查
        if held is None:
            if not soup_ready and not soup_cooking and ml_action in ['pickup_dish', 'fill_dish_with_soup']:
                warning_msg = f"[ITDP-WARNING] LLM chose '{ml_action}' but no soup! ITDP suggests: pickup_onion"

        # 打印警告（如果有）
        if warning_msg:
            print(warning_msg)
            print(f"[LLM DECISION KEPT] Final action remains: {ml_action}")

        # 始终返回LLM的决策，不覆盖
        return ml_action


# ============================================================
# DEIA-Agent: Dual Expert Intent-Aware Agent
# 双重专家意图感知智能体
# ============================================================
#
# 核心创新：解决LLM在协作场景中的三大痛点
#
# 痛点1：LLM推理时间长，无法实时协作
#   → 用规则/贝叶斯预计算结构化上下文，LLM只做最终判断
#
# 痛点2：LLM识别队友意图耗时
#   → BayesianTaskBelief实时维护队友意图概率分布，直接注入提示词
#
# 痛点3：被动等待队友行动，协作效率低
#   → 预判队友最可能做的事，主动选择互补任务
#
# 决策流程：
# 1. ITDP快速预分析（无LLM）
#    → 队友意图概率分布 P(队友做X) for each X
#    → 任务瓶颈优先级队列 [B1>B2>B3...]
#    → 规则推荐动作
# 2. 将预分析结果注入LLM提示词
# 3. LLM基于结构化上下文做最终决策（不再从头分析）
# ============================================================

class DEIAAgent(ITDPAgent):
    """
    DEIA-Agent: Dual Expert Intent-Aware Agent
    双重专家意图感知智能体

    与其他方法对比：
    - ProAgent : LLM从原始状态推理一切（慢、易出错）
    - ITDPAgent: 100%规则，不调用LLM
    - DEIAAgent : 规则预计算上下文 + LLM做最终决策（快且准）
    """

    def __init__(
            self,
            mlam,
            layout,
            model='Qwen/Qwen2.5-7B-Instruct',
            prompt_level='l2-ap',
            belief_revision=False,
            retrival_method="recent_k",
            K=1,
            auto_unstuck=True,
            controller_mode='new',
            debug_mode='N',
            agent_index=None,
            outdir=None
    ):
        super().__init__(
            mlam=mlam,
            layout=layout,
            model=model,
            prompt_level=prompt_level,
            belief_revision=belief_revision,
            retrival_method=retrival_method,
            K=K,
            auto_unstuck=auto_unstuck,
            controller_mode=controller_mode,
            debug_mode=debug_mode,
            agent_index=agent_index,
            outdir=outdir
        )

        self._prev_tm_held = None  # 用于追踪队友手持物品变化

        print(f"\n{'='*60}")
        print(f"[DEIA-Agent] Dual Expert Intent-Aware Agent")
        print(f"  Mode    : ITDP pre-analysis + LLM final decision")
        print(f"  Solves  : slow reasoning / passive collab / intent blindness")
        print(f"{'='*60}\n")

    def reset(self):
        super().reset()
        self._prev_tm_held = None

    def action(self, state):
        """Override: 追踪手持物品变化，为BC等无ml_action队友合成动作信号"""
        teammate = state.players[1 - self.agent_index]
        tm_held = teammate.get_object().name if teammate.has_object() else None

        # 从手持物品变化推断队友执行了什么动作（解决BC无ml_action问题）
        if tm_held != self._prev_tm_held:
            kitchen_state = self._get_kitchen_state(state)
            synthetic_action = self._infer_tm_action_from_held(
                self._prev_tm_held, tm_held, kitchen_state
            )
            if synthetic_action:
                self.itdp.update_teammate_observation(synthetic_action, teammate.position)

        self._prev_tm_held = tm_held
        return super().action(state)

    def _infer_tm_action_from_held(self, prev_held, curr_held, kitchen_state):
        """从手持物品前后变化推断队友执行的中层动作"""
        pot_items = kitchen_state.get('pot_items', 0)
        soup_ready = kitchen_state.get('soup_ready', False)

        if prev_held is None and curr_held in ['onion', 'tomato']:
            return f'pickup_{curr_held}'
        if prev_held is None and curr_held == 'dish':
            return 'pickup_dish'
        if prev_held in ['onion', 'tomato'] and curr_held is None:
            return 'put_onion_in_pot' if pot_items > 0 else 'place_obj_on_counter'
        if prev_held == 'dish' and curr_held == 'soup':
            return 'fill_dish_with_soup'
        if prev_held == 'soup' and curr_held is None:
            return 'deliver_soup'
        if prev_held == 'dish' and curr_held is None:
            return 'place_obj_on_counter'
        return None

    def _reinforce_belief_from_tm_held(self, tm_held, kitchen_state):
        """
        根据队友手持物品直接强化贝叶斯信念。

        predict_intent() 在检测到强信号后会 soft_reset 信念，
        导致信念始终停留在均匀分布（LOW confidence）。
        此方法在 decide() 返回后重注入信念，覆盖 soft_reset 的破坏。
        """
        from .itdp_module import Bottleneck
        belief = self.itdp.intent_predictor.bayesian_belief
        if belief is None or tm_held is None:
            return

        soup_ready = kitchen_state.get('soup_ready', False)

        # 手持物品 → 对应瓶颈的强先验
        target_map = {
            'soup':   Bottleneck.NEED_DELIVERY,
            'dish':   Bottleneck.NEED_PLATING if soup_ready else Bottleneck.NEED_DISH,
            'onion':  Bottleneck.NEED_POT_FILLING,
            'tomato': Bottleneck.NEED_POT_FILLING,
        }
        target = target_map.get(tm_held)
        if target is None:
            return

        # 目标瓶颈设为0.65，其余均分剩余0.35
        n = len(belief.bottlenecks)
        rest = 0.35 / max(n - 1, 1)
        for b in belief.bottlenecks:
            belief.belief[b] = 0.65 if b == target else rest

        print(f"[DEIA] Belief reinforced: teammate holding '{tm_held}' → {target.value} = 65%")

    def generate_ml_action(self, state):
        """DEIA核心决策：规则预分析 + LLM最终决策"""

        # === Step 1: 快速预分析（不调用LLM）===
        kitchen_state = self._get_kitchen_state(state)
        my_state = self._get_player_state(state.players[self.agent_index])
        teammate_state = self._get_player_state(state.players[1 - self.agent_index])
        reachability = self._check_reachability(state)

        itdp_action, itdp_reason, debug_info = self.itdp.decide(
            kitchen_state, my_state, teammate_state,
            self.agent_index, reachability=reachability
        )

        # decide() 内的 predict_intent() 会 soft_reset 信念，在此重注入
        self._reinforce_belief_from_tm_held(
            teammate_state.get('held_object'), kitchen_state
        )
        # 重注入后刷新 debug_info 中的置信度，让 prompt 显示真实值
        debug_info['intent_confidence'] = self.itdp.intent_predictor.get_confidence()

        # === Step 2: 构建结构化上下文块 ===
        deia_block = self._build_deia_prompt_block(
            kitchen_state, my_state, teammate_state,
            itdp_action, itdp_reason, debug_info, reachability
        )

        # === Step 3: 构建完整提示词（标准状态描述 + DEIA分析块）===
        if self.prompt_level == "l3-aip" and self.belief_revision:
            belief_prompt = self.generate_belief_prompt()
        else:
            belief_prompt = ''
        state_prompt = belief_prompt + self.generate_state_prompt(state)
        full_prompt = state_prompt + deia_block

        print(f"\n### [DEIA] Expert Pre-Analysis Block")
        print(deia_block)

        # === Step 4: LLM决策 ===
        state_message = {"role": "user", "content": full_prompt}
        self.planner.current_user_message = state_message
        response = self.planner.query(key=self.openai_api_key(), stop='Scene', trace=self.trace)

        if 'wait' not in response:
            self.planner.add_msg_to_dialog_history(state_message)
            self.planner.add_msg_to_dialog_history({"role": "assistant", "content": response})

        print(f"\n### [DEIA] LLM Response\n{response}")

        # === Step 5: 解析动作（复用ProMediumLevelAgent的解析器）===
        ml_action = self.parse_ml_action(response, self.agent_index)

        # --- Fallback #1：格式解析失败 ---
        # parse_ml_action 解析失败时默认返回 wait(1)，但 response 里并无 "wait"
        # 此时直接用 ITDP 推荐动作，避免浪费一步
        if ml_action == 'wait(1)' and 'wait' not in response.lower():
            print(f"[DEIA] Parse failed (response had no 'wait'), fallback to ITDP: {itdp_action}")
            ml_action = itdp_action

        # --- Fallback #2：wait 不合理时替换 ---
        # LLM 选了 wait，但当前没有真实阻塞原因 → 用 ITDP 推荐动作
        if 'wait' in ml_action:
            pot_blocked = reachability.get('pot_blocked', False)
            serve_blocked = reachability.get('serve_blocked', False)
            is_legitimately_blocked = pot_blocked or serve_blocked
            if not is_legitimately_blocked and 'wait' not in itdp_action:
                print(f"[DEIA] LLM chose wait without blocking reason, fallback to ITDP: {itdp_action}")
                ml_action = itdp_action

        if "wait" not in ml_action:
            self.planner.add_msg_to_dialog_history({"role": "assistant", "content": ml_action})

        print(f"[DEIA] Player {self.agent_index} final action: {ml_action}")
        self.current_ml_action_steps = 0

        if "wait" in ml_action:
            import re as _re
            match = _re.search(r'wait\((\d+)\)', ml_action)
            self.time_to_wait = int(match.group(1)) if match else 3

        return ml_action

    def _build_deia_prompt_block(self, kitchen_state, my_state, teammate_state,
                                  itdp_action, itdp_reason, debug_info, reachability):
        """构建DEIA结构化分析块，注入LLM提示词"""

        my_held = my_state.get('held_object') or 'nothing'
        tm_held = teammate_state.get('held_object') or 'nothing'

        intent_block = self._format_intent_distribution_block()
        teammate_handling = debug_info.get('teammate_handling', [])
        intent_conf = debug_info.get('intent_confidence', 0.0)
        conf_label = "HIGH" if intent_conf > 0.6 else "MEDIUM" if intent_conf > 0.3 else "LOW"
        tm_handling_str = ', '.join(teammate_handling) if teammate_handling else 'unknown'

        bottleneck_block = self._format_bottleneck_block(kitchen_state)

        reach_lines = []
        for loc in ['pot', 'serve', 'onion', 'dish']:
            ok = "OK" if reachability.get(loc, True) else "BLOCKED(wall)"
            blk = " [teammate blocking, consider waiting]" if reachability.get(f'{loc}_blocked', False) else ""
            reach_lines.append(f"  {loc:<8}: {ok}{blk}")
        reach_block = "\n".join(reach_lines)

        sep = "=" * 58
        return f"""

{sep}
[DEIA Pre-Analysis] Instant expert system (no LLM needed here)
{sep}
[Teammate Intent Distribution] (Bayesian inference from observations)
{intent_block}
  --> Most likely handling : {tm_handling_str}
  --> Inference confidence  : {conf_label} ({intent_conf:.0%})

[Task Priority Queue] (what the kitchen needs most urgently)
{bottleneck_block}

[Reachability Status]
{reach_block}

[Expert Recommendation]
  Action    : {itdp_action}
  Reasoning : {itdp_reason}
  Logic     : pick highest-priority task that teammate is NOT handling
{sep}
[YOUR DECISION]
You are holding  : {my_held}
Teammate holding : {tm_held}
Teammate likely doing : {tm_handling_str}

Instructions (read carefully):
  1. Do NOT re-derive what teammate is doing — use the distribution above.
  2. Do NOT re-analyze task priorities — use the queue above.
  3. Pick the highest-priority task that teammate is NOT doing.
  4. If the expert recommendation matches your state, use it.
  5. Only override if you hold something incompatible — state reason briefly.
  6. WAIT IS LAST RESORT: only choose wait(N) if a target location is physically
     blocked by your teammate right now. NEVER wait just to be cautious or safe.
     If in doubt, take the expert recommendation — acting beats waiting.

Valid actions: pickup_onion, pickup_dish, put_onion_in_pot, fill_dish_with_soup,
               deliver_soup, place_obj_on_counter, wait(N) [only if truly blocked]

Player {self.agent_index}: [your action here]
{sep}
"""

    def _format_intent_distribution_block(self):
        """格式化队友意图概率分布（ASCII可视化）"""
        if not self.itdp.intent_predictor.bayesian_belief:
            return "  (Bayesian inference not available)"

        belief = self.itdp.intent_predictor.bayesian_belief.belief
        sorted_items = sorted(belief.items(), key=lambda x: x[1], reverse=True)

        lines = []
        for bottleneck, prob in sorted_items:
            if prob < 0.02:
                continue
            filled = int(prob * 20)
            bar = '#' * filled + '-' * (20 - filled)
            lines.append(f"  {bottleneck.value:<22} {prob:>5.1%}  [{bar}]")

        return "\n".join(lines) if lines else "  (no observations yet — uniform prior)"

    def _format_bottleneck_block(self, kitchen_state):
        """格式化任务瓶颈优先级队列"""
        bottlenecks = self.itdp.pipeline_analyzer.analyze(kitchen_state)

        urgency_labels = {
            0: "[CRITICAL]", 1: "[CRITICAL]",
            2: "[HIGH    ]", 3: "[MEDIUM  ]",
            4: "[LOW     ]", 5: "[LOW     ]"
        }

        lines = []
        seen_types = set()
        for bn in bottlenecks:
            if bn.type in seen_types:
                continue
            seen_types.add(bn.type)
            label = urgency_labels.get(bn.priority, "[LOW     ]")
            lines.append(f"  {label} {bn.description:<38} --> {bn.required_action}")
            if len(lines) >= 5:
                break

        return "\n".join(lines) if lines else "  (no bottlenecks detected)"