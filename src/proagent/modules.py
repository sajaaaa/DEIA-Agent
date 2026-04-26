import openai
from rich import print as rprint
import time
from typing import Union
from .utils import convert_messages_to_prompt, retry_with_exponential_backoff

# Refer to https://platform.openai.com/docs/models/overview
TOKEN_LIMIT_TABLE = {
    "text-davinci-003": 4080,
    "gpt-3.5-turbo": 4096,
    "gpt-3.5-turbo-0301": 4096,
    "gpt-3.5-turbo-16k": 16384,
    "gpt-4": 8192,
    "gpt-4-0314": 8192,
    "gpt-4-32k": 32768,
    "gpt-4-32k-0314": 32768,
    # 硅基流动 Qwen 模型
    "Qwen/Qwen2-7B-Instruct": 32768,
    "Qwen/Qwen2.5-7B-Instruct": 32768,
    "Qwen/Qwen2.5-72B-Instruct": 32768,
}

# 硅基流动API配置
SILICONFLOW_API_BASE = "https://api.siliconflow.cn/v1"


class Module(object):
    """
    This module is responsible for communicating with GPTs.
    支持 OpenAI API 和 硅基流动(SiliconFlow) API
    """
    def __init__(self, 
                 role_messages, 
                 model="gpt-3.5-turbo",
                 retrival_method="recent_k",
                 K=3):
        '''
        args:  
        use_similarity: 
        dia_num: the num of dia use need retrival from dialog history
        '''

        self.model = model
        self.retrival_method = retrival_method
        self.K = K

        self.chat_model = self._is_chat_model()
        self.use_siliconflow = self._is_siliconflow_model()
        
        self.instruction_head_list = role_messages
        self.dialog_history_list = []
        self.current_user_message = None
        self.cache_list = None

    def _is_chat_model(self):
        """判断是否为聊天模型"""
        keywords = ["gpt", "qwen", "deepseek", "glm", "chat"]
        return any(kw in self.model.lower() for kw in keywords)
    
    def _is_siliconflow_model(self):
        """判断是否使用硅基流动的模型"""
        keywords = ["Qwen", "deepseek", "glm", "THUDM"]
        return any(kw.lower() in self.model.lower() for kw in keywords)

    def add_msgs_to_instruction_head(self, messages: Union[list, dict]):
        if isinstance(messages, list):
            self.instruction_head_list += messages
        elif isinstance(messages, dict):
            self.instruction_head_list += [messages]

    def add_msg_to_dialog_history(self, message: dict):
        self.dialog_history_list.append(message)
    
    def get_cache(self)->list:
        if self.retrival_method == "recent_k":
            if self.K > 0:
                return self.dialog_history_list[-self.K:]
            else: 
                return []
        else:
            return None 
           
    @property
    def query_messages(self)->list:
        return self.instruction_head_list + self.cache_list + [self.current_user_message]
    
    @retry_with_exponential_backoff
    def query(self, key, stop=None, temperature=0.0, debug_mode = 'Y', trace = True):
        openai.api_key = key
        
        # 设置API endpoint
        if self.use_siliconflow:
            openai.api_base = SILICONFLOW_API_BASE
            rprint(f"[blue][INFO][/blue]: Using SiliconFlow API with model: {self.model}")
        else:
            openai.api_base = "https://api.openai.com/v1"
        
        rec = self.K  
        if trace == True: 
            self.K = 0 
        self.cache_list = self.get_cache()
        messages = self.query_messages
        if trace == False: 
            messages[len(messages) - 1]['content'] += " Based on the failure explanation and scene description, analyze and plan again." 
        self.K = rec 
        response = "" 
        # print('\n\nmessages = \n\n{}\n\n'.format(messages))
        get_response = False
        retry_count = 0
        
        while not get_response:  
            if retry_count > 3:
                rprint("[red][ERROR][/red]: Query LLM failed for over 3 times!")
                return "wait(1)"
            try:  
                if self.model in ['text-davinci-003']:
                    prompt = convert_messages_to_prompt(messages) 
                    response = openai.Completion.create(
                        model=self.model,
                        prompt=prompt,
                        stop=stop,
                        temperature=temperature, 
                        max_tokens = 256
                    )
                    time.sleep(10)  
                elif self.chat_model:
                    response = openai.ChatCompletion.create(
                        model=self.model,
                        messages=messages,
                        stop=stop,
                        temperature=temperature, 
                        max_tokens = 256
                    )
                    # 硅基流动请求间隔可以短一些
                    if self.use_siliconflow:
                        time.sleep(1)
                    else:
                        time.sleep(10) 
                else:
                    raise Exception(f"Model {self.model} not supported.")
                
                get_response = True

            except Exception as e:
                retry_count += 1
                error_source = "SILICONFLOW" if self.use_siliconflow else "OPENAI"
                rprint(f"[red][{error_source} ERROR][/red]:", e)
                time.sleep(20)  
        return self.parse_response(response)

    def parse_response(self, response):
        if self.model == 'claude': 
            return response 
        elif self.model in ['text-davinci-003']:
            return response["choices"][0]["text"]
        elif self.chat_model:
            return response["choices"][0]["message"]["content"]
        else:
            # 默认尝试chat格式
            try:
                return response["choices"][0]["message"]["content"]
            except:
                return response["choices"][0]["text"]

    def restrict_dialogue(self):
        """
        The limit on token length for gpt-3.5-turbo-0301 is 4096.
        If token length exceeds the limit, we will remove the oldest messages.
        """
        limit = TOKEN_LIMIT_TABLE[self.model]
        print(f'Current token: {self.prompt_token_length}')
        while self.prompt_token_length >= limit:
            self.cache_list.pop(0)
            self.cache_list.pop(0)
            self.cache_list.pop(0)
            self.cache_list.pop(0)
            print(f'Update token: {self.prompt_token_length}')
        
    def reset(self):
        self.dialog_history_list = []
