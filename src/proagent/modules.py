"""
ProAgent Module - 修改版
将OpenAI API替换为硅基流动(SiliconFlow) API
使用模型: Qwen/Qwen2-7B-Instruct
"""

import openai
from rich import print as rprint
import time
from typing import Union
from .utils import convert_messages_to_prompt, retry_with_exponential_backoff

# 硅基流动API配置
SILICONFLOW_API_BASE = "https://api.siliconflow.cn/v1"

# Token限制表 - 添加Qwen模型
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
    "Qwen/Qwen2-72B-Instruct": 32768,
    "Qwen/Qwen2.5-7B-Instruct": 32768,
    "Qwen/Qwen2.5-72B-Instruct": 32768,
}

# 硅基流动支持的模型列表
SILICONFLOW_MODELS = [
    "Qwen/Qwen2-7B-Instruct",
    "Qwen/Qwen2-72B-Instruct", 
    "Qwen/Qwen2.5-7B-Instruct",
    "Qwen/Qwen2.5-72B-Instruct",
    "deepseek-ai/DeepSeek-V2.5",
    "THUDM/glm-4-9b-chat",
]


class Module(object):
    """
    This module is responsible for communicating with LLMs.
    支持 OpenAI API 和 硅基流动(SiliconFlow) API
    """
    def __init__(self, 
                 role_messages, 
                 model="Qwen/Qwen2-7B-Instruct",  # 默认使用Qwen模型
                 retrival_method="recent_k",
                 K=3):
        '''
        args:  
        model: 模型名称，支持OpenAI模型和硅基流动模型
        retrival_method: 检索方法
        K: 检索的对话历史数量
        '''

        self.model = model
        self.retrival_method = retrival_method
        self.K = K

        # 判断是否为聊天模型
        self.chat_model = self._is_chat_model()
        
        # 判断是否使用硅基流动API
        self.use_siliconflow = self._is_siliconflow_model()
        
        self.instruction_head_list = role_messages
        self.dialog_history_list = []
        self.current_user_message = None
        self.cache_list = None

    def _is_chat_model(self) -> bool:
        """判断是否为聊天模型"""
        # GPT系列和Qwen系列都是聊天模型
        if "gpt" in self.model.lower():
            return True
        if "qwen" in self.model.lower():
            return True
        if "deepseek" in self.model.lower():
            return True
        if "glm" in self.model.lower():
            return True
        return False
    
    def _is_siliconflow_model(self) -> bool:
        """判断是否使用硅基流动的模型"""
        # 检查模型名是否包含硅基流动支持的模型关键词
        siliconflow_keywords = ["Qwen", "deepseek", "glm", "THUDM"]
        for keyword in siliconflow_keywords:
            if keyword.lower() in self.model.lower():
                return True
        return False

    def add_msgs_to_instruction_head(self, messages: Union[list, dict]):
        if isinstance(messages, list):
            self.instruction_head_list += messages
        elif isinstance(messages, dict):
            self.instruction_head_list += [messages]

    def add_msg_to_dialog_history(self, message: dict):
        self.dialog_history_list.append(message)
    
    def get_cache(self) -> list:
        if self.retrival_method == "recent_k":
            if self.K > 0:
                return self.dialog_history_list[-self.K:]
            else: 
                return []
        else:
            return None 
           
    @property
    def query_messages(self) -> list:
        return self.instruction_head_list + self.cache_list + [self.current_user_message]
    
    @retry_with_exponential_backoff
    def query(self, key, stop=None, temperature=0.0, debug_mode='Y', trace=True):
        """
        发送查询请求到LLM API
        
        Args:
            key: API密钥
            stop: 停止标记
            temperature: 温度参数
            debug_mode: 调试模式
            trace: 是否追踪
        """
        # 设置API密钥
        openai.api_key = key
        
        # 如果使用硅基流动，设置API base
        if self.use_siliconflow:
            openai.api_base = SILICONFLOW_API_BASE
            rprint(f"[blue][INFO][/blue]: Using SiliconFlow API with model: {self.model}")
        else:
            # 重置为OpenAI默认
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
        get_response = False
        retry_count = 0
        
        while not get_response:  
            if retry_count > 3:
                rprint("[red][ERROR][/red]: Query LLM failed for over 3 times!")
                return {}
            try:  
                if self.model in ['text-davinci-003']:
                    # 旧版Completion API (仅OpenAI)
                    prompt = convert_messages_to_prompt(messages) 
                    response = openai.Completion.create(
                        model=self.model,
                        prompt=prompt,
                        stop=stop,
                        temperature=temperature, 
                        max_tokens=256
                    )
                    time.sleep(10)  
                elif self.chat_model:
                    # Chat Completion API (OpenAI和硅基流动都支持)
                    response = openai.ChatCompletion.create(
                        model=self.model,
                        messages=messages,
                        stop=stop,
                        temperature=temperature, 
                        max_tokens=256
                    )
                    # 硅基流动API响应较快，可以减少等待时间
                    if self.use_siliconflow:
                        time.sleep(1)  # 硅基流动等待时间较短
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
        """解析API响应"""
        if self.model == 'claude': 
            return response 
        elif self.model in ['text-davinci-003']:
            return response["choices"][0]["text"]
        elif self.chat_model:
            # 统一处理所有聊天模型的响应 (GPT, Qwen等)
            return response["choices"][0]["message"]["content"]
        else:
            # 默认处理方式
            try:
                return response["choices"][0]["message"]["content"]
            except:
                return response["choices"][0]["text"]

    def restrict_dialogue(self):
        """
        限制对话长度，防止超出token限制
        """
        limit = TOKEN_LIMIT_TABLE.get(self.model, 4096)  # 默认4096
        print(f'Current token: {self.prompt_token_length}')
        while self.prompt_token_length >= limit:
            self.cache_list.pop(0)
            self.cache_list.pop(0)
            self.cache_list.pop(0)
            self.cache_list.pop(0)
            print(f'Update token: {self.prompt_token_length}')
        
    def reset(self):
        self.dialog_history_list = []
