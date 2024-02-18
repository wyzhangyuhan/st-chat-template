import re, os, glob, json, fitz, string, random, uuid, jsonlines, requests
import base64
import logging
from datetime import datetime
from copy import deepcopy
import numpy as np
from tqdm import tqdm
from typing import *
from pathlib import Path
# 此处导入的配置为发起请求（如WEBUI）机器上的配置，主要用于为前端设置默认值。分布式部署时可以与服务器上的不同
from configs import (
    TEMPERATURE,
    SCORE_THRESHOLD,
    CHUNK_SIZE,
    OVERLAP_SIZE,
    ZH_TITLE_ENHANCE,
    VECTOR_SEARCH_TOP_K,
    MAX_TOKENS,
    PRESENCE_PENALTY,
    FREQUENCY_PENALTY,
)

import base64   


class CoreFunc:
    def __init__(self):
        self.url = 'http://127.0.0.1:31100/v1/chat/completions'
        self.task_url = 'http://127.0.0.1:31101/v1/chat/completions'
        self.headers = {"Content-Type": "application/json"}

    """
    chat
    """

    def gs_chat(self, query: str, history: List[Dict] = []):
        """
        basic chat
        """
        # print(query, history)
        if history is None:
            history = []
        chat_session = history
        chat_session.append({"role":"user", "content": query})
        # print(chat_session)
        data = {
            'model': 'task',
            "messages": chat_session,
            "presence_penalty": PRESENCE_PENALTY,
            "frequency_penalty": FREQUENCY_PENALTY,
            "temperature": TEMPERATURE,
            "max_tokens": MAX_TOKENS,
            'top_p': 1,
            "stream": True,
        }

        gs = gs_infer()
        f = gs.infer(data)
        return f

    def gs_chat_stream(self, query: str, history: List[Dict] = []):
        """
        basic chat
        """
        # print(query, history)
        if history is None:
            history = []
        chat_session = history
        chat_session.append({"role":"user", "content": query})
        # print(chat_session)
        data = {
            'model': 'task',
            "messages": chat_session,
            "presence_penalty": PRESENCE_PENALTY,
            "frequency_penalty": FREQUENCY_PENALTY,
            "temperature": TEMPERATURE,
            "max_tokens": MAX_TOKENS,
            'top_p': 1,
            "stream": True,
        }

        response = requests.post(self.url, headers=self.headers, json=data, stream=True)
        return response

    def gs_instruct_tuning_chat(self, query: str) -> str:
        prompt = f"""
            接下来我将给你用户的提问，你需要提取出问题中的关键词，并通过该关键词联想相关的1-2个词返回给我。以下是一些例子：
            
            用户输入: 我旷工了1周会受到什么惩罚？
            输出: 旷工, 旷工一周, 惩罚, 惩罚条款

            用户输入: 我在上班路上出车祸了算工伤吗？
            输出: 工伤条款, 工伤赔偿, 上班路上, 车祸

            现在我将给你用户的输入，请完成任务。
            用户输入: {query}
            输出:
        """

        input = [{"role":"user", "content": prompt}]
        data = {
            'model': 'task',
            "messages": input,
            "presence_penalty": PRESENCE_PENALTY,
            "frequency_penalty": FREQUENCY_PENALTY,
            "temperature": TEMPERATURE,
            "max_tokens": MAX_TOKENS,
            'top_p': 1,
            "stream": True,
        }

        gs = gs_infer()
        f = gs.infer(data)

        text = ""
        for t in f:
            t = t[6::]
            if "[DONE]" in t:
                break
            t = json.loads(t)
            try:
                text += t["choices"][0]["delta"]["content"]
            except:
                text += ""
        
        return text
    
    def gs_knowledge_chat(self, query: str, db_name: str,
                          top_k: int = VECTOR_SEARCH_TOP_K,
                          score_threshold: float = SCORE_THRESHOLD,
                          history: List[Dict] = [],
                          stream: bool = True):
        
        if history is None:
            history = []

        ref = self.search_milvus(db_name, query, top_k, score_threshold)

        prompt = """
\n以上是我检索到的知识，请分析知识与我所提问题之间的关系。如果知识里包含问题的回答，请抽取回答。并重新组织语言进行输出，你的输出格式希望能尽量美观。
请注意你的输出需要真实、客观、不要捏造我所提供知识中没有的内容。
我的问题是：
        """
        docs = '\n\n'.join(ref)

        input_instruct = docs + '\n' + prompt + '\n' + query
        print(query)

        chat_session = history
        chat_session.append({"role":"user", "content": input_instruct})
        # print(chat_session)
        data = {
            'model': 'task',
            "messages": chat_session,
            "presence_penalty": PRESENCE_PENALTY,
            "frequency_penalty": FREQUENCY_PENALTY,
            "temperature": TEMPERATURE,
            "max_tokens": MAX_TOKENS,
            'top_p': 1,
            "stream": True,
        }

        gs = gs_infer()
        f = gs.task_infer(data)

        return f
    
    def contains_chinese(self, s):
        """
        Check if the string contains any Chinese character.
        :param s: The string to check.
        :return: True if the string contains any Chinese character, False otherwise.
        """
        for char in s:
            if '\u4e00' <= char <= '\u9fff':
                return True
        return False

    """
    utils
    """
    def save_jsonl(self, save_dict, db_name, save_path="./db_bk"):
        with jsonlines.open(f"{save_path}/{db_name}.jsonl", "a") as w:
            for item in save_dict:
                w.write(item)

    def save_split(self, save_dict, db_name, save_path="./db_bk"):
        with jsonlines.open(f"{save_path}/{db_name}.jsonl", "a") as w:
            for item in save_dict:
                w.write(item)

    def locate_file(self, data_name):
        lines = []
        id2idx  = {}
        with jsonlines.open(f"./db_bk/{data_name}.jsonl", "r") as reader:
            for idx, item in enumerate(reader):
                lines.append(item)
                id2idx[item["id"]] = idx
        return lines, id2idx

    def extend_span(self, data, id2idx, id, pages=5):
        base_idx = id2idx[id]
        st_idx = max(0, base_idx-pages)
        ed_idx = min(len(data)-1, base_idx+pages)
        text_list = [data[i]['content'].split('|')[-1] for i in range(st_idx, ed_idx+1)]
        info = ' | '.join(data[base_idx]['content'].split('|')[:-1])
        return info + "\n\n" + ''.join(text_list)
