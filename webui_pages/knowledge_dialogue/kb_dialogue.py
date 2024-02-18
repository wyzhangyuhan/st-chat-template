import streamlit as st
from streamlit_chatbox import *
from datetime import datetime
import os
import io
import uuid
import string
import random
import json
import logging
from configs import (TEMPERATURE, HISTORY_LEN,DEFAULT_KNOWLEDGE_BASE, DEFAULT_SEARCH_ENGINE, SUPPORT_AGENT_MODEL, VECTOR_SEARCH_TOP_K, SCORE_THRESHOLD)
from typing import List, Dict
from core.core import CoreFunc

chat_box = ChatBox(
    assistant_avatar=os.path.join(
        "img",
        "chatchat_icon_blue_square_v2.png"
    )
)

def get_messages_history(history_len: int, content_in_expander: bool = False) -> List[Dict]:
    '''
    返回消息历史。
    content_in_expander控制是否返回expander元素中的内容，一般导出的时候可以选上，传入LLM的history不需要
    '''

    def filter(msg):
        content = [x for x in msg["elements"] if x._output_method in ["markdown", "text"]]
        if not content_in_expander:
            content = [x for x in content if not x._in_expander]
        content = [x.content for x in content]

        return {
            "role": msg["role"],
            "content": "\n\n".join(content),
        }

    return chat_box.filter_history(history_len=history_len, filter=filter)


def knowledge_dialogue(api: CoreFunc, is_lite=False):
    if not chat_box.chat_inited:
        default_model = "GS-LLM"
        st.toast(
            f"当前运行的模型`{default_model}`, 上传文件后, 您就可以开始提问了."
        )
        chat_box.init_session()


    if 'file_processed' not in st.session_state:
        st.session_state["file_processed"] = False
        st.session_state["db_name"] = ""
        st.session_state["file_name"] = ""

    with st.sidebar:
        files = st.file_uploader("上传知识文件：",
                                 [".docx", ".pdf"],
                                 accept_multiple_files=False,
                                )
        if files is None:
            files = [] 
        else:
            files = [files]        
        progress_placeholder = st.empty()


        if len(files)!=0:
            if st.session_state["file_name"] != files[0].name:
                gen_db_name = ''.join(random.choice(string.ascii_letters) for _ in range(8))
                logging.info(f"[DB-CREATE] db_name: {gen_db_name} file_name: {files[0].name}")
                st.session_state["db_name"] = gen_db_name
                st.session_state["file_name"] = files[0].name
                api.create_milvus(gen_db_name)
                progress_bar = progress_placeholder.progress(10)
                file_bytes, file_names = [], []

                for file in files:
                    file_bytes.append(io.BytesIO(file.read()))
                    file_names.append(file.name)
                
                ret = api.upload_milvus_docs(file_bytes, file_names, gen_db_name)
                progress_bar = progress_placeholder.progress(100)
                st.session_state["file_processed"] = True
                st.toast(ret)
        else:
            st.write("请上传文件后再进行知识问答～")

    chat_box.output_messages()
    file_uploaded = (len(files)!=0 and st.session_state["file_processed"])
    chat_input_placeholder = "请输入对话内容，换行请使用Shift+Enter "

    if prompt := st.chat_input(chat_input_placeholder, key="kb_prompt", disabled=not file_uploaded):
        chat_box.user_say(prompt)
        gen_db_name = st.session_state["db_name"]
        
        it_prompt = api.gs_instruct_tuning_chat(prompt)
        logging.info(f"[KB-DIALOGUE1] Infer: {it_prompt} Input: {prompt}")
        
        chat_box.ai_say([
                f"正在查询知识库 ...",
                Markdown("...", in_expander=True, title="知识库匹配结果", state="complete"),
        ])
        text = ""
        docs = api.search_milvus(gen_db_name, prompt + " "+ it_prompt, 1, 0.9)
        logging.info(f"[SEARCH-RES] : Input: {prompt + ' ' + it_prompt} RES: {docs}")
        r = api.gs_knowledge_chat(prompt + " "+ it_prompt,
                                    db_name=gen_db_name,
                                    top_k=5,
                                    score_threshold=0.9,
                                    history=None)
        
        for t in r:
            t = t[6::]
            if "[DONE]" in t:
                break
            t = json.loads(t)
            try:
                text += t["choices"][0]["delta"]["content"]
            except:
                text += ""
            # print("text:", text)
            chat_box.update_msg(text)
            
        logging.info(f"[KB-DIALOGUE2] Infer: {text} Input: {prompt + it_prompt}")
        
        md_search_res = "# 知识库信息\n"
        for idx, doc in enumerate(docs):
            md_search_res += f"\n## Top {idx} \n"
            md_search_res = md_search_res + "> " + doc

        chat_box.update_msg(text, element_index=0, streaming=False)
        chat_box.update_msg(md_search_res, element_index=1, streaming=False)
    
    if st.session_state.get("need_rerun"):
        st.session_state["need_rerun"] = False
        st.rerun()

    now = datetime.now()
    with st.sidebar:
        cols = st.columns(2)
        export_btn = cols[0]
        if cols[1].button(
                "清空对话",
                use_container_width=True,
        ):
            chat_box.reset_history()
            st.rerun()

    export_btn.download_button(
        "导出记录",
        "".join(chat_box.export2md()),
        file_name=f"{now:%Y-%m-%d %H.%M}_对话记录.md",
        mime="text/markdown",
        use_container_width=True,
    )