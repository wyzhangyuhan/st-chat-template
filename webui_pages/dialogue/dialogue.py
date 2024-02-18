import streamlit as st
from streamlit_chatbox import *
from datetime import datetime
import os
import uuid
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


def dialogue_page(api: CoreFunc, is_lite: bool = False):
    if not chat_box.chat_inited:
        default_model = "GS-LLM"
        st.toast(
            f"当前运行的模型`{default_model}`, 您可以开始提问了."
        )
        chat_box.init_session()

    if "dialogue_history" not in st.session_state:
        st.session_state['dialogue_history'] = True
        chat_box.reset_history()
        
    with st.sidebar:
        dialogue_modes = ["LLM 对话"]
        dialogue_mode = st.selectbox("请选择对话模式：",
                                     dialogue_modes,
                                     index=0,
                                     key="dialogue_mode",
                                     )

        available_models = ["GS-LLM"]
        temperature = st.slider("Temperature：", 0.0, 1.0, TEMPERATURE, 0.05)
        history_len = st.number_input("历史对话轮数：", 0, 20, HISTORY_LEN)

    # Display chat messages from history on app rerun
    chat_box.output_messages()

    chat_input_placeholder = "请输入对话内容，换行请使用Shift+Enter "

    def on_feedback(
        feedback,
        chat_history_id: str = "",
        history_index: int = -1,
    ):
        reason = feedback["text"]
        score_int = chat_box.set_feedback(feedback=feedback, history_index=history_index)
        api.chat_feedback(chat_history_id=chat_history_id,
                          score=score_int,
                          reason=reason)
        st.session_state["need_rerun"] = True

    feedback_kwargs = {
        "feedback_type": "thumbs",
        "optional_text_label": "欢迎反馈您打分的理由",
    }

    if prompt := st.chat_input(chat_input_placeholder, key="prompt"):
        history = get_messages_history(history_len)
        chat_box.user_say(prompt)
        if dialogue_mode == "LLM 对话":
            chat_box.ai_say("")
            text = ""
            chat_history_id = ""
            r = api.gs_chat_stream(prompt, history=history)
            for t in r.iter_lines(chunk_size=8192, decode_unicode=False, delimiter=b"data:"):
                if b"[DONE]" in t:
                    break
                if t:
                    data = json.loads(t)
                
                    try:
                        text += data["choices"][0]["delta"]["content"]
                    except:
                        text += ""

                    chat_box.update_msg(text, streaming=True)
                    chat_history_id = data.get("chat_history_id", str(uuid.uuid4()))
            logging.info(f"[GENERAL] Infer: {text} Input: {prompt}")

            metadata = {
                "chat_history_id": chat_history_id,
            }
            chat_box.update_msg(text, streaming=False, metadata=metadata)  # 更新最终的字符串，去除光标
            chat_box.show_feedback(**feedback_kwargs,
                                   key=chat_history_id,
                                   on_submit=on_feedback,
                                   kwargs={"chat_history_id": chat_history_id, "history_index": len(chat_box.history) - 1})
 
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