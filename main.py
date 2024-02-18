import streamlit as st
import logging 
logging.basicConfig(filename="./logs/general.log",level=logging.INFO)

# from webui_pages.utils import *
from streamlit_option_menu import option_menu
from webui_pages.dialogue.dialogue import dialogue_page, chat_box
from webui_pages.knowledge_dialogue.kb_dialogue import knowledge_dialogue
import os
import sys
from core.core import CoreFunc

api = CoreFunc()

if __name__ == "__main__":
    is_lite = "lite" in sys.argv

    st.set_page_config(
        "GS-Demo WebUI",
        os.path.join("img", "favicon.ico"),
        initial_sidebar_state="expanded",
        layout="wide",
        menu_items={
            'About': f"""欢迎体验XX demo！"""
        }
    )

    pages = {
        "对话": {
            "icon": "chat",
            "func": dialogue_page,
        },
        "文件问答": {
            "icon": "file",
            "func": knowledge_dialogue,
        },
    }

    with st.sidebar:

        options = list(pages)
        icons = [x["icon"] for x in pages.values()]

        default_index = 0
        selected_page = option_menu(
            "",
            options=options,
            icons=icons,
            # menu_icon="chat-quote",
            default_index=default_index,
        )


    pages[selected_page]["func"](api=api, is_lite=is_lite)
