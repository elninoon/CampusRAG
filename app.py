"""CampusRAG 的 Streamlit 问答界面。"""
from pathlib import Path
from typing import Any, Dict, List

import streamlit as st

from config import get_settings
from src.parser import load_directory
from src.pipeline import RAGPipeline


st.set_page_config(page_title="CampusRAG", page_icon="C", layout="wide")


@st.cache_data(show_spinner=False)
def load_filter_options() -> Dict[str, List[Any]]:
    """从已有文档提取可选过滤字段，不把筛选项写死在界面里。"""
    docs = load_directory(get_settings().data_dir)
    return {
        "year": sorted({doc.metadata["year"] for doc in docs if "year" in doc.metadata}, reverse=True),
        "category": sorted({doc.metadata["category"] for doc in docs if "category" in doc.metadata}),
        "department": sorted({doc.metadata["department"] for doc in docs if "department" in doc.metadata}),
    }


@st.cache_resource(show_spinner=False)
def get_pipeline() -> RAGPipeline:
    return RAGPipeline()


def selected_filters(options: Dict[str, List[Any]]) -> Dict[str, Any]:
    """只返回用户实际选择的筛选条件。"""
    with st.sidebar:
        st.header("检索范围")
        year = st.selectbox("年份", ["全部", *options["year"]])
        category = st.selectbox("类别", ["全部", *options["category"]])
        department = st.selectbox("部门", ["全部", *options["department"]])
        st.divider()
        if st.button("清空对话", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

    return {
        key: value
        for key, value in {
            "year": year,
            "category": category,
            "department": department,
        }.items()
        if value != "全部"
    }


def render_sources(sources: List[Any]) -> None:
    if not sources:
        return
    with st.expander(f"查看来源 ({len(sources)})", expanded=False):
        for number, source in enumerate(sources, start=1):
            metadata = source.metadata
            title = metadata.get("title", "未命名文档")
            heading = metadata.get("heading", "")
            source_url = metadata.get("source_url")
            st.markdown(f"**[{number}] {title}**")
            if heading:
                st.caption(heading)
            if source_url:
                st.link_button("打开原始链接", source_url)
            else:
                st.caption(metadata.get("source_path", "未知来源"))
            st.code(source.text, language=None)


def render_message(message: Dict[str, Any]) -> None:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant":
            for warning in message.get("citation_warnings", []):
                st.warning(f"引用校验：{warning}")
            render_sources(message.get("sources", []))


def main() -> None:
    st.title("CampusRAG")
    st.caption("校园制度与通知问答")

    options = load_filter_options()
    filters = selected_filters(options)
    settings = get_settings()
    if not Path(settings.index_dir).exists():
        st.info("尚未发现本地索引。请先在终端运行 python scripts\\build_index.py。")

    if "messages" not in st.session_state:
        st.session_state.messages = []
    for message in st.session_state.messages:
        render_message(message)

    question = st.chat_input("输入问题，例如：奖学金材料什么时候提交？")
    if not question:
        return

    user_message = {"role": "user", "content": question}
    st.session_state.messages.append(user_message)
    render_message(user_message)

    with st.chat_message("assistant"):
        with st.spinner("正在检索资料并组织回答..."):
            try:
                answer = get_pipeline().ask(question, filters=filters)
            except (RuntimeError, ValueError) as exc:
                st.error(str(exc))
                return
        st.markdown(answer.text)
        warnings = answer.citation_check.errors
        for warning in warnings:
            st.warning(f"引用校验：{warning}")
        render_sources(answer.sources)

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer.text,
            "sources": answer.sources,
            "citation_warnings": warnings,
        }
    )


if __name__ == "__main__":
    main()
