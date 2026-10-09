"""CampusRAG 的 Streamlit 问答界面。"""
from pathlib import Path
from typing import Any, Dict, List

import streamlit as st

from config import get_settings
from src.indexer import INDEX_VERSION_FILE
from src.parser import load_directory
from src.pipeline import QueryTrace, RAGPipeline
from src.retriever import SearchResult


st.set_page_config(page_title="CampusRAG", page_icon="C", layout="wide")


@st.cache_data(show_spinner=False)
def load_filter_options() -> tuple[Dict[str, List[Any]], List[str]]:
    """从已有文档提取可选过滤字段，不把筛选项写死在界面里。"""
    errors: List[str] = []
    docs = load_directory(
        get_settings().data_dir,
        on_error=lambda path, exc: errors.append(f"{path}: {exc}"),
    )
    options = {
        "year": sorted({doc.metadata["year"] for doc in docs if "year" in doc.metadata}, reverse=True),
        "category": sorted({doc.metadata["category"] for doc in docs if "category" in doc.metadata}),
        "department": sorted({doc.metadata["department"] for doc in docs if "department" in doc.metadata}),
    }
    return options, errors


def current_index_version() -> str:
    index_dir = Path(get_settings().index_dir)
    marker = index_dir / INDEX_VERSION_FILE
    if marker.exists():
        return marker.read_text(encoding="ascii").strip()
    if not index_dir.exists():
        return "missing"
    mtimes = [path.stat().st_mtime_ns for path in index_dir.rglob("*") if path.is_file()]
    return str(max(mtimes, default=0))


@st.cache_resource(show_spinner=False, max_entries=2)
def get_pipeline(index_version: str) -> RAGPipeline:
    del index_version  # 缓存键用于在索引更新后创建新的 Retriever。
    return RAGPipeline()


def selected_filters(options: Dict[str, List[Any]]) -> Dict[str, Any]:
    """只返回用户实际选择的筛选条件。"""
    with st.sidebar:
        st.header("检索范围")
        year = st.selectbox("年份", ["全部", *options["year"]])
        category = st.selectbox("类别", ["全部", *options["category"]])
        department = st.selectbox("部门", ["全部", *options["department"]])
        st.divider()
        if st.button("清空对话", width="stretch"):
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


def _trace_rows(results: List[SearchResult]) -> List[Dict[str, Any]]:
    return [
        {
            "排名": rank,
            "标题": result.metadata.get("title", "未命名文档"),
            "章节": result.metadata.get("heading", ""),
            "分数": round(result.score, 4),
            "向量排名": result.vector_rank,
            "BM25 排名": result.bm25_rank,
            "内容预览": " ".join(result.text.split())[:180],
        }
        for rank, result in enumerate(results, start=1)
    ]


def render_trace(trace: QueryTrace | None) -> None:
    if trace is None:
        return
    with st.expander("检索 Trace", icon=":material/account_tree:"):
        st.caption("原始问题")
        st.code(trace.original_query, language=None)
        st.caption("实际检索问题")
        st.code(trace.retrieval_query, language=None)
        if trace.filters:
            st.caption("Metadata filters")
            st.json(trace.filters)

        stages = (
            ("Vector", trace.vector_results),
            ("BM25", trace.bm25_results),
            ("RRF", trace.fused_results),
            ("Rerank", trace.reranked_results),
        )
        tabs = st.tabs([f"{name} · {len(results)}" for name, results in stages])
        for tab, (_, results) in zip(tabs, stages):
            with tab:
                if results:
                    st.dataframe(
                        _trace_rows(results),
                        hide_index=True,
                        width="stretch",
                        column_config={
                            "排名": st.column_config.NumberColumn(width="small"),
                            "标题": st.column_config.TextColumn(width="medium", pinned=True),
                            "章节": st.column_config.TextColumn(width="medium"),
                            "分数": st.column_config.NumberColumn(format="%.4f", width="small"),
                            "向量排名": st.column_config.NumberColumn(width="small"),
                            "BM25 排名": st.column_config.NumberColumn(width="small"),
                            "内容预览": st.column_config.TextColumn(width="large"),
                        },
                    )
                else:
                    st.caption("该阶段没有结果。")


def render_message(message: Dict[str, Any]) -> None:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant":
            for warning in message.get("citation_warnings", []):
                st.warning(f"引用校验：{warning}")
            render_sources(message.get("sources", []))
            render_trace(message.get("trace"))


def main() -> None:
    st.title("CampusRAG")
    st.caption("校园制度与通知问答")

    options, document_errors = load_filter_options()
    if document_errors:
        st.warning(f"有 {len(document_errors)} 个本地文件无法解析，已跳过；问答界面仍可使用已有索引。")
        with st.expander("查看无法解析的文件"):
            for error in document_errors:
                st.write(error)
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

    history = [
        {"role": message["role"], "content": message["content"]}
        for message in st.session_state.messages
        if message.get("role") in {"user", "assistant"}
        and isinstance(message.get("content"), str)
    ]
    user_message = {"role": "user", "content": question}
    st.session_state.messages.append(user_message)
    render_message(user_message)

    with st.chat_message("assistant"):
        with st.spinner("正在检索资料并组织回答..."):
            try:
                answer = get_pipeline(current_index_version()).ask(
                    question,
                    filters=filters,
                    history=history,
                )
            except (RuntimeError, ValueError) as exc:
                st.error(str(exc))
                return
        st.markdown(answer.text)
        warnings = answer.citation_check.errors
        for warning in warnings:
            st.warning(f"引用校验：{warning}")
        render_sources(answer.sources)
        render_trace(answer.trace)

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer.text,
            "sources": answer.sources,
            "citation_warnings": warnings,
            "trace": answer.trace,
        }
    )


if __name__ == "__main__":
    main()
