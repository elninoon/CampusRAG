# CampusRAG

一个面向校园通知与规章制度的本地 RAG 命令行项目。它支持从 Markdown、TXT、PDF、DOCX 和 HTML 文档建立知识库，并通过混合检索、精排和带来源编号的回答生成来回答问题。

## 功能

- 递归解析 `data/raw` 下的多种文档格式，并保留 YAML frontmatter 元数据。
- 提供固定长度切分和按中文公文标题切分两种策略，默认使用结构化切分。
- 以 Chroma 持久化向量索引，并使用稳定 chunk ID 支持重复建库更新。
- 结合向量检索与 BM25 关键词检索，使用 RRF 融合排序。
- 支持按 `year`、`category`、`department` 等元数据精确筛选。
- 使用 SiliconFlow Rerank API 对候选结果精排。
- 使用 DeepSeek 生成受资料约束的回答，并校验回答中的 `[n]` 引用是否指向真实来源。
- 提供 Hit@K 离线检索评测和单元测试。

## 流程

```text
原始文档 -> 解析 -> 结构化切分 -> Embedding -> Chroma 索引
用户问题 -> 向量召回 + BM25 召回 -> RRF -> Rerank -> LLM 回答 + 来源
```

## 快速开始

需要 Python 3.10 或更高版本，以及可用的 DeepSeek 与 SiliconFlow API 密钥。

```powershell
python -m pip install -r requirements.txt
Copy-Item .env.example .env
```

编辑 `.env`，至少填入：

```dotenv
DEEPSEEK_API_KEY=...
SILICONFLOW_API_KEY=...
```

建立或更新本地索引：

```powershell
python scripts\build_index.py
```

抓取华东师范大学研究生院公开网页并写入 `data/raw/yjsy`：

```powershell
python scripts\crawl_yjsy.py --max-pages 100
python scripts\build_index.py --reset
```

爬虫只保存公开 HTML 详情页，会跳过需要统一身份认证的页面和 PDF 等附件。

提取单个公开网页并保存为 Markdown：

```powershell
python scripts\ingest_webpage.py "https://www.ecnu.edu.cn/wzcd/xxgk/xqjj.htm" `
  --department "华东师范大学" --category "学校概况"
python scripts\build_index.py
```

应传入包含正文的具体页面，而不是只有链接的栏目目录页。如果自动识别不到正文，
可以通过 `--selector ".正文容器类名"` 指定 CSS 选择器。

首次需要从头重建索引时：

```powershell
python scripts\build_index.py --reset
```

提出问题：

```powershell
python scripts\ask.py "奖学金材料什么时候提交" --year 2026 --category 奖学金
```

## 浏览器界面

建立索引后启动本地界面：

```powershell
python -m streamlit run app.py
```

界面默认搜索全部资料。年份、类别和部门是侧边栏中的可选筛选项，只有选定具体值时才会限制检索范围。

只检查检索结果，不调用 reranker 或 LLM：

```powershell
python scripts\search.py "奖学金材料什么时候提交" --year 2026 --category 奖学金
```

排查错误回答时，查看向量召回、BM25、RRF 融合和 Rerank 的完整链路：

```powershell
python scripts\trace_query.py "华东师范大学的副校长是谁"
```

## MCP 工具服务

CampusRAG 可以作为 MCP Server 暴露给外部 Agent，例如 CampusOps Agent。启动方式：

```powershell
python scripts\mcp_server.py
```

当前暴露两个工具：

- `campus_rag_search`：执行向量检索 + BM25 + RRF 融合，只返回检索证据，不调用 reranker 或 LLM。
- `campus_rag_ask`：执行完整 RAGPipeline，返回受资料约束的回答、来源和引用校验结果。

工具参数支持 `year`、`category`、`department` 等 metadata 过滤。推荐在 Agent 里优先用 `campus_rag_search` 查证据；需要完整自然语言回答时再调用 `campus_rag_ask`。

## 数据格式

Markdown 和 TXT 文件可在正文前使用 YAML frontmatter。以下字段会进入检索过滤和来源展示：

```markdown
---
title: 关于开展2026年研究生奖学金评审工作的通知
department: 研究生院
publish_date: 2026-09-15
category: 奖学金
year: 2026
---

正文内容。
```

原始文件放在 `data/raw` 的任意子目录。PDF、DOCX 和 HTML 会直接解析，无需先转换为 Markdown。

非 Markdown 文件可以通过同名 `.meta.yaml` sidecar 补充元数据。例如：

```text
data/raw/yjsy/2026全日制研究生手册.pdf
data/raw/yjsy/2026全日制研究生手册.meta.yaml
```

sidecar 内容：

```yaml
title: 2026全日制研究生手册
department: 华东师范大学研究生院
category: 研究生手册
year: 2026
```

`source_path`、`format` 和 `document_id` 由解析器自动生成。PDF 会优先提取文本层；如果整份文件几乎没有可提取文字，则自动尝试 Tesseract OCR。扫描版中文 PDF 需要系统安装 Tesseract 及 `chi_sim` 中文语言包。

## 评测与测试

`data/eval/retrieval_cases.json` 保存人工标注的查询和目标文档。运行检索评测：

```powershell
python scripts\evaluate.py --top-k 5
```

运行全部单元测试：

```powershell
python -m unittest discover -s tests -v
```

Hit@K 衡量正确文档是否出现在前 K 个检索结果中。它用于发现召回问题，不等同于回答正确率。

## 项目结构

```text
config.py                 模型、路径和环境变量配置
data/raw/                 原始校园资料
data/eval/                检索评测样本
src/parser.py             多格式文档解析
src/web_ingestor.py       通用单网页正文提取与 Markdown 入库
src/chunker.py            文档切分
src/embeddings.py         Embedding API 客户端
src/indexer.py            Chroma 索引构建
src/retriever.py          向量 + BM25 混合检索
src/reranker.py           SiliconFlow 精排
src/generator.py          受约束回答生成
src/citations.py          引用编号校验
src/evaluator.py          Hit@K 评测
src/pipeline.py           端到端问答入口
src/mcp_server.py         CampusRAG MCP 工具服务
scripts/                  建库、检索、问答和评测命令
tests/                    不依赖外部 API 的单元测试
```

## 设计说明

向量检索擅长语义相近的表达，BM25 擅长年份、部门和通知名称等精确词。两者使用 RRF 按排名融合，避免比较不在同一尺度的分数。精排器只处理融合后的少量候选，降低成本并提高最终上下文的相关性。

回答生成被要求仅依据提供的资料作答。CLI 会展示来源列表，并在模型给出不存在或缺失的引用编号时打印校验警告。引用校验只能确认编号有效，不能自动证明每一条文字陈述都被资料完全支撑。

## 常见问题

`缺少 chromadb 依赖`：执行 `python -m pip install -r requirements.txt`。

`未找到本地索引`：先执行 `python scripts\build_index.py`。

`未配置 ... API_KEY`：复制 `.env.example` 为 `.env` 并填写对应密钥。

若更换 Embedding 模型或向量维度，请使用 `--reset` 重建索引，避免混用不同向量空间。
