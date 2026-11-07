"""M8 综合 Demo（交付物 D7 的可运行部分）：旅行助手的记忆 + RAG 知识问答。

默认完全离线（``HashingEmbedder`` + BM25 + SQLite），不联网、不需要 API Key：

    py -3.14 -m examples.demo_travel_memory_rag --out-dir demo_output

可选语义向量后端（需先安装 extras）：

    py -3.14 -m pip install -e ".[embed]"
    py -3.14 -m examples.demo_travel_memory_rag --embed fastembed

演示内容（a–e）：
    a. ReAct / Reflection / Plan-and-Solve 三范式「无记忆 vs 有记忆」对照；
    b. 多轮行程调整：第 2 天北京下雨、用户不想爬坡，记忆驱动改排室内行程；
    c. 跨会话召回：新建会话连接同一 SQLite 长期记忆，仍能想起旧行程；
    d. 用户上传 PDF / DOCX / URL 文档入库，回答问题并给出引用；
    e. 同一问题的「不带 RAG vs 带 RAG」回答对照。

输出（默认写到 ``--out-dir``）：
    trace.jsonl  —— 三范式全链路轨迹
    memory.json  —— 长期记忆状态与各阶段召回上下文
    rag_eval.md  —— 带 / 不带 RAG 的回答对照表

说明：示例 PDF 用 ASCII 英文正文（避免在仓库中引入中文字体文件）；真实中文
PDF 只要内嵌字体、编码正常即可被 ``pypdf`` 解析，不需要额外处理。
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import threading
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Iterable

# 直接以脚本方式运行时，把仓库根目录加进 sys.path，保证 `import agentflow` / `import examples`
if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agentflow.agents import PlanAndSolveAgent, ReActAgent, ReflectionAgent
from agentflow.memory import LongTermMemory, MemoryManager
from agentflow.retrieval import (
    BM25Index,
    Chunker,
    Document,
    Embedder,
    FastEmbedEmbedder,
    HashingEmbedder,
    RAGPipeline,
    SQLiteVectorStore,
    load_document,
)
from agentflow.trace import JSONLFileCallback, Tracer
from examples.demo_e2e import make_registry
from examples.travel_workflow import TravelDeps, build_travel_graph

# --- 知识库语料 -----------------------------------------------------------
# 每篇 4 个二级标题，默认 Chunker 会切成 4 个 chunk；data/eval/travel_qa.jsonl
# 的 gold_chunk_ids 由这些文本 deterministic 计算得出，改动文本后需重新生成。

KNOWLEDGE_DOCS: list[dict[str, str]] = [
    {
        "doc_id": "kb_beijing",
        "title": "北京旅行知识",
        "city": "北京",
        "text": """# 北京旅行知识

## 故宫
故宫博物院位于北京中轴线，旺季门票六十元。开放时间为八点半到十七点，周一闭馆，法定节假日除外。故宫大部分展馆位于室内，雨天可以正常参观。

## 颐和园
颐和园以昆明湖和万寿山为主体，适合半日游。东宫门到长廊一线地势平缓，适合不爬坡的游客；登万寿山需要连续爬台阶，腿脚不便者建议绕行。

## 香山
香山公园以秋季红叶闻名，最佳观赏期是十月中旬到十一月初。登顶需要连续爬坡约一小时，雨天山路湿滑，不建议前往。

## 北京雨天备选
北京雨天优先选择室内景点，例如故宫博物院、中国国家博物馆和首都博物馆。香山、八达岭长城等户外景点在山路湿滑时应避免前往。
""",
    },
    {
        "doc_id": "kb_shanghai",
        "title": "上海旅行知识",
        "city": "上海",
        "text": """# 上海旅行知识

## 外滩
外滩沿黄浦江展开，全长约一点五公里，适合傍晚散步。夜景最佳观赏时间是晚上七点至九点，灯光会持续到深夜。

## 豫园
豫园是明代古典园林，与城隍庙相邻，游览时间约半天。园内假山与回廊较多，节假日人流密集，建议早上开园时进入。

## 上海雨天备选
上海雨天推荐上海博物馆和上海科技馆等室内场馆。外滩夜景可以改到次日晴天再去看。
""",
    },
    {
        "doc_id": "kb_hangzhou",
        "title": "杭州旅行知识",
        "city": "杭州",
        "text": """# 杭州旅行知识

## 西湖
西湖环湖道路约十五公里，适合环湖骑行，也可以乘游船。苏堤和白堤是经典步行路线，环湖一圈大约需要两到三小时。

## 灵隐寺
灵隐寺位于西湖西北的飞来峰下，始建于东晋。寺内台阶较多，雨天石阶湿滑，行动不便的游客需要谨慎。

## 杭州雨天备选
杭州雨天可以参观中国茶叶博物馆和浙江省博物馆。原定的环湖骑行可以改为步行或乘观光车。
""",
    },
    {
        "doc_id": "kb_chengdu",
        "title": "成都旅行知识",
        "city": "成都",
        "text": """# 成都旅行知识

## 大熊猫基地
成都大熊猫繁育研究基地位于北郊斧头山，建议早上开园时前往。上午是熊猫活动高峰，中午以后多数熊猫会休息。

## 都江堰
都江堰景区由鱼嘴、飞沙堰和宝瓶口组成，适合安排一日游。景区步行距离较长，建议穿舒适的鞋。

## 成都雨天备选
成都雨天推荐成都博物馆和四川博物院等室内场馆。都江堰行程建议改期，山区道路在雨天较为湿滑。
""",
    },
]

DEMO_USER_ID = "demo_user"
MAIN_QUESTION = "我想去北京玩一天，有什么景点推荐？"
DEMO_DIR_NAME = "demo_output"

_KNOWLEDGE_TITLE = {doc["doc_id"]: doc["title"] for doc in KNOWLEDGE_DOCS}


def knowledge_documents(namespace: str = "travel") -> list[Document]:
    """把内置语料转成带显式 ``doc_id`` 的 :class:`Document`。

    显式 ``doc_id`` 让评估脚本能在不落盘的情况下复现同一批 chunk_id。
    """
    documents: list[Document] = []
    for item in KNOWLEDGE_DOCS:
        documents.append(
            Document(
                doc_id=item["doc_id"],
                namespace=namespace,
                source_type="file",
                source_uri=f"demo://{item['doc_id']}",
                title=item["title"],
                text=item["text"],
                metadata={
                    "markdown": True,
                    "doc_type": "markdown",
                    "source": item["title"],
                    "city": item["city"],
                },
            )
        )
    return documents


# --- 离线确定性 LLM -------------------------------------------------------


class TravelDemoLLM:
    """可复现的离线 LLM：按提示词形态返回演示文本，不访问网络。

    只实现 ``chat``，满足 Agent / RAGPipeline / 工作流对 LLM 的最小接口要求。
    """

    def __init__(self) -> None:
        self.calls: list[str] = []

    def chat(
        self,
        messages: Iterable[dict[str, Any]],
        *,
        temperature: float | None = None,
        json_mode: bool = False,
        tools: Any = None,
    ) -> str:
        """根据提示词内容生成确定性回复，并记录调用。"""
        message_list = list(messages)
        system = str(message_list[0].get("content", "")) if message_list else ""
        prompt = "\n".join(str(m.get("content", "")) for m in message_list)
        self.calls.append(prompt)

        if json_mode:
            return json.dumps(
                {
                    "thought": "用户想去北京玩一天，关注天气与自然风光。",
                    "intent": "北京一日游，关注天气与景点推荐。",
                    "weather_query": "北京今天天气",
                    "attraction_query": "北京 自然风光 景点 推荐",
                },
                ensure_ascii=False,
            )
        if "知识库资料" in system:
            return self._rag_answer(prompt)
        # 规划器模板独有「必须是一个Python列表」；Reflection 的角色里也含「规划专家」，
        # 因此不能按「规划专家」判断，否则会把反思的初稿误判成计划。
        if "必须是一个Python列表" in prompt:
            return '```python\n["检索旅行知识库中的景点信息", "结合用户记忆组织行程"]\n```'
        if "执行专家" in prompt:
            if _has_memory(prompt):
                return "结合你的偏好（自然风光、不爬坡）：第 1 天故宫，第 2 天颐和园东宫门至长廊一线，第 3 天中国国家博物馆。"
            return "推荐北京故宫、颐和园与八达岭长城。"
        if "评审专家" in prompt:
            return "无需改进，行程已覆盖天气、交通与步行强度。"
        if "输出格式要求" in system:
            return self._react_answer(prompt)
        if _has_memory(prompt):
            return (
                "结合你的记忆：你偏爱自然风光、不喜欢爬坡，"
                "推荐颐和园东宫门到长廊一线，雨天改去中国国家博物馆。"
            )
        return "推荐故宫、颐和园与八达岭长城，都是北京经典景点。"

    @staticmethod
    def _react_answer(prompt: str) -> str:
        """按已出现的 Observation 数推进 ReAct 循环，最后 Finish。"""
        observations = prompt.count("Observation:")
        if observations == 0:
            return 'Thought: 先查北京今天的天气\nAction: get_weather(city="北京")'
        if observations == 1:
            return 'Thought: 再按天气查适合的景点\nAction: get_attraction(city="北京", weather="晴")'
        if _has_memory(prompt):
            return (
                "Thought: 信息齐全，结合记忆给出建议\n"
                "Action: Finish[北京今天晴，结合你偏爱自然风光、不喜欢爬坡的偏好，"
                "推荐颐和园东宫门至长廊一线；如遇雨天改去中国国家博物馆。]"
            )
        return (
            "Thought: 信息齐全\n"
            "Action: Finish[北京今天晴，推荐故宫、颐和园与八达岭长城。]"
        )

    @staticmethod
    def _rag_answer(prompt: str) -> str:
        """从 RAG 上下文中挑与问题最相关的一条资料作答并附引用。

        真实 LLM 会读完整个上下文再选证据；这里用字符二元组重合度模拟这个选择，
        避免总是取第一条命中。
        """
        blocks = re.findall(
            r"【资料 \d+｜([^】]+)】\n(.+?)(?=\n\n【资料 |\n</knowledge>|$)",
            prompt,
            re.DOTALL,
        )
        if not blocks:
            return "知识库中未找到相关内容。"
        question = prompt.rsplit("用户问题：", 1)[-1].split("\n请给出", 1)[0]
        citation, body = max(
            blocks, key=lambda block: _bigram_overlap(question, block[0] + block[1])
        )
        return f"{body.strip()}\n依据：{citation}"


def _has_memory(prompt: str) -> bool:
    """判断提示词里是否注入了记忆上下文。"""
    return "【记忆上下文】" in prompt or "相关记忆：" in prompt


def _bigram_overlap(query: str, text: str) -> int:
    """统计两个字符串的字符二元组重合数，用于挑选最相关证据。"""
    query_grams = {query[i : i + 2] for i in range(len(query) - 1)}
    text_grams = {text[i : i + 2] for i in range(len(text) - 1)}
    return len(query_grams & text_grams)


# --- Demo a：三范式记忆对照 -----------------------------------------------


def _build_agent(paradigm: str, llm: TravelDemoLLM, memory: Any, tracer: Tracer):
    """按范式名构建智能体；三范式共享同一套离线 LLM 与记忆。"""
    if paradigm == "react":
        registry = make_registry(
            weather_fn=lambda city: f"{city}今天晴，22℃，适合户外",
            attraction_fn=lambda city, weather: (
                f"{city}天气{weather}，优先推荐颐和园东宫门至长廊一线"
            ),
        )
        return ReActAgent(
            "react-travel",
            llm,
            tools=registry,
            memory=memory,
            tracer=tracer,
            system_prompt="旅行助手",
        )
    if paradigm == "reflection":
        return ReflectionAgent(
            "reflection-travel",
            llm,
            memory=memory,
            tracer=tracer,
            max_iterations=1,
            role="旅行规划专家",
        )
    return PlanAndSolveAgent(
        "plan-travel", llm, memory=memory, tracer=tracer
    )


def demo_paradigms(memory: MemoryManager, tracer: Tracer) -> list[dict[str, str]]:
    """a. 三范式在有 / 无记忆下的回答对照。"""
    rows: list[dict[str, str]] = []
    for paradigm in ("react", "reflection", "plan_solve"):
        without = _build_agent(paradigm, TravelDemoLLM(), None, tracer).run(MAIN_QUESTION)
        with_memory = _build_agent(
            paradigm, TravelDemoLLM(), memory, tracer
        ).run(MAIN_QUESTION)
        rows.append(
            {
                "paradigm": paradigm,
                "without_memory": without,
                "with_memory": with_memory,
            }
        )
    return rows


def demo_workflow(tracer: Tracer) -> str:
    """低代码工作流同样复用离线 LLM，验证 M8 改动未破坏 v1.0 形态。"""
    graph = build_travel_graph(
        TravelDeps(
            llm=TravelDemoLLM(),
            weather_tool=lambda city: f"{city}今天晴，22℃",
            attraction_tool=lambda city, weather, **_: f"{city}天气{weather}，推荐颐和园",
        )
    )
    state = graph.invoke({"user_query": MAIN_QUESTION, "city": "北京"})
    return str(state.get("final_answer", ""))


# --- Demo b / c：多轮调整与跨会话召回 -------------------------------------


def demo_itinerary(memory: MemoryManager, tracer: Tracer) -> dict[str, Any]:
    """b. 第 2 天下雨且用户不想爬坡：检索记忆后改排室内行程。"""
    question = "第 2 天北京下雨了，帮我调整行程，我不想爬坡。"
    context = memory.build_context(question)
    llm = TravelDemoLLM()
    answer = llm.chat(
        [
            {"role": "system", "content": "你是旅行规划助手，回答需兼顾用户记忆。"},
            {"role": "user", "content": f"{context.text}\n\n{question}"},
        ]
    )
    memory.record_turn(question, answer)
    tracer.emit(
        "memory.demo",
        agent_name="itinerary",
        data={"question": question, "memory_items": len(context.items)},
    )
    return {
        "question": question,
        "memory_context": context.text,
        "answer": answer,
    }


def demo_cross_session(
    embedder: Embedder, memory_path: Path, tracer: Tracer
) -> dict[str, Any]:
    """c. 新会话连接同一 SQLite 长期记忆，仍能召回上一轮的行程。"""
    question = "我第二天原本安排去哪？为什么现在改了？"
    with LongTermMemory(embedder, path=memory_path, user_id=DEMO_USER_ID) as store:
        new_session = MemoryManager(embedder, store, user_id=DEMO_USER_ID)
        context = new_session.build_context(question)
        answer = TravelDemoLLM().chat(
            [
                {"role": "system", "content": "你是旅行规划助手，回答需兼顾用户记忆。"},
                {"role": "user", "content": f"{context.text}\n\n{question}"},
            ]
        )
    tracer.emit(
        "memory.cross_session",
        agent_name="cross-session",
        data={"question": question, "memory_items": len(context.items)},
    )
    return {
        "question": question,
        "memory_context": context.text,
        "answer": answer,
    }


# --- Demo d：用户上传 PDF / DOCX / URL ------------------------------------


def _minimal_pdf(lines: list[str]) -> bytes:
    """生成只含 ASCII 文本的最小合法 PDF（无需任何 PDF 生成依赖）。"""
    content: list[str] = []
    y = 760
    for line in lines:
        safe = line.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
        content.append(f"BT /F1 11 Tf 72 {y} Td ({safe}) Tj ET")
        y -= 16
    stream = "\n".join(content)
    objects = [
        "<< /Type /Catalog /Pages 2 0 R >>",
        "<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 595 842] "
        "/Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >>",
        "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
        f"<< /Length {len(stream)} >>\nstream\n{stream}\nendstream",
    ]
    buffer = bytearray(b"%PDF-1.4\n")
    offsets: list[int] = []
    for index, body in enumerate(objects, start=1):
        offsets.append(len(buffer))
        buffer.extend(f"{index} 0 obj\n{body}\nendobj\n".encode("ascii"))
    xref = len(buffer)
    buffer.extend(f"xref\n0 {len(objects) + 1}\n".encode("ascii"))
    buffer.extend(b"0000000000 65535 f \n")
    for offset in offsets:
        buffer.extend(f"{offset:010d} 00000 n \n".encode("ascii"))
    buffer.extend(
        (
            f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\n"
            f"startxref\n{xref}\n%%EOF\n"
        ).encode("ascii")
    )
    return bytes(buffer)


def _write_docx(path: Path) -> None:
    """用 python-docx 写一份带标题的中文示例文档。"""
    import docx

    document = docx.Document()
    document.add_heading("灵隐寺预约须知", level=1)
    document.add_heading("预约方式", level=2)
    document.add_paragraph(
        "灵隐寺需要提前一天在官方小程序预约，每天限额三万人，约满即止。"
    )
    document.add_heading("门票说明", level=2)
    document.add_paragraph("飞来峰景区需要另行购票，寺庙门票与景区门票分开收取。")
    document.save(str(path))


def _index_document(pipeline: RAGPipeline, document: Document) -> int:
    """把已经加载好的 Document 写入向量库与 BM25（URL 演示用）。"""
    chunks = pipeline.chunker.split(document)
    if not chunks:
        return 0
    vectors = pipeline.embedder.embed([chunk.text for chunk in chunks])
    for chunk, vector in zip(chunks, vectors):
        chunk.embedding = vector
        chunk.embedding_fingerprint = pipeline.embedder.fingerprint
    pipeline.vector_store.upsert(chunks)
    pipeline.bm25.add(chunks)
    return len(chunks)


class _QuietHandler(SimpleHTTPRequestHandler):
    """不打印访问日志的本地静态文件处理器。"""

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
        """静默处理访问日志。"""


def _serve_directory(directory: Path) -> tuple[ThreadingHTTPServer, threading.Thread]:
    """在 127.0.0.1 随机端口启动静态文件服务，供 URL 入库演示使用。"""
    handler = lambda *args, **kwargs: _QuietHandler(  # noqa: E731
        *args, directory=str(directory), **kwargs
    )
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, thread


def demo_uploads(
    pipeline: RAGPipeline, upload_dir: Path, tracer: Tracer
) -> list[dict[str, Any]]:
    """d. PDF / DOCX / URL 三种来源入库、检索并回答（带引用）。"""
    upload_dir.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, Any]] = []

    pdf_path = upload_dir / "beijing_notice.pdf"
    pdf_path.write_bytes(
        _minimal_pdf(
            [
                "Beijing Attraction Notice",
                "The Forbidden City is closed every Monday.",
                "Tickets must be booked online three days in advance.",
            ]
        )
    )
    results.append(
        _ingest_and_answer(
            pipeline,
            str(pdf_path),
            "How many days in advance must tickets be booked?",
            source="PDF 上传",
            tracer=tracer,
        )
    )

    docx_path = upload_dir / "lingyin_notice.docx"
    _write_docx(docx_path)
    results.append(
        _ingest_and_answer(
            pipeline,
            str(docx_path),
            "灵隐寺需要预约吗？",
            source="DOCX 上传",
            tracer=tracer,
        )
    )

    html_path = upload_dir / "west_lake_boat.html"
    html_path.write_text(
        """<!doctype html><html><head><meta charset="utf-8"><title>西湖游船时刻</title></head>
<body>
<h1>西湖游船时刻</h1>
<h2>运营时间</h2>
<p>西湖游船从早上八点运营到下午五点，末班船四点半停止售票。</p>
<h2>购票提示</h2>
<p>旺季建议提前一天在官方渠道购票，雨天视风力情况可能停航。</p>
</body></html>""",
        encoding="utf-8",
    )
    server, thread = _serve_directory(upload_dir)
    try:
        port = server.server_address[1]
        url = f"http://127.0.0.1:{port}/west_lake_boat.html"
        # 本地回环地址只在演示中显式放行；生产默认拒绝内网 URL（SSRF 防护）。
        document = load_document(url, allow_private_hosts=True, namespace=pipeline.namespace)
        _index_document(pipeline, document)
        results.append(
            _answer_question(
                pipeline,
                "西湖游船末班几点停止售票？",
                source=f"URL 入库 {url}",
                tracer=tracer,
            )
        )
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
    return results


def _ingest_and_answer(
    pipeline: RAGPipeline,
    path: str,
    question: str,
    *,
    source: str,
    tracer: Tracer,
) -> dict[str, Any]:
    """入库一个本地文件并回答。"""
    result = pipeline.ingest(path, namespace=pipeline.namespace)
    tracer.emit(
        "rag.ingest",
        agent_name="rag",
        data={
            "source": result.source_uri,
            "doc_id": result.doc_id,
            "chunks": result.chunk_count,
        },
    )
    answer = _answer_question(pipeline, question, source=source, tracer=tracer)
    answer["ingest"] = result.to_dict()
    return answer


def _answer_question(
    pipeline: RAGPipeline, question: str, *, source: str, tracer: Tracer
) -> dict[str, Any]:
    """走 RAG 管线回答并记录命中与引用。"""
    answer = pipeline.answer(question, llm=TravelDemoLLM(), top_k=3)
    tracer.emit(
        "rag.answer",
        agent_name="rag",
        data={
            "question": question,
            "answer": answer.answer,
            "citations": answer.citations,
            "hits": len(answer.hits),
        },
    )
    return {
        "source": source,
        "question": question,
        "answer": answer.answer,
        "citations": answer.citations,
        "hits": [hit.to_dict() for hit in answer.hits],
    }


# --- Demo e：带 / 不带 RAG 对照 -------------------------------------------


def demo_rag_compare(pipeline: RAGPipeline) -> list[dict[str, str]]:
    """e. 同一问题分别走「不经检索」与「RAG 检索 + 引用」的回答对照。"""
    llm = TravelDemoLLM()
    questions = [
        "北京下雨天可以去哪些室内景点？",
        "故宫周几闭馆？",
        "西湖适合骑行吗？",
    ]
    rows: list[dict[str, str]] = []
    for question in questions:
        without = llm.chat([{"role": "user", "content": question}])
        with_rag = pipeline.answer(question, llm=llm, top_k=3)
        rows.append(
            {
                "question": question,
                "without_rag": without,
                "with_rag": with_rag.answer,
                "citations": "；".join(with_rag.citations),
            }
        )
    return rows


# --- 输出与入口 -----------------------------------------------------------


def _write_knowledge_files(out_dir: Path) -> list[Path]:
    """把内置语料写成 Markdown 文件，走真实的文件加载器入库。"""
    knowledge_dir = out_dir / "knowledge"
    knowledge_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for item in KNOWLEDGE_DOCS:
        path = knowledge_dir / f"{item['doc_id']}.md"
        path.write_text(item["text"], encoding="utf-8")
        paths.append(path)
    return paths


def _markdown_table(rows: list[dict[str, str]], columns: list[str]) -> str:
    """把字典行渲染成 Markdown 表格，转义竖线与换行。"""
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join("---" for _ in columns) + " |"
    lines = [header, separator]
    for row in rows:
        cells = []
        for column in columns:
            value = str(row.get(column, "") or "").replace("|", "\\|")
            cells.append(value.replace("\n", "<br>"))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _render_rag_table(rows: list[dict[str, str]]) -> str:
    """渲染带 / 不带 RAG 的对照表。"""
    lines = [
        "## 带 / 不带 RAG 的回答对照",
        "",
        "| 问题 | 不带 RAG | 带 RAG（附引用） | 召回资料（按相关度） |",
        "| --- | --- | --- | --- |",
    ]
    for row in rows:
        without = row["without_rag"].replace("|", "\\|").replace("\n", "<br>")
        with_rag = row["with_rag"].replace("|", "\\|").replace("\n", "<br>")
        citations = row["citations"].replace("|", "\\|")
        lines.append(f"| {row['question']} | {without} | {with_rag} | {citations} |")
    return "\n".join(lines)


def _render_upload_table(rows: list[dict[str, Any]]) -> str:
    """渲染用户上传文档的入库与问答结果。"""
    lines = [
        "## 用户上传文档（PDF / DOCX / URL）",
        "",
        "| 来源 | 问题 | 回答 | 引用 |",
        "| --- | --- | --- | --- |",
    ]
    for row in rows:
        answer = row["answer"].replace("|", "\\|").replace("\n", "<br>")
        citations = "；".join(row["citations"]).replace("|", "\\|")
        lines.append(
            f"| {row['source']} | {row['question']} | {answer} | {citations} |"
        )
    return "\n".join(lines)


def run_demo(
    out_dir: str | Path = DEMO_DIR_NAME,
    *,
    embed_backend: str = "hashing",
    ingest: str | None = None,
    query: str | None = None,
) -> dict[str, Any]:
    """运行完整 Demo 并写出 trace / memory / rag_eval 三个产物。

    参数：
        out_dir: 输出目录，同时也是 SQLite 数据库与知识库文件的落盘位置。
        embed_backend: ``hashing``（默认，离线）或 ``fastembed``（需安装 extras）。
        ingest: 额外入库的本地路径或公网 URL（选修，失败会抛清晰异常）。
        query: 额外问答的问题；为空时不做。

    返回：
        包含各阶段结果的字典，便于测试与二次消费。
    """
    output = Path(out_dir)
    output.mkdir(parents=True, exist_ok=True)
    rag_db = output / "rag.sqlite3"
    memory_db = output / "memory.sqlite3"
    trace_path = output / "trace.jsonl"
    for path in (
        rag_db,
        memory_db,
        Path(f"{rag_db}-wal"),
        Path(f"{rag_db}-shm"),
        Path(f"{memory_db}-wal"),
        Path(f"{memory_db}-shm"),
        trace_path,
    ):
        if path.exists():
            path.unlink()

    embedder: Embedder = (
        FastEmbedEmbedder() if embed_backend == "fastembed" else HashingEmbedder()
    )
    vector_store = SQLiteVectorStore(str(rag_db))
    pipeline = RAGPipeline(embedder, vector_store, BM25Index(), namespace="travel")

    tracer = Tracer()
    jsonl = JSONLFileCallback(str(trace_path))
    handle = tracer.subscribe(jsonl)

    long_term = LongTermMemory(embedder, path=memory_db, user_id=DEMO_USER_ID)
    memory = MemoryManager(embedder, long_term, user_id=DEMO_USER_ID)

    try:
        knowledge_paths = _write_knowledge_files(output)
        ingest_results = [
            pipeline.ingest(
                str(path),
                namespace="travel",
                title=_KNOWLEDGE_TITLE[path.stem],
                metadata={"city": _city_for(path.stem)},
            )
            for path in knowledge_paths
        ]

        seed_memory(memory)
        paradigms = demo_paradigms(memory, tracer)
        workflow_answer = demo_workflow(tracer)
        itinerary = demo_itinerary(memory, tracer)
        cross_session = demo_cross_session(embedder, memory_db, tracer)
        uploads = demo_uploads(pipeline, output / "uploads", tracer)
        rag_rows = demo_rag_compare(pipeline)

        # 先入库再提问，保证 --ingest 的文档能被 --query 检索到。
        if ingest:
            extra_ingest = pipeline.ingest(ingest, namespace="travel")
        else:
            extra_ingest = None
        extra_answer = pipeline.answer(query, llm=TravelDemoLLM()) if query else None

        memory_payload = {
            "user_id": DEMO_USER_ID,
            "stats": memory.stats(),
            "paradigms": paradigms,
            "workflow_answer": workflow_answer,
            "itinerary": itinerary,
            "cross_session": cross_session,
            "long_term_samples": sample_long_term(memory),
        }
        (output / "memory.json").write_text(
            json.dumps(memory_payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )

        report_lines = [
            "# 智游·山水行：记忆 + RAG Demo 输出",
            "",
            f"- 向量后端：`{embed_backend}`（{embedder.fingerprint}）",
            f"- 知识库：{len(KNOWLEDGE_DOCS)} 篇 / {sum(r.chunk_count for r in ingest_results)} chunks",
            f"- 索引统计：{json.dumps(pipeline.stats(), ensure_ascii=False)}",
            "",
            "## 三范式：无记忆 vs 有记忆",
            "",
            _markdown_table(
                paradigms, ["paradigm", "without_memory", "with_memory"]
            ),
            "",
            _render_upload_table(uploads),
            "",
            _render_rag_table(rag_rows),
            "",
            "## 多轮行程调整（记忆驱动）",
            "",
            f"- 问题：{itinerary['question']}",
            f"- 召回上下文：\n\n```text\n{itinerary['memory_context']}\n```",
            f"- 回答：{itinerary['answer']}",
            "",
            "## 跨会话召回",
            "",
            f"- 问题：{cross_session['question']}",
            f"- 召回上下文：\n\n```text\n{cross_session['memory_context']}\n```",
            f"- 回答：{cross_session['answer']}",
            "",
        ]
        if query is not None and extra_answer is not None:
            report_lines.extend(
                [
                    "## 自定义问题",
                    "",
                    f"- 问题：{query}",
                    f"- 回答：{extra_answer.answer}",
                    f"- 依据：{'；'.join(extra_answer.citations) or '无'}",
                    "",
                ]
            )
        if extra_ingest is not None:
            report_lines.extend(
                [
                    "## 额外入库",
                    "",
                    f"- 来源：{extra_ingest.source_uri}",
                    f"- 新增 chunk：{extra_ingest.chunk_count}",
                    "",
                ]
            )
        (output / "rag_eval.md").write_text("\n".join(report_lines), encoding="utf-8")

        return {
            "out_dir": str(output),
            "embed_backend": embed_backend,
            "stats": pipeline.stats(),
            "ingest": [result.to_dict() for result in ingest_results],
            "paradigms": paradigms,
            "uploads": uploads,
            "rag_compare": rag_rows,
            "itinerary": itinerary,
            "cross_session": cross_session,
            "workflow_answer": workflow_answer,
        }
    finally:
        tracer.unsubscribe(handle)
        jsonl.close()
        long_term.close()
        vector_store.close()


def _city_for(doc_id: str) -> str:
    """按 doc_id 查城市名。"""
    for item in KNOWLEDGE_DOCS:
        if item["doc_id"] == doc_id:
            return item["city"]
    return ""


def seed_memory(memory: MemoryManager) -> None:
    """写入演示用的用户画像与行程记忆。"""
    memory.add(
        "用户偏好自然风光，不喜欢爬坡。",
        importance=0.9,
        memory_type="semantic",
        metadata={"topic": "preference"},
    )
    memory.add(
        "用户计划北京三日游：第 1 天故宫，第 2 天香山，第 3 天颐和园。",
        importance=0.8,
        memory_type="episodic",
        metadata={"topic": "itinerary"},
    )


def sample_long_term(memory: MemoryManager) -> list[dict[str, Any]]:
    """为 memory.json 采集几组长期记忆检索结果。"""
    samples: list[dict[str, Any]] = []
    for query in ("北京怎么安排", "雨天去哪些室内景点", "我原来第二天的安排"):
        hits = memory.long_term.search(query, top_k=3)
        samples.append(
            {
                "query": query,
                "hits": [hit.to_dict() for hit in hits],
            }
        )
    return samples


def main(argv: list[str] | None = None) -> int:
    """命令行入口：解析参数并运行 Demo。"""
    parser = argparse.ArgumentParser(description="智游·山水行：记忆 + RAG 综合 Demo")
    parser.add_argument("--out-dir", default=DEMO_DIR_NAME, help="输出目录")
    parser.add_argument(
        "--embed",
        choices=["hashing", "fastembed"],
        default="hashing",
        help="向量后端：默认 hashing（离线），fastembed 需安装 extras",
    )
    parser.add_argument("--ingest", default=None, help="额外入库的本地路径或公网 URL")
    parser.add_argument("--query", default=None, help="额外提问（走 RAG 管线）")
    args = parser.parse_args(argv)

    result = run_demo(
        args.out_dir,
        embed_backend=args.embed,
        ingest=args.ingest,
        query=args.query,
    )
    print(f"Demo 完成：{result['out_dir']}")
    print(f"索引统计：{json.dumps(result['stats'], ensure_ascii=False)}")
    print(f"三范式对照：{len(result['paradigms'])} 组")
    print(f"上传文档问答：{len(result['uploads'])} 组")
    print(f"RAG 对照：{len(result['rag_compare'])} 组")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
